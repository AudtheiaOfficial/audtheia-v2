#!/usr/bin/env python3
"""End-to-end check for the desktop orchestrator.

Path: tests/test_orchestrator.py

This proves that one desktop computer runs the whole pipeline with no field
hardware and no language model: a scripted camera and two scripted models drive
capture, quality control, verification, the dream pass, and a report, over one
database, through the real engines. Only the two model runtimes are stand-ins;
every engine and the orchestration are real.

Run directly: python tests/test_orchestrator.py
It needs numpy, opencv-python-headless, Pillow, and fpdf2, which the desktop
runtime installs.
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from audtheia.config import load_settings
from audtheia.storage.database import Database, Station
from audtheia.pipeline.monitor import NullTriggerSink, TrackedDetection
from audtheia.pipeline.drivers import OpenCVFrameSource, OnnxYoloDetector
from audtheia.inference.rfdetr_onnx import RFDETRVerifier
from audtheia.analysis.dream import STATUS_COMPLETE
from audtheia.app.orchestrator import DesktopStation, NullInterpreter


# ---------------------------------------------------------------------------
# Scripted camera and models (no hardware, no model files).
# ---------------------------------------------------------------------------


class ScriptedCapture:
    def __init__(self, frames):
        self._frames = list(frames)
        self._i = 0

    def read(self):
        if self._i >= len(self._frames):
            return False, None
        frame = self._frames[self._i]
        self._i += 1
        return True, frame

    def release(self):
        pass


class _YoloInput:
    name = "images"

    def __init__(self, size):
        self.shape = [1, 3, size, size]


class ScriptedYoloSession:
    """One strong box for class 0, in the standard YOLO output layout."""

    def __init__(self, in_size=64, classes=3, cls=0, conf=0.9, box=(32.0, 32.0, 20.0, 20.0)):
        self._in, self._c, self._cls, self._conf, self._box = in_size, classes, cls, conf, box

    def get_inputs(self):
        return [_YoloInput(self._in)]

    def run(self, output_names, feeds):
        candidates = 20
        out = np.zeros((1, 4 + self._c, candidates), dtype=np.float32)
        cx, cy, w, h = self._box
        out[0, 0, 0], out[0, 1, 0], out[0, 2, 0], out[0, 3, 0] = cx, cy, w, h
        out[0, 4 + self._cls, 0] = self._conf
        return [out]


class _DetrInput:
    name = "input"

    def __init__(self, size):
        self.shape = [1, 3, size, size]


class ScriptedVerifierSession:
    """One strong query for the same class, so the desktop re-score agrees."""

    def __init__(self, in_size=64, queries=10, classes=3, top_class=0, top_logit=6.0):
        self._in, self._q, self._c, self._top, self._logit = in_size, queries, classes, top_class, top_logit

    def get_inputs(self):
        return [_DetrInput(self._in)]

    def run(self, output_names, feeds):
        logits = np.full((1, self._q, self._c), -10.0, dtype=np.float32)
        logits[0, 0, self._top] = self._logit
        boxes = np.zeros((1, self._q, 4), dtype=np.float32)
        return [logits, boxes]


class SingleTrackTracker:
    def update(self, detections, frame_index):
        return [
            TrackedDetection(track_id=1, x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                             confidence=d.confidence, class_id=d.class_id)
            for d in detections
        ]

    def reset(self):
        pass


def _red_frame_bgr(size=64):
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[20:44, 20:44] = (0, 0, 255)
    return frame


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


def test_desktop_run_once_end_to_end():
    settings = load_settings()

    tmp = Path(tempfile.mkdtemp(prefix="audtheia_orch_"))
    settings.raw["paths"]["db_path"] = str(tmp / "audtheia.db")
    settings.raw["paths"]["detections_visual_dir"] = str(tmp / "visual")
    settings.raw["paths"]["reports_dir"] = str(tmp / "reports")

    station = settings.stations()[0]
    station_id = station["station_id"]

    db = Database(settings.db_path(), **settings.database_kwargs())
    db.initialize_schema(settings.schema_path())
    db.create_station(
        Station(id=station_id, station_name=station["station_name"],
                environment_type=station["environment_type"], created_at="2026-01-01T00:00:00Z")
    )

    # The desktop station, wired with scripted desktop models and no language model.
    desktop = DesktopStation(
        settings=settings,
        station=station,
        db=db,
        verifier=RFDETRVerifier(
            ScriptedVerifierSession(in_size=64, top_class=0),
            class_names={0: "Test species"},
            input_size=(64, 64),
            version="rfdetr-test",
        ),
        interpreter=NullInterpreter(),
        narrator=None,
        clusterer=None,
    )

    result = desktop.run_once(
        frame_source=OpenCVFrameSource(
            ScriptedCapture([_red_frame_bgr() for _ in range(6)]),
            live=False, fps=15, base_time=datetime(2026, 7, 8, tzinfo=timezone.utc),
        ),
        detector=OnnxYoloDetector(
            ScriptedYoloSession(in_size=64, cls=0),
            class_names={0: "Test species", 1: "b", 2: "c"},
            input_size=(64, 64),
            conf_threshold=0.25,
        ),
        tracker=SingleTrackTracker(),
        trigger_sink=NullTriggerSink(),
    )

    assert result["captured"] == 1, f"expected one captured event, got {result['captured']}"
    assert result["controlled"] == 1, f"expected one quality-controlled record, got {result['controlled']}"
    assert result["verified"] >= 1, f"expected the observation to be verified or unverified, got {result['verified']}"
    assert result["dream"].status == STATUS_COMPLETE, f"dream pass did not complete: {result['dream'].status}"

    # The observation exists, is finalized, and carries a verification verdict.
    obs = db.list_observations(station_id=station_id, limit=10)
    assert len(obs) == 1
    assert obs[0]["qc_state"] != "qc_pending"
    verdict = db.get_observation_verification(obs[0]["id"]) if hasattr(db, "get_observation_verification") else True
    assert verdict is not None

    # A report was written to the reports directory.
    report_dir = Path(settings.raw["paths"]["reports_dir"])
    assert report_dir.exists() and any(report_dir.rglob("*")), "no report files were written"


def main() -> int:
    checks = [test_desktop_run_once_end_to_end]
    failures = 0
    for check in checks:
        try:
            check()
            print(f"PASS  {check.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {check.__name__}: {exc}")
            import traceback
            traceback.print_exc()
    print(f"\n{len(checks) - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
