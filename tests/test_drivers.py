#!/usr/bin/env python3
"""Checks for the desktop capture drivers.

Path: tests/test_drivers.py

These prove that an ordinary video source flows through the unchanged detection
loop to a written, quality-controlled observation, with no field hardware and no
model file present. The camera is a scripted capture that replays a few frames,
and the detection model is a scripted ONNX session that returns one box, so the
drivers' own logic (colour conversion, timestamping, letterbox preprocessing,
and YOLO decoding) runs for real while only the two external backends are
stand-ins. The end-to-end check then runs the real Monitor, the real storage
layer, and the real quality-control engine over that feed and confirms one
observation is written and finalized.

Run directly: python tests/test_drivers.py
It needs numpy, opencv-python-headless, and Pillow, which the desktop runtime
installs.
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
from audtheia.analysis.observation import QCEngine, QC_PENDING
from audtheia.pipeline.monitor import Monitor, NullTriggerSink, TrackedDetection
from audtheia.pipeline.drivers import (
    OpenCVFrameSource,
    OnnxYoloDetector,
    _parse_video_spec,
    _resolve_stream_url,
    _best_stream_url,
)


# ---------------------------------------------------------------------------
# Scripted backends: a camera and a model that need no hardware or model file.
# ---------------------------------------------------------------------------


class ScriptedCapture:
    """An OpenCV-style capture that replays a fixed list of BGR frames."""

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


class _Input:
    name = "images"

    def __init__(self, size):
        self.shape = [1, 3, size, size]


class ScriptedSession:
    """An ONNX-Runtime-style session returning one strong box for class 0.

    The output uses the standard export layout (one row of box centre and size
    plus one score per class, transposed to channels-first), so the detector's
    real decode path is exercised end to end.
    """

    def __init__(self, num_classes=3, in_size=64, box=(32.0, 32.0, 20.0, 20.0), conf=0.9, cls=0):
        self._nc = int(num_classes)
        self._in = int(in_size)
        self._box = box
        self._conf = float(conf)
        self._cls = int(cls)

    def get_inputs(self):
        return [_Input(self._in)]

    def run(self, output_names, feeds):
        # A realistic export has far more candidate boxes than feature rows
        # (a real YOLO export is 84 rows by 8400 boxes), which is what lets the
        # decoder tell the channels-first layout from an already-transposed one.
        candidates = 20
        out = np.zeros((1, 4 + self._nc, candidates), dtype=np.float32)
        cx, cy, w, h = self._box
        out[0, 0, 0] = cx
        out[0, 1, 0] = cy
        out[0, 2, 0] = w
        out[0, 3, 0] = h
        out[0, 4 + self._cls, 0] = self._conf
        return [out]


class SingleTrackTracker:
    """A stand-in tracker that keeps every detection on one track, so a short
    clip of one object becomes exactly one event. The real ByteTrack tracker is
    exercised by the detection-loop check; here the point is the drivers and the
    end-to-end write, not the tracking maths."""

    def update(self, detections, frame_index):
        return [
            TrackedDetection(
                track_id=1,
                x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                confidence=d.confidence, class_id=d.class_id,
            )
            for d in detections
        ]

    def reset(self):
        pass


def _red_frame_bgr(size=64):
    """A small BGR frame with a red square, so a colour flip to RGB is visible."""
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[20:44, 20:44] = (0, 0, 255)  # red in BGR
    return frame


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def test_parse_video_spec():
    assert _parse_video_spec("webcam") == (0, True)
    assert _parse_video_spec("webcam:2") == (2, True)
    assert _parse_video_spec("url:rtsp://host/stream") == ("rtsp://host/stream", True)
    assert _parse_video_spec("http://host/live") == ("http://host/live", True)
    assert _parse_video_spec("file:/data/clip.mp4") == ("/data/clip.mp4", False)
    assert _parse_video_spec("/data/clip.mp4") == ("/data/clip.mp4", False)
    assert _parse_video_spec("3") == (3, True)
    assert _parse_video_spec("stream:https://youtu.be/2CP8QA_xKx4") == ("https://youtu.be/2CP8QA_xKx4", True)


class _FakeYDL:
    """A stand-in for yt-dlp's extractor, so stream resolution is tested without
    reaching the network."""

    def __init__(self, info):
        self._info = info

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False):
        return self._info


def test_stream_resolution_and_best_url():
    # A single resolved address is returned as is.
    assert _best_stream_url({"url": "https://cdn.example/live.m3u8"}) == "https://cdn.example/live.m3u8"
    # From a list of formats, the highest-resolution video stream is chosen and an
    # audio-only track is ignored.
    info = {
        "formats": [
            {"url": "low", "vcodec": "avc1", "height": 360},
            {"url": "high", "vcodec": "avc1", "height": 1080},
            {"url": "audio", "vcodec": "none", "height": None},
        ]
    }
    assert _best_stream_url(info) == "high"
    # End to end through an injected extractor: a page address resolves to a direct
    # stream address that OpenCV would then open.
    resolved = _resolve_stream_url(
        "https://youtu.be/2CP8QA_xKx4",
        ydl_factory=lambda: _FakeYDL({"url": "https://cdn.example/stream"}),
    )
    assert resolved == "https://cdn.example/stream"


def test_frame_source_converts_bgr_and_ends():
    src = OpenCVFrameSource(
        ScriptedCapture([_red_frame_bgr(), _red_frame_bgr()]),
        live=False,
        fps=10,
        base_time=datetime(2026, 7, 8, tzinfo=timezone.utc),
    )
    first = src.read()
    assert first is not None
    assert first.index == 0
    # The red square is BGR (0,0,255); after the flip to RGB the centre pixel is
    # (255,0,0), which is what the loop and the frame writer expect.
    assert tuple(int(v) for v in first.image[30, 30]) == (255, 0, 0)
    assert first.captured_at.endswith("Z")
    assert first.time_provisional == 0

    second = src.read()
    assert second is not None and second.index == 1
    # File timestamps advance from the frame rate, so a replayed clip yields a
    # real duration rather than every frame claiming the same instant.
    assert second.captured_at != first.captured_at

    assert src.read() is None  # end of the scripted stream
    src.close()


def test_detector_decodes_one_box():
    det = OnnxYoloDetector(
        ScriptedSession(num_classes=3, in_size=64),
        class_names={0: "test_species", 1: "b", 2: "c"},
        input_size=(64, 64),
        conf_threshold=0.25,
    )
    from audtheia.pipeline.monitor import Frame

    frame = Frame(index=0, image=_red_frame_bgr()[:, :, ::-1].copy(), captured_at="2026-07-08T00:00:00.000000Z")
    detections = det.detect(frame)
    assert len(detections) == 1, f"expected one detection, got {len(detections)}"
    d = detections[0]
    assert d.class_id == 0 and d.class_name == "test_species"
    assert abs(d.x1 - 22) < 2 and abs(d.x2 - 42) < 2, (d.x1, d.x2)
    assert d.confidence > 0.8


def test_end_to_end_capture_to_qc():
    settings = load_settings()

    tmp = Path(tempfile.mkdtemp(prefix="audtheia_drivers_"))
    settings.raw["paths"]["db_path"] = str(tmp / "audtheia.db")
    settings.raw["paths"]["detections_visual_dir"] = str(tmp / "visual")

    station = settings.stations()[0]
    station_id = station["station_id"]

    db = Database(settings.db_path(), **settings.database_kwargs())
    db.initialize_schema(settings.schema_path())

    # Register the station so the observation's foreign key resolves, the same
    # way the desktop registers a station before its observations arrive.
    db.create_station(
        Station(
            id=station_id,
            station_name=station["station_name"],
            environment_type=station["environment_type"],
            created_at="2026-01-01T00:00:00Z",
        )
    )

    frames = [_red_frame_bgr() for _ in range(5)]
    frame_source = OpenCVFrameSource(
        ScriptedCapture(frames),
        live=False,
        fps=15,
        base_time=datetime(2026, 7, 8, tzinfo=timezone.utc),
    )
    detector = OnnxYoloDetector(
        ScriptedSession(num_classes=3, in_size=64),
        class_names={0: "test_species", 1: "b", 2: "c"},
        input_size=(64, 64),
        conf_threshold=0.25,
    )

    monitor = Monitor(
        settings=settings,
        station=station,
        db=db,
        frame_source=frame_source,
        detector=detector,
        tracker=SingleTrackTracker(),
        trigger_sink=NullTriggerSink(),
    )
    monitor.run()

    observations = db.list_observations(station_id=station_id, limit=10)
    assert len(observations) == 1, f"expected one written observation, got {len(observations)}"
    obs = observations[0]
    assert obs["trigger_source"] == "vision"
    assert obs["data_source"] == "model"
    assert obs["qc_state"] == QC_PENDING, f"a freshly written observation should be pending, was {obs['qc_state']}"

    engine = QCEngine(settings=settings, db=db)
    result = engine.process(obs["id"])

    finalized = db.list_observations(station_id=station_id, limit=10)[0]
    assert finalized["qc_state"] != QC_PENDING, (
        f"quality control did not finalize the observation; state is still "
        f"{finalized['qc_state']} (result: {result.outcome})"
    )


def main() -> int:
    checks = [
        test_parse_video_spec,
        test_stream_resolution_and_best_url,
        test_frame_source_converts_bgr_and_ends,
        test_detector_decodes_one_box,
        test_end_to_end_capture_to_qc,
    ]
    failures = 0
    for check in checks:
        try:
            check()
            print(f"PASS  {check.__name__}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"FAIL  {check.__name__}: {exc}")
            import traceback
            traceback.print_exc()
    print(f"\n{len(checks) - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
