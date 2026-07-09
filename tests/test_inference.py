#!/usr/bin/env python3
"""Checks for the desktop inference adapters.

Path: tests/test_inference.py

These prove that the RF-DETR verification adapter turns a frame into the single
detection the verification engine expects, and that it plugs into the real engine
without any change to the engine. The model is a scripted ONNX session returning
one strong query, so the adapter's own preprocessing and decode run for real
while only the model runtime is a stand-in. The final check runs the real
VerifyEngine's frame-scoring path over the adapter and confirms it coerces the
adapter's output into the engine's own detection type.

Run directly: python tests/test_inference.py
It needs numpy and Pillow, which the desktop runtime installs.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from audtheia.config import load_settings
from audtheia.storage.database import Database
from audtheia.analysis.verify import VerifyEngine, FrameDetection
from audtheia.inference.rfdetr_onnx import RFDETRVerifier


class _Input:
    name = "input"

    def __init__(self, size):
        self.shape = [1, 3, size, size]


class ScriptedVerifierSession:
    """An ONNX-Runtime-style session returning per-query class logits and boxes,
    with one strong query for a chosen class, so the adapter's sigmoid-and-argmax
    decode runs against a realistic two-tensor RF-DETR output."""

    def __init__(self, in_size=64, queries=10, classes=5, top_class=2, top_logit=6.0):
        self._in = int(in_size)
        self._q = int(queries)
        self._c = int(classes)
        self._top = int(top_class)
        self._logit = float(top_logit)

    def get_inputs(self):
        return [_Input(self._in)]

    def run(self, output_names, feeds):
        logits = np.full((1, self._q, self._c), -10.0, dtype=np.float32)
        logits[0, 0, self._top] = self._logit  # strong detection: query 0, chosen class
        boxes = np.zeros((1, self._q, 4), dtype=np.float32)
        return [logits, boxes]


class _NullInterpreter:
    """An interpreter that adds no points, so the verifier can be exercised on its
    own. The real desktop language-model interpreter is a drop-in replacement."""

    version = None

    def interpret(self, context):
        return []


def _write_image(path: Path, size=80, colour=(120, 120, 120)):
    from PIL import Image

    Image.new("RGB", (size, size), colour).save(str(path))


def test_verifier_decodes_top_taxon():
    tmp = Path(tempfile.mkdtemp(prefix="audtheia_infer_"))
    frame = tmp / "frame.png"
    _write_image(frame)

    verifier = RFDETRVerifier(
        ScriptedVerifierSession(in_size=64, top_class=2),
        class_names={2: "Test species", 0: "a", 1: "b"},
        input_size=(64, 64),
        version="rfdetr-test",
    )
    detections = verifier.verify_frames([frame])
    assert len(detections) == 1
    d = detections[0]
    assert d["scientific_name"] == "Test species", d
    assert d["confidence"] > 0.9, d
    assert d["gbif_usage_key"] is None
    assert verifier.version == "rfdetr-test"


def test_missing_frame_counts_as_empty():
    verifier = RFDETRVerifier(
        ScriptedVerifierSession(in_size=64),
        class_names={2: "Test species"},
        input_size=(64, 64),
    )
    detections = verifier.verify_frames([Path("/no/such/frame.png")])
    assert detections == [{"gbif_usage_key": None, "scientific_name": None, "confidence": None}]


def test_plugs_into_verify_engine_scoring():
    settings = load_settings()
    tmp = Path(tempfile.mkdtemp(prefix="audtheia_infer_"))
    settings.raw["paths"]["db_path"] = str(tmp / "audtheia.db")

    db = Database(settings.db_path(), **settings.database_kwargs())
    db.initialize_schema(settings.schema_path())

    frame_a = tmp / "a.png"
    frame_b = tmp / "b.png"
    _write_image(frame_a)
    _write_image(frame_b)

    engine = VerifyEngine(
        settings=settings,
        db=db,
        verifier=RFDETRVerifier(
            ScriptedVerifierSession(in_size=64, top_class=2),
            class_names={2: "Test species"},
            input_size=(64, 64),
            version="rfdetr-test",
        ),
        interpreter=_NullInterpreter(),
    )

    # The engine's own frame-scoring path calls the adapter and coerces its output
    # into the engine's FrameDetection type, which is the exact seam the verdict
    # aggregation is built on.
    scored = engine._score_frames([frame_a, frame_b])
    assert len(scored) == 2
    assert all(isinstance(fd, FrameDetection) for fd in scored)
    assert all(fd.scientific_name == "Test species" for fd in scored), scored
    assert all(fd.confidence and fd.confidence > 0.9 for fd in scored)


def main() -> int:
    checks = [
        test_verifier_decodes_top_taxon,
        test_missing_frame_counts_as_empty,
        test_plugs_into_verify_engine_scoring,
    ]
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
