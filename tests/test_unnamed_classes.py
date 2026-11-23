"""A detection whose top class has no name is never stored under a number.

Path: tests/test_unnamed_classes.py

A detection model can output more classes than its label map names. RF-DETR
sizes its output to the largest class id plus one, so a model with 35 named
birds can still score class 40, and a label map may name only some of the
classes a model was trained on. Such a box used to be stored with its bare class
number as the species, for example "40". These checks feed each decoder a model
output whose strongest box is an unnamed class and prove that:

  - The desktop RF-DETR and YOLO detectors, the raw YOLO decoder used on a field
    station, and the field station's on-chip decoder leave the unnamed box out
    and keep a named box beside it.
  - An unnamed box does not suppress an overlapping named box when boxes are
    merged.
  - The desktop RF-DETR verifier reports a frame whose top class is unnamed as
    scored with nothing recognised, never as a call under a number.
  - With no label map at all, boxes are kept as before.

Model outputs are built in memory; nothing is read from or written to disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from audtheia.inference.rfdetr_onnx import RFDETRVerifier  # noqa: E402
from audtheia.pipeline import drivers  # noqa: E402
from audtheia.pipeline.monitor import Frame  # noqa: E402

BIRDS = {13: "harris-hawk", 17: "northern-cardinal"}
_checks = 0


def check(label: str, condition: bool) -> None:
    global _checks
    if not condition:
        raise AssertionError(label)
    _checks += 1
    print(f"  ok  {label}")


class _Input:
    def __init__(self, size):
        self.name = "images"
        self.shape = [1, 3, size, size]


class RfDetrSession:
    """Two queries on the same spot: an unnamed class 40, strongest, and a named cardinal."""

    def __init__(self, size=64, overlap=True):
        self._size = size
        self._overlap = overlap

    def get_inputs(self):
        return [_Input(self._size)]

    def run(self, output_names, feeds):
        boxes = np.zeros((1, 3, 4), dtype=np.float32)
        logits = np.full((1, 3, 41), -9.0, dtype=np.float32)
        boxes[0, 0] = (0.5, 0.5, 0.3, 0.3)
        logits[0, 0, 40] = 3.0  # unnamed, strongest
        boxes[0, 1] = (0.5, 0.5, 0.3, 0.3) if self._overlap else (0.2, 0.2, 0.1, 0.1)
        logits[0, 1, 17] = 1.0  # named, weaker, same place
        return [boxes, logits]


class YoloSession:
    """A raw YOLO output with an unnamed strongest class and a named box elsewhere."""

    def get_inputs(self):
        return [_Input(64)]

    def run(self, output_names, feeds):
        # More candidate boxes than feature rows, as in a real export, so the
        # decoder reads the channels-first layout the right way round.
        out = np.zeros((1, 4 + 41, 200), dtype=np.float32)
        out[0, 0:4, 0] = (32.0, 32.0, 20.0, 20.0)
        out[0, 4 + 40, 0] = 0.9
        out[0, 0:4, 1] = (10.0, 10.0, 8.0, 8.0)
        out[0, 4 + 17, 1] = 0.8
        return [out]


def frame() -> Frame:
    return Frame(index=0, image=np.zeros((64, 64, 3), dtype=np.uint8), captured_at="2026-11-23T00:00:00.000000Z")


def run() -> None:
    print("\n[1] Desktop RF-DETR detector")
    det = drivers.OnnxRfDetrDetector(RfDetrSession(), class_names=BIRDS, input_size=(64, 64), conf_threshold=0.25)
    found = det.detect(frame())
    check("the unnamed class 40 is left out", all(d.class_id != 40 for d in found))
    check("no detection carries a bare number as its name", all(not d.class_name.isdigit() for d in found))
    check("the named cardinal on the same spot is kept, not suppressed by the unnamed box",
          [d.class_name for d in found] == ["northern-cardinal"])
    bare = drivers.OnnxRfDetrDetector(RfDetrSession(overlap=False), class_names={}, input_size=(64, 64),
                                      conf_threshold=0.25).detect(frame())
    check("with no label map at all, boxes are kept as before", sorted(d.class_id for d in bare) == [17, 40])

    print("\n[2] Desktop YOLO detector and the raw YOLO decoder")
    yolo = drivers.OnnxYoloDetector(YoloSession(), class_names=BIRDS, input_size=(64, 64), conf_threshold=0.25)
    found = yolo.detect(frame())
    check("the YOLO detector keeps only the named box", [d.class_name for d in found] == ["northern-cardinal"])
    raw = drivers._yolo_raw_to_detections(YoloSession().run(None, None)[0], 1.0, (0, 0), 64, 64, BIRDS, 0.25, 0.5)
    check("the raw YOLO decoder keeps only the named box", [d.class_name for d in raw] == ["northern-cardinal"])

    print("\n[3] Field station on-chip decoder")
    per_class = [np.zeros((0, 5), dtype=np.float32) for _ in range(41)]
    per_class[40] = np.array([[0.1, 0.1, 0.5, 0.5, 0.9]], dtype=np.float32)
    per_class[13] = np.array([[0.2, 0.2, 0.6, 0.6, 0.7]], dtype=np.float32)
    hef = drivers._hailo_nms_to_detections(per_class, 1.0, (0, 0), 64, 64, 64, 64, BIRDS, 0.25)
    check("the on-chip decoder keeps only the named box", [d.class_name for d in hef] == ["harris-hawk"])

    print("\n[4] Desktop RF-DETR verifier")
    verifier = RFDETRVerifier.__new__(RFDETRVerifier)
    verifier._class_names = dict(BIRDS)
    verifier._score_threshold = 0.25
    top = verifier._decode_top(RfDetrSession().run(None, None))
    check("a frame whose top class is unnamed is scored with nothing recognised",
          top["scientific_name"] is None and top["confidence"] is None)

    print(f"\nAll {_checks} unnamed-class checks passed.")


if __name__ == "__main__":
    try:
        run()
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
