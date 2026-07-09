"""RF-DETR desktop verification adapter (ONNX Runtime).

Path: audtheia/inference/rfdetr_onnx.py

The desktop verification engine (audtheia/analysis/verify.py) re-scores an
event's frames through an injected verifier and aggregates the per-frame results
into one authoritative verdict that opens or holds the gate the dream pass reads.
The engine never imports a model runtime itself; it asks the injected verifier
for one detection per frame. This module is that verifier for the desktop: it
runs the high-accuracy RF-DETR model through ONNX Runtime and returns, for each
frame, the single most confident taxon it found.

The engine coerces whatever the verifier returns into its own frame-detection
type, so this adapter returns plain dictionaries with the three keys the engine
reads (the taxonomic backbone key, the scientific name, and the confidence),
which keeps the adapter independent of the engine's internals. A frame with no
model file, a missing image, or nothing above the score floor is reported as a
frame that was looked at but supported no taxon, which is exactly how the engine
counts a scored-but-empty frame.

ONNX Runtime is imported lazily inside the loader, so importing this module never
requires it, and the model session is an argument to the verifier, so the decode
and aggregation logic can be exercised end to end against a scripted model output
with no model file present. The exact output-tensor layout and preprocessing of a
specific RF-DETR export are confirmed against that model when it is placed; the
decode here targets the standard RF-DETR export (per-query class logits with
sigmoid scores, and a companion box tensor that the top-taxon decision does not
need).
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("audtheia.inference.rfdetr_onnx")

__all__ = [
    "RFDETRVerifier",
    "build_verifier",
    "VerifierError",
    "VerifierDependencyError",
    "DEFAULT_INPUT_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]

# The RF-DETR square input edge, used only as a fallback when the model does not
# declare its own input size. ImageNet channel statistics are the standard
# normalization RF-DETR is trained and exported with. These are model-family
# constants, not per-deployment values, so they live here rather than in the
# configuration.
DEFAULT_INPUT_SIZE = (560, 560)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# A frame that supports no taxon, returned in the shape the verification engine
# coerces. It counts as one scored frame with no detection.
_EMPTY_DETECTION = {"gbif_usage_key": None, "scientific_name": None, "confidence": None}


class VerifierError(RuntimeError):
    """The desktop verifier could not start for an operational reason."""


class VerifierDependencyError(VerifierError):
    """The desktop verifier needs a library that is not installed."""


def _import_onnxruntime():
    try:
        import onnxruntime as ort  # imported here so this module loads without it
        return ort
    except Exception as exc:  # noqa: BLE001
        raise VerifierDependencyError(
            "ONNX Runtime is required to run the RF-DETR verification model, but it "
            "is not installed. Install onnxruntime."
        ) from exc


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # A numerically stable logistic, since RF-DETR emits per-class logits scored
    # with a sigmoid rather than a softmax over a no-object class.
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


class RFDETRVerifier:
    """A per-event verifier that re-scores frames with RF-DETR through ONNX Runtime.

    The session is any object exposing ONNX Runtime's run and get_inputs, so a
    test drives the decode with a scripted model output and no model file. Each
    frame is opened, letterboxed to the model input, normalized, and scored; the
    single highest-confidence query becomes that frame's detection. The verifier
    exposes a version attribute so the engine can stamp which model produced a
    verdict.
    """

    def __init__(
        self,
        session,
        *,
        class_names: dict,
        input_size: tuple = DEFAULT_INPUT_SIZE,
        score_threshold: float = 0.0,
        input_name: Optional[str] = None,
        version: Optional[str] = None,
    ) -> None:
        self._session = session
        self._class_names = {int(k): str(v) for k, v in dict(class_names).items()}
        self._in_w = int(input_size[0])
        self._in_h = int(input_size[1])
        self._score_threshold = float(score_threshold)
        self._input_name = input_name or session.get_inputs()[0].name
        self.version = version

    def verify_frames(self, frame_paths) -> list:
        """Score every frame and return one detection dictionary per frame."""
        return [self._score_one(Path(p)) for p in frame_paths]

    def close(self) -> None:
        self._session = None

    # -- internals -------------------------------------------------------

    def _score_one(self, path: Path) -> dict:
        image = self._load_rgb(path)
        if image is None:
            return dict(_EMPTY_DETECTION)
        blob = self._preprocess(image)
        outputs = self._session.run(None, {self._input_name: blob})
        return self._decode_top(outputs)

    @staticmethod
    def _load_rgb(path: Path):
        if not path.exists():
            logger.warning("verification frame not found at %s; counted as a scored frame with no detection", path)
            return None
        from PIL import Image

        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"))

    def _preprocess(self, image_rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        resized = Image.fromarray(image_rgb).resize((self._in_w, self._in_h))
        arr = np.asarray(resized).astype(np.float32) / 255.0
        arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        blob = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
        return np.ascontiguousarray(blob, dtype=np.float32)

    def _decode_top(self, outputs) -> dict:
        logits = self._class_logits(outputs)
        if logits is None or logits.size == 0:
            return dict(_EMPTY_DETECTION)
        scores = _sigmoid(logits)  # (queries, classes)
        flat = int(np.argmax(scores))
        query, class_id = np.unravel_index(flat, scores.shape)
        confidence = float(scores[query, class_id])
        if confidence < self._score_threshold:
            return dict(_EMPTY_DETECTION)
        return {
            "gbif_usage_key": None,
            "scientific_name": self._class_names.get(int(class_id), str(int(class_id))),
            "confidence": confidence,
        }

    @staticmethod
    def _class_logits(outputs):
        """Pick the per-query class-logit tensor from the model outputs.

        A standard RF-DETR export returns two tensors, the per-query class logits
        and the per-query boxes. The boxes have a last dimension of four; the
        other two-dimensional tensor is the class logits. The top-taxon decision
        needs only the logits, so the boxes are read past.
        """
        logits = None
        for out in outputs:
            arr = np.asarray(out, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[0]
            if arr.ndim != 2:
                continue
            if arr.shape[1] == 4:
                continue  # the box tensor
            logits = arr
        return logits


def _class_names_from_session(session) -> dict:
    """Read a class-name map from the model's ONNX metadata, if present."""
    try:
        meta = session.get_modelmeta().custom_metadata_map
    except Exception:  # noqa: BLE001
        return {}
    raw = meta.get("names") if isinstance(meta, dict) else None
    if not raw:
        return {}
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {}
    if isinstance(parsed, dict):
        return {int(k): str(v) for k, v in parsed.items()}
    if isinstance(parsed, (list, tuple)):
        return {i: str(v) for i, v in enumerate(parsed)}
    return {}


def build_verifier(settings) -> RFDETRVerifier:
    """Load the desktop RF-DETR model into an ONNX Runtime verifier."""
    entry = settings.raw.get("desktop_models", {}).get("visual_rfdetr", {})
    path = entry.get("path")
    if not path:
        raise VerifierError(
            "no desktop verification model is configured under "
            "desktop_models.visual_rfdetr.path."
        )

    model_path = Path(path)
    if not model_path.is_absolute():
        model_path = Path(settings.repo_root) / model_path
    if not model_path.exists():
        raise VerifierError(
            f"the RF-DETR verification model was not found at {model_path}. Export it "
            f"to ONNX and place it there, or set its download source."
        )

    ort = _import_onnxruntime()
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    class_names = _class_names_from_session(session)
    if not class_names:
        logger.warning(
            "the RF-DETR model has no class-name metadata; verified taxa will be "
            "labelled by their numeric class index until names are added."
        )

    in_w, in_h = DEFAULT_INPUT_SIZE
    shape = session.get_inputs()[0].shape
    if isinstance(shape, (list, tuple)) and len(shape) == 4:
        if isinstance(shape[3], int):
            in_w = shape[3]
        if isinstance(shape[2], int):
            in_h = shape[2]

    return RFDETRVerifier(
        session,
        class_names=class_names,
        input_size=(in_w, in_h),
        version=entry.get("version"),
    )
