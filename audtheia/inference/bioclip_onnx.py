"""BioCLIP 2 desktop verification adapter (ONNX Runtime).

Path: audtheia/inference/bioclip_onnx.py

BioCLIP 2 (Gu et al., NeurIPS 2025; MIT licence) names the organism in an image
by comparing the image's embedding with the embeddings of candidate species
names. The image encoder and the name embeddings are produced once by the
Prepare step (audtheia/training/bioclip_export.py); this adapter runs them with
ONNX Runtime alone, so the desktop never needs PyTorch.

For each frame the verification engine hands it, the adapter crops the organism
using the box the screening model saved with that frame, with a small margin,
and falls back to the whole frame when no box is recorded. The crop is resized
and normalised exactly as the model's own configuration says (read from the
sidecar the Prepare step wrote), embedded, and compared with every candidate
name. The probability of the best name is the softmax of the model's own logit
scale times the cosine similarities, over every candidate, so a confident answer
means the model prefers that species over thousands of others it was offered.

The adapter returns the same per-frame dictionaries as the RF-DETR adapter, so
the verification engine aggregates and words a BioCLIP 2 verdict the same way.
ONNX Runtime is imported lazily and the session is an argument, so everything
here can be tested with a scripted encoder.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("audtheia.inference.bioclip_onnx")

__all__ = [
    "BioCLIPVerifier",
    "build_bioclip_verifier",
    "bioclip_configured",
    "bioclip_label_set",
    "BioCLIPError",
    "CROP_MARGIN",
]

# The fraction of the box added on every side before cropping, so the organism's
# outline is not cut off. A property of how crops are taken, not of a deployment.
CROP_MARGIN = 0.10
BATCH = 8


class BioCLIPError(RuntimeError):
    """The BioCLIP 2 verifier could not be loaded."""


def _import_onnxruntime():
    try:
        import onnxruntime as ort  # noqa: PLC0415 - lazy by design
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise BioCLIPError("ONNX Runtime is not installed, so BioCLIP 2 cannot run.") from exc
    return ort


def _annotation_boxes(folder: Path, cache: dict) -> dict:
    """The saved box of each frame in an event folder, by file name."""
    if folder in cache:
        return cache[folder]
    boxes: dict = {}
    path = folder / "annotations.jsonl"
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            box = row.get("bbox_xyxy")
            name = row.get("file")
            if name and isinstance(box, (list, tuple)) and len(box) == 4:
                best = boxes.get(name)
                conf = row.get("confidence") or 0.0
                if best is None or conf > best[1]:
                    boxes[name] = ([float(v) for v in box], conf)
    except (OSError, ValueError):
        pass
    cache[folder] = {k: v[0] for k, v in boxes.items()}
    return cache[folder]


class BioCLIPVerifier:
    """Names the organism in each frame against every candidate species name."""

    configured = True

    def __init__(self, session, embeddings: np.ndarray, labels: list, sidecar: dict, *, version: str):
        if len(labels) != len(embeddings):
            raise BioCLIPError(f"{len(labels)} names but {len(embeddings)} name embeddings; they do not belong together")
        self._session = session
        self._embeddings = np.asarray(embeddings, dtype=np.float32)
        self._labels = list(labels)
        self._size = int(sidecar.get("image_size", 224))
        self._mean = np.asarray(sidecar.get("mean", (0.48145466, 0.4578275, 0.40821073)), dtype=np.float32)
        self._std = np.asarray(sidecar.get("std", (0.26862954, 0.26130258, 0.27577711)), dtype=np.float32)
        self._scale = float(sidecar.get("logit_scale", 100.0))
        self.version = version
        self._boxes: dict = {}

    @property
    def labels(self) -> list:
        return list(self._labels)

    def verify_frames(self, frame_paths) -> list:
        """One dictionary per frame: scientific_name and confidence, or no taxon for an unreadable frame."""
        out: list = []
        pending: list = []
        for path in frame_paths:
            pixels = self._prepare(Path(path))
            out.append(None if pixels is None else len(pending))
            if pixels is not None:
                pending.append(pixels)
        scored: list = []
        for start in range(0, len(pending), BATCH):
            batch = np.stack(pending[start:start + BATCH]).astype(np.float32)
            embeddings = self._session.run(["embedding"], {self._input_name(): batch})[0]
            scored.extend(self._top(row) for row in np.asarray(embeddings, dtype=np.float32))
        empty = {"gbif_usage_key": None, "scientific_name": None, "confidence": None}
        return [empty if slot is None else scored[slot] for slot in out]

    def _input_name(self) -> str:
        try:
            return self._session.get_inputs()[0].name
        except Exception:  # noqa: BLE001 - a scripted session may not describe itself
            return "pixels"

    def _top(self, embedding: np.ndarray) -> dict:
        norm = float(np.linalg.norm(embedding)) or 1.0
        sims = self._embeddings @ (embedding / norm)
        logits = self._scale * sims
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        index = int(probs.argmax())
        return {"gbif_usage_key": None, "scientific_name": self._labels[index], "confidence": float(probs[index])}

    def _prepare(self, path: Path) -> Optional[np.ndarray]:
        try:
            from PIL import Image  # noqa: PLC0415 - the desktop runtime ships Pillow
            with Image.open(path) as image:
                image = image.convert("RGB")
                box = _annotation_boxes(path.parent, self._boxes).get(path.name)
                if box:
                    x1, y1, x2, y2 = box
                    mx, my = (x2 - x1) * CROP_MARGIN, (y2 - y1) * CROP_MARGIN
                    left, top = max(0, int(x1 - mx)), max(0, int(y1 - my))
                    right, bottom = min(image.width, int(x2 + mx)), min(image.height, int(y2 + my))
                    if right - left >= 4 and bottom - top >= 4:
                        image = image.crop((left, top, right, bottom))
                image = self._resize_crop(image)
                arr = np.asarray(image, dtype=np.float32) / 255.0
        except (OSError, ValueError) as exc:
            logger.warning("could not read the frame %s: %s", path.name, exc)
            return None
        arr = (arr - self._mean) / self._std
        return arr.transpose(2, 0, 1)

    def _resize_crop(self, image):
        from PIL import Image  # noqa: PLC0415

        size = self._size
        scale = size / min(image.width, image.height)
        resized = image.resize((max(size, round(image.width * scale)), max(size, round(image.height * scale))),
                               Image.BICUBIC)
        left = (resized.width - size) // 2
        top = (resized.height - size) // 2
        return resized.crop((left, top, left + size, top + size))

    def close(self) -> None:
        self._session = None


def _entry(settings) -> dict:
    return (settings.raw.get("desktop_models", {}) or {}).get("visual_bioclip") or {}


def _resolve(settings, value) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else Path(settings.repo_root) / path


def _files(settings) -> tuple:
    entry = _entry(settings)
    model = _resolve(settings, entry.get("path"))
    embeddings = _resolve(settings, entry.get("embeddings_path"))
    labels = _resolve(settings, entry.get("labels_path"))
    sidecar = model.with_suffix(".json") if model else None
    if sidecar is not None and not sidecar.is_file():
        sidecar = model.parent / "bioclip2.json"
    return model, embeddings, labels, sidecar


def bioclip_configured(settings) -> bool:
    """Whether the encoder, the name embeddings and the names are configured and on disk."""
    model, embeddings, labels, _ = _files(settings)
    return bool(model and embeddings and labels and model.is_file() and embeddings.is_file() and labels.is_file())


def bioclip_label_set(settings) -> tuple:
    """(version, sorted names) of the configured BioCLIP 2 verifier, or (None, None)."""
    if not bioclip_configured(settings):
        return None, None
    _, _, labels, _ = _files(settings)
    try:
        names = json.loads(labels.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _entry(settings).get("version"), None
    return _entry(settings).get("version") or "BioCLIP 2", sorted(names)


def build_bioclip_verifier(settings, session=None) -> Optional[BioCLIPVerifier]:
    """The configured BioCLIP 2 verifier, or None when it is not configured."""
    if not _entry(settings).get("path"):
        return None
    if not bioclip_configured(settings):
        raise BioCLIPError("BioCLIP 2 is configured but its files are not all on this computer; run Prepare again")
    model, embeddings_path, labels_path, sidecar_path = _files(settings)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    embeddings = np.load(embeddings_path)
    sidecar = {}
    if sidecar_path is not None and sidecar_path.is_file():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if session is None:
        ort = _import_onnxruntime()
        session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    version = _entry(settings).get("version") or "BioCLIP 2"
    return BioCLIPVerifier(session, embeddings, labels, sidecar, version=version)
