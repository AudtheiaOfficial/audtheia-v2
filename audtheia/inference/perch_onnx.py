"""Perch 2.0 desktop sound verification adapter (ONNX Runtime).

Path: audtheia/inference/perch_onnx.py

A pure sound event has no frame for a visual verifier to re-score. Without a
second model it can only pass the acoustic confidence gate, which rests on the
field model's own confidence. This module gives the desktop a second, independent
sound model: Google DeepMind's Perch 2.0, a supervised bioacoustics model trained
on 14,795 classes (14,597 species of birds, mammals, amphibians and insects, plus
general sound classes), run through ONNX Runtime on this computer.

The verification engine hands this adapter an event's stored clip. The clip is
read, mixed to mono, resampled to the 32 kHz the model expects, and cut into
consecutive five-second windows; the last partial window is padded with silence.
For each window the adapter returns the single most probable class and its
probability. The model's raw outputs are per-class logits; they are turned into
probabilities with a softmax over all classes, so a window's probability says how
strongly the model prefers that class over every other one it knows. A sigmoid
was rejected because it saturates near one for many classes at once on a clear
call, which would make a floor meaningless.

ONNX Runtime is imported lazily, the session is an argument, and the resampler is
written with numpy alone, so the decode and windowing can be tested with a
scripted model and no model file.
"""

from __future__ import annotations

import csv
import logging
import wave
from fractions import Fraction
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("audtheia.inference.perch_onnx")

__all__ = [
    "PerchVerifier",
    "build_acoustic_verifier",
    "acoustic_verifier_configured",
    "load_labels",
    "read_wav_mono",
    "resample",
    "AcousticVerifierError",
    "SAMPLE_RATE",
    "WINDOW_SECONDS",
]

# The Perch 2.0 input contract: five seconds of mono audio at 32 kHz, one window
# per row. These are properties of the model, not of a deployment.
SAMPLE_RATE = 32000
WINDOW_SECONDS = 5.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
OUTPUT_NAME = "label"


class AcousticVerifierError(RuntimeError):
    """The sound verification model could not be loaded or run."""


def _import_onnxruntime():
    try:
        import onnxruntime as ort  # noqa: PLC0415 - lazy by design
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise AcousticVerifierError(
            "ONNX Runtime is not installed, so the sound verification model cannot run."
        ) from exc
    return ort


# -- audio -----------------------------------------------------------------------


def read_wav_mono(path: Path) -> tuple:
    """Read a PCM WAV file as float32 mono in [-1, 1]. Returns (samples, rate)."""
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        raw = handle.readframes(handle.getnframes())
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise AcousticVerifierError(f"unsupported sample width of {width} bytes in {path.name}")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32), rate


def _lowpass(cutoff: float, taps: int) -> np.ndarray:
    """A Kaiser-windowed sinc low-pass filter, cutoff as a fraction of Nyquist."""
    n = np.arange(taps) - (taps - 1) / 2.0
    h = cutoff * np.sinc(cutoff * n) * np.kaiser(taps, 5.0)
    return h / h.sum()


def resample(samples: np.ndarray, rate_in: int, rate_out: int) -> np.ndarray:
    """Rational-ratio polyphase resampling with numpy alone.

    Upsample by inserting zeros, low-pass below the lower of the two Nyquist
    frequencies, and keep every down-th sample. Close to scipy's resample_poly,
    without making scipy a runtime dependency.
    """
    if rate_in == rate_out:
        return samples.astype(np.float32)
    ratio = Fraction(rate_out, rate_in)
    up, down = ratio.numerator, ratio.denominator
    taps = 20 * max(up, down) + 1
    h = _lowpass(1.0 / max(up, down), taps) * up
    stuffed = np.zeros(len(samples) * up, dtype=np.float64)
    stuffed[::up] = samples
    filtered = np.convolve(stuffed, h, mode="same")
    out_len = int(np.ceil(len(samples) * up / down))
    return filtered[::down][:out_len].astype(np.float32)


def windows(samples: np.ndarray) -> np.ndarray:
    """Consecutive five-second windows, the last one padded with silence."""
    if len(samples) == 0:
        return np.zeros((0, WINDOW_SAMPLES), dtype=np.float32)
    count = int(np.ceil(len(samples) / WINDOW_SAMPLES))
    padded = np.zeros(count * WINDOW_SAMPLES, dtype=np.float32)
    padded[: len(samples)] = samples
    return padded.reshape(count, WINDOW_SAMPLES)


def load_labels(path: Path) -> list:
    """Class names in model output order, from a CSV whose first row is a header."""
    with open(path, encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.reader(handle) if row]
    if not rows:
        raise AcousticVerifierError(f"the labels file {path.name} is empty")
    return [row[0].strip() for row in rows[1:]]


# -- the verifier ----------------------------------------------------------------


class PerchVerifier:
    """Scores a sound clip window by window and reports each window's top class.

    configured is True, so the verification engine knows a second model is
    present. version names the model on every verdict.
    """

    configured = True

    def __init__(self, session, labels: list, *, version: str, max_windows: int = 24):
        self._session = session
        self._labels = list(labels)
        self.version = version
        self._max_windows = int(max_windows)

    def verify_clip(self, clip_path) -> list:
        """One dictionary per scored window: scientific_name and confidence.

        A missing or unreadable clip returns an empty list, which the engine
        reads as nothing scored.
        """
        path = Path(clip_path)
        if not path.is_file():
            return []
        try:
            samples, rate = read_wav_mono(path)
        except (wave.Error, OSError, EOFError, AcousticVerifierError) as exc:
            logger.warning("could not read the clip %s: %s", path.name, exc)
            return []
        batch = windows(resample(samples, rate, SAMPLE_RATE))[: self._max_windows]
        if not len(batch):
            return []
        logits = self._session.run([OUTPUT_NAME], {self._input_name(): batch})[0]
        return [self._top(row) for row in np.asarray(logits, dtype=np.float64)]

    def _input_name(self) -> str:
        try:
            return self._session.get_inputs()[0].name
        except Exception:  # noqa: BLE001 - a scripted session may not describe itself
            return "inputs"

    def _top(self, row: np.ndarray) -> dict:
        shifted = np.exp(row - row.max())
        probs = shifted / shifted.sum()
        index = int(probs.argmax())
        name = self._labels[index] if index < len(self._labels) else f"class {index}"
        return {"gbif_usage_key": None, "scientific_name": name, "confidence": float(probs[index])}

    def close(self) -> None:
        self._session = None


def _resolve(settings, value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else Path(settings.repo_root) / path


def acoustic_verifier_configured(settings) -> bool:
    """Whether a sound verification model and its labels are configured and on disk."""
    entry = (settings.raw.get("desktop_models", {}) or {}).get("acoustic_perch") or {}
    model, labels = _resolve(settings, entry.get("path")), _resolve(settings, entry.get("labels_path"))
    return bool(model and labels and model.is_file() and labels.is_file())


def build_acoustic_verifier(settings, session=None) -> Optional[PerchVerifier]:
    """The configured sound verifier, or None when none is configured.

    Raises AcousticVerifierError when a model is configured but cannot be used,
    such as a labels file whose length differs from the model's output.
    """
    entry = (settings.raw.get("desktop_models", {}) or {}).get("acoustic_perch") or {}
    model_path = _resolve(settings, entry.get("path"))
    labels_path = _resolve(settings, entry.get("labels_path"))
    if not model_path:
        return None
    if not model_path.is_file():
        raise AcousticVerifierError(f"the sound verification model was not found at {model_path}")
    if not labels_path or not labels_path.is_file():
        raise AcousticVerifierError("the sound verification model has no labels file configured")
    labels = load_labels(labels_path)
    if session is None:
        ort = _import_onnxruntime()
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    try:
        width = session.get_outputs()[[o.name for o in session.get_outputs()].index(OUTPUT_NAME)].shape[-1]
    except Exception:  # noqa: BLE001 - a scripted session may not describe itself
        width = None
    if isinstance(width, int) and width != len(labels):
        raise AcousticVerifierError(
            f"the labels file lists {len(labels)} classes but the model scores {width}; "
            f"they do not belong together")
    version = entry.get("version") or model_path.stem
    return PerchVerifier(session, labels, version=version)
