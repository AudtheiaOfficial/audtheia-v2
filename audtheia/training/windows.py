"""Choosing and cutting the windows of a recorded clip that become training examples.

Path: audtheia/training/windows.py

A confirmed clip contains the species somewhere, not in every moment. Training on
every window would teach silence as the species. This module picks the windows most
likely to hold the sound and cuts them out at the clip's own sample rate, without
changing the saved clip.

How windows are chosen

  A scorer, when one is supplied, rates each candidate window for the species the
  expert named, and windows at or above a floor are kept. The scorer is a deployed
  model, so the windows it prefers lean toward sounds that model already recognises.
  The interface says so.

  Without a scorer, or when the scorer finds nothing, windows are chosen by energy.
  The strength of sound inside a band where animal calls sit is measured along the
  clip, and the windows where it rises well above the clip's typical level are kept.
  This needs no model and carries no model bias, but a loud sound that is not the
  target can win.

  When neither finds a clear window, one window centred in the clip is used.

A clip shorter than the window is padded with silence, and one shorter than half a
window is skipped, because most of what would be learned from it is padding.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Callable, Optional

__all__ = [
    "AudioSourceError",
    "METHOD_SCORED",
    "METHOD_ENERGY",
    "METHOD_CENTRE",
    "METHOD_NOISE",
    "read_wav",
    "select_windows",
    "noise_windows",
    "cut_window",
    "write_wav_pcm16",
]

METHOD_SCORED = "model_scored"
METHOD_ENERGY = "energy"
METHOD_CENTRE = "clip_centre"
METHOD_NOISE = "noise"

# Animal calls in air and water sit well inside this band. It is clipped to the clip's
# own sample rate so a low-rate recording never asks for frequencies it cannot hold.
BAND_HZ = (500.0, 10000.0)
HOP_SECONDS = 0.25
ENERGY_FACTOR = 1.5


class AudioSourceError(RuntimeError):
    """A clip cannot be read, with a reason a person can act on."""


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy ships with the desktop
        raise AudioSourceError("preparing audio windows needs the NumPy library, which is part of the desktop requirements") from exc
    return np


def read_wav(path):
    """Read a PCM WAV file as mono floats and its sample rate. The file is not changed."""
    np = _numpy()
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            raw = handle.readframes(handle.getnframes())
    except (wave.Error, EOFError, OSError) as exc:
        raise AudioSourceError(f"the clip could not be read as a PCM WAV file ({exc})") from exc
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise AudioSourceError(f"the clip uses {width * 8} bit samples, which are not supported")
    if channels > 1:
        usable = (len(data) // channels) * channels
        data = data[:usable].reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32), int(rate)


def write_wav_pcm16(samples, path, sample_rate: int) -> None:
    """Write mono samples as a 16 bit PCM WAV file, clipping values outside the valid range."""
    np = _numpy()
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())


def _band_energy(samples, rate: int):
    """Band-limited energy per hop along the clip, as an array."""
    np = _numpy()
    frame = 1
    while frame < int(0.05 * rate):
        frame *= 2
    hop = max(1, int(HOP_SECONDS * rate))
    if len(samples) < frame:
        padded = np.zeros(frame, dtype=np.float32)
        padded[: len(samples)] = samples
        samples = padded
    window = np.hanning(frame).astype(np.float32)
    freqs = np.fft.rfftfreq(frame, d=1.0 / rate)
    low = BAND_HZ[0]
    high = min(BAND_HZ[1], 0.95 * (rate / 2.0))
    mask = (freqs >= low) & (freqs <= high)
    if not mask.any():
        mask = freqs > 0
    energies = []
    for start in range(0, len(samples) - frame + 1, hop):
        spectrum = np.fft.rfft(samples[start : start + frame] * window)
        energies.append(float(np.sum(np.abs(spectrum[mask]) ** 2)))
    return np.asarray(energies if energies else [0.0], dtype=np.float64)


def _centre_window(duration: float, window: float) -> dict:
    start = max(0.0, (duration - window) / 2.0)
    return {"start": round(start, 3), "end": round(start + window, 3), "method": METHOD_CENTRE, "score": None}


def select_windows(
    samples,
    rate: int,
    *,
    window_seconds: float,
    max_windows: int,
    selection: str = "scored_with_energy_fallback",
    scorer: Optional[Callable] = None,
    score_floor: float = 0.10,
) -> list:
    """Pick up to max_windows windows of a clip, as start and end seconds with the method used.

    Returns nothing when the clip is too short to use. Windows never overlap.
    """
    np = _numpy()
    duration = len(samples) / float(rate)
    if duration < 0.5 * window_seconds:
        return []
    if duration <= window_seconds:
        return [_centre_window(duration, window_seconds)]
    if selection == "clip_centre":
        return [_centre_window(duration, window_seconds)]

    last_start = duration - window_seconds
    count = int(last_start / HOP_SECONDS) + 1
    starts = [round(i * HOP_SECONDS, 3) for i in range(count)]
    if starts[-1] < last_start - 1e-6:
        starts.append(round(last_start, 3))

    def greedy(scores: list, accept) -> list:
        order = sorted(range(len(starts)), key=lambda i: (-scores[i], starts[i]))
        picked: list = []
        for i in order:
            if len(picked) >= max_windows:
                break
            if not accept(scores[i]):
                break
            if all(abs(starts[i] - starts[j]) >= window_seconds for j in picked):
                picked.append(i)
        return picked

    if selection == "scored_with_energy_fallback" and scorer is not None:
        scores = [float(v) for v in scorer(samples, rate, starts, window_seconds)]
        picked = greedy(scores, lambda s: s >= score_floor)
        if picked:
            return [
                {"start": starts[i], "end": round(starts[i] + window_seconds, 3), "method": METHOD_SCORED, "score": round(scores[i], 6)}
                for i in sorted(picked)
            ]

    energy = _band_energy(samples, rate)
    hops_per_window = max(1, int(round(window_seconds / HOP_SECONDS)))
    scores = []
    for start in starts:
        first = int(start / HOP_SECONDS)
        chunk = energy[first : first + hops_per_window]
        scores.append(float(chunk.mean()) if len(chunk) else 0.0)
    typical = float(np.median(energy))
    threshold = ENERGY_FACTOR * typical
    picked = greedy(scores, lambda s: s > threshold and s > 0.0)
    if picked:
        return [
            {"start": starts[i], "end": round(starts[i] + window_seconds, 3), "method": METHOD_ENERGY, "score": round(scores[i], 6)}
            for i in sorted(picked)
        ]
    return [_centre_window(duration, window_seconds)]


def noise_windows(duration: float, window_seconds: float, count: int, rng) -> list:
    """Random, non-overlapping windows from a clip an expert rejected, for the noise class."""
    if duration < 0.5 * window_seconds or count <= 0:
        return []
    if duration <= window_seconds:
        return [{"start": 0.0, "end": round(window_seconds, 3), "method": METHOD_NOISE, "score": None}]
    last_start = duration - window_seconds
    chosen: list = []
    attempts = 0
    while len(chosen) < count and attempts < 50 * count:
        attempts += 1
        start = round(rng.uniform(0.0, last_start), 3)
        if all(abs(start - other) >= window_seconds for other in chosen):
            chosen.append(start)
    return [
        {"start": s, "end": round(s + window_seconds, 3), "method": METHOD_NOISE, "score": None}
        for s in sorted(chosen)
    ]


def cut_window(samples, rate: int, start: float, window_seconds: float) -> tuple:
    """The samples of one window, padded with silence to the full length if the clip ends first."""
    np = _numpy()
    length = int(round(window_seconds * rate))
    first = int(round(start * rate))
    piece = samples[first : first + length]
    padded = len(piece) < length
    if padded:
        piece = np.concatenate([piece, np.zeros(length - len(piece), dtype=piece.dtype)])
    return piece, padded
