"""Desktop audio sources for the hardware-free acoustic pipeline.

A field station feeds the acoustic monitor from a live microphone or hydrophone.
On the desktop there is no capture device, so these sources let the same
`AcousticMonitor` read from a saved audio file, or from a URL via yt-dlp and
ffmpeg. Each source decodes to the mono, model-rate PCM the monitor expects and
hands it out in fixed-length blocks with media-time timestamps, so a recording
yields the same event durations no matter how fast the desktop processes it.

`read()` returns `None` once the audio is exhausted, which ends the monitor's
`run()` cleanly, the desktop analogue of a recorded video reaching its end.
"""
from __future__ import annotations

import shutil
import subprocess  # noqa: S404 - used only for a fixed ffmpeg availability probe path
import tempfile
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from audtheia.pipeline.acoustic import AudioBlock
from audtheia.pipeline.monitor import ISO_FORMAT

__all__ = ["FileAudioSource", "UrlAudioSource", "build_desktop_audio_source"]


def _read_wav(path: Path):
    """Decode a PCM WAV to (mono float32 in [-1, 1], sample_rate) with stdlib only.

    Supports 8-, 16-, 24-, and 32-bit integer PCM. Multi-channel audio is mixed
    down to mono by averaging channels, which is what the acoustic models want.
    """
    with wave.open(str(path), "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        rate = w.getframerate()
        frames = w.readframes(w.getnframes())
    if sampwidth == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sampwidth == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        ints = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        ints = np.where(ints >= (1 << 23), ints - (1 << 24), ints)
        data = ints.astype(np.float32) / 8388608.0
    else:
        raise ValueError(f"unsupported WAV sample width: {sampwidth} bytes")
    if n_channels > 1:
        usable = (data.size // n_channels) * n_channels
        data = data[:usable].reshape(-1, n_channels).mean(axis=1)
    return data.astype(np.float32), int(rate)


def _decode_with_ffmpeg(path: Path):
    """Decode any ffmpeg-readable file to (mono float32 in [-1, 1], rate) via a temp WAV.

    ffmpeg downmixes to mono and writes a plain PCM WAV, which `_read_wav` then
    loads; the temp file is always removed. This is the universal fallback for
    mp3, m4a, flac, ogg, and the rest when `soundfile` is not installed.
    """
    out_dir = Path(tempfile.mkdtemp(prefix="audtheia-audio-"))
    tmp = out_dir / "decoded.wav"
    try:
        subprocess.run(  # noqa: S603 - fixed argument vector, path is a decoded file path
            ["ffmpeg", "-nostdin", "-y", "-i", str(path), "-ac", "1", "-f", "wav", str(tmp)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return _read_wav(tmp)
    finally:
        for p in (tmp, out_dir):
            try:
                p.unlink() if p.is_file() else p.rmdir()
            except OSError:
                pass


def _load_audio(path: Path):
    """Decode any supported audio file to (mono float32 in [-1, 1], rate).

    A `.wav` file is read with the standard library, so the common case needs no
    extra dependency. Any other format (mp3, m4a, flac, ogg, ...) is read with
    `soundfile` when it is installed, and otherwise with ffmpeg if it is on PATH;
    only if neither is available does a clear message point at converting to WAV.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"audio file not found: {path}")
    if path.suffix.lower() == ".wav":
        return _read_wav(path)
    try:
        import soundfile as sf  # noqa: PLC0415 - optional, only for non-WAV inputs
    except ImportError:
        sf = None
    if sf is not None:
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        return np.asarray(data).mean(axis=1).astype(np.float32), int(rate)
    if shutil.which("ffmpeg") is not None:
        return _decode_with_ffmpeg(path)
    raise ValueError(
        f"{path.name}: only .wav is decoded with no extra setup. To read "
        f"{path.suffix.lower().lstrip('.') or 'this format'}, put ffmpeg on PATH or "
        f"install 'soundfile', or convert the file to a 48 kHz mono WAV."
    )


def _resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linearly resample mono audio to `dst_rate`.

    Linear interpolation is adequate for the acoustic models here; the cleanest
    result is a file already at the model rate, which needs no resampling at all.
    """
    if src_rate == dst_rate or samples.size == 0:
        return samples.astype(np.float32)
    duration = samples.size / float(src_rate)
    n_out = int(round(duration * dst_rate))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    src_t = np.linspace(0.0, duration, num=samples.size, endpoint=False)
    dst_t = np.linspace(0.0, duration, num=n_out, endpoint=False)
    return np.interp(dst_t, src_t, samples).astype(np.float32)


class FileAudioSource:
    """An `AudioSource` that replays a saved audio file for the acoustic monitor.

    The file is decoded to mono, resampled to the model's rate, and handed out in
    fixed-length blocks whose timestamps advance from a start time by media time.
    `read()` returns `None` once the file is exhausted.
    """

    def __init__(
        self,
        path,
        *,
        target_rate: int,
        block_seconds: float = 1.0,
        started_at: Optional[datetime] = None,
    ) -> None:
        samples, src_rate = _load_audio(Path(path))
        self._samples = _resample(samples, src_rate, int(target_rate))
        self._rate = int(target_rate)
        self._block = max(1, int(round(block_seconds * self._rate)))
        self._pos = 0
        self._start = (started_at or datetime.now(timezone.utc)).astimezone(timezone.utc)

    def read(self) -> Optional[AudioBlock]:
        if self._pos >= self._samples.size:
            return None
        chunk = self._samples[self._pos:self._pos + self._block]
        offset_seconds = self._pos / float(self._rate)
        captured_at = (self._start + timedelta(seconds=offset_seconds)).strftime(ISO_FORMAT)
        self._pos += chunk.size
        return AudioBlock(
            samples=np.ascontiguousarray(chunk, dtype=np.float32),
            sample_rate=self._rate,
            captured_at=captured_at,
            time_provisional=0,
        )

    def close(self) -> None:
        self._samples = np.zeros(0, dtype=np.float32)


def _fetch_url_to_wav(url: str, target_rate: int) -> Path:
    """Download a URL's audio track to a temporary WAV using yt-dlp + ffmpeg.

    yt-dlp fetches the best audio stream and ffmpeg extracts it to WAV; the
    `FileAudioSource` then handles the mono mixdown and resampling. Both tools are
    required, so a missing one raises a clear, actionable message rather than a
    deep stack trace.
    """
    if shutil.which("ffmpeg") is None:
        raise ValueError(
            "reading audio from a URL needs ffmpeg on PATH (yt-dlp uses it to "
            "extract the audio track). Install ffmpeg, or use a local .wav file."
        )
    try:
        from yt_dlp import YoutubeDL  # noqa: PLC0415 - optional, only for URL audio
    except ImportError as exc:
        raise ValueError(
            "reading audio from a URL needs yt-dlp. Install yt-dlp, or use a local "
            ".wav file."
        ) from exc
    out_dir = Path(tempfile.mkdtemp(prefix="audtheia-audio-"))
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "audio.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
    }
    with YoutubeDL(opts) as ydl:
        ydl.download([url])
    wavs = list(out_dir.glob("*.wav"))
    if not wavs:
        raise ValueError(f"could not extract an audio track from {url!r}")
    return wavs[0]


class UrlAudioSource(FileAudioSource):
    """Fetch a URL's audio to a temporary WAV (yt-dlp + ffmpeg) then replay it.

    The temporary file is removed on `close()`. Needs ffmpeg and yt-dlp present;
    otherwise `_fetch_url_to_wav` raises a message pointing at using a local file.
    """

    def __init__(self, url: str, *, target_rate: int, block_seconds: float = 1.0,
                 started_at: Optional[datetime] = None) -> None:
        self._tmp = _fetch_url_to_wav(url, int(target_rate))
        super().__init__(self._tmp, target_rate=target_rate,
                         block_seconds=block_seconds, started_at=started_at)

    def close(self) -> None:
        super().close()
        try:
            tmp = getattr(self, "_tmp", None)
            if tmp and Path(tmp).exists():
                Path(tmp).unlink()
        except OSError:
            pass


def _unquote(text: str) -> str:
    """Strip one matched pair of surrounding quotes, then whitespace.

    Windows "Copy as path" wraps a path in double quotes, and users naturally
    paste that, so `"C:\\clip.wav"` must resolve to the same file as the bare
    path. Only a genuinely matched leading/trailing quote pair is removed.
    """
    t = str(text).strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in ("'", '"'):
        t = t[1:-1].strip()
    return t


def build_desktop_audio_source(spec, *, target_rate: int, block_seconds: float = 1.0):
    """Turn a configured audio-source string into a desktop `AudioSource`.

    Accepted forms mirror the video source: `file:/path` or a bare path for a
    saved recording; `url:...`, `stream:...`, or a bare `http(s)://` address for a
    web audio source. A bare path is treated as a file so a user can paste either
    without a prefix, and a path may be quoted (as Windows "Copy as path" gives).
    """
    s = _unquote(spec)
    if not s:
        raise ValueError("no audio source configured")
    low = s.lower()
    if low.startswith("file:"):
        return FileAudioSource(_unquote(s.split(":", 1)[1]), target_rate=target_rate, block_seconds=block_seconds)
    if low.startswith(("url:", "stream:")):
        return UrlAudioSource(_unquote(s.split(":", 1)[1]), target_rate=target_rate, block_seconds=block_seconds)
    if low.startswith(("http://", "https://")):
        return UrlAudioSource(s, target_rate=target_rate, block_seconds=block_seconds)
    return FileAudioSource(s, target_rate=target_rate, block_seconds=block_seconds)
