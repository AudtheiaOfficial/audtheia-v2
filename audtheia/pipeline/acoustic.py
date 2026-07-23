"""Audtheia field-station acoustic capture and detection.

Path: audtheia/pipeline/acoustic.py

Sound is a first-class sense on a field station, not an afterthought to vision.
This module gives the station two acoustic capabilities that share one
microphone or hydrophone and one continuous audio buffer:

  Triggered capture. When the vision loop closes an encounter, it asks for the
  audio that surrounded that encounter. A continuous ring buffer means the clip
  can begin a little before the animal was first seen (pre-roll) and end a
  little after it was last seen (post-roll), so the sound that led into and out
  of the event is preserved. The clip length is capped so a very long encounter
  never writes a very long file, and whenever a clip is capped the true event
  duration is still recorded, so a shortened file never hides how long the event
  really lasted.

  Independent detection. Sound alone can announce an animal the camera never
  sees: a bird in the canopy, a whale beyond the lens. An acoustic model listens
  to the same audio stream continuously, and when it recognises something it
  opens its own observation, captures the visual frames being processed at that
  moment, and records every species it heard in that stretch of sound. A stretch
  of sound with the same voices is one observation with one identity for its
  whole length, never split into many, exactly as one lingering animal in view
  is one visual record.

Both capabilities discard sound that is not part of an event, by symmetry with
the way the vision loop discards a frame that holds no detection: storage is
spent only on real events. An optional, off-by-default soundscape sampler can
record continuous acoustic indices for deployments that want an ambient time
series alongside the event record.

The audio device and the acoustic model are reached through small interfaces,
so this module runs end to end against scripted audio with no microphone,
hydrophone, or model library present, and the real drivers drop in later without
touching this code. The acoustic model a station uses is chosen entirely in
configuration; changing it there requires no change here.
"""

from __future__ import annotations

import logging
import queue
import struct
import threading
import wave
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from audtheia.storage.database import (
    ChildDetection,
    Database,
    EnvironmentalReading,
    Observation,
    SoundscapeReading,
    new_id,
    utc_now_iso,
)
from audtheia.pipeline.monitor import (
    CaptureResult,
    ISO_FORMAT,
    TrackEvent,
)
from audtheia.pipeline.salience import compute_salience

__all__ = [
    "AudioBlock",
    "AcousticDetection",
    "VisualSnapshot",
    "AcousticEvent",
    "AudioSource",
    "AcousticModel",
    "VisualContext",
    "AudioRingBuffer",
    "NullVisualContext",
    "AcousticTriggerSink",
    "AcousticMonitor",
    "TFLiteAcousticModel",
    "SavedModelAcousticModel",
    "build_acoustic_model",
    "probe_acoustic_model",
    "write_wav_pcm16",
    "DEFAULT_AUDIO_FORMAT",
    "DEFAULT_AUDIO_SAMPLE_WIDTH_BYTES",
    "DEFAULT_ONSET_THRESHOLD",
    "DEFAULT_SILENCE_CLOSE_SECONDS",
    "DEFAULT_QUEUE_MAXSIZE",
]

logger = logging.getLogger("audtheia.pipeline.acoustic")

# Stored-clip encoding. The configuration file is the single home for tuning,
# but it does not currently carry an audio-encoding key, so these are named
# constants here rather than silently chosen. WAV PCM at 16 bits is lossless,
# universally readable, and needs no third-party library to write, which keeps
# a field station's dependencies small. Both can be promoted to configuration
# later with no change to the rest of this module.
DEFAULT_AUDIO_FORMAT = "wav"
DEFAULT_AUDIO_SAMPLE_WIDTH_BYTES = 2  # 16-bit PCM

# Detection tuning that the configuration file does not yet carry. A call at or
# above the onset threshold is treated as the start of an acoustic event; the
# event stays open until this many seconds pass with no call, so a bird that
# sings, pauses, and sings again stays one observation rather than becoming
# several. These are deployment-sensitive (a noisy reef and a quiet forest want
# different sensitivity), so they are surfaced as named constants a deployment
# can tune, pending a configuration home for per-station acoustic sensitivity.
DEFAULT_ONSET_THRESHOLD = 0.5
DEFAULT_SILENCE_CLOSE_SECONDS = 3.0

# How many finished acoustic events may wait for the writer before the reader
# is made to pause. As in the vision loop, the pause protects the never-drop
# rule while keeping memory bounded.
DEFAULT_QUEUE_MAXSIZE = 128

# Fallback audio shape, used only when a configuration carries no sample rate or
# window and none can be read from the model file. These are not tied to any
# model family: they are a last resort so a misconfigured station fails at the
# model with a clear message rather than dividing by a missing rate. A real
# deployment sets the rate once, confirmed against its own file.
DEFAULT_ACOUSTIC_SAMPLE_RATE = 48000
DEFAULT_ACOUSTIC_WINDOW_SECONDS = 3.0


# ===========================================================================
# Value types crossing the interfaces
# ===========================================================================


@dataclass
class AudioBlock:
    """One block of audio handed from the device to this module.

    samples is a one-dimensional float array of mono audio in the range -1..1.
    sample_rate is its rate in hertz. captured_at is the UTC time the block
    began, recorded by the device. time_provisional is 1 when that time was
    taken before the station's clock was disciplined by a satellite fix, so a
    downstream reader can tell a trusted timestamp from a provisional one.
    """

    samples: np.ndarray
    sample_rate: int
    captured_at: str
    time_provisional: int = 0


@dataclass
class AcousticDetection:
    """One class the acoustic model recognised in one window of audio.

    class_id is the model's class index; class_name is its label. confidence is
    the model's score in the range 0..1. A single window may yield several of
    these when several voices overlap, which is why the audio record supports
    more than one species per event.
    """

    class_id: int
    class_name: str
    confidence: float


@dataclass
class VisualSnapshot:
    """The visual context captured for a sound-triggered event.

    representative_frame is a stored-frame path if the vision pipeline had a
    usable frame in hand when the sound was heard, or empty for a pure-audio
    event. children carries any per-taxon visual detections seen in that moment,
    each a plain dictionary matching the visual child-detection shape, so a
    sound event that coincides with something in view records that co-occurrence
    as a measured fact. It never asserts that the sound came from the animal in
    view; that attribution is made downstream, clearly labelled.
    """

    representative_frame: Optional[str] = None
    children: list[dict] = field(default_factory=list)


@dataclass
class AcousticEvent:
    """A finished sound event handed from the reader to the writer.

    Carries the event's identity, its true window and duration, the audio-clip
    facts, the acoustic model version behind the calls, the visual context
    captured at the moment of the sound, and one resolved detection per species
    heard across the event.
    """

    observation_id: str
    event_name: str
    station_id: str

    first_seen: str
    last_seen: str
    duration: float
    time_provisional: int

    audio_clip_path: Optional[str]
    audio_true_duration_seconds: float
    audio_capped: int
    acoustic_model_version: Optional[str]

    representative_frame: Optional[str] = None
    children: list[dict] = field(default_factory=list)
    visual_children: list[dict] = field(default_factory=list)

    # Location and environmental sensors captured for this sound event, so an
    # audio trigger records the same physical conditions a visual trigger does.
    # These are filled by the shared location-and-environment capture when one is
    # wired, and left empty on a station that has no receiver or sensors.
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    gps_elevation: Optional[float] = None
    gps_status: Optional[str] = None
    environmental_readings: list = field(default_factory=list)


# ===========================================================================
# Interfaces (the seams real hardware and models drop into)
# ===========================================================================


@runtime_checkable
class AudioSource(Protocol):
    """A source of audio blocks. The real one wraps the microphone or the
    hydrophone-through-a-USB-converter; a test one replays scripted audio. read
    returns the next block, or None at end of stream."""

    def read(self) -> Optional[AudioBlock]: ...

    def close(self) -> None: ...


@runtime_checkable
class AcousticModel(Protocol):
    """A swappable acoustic classifier.

    sample_rate and window_seconds are the audio shape the model expects, so the
    reader can gather the right amount of audio before asking for a call.
    class_names maps each class index to its label. version and citation come
    from configuration and travel with every record for provenance and credit.
    """

    @property
    def version(self) -> Optional[str]: ...

    @property
    def citation(self) -> Optional[str]: ...

    @property
    def sample_rate(self) -> int: ...

    @property
    def window_seconds(self) -> float: ...

    @property
    def class_names(self) -> dict[int, str]: ...

    def detect(self, samples: np.ndarray, sample_rate: int) -> list[AcousticDetection]: ...

    def close(self) -> None: ...


@runtime_checkable
class VisualContext(Protocol):
    """The bridge that lets a sound event capture the frames being processed.

    When the acoustic loop opens an event, it asks the vision side for whatever
    it currently has in view. The real one reads the live vision pipeline; the
    test one returns a scripted snapshot; the null one returns nothing, which is
    correct for a station with no camera."""

    def snapshot(self, captured_at: str) -> VisualSnapshot: ...


# ===========================================================================
# A visual context that returns nothing
# ===========================================================================


class NullVisualContext:
    """A visual context that captures nothing, for a station with no camera or
    for running the acoustic loop before the vision bridge is wired. A sound
    event still writes a complete, valid observation, simply with no frame."""

    def snapshot(self, captured_at: str) -> VisualSnapshot:
        return VisualSnapshot()


# ===========================================================================
# Continuous audio ring buffer
# ===========================================================================


def _parse_iso(value: str) -> datetime:
    return datetime.strptime(value, ISO_FORMAT).replace(tzinfo=timezone.utc)


def _format_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime(ISO_FORMAT)


class AudioRingBuffer:
    """A bounded, time-stamped window of the most recent audio.

    Blocks are appended as they arrive and the oldest are dropped once the
    buffer is longer than its capacity, so memory stays flat no matter how long
    the station runs. A clip is cut from it by absolute time, which is what lets
    a vision event, closed after the fact, still recover the audio that
    surrounded it. The capacity is set from the longest clip the station can
    store plus its rolls, so the audio a clip needs is always still present.
    """

    def __init__(self, capacity_seconds: float) -> None:
        self.capacity_seconds = float(capacity_seconds)
        self._blocks: deque[AudioBlock] = deque()
        self._lock = threading.Lock()

    def append(self, block: AudioBlock) -> None:
        with self._lock:
            self._blocks.append(block)
            self._trim_locked()

    def _block_span(self, block: AudioBlock) -> tuple[datetime, datetime]:
        start = _parse_iso(block.captured_at)
        seconds = len(block.samples) / float(block.sample_rate)
        return start, start + timedelta(seconds=seconds)

    def _trim_locked(self) -> None:
        if not self._blocks:
            return
        newest_end = self._block_span(self._blocks[-1])[1]
        cutoff = newest_end - timedelta(seconds=self.capacity_seconds)
        while len(self._blocks) > 1:
            _, end = self._block_span(self._blocks[0])
            if end <= cutoff:
                self._blocks.popleft()
            else:
                break

    def extract(self, start: datetime, end: datetime) -> tuple[np.ndarray, int]:
        """Return the audio between two absolute times, and its sample rate.

        Only the portion still held is returned, so a request reaching further
        back than the buffer keeps simply yields what remains. An empty result
        is returned as an empty array rather than an error, so a clip is never
        blocked on the buffer.
        """
        with self._lock:
            blocks = list(self._blocks)
        if not blocks:
            return np.zeros(0, dtype=np.float32), 0
        rate = blocks[0].sample_rate
        pieces: list[np.ndarray] = []
        for block in blocks:
            b_start, b_end = self._block_span(block)
            if b_end <= start or b_start >= end:
                continue
            lead = max(0.0, (start - b_start).total_seconds())
            tail = max(0.0, (b_end - end).total_seconds())
            i0 = int(round(lead * block.sample_rate))
            i1 = len(block.samples) - int(round(tail * block.sample_rate))
            if i1 > i0:
                pieces.append(block.samples[i0:i1])
        if not pieces:
            return np.zeros(0, dtype=np.float32), rate
        return np.concatenate(pieces), rate


# ===========================================================================
# Writing a clip to disk
# ===========================================================================


def write_wav_pcm16(samples: np.ndarray, path: Path, sample_rate: int) -> None:
    """Write mono float audio to a 16-bit PCM WAV file.

    Kept as a small, replaceable function so a deployment can substitute a
    different encoder without touching the capture logic. Samples outside -1..1
    are clipped so a hot signal never wraps to noise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    ints = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(DEFAULT_AUDIO_SAMPLE_WIDTH_BYTES)
        wav.setframerate(int(sample_rate))
        wav.writeframes(struct.pack("<%dh" % len(ints), *ints.tolist()))


def _relpath(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        # The data directory lives outside the repository (an external drive);
        # store the absolute path so the desktop can still find it.
        return path.resolve().as_posix()


def _clip_window(
    event_start: datetime,
    event_end: datetime,
    *,
    pre_roll: float,
    post_roll: float,
    max_clip_seconds: float,
) -> tuple[datetime, datetime, float, int]:
    """Work out the clip's time span and whether it was capped.

    The clip wants to span from pre_roll before the event to post_roll after it.
    If that exceeds the maximum stored length, the span is trimmed to the cap,
    keeping the lead-in and dropping the tail, and the clip is marked capped. The
    event's true duration is returned unchanged either way, so a capped clip
    never hides how long the event really was.
    """
    true_duration = (event_end - event_start).total_seconds()
    desired_start = event_start - timedelta(seconds=pre_roll)
    desired_end = event_end + timedelta(seconds=post_roll)
    desired_len = (desired_end - desired_start).total_seconds()
    if desired_len > max_clip_seconds:
        clip_start = desired_start
        clip_end = desired_start + timedelta(seconds=max_clip_seconds)
        capped = 1
    else:
        clip_start = desired_start
        clip_end = desired_end
        capped = 0
    return clip_start, clip_end, true_duration, capped


# ===========================================================================
# Role one: triggered audio capture for a vision event
# ===========================================================================


class AcousticTriggerSink:
    """Fills the audio side of the capture for a vision event.

    This is the seam the vision loop calls when an encounter closes. It reads
    the audio that surrounded the encounter from the shared ring buffer, writes
    it as a clip, and returns the audio facts on the capture result. It leaves
    the location and environmental fields empty; those are filled by their own
    capture and merged alongside this one.

    The ring buffer is filled elsewhere (by the independent acoustic reader, or
    by a dedicated pump), so one microphone feeds both this capture and the
    independent detector.
    """

    def __init__(
        self,
        *,
        settings,
        station: dict,
        ring_buffer: AudioRingBuffer,
        acoustic_model_version: Optional[str] = None,
        clip_writer=write_wav_pcm16,
        audio_format: str = DEFAULT_AUDIO_FORMAT,
    ) -> None:
        self._ring = ring_buffer
        self._clip_writer = clip_writer
        self._audio_format = audio_format
        self._acoustic_model_version = acoustic_model_version

        audio_cfg = station["capture"]["audio"]
        self._pre_roll = float(audio_cfg["pre_roll_seconds"])
        self._post_roll = float(audio_cfg["post_roll_seconds"])
        self._max_clip_seconds = float(audio_cfg["max_clip_seconds"])

        self._audio_dir = Path(settings.path("detections_audio_dir"))
        self._repo_root = Path(settings.repo_root)

    def on_event(self, event: TrackEvent) -> CaptureResult:
        start = _parse_iso(event.first_seen)
        end = _parse_iso(event.last_seen)
        clip_start, clip_end, _true_from_window, capped = _clip_window(
            start,
            end,
            pre_roll=self._pre_roll,
            post_roll=self._post_roll,
            max_clip_seconds=self._max_clip_seconds,
        )
        samples, rate = self._ring.extract(clip_start, clip_end)

        clip_rel: Optional[str] = None
        if samples.size > 0 and rate > 0:
            clip_path = self._audio_dir / f"{event.event_name}.{self._audio_format}"
            self._clip_writer(samples, clip_path, rate)
            clip_rel = _relpath(clip_path, self._repo_root)

        return CaptureResult(
            audio_clip_path=clip_rel,
            # The true duration is the encounter's own length, never the possibly
            # shorter clip, so a capped clip never misrepresents the event.
            audio_true_duration_seconds=event.duration,
            audio_capped=capped,
            acoustic_model_version=self._acoustic_model_version,
        )


# ===========================================================================
# Role two: the independent acoustic detector
# ===========================================================================


class _AcousticAggregator:
    """Accumulates one stretch of continuous calling into one pending event.

    It holds only small running state: the event window, the per-class peak
    confidence, and when it last heard a call. A stretch of sound with the same
    voices stays one event with one identity for its whole length; a gap longer
    than the silence threshold ends it, so a later call is a new event.
    """

    def __init__(
        self,
        *,
        station_id: str,
        station_name: str,
        silence_close_seconds: float,
    ) -> None:
        self.observation_id = new_id()
        self.station_id = station_id
        self._station_name = station_name
        self._silence_close_seconds = silence_close_seconds

        self.event_name: Optional[str] = None
        self.first_seen: Optional[str] = None
        self.last_seen: Optional[str] = None
        self.time_provisional = 0
        self.class_peak: dict[int, float] = {}
        self.class_label: dict[int, str] = {}

    def _begin(self, window_start: str, time_provisional: int) -> None:
        date_part = window_start[:10]
        short = self.observation_id.split("-")[0]
        self.event_name = f"{self._station_name}_{date_part}_{short}"
        self.first_seen = window_start
        self.time_provisional = int(time_provisional)

    def add_window(
        self,
        window_start: str,
        window_end: str,
        time_provisional: int,
        detections: list[AcousticDetection],
    ) -> None:
        if self.event_name is None:
            self._begin(window_start, time_provisional)
        self.last_seen = window_end
        for det in detections:
            self.class_peak[det.class_id] = max(
                self.class_peak.get(det.class_id, -1.0), det.confidence
            )
            self.class_label[det.class_id] = det.class_name

    def is_stale(self, now: datetime) -> bool:
        if self.last_seen is None:
            return False
        gap = (now - _parse_iso(self.last_seen)).total_seconds()
        return gap > self._silence_close_seconds

    def children(self) -> list[dict]:
        # One resolved detection per species heard across the event, each at its
        # peak confidence. Every recognised voice is kept, because a soundscape
        # genuinely holds several species at once.
        out: list[dict] = []
        for class_id, peak in sorted(self.class_peak.items()):
            out.append(
                {
                    "class_id": class_id,
                    "class_name": self.class_label.get(class_id, str(class_id)),
                    "confidence": peak,
                }
            )
        return out


class AcousticMonitor:
    """The independent acoustic detection loop.

    A reader pulls audio blocks from the device, feeds the shared ring buffer,
    and runs the acoustic model over successive windows. When a window is
    recognised, an event opens; while calling continues the event grows; when
    the sound falls silent for long enough the event closes and a writer captures
    the visual context, cuts the audio clip, and writes one observation with one
    child per species heard. Sound that belongs to no event is never written.

    The reader never blocks on the database or the visual capture: a bounded
    queue hands finished events to the writer, and when the writer falls behind
    the reader pauses briefly and records that it happened rather than dropping
    an event.
    """

    _SENTINEL = object()

    def __init__(
        self,
        *,
        settings,
        station: dict,
        db: Database,
        audio_source: AudioSource,
        model: AcousticModel,
        visual_context: Optional[VisualContext] = None,
        environment_capture=None,
        ring_buffer: Optional[AudioRingBuffer] = None,
        clip_writer=write_wav_pcm16,
        onset_threshold: float = DEFAULT_ONSET_THRESHOLD,
        silence_close_seconds: float = DEFAULT_SILENCE_CLOSE_SECONDS,
        queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
        audio_format: str = DEFAULT_AUDIO_FORMAT,
    ) -> None:
        self._settings = settings
        self._station = station
        self._db = db
        self._audio_source = audio_source
        self._model = model
        self._visual_context = visual_context or NullVisualContext()
        # The location-and-environment capture, shared with the vision path, so a
        # sound-triggered event records where the station is and what its sensors
        # read exactly as a vision-triggered event does. Left unset on a station
        # with no receiver or sensors, in which case those fields stay empty.
        self._environment_capture = environment_capture
        self._clip_writer = clip_writer
        self._onset_threshold = float(onset_threshold)
        self._silence_close_seconds = float(silence_close_seconds)
        self._audio_format = audio_format

        self._station_id = station["station_id"]
        self._station_name = station["station_name"]

        audio_cfg = station["capture"]["audio"]
        self._pre_roll = float(audio_cfg["pre_roll_seconds"])
        self._post_roll = float(audio_cfg["post_roll_seconds"])
        self._max_clip_seconds = float(audio_cfg["max_clip_seconds"])

        self._audio_dir = Path(settings.path("detections_audio_dir"))
        self._repo_root = Path(settings.repo_root)

        # The buffer must hold the longest clip a station can store plus its
        # rolls, so a just-closed event can still recover its audio.
        capacity = self._max_clip_seconds + self._pre_roll + self._post_roll + 1.0
        self._ring = ring_buffer or AudioRingBuffer(capacity)

        self._window_seconds = float(model.window_seconds)
        self._target_rate = int(model.sample_rate)

        self._queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._pending: Optional[_AcousticAggregator] = None

        # Audio waiting to fill one model window, and the wall-clock start of the
        # oldest waiting sample.
        self._pending_samples: list[np.ndarray] = []
        self._pending_count = 0
        self._window_start_at: Optional[datetime] = None
        self._window_provisional = 0

        self._stop = threading.Event()
        self._writer_thread: Optional[threading.Thread] = None

        # Counters surfaced for station telemetry, mirroring the vision loop.
        self.events_written = 0
        self.events_failed = 0
        self.queue_saturation_events = 0

    # -- shared buffer access -------------------------------------------

    @property
    def ring_buffer(self) -> AudioRingBuffer:
        """The shared ring buffer, so a triggered-capture sink can read the same
        audio this reader is filling from one device."""
        return self._ring

    # -- public control --------------------------------------------------

    def run(self) -> None:
        """Process the audio stream to its end, then close cleanly."""
        self._start_writer()
        try:
            self._read_loop()
        finally:
            self._finalize_pending()
            self._queue.put(self._SENTINEL)
            if self._writer_thread is not None:
                self._writer_thread.join()

    def start(self) -> None:
        """Run the loop in the background; pair with stop for a live device."""
        self._reader_thread = threading.Thread(
            target=self.run, name="audtheia-acoustic", daemon=True
        )
        self._reader_thread.start()

    def stop(self) -> None:
        """Ask the loop to finish the current block and shut down."""
        self._stop.set()

    # -- reader ----------------------------------------------------------

    def _start_writer(self) -> None:
        self._writer_thread = threading.Thread(
            target=self._writer_loop, name="audtheia-acoustic-writer", daemon=True
        )
        self._writer_thread.start()

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            block = self._audio_source.read()
            if block is None:
                break
            self._ring.append(block)
            self._ingest(block)

    def _ingest(self, block: AudioBlock) -> None:
        # Gather audio until a full model window is available, run the model on
        # that window, then step forward. Windows do not overlap, so each moment
        # of sound is judged once.
        if self._window_start_at is None:
            self._window_start_at = _parse_iso(block.captured_at)
            self._window_provisional = int(block.time_provisional)
        self._pending_samples.append(block.samples)
        self._pending_count += len(block.samples)

        need = int(round(self._window_seconds * self._target_rate))
        while self._pending_count >= need:
            joined = np.concatenate(self._pending_samples)
            window = joined[:need]
            remainder = joined[need:]

            window_start = self._window_start_at
            window_end = window_start + timedelta(seconds=self._window_seconds)
            self._judge_window(window, window_start, window_end, self._window_provisional)

            # Advance to the next, non-overlapping window.
            self._pending_samples = [remainder] if remainder.size else []
            self._pending_count = remainder.size
            self._window_start_at = window_end

    def _judge_window(
        self,
        window: np.ndarray,
        window_start: datetime,
        window_end: datetime,
        time_provisional: int,
    ) -> None:
        detections = [
            d
            for d in self._model.detect(window, self._target_rate)
            if d.confidence >= self._onset_threshold
        ]
        start_iso = _format_iso(window_start)
        end_iso = _format_iso(window_end)

        if detections:
            if self._pending is None:
                self._pending = _AcousticAggregator(
                    station_id=self._station_id,
                    station_name=self._station_name,
                    silence_close_seconds=self._silence_close_seconds,
                )
            self._pending.add_window(start_iso, end_iso, time_provisional, detections)
        elif self._pending is not None and self._pending.is_stale(window_end):
            self._close_pending()

    def _close_pending(self) -> None:
        if self._pending is None:
            return
        agg = self._pending
        self._pending = None
        self._enqueue(self._build_event(agg))

    def _finalize_pending(self) -> None:
        if self._pending is not None:
            self._close_pending()

    def _build_event(self, agg: _AcousticAggregator) -> AcousticEvent:
        assert agg.first_seen is not None and agg.last_seen is not None
        start = _parse_iso(agg.first_seen)
        end = _parse_iso(agg.last_seen)
        clip_start, clip_end, true_duration, capped = _clip_window(
            start,
            end,
            pre_roll=self._pre_roll,
            post_roll=self._post_roll,
            max_clip_seconds=self._max_clip_seconds,
        )
        samples, rate = self._ring.extract(clip_start, clip_end)

        clip_rel: Optional[str] = None
        if samples.size > 0 and rate > 0:
            clip_path = self._audio_dir / f"{agg.event_name}.{self._audio_format}"
            self._clip_writer(samples, clip_path, rate)
            clip_rel = _relpath(clip_path, self._repo_root)

        snapshot = self._visual_context.snapshot(agg.first_seen)

        # The shared location-and-environment capture runs for the same window,
        # so a sound event carries where the station is and what its sensors read.
        # A capture that is not wired, or one that fails, leaves these fields
        # empty rather than blocking the record.
        gps_latitude = None
        gps_longitude = None
        gps_elevation = None
        gps_status = None
        environmental_readings: list = []
        if self._environment_capture is not None:
            try:
                env = self._environment_capture.capture(agg.first_seen, agg.last_seen)
                gps_latitude = env.gps_latitude
                gps_longitude = env.gps_longitude
                gps_elevation = env.gps_elevation
                gps_status = env.gps_status
                environmental_readings = list(env.environmental_readings)
            except Exception:  # noqa: BLE001 - an environment fault is isolated, never fatal
                logger.exception(
                    "environment capture failed for acoustic event %s; its "
                    "location and sensor fields will be absent",
                    agg.event_name,
                )

        return AcousticEvent(
            observation_id=agg.observation_id,
            event_name=agg.event_name,  # type: ignore[arg-type]
            station_id=agg.station_id,
            first_seen=agg.first_seen,
            last_seen=agg.last_seen,
            duration=true_duration,
            time_provisional=agg.time_provisional,
            audio_clip_path=clip_rel,
            audio_true_duration_seconds=true_duration,
            audio_capped=capped,
            acoustic_model_version=self._model.version,
            representative_frame=snapshot.representative_frame,
            children=agg.children(),
            visual_children=snapshot.children,
            gps_latitude=gps_latitude,
            gps_longitude=gps_longitude,
            gps_elevation=gps_elevation,
            gps_status=gps_status,
            environmental_readings=environmental_readings,
        )

    def _enqueue(self, event: AcousticEvent) -> None:
        # Never drop a finished event. Try once without waiting; if the writer is
        # behind, record the back-pressure and then wait for room.
        try:
            self._queue.put_nowait(event)
            return
        except queue.Full:
            self.queue_saturation_events += 1
            logger.warning(
                "acoustic writer queue full; pausing reader briefly so no event "
                "is lost (saturation count %d)",
                self.queue_saturation_events,
            )
            self._queue.put(event)

    # -- writer ----------------------------------------------------------

    def _writer_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._SENTINEL:
                break
            try:
                self._write_event(item)
                self.events_written += 1
            except Exception:  # one bad event must not stop the pipeline
                self.events_failed += 1
                logger.exception(
                    "failed to write acoustic observation for event %s",
                    getattr(item, "event_name", "?"),
                )

    def _write_event(self, event: AcousticEvent) -> None:
        created_at = utc_now_iso()

        # Provisional salience per docs/salience.md (#106): the acoustic path's
        # detection evidence is the loudest call's confidence (A_eff), corroborated
        # by any coincident visual detection (C_eff) via noisy-OR, gating importance
        # from the species' local novelty and dataset rarity. Counts are read before
        # insert so the event never seeds its own baseline. Provisional slot only;
        # the desktop computes the authoritative value later.
        a_eff = max((c.get("confidence") or 0.0 for c in event.children), default=0.0)
        c_eff = max((v.get("confidence") or 0.0 for v in event.visual_children), default=0.0)
        dominant = max(event.children, key=lambda c: c.get("confidence") or 0.0, default=None)
        species_key = dominant.get("class_name") if dominant else None
        counts = self._db.salience_counts(event.station_id, species_key)
        species_universe = len(self._model.class_names)
        provisional_salience = compute_salience(c_eff, a_eff, counts, k=species_universe)

        observation = Observation(
            id=event.observation_id,
            event_name=event.event_name,
            station_id=event.station_id,
            trigger_source="audio",
            first_seen=event.first_seen,
            last_seen=event.last_seen,
            duration=event.duration,
            data_source="model",
            created_at=created_at,
            time_provisional=event.time_provisional,
            qc_state="qc_pending",
            representative_frame=event.representative_frame,
            frame_count=len(event.visual_children) or None,
            screening_confidence=None,
            acoustic_model_version=event.acoustic_model_version,
            salience_provisional=provisional_salience,
            anomaly_magnitude_provisional=None,
            audio_clip_path=event.audio_clip_path,
            audio_true_duration_seconds=event.audio_true_duration_seconds,
            audio_capped=event.audio_capped,
            gps_latitude=event.gps_latitude,
            gps_longitude=event.gps_longitude,
            gps_elevation=event.gps_elevation,
            gps_status=event.gps_status,
        )

        children = [
            ChildDetection(
                id=new_id(),
                observation_id=event.observation_id,
                modality="audio",
                created_at=created_at,
                confidence=child["confidence"],
                # The model's label is kept as the common name. The taxonomic
                # backbone fills the scientific name and key downstream, so the
                # record is never blocked on that lookup at capture time.
                common_name=child["class_name"],
            )
            for child in event.children
        ]

        # Any visual detection that coincided with the sound is recorded as a
        # measured co-occurrence in the same event, tagged as the vision
        # modality. This never asserts the sound came from the animal in view.
        for vchild in event.visual_children:
            children.append(
                ChildDetection(
                    id=new_id(),
                    observation_id=event.observation_id,
                    modality="vision",
                    created_at=created_at,
                    confidence=vchild.get("confidence"),
                    common_name=vchild.get("class_name"),
                    bbox_x=vchild.get("bbox_x"),
                    bbox_y=vchild.get("bbox_y"),
                    bbox_w=vchild.get("bbox_w"),
                    bbox_h=vchild.get("bbox_h"),
                )
            )

        # The sensor channels captured for this sound event, each carrying its
        # own measurement status and, for marine channels, its quality flag.
        readings = [
            EnvironmentalReading(
                id=new_id(),
                observation_id=event.observation_id,
                channel=r.channel,
                status=r.status,
                created_at=created_at,
                value=r.value,
                unit=r.unit,
                qartod_flag=r.qartod_flag,
            )
            for r in event.environmental_readings
        ]

        self._db.insert_observation(
            observation, children=children, environmental_readings=readings
        )


# ===========================================================================
# Concrete acoustic models (heavy libraries import lazily)
# ===========================================================================


def _load_class_names(csv_path: Optional[Path]) -> dict[int, str]:
    """Read a model's class labels from a labels file next to the weights.

    The file is one label per line, its line number the class index. A missing
    file yields an empty map, and callers fall back to the numeric index, so a
    model with no labels file still runs and records something meaningful.
    """
    names: dict[int, str] = {}
    if csv_path is None or not csv_path.exists():
        return names
    for i, line in enumerate(csv_path.read_text(encoding="utf-8").splitlines()):
        label = line.strip()
        if label:
            names[i] = label
    return names


class SavedModelAcousticModel:
    """A TensorFlow SavedModel acoustic classifier, read through one output head.

    This is the shape SurfPerch and its relatives ship in: a saved model that,
    for one window of audio at a fixed rate, returns per-class logits under a
    named output. The heavy library is imported only when a real model is built,
    so importing this module never requires TensorFlow.
    """

    def __init__(
        self,
        *,
        model_path: Path,
        output_key: str,
        sample_rate: int,
        window_seconds: float,
        version: Optional[str] = None,
        citation: Optional[str] = None,
        labels_path: Optional[Path] = None,
    ) -> None:
        import tensorflow as tf  # imported here so this stays optional at import time

        self._tf = tf
        self._model = tf.saved_model.load(str(model_path))
        self._output_key = output_key
        self._sample_rate = int(sample_rate)
        self._window_seconds = float(window_seconds)
        self._version = version
        self._citation = citation
        self._class_names = _load_class_names(labels_path)

    @property
    def version(self) -> Optional[str]:
        return self._version

    @property
    def citation(self) -> Optional[str]:
        return self._citation

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def window_seconds(self) -> float:
        return self._window_seconds

    @property
    def class_names(self) -> dict[int, str]:
        return self._class_names

    def detect(self, samples: np.ndarray, sample_rate: int) -> list[AcousticDetection]:
        tf = self._tf
        audio = tf.convert_to_tensor(samples[np.newaxis, :], dtype=tf.float32)
        outputs = self._model.infer_tf(audio) if hasattr(self._model, "infer_tf") else self._model(audio)
        logits = np.asarray(outputs[self._output_key]).reshape(-1)
        scores = 1.0 / (1.0 + np.exp(-logits))  # logits to 0..1 scores
        out: list[AcousticDetection] = []
        for class_id, score in enumerate(scores):
            out.append(
                AcousticDetection(
                    class_id=class_id,
                    class_name=self._class_names.get(class_id, str(class_id)),
                    confidence=float(score),
                )
            )
        return out

    def close(self) -> None:
        self._model = None


def _tflite_interpreter(model_path):
    """Return an allocated TFLite interpreter, trying the runtimes in order.

    `ai-edge-litert` (LiteRT) is the current, lightweight runtime with wheels for
    Windows, macOS, and Linux, so it is tried first; `tflite_runtime` is the slim
    field build used on the Pi; `tensorflow.lite` is the last-resort fallback for
    an environment that only has full TensorFlow (whose `tf.lite` path has become
    unreliable in recent releases). The first runtime that imports is used.
    """
    last_err = None
    for import_interpreter in (
        lambda: __import__("ai_edge_litert.interpreter", fromlist=["Interpreter"]).Interpreter,
        lambda: __import__("tflite_runtime.interpreter", fromlist=["Interpreter"]).Interpreter,
        lambda: __import__("tensorflow.lite", fromlist=["Interpreter"]).Interpreter,
    ):
        try:
            Interpreter = import_interpreter()
        except Exception as exc:  # noqa: BLE001 - try the next runtime
            last_err = exc
            continue
        interpreter = Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        return interpreter
    raise ImportError(
        "no TFLite runtime is available to load the acoustic model. Install one of "
        "'ai-edge-litert' (recommended), 'tflite-runtime', or 'tensorflow'. Last "
        f"import error: {last_err}"
    )


class TFLiteAcousticModel:
    """A TFLite acoustic classifier, run through the first available runtime.

    The model takes one window of audio at a fixed sample rate and returns a
    score per class. Its audio shape (sample rate and window length) comes from
    configuration rather than being assumed, because the same TFLite form is used
    by models that want different rates and windows, and nothing here is tied to
    one model family or one taxon. The interpreter library is imported only when
    a real model is built, so importing this module never requires it.
    """

    def __init__(
        self,
        *,
        model_path: Path,
        sample_rate: int,
        window_seconds: float,
        version: Optional[str] = None,
        citation: Optional[str] = None,
        labels_path: Optional[Path] = None,
    ) -> None:
        self._interpreter = _tflite_interpreter(model_path)
        self._input = self._interpreter.get_input_details()[0]
        self._output = self._interpreter.get_output_details()[0]
        self._sample_rate = int(sample_rate)
        self._window_seconds = float(window_seconds)
        self._version = version
        self._citation = citation
        self._class_names = _load_class_names(labels_path)

    @property
    def version(self) -> Optional[str]:
        return self._version

    @property
    def citation(self) -> Optional[str]:
        return self._citation

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def window_seconds(self) -> float:
        return self._window_seconds

    @property
    def class_names(self) -> dict[int, str]:
        return self._class_names

    def detect(self, samples: np.ndarray, sample_rate: int) -> list[AcousticDetection]:
        audio = samples.astype(np.float32)[np.newaxis, :]
        self._interpreter.set_tensor(self._input["index"], audio)
        self._interpreter.invoke()
        logits = np.asarray(self._interpreter.get_tensor(self._output["index"])).reshape(-1)
        # A TFLite classifier emits per-class logits, not probabilities; a
        # numerically stable sigmoid maps them to the [0, 1] confidences the
        # onset threshold expects.
        scores = np.where(
            logits >= 0,
            1.0 / (1.0 + np.exp(-logits)),
            np.exp(logits) / (1.0 + np.exp(logits)),
        )
        out: list[AcousticDetection] = []
        for class_id, score in enumerate(scores):
            out.append(
                AcousticDetection(
                    class_id=class_id,
                    class_name=self._class_names.get(class_id, str(class_id)),
                    confidence=float(score),
                )
            )
        return out

    def close(self) -> None:
        self._interpreter = None


def _is_saved_model_dir(path: Path) -> bool:
    """Whether a path is a TensorFlow SavedModel directory."""
    return path.is_dir() and (path / "saved_model.pb").exists()


def _probe_tflite_shape(model_path: Path) -> dict:
    """Read what a TFLite acoustic model's own tensors state, no rate assumed.

    Returns the input window length in samples and the class-head width, both
    read from the file, plus the adapter name. The sample rate is deliberately
    absent: it is not stored in the file and cannot be derived from the sample
    count alone, so it is proposed or entered elsewhere, never guessed here.
    """
    interpreter = _tflite_interpreter(model_path)
    in_shape = list(interpreter.get_input_details()[0].get("shape", []))
    out_shape = list(interpreter.get_output_details()[0].get("shape", []))
    input_samples = int(in_shape[-1]) if in_shape else None
    class_count = int(out_shape[-1]) if out_shape else None
    return {"adapter": "tflite", "input_samples": input_samples, "class_count": class_count}


def probe_acoustic_model(model_path, *, proposals: Optional[dict] = None) -> dict:
    """Read an acoustic model file's audio shape, keeping read and proposed apart.

    What can be read from the file is read; what cannot be read is proposed, and
    the two are never conflated, so a configuration never presents a guessed
    sample rate as a measured one. A `.tflite` file's input window length and
    class-head width are read from its tensors. A SavedModel directory is
    reported by form only, since its input shape is not reliably introspectable
    here. The sample rate is never in the file: when a `proposals` table keyed by
    the measured fingerprint `"<input_samples>x<class_count>"` carries an entry,
    that rate is offered as proposed and the window is derived from it; otherwise
    the rate is left for a person to enter.

    Returns a dict with three parts kept separate:
      read     - facts taken from the file (adapter, input_samples, class_count)
      proposed - values offered but not read (sample_rate)
      derived  - values computed from a proposed value (window_seconds)
    """
    path = Path(model_path)
    read: dict = {"adapter": None, "input_samples": None, "class_count": None}
    if _is_saved_model_dir(path):
        read["adapter"] = "savedmodel"
    elif path.suffix.lower() == ".tflite":
        read = _probe_tflite_shape(path)

    proposed: dict = {"sample_rate": None}
    derived: dict = {"window_seconds": None}

    input_samples = read.get("input_samples")
    class_count = read.get("class_count")
    if proposals and input_samples is not None and class_count is not None:
        fingerprint = f"{input_samples}x{class_count}"
        entry = proposals.get(fingerprint)
        rate = entry.get("sample_rate") if isinstance(entry, dict) else entry
        if isinstance(rate, (int, float)) and not isinstance(rate, bool) and rate > 0:
            proposed["sample_rate"] = int(rate)
            derived["window_seconds"] = round(input_samples / float(rate), 4)

    return {"read": read, "proposed": proposed, "derived": derived}


def build_acoustic_model(station: dict, settings) -> AcousticModel:
    """Construct the station's single acoustic model from its flat configuration.

    A station listens with one model, described by one flat block: models.acoustic
    carries the path, the labels path, the audio shape (sample_rate,
    window_seconds, output_key), and the version and citation. The adapter is
    chosen from the file's own form, never from a configured name, so the platform
    stays indifferent to what is being studied: a `.tflite` file is run through the
    TFLite runtime, a SavedModel directory through TensorFlow. A block with no path
    raises a clear error naming models.acoustic.path rather than starting a station
    that cannot listen.
    """
    acoustic = station["models"].get("acoustic") or {}
    path_value = acoustic.get("path")
    if not path_value:
        raise ValueError(
            f"station {station.get('station_name')!r} has no acoustic model set; "
            f"set models.acoustic.path to a model file"
        )
    model_path = settings.resolve_path(path_value)
    if not model_path.exists():
        raise FileNotFoundError(f"the acoustic model file was not found at {model_path}")
    labels_value = acoustic.get("labels_path")
    labels_path = settings.resolve_path(labels_value) if labels_value else None
    version = acoustic.get("version")
    citation = acoustic.get("citation")
    sample_rate = acoustic.get("sample_rate")
    window_seconds = acoustic.get("window_seconds")
    rate = int(sample_rate) if sample_rate else DEFAULT_ACOUSTIC_SAMPLE_RATE
    window = float(window_seconds) if window_seconds else DEFAULT_ACOUSTIC_WINDOW_SECONDS

    if _is_saved_model_dir(model_path):
        return SavedModelAcousticModel(
            model_path=model_path,
            output_key=acoustic.get("output_key") or "label",
            sample_rate=rate,
            window_seconds=window,
            version=version,
            citation=citation,
            labels_path=labels_path,
        )
    if model_path.suffix.lower() == ".tflite":
        return TFLiteAcousticModel(
            model_path=model_path,
            sample_rate=rate,
            window_seconds=window,
            version=version,
            citation=citation,
            labels_path=labels_path,
        )
    raise ValueError(
        f"the acoustic model at {model_path} is not a form this runtime can load; "
        f"a '.tflite' file or a TensorFlow SavedModel directory is expected"
    )


# ===========================================================================
# Optional continuous soundscape sampler (off by default)
# ===========================================================================


class SoundscapeSampler:
    """Records continuous acoustic indices, when a deployment turns it on.

    This is off by default and independent of any single event: it samples the
    ambient sound on a fixed cadence and writes a small time series. A metric
    function turns one window of audio into a named number; a deployment supplies
    the metrics it wants. When the configuration leaves it disabled, it never
    runs and writes nothing.
    """

    def __init__(
        self,
        *,
        settings,
        station: dict,
        db: Database,
        metric_functions: Optional[dict] = None,
    ) -> None:
        self._db = db
        self._station_id = station["station_id"]
        soundscape = station["capture"].get("soundscape", {})
        self._enabled = bool(soundscape.get("enabled", False))
        self._metrics = list(soundscape.get("metrics", []))
        self._cadence_seconds = float(soundscape.get("cadence_seconds", 60))
        self._metric_functions = metric_functions or {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def sample(self, samples: np.ndarray, sample_rate: int, recorded_at: Optional[str] = None) -> int:
        """Compute and store the configured indices for one window of audio.

        Returns the number of readings written, which is zero when the sampler
        is disabled or no metric is configured, so a caller can loop over it
        unconditionally without special-casing the off state.
        """
        if not self._enabled or not self._metrics:
            return 0
        stamp = recorded_at or utc_now_iso()
        written = 0
        for metric in self._metrics:
            fn = self._metric_functions.get(metric)
            if fn is None:
                continue
            value = float(fn(samples, sample_rate))
            self._db.insert_soundscape_reading(
                SoundscapeReading(
                    id=new_id(),
                    station_id=self._station_id,
                    recorded_at=stamp,
                    metric=metric,
                    value=value,
                    created_at=utc_now_iso(),
                )
            )
            written += 1
        return written
