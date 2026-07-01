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
    "SurfPerchModel",
    "BirdNetModel",
    "SavedModelAcousticModel",
    "build_acoustic_model",
    "write_wav_pcm16",
    "DEFAULT_AUDIO_FORMAT",
    "DEFAULT_AUDIO_SAMPLE_WIDTH_BYTES",
    "DEFAULT_ONSET_THRESHOLD",
    "DEFAULT_SILENCE_CLOSE_SECONDS",
    "DEFAULT_QUEUE_MAXSIZE",
    "MARINE_MODEL_KEY",
    "BIRDNET_MODEL_KEY",
    "CUSTOM_MODEL_KEY",
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

# The three configuration slots a station may select as its active acoustic
# model. The slot name selects the adapter and its audio preprocessing; the
# path, version, and citation for the selected slot come from configuration.
MARINE_MODEL_KEY = "marine"
BIRDNET_MODEL_KEY = "birdnet"
CUSTOM_MODEL_KEY = "custom"


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

        # Until the longitudinal baselines exist on the desktop, the provisional
        # salience is the loudest call's own confidence, written to the
        # provisional slot only; the desktop computes the authoritative value
        # later and this value is never treated as final.
        best_conf = max((c["confidence"] for c in event.children), default=0.0)
        provisional_salience = float(min(max(best_conf, 0.0), 1.0))

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

        self._db.insert_observation(observation, children=children)


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


class SurfPerchModel(SavedModelAcousticModel):
    """The coral-reef acoustic classifier, read through its reef-sound head.

    SurfPerch is a domain-adapted reef model that accepts five-second windows at
    thirty-two kilohertz and exposes a coral-reef classification head alongside
    its embedding and its inherited bird heads. Audtheia reads the reef head for
    marine deployments; the embedding and other heads are available to downstream
    analysis but are not the field-station's detection signal.
    """

    REEF_OUTPUT_KEY = "reef_label"
    SAMPLE_RATE = 32000
    WINDOW_SECONDS = 5.0

    def __init__(
        self,
        *,
        model_path: Path,
        version: Optional[str] = None,
        citation: Optional[str] = None,
        labels_path: Optional[Path] = None,
    ) -> None:
        super().__init__(
            model_path=model_path,
            output_key=self.REEF_OUTPUT_KEY,
            sample_rate=self.SAMPLE_RATE,
            window_seconds=self.WINDOW_SECONDS,
            version=version,
            citation=citation,
            labels_path=labels_path,
        )


class BirdNetModel:
    """The terrestrial bird and wildlife classifier, run through its TFLite build.

    BirdNET accepts three-second windows at forty-eight kilohertz and returns a
    score per class. The interpreter library is imported only when a real model
    is built, so importing this module never requires it.
    """

    SAMPLE_RATE = 48000
    WINDOW_SECONDS = 3.0

    def __init__(
        self,
        *,
        model_path: Path,
        version: Optional[str] = None,
        citation: Optional[str] = None,
        labels_path: Optional[Path] = None,
    ) -> None:
        try:
            from tflite_runtime.interpreter import Interpreter  # lightweight field build
        except Exception:  # noqa: BLE001 - fall back to the interpreter bundled with TensorFlow
            from tensorflow.lite import Interpreter

        self._interpreter = Interpreter(model_path=str(model_path))
        self._interpreter.allocate_tensors()
        self._input = self._interpreter.get_input_details()[0]
        self._output = self._interpreter.get_output_details()[0]
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
        return self.SAMPLE_RATE

    @property
    def window_seconds(self) -> float:
        return self.WINDOW_SECONDS

    @property
    def class_names(self) -> dict[int, str]:
        return self._class_names

    def detect(self, samples: np.ndarray, sample_rate: int) -> list[AcousticDetection]:
        audio = samples.astype(np.float32)[np.newaxis, :]
        self._interpreter.set_tensor(self._input["index"], audio)
        self._interpreter.invoke()
        scores = np.asarray(self._interpreter.get_tensor(self._output["index"])).reshape(-1)
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


def build_acoustic_model(station: dict, settings) -> AcousticModel:
    """Construct the acoustic model the station selected in configuration.

    The active slot name chooses the adapter and its audio preprocessing; the
    path, version, and citation for that slot come from configuration. Changing
    the active slot in configuration changes the model with no change to any
    code here. A slot with no model file configured raises a clear error rather
    than starting a station that cannot listen.
    """
    acoustic = station["models"]["acoustic"]
    active = acoustic["active"]
    options = acoustic.get("options", {})
    if active not in options:
        raise ValueError(
            f"active acoustic model {active!r} has no entry under "
            f"models.acoustic.options for station {station.get('station_name')!r}"
        )
    option = options[active]
    path_value = option.get("path")
    if not path_value:
        raise ValueError(
            f"acoustic model slot {active!r} has no path configured for station "
            f"{station.get('station_name')!r}; set models.acoustic.options.{active}.path"
        )
    model_path = settings.resolve_path(path_value)
    if not model_path.exists():
        raise FileNotFoundError(
            f"acoustic model file for slot {active!r} was not found at {model_path}"
        )
    labels_value = option.get("labels_path")
    labels_path = settings.resolve_path(labels_value) if labels_value else None
    version = option.get("version")
    citation = option.get("citation")

    if active == MARINE_MODEL_KEY:
        return SurfPerchModel(
            model_path=model_path, version=version, citation=citation, labels_path=labels_path
        )
    if active == BIRDNET_MODEL_KEY:
        return BirdNetModel(
            model_path=model_path, version=version, citation=citation, labels_path=labels_path
        )
    if active == CUSTOM_MODEL_KEY:
        # A user classifier in the SavedModel shape. The output head, rate, and
        # window come from the slot so a custom model needs no code here.
        return SavedModelAcousticModel(
            model_path=model_path,
            output_key=option.get("output_key", "label"),
            sample_rate=int(option.get("sample_rate", SurfPerchModel.SAMPLE_RATE)),
            window_seconds=float(option.get("window_seconds", SurfPerchModel.WINDOW_SECONDS)),
            version=version,
            citation=citation,
            labels_path=labels_path,
        )
    raise ValueError(f"unrecognised acoustic model slot {active!r}")


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
