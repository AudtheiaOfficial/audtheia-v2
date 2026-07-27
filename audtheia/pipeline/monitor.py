"""Audtheia field-station detection loop.

Path: audtheia/pipeline/monitor.py

This is the heartbeat of a field station. A camera streams continuously, a
vision model checks every frame up to the accelerator's throughput, and an
object tracker associates the per-frame detections of one animal across many
frames into a single event. When that animal leaves and its track closes, the
loop fires one capture of the other sensor streams and writes exactly one
observation for the whole encounter. A frame with no detection is discarded
with nothing written, so storage is spent only on real events.

Three rules shape everything here:

  One specimen is one record. Every frame in which the tracker holds the same
  object belongs to the same observation, identified by one identifier for the
  entire life of the track, no matter how long the animal lingers. A long stay
  is never broken into separate records, because doing so would count one
  animal as many and destroy the continuous behavioural and environmental data
  that long encounters (a nesting bird, a slow-moving invertebrate) exist to
  capture. The maximum-event-duration setting bounds the length of the stored
  media segments on disk only; it never splits the record's identity.

  Detection never blocks. Capturing frames and running the model must not wait
  on the database or on the slower sensor captures. A fast producer reads,
  detects, and tracks; a separate consumer drains finished events, gathers the
  other sensors, and writes. The hand-off queue is bounded and, when full,
  briefly makes the producer wait and records that it happened, but a finished
  event is never silently dropped.

  Stored frames stay raw. Every detected frame is written to disk unaltered so
  it can be used to retrain the model, and a companion annotations file records
  the boxes and labels per frame so the interface can draw them on demand and
  reconstruct the event as video. Boxes are never burned into the saved pixels.

The camera, the accelerator, the object tracker, and the other-sensor capture
are all reached through small interfaces, so the loop runs end to end against a
scripted feed with no field hardware present, and the real drivers drop in
later without touching this loop.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from audtheia.storage.database import (
    ChildDetection,
    Database,
    EnvironmentalReading,
    Observation,
    new_id,
    utc_now_iso,
)
from audtheia.pipeline.salience import compute_salience

__all__ = [
    "Frame",
    "RawDetection",
    "TrackedDetection",
    "ChannelReading",
    "CaptureResult",
    "TrackEvent",
    "FrameSource",
    "Detector",
    "Tracker",
    "TriggerSink",
    "SupervisionByteTracker",
    "NullTriggerSink",
    "Monitor",
    "build_tracker_from_capture",
    "ISO_FORMAT",
    "DEFAULT_QUEUE_MAXSIZE",
    "DEFAULT_IMAGE_FORMAT",
    "DEFAULT_IMAGE_QUALITY",
]

logger = logging.getLogger("audtheia.pipeline.monitor")

# UTC timestamp format shared with the storage layer, so a value written here
# and a value read back compare and parse identically.
ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

# How many finished events may wait for the writer before the producer is made
# to pause. The pause protects the never-drop rule; the size keeps memory
# bounded. It is a starting value a deployment can raise if its disk is fast.
DEFAULT_QUEUE_MAXSIZE = 256

# Distinguishes "no screening model version was passed" from "the model that ran
# genuinely has no version recorded". The first falls back to the station's field
# model; the second is stored as it is, because an unstated version is honest and
# a borrowed one is not.
_MODEL_VERSION_UNSET = object()

# Saved-frame encoding fallback. The media block in the configuration is the
# home for these, and the loop reads them from there; these named constants are
# only the fallback used when a configuration does not specify a value. JPEG
# keeps the rolling buffer small while staying high enough quality for
# retraining.
DEFAULT_IMAGE_FORMAT = "jpg"
DEFAULT_IMAGE_QUALITY = 95


# ===========================================================================
# Value types crossing the interfaces
# ===========================================================================


@dataclass
class Frame:
    """One captured frame handed from the camera to the loop.

    captured_at is the UTC time the frame was taken, recorded by the camera
    source. time_provisional is 1 when that time was taken before the station's
    clock was disciplined by a satellite fix, so a downstream reader can tell a
    trusted timestamp from a provisional one.
    """

    index: int
    image: np.ndarray
    captured_at: str
    time_provisional: int = 0


@dataclass
class RawDetection:
    """One model detection in one frame, before any tracking.

    Coordinates are pixel corners (top-left x1, y1 and bottom-right x2, y2) in
    the frame. class_id is the model's class index; class_name is its label.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str


@dataclass
class TrackedDetection:
    """One detection after tracking, now carrying a stable track identity.

    track_id is the same for every frame the tracker believes shows the same
    object, which is what lets the loop treat a whole encounter as one record.
    class_name is resolved from class_id by the loop, so the tracker only has
    to preserve the reliable numeric class index.
    """

    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


@dataclass
class ChannelReading:
    """One environmental-sensor channel value captured for an event.

    Returned by the sensor capture rather than the storage row directly,
    because the loop owns the event identity and the write time and fills them
    in. status follows the missing-data vocabulary; qartod_flag is set only for
    marine channels.
    """

    channel: str
    status: str
    value: Optional[float] = None
    unit: Optional[str] = None
    qartod_flag: Optional[int] = None


@dataclass
class CaptureResult:
    """Everything the simultaneous capture gathered for one event.

    The audio clip, the satellite fix, and the environmental channels are all
    captured together for the event's window. Each field is optional so a
    capture source that is not present on this station simply leaves it empty,
    which a later reader sees as absent rather than as a measured value.
    """

    audio_clip_path: Optional[str] = None
    audio_true_duration_seconds: Optional[float] = None
    audio_capped: Optional[int] = None

    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    gps_elevation: Optional[float] = None
    gps_status: Optional[str] = None

    acoustic_model_version: Optional[str] = None
    gbif_snapshot_date: Optional[str] = None
    iucn_fetch_date: Optional[str] = None

    environmental_readings: list[ChannelReading] = field(default_factory=list)


@dataclass
class TrackEvent:
    """A finished encounter handed from the producer to the writer.

    Carries the event's identity, its true window and duration, the metadata
    the desktop uses to re-weigh the detection, and the resolved per-taxon
    children. The raw frames and the per-frame annotations have already been
    written to event_dir during the encounter; this object closes the record.
    """

    observation_id: str
    event_name: str
    station_id: str
    track_id: int

    first_seen: str
    last_seen: str
    duration: float
    frame_count: int
    time_provisional: int

    best_confidence: float
    representative_frame: Optional[str]
    screening_model_version: Optional[str]

    event_dir: Path
    segment_count: int

    # One resolved detection per taxon in the event. Each carries the label,
    # its peak confidence, and the box from the representative frame.
    children: list[dict] = field(default_factory=list)


# ===========================================================================
# Interfaces (the seams real hardware drops into)
# ===========================================================================


@runtime_checkable
class FrameSource(Protocol):
    """A source of frames. The real one wraps the camera; a test one replays a
    scripted sequence. read returns the next frame, or None at end of stream."""

    def read(self) -> Optional[Frame]: ...

    def close(self) -> None: ...


@runtime_checkable
class Detector(Protocol):
    """A per-frame detector. The real one runs the vision model on the
    accelerator; a test one returns scripted detections. class_names maps each
    class index to its label so the loop can name a tracked detection."""

    @property
    def class_names(self) -> dict[int, str]: ...

    def detect(self, frame: Frame) -> list[RawDetection]: ...

    def close(self) -> None: ...


@runtime_checkable
class Tracker(Protocol):
    """An object tracker. update is called once per frame, including frames
    with no detections, so it can age tracks whose object has left."""

    def update(self, detections: list[RawDetection], frame_index: int) -> list[TrackedDetection]: ...

    def reset(self) -> None: ...


@runtime_checkable
class TriggerSink(Protocol):
    """The simultaneous capture of the other sensor streams for one event.
    The real one composes the audio, location, and environmental captures; the
    test one captures nothing."""

    def on_event(self, event: TrackEvent) -> CaptureResult: ...


# ===========================================================================
# Object tracker built on the current, fully functional ByteTrack
# ===========================================================================


class SupervisionByteTracker:
    """Object tracker backed by the maintained ByteTrack implementation.

    The configuration names a close threshold in frames, which this maps to the
    tracker's lost-track buffer, so a brief occlusion keeps the same identity
    and the loop and the tracker agree on when a track has ended. The library
    is moving this tracker into a separate package over time; keeping it behind
    the Tracker interface means that future swap touches only this class.
    """

    def __init__(
        self,
        *,
        track_activation_threshold: float,
        lost_track_buffer: int,
        minimum_matching_threshold: float,
        frame_rate: int,
    ) -> None:
        import supervision as sv  # imported here so this stays optional at import time

        self._sv = sv
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._tracker = sv.ByteTrack(
                track_activation_threshold=track_activation_threshold,
                lost_track_buffer=lost_track_buffer,
                minimum_matching_threshold=minimum_matching_threshold,
                frame_rate=frame_rate,
            )

    def update(self, detections: list[RawDetection], frame_index: int) -> list[TrackedDetection]:
        sv = self._sv
        if detections:
            xyxy = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=float)
            confidence = np.array([d.confidence for d in detections], dtype=float)
            class_id = np.array([d.class_id for d in detections], dtype=int)
        else:
            # An empty frame is still fed in so the tracker can age and close
            # tracks whose object has gone.
            xyxy = np.empty((0, 4), dtype=float)
            confidence = np.empty((0,), dtype=float)
            class_id = np.empty((0,), dtype=int)
        dets = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            tracked = self._tracker.update_with_detections(dets)

        out: list[TrackedDetection] = []
        for i in range(len(tracked)):
            tid = tracked.tracker_id[i] if tracked.tracker_id is not None else None
            if tid is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in tracked.xyxy[i])
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
            cid = int(tracked.class_id[i]) if tracked.class_id is not None else -1
            out.append(
                TrackedDetection(
                    track_id=int(tid),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=conf,
                    class_id=cid,
                )
            )
        return out

    def reset(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._tracker.reset()


def build_tracker_from_capture(capture: dict) -> SupervisionByteTracker:
    """Build the tracker from a station's capture tuning.

    Every value comes from the configuration. The configured close-in-frames
    becomes the tracker's lost-track buffer so a momentary gap does not start a
    second record for the same animal.
    """
    bt = capture["bytetrack"]
    return SupervisionByteTracker(
        track_activation_threshold=bt["track_activation_threshold"],
        lost_track_buffer=bt["track_close_frames"],
        minimum_matching_threshold=bt["minimum_matching_threshold"],
        frame_rate=bt["frame_rate"],
    )


# ===========================================================================
# A capture sink that gathers nothing
# ===========================================================================


class NullTriggerSink:
    """A capture that gathers nothing, for running the loop before the audio,
    location, and environmental captures exist. It returns an empty result, so
    the written observation is complete and valid with those fields simply
    absent, and the real captures attach later without changing this loop."""

    def on_event(self, event: TrackEvent) -> CaptureResult:
        return CaptureResult()


# ===========================================================================
# Per-track aggregation
# ===========================================================================


def _parse_iso(value: str) -> datetime:
    return datetime.strptime(value, ISO_FORMAT).replace(tzinfo=timezone.utc)


class _TrackAggregator:
    """Accumulates one track's frames into one pending event.

    Holds only small running state in memory: the window, the running best
    frame, and per-class peak confidence. Every detected frame is written to
    disk as it arrives and its annotation appended, so memory stays flat no
    matter how long the encounter runs.
    """

    def __init__(
        self,
        *,
        track_id: int,
        station_id: str,
        station_name: str,
        visual_dir: Path,
        repo_root: Path,
        screening_model_version: Optional[str],
        max_event_duration_seconds: float,
        representative_frame_rule: str,
        image_format: str,
        image_quality: int,
    ) -> None:
        self.track_id = track_id
        self.station_id = station_id
        self.observation_id = new_id()
        self.screening_model_version = screening_model_version
        self.max_event_duration_seconds = max_event_duration_seconds
        self.representative_frame_rule = representative_frame_rule
        self.image_format = image_format
        self.image_quality = image_quality
        self.repo_root = repo_root

        self.frame_count = 0
        self.first_seen: Optional[str] = None
        self.last_seen: Optional[str] = None
        self.last_index: int = -1
        self.time_provisional = 0

        # Running best frame for the representative-frame rule.
        self.best_confidence = -1.0
        self.best_frame_file: Optional[str] = None
        self.best_box: Optional[tuple[float, float, float, float]] = None  # x, y, w, h
        self.best_class_id: Optional[int] = None

        # Peak confidence per class, to resolve the event's taxon.
        self.class_peak: dict[int, float] = {}

        # Media-segment bookkeeping. Segments bound stored-media length only;
        # the event identity never changes with them.
        self.segment_index = 0
        self._segment_start: Optional[datetime] = None

        # The event name is fixed at first sight so the on-disk folder and the
        # record share one identity from the start.
        self.event_name: Optional[str] = None
        self._station_name = station_name
        self._visual_dir = visual_dir
        self.event_dir: Optional[Path] = None
        self._annotations_path: Optional[Path] = None

    def _begin(self, frame: Frame) -> None:
        date_part = frame.captured_at[:10]  # ISO date prefix
        short = self.observation_id.split("-")[0]
        self.event_name = f"{self._station_name}_{date_part}_{short}"
        self.event_dir = self._visual_dir / self.event_name
        self.event_dir.mkdir(parents=True, exist_ok=True)
        self._annotations_path = self.event_dir / "annotations.jsonl"
        self.first_seen = frame.captured_at
        self.time_provisional = int(frame.time_provisional)
        self._segment_start = _parse_iso(frame.captured_at)

    def _roll_segment_if_needed(self, captured_at: str) -> None:
        if self._segment_start is None:
            return
        elapsed = (_parse_iso(captured_at) - self._segment_start).total_seconds()
        if elapsed >= self.max_event_duration_seconds:
            self.segment_index += 1
            self._segment_start = _parse_iso(captured_at)

    def update(self, frame: Frame, det: TrackedDetection, class_name: str, image_writer) -> None:
        if self.event_name is None:
            self._begin(frame)
        self._roll_segment_if_needed(frame.captured_at)

        self.frame_count += 1
        self.last_seen = frame.captured_at
        self.last_index = frame.index

        frame_file = f"frame_{frame.index:06d}.{self.image_format}"
        frame_path = self.event_dir / frame_file  # type: ignore[union-attr]
        # Raw pixels only. The box is recorded alongside, never drawn into the
        # saved image, so the frame stays a clean training sample.
        image_writer(frame.image, frame_path, self.image_quality)

        width = det.x2 - det.x1
        height = det.y2 - det.y1
        self.class_peak[det.class_id] = max(self.class_peak.get(det.class_id, -1.0), det.confidence)
        if det.confidence > self.best_confidence:
            self.best_confidence = det.confidence
            self.best_frame_file = frame_file
            self.best_box = (det.x1, det.y1, width, height)
            self.best_class_id = det.class_id

        record = {
            "index": frame.index,
            "file": frame_file,
            "captured_at": frame.captured_at,
            "segment": self.segment_index,
            "track_id": det.track_id,
            "class_id": det.class_id,
            "class_name": class_name,
            "confidence": det.confidence,
            "bbox_xyxy": [det.x1, det.y1, det.x2, det.y2],
        }
        with self._annotations_path.open("a", encoding="utf-8") as fh:  # type: ignore[union-attr]
            fh.write(json.dumps(record) + "\n")

    def _relpath(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.repo_root).as_posix()
        except ValueError:
            # The data directory lives outside the repository (an external
            # drive); store the absolute path so the desktop can still find it.
            return path.resolve().as_posix()

    def finalize(self, class_names: dict[int, str]) -> TrackEvent:
        assert self.first_seen is not None and self.last_seen is not None
        duration = (_parse_iso(self.last_seen) - _parse_iso(self.first_seen)).total_seconds()

        representative = None
        if self.best_frame_file is not None:
            representative = self._relpath(self.event_dir / self.best_frame_file)  # type: ignore[union-attr]

        # Resolve the event's taxon as the class with the highest peak
        # confidence. A flickering second guess stays in the per-frame
        # annotations rather than inflating the taxon count.
        children: list[dict] = []
        if self.class_peak:
            dominant_id = max(self.class_peak, key=self.class_peak.get)
            x = y = w = h = None
            if self.best_class_id == dominant_id and self.best_box is not None:
                x, y, w, h = self.best_box
            children.append(
                {
                    "class_id": dominant_id,
                    "class_name": class_names.get(dominant_id, str(dominant_id)),
                    "confidence": self.class_peak[dominant_id],
                    "bbox_x": x,
                    "bbox_y": y,
                    "bbox_w": w,
                    "bbox_h": h,
                }
            )

        # A manifest the interface reads to draw labels on demand and to
        # rebuild the event as video, split into bounded segments.
        manifest = {
            "observation_id": self.observation_id,
            "event_name": self.event_name,
            "station_id": self.station_id,
            "trigger_source": "vision",
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "duration_seconds": duration,
            "frame_count": self.frame_count,
            "representative_frame": self.best_frame_file,
            "screening_model_version": self.screening_model_version,
            "max_event_duration_seconds": self.max_event_duration_seconds,
            "segment_count": self.segment_index + 1,
            "frames_index_file": "annotations.jsonl",
        }
        (self.event_dir / "annotations.json").write_text(  # type: ignore[union-attr]
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        return TrackEvent(
            observation_id=self.observation_id,
            event_name=self.event_name,  # type: ignore[arg-type]
            station_id=self.station_id,
            track_id=self.track_id,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            duration=duration,
            frame_count=self.frame_count,
            time_provisional=self.time_provisional,
            best_confidence=max(self.best_confidence, 0.0),
            representative_frame=representative,
            screening_model_version=self.screening_model_version,
            event_dir=self.event_dir,  # type: ignore[arg-type]
            segment_count=self.segment_index + 1,
            children=children,
        )


# ===========================================================================
# The monitor
# ===========================================================================


def _default_image_writer(image: np.ndarray, path: Path, quality: int) -> None:
    """Write one frame to disk as raw pixels in the configured format.

    Kept as a small, replaceable function so a deployment can substitute a
    faster encoder without touching the loop.
    """
    from PIL import Image

    Image.fromarray(image).save(str(path), quality=quality)


class Monitor:
    """The field-station detection loop.

    Construct it with the resolved configuration, the station it runs as, an
    open database, and the four interfaces. Call run to process the stream to
    its end, or start and stop to run it against a live camera.

    The producer reads, detects, tracks, and closes events without ever waiting
    on the database. A single worker drains closed events, captures the other
    sensors, and writes one observation each. Tuning comes entirely from the
    station's capture block.
    """

    _SENTINEL = object()

    def __init__(
        self,
        *,
        settings,
        station: dict,
        db: Database,
        frame_source: FrameSource,
        detector: Detector,
        tracker: Tracker,
        trigger_sink: TriggerSink,
        queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
        screening_model_version=_MODEL_VERSION_UNSET,
        image_format: Optional[str] = None,
        image_quality: Optional[int] = None,
        image_writer=_default_image_writer,
    ) -> None:
        self._settings = settings
        self._station = station
        self._db = db
        self._frame_source = frame_source
        self._detector = detector
        self._tracker = tracker
        self._trigger_sink = trigger_sink
        self._image_writer = image_writer
        # Saved-frame encoding comes from the media configuration, with an
        # explicit constructor argument taking precedence when one is passed, so
        # the frames this loop writes and the frames the desktop verifier looks
        # for share one configured format and quality.
        image_encoding = settings.image_encoding()
        self._image_format = (
            image_format if image_format is not None
            else image_encoding.get("format", DEFAULT_IMAGE_FORMAT)
        )
        self._image_quality = int(
            image_quality if image_quality is not None
            else image_encoding.get("quality", DEFAULT_IMAGE_QUALITY)
        )

        # Privacy: detections the deployment classes as human are discarded
        # before any frame is written. The set is keyed off the station's own
        # detection-model labels, so an empty set (the default) matches nothing
        # and the discard is inert until a deployment names its human class.
        privacy = settings.privacy_config()
        self._discard_human = bool(privacy["discard_human_detections"])
        self._human_class_names = set(privacy["human_class_names"])

        capture = station["capture"]
        self._capture = capture
        self._track_close_frames = int(capture["bytetrack"]["track_close_frames"])
        self._max_event_duration_seconds = float(capture["max_event_duration_seconds"])
        self._representative_frame_rule = capture["representative_frame_rule"]
        if self._representative_frame_rule != "highest_confidence":
            raise ValueError(
                f"representative_frame_rule {self._representative_frame_rule!r} is not "
                f"supported; this loop selects the highest-confidence frame"
            )

        self._station_id = station["station_id"]
        self._station_name = station["station_name"]
        # The version recorded on an observation must name the model that actually
        # produced the detection. A caller that loaded a different screening model,
        # as the desktop does when it runs without field hardware, passes that
        # model's version here. Only when nothing is passed does this fall back to
        # the station's own field model, which is the model the field tier loads.
        # Recording a field model's version beside a detection some other model
        # made would be false provenance, which is worse than recording none.
        self._screening_model_version = (
            station["models"]["visual_pi"].get("version")
            if screening_model_version is _MODEL_VERSION_UNSET
            else screening_model_version
        )
        self._max_embedding_bytes = settings.max_embedding_bytes()

        self._visual_dir = Path(settings.path("detections_visual_dir"))
        self._repo_root = Path(settings.repo_root)

        self._queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._aggregators: dict[int, _TrackAggregator] = {}
        self._frame_index = 0

        self._stop = threading.Event()
        self._writer_thread: Optional[threading.Thread] = None

        # Counters surfaced for station telemetry. The saturation count records
        # how often the writer fell behind and the producer had to wait; it is
        # how back-pressure becomes visible without ever dropping an event.
        self.events_written = 0
        self.events_failed = 0
        self.queue_saturation_events = 0
        self.observations_skipped_no_track = 0
        self.frames_discarded_human = 0

    # -- public control --------------------------------------------------

    def run(self) -> None:
        """Process the stream to its end, then close cleanly.

        Starts the writer, runs the capture loop until the source is exhausted
        or stop is called, finalizes every still-open track, and drains the
        writer before returning.
        """
        self._start_writer()
        try:
            self._capture_loop()
        finally:
            self._finalize_all_open_tracks()
            self._queue.put(self._SENTINEL)
            if self._writer_thread is not None:
                self._writer_thread.join()

    def start(self) -> None:
        """Run the loop in the background; pair with stop for a live camera."""
        self._producer_thread = threading.Thread(target=self.run, name="audtheia-monitor", daemon=True)
        self._producer_thread.start()

    def stop(self) -> None:
        """Ask the loop to finish the current frame and shut down."""
        self._stop.set()

    # -- producer --------------------------------------------------------

    def _start_writer(self) -> None:
        self._writer_thread = threading.Thread(target=self._writer_loop, name="audtheia-monitor-writer", daemon=True)
        self._writer_thread.start()

    def _capture_loop(self) -> None:
        class_names = self._detector.class_names
        while not self._stop.is_set():
            frame = self._frame_source.read()
            if frame is None:
                break
            self._frame_index = frame.index

            detections = self._detector.detect(frame)
            tracked = self._tracker.update(detections, frame.index)

            for det in tracked:
                class_name = class_names.get(det.class_id, str(det.class_id))
                if self._is_human_class(class_name):
                    # A detection the deployment classes as human is discarded
                    # before any frame is written, so a person's image is never
                    # stored. The discard is counted, and the running count is
                    # logged, because an unexpectedly high count can mean the
                    # detection model is confusing animals for people; nothing is
                    # ever silently dropped without a trace.
                    self.frames_discarded_human += 1
                    if self.frames_discarded_human % 100 == 1:
                        logger.info(
                            "discarding frames classified as human for privacy; "
                            "%d discarded so far this run (an unexpectedly high "
                            "count can indicate the detection model confusing "
                            "animals for people)",
                            self.frames_discarded_human,
                        )
                    continue
                agg = self._aggregators.get(det.track_id)
                if agg is None:
                    agg = _TrackAggregator(
                        track_id=det.track_id,
                        station_id=self._station_id,
                        station_name=self._station_name,
                        visual_dir=self._visual_dir,
                        repo_root=self._repo_root,
                        screening_model_version=self._screening_model_version,
                        max_event_duration_seconds=self._max_event_duration_seconds,
                        representative_frame_rule=self._representative_frame_rule,
                        image_format=self._image_format,
                        image_quality=self._image_quality,
                    )
                    self._aggregators[det.track_id] = agg
                agg.update(frame, det, class_name, self._image_writer)

            self._close_stale_tracks(frame.index, class_names)

    def _is_human_class(self, class_name: str) -> bool:
        """Whether a detection's class is one the deployment treats as human.

        Discarding is on by default but keyed entirely off the configured human
        class set, which names the exact labels the station's own detection model
        uses for people. An empty set matches nothing, so a model with no human
        class makes this inert. The station's own model decides what is human;
        this never guesses and never confuses an animal for a person on its own.
        """
        return self._discard_human and class_name in self._human_class_names

    def _close_stale_tracks(self, current_index: int, class_names: dict[int, str]) -> None:
        # A track whose object has not been seen for the configured number of
        # frames is finished. The same threshold is the tracker's lost-track
        # buffer, so by the time a track is closed here the tracker has also
        # released its identity; an object that returns after that is genuinely
        # a new encounter and correctly becomes a new record.
        stale = [
            tid
            for tid, agg in self._aggregators.items()
            if agg.last_index >= 0 and (current_index - agg.last_index) > self._track_close_frames
        ]
        for tid in stale:
            agg = self._aggregators.pop(tid)
            self._enqueue(agg.finalize(class_names))

    def _finalize_all_open_tracks(self) -> None:
        class_names = self._detector.class_names
        for tid in list(self._aggregators.keys()):
            agg = self._aggregators.pop(tid)
            self._enqueue(agg.finalize(class_names))

    def _enqueue(self, event: TrackEvent) -> None:
        # Never drop a finished event. Try once without waiting; if the writer
        # is behind, record the back-pressure and then wait for room.
        try:
            self._queue.put_nowait(event)
            return
        except queue.Full:
            self.queue_saturation_events += 1
            logger.warning(
                "writer queue full; pausing capture briefly so no event is lost "
                "(saturation count %d)",
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
                logger.exception("failed to write observation for event %s", getattr(item, "event_name", "?"))

    def _write_event(self, event: TrackEvent) -> None:
        if not event.children:
            # No resolved taxon means no confirmed detection in the track; there
            # is nothing to record. This should not occur for a vision event but
            # is guarded so a malformed track never writes an empty observation.
            self.observations_skipped_no_track += 1
            logger.warning("event %s closed with no detection; nothing written", event.event_name)
            return

        capture = self._trigger_sink.on_event(event)
        created_at = utc_now_iso()

        # Provisional salience per docs/salience.md (#106): confidence-gated
        # importance from local novelty and dataset rarity. The species is the
        # track's highest-confidence taxon; its novelty and rarity counts are read
        # before this observation is inserted, so an event never seeds its own
        # baseline. A_eff is 0 on the vision path (no matched acoustic detection),
        # so the detection evidence reduces to the visual confidence. This writes
        # the provisional slot only; the desktop computes the authoritative value
        # later and this value is never treated as final.
        dominant = max(event.children, key=lambda c: c.get("confidence") or 0.0)
        species_key = dominant.get("class_name")
        # Stamp the reference snapshot behind this taxon, when its reference has
        # been fetched, so the record can disclose how current its taxonomy and
        # conservation status are. A taxon whose reference was never fetched, or
        # whose label does not match a fetched name, leaves these unset rather
        # than guessing, which reads honestly as "not stated" downstream.
        reference = self._db.find_species_reference_by_name(species_key)
        gbif_snapshot_date = reference.get("gbif_snapshot_date") if reference else capture.gbif_snapshot_date
        iucn_fetch_date = reference.get("iucn_fetch_date") if reference else capture.iucn_fetch_date
        counts = self._db.salience_counts(event.station_id, species_key)
        species_universe = len(self._detector.class_names)
        provisional_salience = compute_salience(
            event.best_confidence, 0.0, counts, k=species_universe
        )

        observation = Observation(
            id=event.observation_id,
            event_name=event.event_name,
            station_id=event.station_id,
            trigger_source="vision",
            first_seen=event.first_seen,
            last_seen=event.last_seen,
            duration=event.duration,
            data_source="model",
            created_at=created_at,
            time_provisional=event.time_provisional,
            qc_state="qc_pending",
            representative_frame=event.representative_frame,
            frame_count=event.frame_count,
            screening_confidence=event.best_confidence,
            screening_model_version=event.screening_model_version,
            acoustic_model_version=capture.acoustic_model_version,
            gbif_snapshot_date=gbif_snapshot_date,
            iucn_fetch_date=iucn_fetch_date,
            salience_provisional=provisional_salience,
            anomaly_magnitude_provisional=None,
            audio_clip_path=capture.audio_clip_path,
            audio_true_duration_seconds=capture.audio_true_duration_seconds,
            audio_capped=capture.audio_capped,
            gps_latitude=capture.gps_latitude,
            gps_longitude=capture.gps_longitude,
            gps_elevation=capture.gps_elevation,
            gps_status=capture.gps_status,
        )

        children = [
            ChildDetection(
                id=new_id(),
                observation_id=event.observation_id,
                modality="vision",
                created_at=created_at,
                confidence=child["confidence"],
                bbox_x=child["bbox_x"],
                bbox_y=child["bbox_y"],
                bbox_w=child["bbox_w"],
                bbox_h=child["bbox_h"],
                # The model's label is kept as the common name. The taxonomic
                # backbone fills the scientific name and key downstream, so the
                # record is never blocked on that lookup at capture time.
                common_name=child["class_name"],
            )
            for child in event.children
        ]

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
            for r in capture.environmental_readings
        ]

        self._db.insert_observation(
            observation,
            children=children,
            environmental_readings=readings,
            max_embedding_bytes=self._max_embedding_bytes,
        )
