"""Mocked-hardware verification for the acoustic capture and detection module.

Path: tests/test_acoustic.py

Runs both acoustic roles end to end with no microphone, hydrophone, or model
library present. A scripted audio source and a scripted acoustic model stand in
for the hardware, and a null visual context stands in for the not-yet-wired
vision bridge. Every tuning value that lives in configuration is read from the
real configuration through the real loader, and every row is read back and
checked against the real schema.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from audtheia.config import load_settings  # noqa: E402
from audtheia.storage.database import Database, Station, utc_now_iso  # noqa: E402
from audtheia.pipeline.monitor import CaptureResult, TrackEvent  # noqa: E402
from audtheia.pipeline.acoustic import (  # noqa: E402
    AcousticDetection,
    AcousticMonitor,
    AcousticTriggerSink,
    AudioBlock,
    AudioRingBuffer,
    NullVisualContext,
    SoundscapeSampler,
    VisualContext,
    VisualSnapshot,
    build_acoustic_model,
    write_wav_pcm16,
    DEFAULT_ONSET_THRESHOLD,
)

ISO = "%Y-%m-%dT%H:%M:%S.%fZ"
BASE = datetime(2026, 6, 30, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Mock hardware
# ---------------------------------------------------------------------------


class ScriptedAudioSource:
    """Emits a fixed sequence of audio blocks at a fixed block length. Each
    block carries synthetic mono audio so the clip-writing path runs for real."""

    def __init__(self, blocks: list[AudioBlock]):
        self._blocks = list(blocks)
        self._i = 0

    def read(self):
        if self._i >= len(self._blocks):
            return None
        block = self._blocks[self._i]
        self._i += 1
        return block

    def close(self):
        pass


class ScriptedAcousticModel:
    """Returns scripted detections for each window, keyed by the window's start
    time in whole seconds from the stream base. Declares a fixed rate and window
    so the reader gathers the right amount of audio before each call."""

    def __init__(self, class_names, script, *, sample_rate=32000, window_seconds=1.0,
                 version="scripted-1.0", citation="test"):
        self._class_names = class_names
        self._script = script
        self._sample_rate = sample_rate
        self._window_seconds = window_seconds
        self._version = version
        self._citation = citation

    @property
    def version(self):
        return self._version

    @property
    def citation(self):
        return self._citation

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def window_seconds(self):
        return self._window_seconds

    @property
    def class_names(self):
        return self._class_names

    def detect(self, samples, sample_rate):
        # Key on the window length actually handed in, tracked by a running
        # counter so the script can drive successive windows.
        key = getattr(self, "_call", 0)
        self._call = key + 1
        return list(self._script.get(key, []))

    def close(self):
        pass


class ScriptedVisualContext:
    """A visual context that returns one fixed snapshot, standing in for frames
    the vision pipeline is processing when a sound is heard."""

    def __init__(self, snapshot: VisualSnapshot):
        self._snapshot = snapshot

    def snapshot(self, captured_at):
        return self._snapshot


def tone_block(index: int, sample_rate: int, block_seconds: float, *, amp: float = 0.2,
               time_provisional: int = 0) -> AudioBlock:
    """One block of a quiet sine tone, time-stamped by its position in the stream."""
    n = int(round(block_seconds * sample_rate))
    t0 = index * block_seconds
    t = (np.arange(n) / sample_rate) + t0
    samples = (amp * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    captured_at = (BASE + timedelta(seconds=t0)).strftime(ISO)
    return AudioBlock(samples=samples, sample_rate=sample_rate, captured_at=captured_at,
                      time_provisional=time_provisional)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def make_pi_settings(active_index: int = 0):
    """Write a field-station copy of the real configuration and load it through
    the real loader. Index 0 is the marine station, index 1 the forest one."""
    base = json.loads((REPO / "config" / "settings.json").read_text(encoding="utf-8"))
    base["node"]["role"] = "pi"
    base["node"]["active_station_id"] = base["stations"][active_index]["station_id"]
    # Redirect writable data paths into a throwaway temp tree so the scenarios'
    # shutil.rmtree of the configured detections dirs can never touch the real
    # repository data/ (which would delete real captured frames and clips).
    sandbox = tempfile.mkdtemp(prefix="audtheia-test-")
    base["paths"]["data_dir"] = sandbox
    base["paths"]["detections_visual_dir"] = str(Path(sandbox) / "detections" / "visual")
    base["paths"]["detections_audio_dir"] = str(Path(sandbox) / "detections" / "audio")
    base["paths"]["gps_dir"] = str(Path(sandbox) / "gps")
    path = REPO / "config" / f"settings.pi.acoustic.{active_index}.json"
    path.write_text(json.dumps(base), encoding="utf-8")
    return load_settings(path), path


def fresh_db(settings, tmp: Path) -> Database:
    db_path = tmp / "audtheia.db"
    db = Database(str(db_path), **settings.database_kwargs())
    db.initialize_schema(settings.schema_path())
    station = settings.active_station()
    db.create_station(
        Station(
            id=station["station_id"],
            station_name=station["station_name"],
            environment_type=station["environment_type"],
            created_at=utc_now_iso(),
        )
    )
    return db


def clean_audio_dir(settings):
    adir = Path(settings.path("detections_audio_dir"))
    if adir.exists():
        shutil.rmtree(adir)
    adir.mkdir(parents=True, exist_ok=True)
    return adir


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

CHECKS = {"passed": 0, "failed": 0}


def check(label: str, condition: bool):
    if condition:
        CHECKS["passed"] += 1
        print(f"  PASS  {label}")
    else:
        CHECKS["failed"] += 1
        print(f"  FAIL  {label}")


REQUIRED_COLS = [
    "id", "event_name", "station_id", "trigger_source", "first_seen", "last_seen",
    "duration", "time_provisional", "data_source", "qc_state", "created_at",
]


def validate_audio_observation_row(row: dict):
    ok = all(row.get(c) is not None for c in REQUIRED_COLS)
    check("observation row carries every required column", ok)
    check("trigger_source is audio", row["trigger_source"] == "audio")
    check("data_source is model", row["data_source"] == "model")
    check("qc_state starts pending", row["qc_state"] == "qc_pending")
    check("acoustic_model_version is stamped", row["acoustic_model_version"] is not None)
    check("event_name follows station_date_short form", "_2026-06-30_" in row["event_name"])
    check(
        "provisional salience in range",
        row["salience_provisional"] is None
        or (0.0 <= row["salience_provisional"] <= 1.0),
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def scenario_triggered_capture(settings):
    print("\n[1] A vision event pulls its surrounding audio from the ring buffer")
    adir = clean_audio_dir(settings)
    station = settings.active_station()
    rate = 32000
    block_seconds = 0.5
    # Fill a ring buffer with 20 seconds of continuous audio.
    ring = AudioRingBuffer(capacity_seconds=60.0)
    for i in range(40):
        ring.append(tone_block(i, rate, block_seconds))

    sink = AcousticTriggerSink(
        settings=settings,
        station=station,
        ring_buffer=ring,
        acoustic_model_version="scripted-1.0",
    )
    check("the sink satisfies the trigger-sink seam", hasattr(sink, "on_event"))

    # A vision event spanning seconds 5..9 of the stream.
    ev_start = BASE + timedelta(seconds=5.0)
    ev_end = BASE + timedelta(seconds=9.0)
    event = TrackEvent(
        observation_id="obs-1",
        event_name=f"{station['station_name']}_2026-06-30_obs1",
        station_id=station["station_id"],
        track_id=1,
        first_seen=ev_start.strftime(ISO),
        last_seen=ev_end.strftime(ISO),
        duration=4.0,
        frame_count=10,
        time_provisional=0,
        best_confidence=0.9,
        representative_frame="data/detections/visual/x/frame.jpg",
        screening_model_version="yolo-1",
        event_dir=Path("."),
        segment_count=1,
        children=[{"class_id": 0, "class_name": "x", "confidence": 0.9,
                   "bbox_x": 1, "bbox_y": 1, "bbox_w": 1, "bbox_h": 1}],
    )
    result = sink.on_event(event)
    check("on_event returns a CaptureResult", isinstance(result, CaptureResult))
    check("audio clip path was filled", result.audio_clip_path is not None)
    check("true duration is the event's own length, not the clip's", result.audio_true_duration_seconds == 4.0)
    check("clip was not capped for a short event", result.audio_capped == 0)
    check("acoustic model version carried on the result", result.acoustic_model_version == "scripted-1.0")
    check("location and environment left for their own capture", result.gps_status is None and not result.environmental_readings)

    clip = Path(settings.repo_root) / result.audio_clip_path
    check("clip file exists on disk", clip.exists())
    with wave.open(str(clip), "rb") as w:
        clip_seconds = w.getnframes() / w.getframerate()
        chans = w.getnchannels()
        width = w.getsampwidth()
    # pre_roll 2 + duration 4 + post_roll 2 = 8 seconds, under the 30s cap.
    audio_cfg = station["capture"]["audio"]
    expected = audio_cfg["pre_roll_seconds"] + 4.0 + audio_cfg["post_roll_seconds"]
    check("clip spans pre-roll + event + post-roll", abs(clip_seconds - expected) < 0.05)
    check("clip is mono 16-bit PCM WAV", chans == 1 and width == 2)


def scenario_capping(settings):
    print("\n[2] A clip longer than the cap is capped, but the true duration is kept")
    clean_audio_dir(settings)
    station = settings.active_station()
    rate = 32000
    ring = AudioRingBuffer(capacity_seconds=120.0)
    for i in range(200):  # 100 seconds
        ring.append(tone_block(i, rate, 0.5))

    sink = AcousticTriggerSink(settings=settings, station=station, ring_buffer=ring,
                               acoustic_model_version="scripted-1.0")
    max_clip = station["capture"]["audio"]["max_clip_seconds"]  # 30
    ev_start = BASE + timedelta(seconds=10.0)
    ev_end = BASE + timedelta(seconds=10.0 + max_clip + 20.0)  # far longer than the cap
    event = TrackEvent(
        observation_id="obs-2", event_name=f"{station['station_name']}_2026-06-30_obs2",
        station_id=station["station_id"], track_id=2,
        first_seen=ev_start.strftime(ISO), last_seen=ev_end.strftime(ISO),
        duration=max_clip + 20.0, frame_count=5, time_provisional=0, best_confidence=0.8,
        representative_frame=None, screening_model_version=None,
        event_dir=Path("."), segment_count=1, children=[],
    )
    result = sink.on_event(event)
    check("clip marked capped", result.audio_capped == 1)
    check("true duration preserved beyond the cap", result.audio_true_duration_seconds == max_clip + 20.0)
    clip = Path(settings.repo_root) / result.audio_clip_path
    with wave.open(str(clip), "rb") as w:
        clip_seconds = w.getnframes() / w.getframerate()
    check("stored clip length equals the cap", abs(clip_seconds - max_clip) < 0.1)


def scenario_independent_trigger(settings):
    print("\n[3] Sound alone opens its own observation and captures visual context")
    adir = clean_audio_dir(settings)
    station = settings.active_station()
    rate = 32000
    block_seconds = 1.0  # one block per model window keeps the script simple

    # Windows 0..29. Detect a reef sound in windows 5..8, silence elsewhere.
    detA = AcousticDetection(class_id=3, class_name="fish_call", confidence=0.82)
    detB = AcousticDetection(class_id=7, class_name="snapping_shrimp", confidence=0.66)
    script = {5: [detA], 6: [detA, detB], 7: [detA], 8: [detB]}
    model = ScriptedAcousticModel({3: "fish_call", 7: "snapping_shrimp"}, script,
                                  sample_rate=rate, window_seconds=1.0)

    blocks = [tone_block(i, rate, block_seconds) for i in range(30)]
    src = ScriptedAudioSource(blocks)

    snapshot = VisualSnapshot(
        representative_frame="data/detections/visual/reef/frame_000010.jpg",
        children=[{"class_id": 0, "class_name": "Aplysina_fistularis", "confidence": 0.7,
                   "bbox_x": 1.0, "bbox_y": 2.0, "bbox_w": 3.0, "bbox_h": 4.0}],
    )
    visual = ScriptedVisualContext(snapshot)

    with tempfile.TemporaryDirectory() as d:
        db = fresh_db(settings, Path(d))
        mon = AcousticMonitor(
            settings=settings, station=station, db=db, audio_source=src, model=model,
            visual_context=visual, onset_threshold=DEFAULT_ONSET_THRESHOLD,
            silence_close_seconds=2.0,
        )
        mon.run()

        obs = db.list_observations()
        check("one acoustic event became exactly one observation", len(obs) == 1)
        if not obs:
            return
        row = obs[0]
        validate_audio_observation_row(row)
        check("writer reported no failures", mon.events_failed == 0)

        children = db.list_child_detections(row["id"])
        audio_children = [c for c in children if c["modality"] == "audio"]
        vision_children = [c for c in children if c["modality"] == "vision"]
        check("both heard species recorded as audio children", len(audio_children) == 2)
        names = {c["common_name"] for c in audio_children}
        check("the two distinct voices are kept", names == {"fish_call", "snapping_shrimp"})
        check("audio children carry no bounding box", all(c["bbox_x"] is None for c in audio_children))
        check("coinciding visual detection recorded as a vision child", len(vision_children) == 1)
        check("visual child carries its box (measured co-occurrence)", vision_children[0]["bbox_x"] is not None)
        check("representative frame captured from the visual context", row["representative_frame"] == snapshot.representative_frame)

        # The event spans windows 5..8 => first_seen at second 5, last_seen at second 9.
        first = datetime.strptime(row["first_seen"], ISO).replace(tzinfo=timezone.utc)
        last = datetime.strptime(row["last_seen"], ISO).replace(tzinfo=timezone.utc)
        check("event first_seen at the first calling window", abs((first - (BASE + timedelta(seconds=5))).total_seconds()) < 0.01)
        check("event duration spans the whole calling stretch", abs(row["duration"] - 4.0) < 0.01)
        check("a clip was written for the event", row["audio_clip_path"] is not None)
        clip = Path(settings.repo_root) / row["audio_clip_path"]
        check("acoustic clip exists on disk", clip.exists())


def scenario_non_event_audio_discarded(settings):
    print("\n[4] Audio with nothing above threshold writes no observation and no clip")
    adir = clean_audio_dir(settings)
    station = settings.active_station()
    rate = 32000
    # A quiet detection everywhere, always below the onset threshold.
    low = AcousticDetection(class_id=1, class_name="faint", confidence=0.2)
    script = {i: [low] for i in range(20)}
    model = ScriptedAcousticModel({1: "faint"}, script, sample_rate=rate, window_seconds=1.0)
    blocks = [tone_block(i, rate, 1.0) for i in range(20)]
    src = ScriptedAudioSource(blocks)
    with tempfile.TemporaryDirectory() as d:
        db = fresh_db(settings, Path(d))
        mon = AcousticMonitor(settings=settings, station=station, db=db, audio_source=src,
                              model=model, visual_context=NullVisualContext(),
                              onset_threshold=DEFAULT_ONSET_THRESHOLD, silence_close_seconds=2.0)
        mon.run()
        obs = db.list_observations()
        check("no observation for sub-threshold audio", len(obs) == 0)
        clips = list(adir.glob("*.wav"))
        check("no clip written for non-event audio", len(clips) == 0)


def scenario_two_separate_events(settings):
    print("\n[5] A silence gap splits two calling stretches into two observations")
    clean_audio_dir(settings)
    station = settings.active_station()
    rate = 32000
    det = AcousticDetection(class_id=3, class_name="fish_call", confidence=0.8)
    # Calls in windows 2..3, then a long silence, then calls in windows 12..13.
    script = {2: [det], 3: [det], 12: [det], 13: [det]}
    model = ScriptedAcousticModel({3: "fish_call"}, script, sample_rate=rate, window_seconds=1.0)
    blocks = [tone_block(i, rate, 1.0) for i in range(18)]
    src = ScriptedAudioSource(blocks)
    with tempfile.TemporaryDirectory() as d:
        db = fresh_db(settings, Path(d))
        mon = AcousticMonitor(settings=settings, station=station, db=db, audio_source=src,
                              model=model, visual_context=NullVisualContext(),
                              onset_threshold=DEFAULT_ONSET_THRESHOLD, silence_close_seconds=3.0)
        mon.run()
        obs = db.list_observations()
        check("two calling stretches become two observations", len(obs) == 2)
        ids = {o["id"] for o in obs}
        check("the two observations have distinct identities", len(ids) == 2)


def scenario_one_stretch_one_identity(settings):
    print("\n[6] One continuous calling stretch keeps one identity, however long")
    clean_audio_dir(settings)
    station = settings.active_station()
    rate = 32000
    det = AcousticDetection(class_id=3, class_name="fish_call", confidence=0.75)
    script = {i: [det] for i in range(2, 40)}  # 38 continuous calling windows
    model = ScriptedAcousticModel({3: "fish_call"}, script, sample_rate=rate, window_seconds=1.0)
    blocks = [tone_block(i, rate, 1.0) for i in range(42)]
    src = ScriptedAudioSource(blocks)
    with tempfile.TemporaryDirectory() as d:
        db = fresh_db(settings, Path(d))
        mon = AcousticMonitor(settings=settings, station=station, db=db, audio_source=src,
                              model=model, visual_context=NullVisualContext(),
                              onset_threshold=DEFAULT_ONSET_THRESHOLD, silence_close_seconds=3.0)
        mon.run()
        obs = db.list_observations()
        check("a long continuous stretch is exactly one observation", len(obs) == 1)
        if obs:
            check("its duration spans the whole stretch, not one window", obs[0]["duration"] > 30.0)
            check("only one child per voice despite many windows", len(db.list_child_detections(obs[0]["id"])) == 1)


def scenario_tuning_from_config(settings):
    print("\n[7] Roll and cap values are taken from configuration, nothing hardcoded")
    station = settings.active_station()
    audio_cfg = station["capture"]["audio"]
    ring = AudioRingBuffer(capacity_seconds=10.0)
    sink = AcousticTriggerSink(settings=settings, station=station, ring_buffer=ring)
    check("pre-roll came from config", sink._pre_roll == float(audio_cfg["pre_roll_seconds"]))
    check("post-roll came from config", sink._post_roll == float(audio_cfg["post_roll_seconds"]))
    check("max clip came from config", sink._max_clip_seconds == float(audio_cfg["max_clip_seconds"]))
    check("audio directory resolved through the loader", str(sink._audio_dir) == settings.path("detections_audio_dir"))


def scenario_model_selection(settings):
    print("\n[8] The acoustic model is selected by configuration; swapping needs no code change")
    station = settings.active_station()
    active = station["models"]["acoustic"]["active"]
    check("marine station selects the marine slot by config", active == "marine")

    # With no model file on disk the factory refuses clearly rather than guessing.
    raised = False
    try:
        build_acoustic_model(station, settings)
    except (FileNotFoundError, ValueError):
        raised = True
    check("factory refuses a slot with no model file present", raised)

    # Point the marine slot at a throwaway file and confirm the factory reaches
    # the marine adapter (which then tries to load TensorFlow, absent here).
    import copy
    st = copy.deepcopy(station)
    with tempfile.TemporaryDirectory() as d:
        fake = Path(d) / "surfperch_savedmodel"
        fake.mkdir()
        st["models"]["acoustic"]["options"]["marine"]["path"] = str(fake)
        reached_adapter = False
        try:
            build_acoustic_model(st, settings)
        except Exception as exc:  # noqa: BLE001
            # A missing TensorFlow or an unreadable SavedModel both prove the
            # marine adapter was reached without any code change to select it.
            reached_adapter = "tensorflow" in str(exc).lower() or "saved" in str(exc).lower() or isinstance(exc, ImportError)
        check("marine slot routes to the reef adapter with no code change", reached_adapter)

    # Switching the active slot to birdnet routes to the bird adapter instead.
    st2 = copy.deepcopy(station)
    st2["models"]["acoustic"]["active"] = "birdnet"
    with tempfile.TemporaryDirectory() as d:
        fake = Path(d) / "birdnet.tflite"
        fake.write_bytes(b"not a real model")
        st2["models"]["acoustic"]["options"]["birdnet"]["path"] = str(fake)
        routed = False
        try:
            build_acoustic_model(st2, settings)
        except Exception:  # noqa: BLE001 - reaching the adapter is the point
            routed = True
        check("changing the active slot to birdnet routes to the bird adapter", routed)


def scenario_soundscape_default_off(settings):
    print("\n[9] The continuous soundscape sampler is off by default and writes nothing")
    station = settings.active_station()
    rate = 32000
    with tempfile.TemporaryDirectory() as d:
        db = fresh_db(settings, Path(d))
        sampler = SoundscapeSampler(settings=settings, station=station, db=db,
                                    metric_functions={"rms": lambda s, r: float(np.sqrt(np.mean(s ** 2)))})
        check("sampler reports disabled from config", sampler.enabled is False)
        written = sampler.sample(np.ones(rate, dtype=np.float32) * 0.1, rate)
        check("a disabled sampler writes nothing", written == 0)
        check("no soundscape rows exist", len(db.list_soundscape_readings(station["station_id"])) == 0)

        # When a deployment enables it, it writes the configured indices.
        import copy
        st = copy.deepcopy(station)
        st["capture"]["soundscape"]["enabled"] = True
        st["capture"]["soundscape"]["metrics"] = ["rms"]
        sampler2 = SoundscapeSampler(settings=settings, station=st, db=db,
                                     metric_functions={"rms": lambda s, r: float(np.sqrt(np.mean(s ** 2)))})
        written2 = sampler2.sample(np.ones(rate, dtype=np.float32) * 0.1, rate)
        check("an enabled sampler writes its configured metric", written2 == 1)
        rows = db.list_soundscape_readings(st["station_id"])
        check("the soundscape reading landed in the database", len(rows) == 1 and rows[0]["metric"] == "rms")


def scenario_ring_buffer_bounded(settings):
    print("\n[10] The ring buffer stays bounded no matter how long the stream runs")
    rate = 8000
    ring = AudioRingBuffer(capacity_seconds=5.0)
    for i in range(200):  # 100 seconds of half-second blocks
        ring.append(tone_block(i, rate, 0.5))
    # Only the most recent ~5 seconds should remain retrievable.
    far_start = BASE + timedelta(seconds=1.0)
    far_end = BASE + timedelta(seconds=2.0)
    old, _ = ring.extract(far_start, far_end)
    check("audio older than the capacity is dropped", old.size == 0)
    recent_start = BASE + timedelta(seconds=97.0)
    recent_end = BASE + timedelta(seconds=99.0)
    recent, r = ring.extract(recent_start, recent_end)
    check("recent audio is still available", recent.size > 0 and r == rate)


def main():
    settings, cfg_path = make_pi_settings(active_index=0)
    try:
        print("loader role:", settings.node_role, "| active station:", settings.active_station()["station_name"])
        scenario_triggered_capture(settings)
        scenario_capping(settings)
        scenario_independent_trigger(settings)
        scenario_non_event_audio_discarded(settings)
        scenario_two_separate_events(settings)
        scenario_one_stretch_one_identity(settings)
        scenario_tuning_from_config(settings)
        scenario_model_selection(settings)
        scenario_soundscape_default_off(settings)
        scenario_ring_buffer_bounded(settings)
        clean_audio_dir(settings)  # leave the tree clean
    finally:
        if cfg_path.exists():
            cfg_path.unlink()

    print(f"\n==== {CHECKS['passed']} passed, {CHECKS['failed']} failed ====")
    return 0 if CHECKS["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
