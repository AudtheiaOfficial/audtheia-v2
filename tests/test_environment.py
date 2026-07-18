"""Mocked-hardware verification for environmental capture and the composer.

Path: tests/test_environment.py

Runs the location-and-environment capture and the capture composer end to end
with no satellite receiver, no sensors, no camera, and no microphone present.
Scripted receivers and sensor banks stand in for the hardware, and the real
configuration is read through the real loader so every tuning value and every
channel definition comes from the same file the field station uses. Every row
is read back and checked against the real schema.

The checks prove, in one run:

  - GPS and every configured sensor are read on a trigger, and each writes a
    value with its own missing-data status.
  - Marine channels carry an oceanographic quality flag; non-marine channels do
    not.
  - A simulated sensor fault records a sensor error, not a silent gap.
  - The satellite receiver is the clock: before the first fix a capture is
    marked as having a provisional time, and after a fix it is not, with the
    hardware clock holding time between fixes.
  - The composer runs every capture leg for one trigger and merges them into one
    record, on both the visual trigger and the acoustic trigger, so the
    capture-everything rule holds end to end.
  - One leg's failure never denies the record the other legs captured.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

# The composer logs isolated leg failures on purpose; in this verification those
# failures are deliberately induced, so the log is quieted to keep the pass/fail
# output readable. The behavior under test is unchanged.
logging.disable(logging.CRITICAL)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from audtheia.config import load_settings  # noqa: E402
from audtheia.storage.database import Database, Station, utc_now_iso  # noqa: E402
from audtheia.pipeline.monitor import CaptureResult, TrackEvent  # noqa: E402
from audtheia.pipeline.environment import (  # noqa: E402
    EnvironmentCapture,
    FieldClock,
    GpsRead,
    NullEnvironmentCapture,
    NullGpsSource,
    NullSensorBank,
    SensorRead,
    STATUS_BELOW_DETECTION_LIMIT,
    STATUS_MEASURED,
    STATUS_NOT_APPLICABLE,
    STATUS_NOT_MEASURED,
    STATUS_SENSOR_ERROR,
    STATUS_STATION_CONFIGURED,
    QARTOD_FAIL,
    QARTOD_PASS,
    QARTOD_SUSPECT,
)
from audtheia.pipeline.composer import (  # noqa: E402
    CaptureComposer,
    ComposedTriggerSink,
    merge_capture_results,
)
from audtheia.pipeline.acoustic import (  # noqa: E402
    AcousticDetection,
    AcousticMonitor,
    AcousticTriggerSink,
    AudioBlock,
    AudioRingBuffer,
)

ISO = "%Y-%m-%dT%H:%M:%S.%fZ"
BASE = datetime(2026, 6, 30, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Mock hardware
# ---------------------------------------------------------------------------


class ScriptedGpsSource:
    """Returns a fixed queue of satellite reads in order, so a test can drive the
    receiver from no-fix to fix and see the clock respond."""

    def __init__(self, reads: list[GpsRead]):
        self._reads = list(reads)
        self._i = 0

    def read(self) -> GpsRead:
        if self._i < len(self._reads):
            r = self._reads[self._i]
            self._i += 1
            return r
        return self._reads[-1] if self._reads else GpsRead(attempted=True, ok=False)

    def close(self):
        pass


class ScriptedSensorBank:
    """Returns scripted readings by channel id. A channel mapped to an exception
    raises on read, so a driver fault can be simulated; a channel mapped to a
    SensorRead is returned as-is; an unmapped channel returns a no-value read."""

    def __init__(self, by_channel: dict):
        self._by_channel = dict(by_channel)

    def read_all(self, channel_ids: list[str]) -> list[SensorRead]:
        out = []
        for cid in channel_ids:
            spec = self._by_channel.get(cid)
            if isinstance(spec, Exception):
                out.append(SensorRead(channel=cid, attempted=True, ok=False, error=str(spec)))
            elif isinstance(spec, SensorRead):
                out.append(spec)
            else:
                out.append(SensorRead(channel=cid, attempted=True, ok=False))
        return out

    def close(self):
        pass


class ScriptedAudioSource:
    """Emits a fixed sequence of audio blocks, then end of stream."""

    def __init__(self, blocks: list[AudioBlock]):
        self._blocks = list(blocks)
        self._i = 0

    def read(self):
        if self._i >= len(self._blocks):
            return None
        b = self._blocks[self._i]
        self._i += 1
        return b

    def close(self):
        pass


class ScriptedAcousticModel:
    """A stand-in acoustic model that scores one fixed class per window from a
    script keyed by window index, so the independent acoustic trigger runs
    without any model library."""

    SAMPLE_RATE = 32000
    WINDOW_SECONDS = 1.0

    def __init__(self, script: dict, version="test-acoustic-1"):
        self._script = script
        self._version = version
        self._i = 0

    @property
    def version(self):
        return self._version

    @property
    def citation(self):
        return None

    @property
    def sample_rate(self):
        return self.SAMPLE_RATE

    @property
    def window_seconds(self):
        return self.WINDOW_SECONDS

    @property
    def class_names(self):
        return {0: "Test_species"}

    def detect(self, samples, sample_rate):
        conf = self._script.get(self._i, 0.0)
        self._i += 1
        return [AcousticDetection(class_id=0, class_name="Test_species", confidence=conf)]

    def close(self):
        pass


class FixedVisualContext:
    """A visual context that always returns a frame and one visual detection, so
    the audio path's visual leg is exercised without a camera."""

    def snapshot(self, captured_at):
        from audtheia.pipeline.acoustic import VisualSnapshot

        return VisualSnapshot(
            representative_frame="data/detections/visual/mock/frame.jpg",
            children=[{"class_name": "Seen_thing", "confidence": 0.7,
                       "bbox_x": 1.0, "bbox_y": 2.0, "bbox_w": 3.0, "bbox_h": 4.0}],
        )


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def make_pi_settings(station_index: int):
    """Write a field-station copy of the real configuration (role pi, active
    station = the chosen one) and load it through the real loader."""
    base = json.loads((REPO / "config" / "settings.json").read_text(encoding="utf-8"))
    base["node"]["role"] = "pi"
    base["node"]["active_station_id"] = base["stations"][station_index]["station_id"]
    # Redirect writable data paths into a throwaway temp tree so the scenarios'
    # shutil.rmtree of the configured detections dirs can never touch the real
    # repository data/ (which would delete real captured frames and clips).
    sandbox = tempfile.mkdtemp(prefix="audtheia-test-")
    base["paths"]["data_dir"] = sandbox
    base["paths"]["detections_visual_dir"] = str(Path(sandbox) / "detections" / "visual")
    base["paths"]["detections_audio_dir"] = str(Path(sandbox) / "detections" / "audio")
    base["paths"]["gps_dir"] = str(Path(sandbox) / "gps")
    path = REPO / "config" / f"settings.env.test.{station_index}.json"
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


def clean_dirs(settings):
    for key in ("detections_visual_dir", "detections_audio_dir"):
        d = Path(settings.path(key))
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)


def track_event(station: dict, start: datetime, end: datetime) -> TrackEvent:
    """A minimal finished vision event, enough for a trigger sink to act on."""
    short = "abcd1234"
    name = f"{station['station_name']}_{start.strftime('%Y-%m-%d')}_{short}"
    return TrackEvent(
        observation_id="00000000-0000-0000-0000-0000000000aa",
        event_name=name,
        station_id=station["station_id"],
        track_id=1,
        first_seen=start.strftime(ISO),
        last_seen=end.strftime(ISO),
        duration=(end - start).total_seconds(),
        frame_count=5,
        time_provisional=0,
        best_confidence=0.9,
        representative_frame="data/detections/visual/mock/rep.jpg",
        screening_model_version="yolo11-test-1",
        event_dir=Path("data/detections/visual/mock"),
        segment_count=1,
        children=[{"class_name": "Aplysina_fistularis", "confidence": 0.9,
                   "bbox_x": 10.0, "bbox_y": 10.0, "bbox_w": 40.0, "bbox_h": 30.0}],
    )


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


# ---------------------------------------------------------------------------
# 1. Environment capture: statuses, QARTOD, sensor fault
# ---------------------------------------------------------------------------


def test_environment_capture_marine():
    print("\n[1] Environment capture on the marine station (statuses + QARTOD + fault)")
    settings, path = make_pi_settings(0)
    try:
        station = settings.active_station()
        channels = {c["id"]: c for c in station["channels"]}

        # A good temperature, a pH physically impossible for the sensor range
        # (fail), a dissolved-oxygen value under its detection limit, a salinity
        # that raises a driver fault, all marine.
        gps = ScriptedGpsSource([
            GpsRead(attempted=True, ok=True, latitude=18.2, longitude=-67.1,
                    elevation=0.0, utc_time=BASE.strftime(ISO)),
        ])
        bank = ScriptedSensorBank({
            "water_temp_c": SensorRead(channel="water_temp_c", attempted=True, ok=True, value=27.5),
            "ph": SensorRead(channel="ph", attempted=True, ok=True, value=99.0),  # outside sensor_range -> fail
            "dissolved_oxygen_mg_l": SensorRead(channel="dissolved_oxygen_mg_l", attempted=True, ok=True, value=0.005),  # below detection_limit 0.01
            "salinity_psu": RuntimeError("i2c timeout"),
        })
        cap = EnvironmentCapture(settings=settings, station=station, gps_source=gps, sensor_bank=bank)
        result = cap.capture(BASE.strftime(ISO), (BASE + timedelta(seconds=5)).strftime(ISO))

        by = {r.channel: r for r in result.environmental_readings}

        check("GPS read on trigger and measured", result.gps_status == STATUS_MEASURED and result.gps_latitude == 18.2)
        check("all enabled channels produced a reading", set(by) == set(channels))
        check("good value is measured", by["water_temp_c"].status == STATUS_MEASURED and by["water_temp_c"].value == 27.5)
        check("value under detection limit is below_detection_limit",
              by["dissolved_oxygen_mg_l"].status == STATUS_BELOW_DETECTION_LIMIT)
        check("driver fault is sensor_error, not a silent gap",
              by["salinity_psu"].status == STATUS_SENSOR_ERROR and by["salinity_psu"].value is None)
        check("units come from configuration", by["water_temp_c"].unit == "degC")

        # QARTOD on marine channels only
        check("in-range marine value flags QARTOD pass", by["water_temp_c"].qartod_flag == QARTOD_PASS)
        check("out-of-sensor-range value flags QARTOD fail", by["ph"].qartod_flag == QARTOD_FAIL)
        check("sensor-error channel flags QARTOD missing (9)", by["salinity_psu"].qartod_flag == 9)
        check("every marine reading carries a QARTOD flag",
              all(by[c].qartod_flag is not None for c in by))
    finally:
        path.unlink(missing_ok=True)


def test_environment_capture_terrestrial():
    print("\n[2] Environment capture on the terrestrial station (no QARTOD, suspect range)")
    settings, path = make_pi_settings(1)
    try:
        station = settings.active_station()
        gps = ScriptedGpsSource([GpsRead(attempted=True, ok=False)])  # attempted, no fix
        bank = ScriptedSensorBank({
            "air_temp_c": SensorRead(channel="air_temp_c", attempted=True, ok=True, value=24.0),
            "relative_humidity_pct": SensorRead(channel="relative_humidity_pct", attempted=True, ok=True, value=61.0),
            # soil moisture and illuminance left unmapped -> no value this event
        })
        cap = EnvironmentCapture(settings=settings, station=station, gps_source=gps, sensor_bank=bank)
        result = cap.capture(BASE.strftime(ISO), (BASE + timedelta(seconds=5)).strftime(ISO))
        by = {r.channel: r for r in result.environmental_readings}

        check("GPS attempted but no fix is not_measured", result.gps_status == STATUS_NOT_MEASURED)
        check("terrestrial channels carry no QARTOD flag",
              all(by[c].qartod_flag is None for c in by))
        check("unmapped channel is not_measured, not dropped",
              by["soil_moisture_pct"].status == STATUS_NOT_MEASURED and by["soil_moisture_pct"].value is None)
        check("measured terrestrial value has value and status",
              by["air_temp_c"].status == STATUS_MEASURED and by["air_temp_c"].value == 24.0)
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 3. The satellite receiver as the clock
# ---------------------------------------------------------------------------


def test_field_clock():
    print("\n[3] GPS-as-clock: provisional before first fix, disciplined after, holds between fixes")
    clock = FieldClock()
    check("clock starts provisional (time_provisional == 1)", clock.time_provisional() == 1 and not clock.disciplined)

    # A read with no fix does not discipline the clock.
    clock.observe(GpsRead(attempted=True, ok=False))
    check("no-fix read leaves the clock provisional", clock.time_provisional() == 1)

    # The first real fix disciplines the clock.
    clock.observe(GpsRead(attempted=True, ok=True, latitude=18.2, longitude=-67.1,
                          utc_time=BASE.strftime(ISO)))
    check("first fix disciplines the clock (time_provisional == 0)", clock.time_provisional() == 0 and clock.disciplined)
    check("clock remembers the satellite time", clock.last_utc == BASE.strftime(ISO))

    # A later no-fix read (receiver briefly loses lock) keeps the clock
    # disciplined: the hardware clock holds time between fixes.
    clock.observe(GpsRead(attempted=True, ok=False))
    check("clock stays disciplined between fixes (RTC hold)", clock.time_provisional() == 0)


def test_clock_drives_capture_state():
    print("\n[4] The shared clock a capture disciplines is the one a record reads")
    settings, path = make_pi_settings(0)
    try:
        station = settings.active_station()
        clock = FieldClock()
        # Before any capture, a record created now would be provisional.
        provisional_before = clock.time_provisional()

        gps = ScriptedGpsSource([
            GpsRead(attempted=True, ok=True, latitude=18.2, longitude=-67.1,
                    elevation=0.0, utc_time=BASE.strftime(ISO)),
        ])
        bank = ScriptedSensorBank({})
        cap = EnvironmentCapture(settings=settings, station=station,
                                 gps_source=gps, sensor_bank=bank, clock=clock)
        # A capture that gets a fix disciplines the shared clock as a side effect.
        cap.capture(BASE.strftime(ISO), (BASE + timedelta(seconds=1)).strftime(ISO))
        provisional_after = cap.clock.time_provisional()

        check("record before first fix would be provisional", provisional_before == 1)
        check("capture with a fix disciplines the shared clock", provisional_after == 0)
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 5. The composer merges every leg on the visual trigger, and writes the row
# ---------------------------------------------------------------------------


def test_composer_vision_path():
    print("\n[5] Composer on the visual trigger: audio + environment merged into one record")
    settings, path = make_pi_settings(0)
    with tempfile.TemporaryDirectory() as td:
        try:
            clean_dirs(settings)
            station = settings.active_station()
            db = fresh_db(settings, Path(td))

            start = BASE
            end = BASE + timedelta(seconds=4)

            # Audio leg: a ring buffer pre-filled so a clip can be cut.
            ring = AudioRingBuffer(60.0)
            rate = 32000
            t = start - timedelta(seconds=3)
            for _ in range(10):
                ring.append(AudioBlock(samples=np.zeros(rate, dtype=np.float32),
                                       sample_rate=rate, captured_at=t.strftime(ISO)))
                t += timedelta(seconds=1)
            acoustic_sink = AcousticTriggerSink(
                settings=settings, station=station, ring_buffer=ring,
                acoustic_model_version="test-acoustic-1",
            )

            # Environment leg: a fix and two good marine readings.
            gps = ScriptedGpsSource([
                GpsRead(attempted=True, ok=True, latitude=18.2, longitude=-67.1,
                        elevation=0.0, utc_time=start.strftime(ISO)),
            ])
            bank = ScriptedSensorBank({
                "water_temp_c": SensorRead(channel="water_temp_c", attempted=True, ok=True, value=27.5),
                "ph": SensorRead(channel="ph", attempted=True, ok=True, value=8.1),
                "dissolved_oxygen_mg_l": SensorRead(channel="dissolved_oxygen_mg_l", attempted=True, ok=True, value=6.2),
                "salinity_psu": SensorRead(channel="salinity_psu", attempted=True, ok=True, value=35.0),
            })
            env_cap = EnvironmentCapture(settings=settings, station=station, gps_source=gps, sensor_bank=bank)

            sink = ComposedTriggerSink(acoustic_sink=acoustic_sink, environment_capture=env_cap)
            event = track_event(station, start, end)
            merged = sink.on_event(event)

            check("merged result carries the audio clip (audio leg ran)", merged.audio_clip_path is not None)
            check("merged result carries the acoustic model version", merged.acoustic_model_version == "test-acoustic-1")
            check("merged result carries the GPS fix (environment leg ran)", merged.gps_status == STATUS_MEASURED)
            check("merged result carries every sensor channel (environment leg ran)",
                  len(merged.environmental_readings) == 4)

            # Write the observation the way the monitor loop would, then read the
            # row back against the real schema.
            from audtheia.storage.database import Observation, ChildDetection, EnvironmentalReading, new_id
            created = utc_now_iso()
            obs = Observation(
                id=event.observation_id, event_name=event.event_name, station_id=event.station_id,
                trigger_source="vision", first_seen=event.first_seen, last_seen=event.last_seen,
                duration=event.duration, data_source="model", created_at=created,
                time_provisional=event.time_provisional, qc_state="qc_pending",
                representative_frame=event.representative_frame, frame_count=event.frame_count,
                screening_confidence=event.best_confidence, screening_model_version=event.screening_model_version,
                acoustic_model_version=merged.acoustic_model_version,
                salience_provisional=event.best_confidence,
                audio_clip_path=merged.audio_clip_path,
                audio_true_duration_seconds=merged.audio_true_duration_seconds,
                audio_capped=merged.audio_capped,
                gps_latitude=merged.gps_latitude, gps_longitude=merged.gps_longitude,
                gps_elevation=merged.gps_elevation, gps_status=merged.gps_status,
            )
            children = [ChildDetection(id=new_id(), observation_id=event.observation_id, modality="vision",
                                       created_at=created, confidence=c["confidence"], common_name=c["class_name"],
                                       bbox_x=c["bbox_x"], bbox_y=c["bbox_y"], bbox_w=c["bbox_w"], bbox_h=c["bbox_h"])
                        for c in event.children]
            readings = [EnvironmentalReading(id=new_id(), observation_id=event.observation_id, channel=r.channel,
                                             status=r.status, created_at=created, value=r.value, unit=r.unit,
                                             qartod_flag=r.qartod_flag)
                        for r in merged.environmental_readings]
            db.insert_observation(obs, children=children, environmental_readings=readings)

            row = db.get_observation(event.observation_id)
            check("observation row persisted with GPS and audio", row is not None and row["gps_status"] == "measured" and row["audio_clip_path"] is not None)

            # Read the environmental rows straight from the connection to confirm
            # they hit the real table with data_source 'sensor'.
            with db.connect() as conn:
                erows = conn.execute("SELECT channel, status, data_source, qartod_flag FROM environmental_readings WHERE observation_id=?",
                                     (event.observation_id,)).fetchall()
            check("four environmental rows stored under this observation", len(erows) == 4)
            check("every environmental row is data_source 'sensor'", all(r["data_source"] == "sensor" for r in erows))
            check("marine rows carry a QARTOD flag in the database", all(r["qartod_flag"] is not None for r in erows))
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 6. Leg isolation
# ---------------------------------------------------------------------------


def test_composer_leg_isolation():
    print("\n[6] One leg's failure never denies the record the others captured")
    settings, path = make_pi_settings(0)
    try:
        station = settings.active_station()

        def good_env_leg(fs, ls):
            r = CaptureResult()
            r.gps_status = STATUS_MEASURED
            r.gps_latitude = 18.2
            return r

        def failing_audio_leg(fs, ls):
            raise RuntimeError("hydrophone unplugged")

        composer = CaptureComposer([("audio", failing_audio_leg), ("environment", good_env_leg)])
        merged = composer.compose(BASE.strftime(ISO), (BASE + timedelta(seconds=1)).strftime(ISO))

        check("failing leg contributes nothing (audio absent)", merged.audio_clip_path is None)
        check("surviving leg's fields are present (GPS measured)", merged.gps_status == STATUS_MEASURED and merged.gps_latitude == 18.2)
    finally:
        path.unlink(missing_ok=True)


def test_merge_collision_keeps_first():
    print("\n[7] Two legs setting the same field keep the first, never silently clobber")
    a = CaptureResult(gps_status=STATUS_MEASURED)
    b = CaptureResult(gps_status=STATUS_SENSOR_ERROR)
    merged = merge_capture_results([a, b])
    check("first leg's value wins on collision", merged.gps_status == STATUS_MEASURED)


# ---------------------------------------------------------------------------
# 8. The acoustic trigger path now records environment + GPS too
# ---------------------------------------------------------------------------


def test_acoustic_path_records_environment():
    print("\n[8] Acoustic trigger: visual + environment + GPS captured, all merged into the audio record")
    settings, path = make_pi_settings(0)
    with tempfile.TemporaryDirectory() as td:
        try:
            clean_dirs(settings)
            station = settings.active_station()
            db = fresh_db(settings, Path(td))

            model = ScriptedAcousticModel(script={0: 0.9, 1: 0.9, 2: 0.0, 3: 0.0, 4: 0.0})
            rate = model.SAMPLE_RATE
            blocks = []
            t = BASE
            for _ in range(6):
                blocks.append(AudioBlock(samples=np.zeros(rate, dtype=np.float32),
                                         sample_rate=rate, captured_at=t.strftime(ISO)))
                t += timedelta(seconds=1)
            audio_source = ScriptedAudioSource(blocks)

            gps = ScriptedGpsSource([
                GpsRead(attempted=True, ok=True, latitude=18.2, longitude=-67.1,
                        elevation=0.0, utc_time=BASE.strftime(ISO)),
            ])
            bank = ScriptedSensorBank({
                "water_temp_c": SensorRead(channel="water_temp_c", attempted=True, ok=True, value=27.5),
                "ph": SensorRead(channel="ph", attempted=True, ok=True, value=8.1),
                "dissolved_oxygen_mg_l": SensorRead(channel="dissolved_oxygen_mg_l", attempted=True, ok=True, value=6.2),
                "salinity_psu": SensorRead(channel="salinity_psu", attempted=True, ok=True, value=35.0),
            })
            env_cap = EnvironmentCapture(settings=settings, station=station, gps_source=gps, sensor_bank=bank)

            mon = AcousticMonitor(
                settings=settings, station=station, db=db,
                audio_source=audio_source, model=model,
                visual_context=FixedVisualContext(),
                environment_capture=env_cap,
                onset_threshold=0.5, silence_close_seconds=2.0,
            )
            mon.run()

            check("exactly one acoustic observation written", mon.events_written == 1 and mon.events_failed == 0)

            with db.connect() as conn:
                obs = conn.execute("SELECT * FROM observations WHERE trigger_source='audio'").fetchall()
                check("the audio observation carries the GPS fix", len(obs) == 1 and obs[0]["gps_status"] == "measured")
                oid = obs[0]["id"]
                erows = conn.execute("SELECT channel, data_source FROM environmental_readings WHERE observation_id=?",
                                     (oid,)).fetchall()
                check("the audio observation carries all four sensor channels", len(erows) == 4)
                check("those sensor rows are data_source 'sensor'", all(r["data_source"] == "sensor" for r in erows))
                vis = conn.execute("SELECT COUNT(*) AS n FROM child_detections WHERE observation_id=? AND modality='vision'",
                                   (oid,)).fetchone()
                check("the visual co-occurrence was captured on the audio event", vis["n"] == 1)
        finally:
            path.unlink(missing_ok=True)


def test_null_environment_capture():
    print("\n[9] A station with no receiver or sensors still produces a valid, empty leg")
    cap = NullEnvironmentCapture()
    r = cap.capture(BASE.strftime(ISO), (BASE + timedelta(seconds=1)).strftime(ISO))
    check("null capture returns an empty result", r.gps_status is None and not r.environmental_readings)


def test_station_configured_location():
    print("\n[10] A station with no receiver but an entered position records it distinctly")
    settings, path = make_pi_settings(0)
    try:
        station = settings.active_station()

        # No live receiver on this station for this scenario. The entered position
        # is set explicitly so the check does not depend on the reference file
        # keeping any particular coordinates.
        station["sensors"]["gps"]["enabled"] = False
        station["location"] = {"latitude": 18.21, "longitude": -67.15, "elevation": None}
        cap = EnvironmentCapture(settings=settings, station=station,
                                 gps_source=NullGpsSource(), sensor_bank=NullSensorBank())
        result = cap.capture(BASE.strftime(ISO), (BASE + timedelta(seconds=1)).strftime(ISO))
        check("entered coordinates are recorded on the event",
              result.gps_latitude == 18.21 and result.gps_longitude == -67.15)
        check("their status is station_configured, distinct from a measured fix",
              result.gps_status == STATUS_STATION_CONFIGURED)

        # No receiver and no entered position: the location simply does not apply.
        station["location"] = {"latitude": None, "longitude": None, "elevation": None}
        result2 = cap.capture(BASE.strftime(ISO), (BASE + timedelta(seconds=1)).strftime(ISO))
        check("no receiver and no entered position is not_applicable",
              result2.gps_status == STATUS_NOT_APPLICABLE and result2.gps_latitude is None)

        # A live receiver still measures its own fix; an entered position never
        # masks a real one, so the measured-versus-entered line is never blurred.
        station["sensors"]["gps"]["enabled"] = True
        station["location"] = {"latitude": 18.21, "longitude": -67.15, "elevation": None}
        gps = ScriptedGpsSource([
            GpsRead(attempted=True, ok=True, latitude=18.0, longitude=-67.0,
                    elevation=0.0, utc_time=BASE.strftime(ISO)),
        ])
        cap2 = EnvironmentCapture(settings=settings, station=station,
                                  gps_source=gps, sensor_bank=NullSensorBank())
        result3 = cap2.capture(BASE.strftime(ISO), (BASE + timedelta(seconds=1)).strftime(ISO))
        check("a live fix stays measured and is not overridden by an entered position",
              result3.gps_status == STATUS_MEASURED and result3.gps_latitude == 18.0)
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main():
    print("=" * 72)
    print("Environmental capture + composer: mocked-hardware verification")
    print("=" * 72)
    test_environment_capture_marine()
    test_environment_capture_terrestrial()
    test_field_clock()
    test_clock_drives_capture_state()
    test_composer_vision_path()
    test_composer_leg_isolation()
    test_merge_collision_keeps_first()
    test_acoustic_path_records_environment()
    test_null_environment_capture()
    test_station_configured_location()

    print("\n" + "=" * 72)
    print(f"RESULT: {CHECKS['passed']} passed, {CHECKS['failed']} failed")
    print("=" * 72)
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
