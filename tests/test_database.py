"""End-to-end checks for the Audtheia V2 data-access layer.

Path: tests/test_database.py

Runs against the real schema.sql found relative to this file, so it needs no
configuration and no hardware. Two temporary databases stand in for a field
station and the desktop hub. The script exercises every read and write path,
then proves the three properties the sync layer must guarantee:

  - a batch delivered twice never produces a duplicate,
  - an interrupted sync recovers with no gaps and no duplicates,
  - rows not yet confirmed by the desktop are never deleted from the station.

Run from the repository root:  python tests/test_database.py
Exits 0 only if every check passes.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = REPO_ROOT / "audtheia" / "storage" / "schema.sql"
sys.path.insert(0, str(REPO_ROOT / "audtheia" / "storage"))

import database as db  # noqa: E402
from database import (  # noqa: E402
    Database,
    Station,
    Observation,
    ChildDetection,
    EnvironmentalReading,
    SoundscapeReading,
    SpeciesReference,
    Skill,
    ObservationVerification,
    Interpretation,
    StationTelemetry,
    TelemetryError,
    DreamPass,
    Pattern,
    utc_now_iso,
    new_id,
)

PASS = 0
FAIL = 0


def check(label: str, condition: bool) -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ok   {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label}")


def fresh_db() -> Database:
    handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    handle.close()
    d = Database(handle.name)
    d.initialize_schema(SCHEMA)
    return d


def make_station(d: Database, name: str = "ReefAlpha") -> str:
    sid = new_id()
    d.upsert_station(
        Station(
            id=sid,
            station_name=name,
            environment_type="marine",
            created_at=utc_now_iso(),
        )
    )
    return sid


def make_observation(station_id: str, *, ordinal: int, embedding: bytes | None = None) -> Observation:
    oid = new_id()
    short = oid.split("-")[0]
    now = utc_now_iso()
    return Observation(
        id=oid,
        event_name=f"ReefAlpha_2026-06-29_{short}_{ordinal}",
        station_id=station_id,
        trigger_source="vision",
        first_seen=now,
        last_seen=now,
        duration=4.2,
        data_source="model",
        created_at=now,
        screening_confidence=0.87,
        screening_model_version="yolo11-porifera-v3",
        gps_latitude=18.21,
        gps_longitude=-67.15,
        gps_status="measured",
        salience_provisional=0.4,
        feature_embedding=embedding,
    )


def section(title: str) -> None:
    print(f"\n== {title} ==")


def test_connection_pragmas() -> None:
    section("connection settings")
    d = fresh_db()
    with d.connect() as conn:
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
        busy = conn.execute("PRAGMA busy_timeout").fetchone()[0]
    check("foreign keys enforced on the connection", fk == 1)
    check("write-ahead logging enabled", str(journal).lower() == "wal")
    check("busy timeout applied", busy == db.DEFAULT_BUSY_TIMEOUT_MS)


def test_observation_write_read() -> None:
    section("observation write and read with captured children")
    d = fresh_db()
    sid = make_station(d)
    obs = make_observation(sid, ordinal=1)
    children = [
        ChildDetection(
            id=new_id(),
            observation_id=obs.id,
            modality="vision",
            created_at=utc_now_iso(),
            scientific_name="Aplysina fistularis",
            confidence=0.91,
            bbox_x=0.1,
            bbox_y=0.1,
            bbox_w=0.2,
            bbox_h=0.2,
        ),
        ChildDetection(
            id=new_id(),
            observation_id=obs.id,
            modality="vision",
            created_at=utc_now_iso(),
            scientific_name="Xestospongia muta",
            confidence=0.74,
        ),
    ]
    readings = [
        EnvironmentalReading(
            id=new_id(),
            observation_id=obs.id,
            channel="water_temp_c",
            status="measured",
            created_at=utc_now_iso(),
            value=28.4,
            unit="C",
            qartod_flag=1,
        ),
        EnvironmentalReading(
            id=new_id(),
            observation_id=obs.id,
            channel="salinity_psu",
            status="sensor_error",
            created_at=utc_now_iso(),
            qartod_flag=4,
        ),
    ]
    d.insert_observation(obs, children=children, environmental_readings=readings)

    got = d.get_observation(obs.id)
    check("observation round-trips", got is not None and got["event_name"] == obs.event_name)
    check("provisional salience stored", got["salience_provisional"] == 0.4)
    check("two child detections stored", len(d.list_child_detections(obs.id)) == 2)
    env = d.list_environmental_readings(obs.id)
    check("two environmental readings stored", len(env) == 2)
    err = [r for r in env if r["channel"] == "salinity_psu"][0]
    check("absent value recorded as a sensor error, not a silent gap", err["status"] == "sensor_error")
    check("missing value left null rather than zeroed", err["value"] is None)

    d.set_observation_qc(obs.id, "qc_passed")
    check("quality-control state updated", d.get_observation(obs.id)["qc_state"] == "qc_passed")
    d.set_observation_provisional_salience(obs.id, 0.55, 0.3)
    check("provisional salience update applied", d.get_observation(obs.id)["salience_provisional"] == 0.55)


def test_provenance_and_constraints() -> None:
    section("provenance and constraint enforcement")
    d = fresh_db()
    sid = make_station(d)

    bad = make_observation(sid, ordinal=2)
    bad.data_source = "guess"  # not in the controlled vocabulary
    raised = False
    try:
        d.insert_observation(bad)
    except Exception:
        raised = True
    check("an out-of-vocabulary data_source is rejected", raised)

    bad_salience = make_observation(sid, ordinal=3)
    bad_salience.salience_provisional = 1.7  # outside the normalized range
    raised = False
    try:
        d.insert_observation(bad_salience)
    except Exception:
        raised = True
    check("salience outside the 0 to 1 range is rejected", raised)

    orphan = ChildDetection(
        id=new_id(), observation_id=new_id(), modality="vision", created_at=utc_now_iso()
    )
    raised = False
    try:
        with d.connect() as conn:
            Database._insert_row(conn, orphan)
    except Exception:
        raised = True
    check("a child detection with no parent event is rejected", raised)


def test_embedding_guard() -> None:
    section("embedding size guard, no silent loss")
    d = fresh_db()
    sid = make_station(d)
    big = make_observation(sid, ordinal=4, embedding=b"x" * 40)
    raised = False
    try:
        d.insert_observation(big, max_embedding_bytes=16)
    except ValueError:
        raised = True
    check("an oversized embedding is refused at write time", raised)
    check("the oversized row was not stored", d.get_observation(big.id) is None)

    ok = make_observation(sid, ordinal=5, embedding=b"y" * 8)
    d.insert_observation(ok, max_embedding_bytes=16)
    stored = d.get_observation(ok.id)
    check("a within-limit embedding is stored intact", stored["feature_embedding"] == b"y" * 8)


def test_desktop_owned_tables() -> None:
    section("desktop-owned writes")
    d = fresh_db()
    sid = make_station(d)
    obs = make_observation(sid, ordinal=6)
    d.insert_observation(obs)

    d.upsert_observation_verification(
        ObservationVerification(
            observation_id=obs.id,
            created_at=utc_now_iso(),
            verified=1,
            rfdetr_version="rfdetr-porifera-v12",
            salience_authoritative=0.66,
            rarity_score=0.9,
            verified_at=utc_now_iso(),
        )
    )
    check("verification stored", d.get_observation_verification(obs.id)["verified"] == 1)
    check("verified gate lists the cleared event", obs.id in d.list_verified_observation_ids())

    d.insert_interpretation(
        Interpretation(
            id=new_id(),
            observation_id=obs.id,
            point_type="ecological_role",
            value="filter feeder",
            produced_by="verify",
            created_at=utc_now_iso(),
            confidence=0.8,
        )
    )
    check("interpretation stored and labelled inferred",
          d.list_interpretations(obs.id)[0]["data_source"] == "llm_inferred")

    dp = DreamPass(
        id=new_id(),
        phase_reached="nrem_a",
        status="running",
        started_at=utc_now_iso(),
        created_at=utc_now_iso(),
    )
    d.create_dream_pass(dp)
    d.update_dream_pass(dp.id, phase_reached="rem", cycles_completed=2, checkpoint_watermark="wm-2")
    refreshed = d.get_dream_pass(dp.id)
    check("dream pass progress checkpointed", refreshed["cycles_completed"] == 2 and refreshed["checkpoint_watermark"] == "wm-2")

    pat = Pattern(
        id=new_id(),
        dream_pass_id=dp.id,
        dream_phase="rem",
        data_span_start=utc_now_iso(),
        data_span_end=utc_now_iso(),
        n=12,
        description="candidate seasonal shift in sponge cover",
        created_at=utc_now_iso(),
        effect_size=0.42,
        effect_size_type="r",
    )
    d.insert_pattern(pat, observation_ids=[obs.id])
    check("pattern stored as a candidate hypothesis", d.get_pattern(pat.id)["status"] == "candidate")
    check("pattern traceable to its source memory", d.list_pattern_observations(pat.id) == [obs.id])

    d.upsert_skill(
        Skill(
            id=new_id(),
            title="Thermal stress flag",
            trigger_condition="water_temp_c above 29",
            instruction="flag as potential thermal stress",
            tier="deterministic_flag",
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
        )
    )
    check("skill stored with its tier", d.list_skills(tier="deterministic_flag")[0]["tier"] == "deterministic_flag")

    d.upsert_species_reference(
        SpeciesReference(
            gbif_usage_key="2253869",
            scientific_name="Xestospongia muta",
            fetched_at=utc_now_iso(),
            iucn_status="LC",
            gbif_snapshot_date="2026-06-01",
        )
    )
    check("species reference cached", d.get_species_reference("2253869")["iucn_status"] == "LC")


def test_station_upsert_propagates() -> None:
    section("station registry upsert (desktop authored)")
    d = fresh_db()
    sid = new_id()
    d.upsert_station(Station(id=sid, station_name="ReefAlpha", environment_type="marine", created_at=utc_now_iso()))
    d.upsert_station(Station(id=sid, station_name="ReefAlpha-North", environment_type="marine", created_at=utc_now_iso()))
    check("a rename replaces the earlier copy", d.get_station(sid)["station_name"] == "ReefAlpha-North")
    check("upsert did not create a second station", len(d.list_stations()) == 1)


def test_soundscape_and_telemetry() -> None:
    section("soundscape and telemetry writes")
    d = fresh_db()
    sid = make_station(d)
    d.insert_soundscape_reading(
        SoundscapeReading(id=new_id(), station_id=sid, recorded_at=utc_now_iso(),
                          metric="spl_low_band", value=92.3, created_at=utc_now_iso())
    )
    check("soundscape reading stored", len(d.list_soundscape_readings(sid)) == 1)

    tel = StationTelemetry(
        id=new_id(), station_id=sid, recorded_at=utc_now_iso(), created_at=utc_now_iso(),
        frames_processed=12000, frames_dropped=3, buffer_fill_pct=41.5,
    )
    d.insert_station_telemetry(tel, errors=[TelemetryError(id=new_id(), telemetry_id=tel.id, channel="ph", error_count=2)])
    check("telemetry heartbeat stored", len(d.list_station_telemetry(sid)) == 1)
    check("per-channel error stored with its heartbeat", d.list_telemetry_errors(tel.id)[0]["error_count"] == 2)
    check("energy fields left null when no meter is configured", d.list_station_telemetry(sid)[0]["avg_power_w"] is None)


def _seed_station_and_events(station_db: Database, desktop_db: Database, n: int) -> str:
    """Create one station on both sides and n events on the station only."""
    sid = new_id()
    reg = Station(id=sid, station_name="ReefAlpha", environment_type="marine", created_at=utc_now_iso())
    station_db.upsert_station(reg)
    desktop_db.upsert_station(reg)  # the registry reaches the desktop ahead of the events
    for i in range(n):
        obs = make_observation(sid, ordinal=i)
        child = ChildDetection(id=new_id(), observation_id=obs.id, modality="vision",
                              created_at=utc_now_iso(), scientific_name="Aplysina fistularis", confidence=0.9)
        env = EnvironmentalReading(id=new_id(), observation_id=obs.id, channel="water_temp_c",
                                  status="measured", created_at=utc_now_iso(), value=28.0, qartod_flag=1)
        station_db.insert_observation(obs, children=[child], environmental_readings=[env])
    return sid


def test_sync_basic_and_idempotent() -> None:
    section("sync: append-only, parent drags children, idempotent")
    station_db = fresh_db()
    desktop_db = fresh_db()
    _seed_station_and_events(station_db, desktop_db, 5)

    check("five events queued before sync", station_db.count_unsynced()["observations"] == 5)
    confirmed = db.run_sync_round(station_db, desktop_db)
    check("five events confirmed this round", confirmed["observations"] == 5)
    check("desktop now holds five events", len(desktop_db.list_observations()) == 5)

    # Children travelled with their parents.
    a_parent = desktop_db.list_observations()[0]["id"]
    check("captured child detection arrived with its event", len(desktop_db.list_child_detections(a_parent)) == 1)
    check("captured sensor reading arrived with its event", len(desktop_db.list_environmental_readings(a_parent)) == 1)

    check("station queue is empty after confirmation", station_db.count_unsynced()["observations"] == 0)
    check("desktop stamped a received time", desktop_db.list_observations()[0]["synced_at"] is not None)
    check("station stamped its own confirmed time", station_db.get_observation(a_parent)["synced_at"] is not None)

    # Deliver the exact same batch again: nothing should duplicate.
    replay = {"observations": [], "soundscape_index_readings": [], "station_telemetry": []}
    # Rebuild a batch by exporting from a copy of the station before it was marked synced.
    station_again = fresh_db()
    desktop_again = fresh_db()
    _seed_station_and_events(station_again, desktop_again, 5)
    batch = station_again.export_unsynced_batch()
    desktop_again.import_batch(batch)
    desktop_again.import_batch(batch)  # second identical delivery
    check("a re-delivered batch does not duplicate events", len(desktop_again.list_observations()) == 5)
    first_parent = desktop_again.list_observations()[0]["id"]
    check("a re-delivered batch does not duplicate children",
          len(desktop_again.list_child_detections(first_parent)) == 1)


def test_sync_interruption_recovers() -> None:
    section("sync: interruption recovers with no gaps or duplicates")
    station_db = fresh_db()
    desktop_db = fresh_db()
    _seed_station_and_events(station_db, desktop_db, 4)

    # Desktop imports, but the confirmation never reaches the station.
    batch = station_db.export_unsynced_batch()
    desktop_db.import_batch(batch)
    # No mark_synced here: the station still believes nothing is confirmed.
    check("station still shows all four queued after a lost confirmation",
          station_db.count_unsynced()["observations"] == 4)

    # Next round re-exports the same rows; the desktop ignores what it has.
    confirmed = db.run_sync_round(station_db, desktop_db)
    check("the retry confirms all four", confirmed["observations"] == 4)
    check("desktop holds exactly four, no duplicates from the double delivery",
          len(desktop_db.list_observations()) == 4)
    check("station queue clears after the successful retry",
          station_db.count_unsynced()["observations"] == 0)


def test_never_evict_unsynced() -> None:
    section("rolling buffer: confirmed rows cleaned, unconfirmed never touched")
    station_db = fresh_db()
    desktop_db = fresh_db()
    sid = _seed_station_and_events(station_db, desktop_db, 6)

    # Sync only the first three by exporting a small batch, then add fresh ones.
    batch = station_db.export_unsynced_batch(batch_size=3)
    confirmed = desktop_db.import_batch(batch)
    station_db.mark_synced("observations", confirmed["observations"])
    check("three events confirmed", station_db.count_unsynced()["observations"] == 3)

    cleaned = station_db.clean_synced()
    check("clean removed exactly the three confirmed events", cleaned["observations"] == 3)
    check("the three unconfirmed events remain on the station", station_db.count_unsynced()["observations"] == 3)
    check("station still holds only the unconfirmed events", len(station_db.list_observations()) == 3)

    # The children of a still-present event must survive; the children of a
    # cleaned event must be gone with it.
    remaining = station_db.list_observations()
    check("a surviving event kept its child detection", len(station_db.list_child_detections(remaining[0]["id"])) == 1)
    desktop_children_total = sum(
        len(desktop_db.list_child_detections(o["id"])) for o in desktop_db.list_observations()
    )
    check("the desktop kept the children of the cleaned events", desktop_children_total == 3)


def test_telemetry_and_soundscape_sync() -> None:
    section("sync covers telemetry and soundscape streams too")
    station_db = fresh_db()
    desktop_db = fresh_db()
    sid = new_id()
    reg = Station(id=sid, station_name="ReefAlpha", environment_type="marine", created_at=utc_now_iso())
    station_db.upsert_station(reg)
    desktop_db.upsert_station(reg)

    tel = StationTelemetry(id=new_id(), station_id=sid, recorded_at=utc_now_iso(),
                          created_at=utc_now_iso(), frames_processed=9000)
    station_db.insert_station_telemetry(tel, errors=[TelemetryError(id=new_id(), telemetry_id=tel.id, channel="do", error_count=1)])
    station_db.insert_soundscape_reading(
        SoundscapeReading(id=new_id(), station_id=sid, recorded_at=utc_now_iso(),
                         metric="aci", value=3.1, created_at=utc_now_iso())
    )

    confirmed = db.run_sync_round(station_db, desktop_db)
    check("telemetry heartbeat synced", confirmed["station_telemetry"] == 1)
    check("soundscape reading synced", confirmed["soundscape_index_readings"] == 1)
    check("desktop received the telemetry error row",
          len(desktop_db.list_telemetry_errors(tel.id)) == 1)
    check("telemetry and soundscape queues clear after confirmation",
          station_db.count_unsynced()["station_telemetry"] == 0
          and station_db.count_unsynced()["soundscape_index_readings"] == 0)


def main() -> int:
    print(f"schema: {SCHEMA}")
    print(f"exists: {SCHEMA.exists()}")
    test_connection_pragmas()
    test_observation_write_read()
    test_provenance_and_constraints()
    test_embedding_guard()
    test_desktop_owned_tables()
    test_station_upsert_propagates()
    test_soundscape_and_telemetry()
    test_sync_basic_and_idempotent()
    test_sync_interruption_recovers()
    test_never_evict_unsynced()
    test_telemetry_and_soundscape_sync()
    print(f"\n==============================")
    print(f"  {PASS} passed, {FAIL} failed")
    print(f"==============================")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
