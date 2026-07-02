"""Mocked-hardware verification for the field-tier quality-control engine.

Path: tests/test_observation.py

Runs the deterministic quality-control and consolidation engine end to end with
no field hardware present. There is no camera, hydrophone, satellite receiver,
or sensor here: capture is stood in for by writing pending records straight
through the real storage layer, exactly as the field pipeline does, and then
the engine is run on them by identifier. The real configuration is read through
the real loader, so every channel manifest and tuning value comes from the same
file a field station uses, and every row is read back and checked against the
real schema.

The checks prove, in one run:

  - A well-formed record is validated, consolidated into one snapshot, and
    advanced to passed.
  - A channel the configuration expects but that produced no reading is filled
    with an explicit not-measured status and no invented value, a marine
    channel with the oceanographic missing flag, while a channel already
    present is left untouched.
  - Provisional salience is confirmed when capture wrote it and written from
    the detection's own confidence when capture left it empty, always to the
    provisional slot alone.
  - A deterministic-flag skill runs and its flag reaches the snapshot; an
    interpretive skill never runs at the field tier.
  - The measured-versus-inferred firewall rejects a record carrying inferred
    provenance and rejects a skill that tries to emit anything other than a
    plain flag, routing each to the desktop with a firewall reason.
  - Each unclassifiable shape is deferred with the correct controlled reason.
  - The engine writes no interpretation for a passed record.
  - The bounded queue and worker process submitted records, a full queue drops
    the hint rather than blocking and a sweep recovers it, and re-processing a
    finalized record changes nothing.
  - A node that defers quality control to the desktop leaves records pending.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The engine logs refused skills and firewall breaches on purpose; in this
# verification those are deliberately induced, so the log is quieted to keep the
# pass/fail output readable. The behavior under test is unchanged.
logging.disable(logging.CRITICAL)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from audtheia.config import load_settings  # noqa: E402
from audtheia.storage.database import (  # noqa: E402
    ChildDetection,
    Database,
    EnvironmentalReading,
    Observation,
    Skill,
    Station,
    new_id,
    utc_now_iso,
)
from audtheia.analysis.observation import (  # noqa: E402
    QCEngine,
    QCWorker,
    SkillFlag,
    QC_PASSED,
    QC_PENDING,
    QC_DEFERRED,
    REASON_SCHEMA_NOVEL_SHAPE,
    REASON_INCOMPLETE_RECORD,
    REASON_SENSOR_CONFLICT,
    REASON_LOW_CONFIDENCE,
    REASON_FIREWALL_VIOLATION,
    STATUS_MEASURED,
    STATUS_NOT_MEASURED,
    QARTOD_MISSING,
    TIER_DETERMINISTIC_FLAG,
    TIER_INTERPRETIVE,
)

ISO = "%Y-%m-%dT%H:%M:%S.%fZ"
BASE = datetime(2026, 6, 30, 12, 0, 0, tzinfo=timezone.utc)


CHECKS = {"passed": 0, "failed": 0}


def check(label: str, condition: bool):
    if condition:
        CHECKS["passed"] += 1
        print(f"  PASS  {label}")
    else:
        CHECKS["failed"] += 1
        print(f"  FAIL  {label}")


# ---------------------------------------------------------------------------
# Harness: real loader, real schema, capture stood in by direct writes
# ---------------------------------------------------------------------------


def make_settings(station_index: int):
    """Write a field-station copy of the real configuration (role pi, active
    station = the chosen one) and load it through the real loader."""
    import json

    base = json.loads((REPO / "config" / "settings.json").read_text(encoding="utf-8"))
    base["node"]["role"] = "pi"
    base["node"]["active_station_id"] = base["stations"][station_index]["station_id"]
    path = REPO / "config" / f"settings.qc.test.{station_index}.json"
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


def enabled_channels(station: dict) -> list[dict]:
    return [c for c in station.get("channels", []) if c.get("enabled", False)]


def write_pending_observation(
    db: Database,
    station: dict,
    *,
    oid: str,
    trigger: str = "vision",
    data_source: str = "model",
    screening_confidence=0.9,
    salience=0.9,
    duration: float = 4.0,
    first: datetime = BASE,
    last: datetime = BASE + timedelta(seconds=4),
    child_confidence=0.9,
    readings: list = None,
) -> str:
    """Write one pending record exactly as the capture stage would.

    Mirrors what the vision and audio capture loops persist: an event row in the
    pending state with provisional salience, one child detection of the matching
    modality, and whatever sensor channels capture had in hand.
    """
    created = utc_now_iso()
    modality = "audio" if trigger == "audio" else "vision"
    obs = Observation(
        id=oid,
        event_name=f"{station['station_name']}_{first.strftime('%Y-%m-%d')}_{oid[:8]}",
        station_id=station["station_id"],
        trigger_source=trigger,
        first_seen=first.strftime(ISO),
        last_seen=last.strftime(ISO),
        duration=duration,
        data_source=data_source,
        created_at=created,
        qc_state=QC_PENDING,
        screening_confidence=screening_confidence,
        salience_provisional=salience,
        anomaly_magnitude_provisional=None,
    )
    children = [
        ChildDetection(
            id=new_id(),
            observation_id=oid,
            modality=modality,
            created_at=created,
            confidence=child_confidence,
            common_name="Aplysina_fistularis",
        )
    ]
    env = [
        EnvironmentalReading(
            id=new_id(),
            observation_id=oid,
            channel=r["channel"],
            status=r["status"],
            created_at=created,
            value=r.get("value"),
            unit=r.get("unit"),
            qartod_flag=r.get("qartod_flag"),
        )
        for r in (readings or [])
    ]
    db.insert_observation(obs, children=children, environmental_readings=env)
    return oid


# ---------------------------------------------------------------------------
# 1. A well-formed record passes and is consolidated
# ---------------------------------------------------------------------------


def test_pass_and_consolidate():
    print("\n[1] Well-formed marine record: validated, consolidated, passed")
    settings, _ = make_settings(0)
    station = settings.active_station()
    with tempfile.TemporaryDirectory() as td:
        db = fresh_db(settings, Path(td))
        engine = QCEngine(settings=settings, db=db)
        oid = new_id()
        # One real reading; the other three channels will be completed.
        write_pending_observation(
            db, station, oid=oid,
            readings=[{"channel": "water_temp_c", "status": STATUS_MEASURED,
                       "value": 26.4, "unit": "degC", "qartod_flag": 1}],
        )
        result = engine.process(oid)
        check("outcome is passed", result.outcome == "passed")
        check("record state is qc_passed", result.qc_state == QC_PASSED)
        check("no defer reason on a pass", result.qc_reason is None)
        row = db.get_observation(oid)
        check("database shows qc_passed", row["qc_state"] == QC_PASSED)
        snap = result.snapshot
        check("snapshot carries the event identity", snap is not None and snap.event_name == row["event_name"])
        check("snapshot carries the child detection", len(snap.child_detections) == 1)
        check("snapshot lists every expected channel",
              set(snap.expected_channels) == {c["id"] for c in enabled_channels(station)})
        check("engine writes no interpretation for a passed record",
              db.list_interpretations(oid) == [])


# ---------------------------------------------------------------------------
# 2. Completeness: missing channels filled, present channel untouched
# ---------------------------------------------------------------------------


def test_completeness_fill_marine():
    print("\n[2] Completeness: a dropped environment leg is filled, never invented")
    settings, _ = make_settings(0)
    station = settings.active_station()
    expected = {c["id"] for c in enabled_channels(station)}
    with tempfile.TemporaryDirectory() as td:
        db = fresh_db(settings, Path(td))
        engine = QCEngine(settings=settings, db=db)
        # No readings at all, the case the composer's per-leg isolation creates.
        oid = write_pending_observation(db, station, oid=new_id(), readings=[])
        result = engine.process(oid)
        rows = {r["channel"]: r for r in db.list_environmental_readings(oid)}
        check("every expected channel now has a reading", set(rows) == expected)
        check("filled channels reported on the snapshot", set(result.snapshot.filled_channels) == expected)
        check("filled rows are not_measured", all(r["status"] == STATUS_NOT_MEASURED for r in rows.values()))
        check("filled rows carry no invented value", all(r["value"] is None for r in rows.values()))
        check("filled marine channels carry the missing flag",
              all(r["qartod_flag"] == QARTOD_MISSING for r in rows.values()))
        check("record still passes after completion", result.outcome == "passed")


def test_completeness_present_untouched_terrestrial():
    print("\n[3] Completeness: a present reading is left untouched (terrestrial, no QARTOD)")
    settings, _ = make_settings(1)
    station = settings.active_station()
    with tempfile.TemporaryDirectory() as td:
        db = fresh_db(settings, Path(td))
        engine = QCEngine(settings=settings, db=db)
        oid = write_pending_observation(
            db, station, oid=new_id(),
            readings=[{"channel": "air_temp_c", "status": STATUS_MEASURED,
                       "value": 24.1, "unit": "degC"}],
        )
        engine.process(oid)
        rows = {r["channel"]: r for r in db.list_environmental_readings(oid)}
        check("the present channel keeps its measured value", rows["air_temp_c"]["value"] == 24.1)
        check("the present channel keeps its measured status", rows["air_temp_c"]["status"] == STATUS_MEASURED)
        filled = [cid for cid, r in rows.items() if r["status"] == STATUS_NOT_MEASURED]
        check("the other terrestrial channels are filled", len(filled) == 3)
        check("filled terrestrial channels carry no QARTOD flag",
              all(rows[cid]["qartod_flag"] is None for cid in filled))


# ---------------------------------------------------------------------------
# 4. Provisional salience: confirm when present, write when absent
# ---------------------------------------------------------------------------


def test_salience_confirmed_when_present():
    print("\n[4] Salience: capture's provisional value is confirmed, not rewritten")
    settings, _ = make_settings(0)
    station = settings.active_station()
    with tempfile.TemporaryDirectory() as td:
        db = fresh_db(settings, Path(td))
        engine = QCEngine(settings=settings, db=db)
        oid = write_pending_observation(db, station, oid=new_id(), screening_confidence=0.83, salience=0.83)
        engine.process(oid)
        row = db.get_observation(oid)
        check("provisional salience unchanged", row["salience_provisional"] == 0.83)
        check("authoritative salience never written at the field tier",
              db.get_observation_verification(oid) is None)


def test_salience_written_when_absent_vision():
    print("\n[5] Salience: written from screening confidence when capture left it empty")
    settings, _ = make_settings(0)
    station = settings.active_station()
    with tempfile.TemporaryDirectory() as td:
        db = fresh_db(settings, Path(td))
        engine = QCEngine(settings=settings, db=db)
        oid = write_pending_observation(db, station, oid=new_id(), screening_confidence=0.71, salience=None)
        engine.process(oid)
        row = db.get_observation(oid)
        check("provisional salience written from screening confidence", row["salience_provisional"] == 0.71)


def test_salience_written_from_child_for_audio():
    print("\n[6] Salience: written from the strongest child for an audio event with no screening confidence")
    settings, _ = make_settings(0)
    station = settings.active_station()
    with tempfile.TemporaryDirectory() as td:
        db = fresh_db(settings, Path(td))
        engine = QCEngine(settings=settings, db=db)
        oid = write_pending_observation(
            db, station, oid=new_id(), trigger="audio",
            screening_confidence=None, salience=None, child_confidence=0.62,
        )
        engine.process(oid)
        row = db.get_observation(oid)
        check("provisional salience written from the child confidence", row["salience_provisional"] == 0.62)


# ---------------------------------------------------------------------------
# 7. Skills: deterministic flag runs; interpretive never runs at the field tier
# ---------------------------------------------------------------------------


def test_deterministic_flag_runs():
    print("\n[7] A deterministic-flag skill runs and its flag reaches the snapshot")
    settings, _ = make_settings(0)
    station = settings.active_station()
    with tempfile.TemporaryDirectory() as td:
        db = fresh_db(settings, Path(td))
        skill_id = new_id()
        db.upsert_skill(Skill(
            id=skill_id, title="Warm water flag",
            trigger_condition="water temperature above a threshold",
            instruction="raise a thermal flag",
            tier=TIER_DETERMINISTIC_FLAG, created_at=utc_now_iso(), updated_at=utc_now_iso(),
        ))

        def warm_water(snapshot):
            for r in snapshot.environmental_readings:
                if r["channel"] == "water_temp_c" and r["value"] is not None and r["value"] > 29.0:
                    return SkillFlag(skill_id=skill_id, skill_title="Warm water flag",
                                     name="thermal_stress", value=True)
            return None

        engine = QCEngine(settings=settings, db=db, flag_evaluators={skill_id: warm_water})
        oid = write_pending_observation(
            db, station, oid=new_id(),
            readings=[{"channel": "water_temp_c", "status": STATUS_MEASURED,
                       "value": 30.5, "unit": "degC", "qartod_flag": 3}],
        )
        result = engine.process(oid)
        check("record passes with a flag attached", result.outcome == "passed")
        check("the derived flag is on the snapshot",
              any(f.name == "thermal_stress" and f.value is True for f in result.snapshot.flags))
        check("no interpretation row is written for a field flag", db.list_interpretations(oid) == [])


def test_interpretive_skill_never_runs_at_field():
    print("\n[8] An interpretive skill in the store never runs at the field tier")
    settings, _ = make_settings(0)
    station = settings.active_station()
    with tempfile.TemporaryDirectory() as td:
        db = fresh_db(settings, Path(td))
        db.upsert_skill(Skill(
            id=new_id(), title="Competition note",
            trigger_condition="a sponge is detected near coral",
            instruction="note potential competition for substrate",
            tier=TIER_INTERPRETIVE, created_at=utc_now_iso(), updated_at=utc_now_iso(),
        ))
        engine = QCEngine(settings=settings, db=db)
        oid = write_pending_observation(db, station, oid=new_id(),
                                        readings=[{"channel": "water_temp_c", "status": STATUS_MEASURED,
                                                   "value": 26.0, "unit": "degC", "qartod_flag": 1}])
        result = engine.process(oid)
        check("the record passes on its own merits", result.outcome == "passed")
        check("the interpretive skill produced no field flag", result.snapshot.flags == [])


# ---------------------------------------------------------------------------
# 9. Firewall: reject a fabricated interpretive field and a mis-tagged skill
# ---------------------------------------------------------------------------


def test_firewall_rejects_inferred_record():
    print("\n[9] Firewall: a record tagged with inferred provenance is deferred, untouched")
    settings, _ = make_settings(0)
    station = settings.active_station()
    with tempfile.TemporaryDirectory() as td:
        db = fresh_db(settings, Path(td))
        engine = QCEngine(settings=settings, db=db)
        # A fabricated interpretive field: a field observation claiming inferred
        # provenance. It has no readings, so if the engine wrongly completed it,
        # rows would appear.
        oid = write_pending_observation(db, station, oid=new_id(), data_source="llm_inferred", readings=[])
        result = engine.process(oid)
        check("outcome is deferred", result.outcome == "deferred")
        check("reason is firewall_violation", result.qc_reason == REASON_FIREWALL_VIOLATION)
        check("the quarantined record was not completed",
              db.list_environmental_readings(oid) == [])


def test_firewall_rejects_mistagged_interpretive_skill():
    print("\n[10] Firewall: an interpretive skill mis-tagged for the field tier is refused")
    settings, _ = make_settings(0)
    station = settings.active_station()
    with tempfile.TemporaryDirectory() as td:
        db = fresh_db(settings, Path(td))
        engine = QCEngine(settings=settings, db=db)
        oid = write_pending_observation(db, station, oid=new_id(), readings=[])
        snap = engine._build_snapshot(db.get_observation(oid), [], [], enabled_channels(station), [], [])
        # A skill whose content is interpretive but that reaches the field run
        # path: the engine decides by the tag and refuses it.
        mistagged = {"id": new_id(), "tier": TIER_INTERPRETIVE, "title": "role"}
        flags, breach = engine.evaluate_flag_skills(snap, [mistagged])
        check("the mis-tagged skill is refused", breach is True)
        check("the mis-tagged skill produced no flag", flags == [])


def test_firewall_rejects_non_flag_output():
    print("\n[11] Firewall: a field skill that emits a non-flag value is rejected and the record deferred")
    settings, _ = make_settings(0)
    station = settings.active_station()
    with tempfile.TemporaryDirectory() as td:
        db = fresh_db(settings, Path(td))
        skill_id = new_id()
        db.upsert_skill(Skill(
            id=skill_id, title="Sneaky interpretive",
            trigger_condition="always", instruction="emit an ecological claim",
            tier=TIER_DETERMINISTIC_FLAG, created_at=utc_now_iso(), updated_at=utc_now_iso(),
        ))

        def emits_a_claim(snapshot):
            # A mis-tagged interpretive skill trying to smuggle a free-text claim
            # through as a flag. The engine rejects any value that is not a plain
            # measured or derived flag.
            return SkillFlag(skill_id=skill_id, skill_title="Sneaky interpretive",
                             name="ecological_role", value="this sponge competes with coral")

        engine = QCEngine(settings=settings, db=db, flag_evaluators={skill_id: emits_a_claim})
        oid = write_pending_observation(db, station, oid=new_id(),
                                        readings=[{"channel": "water_temp_c", "status": STATUS_MEASURED,
                                                   "value": 26.0, "unit": "degC", "qartod_flag": 1}])
        result = engine.process(oid)
        check("outcome is deferred", result.outcome == "deferred")
        check("reason is firewall_violation", result.qc_reason == REASON_FIREWALL_VIOLATION)
        check("no interpretation was ever written", db.list_interpretations(oid) == [])


# ---------------------------------------------------------------------------
# 12. Defer reasons: each unclassifiable shape gets the correct code
# ---------------------------------------------------------------------------


def test_defer_reasons():
    print("\n[12] Each unclassifiable shape defers with the correct controlled reason")
    settings, _ = make_settings(0)
    station = settings.active_station()
    with tempfile.TemporaryDirectory() as td:
        db = fresh_db(settings, Path(td))
        engine = QCEngine(settings=settings, db=db)

        # Backwards event window cannot be repaired deterministically.
        oid_a = write_pending_observation(
            db, station, oid=new_id(),
            first=BASE + timedelta(seconds=10), last=BASE, duration=4.0,
            readings=[{"channel": "water_temp_c", "status": STATUS_MEASURED, "value": 26.0, "qartod_flag": 1}],
        )
        r_a = engine.process(oid_a)
        check("backwards window -> incomplete_record", r_a.qc_reason == REASON_INCOMPLETE_RECORD)

        # A reserved trigger shape the field engine has no path to validate.
        oid_b = write_pending_observation(db, station, oid=new_id(), trigger="sensor", readings=[])
        r_b = engine.process(oid_b)
        check("reserved trigger shape -> schema_novel_shape", r_b.qc_reason == REASON_SCHEMA_NOVEL_SHAPE)

        # A channel that claims a measurement but carries no value.
        oid_c = write_pending_observation(
            db, station, oid=new_id(),
            readings=[{"channel": "water_temp_c", "status": STATUS_MEASURED, "value": None, "qartod_flag": 1}],
        )
        r_c = engine.process(oid_c)
        check("value contradicting its status -> sensor_conflict", r_c.qc_reason == REASON_SENSOR_CONFLICT)

        # A detection below the field-pass confidence floor.
        oid_d = write_pending_observation(
            db, station, oid=new_id(), screening_confidence=0.05, salience=0.05, child_confidence=0.05,
            readings=[{"channel": "water_temp_c", "status": STATUS_MEASURED, "value": 26.0, "qartod_flag": 1}],
        )
        r_d = engine.process(oid_d)
        check("weak detection -> low_confidence_unclassified", r_d.qc_reason == REASON_LOW_CONFIDENCE)

        check("every deferred record is qc_deferred in the database",
              all(db.get_observation(o)["qc_state"] == QC_DEFERRED for o in (oid_a, oid_b, oid_c, oid_d)))


# ---------------------------------------------------------------------------
# 13. Queue, worker, idempotency, sweep, and the desktop-deferral gate
# ---------------------------------------------------------------------------


def test_worker_drains_queue():
    print("\n[13] The bounded worker drains submitted records")
    settings, _ = make_settings(0)
    station = settings.active_station()
    with tempfile.TemporaryDirectory() as td:
        db = fresh_db(settings, Path(td))
        engine = QCEngine(settings=settings, db=db)
        worker = QCWorker(engine, maxsize=8)
        ids = [write_pending_observation(db, station, oid=new_id(), readings=[]) for _ in range(5)]
        worker.start()
        for oid in ids:
            worker.submit(oid)
        worker.stop()
        check("worker processed every submitted record", worker.processed == 5)
        check("all records advanced out of pending",
              all(db.get_observation(o)["qc_state"] != QC_PENDING for o in ids))


def test_full_queue_drops_hint_and_sweep_recovers():
    print("\n[14] A full queue drops the hint without losing work; a sweep recovers it")
    settings, _ = make_settings(0)
    station = settings.active_station()
    with tempfile.TemporaryDirectory() as td:
        db = fresh_db(settings, Path(td))
        engine = QCEngine(settings=settings, db=db)
        worker = QCWorker(engine, maxsize=1)  # tiny queue, worker not started
        ids = [write_pending_observation(db, station, oid=new_id(), readings=[]) for _ in range(3)]
        accepted = [worker.submit(o) for o in ids]
        check("the queue accepted one and dropped the rest", accepted.count(True) == 1 and accepted.count(False) == 2)
        check("saturation was counted", worker.queue_saturation_events == 2)
        check("dropped records are still pending in the database",
              all(db.get_observation(o)["qc_state"] == QC_PENDING for o in ids))
        advanced = engine.sweep_pending(station_id=station["station_id"])
        check("the sweep processed every pending record", advanced == 3)
        check("no record is left pending after the sweep",
              all(db.get_observation(o)["qc_state"] != QC_PENDING for o in ids))


def test_idempotent_reprocess():
    print("\n[15] Re-processing a finalized record changes nothing")
    settings, _ = make_settings(0)
    station = settings.active_station()
    with tempfile.TemporaryDirectory() as td:
        db = fresh_db(settings, Path(td))
        engine = QCEngine(settings=settings, db=db)
        oid = write_pending_observation(db, station, oid=new_id(), readings=[])
        first = engine.process(oid)
        readings_after_first = len(db.list_environmental_readings(oid))
        second = engine.process(oid)
        readings_after_second = len(db.list_environmental_readings(oid))
        check("first run advances the record", first.outcome in ("passed", "deferred"))
        check("second run is skipped", second.outcome == "skipped")
        check("completion is not repeated", readings_after_first == readings_after_second)


def test_desktop_deferral_gate():
    print("\n[16] A node that defers to the desktop leaves records pending")
    settings, _ = make_settings(0)
    station = settings.active_station()
    # Flip the node to defer per-observation quality control to the desktop.
    settings.raw["analysis"]["per_observation_analysis_location"] = "desktop"
    with tempfile.TemporaryDirectory() as td:
        db = fresh_db(settings, Path(td))
        engine = QCEngine(settings=settings, db=db)
        oid = write_pending_observation(db, station, oid=new_id(), readings=[])
        result = engine.process(oid)
        check("outcome is skipped", result.outcome == "skipped")
        check("record stays pending for the desktop", db.get_observation(oid)["qc_state"] == QC_PENDING)
        check("nothing was completed on the deferring node", db.list_environmental_readings(oid) == [])


def cleanup():
    for i in (0, 1):
        p = REPO / "config" / f"settings.qc.test.{i}.json"
        if p.exists():
            p.unlink()


def main():
    print("=" * 72)
    print("Field-tier quality-control engine: mocked-hardware verification")
    print("=" * 72)
    try:
        test_pass_and_consolidate()
        test_completeness_fill_marine()
        test_completeness_present_untouched_terrestrial()
        test_salience_confirmed_when_present()
        test_salience_written_when_absent_vision()
        test_salience_written_from_child_for_audio()
        test_deterministic_flag_runs()
        test_interpretive_skill_never_runs_at_field()
        test_firewall_rejects_inferred_record()
        test_firewall_rejects_mistagged_interpretive_skill()
        test_firewall_rejects_non_flag_output()
        test_defer_reasons()
        test_worker_drains_queue()
        test_full_queue_drops_hint_and_sweep_recovers()
        test_idempotent_reprocess()
        test_desktop_deferral_gate()
    finally:
        cleanup()

    print("\n" + "=" * 72)
    print(f"RESULT: {CHECKS['passed']} passed, {CHECKS['failed']} failed")
    print("=" * 72)
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
