"""Mocked-model verification for the desktop verification engine.

Path: tests/test_verify.py

Runs verify.py end to end with no accelerator runtime, no language-model
runtime, and no frames on disk. A scripted verifier stands in for the desktop
model and a scripted interpreter stands in for the desktop language model and
its interpretive skills, so the module loads and its verification runs with
neither library present. The real configuration is read through the real loader
and every row is read back against the real schema.

The checks prove, in one run:

  - An agreeing, confident re-score opens the verification gate, writes the
    authoritative salience as the normalized verification confidence, and
    records the verdict as measured desktop model facts.
  - A disagreeing re-score overrides the field call: the verdict and the
    disagreement are recorded in the desktop-owned table, the gate stays closed,
    and the station-owned observation row is never altered.
  - A deferred record is adjudicated by the same path, so a record the field
    tier could not classify is the desktop's to clear or leave uncleared.
  - Interpretation is written as labelled inference: the desktop model's points
    are produced_by verify, an interpretive skill's point is produced_by skill
    and carries its skill id, and rarity is written both as a numeric ingredient
    and as a labelled interpretive point.
  - The firewall holds: a point with an unrecognized type, or one claiming a
    dream provenance, is refused rather than stored, and nothing interpretive is
    ever written as a measured field value.
  - Verification is idempotent: a record already verified is skipped, so a
    re-queued identifier neither changes the verdict nor duplicates
    interpretation.
  - A pending record is not eligible, and a sweep advances every eligible record
    at once.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

logging.disable(logging.CRITICAL)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from audtheia.config import load_settings  # noqa: E402
from audtheia.storage.database import (  # noqa: E402
    Database,
    Station,
    Observation,
    ChildDetection,
    EnvironmentalReading,
    Skill,
    new_id,
    utc_now_iso,
)
from audtheia.analysis.verify import (  # noqa: E402
    VerifyEngine,
    VerifyWorker,
    FrameDetection,
    InterpretationPoint,
    PRODUCED_BY_VERIFY,
    PRODUCED_BY_SKILL,
    PRODUCED_BY_DREAM,
    DEFAULT_VERIFY_CLEAR_CONFIDENCE,
)

ISO = "%Y-%m-%dT%H:%M:%S.%fZ"


# ---------------------------------------------------------------------------
# Mock desktop model and interpreter
# ---------------------------------------------------------------------------


class ScriptedVerifier:
    """Returns a fixed detection for every frame it is handed.

    A test drives agreement or disagreement by choosing which taxon and
    confidence the stand-in resolves, without any model library. It records the
    frame paths it was asked to score so a test can confirm which frames the
    engine resolved.
    """

    def __init__(self, detection: FrameDetection, *, version="rfdetr-test-1"):
        self._detection = detection
        self._version = version
        self.scored_paths = []

    @property
    def version(self):
        return self._version

    def verify_frames(self, frame_paths):
        self.scored_paths = list(frame_paths)
        # One detection per frame handed in; an empty set yields no detections.
        return [self._detection for _ in frame_paths]


class ScriptedInterpreter:
    """Returns a fixed list of interpretation points for any context.

    Stands in for the desktop language model and any interpretive skills, so the
    interpretation path runs with no language-model library present.
    """

    def __init__(self, points, *, version="llm-test-1"):
        self._points = list(points)
        self._version = version
        self.calls = 0

    @property
    def version(self):
        return self._version

    def interpret(self, context):
        self.calls += 1
        return list(self._points)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def make_desktop_settings():
    """Write a desktop copy of the real configuration and load it through the
    real loader. The shipped file already runs as the desktop node."""
    base = json.loads((REPO / "config" / "settings.json").read_text(encoding="utf-8"))
    base["node"]["role"] = "desktop"
    base["node"]["active_station_id"] = None
    path = REPO / "config" / "settings.verify.test.json"
    path.write_text(json.dumps(base), encoding="utf-8")
    return load_settings(path), path


def fresh_db(settings, tmp: Path, station_index: int = 0) -> tuple[Database, dict]:
    db_path = tmp / "audtheia.db"
    db = Database(str(db_path), **settings.database_kwargs())
    db.initialize_schema(settings.schema_path())
    station = settings.stations()[station_index]
    db.create_station(
        Station(
            id=station["station_id"],
            station_name=station["station_name"],
            environment_type=station["environment_type"],
            created_at=utc_now_iso(),
        )
    )
    return db, station


def make_observation(
    db: Database,
    station: dict,
    *,
    field_name: str,
    field_confidence: float,
    qc_state: str = "qc_passed",
) -> str:
    """Insert one vision event with a single visual child detection.

    The child detection is the field screening call the desktop re-checks. No
    frame is written to disk; the injected verifier never opens the paths.
    """
    oid = new_id()
    created = utc_now_iso()
    short = oid.split("-")[0]
    event_name = f"{station['station_name']}_2026-07-02_{short}"
    obs = Observation(
        id=oid,
        event_name=event_name,
        station_id=station["station_id"],
        trigger_source="vision",
        first_seen="2026-07-02T12:00:00.000000Z",
        last_seen="2026-07-02T12:00:05.000000Z",
        duration=5.0,
        data_source="model",
        created_at=created,
        qc_state=qc_state,
        representative_frame=f"data/detections/visual/{event_name}/rep.jpg",
        frame_count=5,
        screening_confidence=field_confidence,
        screening_model_version="yolo11-test-1",
        salience_provisional=field_confidence,
    )
    child = ChildDetection(
        id=new_id(),
        observation_id=oid,
        modality="vision",
        created_at=created,
        confidence=field_confidence,
        scientific_name=field_name,
        common_name=field_name,
        bbox_x=10.0,
        bbox_y=10.0,
        bbox_w=40.0,
        bbox_h=30.0,
    )
    reading = EnvironmentalReading(
        id=new_id(),
        observation_id=oid,
        channel="water_temp_c",
        status="measured",
        created_at=created,
        value=27.5,
        unit="degC",
        qartod_flag=1,
    )
    db.insert_observation(obs, children=[child], environmental_readings=[reading])
    return oid


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


def engine_for(settings, db, verifier, interpreter):
    return VerifyEngine(
        settings=settings, db=db, verifier=verifier, interpreter=interpreter
    )


# ---------------------------------------------------------------------------
# 1. Agreeing, confident re-score clears the gate and writes salience + verdict
# ---------------------------------------------------------------------------


def test_agreeing_rescore_clears_gate():
    print("\n[1] An agreeing, confident re-score opens the gate and records the verdict")
    settings, path = make_desktop_settings()
    with tempfile.TemporaryDirectory() as td:
        try:
            db, station = fresh_db(settings, Path(td))
            oid = make_observation(
                db, station, field_name="Aplysina fistularis", field_confidence=0.80
            )

            verifier = ScriptedVerifier(
                FrameDetection(scientific_name="Aplysina fistularis", confidence=0.92)
            )
            interpreter = ScriptedInterpreter(
                [
                    InterpretationPoint(
                        point_type="ecological_role",
                        value="reef sponge, substrate competitor",
                        produced_by=PRODUCED_BY_VERIFY,
                        confidence=0.7,
                    )
                ]
            )
            engine = engine_for(settings, db, verifier, interpreter)
            result = engine.process(oid)

            check("outcome is verified", result.outcome == "verified" and result.verified == 1)

            row = db.get_observation_verification(oid)
            check("a verification row was written", row is not None)
            check("the gate is open (verified = 1)", row["verified"] == 1)
            check(
                "authoritative salience is the normalized verification confidence",
                abs(row["salience_authoritative"] - 0.92) < 1e-9,
            )
            check("the verdict records the resolved taxon", row["rfdetr_scientific_name"] == "Aplysina fistularis")
            check("the verdict records agreement with the field call", row["rfdetr_agrees_with_field"] == 1)
            check("the verdict records the model version from the verifier", row["rfdetr_version"] == "rfdetr-test-1")
            check("frames scored and frames in agreement are recorded", row["frames_scored"] >= 1 and row["frames_in_agreement"] >= 1)
            check("verified_at is stamped when cleared", row["verified_at"] is not None)

            # The gate the dream pass reads returns this observation.
            check("the observation is listed as verified", oid in db.list_verified_observation_ids())

            # The station-owned observation row is untouched by verification.
            obs = db.get_observation(oid)
            check("the station observation qc_state is unchanged", obs["qc_state"] == "qc_passed")
            check("the station observation carries no authoritative salience column", "salience_authoritative" not in obs)
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 2. Disagreeing re-score overrides the field call without touching the row
# ---------------------------------------------------------------------------


def test_disagreeing_rescore_overrides():
    print("\n[2] A disagreeing re-score overrides the field call, gate stays closed")
    settings, path = make_desktop_settings()
    with tempfile.TemporaryDirectory() as td:
        try:
            db, station = fresh_db(settings, Path(td))
            oid = make_observation(
                db, station, field_name="Aplysina fistularis", field_confidence=0.75
            )

            # The desktop model resolves a different taxon at high confidence.
            verifier = ScriptedVerifier(
                FrameDetection(scientific_name="Aplysina cauliformis", confidence=0.95)
            )
            interpreter = ScriptedInterpreter([])
            engine = engine_for(settings, db, verifier, interpreter)
            result = engine.process(oid)

            check("outcome is unverified", result.outcome == "unverified" and result.verified == 0)

            row = db.get_observation_verification(oid)
            check("the disagreement is recorded", row["rfdetr_agrees_with_field"] == 0)
            check("the desktop taxon is recorded, not the field taxon", row["rfdetr_scientific_name"] == "Aplysina cauliformis")
            check("the gate stays closed (verified = 0)", row["verified"] == 0)
            check("verified_at is left empty when uncleared", row["verified_at"] is None)
            check("the observation is not listed as verified", oid not in db.list_verified_observation_ids())

            # The station's own detection is never rewritten by the desktop.
            children = db.list_child_detections(oid)
            check("the field child detection is unchanged", len(children) == 1 and children[0]["scientific_name"] == "Aplysina fistularis")
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 3. A deferred record is adjudicated by the same path
# ---------------------------------------------------------------------------


def test_deferred_record_is_adjudicated():
    print("\n[3] A deferred record is the desktop's to adjudicate")
    settings, path = make_desktop_settings()
    with tempfile.TemporaryDirectory() as td:
        try:
            db, station = fresh_db(settings, Path(td))
            oid = make_observation(
                db, station, field_name="Aplysina fistularis",
                field_confidence=0.08, qc_state="qc_deferred",
            )

            verifier = ScriptedVerifier(
                FrameDetection(scientific_name="Aplysina fistularis", confidence=0.88)
            )
            interpreter = ScriptedInterpreter([])
            engine = engine_for(settings, db, verifier, interpreter)
            result = engine.process(oid)

            check("a deferred record is processed, not skipped", result.outcome in ("verified", "unverified"))
            check("the desktop cleared the deferred record on agreement", result.outcome == "verified")
            row = db.get_observation_verification(oid)
            check("a verification row exists for the deferred record", row is not None and row["verified"] == 1)
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 4. Interpretation is written as labelled inference (verify, skill, rarity)
# ---------------------------------------------------------------------------


def test_interpretation_is_labelled_inference():
    print("\n[4] Interpretation is labelled inference; rarity is both a number and a point")
    settings, path = make_desktop_settings()
    with tempfile.TemporaryDirectory() as td:
        try:
            db, station = fresh_db(settings, Path(td))
            oid = make_observation(
                db, station, field_name="Aplysina fistularis", field_confidence=0.80
            )

            skill_id = new_id()
            db.upsert_skill(
                Skill(
                    id=skill_id,
                    title="Substrate competition note",
                    trigger_condition="a sponge is detected near coral",
                    instruction="note potential competition for substrate",
                    tier="interpretive",
                    created_at=utc_now_iso(),
                    updated_at=utc_now_iso(),
                )
            )
            db_skill_points = [
                InterpretationPoint(
                    point_type="ecological_role",
                    value="reef sponge",
                    produced_by=PRODUCED_BY_VERIFY,
                ),
                InterpretationPoint(
                    point_type="rarity_score",
                    value="locally uncommon at this site",
                    produced_by=PRODUCED_BY_VERIFY,
                    numeric_value=0.83,
                ),
                InterpretationPoint(
                    point_type="skill_note",
                    value="possible substrate competition with nearby coral",
                    produced_by=PRODUCED_BY_SKILL,
                    skill_id=skill_id,
                ),
            ]
            verifier = ScriptedVerifier(
                FrameDetection(scientific_name="Aplysina fistularis", confidence=0.90)
            )
            interpreter = ScriptedInterpreter(db_skill_points)
            engine = engine_for(settings, db, verifier, interpreter)
            engine.process(oid)

            points = db.list_interpretations(oid)
            check("three interpretation points were written", len(points) == 3)
            check("every interpretation is data_source llm_inferred", all(p["data_source"] == "llm_inferred" for p in points))
            by_type = {p["point_type"]: p for p in points}
            check("the model's point is produced_by verify", by_type["ecological_role"]["produced_by"] == "verify")
            check("the skill's point is produced_by skill and carries its skill id",
                  by_type["skill_note"]["produced_by"] == "skill" and by_type["skill_note"]["skill_id"] == skill_id)
            check("the interpreter model version is stamped on a point", by_type["ecological_role"]["model_version"] == "llm-test-1")

            # Rarity has both homes: the labelled point and the numeric ingredient.
            check("rarity is written as a labelled interpretive point", "rarity_score" in by_type)
            row = db.get_observation_verification(oid)
            check("rarity is also written as the numeric salience ingredient", abs(row["rarity_score"] - 0.83) < 1e-9)

            # The deferred salience ingredients stay empty until their combination formula exists.
            check("baseline deviation is left for the later salience formula", row["baseline_deviation"] is None)
            check("authoritative anomaly magnitude is left for the later salience formula", row["anomaly_magnitude_authoritative"] is None)
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 5. The firewall refuses points it may not write
# ---------------------------------------------------------------------------


def test_firewall_refuses_bad_points():
    print("\n[5] The firewall refuses an unrecognized type and a dream-provenance point")
    settings, path = make_desktop_settings()
    with tempfile.TemporaryDirectory() as td:
        try:
            db, station = fresh_db(settings, Path(td))
            oid = make_observation(
                db, station, field_name="Aplysina fistularis", field_confidence=0.80
            )

            bad_points = [
                InterpretationPoint(
                    point_type="made_up_point",
                    value="not a recognized point",
                    produced_by=PRODUCED_BY_VERIFY,
                ),
                InterpretationPoint(
                    point_type="anomaly_flag",
                    value="a downstream dream claim",
                    produced_by=PRODUCED_BY_DREAM,
                ),
                InterpretationPoint(
                    point_type="ecological_role",
                    value="a valid point that should be written",
                    produced_by=PRODUCED_BY_VERIFY,
                ),
            ]
            verifier = ScriptedVerifier(
                FrameDetection(scientific_name="Aplysina fistularis", confidence=0.90)
            )
            interpreter = ScriptedInterpreter(bad_points)
            engine = engine_for(settings, db, verifier, interpreter)
            engine.process(oid)

            points = db.list_interpretations(oid)
            check("only the valid point was written", len(points) == 1 and points[0]["point_type"] == "ecological_role")
            check("no point carries a dream provenance", all(p["produced_by"] != "dream" for p in points))
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 6. Verification is idempotent
# ---------------------------------------------------------------------------


def test_verification_is_idempotent():
    print("\n[6] A verified record is skipped; interpretation is never duplicated")
    settings, path = make_desktop_settings()
    with tempfile.TemporaryDirectory() as td:
        try:
            db, station = fresh_db(settings, Path(td))
            oid = make_observation(
                db, station, field_name="Aplysina fistularis", field_confidence=0.80
            )
            verifier = ScriptedVerifier(
                FrameDetection(scientific_name="Aplysina fistularis", confidence=0.90)
            )
            interpreter = ScriptedInterpreter(
                [InterpretationPoint(point_type="ecological_role", value="reef sponge")]
            )
            engine = engine_for(settings, db, verifier, interpreter)

            first = engine.process(oid)
            second = engine.process(oid)

            check("the first run verifies", first.outcome == "verified")
            check("the second run is skipped", second.outcome == "skipped")
            check("interpretation was written once, not twice", len(db.list_interpretations(oid)) == 1)
            check("the interpreter was called once", interpreter.calls == 1)
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 7. A pending record is not eligible
# ---------------------------------------------------------------------------


def test_pending_record_not_eligible():
    print("\n[7] A pending record is not yet the desktop's to verify")
    settings, path = make_desktop_settings()
    with tempfile.TemporaryDirectory() as td:
        try:
            db, station = fresh_db(settings, Path(td))
            oid = make_observation(
                db, station, field_name="Aplysina fistularis",
                field_confidence=0.80, qc_state="qc_pending",
            )
            verifier = ScriptedVerifier(
                FrameDetection(scientific_name="Aplysina fistularis", confidence=0.90)
            )
            engine = engine_for(settings, db, verifier, ScriptedInterpreter([]))
            result = engine.process(oid)

            check("a pending record is skipped", result.outcome == "skipped")
            check("no verification row was written for a pending record", db.get_observation_verification(oid) is None)
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 8. A sweep advances every eligible record at once
# ---------------------------------------------------------------------------


def test_sweep_advances_backlog():
    print("\n[8] A sweep verifies every eligible record and leaves the pending one")
    settings, path = make_desktop_settings()
    with tempfile.TemporaryDirectory() as td:
        try:
            db, station = fresh_db(settings, Path(td))
            passed_ids = [
                make_observation(db, station, field_name="Aplysina fistularis",
                                 field_confidence=0.80)
                for _ in range(3)
            ]
            deferred_id = make_observation(
                db, station, field_name="Aplysina fistularis",
                field_confidence=0.08, qc_state="qc_deferred",
            )
            pending_id = make_observation(
                db, station, field_name="Aplysina fistularis",
                field_confidence=0.80, qc_state="qc_pending",
            )

            verifier = ScriptedVerifier(
                FrameDetection(scientific_name="Aplysina fistularis", confidence=0.90)
            )
            engine = engine_for(settings, db, verifier, ScriptedInterpreter([]))
            advanced = engine.sweep()

            check("the sweep advanced the passed and deferred records", advanced == 4)
            check("every passed record now has a verification row", all(db.get_observation_verification(i) is not None for i in passed_ids))
            check("the deferred record was adjudicated", db.get_observation_verification(deferred_id) is not None)
            check("the pending record was left untouched", db.get_observation_verification(pending_id) is None)
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 9. The worker drains a queue and never blocks the caller
# ---------------------------------------------------------------------------


def test_worker_drains_queue():
    print("\n[9] The verification worker drains submitted identifiers")
    settings, path = make_desktop_settings()
    with tempfile.TemporaryDirectory() as td:
        try:
            db, station = fresh_db(settings, Path(td))
            oid = make_observation(
                db, station, field_name="Aplysina fistularis", field_confidence=0.80
            )
            verifier = ScriptedVerifier(
                FrameDetection(scientific_name="Aplysina fistularis", confidence=0.90)
            )
            engine = engine_for(settings, db, verifier, ScriptedInterpreter([]))
            worker = VerifyWorker(engine)
            worker.start()
            accepted = worker.submit(oid)
            worker.stop()

            check("the identifier was accepted onto the queue", accepted is True)
            check("the worker processed one record", worker.processed == 1 and worker.failed == 0)
            check("the record was verified through the worker", db.get_observation_verification(oid) is not None)
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 10. Frame resolution selects the representative frame from settings
# ---------------------------------------------------------------------------


def test_frame_resolution_uses_settings():
    print("\n[10] Frame resolution routes the representative frame through the loader")
    settings, path = make_desktop_settings()
    with tempfile.TemporaryDirectory() as td:
        try:
            db, station = fresh_db(settings, Path(td))
            oid = make_observation(
                db, station, field_name="Aplysina fistularis", field_confidence=0.80
            )
            verifier = ScriptedVerifier(
                FrameDetection(scientific_name="Aplysina fistularis", confidence=0.90)
            )
            engine = engine_for(settings, db, verifier, ScriptedInterpreter([]))
            engine.process(oid)

            check("the verifier was handed at least the representative frame", len(verifier.scored_paths) >= 1)
            check("the resolved path is absolute (routed through the loader)", verifier.scored_paths[0].is_absolute())
            check("the resolved path ends at the representative frame", verifier.scored_paths[0].name == "rep.jpg")
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main():
    print("=" * 72)
    print("Desktop verification engine: mocked-model verification")
    print("=" * 72)
    test_agreeing_rescore_clears_gate()
    test_disagreeing_rescore_overrides()
    test_deferred_record_is_adjudicated()
    test_interpretation_is_labelled_inference()
    test_firewall_refuses_bad_points()
    test_verification_is_idempotent()
    test_pending_record_not_eligible()
    test_sweep_advances_backlog()
    test_worker_drains_queue()
    test_frame_resolution_uses_settings()

    print("\n" + "=" * 72)
    print(f"RESULT: {CHECKS['passed']} passed, {CHECKS['failed']} failed")
    print("=" * 72)
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
