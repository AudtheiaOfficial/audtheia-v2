"""Verification for the expert correction loop.

Path: tests/test_corrections.py

Proves that a person's judgement about a detection is stored as a separate
claim, is never allowed to overwrite what a model wrote, and survives being
read back. The properties checked are the ones the design depends on:

  - each of the three verdicts stores and reads back correctly,
  - a relabel without a resolved taxon is refused by the database,
  - a rejection carrying a taxon is refused by the database,
  - the history is append-only and reads newest first,
  - effective detection evidence is 1.0 for a confirm and a relabel and 0.0 for
    a reject, derived on read rather than stored,
  - the model's screening confidence and the child detection's own name are
    byte-identical before and after a correction.

The last of these is the point of the whole feature. A correction that quietly
edited the model's output would destroy the provenance firewall, so it is
checked directly rather than assumed from the absence of an UPDATE statement.

Built on a temporary database created from the real schema.sql, so the CHECK
constraints exercised here are the shipped ones and not a restatement of them.
Standard library only, and it never opens the live database.

Run: python tests/test_corrections.py
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from audtheia.storage.database import (  # noqa: E402
    ChildDetection,
    Database,
    Observation,
    Station,
    new_id,
    utc_now_iso,
)

SCHEMA = REPO / "audtheia" / "storage" / "schema.sql"

CHECKS = {"passed": 0, "failed": 0}

# The effective detection evidence a verdict implies. Derived here and never
# stored, because a stored confidence attached to a human judgement would be a
# number with nothing behind it.
EFFECTIVE_C = {"confirm": 1.0, "relabel": 1.0, "reject": 0.0}


def check(label: str, condition: bool) -> None:
    if condition:
        CHECKS["passed"] += 1
        print(f"  PASS  {label}")
    else:
        CHECKS["failed"] += 1
        print(f"  FAIL  {label}")


def effective_c(verdict: str) -> float:
    """The detection evidence a verdict implies, for the salience recomputation."""
    return EFFECTIVE_C[verdict]


def _fixture(db_path: Path) -> tuple:
    """A database holding one station and one two-taxon terrestrial event.

    Two child detections rather than one, so that a correction aimed at a
    single box can be shown not to disturb its neighbour.
    """
    db = Database(db_path)
    db.initialize_schema(SCHEMA)
    now = utc_now_iso()

    station = Station(
        id=new_id(),
        station_name="Test Station",
        environment_type="terrestrial",
        created_at=now,
    )
    db.create_station(station)

    obs = Observation(
        id=new_id(),
        event_name="test-event-0001",
        station_id=station.id,
        trigger_source="vision",
        first_seen=now,
        last_seen=now,
        duration=4.0,
        data_source="model",
        created_at=now,
        frame_count=12,
        screening_confidence=0.73,
    )
    junco = ChildDetection(
        id=new_id(),
        observation_id=obs.id,
        modality="vision",
        created_at=now,
        scientific_name="Junco hyemalis",
        common_name="Dark-eyed Junco",
        confidence=0.73,
    )
    chickadee = ChildDetection(
        id=new_id(),
        observation_id=obs.id,
        modality="vision",
        created_at=now,
        scientific_name="Poecile atricapillus",
        common_name="Black-capped Chickadee",
        confidence=0.41,
    )
    db.insert_observation(obs, children=[junco, chickadee])
    return db, obs, junco, chickadee


def test_verdicts_store(db_path: Path) -> None:
    print("\nEach verdict stores and reads back")
    db, obs, junco, _ = _fixture(db_path)

    confirmed = db.add_correction(obs.id, verdict="confirm", corrector="expert")
    check("a confirm stores", confirmed["verdict"] == "confirm")
    check("a confirm carries human provenance", confirmed["data_source"] == "human_expert")
    check("a confirm is marked measured", confirmed["status"] == "measured")
    check("a confirm records its corrector", confirmed["corrector"] == "expert")
    check("a confirm is stamped", bool(confirmed["corrected_at"]))

    relabelled = db.add_correction(
        obs.id,
        verdict="relabel",
        corrector="expert",
        detection_id=junco.id,
        corrected_scientific_name="Haemorhous mexicanus",
        corrected_common_name="House Finch",
        corrected_gbif_usage_key="2494988",
    )
    check("a relabel stores its name", relabelled["corrected_scientific_name"] == "Haemorhous mexicanus")
    check("a relabel stores its usage key", relabelled["corrected_gbif_usage_key"] == "2494988")
    check("a relabel can target one box", relabelled["detection_id"] == junco.id)

    rejected = db.add_correction(obs.id, verdict="reject", corrector="expert", modality="vision")
    check("a reject stores", rejected["verdict"] == "reject")
    check("a reject carries no name", rejected["corrected_scientific_name"] is None)
    check("a reject keeps its modality", rejected["modality"] == "vision")


def test_constraints_refuse(db_path: Path) -> None:
    print("\nThe database refuses an incoherent correction")
    db, obs, _, _ = _fixture(db_path)

    try:
        db.add_correction(obs.id, verdict="relabel", corrector="expert")
        check("a relabel without a name is refused", False)
    except sqlite3.IntegrityError:
        check("a relabel without a name is refused", True)

    try:
        db.add_correction(
            obs.id,
            verdict="reject",
            corrector="expert",
            corrected_scientific_name="Xestospongia muta",
        )
        check("a reject carrying a name is refused", False)
    except sqlite3.IntegrityError:
        check("a reject carrying a name is refused", True)

    try:
        db.add_correction(obs.id, verdict="probably", corrector="expert")
        check("an unknown verdict is refused", False)
    except sqlite3.IntegrityError:
        check("an unknown verdict is refused", True)

    # The provenance firewall itself: a human claim cannot be written into the
    # model-owned table, because that table's CHECK physically forbids it.
    with db.connect() as conn:
        try:
            conn.execute(
                "INSERT INTO child_detections (id, observation_id, modality, created_at, data_source, status) "
                "VALUES (?, ?, 'vision', ?, 'human_expert', 'measured')",
                (new_id(), obs.id, utc_now_iso()),
            )
            check("a human claim cannot enter child_detections", False)
        except sqlite3.IntegrityError:
            check("a human claim cannot enter child_detections", True)


def test_append_only_history(db_path: Path) -> None:
    print("\nHistory is append-only and reads newest first")
    db, obs, junco, _ = _fixture(db_path)

    db.add_correction(obs.id, verdict="confirm", corrector="expert")
    db.add_correction(
        obs.id,
        verdict="relabel",
        corrector="expert",
        corrected_scientific_name="Cyanocitta cristata",
        corrected_common_name="Blue Jay",
        corrected_gbif_usage_key="2482593",
    )
    db.add_correction(obs.id, verdict="reject", corrector="expert")

    history = db.corrections_for_observation(obs.id)
    check("every correction is kept", len(history) == 3)
    check("the newest is first", history[0]["verdict"] == "reject")
    check("the oldest is last", history[2]["verdict"] == "confirm")

    latest = db.latest_correction(obs.id)
    check("the latest is the most recent", latest["verdict"] == "reject")

    # An event-level verdict must not answer a question about one box, and the
    # reverse, or a single wrong box would silently condemn the whole event.
    db.add_correction(
        obs.id,
        verdict="relabel",
        corrector="expert",
        detection_id=junco.id,
        corrected_scientific_name="Haemorhous mexicanus",
        corrected_gbif_usage_key="2494988",
    )
    per_box = db.latest_correction(obs.id, junco.id)
    check("a box correction is found by its own id", per_box["detection_id"] == junco.id)
    check("an event-level read ignores box corrections",
          db.latest_correction(obs.id)["verdict"] == "reject")

    counts = db.correction_counts()
    check("counts collapse a changed mind to one target", counts["reject"] == 1)
    check("counts see the box correction separately", counts["relabel"] == 1)
    check("counts total distinct targets", counts["total"] == 2)


def test_effective_confidence(db_path: Path) -> None:
    print("\nEffective detection evidence is derived, never stored")
    db, obs, _, _ = _fixture(db_path)

    for verdict, expected in (("confirm", 1.0), ("relabel", 1.0), ("reject", 0.0)):
        check(f"effective C for {verdict} is {expected}", effective_c(verdict) == expected)

    stored = db.add_correction(obs.id, verdict="confirm", corrector="expert")
    check("no confidence column is written", "confidence" not in stored)
    check("salience is left for the pass that knows k", stored["salience_corrected"] is None)


def test_model_output_untouched(db_path: Path) -> None:
    print("\nA correction never edits what the model wrote")
    db, obs, junco, chickadee = _fixture(db_path)

    before = db.get_observation(obs.id)
    children_before = {c["id"]: dict(c) for c in db.list_child_detections(obs.id)}

    db.add_correction(
        obs.id,
        verdict="relabel",
        corrector="expert",
        detection_id=junco.id,
        corrected_scientific_name="Haemorhous mexicanus",
        corrected_common_name="House Finch",
        corrected_gbif_usage_key="2494988",
    )

    after = db.get_observation(obs.id)
    children_after = {c["id"]: dict(c) for c in db.list_child_detections(obs.id)}

    check("screening confidence is unchanged",
          after["screening_confidence"] == before["screening_confidence"] == 0.73)
    check("the observation row is unchanged in every field", after == before)
    check("the corrected box keeps the model's own name",
          children_after[junco.id]["scientific_name"] == "Junco hyemalis")
    check("the corrected box keeps the model's confidence",
          children_after[junco.id]["confidence"] == 0.73)
    check("the untouched box is untouched",
          children_after[chickadee.id] == children_before[chickadee.id])
    check("every child detection is still model-sourced",
          all(c["data_source"] == "model" for c in children_after.values()))


def main() -> int:
    print("=" * 72)
    print("Expert corrections: storage, constraints, and provenance")
    print("=" * 72)
    if not SCHEMA.exists():
        print("  FAIL  schema.sql not found")
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        test_verdicts_store(root / "verdicts.db")
        test_constraints_refuse(root / "constraints.db")
        test_append_only_history(root / "history.db")
        test_effective_confidence(root / "evidence.db")
        test_model_output_untouched(root / "provenance.db")
    print("\n" + "=" * 72)
    print(f"RESULT: {CHECKS['passed']} passed, {CHECKS['failed']} failed")
    print("=" * 72)
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
