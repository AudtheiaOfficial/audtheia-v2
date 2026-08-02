"""The longitudinal pass gate includes expert-identified events, not just auto-verified.

Path: tests/test_pass_gate.py

The generative gate decides which events the pass may rest a claim on. It was
the desktop verifier's cleared set alone; an expert's own identification, the
strongest evidence there is, did not count. This proves the widened gate: an
expert-confirmed or expert-relabelled event is now eligible, an expert-rejected
one is not, and a relabelled event contributes its CORRECTED taxon rather than
the model's mistaken one.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from audtheia.storage.database import (  # noqa: E402
    Database, Station, Observation, ChildDetection, ObservationVerification,
    new_id, utc_now_iso,
)

SCHEMA = REPO / "audtheia" / "storage" / "schema.sql"
CHECKS = {"passed": 0, "failed": 0}


def check(label: str, condition: bool):
    if condition:
        CHECKS["passed"] += 1
        print(f"  PASS  {label}")
    else:
        CHECKS["failed"] += 1
        print(f"  FAIL  {label}")


def _obs(db, station_id, *, usage_key):
    now = utc_now_iso()
    oid = new_id()
    obs = Observation(
        id=oid, event_name="e-" + oid[:8], station_id=station_id, trigger_source="vision",
        first_seen=now, last_seen=now, duration=1.0, data_source="model", created_at=now,
        frame_count=2, screening_confidence=0.9, salience_provisional=0.9,
    )
    det = ChildDetection(
        id=new_id(), observation_id=oid, modality="vision", created_at=now,
        confidence=0.9, gbif_usage_key=usage_key, scientific_name="Model taxon",
    )
    db.insert_observation(obs, children=[det])
    return oid, det.id


def test_gate_and_resolved_species():
    print("\n[1] Expert-confirmed and relabelled events are eligible; rejected is not")
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "gate.db")
        db.initialize_schema(SCHEMA)
        st = Station(id=new_id(), station_name="S", environment_type="terrestrial", created_at=utc_now_iso())
        db.create_station(st)

        a, _ = _obs(db, st.id, usage_key="AUTO")
        b, b_det = _obs(db, st.id, usage_key="MODELB")
        c, c_det = _obs(db, st.id, usage_key="MODELC")
        d, d_det = _obs(db, st.id, usage_key="MODELD")
        e, _ = _obs(db, st.id, usage_key="MODELE")

        # A: desktop auto-verified.
        db.upsert_observation_verification(ObservationVerification(
            observation_id=a, created_at=utc_now_iso(), verified=1))
        # B: expert confirmed (not auto-verified).
        db.add_correction(b, verdict="confirm", corrector="expert", detection_id=b_det)
        # C: expert relabelled to a corrected taxon.
        db.add_correction(c, verdict="relabel", corrector="expert", detection_id=c_det,
                          corrected_scientific_name="Corrected taxon", corrected_gbif_usage_key="CORRECTED")
        # D: expert rejected.
        db.add_correction(d, verdict="reject", corrector="expert", detection_id=d_det)
        # E: nothing.

        eligible = set(db.list_pass_eligible_observation_ids())
        check("an auto-verified event is eligible", a in eligible)
        check("an expert-confirmed event is eligible", b in eligible)
        check("an expert-relabelled event is eligible", c in eligible)
        check("an expert-rejected event is NOT eligible", d not in eligible)
        check("an untouched event is NOT eligible", e not in eligible)

        check("a confirmed event keeps the model taxon", db.expert_resolved_species(b) == ["MODELB"])
        check("a relabelled event uses the corrected taxon", db.expert_resolved_species(c) == ["CORRECTED"])
        check("a rejected detection is dropped from the species", db.expert_resolved_species(d) == [])
        check("an auto-verified event keeps its model taxon", db.expert_resolved_species(a) == ["AUTO"])


def test_change_of_mind_wins():
    print("\n[2] The newest verdict decides eligibility")
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "mind.db")
        db.initialize_schema(SCHEMA)
        st = Station(id=new_id(), station_name="S", environment_type="terrestrial", created_at=utc_now_iso())
        db.create_station(st)
        oid, det = _obs(db, st.id, usage_key="MODEL")
        db.add_correction(oid, verdict="reject", corrector="expert", detection_id=det)
        check("a rejected event is not eligible", oid not in set(db.list_pass_eligible_observation_ids()))
        db.add_correction(oid, verdict="confirm", corrector="expert", detection_id=det)
        check("after a later confirm it becomes eligible", oid in set(db.list_pass_eligible_observation_ids()))


def main() -> int:
    test_gate_and_resolved_species()
    test_change_of_mind_wins()
    print(f"\n==== pass gate: {CHECKS['passed']} passed, {CHECKS['failed']} failed ====")
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
