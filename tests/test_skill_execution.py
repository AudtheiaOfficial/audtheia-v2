"""End-to-end verification for skill execution and its stored effect.

Path: tests/test_skill_execution.py

Proves the behaviour that turns a saved skill from a library entry into a real,
recorded effect on the record, without ever crossing the measured-versus-inferred
firewall:

  - A deterministic-flag skill carrying a structured condition fires during
    field quality control and its outcome is persisted to the skill_flags table
    as a derived reading (data_source 'rule_derived'), never into the
    inferred-only interpretations table and never onto the measured record.
  - A skill whose condition does not hold records nothing.
  - Persistence is idempotent: re-scanning the same event and skill never
    duplicates a flag.
  - Re-scanning a finalized record applies the current skills to it and clears a
    flag whose skill no longer fires, so an edited condition is reflected.
  - The per-skill flagged-event count reads back from the stored flags.
  - An interpretive skill is carried out by the desktop interpreter and returned
    as a labelled inference point (produced_by 'skill', with its skill id), while
    a field-tier skill handed to the interpreter is never run there.

Runs entirely on mocked hardware and the real storage layer and schema, reusing
the field-tier harness from test_observation.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audtheia.storage.database import Skill, new_id, utc_now_iso  # noqa: E402
from audtheia.analysis.observation import QCEngine, TIER_DETERMINISTIC_FLAG, QC_PASSED  # noqa: E402
from audtheia.inference.gguf_llm import GGUFInterpreter  # noqa: E402
from audtheia.analysis.verify import VerificationContext, VerificationVerdict  # noqa: E402

from test_observation import make_settings, fresh_db, write_pending_observation  # noqa: E402


CHECKS = {"passed": 0, "failed": 0}


def check(label: str, condition: bool):
    if condition:
        CHECKS["passed"] += 1
        print(f"  PASS  {label}")
    else:
        CHECKS["failed"] += 1
        print(f"  FAIL  {label}")


def _make_skill(db, *, title, condition, tier=TIER_DETERMINISTIC_FLAG):
    sid = new_id()
    db.upsert_skill(Skill(
        id=sid, title=title,
        trigger_condition="a plain-language note for a person",
        instruction="record a flag",
        tier=tier, created_at=utc_now_iso(), updated_at=utc_now_iso(),
        condition=json.dumps(condition) if condition is not None else None,
    ))
    return sid


# ---------------------------------------------------------------------------
# 1. A field skill fires and its flag is persisted as a derived reading
# ---------------------------------------------------------------------------


def test_field_flag_persists():
    print("\n[1] A deterministic-flag skill fires and persists to skill_flags")
    settings, _ = make_settings(0)
    station = settings.active_station()
    with tempfile.TemporaryDirectory() as td:
        db = fresh_db(settings, Path(td))
        sid = _make_skill(db, title="Weak screening",
                          condition={"source": "observation", "field": "screening_confidence",
                                     "op": "lt", "value": 0.5})
        engine = QCEngine(settings=settings, db=db)

        weak = write_pending_observation(db, station, oid=new_id(), screening_confidence=0.30, salience=0.30)
        strong = write_pending_observation(db, station, oid=new_id(), screening_confidence=0.90, salience=0.90)
        engine.process(weak)
        engine.process(strong)

        weak_flags = db.list_skill_flags(weak)
        strong_flags = db.list_skill_flags(strong)
        check("the matching event carries exactly one flag", len(weak_flags) == 1)
        check("the flag is a derived reading, never inferred",
              weak_flags and weak_flags[0]["data_source"] == "rule_derived")
        check("the flag names its skill and a stable flag name",
              weak_flags and weak_flags[0]["skill_id"] == sid and weak_flags[0]["flag_name"] == "weak_screening")
        check("the non-matching event carries no flag", strong_flags == [])
        check("no interpretation row was written by the field tier", db.list_interpretations(weak) == [])
        check("the measured record is untouched (still passed, no salience change)",
              db.get_observation(weak)["qc_state"] == QC_PASSED
              and abs(db.get_observation(weak)["salience_provisional"] - 0.30) < 1e-9)


# ---------------------------------------------------------------------------
# 2. Persistence is idempotent across repeated scans
# ---------------------------------------------------------------------------


def test_persist_is_idempotent():
    print("\n[2] Re-scanning the same event and skill never duplicates a flag")
    settings, _ = make_settings(0)
    station = settings.active_station()
    with tempfile.TemporaryDirectory() as td:
        db = fresh_db(settings, Path(td))
        _make_skill(db, title="Weak screening",
                    condition={"source": "observation", "field": "screening_confidence",
                               "op": "lt", "value": 0.5})
        engine = QCEngine(settings=settings, db=db)
        oid = write_pending_observation(db, station, oid=new_id(), screening_confidence=0.30, salience=0.30)
        engine.process(oid)
        engine.rescan_flag_skills(oid)
        engine.rescan_flag_skills(oid)
        check("exactly one flag remains after several scans", len(db.list_skill_flags(oid)) == 1)


# ---------------------------------------------------------------------------
# 3. Re-scan applies current skills to a finalized record and clears stale flags
# ---------------------------------------------------------------------------


def test_rescan_finalized_and_clear_stale():
    print("\n[3] Re-scan flags a finalized record, and clears a flag that no longer fires")
    settings, _ = make_settings(0)
    station = settings.active_station()
    with tempfile.TemporaryDirectory() as td:
        db = fresh_db(settings, Path(td))
        engine = QCEngine(settings=settings, db=db)

        # A record finalized before any skill existed.
        oid = write_pending_observation(db, station, oid=new_id(), screening_confidence=0.30, salience=0.30)
        engine.process(oid)
        check("no flags before any skill exists", db.list_skill_flags(oid) == [])

        # Author a matching skill and apply it to the existing record.
        sid = _make_skill(db, title="Weak screening",
                          condition={"source": "observation", "field": "screening_confidence",
                                     "op": "lt", "value": 0.5})
        fired = engine.rescan_flag_skills(oid)
        check("re-scan fires the skill on the finalized record", fired == 1 and len(db.list_skill_flags(oid)) == 1)

        # Edit the condition so it no longer holds; a re-scan must clear the flag.
        db.upsert_skill(Skill(
            id=sid, title="Weak screening",
            trigger_condition="a plain-language note for a person", instruction="record a flag",
            tier=TIER_DETERMINISTIC_FLAG, created_at=utc_now_iso(), updated_at=utc_now_iso(),
            condition=json.dumps({"source": "observation", "field": "screening_confidence",
                                  "op": "lt", "value": 0.1}),
        ))
        engine.rescan_flag_skills(oid)
        check("the stale flag is cleared once the condition no longer holds", db.list_skill_flags(oid) == [])


# ---------------------------------------------------------------------------
# 4. The per-skill flagged-event count reads back from the stored flags
# ---------------------------------------------------------------------------


def test_flag_counts():
    print("\n[4] The flagged-event count is read from the stored flags")
    settings, _ = make_settings(0)
    station = settings.active_station()
    with tempfile.TemporaryDirectory() as td:
        db = fresh_db(settings, Path(td))
        sid = _make_skill(db, title="Weak screening",
                          condition={"source": "observation", "field": "screening_confidence",
                                     "op": "lt", "value": 0.5})
        engine = QCEngine(settings=settings, db=db)
        for conf in (0.10, 0.20, 0.90):
            engine.process(write_pending_observation(db, station, oid=new_id(),
                                                      screening_confidence=conf, salience=conf))
        counts = db.count_skill_flags_by_skill()
        check("two of three events were flagged", counts.get(sid) == 2)


# ---------------------------------------------------------------------------
# 5. An interpretive skill is carried out by the interpreter as a labelled point
# ---------------------------------------------------------------------------


class _ScriptedCompleter:
    def __init__(self, text):
        self._text = text
        self.version = "gguf-test"
        self.prompts = []

    def complete(self, prompt, *, max_tokens=256, temperature=0.2):
        self.prompts.append(prompt)
        return self._text


def test_interpretive_skill_applied():
    print("\n[5] An interpretive skill is applied by the interpreter as a skill point")
    completer = _ScriptedCompleter("This taxon likely forages near dawn in warm shallows.")
    interp = GGUFInterpreter(completer)
    ctx = VerificationContext(
        observation={"id": "obs-1", "station_id": "st-1"},
        child_detections=[],
        environmental_readings=[],
        verdict=VerificationVerdict(resolved_scientific_name="Xestospongia muta"),
        field_scientific_name="Xestospongia muta",
        interpretive_skills=[
            {"id": "skill-42", "tier": "interpretive", "instruction": "note likely foraging behaviour"},
            {"id": "skill-99", "tier": "deterministic_flag", "instruction": "should never run here"},
        ],
    )
    points = interp.interpret(ctx)
    skill_points = [p for p in points if p.get("produced_by") == "skill"]
    check("the interpretive skill produced one skill point", len(skill_points) == 1)
    check("the skill point carries its skill id and skill_note type",
          skill_points and skill_points[0]["skill_id"] == "skill-42"
          and skill_points[0]["point_type"] == "skill_note")
    check("the field-tier skill was never run by the interpreter",
          all(p.get("skill_id") != "skill-99" for p in points))


def main() -> int:
    test_field_flag_persists()
    test_persist_is_idempotent()
    test_rescan_finalized_and_clear_stale()
    test_flag_counts()
    test_interpretive_skill_applied()
    print(f"\n==== skill execution: {CHECKS['passed']} passed, {CHECKS['failed']} failed ====")
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
