"""Checks the migration that lets a candidate pattern be withdrawn.

Path: tests/test_migration_pattern_withdrawn.py

Builds a database in a temporary folder whose patterns table has the layout it
had before withdrawal existed, fills it with a dream pass, events, patterns and
the links between them, and proves:

  - Before the migration, deleting an event still works and says the database
    needs its migration, rather than failing.
  - The migration writes a backup first, keeps every pattern and every pattern
    link unchanged, and leaves the patterns table defined exactly as a fresh
    database defines it. Running it again changes nothing.
  - After the migration, deleting an event withdraws each live pattern resting on
    it, with when and why and the status it had before, keeps a rejected pattern
    rejected, and leaves patterns on other events untouched.

Nothing here touches a real installation's data, database, reports or models.
"""

from __future__ import annotations

import re
import sqlite3
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import migrate_add_pattern_withdrawn as mig  # noqa: E402
from audtheia.storage.database import (  # noqa: E402
    Database, DreamPass, Observation, Pattern, Station, new_id, utc_now_iso,
)

SCHEMA = REPO / "audtheia" / "storage" / "schema.sql"
_checks = 0


def check(label: str, condition: bool) -> None:
    global _checks
    if not condition:
        raise AssertionError(label)
    _checks += 1
    print(f"  ok  {label}")


def old_schema() -> str:
    """schema.sql with the patterns table as it was before withdrawal existed."""
    text = SCHEMA.read_text(encoding="utf-8")
    text = text.replace("CHECK (status IN ('candidate', 'verified', 'rejected', 'withdrawn'))",
                        "CHECK (status IN ('candidate', 'verified', 'rejected'))")
    text = re.sub(r"\n\s*withdrawn_at\s+TEXT,[^\n]*", "", text)
    text = re.sub(r"\n\s*withdrawn_reason\s+TEXT,[^\n]*", "", text)
    assert "withdrawn_at" not in text.split("CREATE TABLE patterns")[1].split(") STRICT")[0]
    return text


def seed(db: Database) -> dict:
    now = utc_now_iso()
    sid = new_id()
    db.upsert_station(Station(id=sid, station_name="Reef", environment_type="marine", created_at=now))
    obs = []
    for i in range(4):
        oid = new_id()
        db.insert_observation(Observation(
            id=oid, event_name=f"ev{i}", station_id=sid, trigger_source="vision",
            first_seen=f"2026-03-0{i + 1}T10:00:00Z", last_seen=f"2026-03-0{i + 1}T10:00:00Z", duration=1.0,
            data_source="model", created_at=now, qc_state="qc_passed"), children=[])
        obs.append(oid)
    db.create_dream_pass(DreamPass(id="pass-1", phase_reached="complete", status="complete",
                                   started_at=now, created_at=now, ended_at=now))

    def pattern(pid, status, links):
        db.insert_pattern(Pattern(id=pid, dream_pass_id="pass-1", dream_phase="rem", status=status,
                                  data_span_start="2026-03-01T10:00:00Z", data_span_end="2026-03-04T10:00:00Z",
                                  n=len(links), description=pid, created_at=now), observation_ids=links)

    pattern("live", "candidate", obs[0:2])
    pattern("expert", "verified", [obs[0], obs[2]])
    pattern("rejected", "rejected", [obs[0], obs[3]])
    pattern("other", "candidate", obs[2:4])
    return {"obs": obs}


def counts(path: Path) -> tuple:
    conn = sqlite3.connect(str(path))
    try:
        return (conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0],
                sorted(conn.execute("SELECT pattern_id, observation_id FROM pattern_observations").fetchall()))
    finally:
        conn.close()


def run() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        old_path = root / "old.db"
        conn = sqlite3.connect(str(old_path))
        conn.executescript(old_schema())
        conn.close()
        db = Database(str(old_path))
        world = seed(db)

        print("\n[1] Before the migration")
        check("the old table cannot record a withdrawal", not Database._patterns_can_withdraw(sqlite3.connect(str(old_path))))
        spare = new_id()
        with db.connect() as c:
            c.execute("INSERT INTO observations (id, event_name, station_id, trigger_source, first_seen, last_seen, "
                      "duration, data_source, created_at, qc_state) SELECT ?, 'spare', station_id, trigger_source, "
                      "first_seen, last_seen, duration, data_source, created_at, qc_state FROM observations LIMIT 1",
                      (spare,))
        result = db.delete_observations([spare])
        check("deleting still works and reports that the migration is needed",
              result["deleted"] == [spare] and result["patterns_withdrawn"] is None)
        before = counts(old_path)

        print("\n[2] The migration")
        check("the check mode reports the migration is needed and changes nothing",
              mig.main(["--db", str(old_path), "--check"]) == 0 and counts(old_path) == before)
        check("the migration succeeds", mig.main(["--db", str(old_path)]) == 0)
        check("a backup was written first", any(p.name.startswith("old.db.backup-") for p in root.iterdir()))
        check("every pattern and every pattern link is kept unchanged", counts(old_path) == before)
        fresh = root / "fresh.db"
        Database(str(fresh)).initialize_schema(SCHEMA)

        def definition(path):
            c = sqlite3.connect(str(path))
            try:
                return [r[0] for r in c.execute(
                    "SELECT sql FROM sqlite_master WHERE tbl_name = 'patterns' ORDER BY name").fetchall()]
            finally:
                c.close()

        check("the patterns table and its indexes match a fresh database exactly", definition(old_path) == definition(fresh))
        c = sqlite3.connect(str(old_path))
        check("the database passes its integrity and foreign key checks",
              c.execute("PRAGMA integrity_check").fetchone()[0] == "ok" and not c.execute("PRAGMA foreign_key_check").fetchall())
        c.close()
        check("running it again changes nothing", mig.main(["--db", str(old_path)]) == 0 and counts(old_path) == before)

        print("\n[3] Withdrawing on deletion")
        db = Database(str(old_path))
        result = db.delete_observations([world["obs"][0]])
        check("two live patterns rested on the deleted event and both are withdrawn", result["patterns_withdrawn"] == 2)
        live, expert = db.get_pattern("live"), db.get_pattern("expert")
        check("a withdrawn candidate keeps when and why", live["status"] == "withdrawn" and live["withdrawn_at"]
              and live["withdrawn_reason"] == "a supporting event was deleted; status before: candidate")
        check("an expert-verified pattern keeps its earlier status in the reason",
              expert["status"] == "withdrawn" and expert["withdrawn_reason"].endswith("status before: verified"))
        check("a rejected pattern stays rejected", db.get_pattern("rejected")["status"] == "rejected"
              and db.get_pattern("rejected")["withdrawn_at"] is None)
        check("a pattern on other events is untouched", db.get_pattern("other")["status"] == "candidate")
        check("a withdrawn pattern is kept, with its remaining links", db.list_pattern_observations("live") == [world["obs"][1]])
        check("a new pattern can still be written after the migration", db.insert_pattern(Pattern(
            id="new", dream_pass_id="pass-1", dream_phase="nrem", data_span_start="2026-03-01T10:00:00Z",
            data_span_end="2026-03-02T10:00:00Z", n=1, description="new", created_at=utc_now_iso()),
            observation_ids=[world["obs"][1]]) is None and db.get_pattern("new")["status"] == "candidate")

    print(f"\nAll {_checks} pattern withdrawal migration checks passed.")


if __name__ == "__main__":
    try:
        run()
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
