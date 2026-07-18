"""Verification for the station-position database migration.

Path: tests/test_migration_gps.py

Proves the one-time migration that widens the location-status constraint does
exactly what it must and nothing else: it preserves every row, keeps foreign-key
references intact, ends with a table identical to a fresh install's, allows the
new "station_configured" status while still rejecting an invalid one, writes a
backup before touching anything, and is safe to run twice.

The check builds a database in the pre-migration shape (the old constraint),
fills it with observations and a child detection, runs the real migration script
as a person would, and reads the result back against the current schema. It uses
only the standard library, so it runs on any machine with no extra packages.
"""

from __future__ import annotations

import re
import sqlite3
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "migrate_add_station_configured.py"
SCHEMA = REPO / "audtheia" / "storage" / "schema.sql"

CHECKS = {"passed": 0, "failed": 0}


def check(label: str, condition: bool) -> None:
    if condition:
        CHECKS["passed"] += 1
        print(f"  PASS  {label}")
    else:
        CHECKS["failed"] += 1
        print(f"  FAIL  {label}")


def _old_schema() -> str:
    """The current schema with the new status removed, i.e. the pre-migration shape.

    Removing the single ", 'station_configured'" entry from the gps_status list
    reproduces exactly the constraint that shipped before this migration, so the
    starting database is a faithful old one rather than an invented approximation.
    """
    text = SCHEMA.read_text(encoding="utf-8")
    old = text.replace(", 'station_configured'", "")
    # Confirm the value is gone from the constraint itself, ignoring any comment
    # that merely mentions it by name.
    code = re.sub(r"--[^\n]*", "", _observations_block(old))
    assert "station_configured" not in code, "failed to build a pre-migration schema"
    return old


def _observations_block(schema_text: str) -> str:
    m = re.search(r"CREATE TABLE observations\s*\(.*?\)\s*STRICT\s*;", schema_text, re.DOTALL)
    return m.group(0) if m else ""


def _norm(sql: str) -> str:
    # SQLite quotes a table name after a rename; normalize that and whitespace so
    # a structural comparison is not tripped by cosmetics.
    return re.sub(r"\s+", " ", sql).replace('"observations"', "observations").strip()


def _seed(db_path: Path) -> tuple[int, int]:
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_old_schema())
    station_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO stations (id, station_name, environment_type, created_at) VALUES (?,?,?,?)",
        (station_id, "MigTestReef", "marine", "2026-07-17T00:00:00Z"),
    )
    obs_ids = []
    for i, status in enumerate(["measured", "not_measured", None]):
        oid = str(uuid.uuid4())
        obs_ids.append(oid)
        conn.execute(
            """INSERT INTO observations
               (id, event_name, station_id, trigger_source, first_seen, last_seen, duration,
                time_provisional, data_source, gps_latitude, gps_longitude, gps_status, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (oid, f"MigTest_{i}_{oid[:8]}", station_id, "vision",
             "2026-07-17T00:00:00Z", "2026-07-17T00:00:01Z", 1.0, 0, "model",
             18.2 if status == "measured" else None,
             -67.1 if status == "measured" else None,
             status, "2026-07-17T00:00:02Z"),
        )
    # A child detection so the foreign-key check has something real to confirm.
    conn.execute(
        """INSERT INTO child_detections
           (id, observation_id, modality, scientific_name, confidence, data_source, status, created_at)
           VALUES (?,?,?,?,?,?,?,?)""",
        (str(uuid.uuid4()), obs_ids[0], "vision", "Aplysina fistularis", 0.9, "model", "measured",
         "2026-07-17T00:00:02Z"),
    )
    obs = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    children = conn.execute("SELECT COUNT(*) FROM child_detections").fetchone()[0]
    conn.commit()
    conn.close()
    return obs, children


def _run_migration(db_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--db", str(db_path), "--schema", str(SCHEMA)],
        capture_output=True, text=True,
    )


def test_migration() -> None:
    print("\n[1] Station-position migration on a copy of an old-shape database")
    work = Path(tempfile.mkdtemp(prefix="audtheia-mig-"))
    db_path = work / "audtheia.db"
    obs_before, children_before = _seed(db_path)

    # The old database must genuinely reject the new value before migrating.
    conn = sqlite3.connect(str(db_path))
    rejected_before = False
    try:
        conn.execute(
            """INSERT INTO observations
               (id, event_name, station_id, trigger_source, first_seen, last_seen, duration,
                time_provisional, data_source, gps_status, created_at)
               VALUES (?,?, (SELECT id FROM stations LIMIT 1), ?,?,?,?,?,?,?,?)""",
            (str(uuid.uuid4()), "PreBad", "vision", "2026-07-17T00:00:00Z",
             "2026-07-17T00:00:01Z", 1.0, 0, "model", "station_configured", "2026-07-17T00:00:02Z"),
        )
    except sqlite3.IntegrityError:
        rejected_before = True
    finally:
        conn.rollback()
        conn.close()
    check("the old database rejects station_configured before migrating", rejected_before)

    result = _run_migration(db_path)
    check("migration reports success", result.returncode == 0)

    backups = list(work.glob("audtheia.db.backup-*"))
    check("a backup was written before any change", len(backups) == 1)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    obs_after = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    children_after = conn.execute("SELECT COUNT(*) FROM child_detections").fetchone()[0]
    check("every observation row is preserved", obs_after == obs_before)
    check("every child detection is preserved", children_after == children_before)

    violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    check("no foreign-key reference was orphaned", violations == [])

    indexes = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='observations' AND sql IS NOT NULL"
    )}
    expected = {
        "idx_observations_station_id_first_seen", "idx_observations_first_seen",
        "idx_observations_synced_pending", "idx_observations_qc_state", "idx_observations_data_source",
    }
    check("every index was recreated", expected <= indexes)

    # The new value is now accepted, and a bogus one is still refused.
    station_id = conn.execute("SELECT id FROM stations LIMIT 1").fetchone()[0]
    conn.execute(
        """INSERT INTO observations
           (id, event_name, station_id, trigger_source, first_seen, last_seen, duration,
            time_provisional, data_source, gps_latitude, gps_longitude, gps_status, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (str(uuid.uuid4()), "PostGood", station_id, "vision", "2026-07-17T00:00:00Z",
         "2026-07-17T00:00:01Z", 1.0, 0, "model", 18.21, -67.15, "station_configured", "2026-07-17T00:00:02Z"),
    )
    check("station_configured is now accepted", True)
    rejected_after = False
    try:
        conn.execute(
            """INSERT INTO observations
               (id, event_name, station_id, trigger_source, first_seen, last_seen, duration,
                time_provisional, data_source, gps_status, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (str(uuid.uuid4()), "PostBad", station_id, "vision", "2026-07-17T00:00:00Z",
             "2026-07-17T00:00:01Z", 1.0, 0, "model", "not_a_status", "2026-07-17T00:00:02Z"),
        )
    except sqlite3.IntegrityError:
        rejected_after = True
    check("an invalid status is still rejected after migrating", rejected_after)
    migrated_sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='observations'"
    ).fetchone()[0]
    conn.rollback()
    conn.close()

    # The migrated table is structurally what a fresh install would create.
    fresh = work / "fresh.db"
    fconn = sqlite3.connect(str(fresh))
    fconn.executescript(SCHEMA.read_text(encoding="utf-8"))
    fresh_sql = fconn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='observations'"
    ).fetchone()[0]
    fconn.close()
    check("the migrated table matches a fresh schema build", _norm(migrated_sql) == _norm(fresh_sql))

    # Running it again changes nothing and reports so.
    again = _run_migration(db_path)
    check("a second run is a safe no-op", again.returncode == 0 and "Already migrated" in again.stdout)


def main() -> int:
    print("=" * 72)
    print("Station-position migration: verification on a copy")
    print("=" * 72)
    if not SCRIPT.exists():
        print("  FAIL  migration script not found")
        return 1
    test_migration()
    print("\n" + "=" * 72)
    print(f"RESULT: {CHECKS['passed']} passed, {CHECKS['failed']} failed")
    print("=" * 72)
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
