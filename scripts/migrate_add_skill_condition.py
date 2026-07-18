"""One-time database migration: give a skill a checkable condition.

Path: scripts/migrate_add_skill_condition.py

A skill has always carried the text a person writes: when it applies, and what to
do. That text is for a human reader, and the field station deliberately never
interprets it, because reading prose and acting on it would be inference, which
the field tier is not permitted to do.

This migration adds one nullable column, "condition", to the skills table. It
holds a small structured comparison, such as "the screening confidence is below
0.45", which the field engine compiles into a plain function of the measured
values in a record. That is what lets a skill someone authored actually run
without the engine ever interpreting a sentence.

Safety:

  - Adding a nullable column does not rewrite the table, so existing skills are
    untouched and simply have no condition until one is set.
  - Nothing is changed until a timestamped backup copy has been written.
  - The migration is idempotent. If the column is already there, it reports that
    and changes nothing, so running it twice is harmless.

Usage (from the repository root):

    python scripts/migrate_add_skill_condition.py            # migrate the configured database
    python scripts/migrate_add_skill_condition.py --check    # report status only, change nothing
    python scripts/migrate_add_skill_condition.py --db path/to/audtheia.db

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

COLUMN = "condition"
TABLE = "skills"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _configured_db(repo_root: Path) -> Path:
    """The database path from the configuration, or the conventional location."""
    settings_file = repo_root / "config" / "settings.json"
    if settings_file.exists():
        try:
            paths = json.loads(settings_file.read_text(encoding="utf-8")).get("paths", {})
            if isinstance(paths, dict) and paths.get("db_path"):
                candidate = Path(paths["db_path"])
                return candidate if candidate.is_absolute() else repo_root / candidate
        except (json.JSONDecodeError, OSError):
            pass
    return repo_root / "database" / "audtheia.db"


def _has_column(conn: sqlite3.Connection) -> bool:
    return any(row[1] == COLUMN for row in conn.execute(f"PRAGMA table_info({TABLE})"))


def _table_exists(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (TABLE,)
    ).fetchone()
    return row is not None


def _backup(db_path: Path) -> Path:
    """Fold any write-ahead log into the file, then copy it aside."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.commit()
    finally:
        conn.close()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = db_path.with_name(f"{db_path.name}.backup-{stamp}")
    shutil.copy2(str(db_path), str(backup_path))
    return backup_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Add the skill condition column to an Audtheia database.")
    parser.add_argument("--db", help="path to the database file (defaults to the configured path)")
    parser.add_argument("--check", action="store_true", help="report whether the migration is needed and change nothing")
    args = parser.parse_args(argv)

    repo_root = _repo_root()
    db_path = Path(args.db).resolve() if args.db else _configured_db(repo_root)

    if not db_path.exists():
        print(f"No database found at {db_path}. Nothing to migrate.")
        return 0

    conn = sqlite3.connect(str(db_path))
    try:
        if not _table_exists(conn):
            print(f"No {TABLE} table exists in {db_path}; initialize it from the schema first.", file=sys.stderr)
            return 2
        already = _has_column(conn)
    finally:
        conn.close()

    if already:
        print(f"Already migrated: {db_path} already has a skill '{COLUMN}' column. No change made.")
        return 0
    if args.check:
        print(f"Migration needed: {db_path} has no skill '{COLUMN}' column yet.")
        return 0

    print(f"Migrating {db_path}")
    backup_path = _backup(db_path)
    print(f"  Backup written to {backup_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        skills_before = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
        conn.execute(f"ALTER TABLE {TABLE} ADD COLUMN {COLUMN} TEXT")
        conn.commit()
        skills_after = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
        if skills_before != skills_after:
            raise RuntimeError(f"skill count changed during migration: {skills_before} then {skills_after}")
    except (sqlite3.Error, RuntimeError) as exc:
        print(f"  Migration failed: {exc}", file=sys.stderr)
        print(f"  A backup is available at {backup_path}.", file=sys.stderr)
        return 1
    finally:
        conn.close()

    print(f"  Added the '{COLUMN}' column. {skills_before} existing skills preserved and unchanged.")
    print("  A skill runs at the field tier once you give it a condition in the Skills panel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
