"""One-time database migration: record a human per-frame verdict on an event.

Path: scripts/migrate_add_frame_review.py

A visual event is one tracked object across many saved frames, and a screening
model can disagree with itself frame to frame (a sponge read as one species on
most frames and a neighbour on a few, a sparrow read as a wren on a handful).
The event still resolves to one taxon, but until now there was nowhere for an
expert to mark an individual frame accurate or not.

This migration adds one desktop-owned table, "frame_review", holding a human
expert's per-frame verdict: 'accurate', 'inaccurate', or 'cleared' (a retraction
that returns a frame to unreviewed without deleting history).

Why a separate table rather than editing the frame annotation or the event:

  - The measured record of what the model did stays exactly as captured. The
    event's frame_count, duration, screening_confidence and salience_provisional
    are never altered by a review; an 'inaccurate' verdict is subtracted only
    from the expert-curated view and from the frame count the analysis trusts.
  - The row is data_source = 'human_expert', desktop-owned, and outside the
    append-only Pi -> desktop pull, so a later sync can never overwrite it. This
    is the same firewall observation_corrections uses.

The table is append-only. A change of mind is a new row, so the review history
of a frame is never destroyed; readers take the most recent row per
(observation_id, frame_index).

Safety:

  - Creating a new table does not alter any existing table, so every
    observation, detection, correction and verification row is untouched.
  - Nothing is changed until a timestamped backup copy has been written.
  - The migration is idempotent. If the table is already there, it reports that
    and changes nothing, so running it twice is harmless.

Usage (from the repository root):

    python scripts/migrate_add_frame_review.py            # migrate the configured database
    python scripts/migrate_add_frame_review.py --check    # report status only, change nothing
    python scripts/migrate_add_frame_review.py --db path/to/audtheia.db

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

TABLE = "frame_review"
REQUIRED_TABLES = ("observations",)

# Kept in full and identical to audtheia/storage/schema.sql, so a database
# created fresh from the schema and a database brought forward by this migration
# end up with the same table definition. SQLite stores the CREATE statement
# verbatim, so the leading and trailing whitespace of this literal is stripped
# before execution.
CREATE_TABLE_SQL = """
CREATE TABLE frame_review (
    id                  TEXT PRIMARY KEY,       -- UUID
    observation_id      TEXT NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
    frame_index         INTEGER NOT NULL,       -- the per-frame annotation index this verdict is about

    verdict             TEXT NOT NULL CHECK (verdict IN ('accurate', 'inaccurate', 'cleared')),

    corrector           TEXT NOT NULL,          -- required: an anonymous review is not reviewable
    reviewed_at         TEXT NOT NULL,          -- UTC ISO8601
    reason              TEXT,

    data_source         TEXT NOT NULL CHECK (data_source = 'human_expert'),
    created_at          TEXT NOT NULL
) STRICT
"""

# The line breaks and indentation here are deliberate and must keep matching
# schema.sql for the same verbatim-storage reason as the table above.
CREATE_INDEX_SQL = (
    "CREATE INDEX idx_frame_review_observation\n"
    "    ON frame_review(observation_id, frame_index, reviewed_at)",
)


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


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def _missing_prerequisites(conn: sqlite3.Connection) -> list[str]:
    """Tables the new one references, which must exist before it is created."""
    return [name for name in REQUIRED_TABLES if not _table_exists(conn, name)]


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
    parser = argparse.ArgumentParser(
        description="Add the per-frame review table to an Audtheia database."
    )
    parser.add_argument("--db", help="path to the database file (defaults to the configured path)")
    parser.add_argument(
        "--check",
        action="store_true",
        help="report whether the migration is needed and change nothing",
    )
    args = parser.parse_args(argv)

    repo_root = _repo_root()
    db_path = Path(args.db).resolve() if args.db else _configured_db(repo_root)

    if not db_path.exists():
        print(f"No database found at {db_path}. Nothing to migrate.")
        return 0

    conn = sqlite3.connect(str(db_path))
    try:
        missing = _missing_prerequisites(conn)
        if missing:
            print(
                f"Cannot migrate {db_path}: it has no {', '.join(missing)} table."
                " Initialize the database from the schema first.",
                file=sys.stderr,
            )
            return 2
        already = _table_exists(conn, TABLE)
        observations_before = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    finally:
        conn.close()

    if already:
        print(f"Already migrated: {db_path} already has a '{TABLE}' table. No change made.")
        return 0
    if args.check:
        print(f"Migration needed: {db_path} has no '{TABLE}' table yet.")
        print(f"  {observations_before} observations are present and will not be modified.")
        return 0

    print(f"Migrating {db_path}")
    backup_path = _backup(db_path)
    print(f"  Backup written to {backup_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(CREATE_TABLE_SQL.strip())
        for statement in CREATE_INDEX_SQL:
            conn.execute(statement)
        conn.commit()
        observations_after = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        if observations_before != observations_after:
            raise RuntimeError(
                f"observation count changed during migration:"
                f" {observations_before} then {observations_after}"
            )
        if not _table_exists(conn, TABLE):
            raise RuntimeError(f"the {TABLE} table was not created")
    except (sqlite3.Error, RuntimeError) as exc:
        print(f"  Migration failed: {exc}", file=sys.stderr)
        print(f"  A backup is available at {backup_path}.", file=sys.stderr)
        return 1
    finally:
        conn.close()

    print(f"  Created the '{TABLE}' table. {observations_before} observations preserved and unchanged.")
    print("  An expert can now mark individual frames accurate or inaccurate on the Detections tab.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
