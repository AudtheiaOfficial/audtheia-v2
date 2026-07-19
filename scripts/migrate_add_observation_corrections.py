"""One-time database migration: record a human correction of a detection.

Path: scripts/migrate_add_observation_corrections.py

A screening model names what it saw, and the desktop verifier re-scores that call
with a second model. Both are model claims. Neither is a taxonomist, and until
now there was nowhere for a taxonomist to disagree.

This migration adds one desktop-owned table, "observation_corrections", holding a
human expert's verdict on an event or on one child detection within it. The
verdict is one of three: the expert agrees with the model ("confirm"), the expert
supplies a different name ("relabel"), or the expert states that no organism is
present at all ("reject").

Why a separate table rather than editing the detection in place:

  - child_detections carries CHECK (data_source = 'model'), so the storage layer
    itself refuses to hold a human claim in that table. The firewall between a
    measured model output and a human judgement is enforced by the schema, not by
    convention, and this migration keeps it that way.
  - The Pi-to-desktop sync is append-only over station-owned rows. A correction
    is desktop-owned, so a later sync can never overwrite a taxonomist's work
    (decision #47).
  - The model's original call stays exactly as recorded, forever. A correction is
    a third, separately sourced claim standing beside the field call and the
    desktop verification, not a replacement for either.

The table is append-only. A change of mind is a new row, so the review history of
a contested identification is never destroyed and can always be read back in
order. Readers take the most recent row for a target.

No confidence number is written anywhere by a correction. An expert
identification is a different kind of statement from a model score, not the top
of the same scale, and writing one into screening_confidence would silently
corrupt every mean-confidence figure in Analytics and every drift measurement in
the Brain versions panel with a value no model produced. The salience implied by
a correction is stored here instead, on the row that caused it, so it is always
traceable to the specific verdict that produced it and neither existing salience
slot is touched.

Safety:

  - Creating a new table does not alter any existing table, so every observation,
    detection, and verification row is untouched.
  - Nothing is changed until a timestamped backup copy has been written.
  - The migration is idempotent. If the table is already there, it reports that
    and changes nothing, so running it twice is harmless.

Usage (from the repository root):

    python scripts/migrate_add_observation_corrections.py            # migrate the configured database
    python scripts/migrate_add_observation_corrections.py --check    # report status only, change nothing
    python scripts/migrate_add_observation_corrections.py --db path/to/audtheia.db

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

TABLE = "observation_corrections"
REQUIRED_TABLES = ("observations", "child_detections")

# The table definition is kept here in full and is the same text that
# audtheia/storage/schema.sql carries, so a database created fresh from the
# schema and a database brought forward by this migration end up identical.
#
# SQLite stores a CREATE statement verbatim in sqlite_master, surrounding
# whitespace included, so this is stripped before execution. Without that, a
# migrated database and a freshly created one would differ by the leading and
# trailing newlines of this literal, and any future check comparing the two
# definitions would report a spurious mismatch.
CREATE_TABLE_SQL = """
CREATE TABLE observation_corrections (
    id                          TEXT PRIMARY KEY,       -- UUID
    observation_id              TEXT NOT NULL REFERENCES observations(id) ON DELETE CASCADE,

    -- Which claim the expert is correcting. NULL means the verdict applies to
    -- the event as a whole; a value names the one child detection being
    -- corrected, which is what lets a single wrong box in a multi-taxon event be
    -- fixed without disturbing the others.
    detection_id                TEXT REFERENCES child_detections(id) ON DELETE CASCADE,
    modality                    TEXT CHECK (modality IS NULL OR modality IN ('vision', 'audio')),

    -- The three-way vocabulary. 'confirm' is an expert agreeing with the model,
    -- which is a real and valuable signal rather than a no-op: it is the only
    -- way a true positive is ever distinguished from an unreviewed one.
    -- 'reject' states that no organism is present, which is what allows a
    -- retraining export to carry true negatives and so teach a model what a
    -- false positive looks like.
    verdict                     TEXT NOT NULL CHECK (verdict IN ('confirm', 'relabel', 'reject')),

    -- The corrected taxon. Resolved against the shipped GBIF backbone before the
    -- row is ever written, so a correction cannot introduce a taxon that does
    -- not exist; an unresolvable name is refused rather than stored as free text.
    corrected_scientific_name   TEXT,
    corrected_common_name       TEXT,
    corrected_gbif_usage_key    TEXT,
    gbif_snapshot_date          TEXT,     -- backbone snapshot the corrected name resolved against

    -- Who made the call and why. The corrector is required: an anonymous expert
    -- identification is not reviewable, and reviewability is the entire point.
    corrector                   TEXT NOT NULL,
    corrected_at                TEXT NOT NULL,          -- UTC ISO8601
    reason                      TEXT,

    -- Salience recomputed under this verdict. Stored here rather than on the
    -- observation so the station-owned provisional value and the desktop-owned
    -- authoritative value both stay exactly as their own producers wrote them.
    salience_corrected          REAL CHECK (salience_corrected IS NULL OR
                                    (salience_corrected >= 0 AND salience_corrected <= 1)),

    data_source                 TEXT NOT NULL CHECK (data_source = 'human_expert'),
    status                      TEXT NOT NULL DEFAULT 'measured' CHECK (status IN
                                    ('measured', 'not_measured', 'below_detection_limit',
                                     'sensor_error', 'not_applicable')),
    created_at                  TEXT NOT NULL,

    -- A rejection asserts that nothing is there, so it must not carry a name; a
    -- relabel exists precisely to supply one, so it must. A confirmation may
    -- echo the model's name or leave it out, since the name it agrees with is
    -- already on the detection.
    CHECK (
        (verdict = 'reject'  AND corrected_scientific_name IS NULL) OR
        (verdict = 'relabel' AND corrected_scientific_name IS NOT NULL) OR
        (verdict = 'confirm')
    )
) STRICT
"""

# Line breaks and indentation here are deliberate and must keep matching
# schema.sql for the same verbatim-storage reason as the table above.
CREATE_INDEX_SQL = (
    "CREATE INDEX idx_observation_corrections_observation_id\n"
    "    ON observation_corrections(observation_id, corrected_at)",
    "CREATE INDEX idx_observation_corrections_detection_id\n"
    "    ON observation_corrections(detection_id)",
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
        description="Add the human-correction table to an Audtheia database."
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
        print(f"Already migrated: {db_path} already has an '{TABLE}' table. No change made.")
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
    print("  A taxonomist can now confirm, relabel, or reject a detection from the Detections and Audio tabs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
