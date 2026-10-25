"""One-time database migration: add the tables the Train tab records into.

Path: scripts/migrate_add_training.py

The Train tab turns an expert's reviewed events into a training set, trains a
model on it, evaluates the result on held-out data, registers it under a name
and lets the user activate it. Four desktop-owned tables record that work:

  - frame_box_correction  an expert's correction to the geometry of a box on a
                          stored frame (adjust, add, remove, cleared). Append-only.
  - training_runs         one row per training attempt: its configuration, its
                          hardware, its status and its progress.
  - trained_models        the registry of issued model versions, immutable once
                          written, unique by name ignoring case.
  - model_activation_log  every activation and revert of a model into a
                          configuration slot, with the slot's previous values so
                          a revert is exact. Append-only.

Why separate tables rather than touching anything that exists:

  - The measured record stays exactly as captured. No observation, detection,
    correction, verification or frame review row is read or changed here.
  - Every row is desktop-owned and outside the append-only station to desktop
    pull, so a later sync can never overwrite it. Training artefacts and metrics
    are model outputs (data_source = 'model'); an expert's box or an activation
    is a human act (data_source = 'human_expert').

Safety:

  - Creating new tables does not alter any existing table, so every row is
    untouched.
  - Nothing is changed until a timestamped backup copy has been written.
  - The migration is idempotent. If every table is already there, it reports
    that and changes nothing, so running it twice is harmless. If only some are
    there, it adds the missing ones.

The table definitions are read from audtheia/storage/schema.sql rather than
repeated here, so a database brought forward by this script and a database
created fresh from the schema end up with the same definitions and cannot drift
apart.

Usage (from the repository root):

    python scripts/migrate_add_training.py            # migrate the configured database
    python scripts/migrate_add_training.py --check    # report status only, change nothing
    python scripts/migrate_add_training.py --db path/to/audtheia.db

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

TABLES = (
    "frame_box_correction",
    "trained_models",
    "training_runs",
    "model_activation_log",
)
INDEXES = (
    "idx_frame_box_correction_observation",
    "idx_training_runs_status",
    "idx_model_activation_slot",
)
REQUIRED_TABLES = ("observations",)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _schema_text(repo_root: Path) -> str:
    return (repo_root / "audtheia" / "storage" / "schema.sql").read_text(encoding="utf-8")


def table_statement(schema: str, name: str) -> str:
    """The CREATE TABLE statement for one table, exactly as schema.sql holds it."""
    match = re.search(
        r"CREATE TABLE " + re.escape(name) + r" \(.*?\) STRICT;", schema, flags=re.DOTALL
    )
    if match is None:
        raise RuntimeError(f"schema.sql has no definition for the {name} table")
    return match.group(0).rstrip(";")


def index_statement(schema: str, name: str) -> str:
    """The CREATE INDEX statement for one index, exactly as schema.sql holds it."""
    match = re.search(r"CREATE INDEX " + re.escape(name) + r"\b.*?;", schema, flags=re.DOTALL)
    if match is None:
        raise RuntimeError(f"schema.sql has no definition for the {name} index")
    return match.group(0).rstrip(";")


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


def _index_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def _missing_prerequisites(conn: sqlite3.Connection) -> list[str]:
    """Tables the new ones reference, which must exist before they are created."""
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
        description="Add the training tables to an Audtheia database."
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

    schema = _schema_text(repo_root)

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
        absent_tables = [t for t in TABLES if not _table_exists(conn, t)]
        absent_indexes = [i for i in INDEXES if not _index_exists(conn, i)]
        observations_before = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    finally:
        conn.close()

    if not absent_tables and not absent_indexes:
        print(f"Already migrated: {db_path} already has every training table. No change made.")
        return 0
    if args.check:
        print(f"Migration needed: {db_path} is missing {', '.join(absent_tables + absent_indexes)}.")
        print(f"  {observations_before} observations are present and will not be modified.")
        return 0

    print(f"Migrating {db_path}")
    backup_path = _backup(db_path)
    print(f"  Backup written to {backup_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        for name in TABLES:
            if name in absent_tables:
                conn.execute(table_statement(schema, name))
        for name in INDEXES:
            if name in absent_indexes:
                conn.execute(index_statement(schema, name))
        conn.commit()
        observations_after = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        if observations_before != observations_after:
            raise RuntimeError(
                f"observation count changed during migration:"
                f" {observations_before} then {observations_after}"
            )
        for name in TABLES:
            if not _table_exists(conn, name):
                raise RuntimeError(f"the {name} table was not created")
    except (sqlite3.Error, RuntimeError) as exc:
        print(f"  Migration failed: {exc}", file=sys.stderr)
        print(f"  A backup is available at {backup_path}.", file=sys.stderr)
        return 1
    finally:
        conn.close()

    print(
        f"  Added {len(absent_tables)} table(s) and {len(absent_indexes)} index(es)."
        f" {observations_before} observations preserved and unchanged."
    )
    print("  The Train tab can now record training runs and registered models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
