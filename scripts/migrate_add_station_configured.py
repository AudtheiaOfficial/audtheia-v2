"""One-time database migration: allow an entered fixed station position.

Path: scripts/migrate_add_station_configured.py

Audtheia records a location status alongside every observation. This migration
adds one more allowed value to that status, "station_configured", so a station
with no live satellite receiver but a known, surveyed position can record that
entered position as its location, kept clearly separate from a live fix.

The status lives in a CHECK constraint on the observations table. SQLite cannot
add a value to an existing CHECK with ALTER TABLE, so the table is rebuilt: a new
table with the wider constraint is created, every row is copied into it, the old
table is dropped, and the new one takes its place, with the indexes recreated and
the foreign keys checked before anything is committed. The whole rebuild runs in
one transaction, so it either completes fully or leaves the database untouched.

Safety:

  - Nothing is changed until a timestamped backup copy of the database has been
    written next to it.
  - The rebuild is a single transaction guarded by a foreign-key check; any
    problem rolls the whole thing back.
  - The migration is idempotent. If the database already allows the value, it
    reports that and changes nothing, so running it twice is harmless.
  - The new table is built from the project's schema file, so its shape always
    matches the current schema rather than a copy that could drift out of date.

Usage (from the repository root):

    python scripts/migrate_add_station_configured.py            # migrate the configured database
    python scripts/migrate_add_station_configured.py --check    # report status only, change nothing
    python scripts/migrate_add_station_configured.py --db path/to/audtheia.db --schema path/to/schema.sql

With no --db, the database path is read from config/settings.json; with no
--schema, the schema file is read from the same configuration, falling back to
audtheia/storage/schema.sql. Only the Python standard library is used.
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

NEW_STATUS = "station_configured"
TABLE = "observations"
TEMP_TABLE = "observations_new"


def _repo_root() -> Path:
    # scripts/ sits directly under the repository root.
    return Path(__file__).resolve().parent.parent


def _resolve_from_settings(repo_root: Path) -> tuple[Path, Path]:
    """Read the database and schema paths from config/settings.json when present.

    Falls back to the conventional locations so the migration still runs against
    a checkout whose configuration has moved or is absent.
    """
    db_path = repo_root / "database" / "audtheia.db"
    schema_path = repo_root / "audtheia" / "storage" / "schema.sql"
    settings_file = repo_root / "config" / "settings.json"
    if settings_file.exists():
        try:
            raw = json.loads(settings_file.read_text(encoding="utf-8"))
            paths = raw.get("paths", {})
            if isinstance(paths, dict):
                if paths.get("db_path"):
                    db_path = _resolve(repo_root, paths["db_path"])
                if paths.get("schema_path"):
                    schema_path = _resolve(repo_root, paths["schema_path"])
        except (json.JSONDecodeError, OSError):
            # A malformed or unreadable configuration falls back to the defaults
            # above rather than stopping the migration.
            pass
    return db_path, schema_path


def _resolve(repo_root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (repo_root / p)


def _extract_observations_ddl(schema_sql: str) -> tuple[str, list[str]]:
    """Pull the observations table and its indexes out of the schema file.

    Building the replacement table straight from the schema keeps the migration
    honest: the rebuilt table is exactly what a fresh install would create, not a
    second copy of the definition that could fall out of step with it.
    """
    match = re.search(
        r"CREATE TABLE observations\s*\((.*?)\)\s*STRICT\s*;",
        schema_sql,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        raise MigrationError("could not find the observations table definition in the schema file")
    create_temp = f"CREATE TABLE {TEMP_TABLE} ({match.group(1)}) STRICT;"

    # Only indexes declared ON observations(...) belong to this table. Matching
    # the open parenthesis avoids catching a differently named table that merely
    # starts with the same word.
    indexes = re.findall(
        r"CREATE INDEX[^;]*?\bON\s+observations\s*\([^;]*?;",
        schema_sql,
        re.DOTALL | re.IGNORECASE,
    )
    if not indexes:
        raise MigrationError("could not find any indexes for the observations table in the schema file")
    return create_temp, [i.strip() for i in indexes]


class MigrationError(RuntimeError):
    """A problem that should stop the migration with a clear message."""


def _table_sql(conn: sqlite3.Connection, table: str) -> str | None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    return row[0] if row else None


def _already_migrated(conn: sqlite3.Connection) -> bool:
    sql = _table_sql(conn, TABLE)
    if sql is None:
        raise MigrationError(
            f"no {TABLE} table exists in this database; initialize it from the schema before migrating"
        )
    # SQLite stores the CREATE TABLE text with its comments, and a comment may
    # name the new status while the constraint does not yet allow it. Strip line
    # comments first so the decision reads the constraint itself, never prose.
    code = re.sub(r"--[^\n]*", "", sql)
    return NEW_STATUS in code


def _backup(db_path: Path) -> Path:
    """Fold any write-ahead log into the main file, then copy it aside.

    Checkpointing first means the backup is a single self-contained file rather
    than one that depends on a separate log the copy would miss.
    """
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


def _migrate(db_path: Path, create_temp: str, indexes: list[str]) -> tuple[int, int]:
    """Rebuild the observations table with the wider status constraint.

    Returns the row count before and after so the caller can confirm no row was
    lost. Foreign keys are turned off only for the drop-and-rename, which is the
    documented way to rebuild a table other tables reference, and the foreign-key
    check inside the transaction proves nothing was orphaned.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        before = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]

        # foreign_keys must be toggled outside a transaction to take effect.
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("BEGIN")
        try:
            conn.execute(create_temp)

            columns = [r[1] for r in conn.execute(f"PRAGMA table_info({TEMP_TABLE})").fetchall()]
            col_list = ", ".join(columns)
            conn.execute(
                f"INSERT INTO {TEMP_TABLE} ({col_list}) SELECT {col_list} FROM {TABLE}"
            )

            conn.execute(f"DROP TABLE {TABLE}")
            conn.execute(f"ALTER TABLE {TEMP_TABLE} RENAME TO {TABLE}")
            for index_sql in indexes:
                conn.execute(index_sql)

            violations = conn.execute("PRAGMA foreign_key_check").fetchall()
            if violations:
                raise MigrationError(f"foreign-key check failed after rebuild: {violations!r}")

            after = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
            if after != before:
                raise MigrationError(f"row count changed during rebuild: {before} before, {after} after")

            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.execute("PRAGMA foreign_keys = ON")
        return before, after
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Add the station_configured location status to an Audtheia database.")
    parser.add_argument("--db", help="path to the database file (defaults to the configured path)")
    parser.add_argument("--schema", help="path to schema.sql (defaults to the configured path)")
    parser.add_argument("--check", action="store_true", help="report whether the migration is needed and change nothing")
    args = parser.parse_args(argv)

    repo_root = _repo_root()
    default_db, default_schema = _resolve_from_settings(repo_root)
    db_path = Path(args.db).resolve() if args.db else default_db
    schema_path = Path(args.schema).resolve() if args.schema else default_schema

    if not db_path.exists():
        print(f"No database found at {db_path}. Nothing to migrate.")
        return 0
    if not schema_path.exists():
        print(f"Schema file not found at {schema_path}.", file=sys.stderr)
        return 2

    try:
        create_temp, indexes = _extract_observations_ddl(schema_path.read_text(encoding="utf-8"))
    except MigrationError as exc:
        print(f"Could not read the schema: {exc}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(str(db_path))
    try:
        try:
            migrated = _already_migrated(conn)
        except MigrationError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    finally:
        conn.close()

    if migrated:
        print(f"Already migrated: {db_path} already allows the '{NEW_STATUS}' location status. No change made.")
        return 0

    if args.check:
        print(f"Migration needed: {db_path} does not yet allow the '{NEW_STATUS}' location status.")
        return 0

    print(f"Migrating {db_path}")
    backup_path = _backup(db_path)
    print(f"  Backup written to {backup_path}")
    try:
        before, after = _migrate(db_path, create_temp, indexes)
    except (MigrationError, sqlite3.Error) as exc:
        print(f"  Migration failed and was rolled back: {exc}", file=sys.stderr)
        print(f"  The database is unchanged. A backup is available at {backup_path}.", file=sys.stderr)
        return 1

    print(f"  Rebuilt the observations table: {before} rows preserved.")
    print(f"  Done. The '{NEW_STATUS}' location status is now allowed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
