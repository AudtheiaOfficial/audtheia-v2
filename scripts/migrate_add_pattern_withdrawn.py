"""One-time database migration: let a candidate pattern be withdrawn.

Path: scripts/migrate_add_pattern_withdrawn.py

A candidate pattern from the longitudinal pass is computed from the events it
rests on. When one of those events is deleted, the pattern's count, effect size
and description describe evidence that no longer exists. Deleting the pattern
would lose the record that it was ever proposed; keeping it as a live candidate
would offer a figure the record no longer supports. It is withdrawn instead.

This migration lets the patterns table hold that state:

  - 'withdrawn' is added to the statuses a pattern may have.
  - withdrawn_at records when, and withdrawn_reason records why, for example
    that a supporting event was deleted, with the status the pattern had before.

SQLite cannot widen a CHECK rule in place, so the table is rebuilt: the old
table is set aside, the new one is created exactly as audtheia/storage/schema.sql
defines it, every row is copied across unchanged, and the old table is dropped.
Foreign keys are switched off for the rebuild so the links from
pattern_observations are neither removed nor rewritten, and are checked after.

Safety:

  - Nothing is changed until a timestamped backup copy has been written.
  - The rebuild runs in one transaction. Every pattern and every pattern link
    is counted before and after, and any difference rolls the whole change back.
  - The migration is idempotent. If the columns are already there, it reports
    that and changes nothing, so running it twice is harmless.

Usage (from the repository root):

    python scripts/migrate_add_pattern_withdrawn.py            # migrate the configured database
    python scripts/migrate_add_pattern_withdrawn.py --check    # report status only, change nothing
    python scripts/migrate_add_pattern_withdrawn.py --db path/to/audtheia.db

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

TABLE = "patterns"
NEW_COLUMNS = ("withdrawn_at", "withdrawn_reason")

# Kept in full and identical to audtheia/storage/schema.sql, so a database
# created fresh from the schema and a database brought forward by this migration
# end up with the same table definition.
CREATE_TABLE_SQL = """
CREATE TABLE patterns (
    id                      TEXT PRIMARY KEY,
    dream_pass_id              TEXT NOT NULL REFERENCES dream_passes(id) ON DELETE CASCADE,
    data_source                 TEXT NOT NULL CHECK (data_source = 'dream'),
    status                       TEXT NOT NULL DEFAULT 'candidate' CHECK (status IN ('candidate', 'verified', 'rejected', 'withdrawn')),
    withdrawn_at                 TEXT,            -- UTC ISO8601, set only when status is withdrawn
    withdrawn_reason             TEXT,            -- why it was withdrawn, for example a supporting event was deleted
    dream_phase                  TEXT NOT NULL CHECK (dream_phase IN ('nrem', 'rem')),
    -- The class of hypothesis, so patterns are filterable by kind rather than
    -- only by free-text description. Nullable so a future class can be added
    -- without a migration; the values below are the classes produced today.
    pattern_type                  TEXT CHECK (pattern_type IS NULL OR pattern_type IN
                                    ('temporal_shift', 'co_occurrence', 'envelope_correlation', 'novel_cluster')),
    confidence                    REAL CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    effect_size                    REAL,
    effect_size_type                TEXT CHECK (effect_size_type IS NULL OR effect_size_type IN ('r', 'cohens_d', 'log_odds')),
    -- The named test or method behind effect_size (for example spearman_rho,
    -- mann_kendall), so the statistic that produced a candidate is auditable.
    statistic                        TEXT,
    data_span_start                  TEXT NOT NULL,   -- UTC ISO8601
    data_span_end                     TEXT NOT NULL,   -- UTC ISO8601
    n                                INTEGER NOT NULL,
    p_value                           REAL,            -- nullable: filled by the statistical-validity layer
    q_value                            REAL,            -- nullable: Benjamini-Hochberg FDR-adjusted
    autocorr_adjusted                   INTEGER CHECK (autocorr_adjusted IS NULL OR autocorr_adjusted IN (0, 1)),
    model_version                        TEXT,
    description                           TEXT NOT NULL,
    created_at                             TEXT NOT NULL
) STRICT
"""

CREATE_INDEX_SQL = (
    "CREATE INDEX idx_patterns_dream_pass_id ON patterns(dream_pass_id)",
    "CREATE INDEX idx_patterns_status ON patterns(status)",
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


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]


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


def _counts(conn: sqlite3.Connection) -> tuple[int, int]:
    patterns = conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
    links = conn.execute("SELECT COUNT(*) FROM pattern_observations").fetchone()[0]
    return patterns, links


def migrate(db_path: Path) -> tuple[int, int]:
    """Rebuild the patterns table with the withdrawal columns. Returns (patterns, links)."""
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        # Keep references in other tables pointing at the name "patterns" while
        # the old table is renamed aside.
        conn.execute("PRAGMA legacy_alter_table = ON")
        conn.execute("BEGIN IMMEDIATE")
        try:
            before = _counts(conn)
            old_cols = _columns(conn, TABLE)
            conn.execute("ALTER TABLE patterns RENAME TO patterns_before_withdrawn")
            conn.execute("DROP INDEX IF EXISTS idx_patterns_dream_pass_id")
            conn.execute("DROP INDEX IF EXISTS idx_patterns_status")
            conn.execute(CREATE_TABLE_SQL.strip())
            cols = ", ".join(old_cols)
            conn.execute(f"INSERT INTO patterns ({cols}) SELECT {cols} FROM patterns_before_withdrawn")
            conn.execute("DROP TABLE patterns_before_withdrawn")
            for statement in CREATE_INDEX_SQL:
                conn.execute(statement)
            after = _counts(conn)
            if before != after:
                raise RuntimeError(f"pattern or link count changed: {before} then {after}")
            problems = conn.execute("PRAGMA foreign_key_check").fetchall()
            if problems:
                raise RuntimeError(f"foreign key check failed after the rebuild: {problems[:3]}")
            missing = [c for c in NEW_COLUMNS if c not in _columns(conn, TABLE)]
            if missing:
                raise RuntimeError(f"columns not created: {missing}")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.execute("PRAGMA legacy_alter_table = OFF")
            conn.execute("PRAGMA foreign_keys = ON")
        return after
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Let a candidate pattern be withdrawn in an Audtheia database."
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
        if not _table_exists(conn, TABLE) or not _table_exists(conn, "pattern_observations"):
            print(
                f"Cannot migrate {db_path}: it has no patterns tables."
                " Initialize the database from the schema first.",
                file=sys.stderr,
            )
            return 2
        already = all(c in _columns(conn, TABLE) for c in NEW_COLUMNS)
        patterns_before, links_before = _counts(conn)
    finally:
        conn.close()

    if already:
        print(f"Already migrated: {db_path} can already withdraw patterns. No change made.")
        return 0
    if args.check:
        print(f"Migration needed: {db_path} cannot withdraw patterns yet.")
        print(f"  {patterns_before} patterns and {links_before} pattern links are present and will be kept unchanged.")
        return 0

    print(f"Migrating {db_path}")
    backup_path = _backup(db_path)
    print(f"  Backup written to {backup_path}")
    try:
        patterns_after, links_after = migrate(db_path)
    except (sqlite3.Error, RuntimeError) as exc:
        print(f"  Migration failed and was rolled back: {exc}", file=sys.stderr)
        print(f"  A backup is available at {backup_path}.", file=sys.stderr)
        return 1

    print(f"  Rebuilt the '{TABLE}' table. {patterns_after} patterns and {links_after} pattern links preserved unchanged.")
    print("  A pattern whose supporting event is deleted is now withdrawn, with when and why, rather than left as a candidate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
