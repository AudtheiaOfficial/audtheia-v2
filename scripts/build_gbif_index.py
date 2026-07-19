"""Build the offline species lookup index from the GBIF backbone.

Path: scripts/build_gbif_index.py

The shipped backbone, audtheia/resources/gbif_backbone/simple.txt, is a
tab-delimited export of roughly 2.4 GB with no header row. Answering "is this a
real species name?" by scanning it costs about 13 seconds when the name is
present and a full-file scan when it is not. A miss has no early exit, and a miss
is exactly what a typo looks like, so the naive approach makes the software feel
broken precisely when it is working correctly.

This script does that scan once and writes a small SQLite index beside the
backbone. Lookups against the index are immediate, which is what allows the
correction control to search as the user types rather than refusing a name after
a long pause.

What goes in, and why:

  - Species-rank rows only. A correction names a species, so genus and family
    rows would only add noise to a search box.
  - Accepted names and synonyms both. A taxonomist may reasonably type an older
    name, and refusing it would be wrong: the scientifically correct behaviour is
    to accept it and resolve it to the currently accepted taxon, which is what
    GBIF's own name matching does. Every synonym therefore carries the usage key
    and name of the taxon it resolves to.
  - Doubtful rows are excluded. A name GBIF itself marks DOUBTFUL should not be
    offered to a scientist as a confident identification.

Column positions in simple.txt, confirmed against known keys rather than assumed
(1-based): 1 usage key, 2 parent key, 4 is_synonym, 5 status, 6 rank, 19
scientific name with authorship, 20 canonical name. For a synonym row, the parent
key is the accepted taxon: `Aa rostrata` (SYNONYM, key 7649122) has parent
2797604, which is `Myrosmodes rostrata` (ACCEPTED).

The index is a derived artifact. It is rebuilt from the backbone rather than
shipped, it is excluded from version control, and deleting it costs only the time
to build it again.

Usage (from the repository root):

    python scripts/build_gbif_index.py             # build, refusing to overwrite
    python scripts/build_gbif_index.py --check     # report status only, build nothing
    python scripts/build_gbif_index.py --force     # rebuild over an existing index
    python scripts/build_gbif_index.py --backbone PATH --out PATH

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

# 0-based positions into a simple.txt row, confirmed against known keys.
COL_USAGE_KEY = 0
COL_PARENT_KEY = 1
COL_STATUS = 4
COL_RANK = 5
COL_SCIENTIFIC_NAME = 18
COL_CANONICAL_NAME = 19

MIN_COLUMNS = COL_CANONICAL_NAME + 1

WANTED_RANK = "SPECIES"
WANTED_STATUS = ("ACCEPTED", "SYNONYM")

INDEX_FILENAME = "index.db"

SCHEMA = """
CREATE TABLE taxon_index (
    usage_key         TEXT PRIMARY KEY,
    canonical_name    TEXT NOT NULL,     -- "Junco hyemalis"
    scientific_name   TEXT,              -- "Junco hyemalis (Linnaeus, 1758)"
    status            TEXT NOT NULL,     -- ACCEPTED or SYNONYM
    accepted_key      TEXT NOT NULL,     -- own key when accepted; the accepted taxon when a synonym
    accepted_name     TEXT,              -- filled in after the scan by resolving accepted_key
    name_lower        TEXT NOT NULL      -- lowercased canonical name, for case-insensitive search
) STRICT;
"""

# A prefix search ("junco hy...") is the query the correction search box issues on
# every keystroke, and LIKE 'prefix%' on this index is answered without a scan.
INDEXES = (
    "CREATE INDEX idx_taxon_index_name_lower ON taxon_index(name_lower)",
    "CREATE INDEX idx_taxon_index_accepted_key ON taxon_index(accepted_key)",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _configured_backbone(repo_root: Path) -> Path:
    """The backbone directory from the configuration, or the conventional one."""
    settings_file = repo_root / "config" / "settings.json"
    if settings_file.exists():
        try:
            paths = json.loads(settings_file.read_text(encoding="utf-8")).get("paths", {})
            if isinstance(paths, dict) and paths.get("gbif_backbone_path"):
                candidate = Path(paths["gbif_backbone_path"])
                if not candidate.is_absolute():
                    candidate = repo_root / candidate
                return candidate / "simple.txt"
        except (json.JSONDecodeError, OSError):
            pass
    return repo_root / "audtheia" / "resources" / "gbif_backbone" / "simple.txt"


def _human(n: int) -> str:
    return f"{n:,}"


def build(backbone: Path, out_path: Path, *, progress_every: int = 2_000_000) -> int:
    """Stream the backbone once and write the index. Returns the row count."""
    tmp_path = out_path.with_name(out_path.name + ".partial")
    if tmp_path.exists():
        tmp_path.unlink()

    conn = sqlite3.connect(str(tmp_path))
    try:
        # These pragmas apply to building a throwaway file that is verified and
        # renamed only on success, so durability during the build buys nothing.
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA synchronous = OFF")
        conn.executescript(SCHEMA)

        total_bytes = backbone.stat().st_size
        started = time.monotonic()
        scanned = 0
        kept = 0
        malformed = 0
        batch: list[tuple] = []

        read_bytes = 0
        # Read in binary and decode each line explicitly. Calling tell() while
        # iterating a file opened in text mode raises OSError, and counting
        # characters instead of bytes would under-report progress on a file whose
        # authorship fields are not all ASCII, so the byte count is accumulated
        # here directly.
        with backbone.open("rb") as handle:
            for raw_line in handle:
                read_bytes += len(raw_line)
                scanned += 1
                line = raw_line.decode("utf-8", errors="replace")
                row = line.rstrip("\r\n").split("\t")
                if len(row) < MIN_COLUMNS:
                    malformed += 1
                    continue
                if row[COL_RANK] != WANTED_RANK or row[COL_STATUS] not in WANTED_STATUS:
                    continue

                canonical = row[COL_CANONICAL_NAME].strip()
                usage_key = row[COL_USAGE_KEY].strip()
                if not canonical or canonical == "\\N" or not usage_key:
                    continue

                status = row[COL_STATUS]
                if status == "SYNONYM":
                    accepted_key = row[COL_PARENT_KEY].strip()
                    # A synonym with no parent resolves to nothing, so it cannot
                    # be offered as an identification and is dropped rather than
                    # stored pointing at nowhere.
                    if not accepted_key or accepted_key == "\\N":
                        continue
                else:
                    accepted_key = usage_key

                scientific = row[COL_SCIENTIFIC_NAME].strip()
                batch.append(
                    (
                        usage_key,
                        canonical,
                        None if scientific in ("", "\\N") else scientific,
                        status,
                        accepted_key,
                        canonical.lower(),
                    )
                )
                kept += 1

                if len(batch) >= 50_000:
                    conn.executemany(
                        "INSERT OR IGNORE INTO taxon_index"
                        " (usage_key, canonical_name, scientific_name, status, accepted_key, name_lower)"
                        " VALUES (?, ?, ?, ?, ?, ?)",
                        batch,
                    )
                    batch.clear()

                if scanned % progress_every == 0:
                    pct = (read_bytes / total_bytes * 100) if total_bytes else 0
                    print(
                        f"  {pct:5.1f}%  scanned {_human(scanned)} rows, kept {_human(kept)}",
                        flush=True,
                    )

        if batch:
            conn.executemany(
                "INSERT OR IGNORE INTO taxon_index"
                " (usage_key, canonical_name, scientific_name, status, accepted_key, name_lower)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                batch,
            )
            batch.clear()

        # Resolve every synonym to the name of the taxon it points at. Done as a
        # self-join after the scan because a synonym can appear in the file
        # before the accepted taxon it refers to.
        conn.execute(
            "UPDATE taxon_index SET accepted_name = ("
            "  SELECT a.canonical_name FROM taxon_index a WHERE a.usage_key = taxon_index.accepted_key"
            ")"
        )
        # A synonym whose accepted key is not itself a species-rank row in this
        # index cannot be resolved to a name, so it is removed rather than left
        # offering an identification that resolves to nothing.
        dangling = conn.execute(
            "DELETE FROM taxon_index WHERE accepted_name IS NULL"
        ).rowcount
        for statement in INDEXES:
            conn.execute(statement)
        conn.commit()
        conn.execute("VACUUM")
        conn.commit()

        final = conn.execute("SELECT COUNT(*) FROM taxon_index").fetchone()[0]
        accepted = conn.execute(
            "SELECT COUNT(*) FROM taxon_index WHERE status = 'ACCEPTED'"
        ).fetchone()[0]
    finally:
        conn.close()

    elapsed = time.monotonic() - started
    os.replace(str(tmp_path), str(out_path))

    print(f"  Scanned {_human(scanned)} rows in {elapsed / 60:.1f} minutes.")
    if malformed:
        print(f"  Skipped {_human(malformed)} rows with too few columns.")
    if dangling:
        print(f"  Dropped {_human(dangling)} synonyms that resolved to no species-rank taxon.")
    print(
        f"  Indexed {_human(final)} species names"
        f" ({_human(accepted)} accepted, {_human(final - accepted)} synonyms)."
    )
    print(f"  Index written to {out_path} ({out_path.stat().st_size / 1_048_576:.0f} MB).")
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the offline species lookup index from the GBIF backbone."
    )
    parser.add_argument("--backbone", help="path to simple.txt (defaults to the configured location)")
    parser.add_argument("--out", help="path to write the index (defaults to index.db beside the backbone)")
    parser.add_argument("--check", action="store_true", help="report status only and build nothing")
    parser.add_argument("--force", action="store_true", help="rebuild over an existing index")
    args = parser.parse_args(argv)

    repo_root = _repo_root()
    backbone = Path(args.backbone).resolve() if args.backbone else _configured_backbone(repo_root)
    out_path = Path(args.out).resolve() if args.out else backbone.with_name(INDEX_FILENAME)

    if not backbone.exists():
        print(
            f"No GBIF backbone found at {backbone}."
            " The species index cannot be built without it.",
            file=sys.stderr,
        )
        return 2

    if out_path.exists():
        try:
            conn = sqlite3.connect(str(out_path))
            count = conn.execute("SELECT COUNT(*) FROM taxon_index").fetchone()[0]
            conn.close()
            summary = f"{_human(count)} species names"
        except sqlite3.Error:
            summary = "unreadable, so a rebuild is needed"
        if not args.force:
            print(f"An index already exists at {out_path} ({summary}).")
            print("  Nothing was changed. Pass --force to rebuild it.")
            return 0
        print(f"Rebuilding the existing index at {out_path} ({summary}).")
    elif args.check:
        size_gb = backbone.stat().st_size / 1_073_741_824
        print(f"Index needed: no index exists at {out_path}.")
        print(f"  It will be built from {backbone} ({size_gb:.1f} GB), which takes several minutes.")
        return 0

    if args.check:
        return 0

    print(f"Building the species index from {backbone}")
    print("  This reads the whole backbone once and takes several minutes.")
    build(backbone, out_path)
    print("  Species names can now be searched instantly when correcting a detection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
