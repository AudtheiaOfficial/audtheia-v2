"""Rolling a rare species up to its genus or family through the shipped backbone.

Path: audtheia/training/taxon_rollup.py

A species with only a few confirmed events cannot be learned or tested fairly, but
its genus or family often can. This module finds the genus and family of a taxon
using the GBIF backbone that ships with Audtheia, so a rare class can be merged
into its parent instead of being thrown away. Nothing is guessed, and nothing is
changed in the record. The caller decides, class by class, whether to roll up.

Where the parents come from

The compact name index used for searching holds names and synonym links but not
the ranks above a species. The full backbone export that setup also downloads does,
as the family and genus keys on every row. The export is large, so it is read once
as a stream, only the rows needed are kept, and the answers are cached in a small
file beside the datasets. A later request for the same taxa reads the cache and
does not scan again. The cache records the size of the backbone it was built from
and is discarded if that changes.

Backbone column layout, checked against the shipped file

  0 usage key, 1 parent key, 2 basionym key, 3 synonym flag, 4 status, 5 rank,
  10 kingdom key, 11 phylum key, 12 class key, 13 order key, 14 family key,
  15 genus key, 16 species key, 19 canonical name.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Callable, Iterable, Optional

__all__ = [
    "RollupError",
    "LEVELS",
    "rollup_targets",
    "parent_for",
]

LEVELS = ("genus", "family")

COL_KEY = 0
COL_STATUS = 4
COL_RANK = 5
COL_FAMILY = 14
COL_GENUS = 15
COL_NAME = 19
MIN_COLUMNS = 20

BACKBONE_FILE = "simple.txt"
INDEX_FILE = "index.db"
CACHE_VERSION = 1


class RollupError(RuntimeError):
    """The backbone needed for a roll-up is missing or unreadable, with a reason a person can act on."""


def _null(value: str) -> Optional[str]:
    value = value.strip()
    return None if value in ("", "\\N") else value


def _open_index(backbone_dir: Path) -> sqlite3.Connection:
    index = Path(backbone_dir) / INDEX_FILE
    if not index.is_file():
        raise RollupError(
            f"the taxonomic index was not found at {index}. Run the species data setup first."
        )
    return sqlite3.connect(index.resolve().as_uri() + "?mode=ro", uri=True)


def _name_for_key(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT canonical_name FROM taxon_index WHERE usage_key = ?", (key,)).fetchone()
    return row[0] if row else None


def _accepted_key(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT accepted_key FROM taxon_index WHERE usage_key = ?", (key,)).fetchone()
    return row[0] if row else None


def _key_for_name(conn: sqlite3.Connection, name: str) -> Optional[str]:
    """The accepted key for a name, only when exactly one accepted taxon carries it.

    A name shared by several accepted taxa, such as a genus name that is also used in
    another kingdom, is ambiguous, and an ambiguous name is never resolved by guessing.
    """
    rows = conn.execute(
        "SELECT DISTINCT accepted_key FROM taxon_index WHERE name_lower = ?", (name.strip().lower(),)
    ).fetchall()
    return rows[0][0] if len(rows) == 1 else None


def _cache_fingerprint(backbone_dir: Path) -> dict:
    full = Path(backbone_dir) / BACKBONE_FILE
    index = Path(backbone_dir) / INDEX_FILE
    return {
        "version": CACHE_VERSION,
        "backbone_bytes": full.stat().st_size if full.is_file() else None,
        "index_bytes": index.stat().st_size if index.is_file() else None,
    }


def _read_cache(cache_path: Optional[Path], fingerprint: dict) -> dict:
    if cache_path is None or not Path(cache_path).is_file():
        return {}
    try:
        data = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if data.get("fingerprint") != fingerprint:
        return {}
    entries = data.get("entries")
    return entries if isinstance(entries, dict) else {}


def _write_cache(cache_path: Optional[Path], fingerprint: dict, entries: dict) -> None:
    if cache_path is None:
        return
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"fingerprint": fingerprint, "entries": entries}, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8", newline="\n")
    tmp.replace(cache_path)


def _scan(backbone_dir: Path, wanted: set, progress: Optional[Callable[[int], None]]) -> dict:
    """Read the backbone once and keep only the rows for the wanted keys."""
    full = Path(backbone_dir) / BACKBONE_FILE
    if not full.is_file():
        raise RollupError(
            f"the full taxonomic backbone was not found at {full}, so genus and family cannot be looked up."
            " Run the species data setup first."
        )
    remaining = {k.encode("utf-8") for k in wanted}
    found: dict = {}
    with open(full, "rb") as handle:
        for number, raw in enumerate(handle, start=1):
            head = raw.split(b"\t", 1)[0]
            if head in remaining:
                cols = raw.decode("utf-8", errors="replace").rstrip("\r\n").split("\t")
                if len(cols) >= MIN_COLUMNS:
                    found[cols[COL_KEY]] = {
                        "rank": _null(cols[COL_RANK]),
                        "family_key": _null(cols[COL_FAMILY]),
                        "genus_key": _null(cols[COL_GENUS]),
                        "name": _null(cols[COL_NAME]),
                    }
                remaining.discard(head)
                if not remaining:
                    break
            if progress and number % 500000 == 0:
                progress(number)
    return found


def rollup_targets(
    backbone_dir,
    class_keys: Iterable[str],
    *,
    cache_path=None,
    progress: Optional[Callable[[int], None]] = None,
) -> dict:
    """Find the genus and family of each class, keyed by the class key it was asked for.

    A class key is either "gbif:<usage key>" or "name:<lower-cased name>", the same
    identities the candidate rules produce. The result maps each one to a dictionary
    with the accepted key, the rank, and the genus and family each as a dictionary of
    key and name, or None when the backbone does not give one. A class that cannot be
    placed at all maps to a dictionary with a reason. The shipped files are only read.
    """
    backbone_dir = Path(backbone_dir)
    requested = [k for k in dict.fromkeys(class_keys) if k]
    fingerprint = _cache_fingerprint(backbone_dir)
    cached = _read_cache(cache_path, fingerprint)
    result: dict = {}
    todo: list = []
    for key in requested:
        if key in cached:
            result[key] = cached[key]
        else:
            todo.append(key)
    if not todo:
        return result

    conn = _open_index(backbone_dir)
    try:
        resolved_accepted: dict = {}
        for key in todo:
            if key.startswith("gbif:"):
                raw_key = key[5:]
                accepted = _accepted_key(conn, raw_key) or raw_key
            elif key.startswith("name:"):
                accepted = _key_for_name(conn, key[5:])
            else:
                accepted = None
            if accepted is None:
                result[key] = {"reason": "the name could not be matched to a single accepted taxon in the backbone"}
            else:
                resolved_accepted[key] = accepted

        rows = _scan(backbone_dir, set(resolved_accepted.values()), progress) if resolved_accepted else {}

        # The index holds species and their synonyms. A genus or family is named
        # from the full backbone instead, read once more for just those keys.
        missing_names = set()
        for row in rows.values():
            for parent_key in (row["genus_key"], row["family_key"]):
                if parent_key and _name_for_key(conn, parent_key) is None:
                    missing_names.add(parent_key)
        parent_rows = _scan(backbone_dir, missing_names, progress) if missing_names else {}

        def name_of(key):
            found = _name_for_key(conn, key)
            if found is None:
                found = (parent_rows.get(key) or rows.get(key) or {}).get("name")
            return found

        for key, accepted in resolved_accepted.items():
            row = rows.get(accepted)
            if row is None:
                result[key] = {"accepted_key": accepted, "reason": "the taxon was not found in the full backbone"}
                continue
            rank = (row["rank"] or "").upper()
            genus_key = row["genus_key"]
            family_key = row["family_key"]
            entry = {
                "accepted_key": accepted,
                "accepted_name": name_of(accepted) or row.get("name"),
                "rank": rank or None,
                "genus": ({"key": genus_key, "name": name_of(genus_key)} if genus_key else None),
                "family": ({"key": family_key, "name": name_of(family_key)} if family_key else None),
            }
            result[key] = entry
    finally:
        conn.close()

    _write_cache(cache_path, fingerprint, {**cached, **{k: result[k] for k in todo}})
    return result


def parent_for(info: Optional[dict], level: str) -> Optional[dict]:
    """The parent a class would merge into at the requested level, or None.

    Rolling to genus from a species returns its genus. A class that is already a genus
    stays itself at the genus level. Rolling to family from a species or a genus
    returns the family. A taxon above that rank has no parent at either level, and
    nothing is returned, so it is never merged upward by guesswork.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {', '.join(LEVELS)}")
    if not info or info.get("reason"):
        return None
    rank = (info.get("rank") or "").upper()
    own = {"key": info.get("accepted_key"), "name": info.get("accepted_name")}
    if level == "genus":
        if rank == "GENUS":
            return own
        found = info.get("genus")
        return found if found and found.get("key") else None
    if rank == "FAMILY":
        return own
    found = info.get("family")
    return found if found and found.get("key") else None
