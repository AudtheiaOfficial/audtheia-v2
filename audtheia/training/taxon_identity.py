"""One identity per taxon before anything is learned.

Path: audtheia/training/taxon_identity.py

The same organism can reach the record under several labels. A vision model writes
its own class label, such as "northern-cardinal". BirdNET writes a label that joins
the scientific and common names, such as "Cardinalis cardinalis_Northern Cardinal".
An expert relabel writes the accepted scientific name and its backbone key. Left
alone, these become separate classes, and a model trained on them is taught to tell
one species apart from itself.

This module gives every label one identity, tried in this order:

  1. The backbone key the record already carries. A synonym key is followed to its
     accepted taxon, so historical names collapse onto the current one.
  2. An exact scientific name in the shipped backbone index, accepted only when a
     single accepted taxon carries it.
  3. The scientific part of a BirdNET label, the text before its first underscore,
     matched the same way.
  4. The label map, where an expert has mapped a label once, for labels such as
     model class names that no rule can read.

A label that none of these settles stays its own class and is listed as unresolved,
so a person can map it. An expert can also mark a label to keep as it is, for a
morphotype or an informal name that has no backbone entry, and it is then no longer
listed. Nothing is merged by guessing, and the record is never changed. The map is
a small file beside the datasets, every entry written by a person.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

__all__ = [
    "IdentityError",
    "INDEX_FILE",
    "LABEL_MAP_FILE",
    "METHODS",
    "Resolver",
    "label_key",
    "load_label_map",
    "save_label_entry",
    "remove_label_entry",
    "resolve_classes",
]

INDEX_FILE = "index.db"
LABEL_MAP_FILE = "taxon_label_map.json"
LABEL_MAP_VERSION = 1
ACTIONS = ("map", "keep")

METHODS = {
    "record_key": "the backbone key stored in the record",
    "record_key_synonym": "the backbone key stored in the record, followed from a synonym to the accepted taxon",
    "backbone_name": "an exact scientific name in the backbone",
    "backbone_name_synonym": "an exact scientific name in the backbone, followed from a synonym to the accepted taxon",
    "birdnet_label": "the scientific part of a BirdNET label",
    "label_map": "the label map, mapped by an expert",
    "kept_as_label": "kept as its own label by an expert",
    "unresolved": "not resolved",
}


class IdentityError(ValueError):
    """A label map entry is not valid, with a reason a person can act on."""


def label_key(label) -> str:
    """A label reduced for lookup: trimmed, lower-cased, runs of spaces made single."""
    return " ".join(str(label or "").strip().lower().split())


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# The label map
# ---------------------------------------------------------------------------


def load_label_map(path) -> dict:
    """The label map as a dictionary keyed by label_key, or empty when absent or unreadable."""
    if path is None:
        return {}
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    entries = data.get("entries") if isinstance(data, dict) else None
    if not isinstance(entries, dict):
        return {}
    clean = {}
    for key, entry in entries.items():
        if not isinstance(entry, dict) or entry.get("action") not in ACTIONS:
            continue
        if entry["action"] == "map" and not (str(entry.get("gbif_usage_key") or "").strip()
                                             and str(entry.get("scientific_name") or "").strip()):
            continue
        clean[label_key(key)] = entry
    return clean


def _write_map(path: Path, entries: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps({"version": LABEL_MAP_VERSION, "entries": dict(sorted(entries.items()))},
                      indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=".label_map.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(body)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_label_entry(path, label: str, *, action: str, scientific_name: Optional[str] = None,
                     gbif_usage_key: Optional[str] = None, corrector: Optional[str] = None) -> dict:
    """Record a person's decision for one label and return the entry written.

    A "map" entry needs the accepted scientific name and its backbone key, which the
    caller has already checked against the backbone. A "keep" entry records that the
    label is meant to stay its own class.
    """
    key = label_key(label)
    if not key:
        raise IdentityError("the label is empty")
    if action not in ACTIONS:
        raise IdentityError("the action must be 'map' or 'keep'")
    entry = {"label": str(label).strip(), "action": action, "decided_at": _utc_now(),
             "data_source": "human_expert"}
    if corrector:
        entry["corrector"] = str(corrector).strip()[:120]
    if action == "map":
        name = str(scientific_name or "").strip()
        usage = str(gbif_usage_key or "").strip()
        if not name or not usage:
            raise IdentityError("a mapping needs both the scientific name and its backbone key")
        if not usage.isdigit():
            raise IdentityError("the backbone key must be a number")
        entry["scientific_name"] = name
        entry["gbif_usage_key"] = usage
    path = Path(path)
    entries = load_label_map(path)
    entries[key] = entry
    _write_map(path, entries)
    return entry


def remove_label_entry(path, label: str) -> bool:
    """Remove a person's decision for a label. Returns whether there was one."""
    path = Path(path)
    entries = load_label_map(path)
    if entries.pop(label_key(label), None) is None:
        return False
    _write_map(path, entries)
    return True


# ---------------------------------------------------------------------------
# Resolving
# ---------------------------------------------------------------------------


class Resolver:
    """Resolves labels against the shipped backbone index and the label map.

    The index is opened read-only on first use. When it is missing, only the keys
    the record already carries and the label map are used, and the report says so.
    """

    def __init__(self, backbone_dir=None, label_map: Optional[dict] = None):
        self.index_path = Path(backbone_dir) / INDEX_FILE if backbone_dir else None
        self.label_map = dict(label_map or {})
        self._conn = None
        self._opened = False

    # -- the index -----------------------------------------------------------

    @property
    def index_available(self) -> bool:
        return self._connection() is not None

    def _connection(self):
        if not self._opened:
            self._opened = True
            if self.index_path is not None and self.index_path.is_file():
                try:
                    self._conn = sqlite3.connect(self.index_path.resolve().as_uri() + "?mode=ro", uri=True)
                    self._conn.execute("SELECT 1 FROM taxon_index LIMIT 1").fetchall()
                except sqlite3.Error:
                    self._conn = None
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
        self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _by_key(self, usage_key: str) -> Optional[dict]:
        conn = self._connection()
        if conn is None:
            return None
        row = conn.execute("SELECT usage_key, canonical_name, status, accepted_key FROM taxon_index"
                           " WHERE usage_key = ?", (usage_key,)).fetchone()
        if row is None:
            return None
        accepted_key = row[3] or row[0]
        name = row[1]
        if accepted_key != row[0]:
            acc = conn.execute("SELECT canonical_name FROM taxon_index WHERE usage_key = ?",
                               (accepted_key,)).fetchone()
            if acc is None:
                return None
            name = acc[0]
        return {"key": accepted_key, "name": name, "synonym": accepted_key != row[0]}

    def _by_name(self, name: str) -> tuple:
        """The accepted taxon for an exact name, and a reason when there is none."""
        conn = self._connection()
        if conn is None:
            return None, "the taxonomic index is not available"
        wanted = label_key(name)
        if not wanted:
            return None, "the label is empty"
        rows = conn.execute("SELECT DISTINCT accepted_key FROM taxon_index WHERE name_lower = ?",
                            (wanted,)).fetchall()
        if not rows:
            return None, "no scientific name in the backbone matches it exactly"
        if len(rows) > 1:
            return None, "the name is shared by more than one accepted taxon, so it is ambiguous"
        found = self._by_key(rows[0][0])
        if found is None:
            return None, "the accepted taxon could not be read from the index"
        own = conn.execute("SELECT 1 FROM taxon_index WHERE name_lower = ? AND usage_key = ?",
                           (wanted, rows[0][0])).fetchone()
        found["synonym"] = own is None
        return found, None

    # -- one label -----------------------------------------------------------

    def resolve(self, class_key: str, class_name: str, gbif_usage_key=None) -> dict:
        """The identity for one class as the record names it.

        Returns a dictionary with the class key, class name and backbone key the class
        should carry, the method that settled it, and for an unresolved label the reason.
        """
        label = str(class_name or "").strip() or class_key
        usage = str(gbif_usage_key or "").strip()
        if usage:
            found = self._by_key(usage)
            if found is not None:
                return self._resolved(found, "record_key_synonym" if found["synonym"] else "record_key", label)
            # A key the index does not hold, such as a genus from a roll-up, is kept as it is.
            return {"class_key": f"gbif:{usage}", "class_name": label, "gbif_usage_key": usage,
                    "method": "record_key", "label": label}

        reasons = []
        found, reason = self._by_name(label)
        if found is not None:
            return self._resolved(found, "backbone_name_synonym" if found["synonym"] else "backbone_name", label)
        reasons.append(reason)

        if "_" in label:
            scientific = label.split("_", 1)[0].strip()
            if scientific:
                found, reason = self._by_name(scientific)
                if found is not None:
                    return self._resolved(found, "birdnet_label", label)
                reasons.append(f"its scientific part: {reason}")

        entry = self.label_map.get(label_key(label))
        if entry is not None:
            if entry["action"] == "map":
                usage = str(entry["gbif_usage_key"]).strip()
                found = self._by_key(usage)
                if found is not None:
                    return self._resolved(found, "label_map", label)
                return {"class_key": f"gbif:{usage}", "class_name": str(entry["scientific_name"]).strip(),
                        "gbif_usage_key": usage, "method": "label_map", "label": label}
            return {"class_key": class_key, "class_name": label, "gbif_usage_key": None,
                    "method": "kept_as_label", "label": label}

        return {"class_key": class_key, "class_name": label, "gbif_usage_key": None,
                "method": "unresolved", "label": label, "reason": "; ".join(r for r in reasons if r)}

    @staticmethod
    def _resolved(found: dict, method: str, label: str) -> dict:
        return {"class_key": f"gbif:{found['key']}", "class_name": found["name"],
                "gbif_usage_key": str(found["key"]), "method": method, "label": label}


def resolve_classes(rows: Iterable[dict], resolver: Resolver) -> dict:
    """Resolve every class and describe what changed.

    rows are dictionaries with class_key, class_name and gbif_usage_key, one per class
    as the record names it. Returns the mapping from each class key to the identity it
    should carry, only for keys whose identity changes, and a report with the merged
    classes, the unresolved labels and the labels kept by an expert.
    """
    resolved: dict = {}
    original: dict = {}
    for row in rows:
        key = row["class_key"]
        if key in resolved or key == "background":
            continue
        resolved[key] = resolver.resolve(key, row.get("class_name"), row.get("gbif_usage_key"))
        original[key] = (row.get("class_name"), str(row.get("gbif_usage_key") or "").strip() or None)

    mapping = {}
    groups: dict = {}
    for key, ident in resolved.items():
        target = {"class_key": ident["class_key"], "class_name": ident["class_name"],
                  "gbif_usage_key": ident["gbif_usage_key"]}
        if (ident["class_key"], ident["class_name"], ident["gbif_usage_key"]) != (key, *original[key]):
            mapping[key] = target
        groups.setdefault(ident["class_key"], []).append({"class_key": key, "label": ident["label"],
                                                          "method": ident["method"]})

    classes = []
    for new_key, members in sorted(groups.items()):
        first = resolved[members[0]["class_key"]]
        classes.append({
            "class_key": new_key,
            "class_name": first["class_name"],
            "gbif_usage_key": first["gbif_usage_key"],
            "labels": sorted(members, key=lambda m: (m["label"].lower(), m["class_key"])),
            "merged": len(members) > 1,
        })
    unresolved = sorted(
        ({"class_key": k, "label": v["label"], "reason": v.get("reason") or ""}
         for k, v in resolved.items() if v["method"] == "unresolved"),
        key=lambda r: r["label"].lower())
    kept = sorted(
        ({"class_key": k, "label": v["label"]} for k, v in resolved.items() if v["method"] == "kept_as_label"),
        key=lambda r: r["label"].lower())
    return {
        "mapping": mapping,
        "report": {
            "index_available": resolver.index_available,
            "classes": classes,
            "merged": [c for c in classes if c["merged"]],
            "unresolved": unresolved,
            "kept_as_label": kept,
            "methods": dict(METHODS),
        },
    }
