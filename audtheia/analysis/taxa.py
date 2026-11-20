"""One identity per species wherever a taxon is counted or shown.

Path: audtheia/analysis/taxa.py

The record keeps every label exactly as the model or the person wrote it. A camera
model writes its own class name, such as "northern-cardinal"; BirdNET writes the
scientific and common names joined, such as "Cardinalis cardinalis_Northern
Cardinal"; a backbone match writes a usage key. Counted as written, one species
becomes two or three taxa, and richness, events per taxon, species profiles and
model accuracy all overstate what was seen.

This module reads those labels through the same identity rules the Train tab uses
to build datasets (audtheia/training/taxon_identity.py): the record's backbone key
followed through synonyms, an exact scientific name, the scientific part of a
BirdNET label, then the expert label map. It changes nothing in the record. Each
label is resolved at display or report time, the raw label is kept beside the
identity as provenance, and a label no rule settles stays its own entry and is
marked unresolved, so nothing is merged by guessing. A mapping an expert makes in
the Brain tab or the Train tab is the same entry in the same file, so it applies
in both places and in every report from then on, without touching a model.

Common names are only shown when the record itself supplies one: the common part
of a BirdNET label, a record that carries both names, or the cached species
reference. A model's class name is never rewritten into a common name.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Iterable, Optional

logger = logging.getLogger(__name__)

__all__ = ["LabelIdentities", "TaxonBook", "raw_label", "label_common_name", "identity_function",
           "trust_snapshot_by_identity"]

# Methods that settle a label onto a backbone taxon.
_RESOLVED = {"record_key", "record_key_synonym", "backbone_name", "backbone_name_synonym",
             "birdnet_label", "label_map"}


def raw_label(detection: dict) -> Optional[str]:
    """The label a detection carries, as written: the scientific name, else the common name."""
    for field in ("scientific_name", "common_name"):
        value = str(detection.get(field) or "").strip()
        if value:
            return value
    return None


def label_common_name(label: Optional[str]) -> Optional[str]:
    """The common-name part of a BirdNET label ("Genus species_Common Name"), or None."""
    text = str(label or "").strip()
    if "_" not in text:
        return None
    common = text.split("_", 1)[1].strip()
    return common or None


def _named_labels(db) -> list:
    """Every distinct label in the record that carries a common name after an underscore."""
    try:
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT common_name FROM child_detections WHERE common_name LIKE '%\\_%' ESCAPE '\\'"
            ).fetchall()
        return [r[0] for r in rows if r[0]]
    except Exception:  # noqa: BLE001 - names are a display aid and never block a view
        return []


class TaxonBook:
    """Resolves detections to one identity per taxon, read-only, for counting and display.

    Build one per request or report with for_settings, call identify on every
    detection, then display or common_name on the identities. Common names are
    gathered as detections are identified, so identify everything in scope before
    rendering names.
    """

    def __init__(self, resolver=None, species_reference: Iterable[dict] = ()):
        self._resolver = resolver
        self._cache: dict = {}
        self._common: dict = {}
        self._labels: dict = {}
        self._reference_common: dict = {}
        for sp in species_reference or ():
            key = str(sp.get("gbif_usage_key") or "").strip()
            common = str(sp.get("common_name") or "").strip()
            if key and common:
                self._reference_common[key] = common

    # -- construction ----------------------------------------------------------

    @classmethod
    def for_settings(cls, settings, db=None) -> "TaxonBook":
        """A book over the shipped backbone index, the expert label map and the species cache.

        Every source is optional. With no index, only backbone keys already on a
        record and the label map resolve anything; everything else is shown as its
        own unresolved label, never guessed.
        """
        resolver = None
        try:
            from audtheia.training import dataset as dataset_rules

            resolver = dataset_rules.default_identity_resolver(settings)
        except Exception as exc:  # noqa: BLE001 - identity is a display aid and never blocks a view
            logger.warning("taxon identity is unavailable (%s); labels are shown as written", exc)
        reference: list = []
        if db is not None:
            try:
                reference = db.list_species_reference()
            except Exception:  # noqa: BLE001 - a database without the species cache simply has none
                reference = []
        book = cls(resolver, reference)
        if db is not None:
            book.prime_common_names(_named_labels(db))
        return book

    def prime_common_names(self, labels: Iterable[str]) -> None:
        """Learn common names from labels that carry one, such as BirdNET "Sci_Common".

        Called once with every such label in the record, so a species is shown
        with the same common name on every page and in every report, whichever
        model's labels a page happens to hold. Only the names are learned: the
        labels are not added to any identity's list of labels as written.
        """
        from audtheia.training.candidates import taxon_key

        for label in labels or ():
            common = label_common_name(label)
            key = taxon_key(None, label)
            if not common or key is None:
                continue
            ident = self._cache.get((None, label))
            if ident is None:
                ident = self._resolve(key, label, None)
                self._cache[(None, label)] = ident
            if ident["resolved"]:
                self._common.setdefault(ident["key"], common)

    def close(self) -> None:
        if self._resolver is not None:
            try:
                self._resolver.close()
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- resolution ------------------------------------------------------------

    def identify(self, detection: dict) -> Optional[dict]:
        """The identity of one detection, or None when it names no taxon.

        Returns a dictionary with key (the identity key used for counting),
        scientific_name (the accepted name, None when unresolved), gbif_usage_key,
        label (the raw label as written), method and the flags resolved,
        unresolved and kept.
        """
        from audtheia.training.candidates import taxon_key

        label = raw_label(detection)
        usage = str(detection.get("gbif_usage_key") or "").strip() or None
        key = taxon_key(usage, label)
        if key is None:
            return None
        cache_key = (usage, label)
        ident = self._cache.get(cache_key)
        if ident is None:
            ident = self._resolve(key, label, usage)
            self._cache[cache_key] = ident
        common = label_common_name(label)
        if common is None:
            sci = str(detection.get("scientific_name") or "").strip()
            cn = str(detection.get("common_name") or "").strip()
            if sci and cn and cn.lower() != sci.lower():
                common = cn
        if common and ident["resolved"]:
            self._common.setdefault(ident["key"], common)
        if ident.get("label"):
            self._labels.setdefault(ident["key"], set()).add(ident["label"])
        return ident

    def identify_label(self, label: Optional[str], gbif_usage_key=None) -> Optional[dict]:
        """identify for a bare label, such as a verifier's class name."""
        return self.identify({"scientific_name": None, "common_name": label, "gbif_usage_key": gbif_usage_key})

    def _resolve(self, key: str, label: Optional[str], usage: Optional[str]) -> dict:
        if self._resolver is not None:
            try:
                got = self._resolver.resolve(key, label, usage)
            except Exception as exc:  # noqa: BLE001 - an unreadable index leaves the label as written
                logger.warning("could not resolve %r (%s); shown as written", label, exc)
                got = None
        else:
            got = None
        if got is None:
            method = "record_key" if usage else "unresolved"
            got = {"class_key": key, "class_name": label or usage, "gbif_usage_key": usage, "method": method,
                   "label": label or usage}
        method = got.get("method") or "unresolved"
        resolved = method in _RESOLVED
        return {
            "key": got["class_key"],
            "scientific_name": got.get("class_name") if resolved else None,
            "gbif_usage_key": got.get("gbif_usage_key"),
            "label": label or got.get("label"),
            "method": method,
            "resolved": resolved,
            "unresolved": method == "unresolved",
            "kept": method == "kept_as_label",
            "reason": got.get("reason"),
        }

    # -- names -----------------------------------------------------------------

    def common_name(self, ident: Optional[dict]) -> Optional[str]:
        if not ident or not ident.get("resolved"):
            return None
        found = self._common.get(ident["key"])
        if found:
            return found
        usage = str(ident.get("gbif_usage_key") or "")
        return self._reference_common.get(usage)

    def display(self, ident: Optional[dict]) -> str:
        """"Accepted name (Common Name)" for a resolved identity, the raw label otherwise."""
        if not ident:
            return "unidentified"
        if ident.get("resolved") and ident.get("scientific_name"):
            common = self.common_name(ident)
            name = ident["scientific_name"]
            return f"{name} ({common})" if common and common.lower() != name.lower() else name
        return ident.get("label") or ident.get("key") or "unidentified"

    def labels(self, ident: Optional[dict]) -> list:
        """Every raw label seen so far that resolves to this identity, sorted."""
        if not ident:
            return []
        return sorted(self._labels.get(ident["key"], set()), key=str.lower)

    def describe(self, ident: Optional[dict]) -> Optional[dict]:
        """The identity with its display name and every label behind it, for an API payload."""
        if not ident:
            return None
        out = dict(ident)
        out["display"] = self.display(ident)
        out["common_name"] = self.common_name(ident)
        out["labels"] = self.labels(ident)
        return out


class LabelIdentities:
    """The identity keys a verifier's labels resolve to, worked out only when asked.

    A verifier may carry thousands of names. Resolving them is needed only to say
    that a field call's taxon lies outside the labels, so it is done on the first
    membership test and kept.
    """

    def __init__(self, labels, book: "TaxonBook"):
        self._labels = list(labels or ())
        self._book = book
        self._keys: Optional[set] = None

    def __contains__(self, key) -> bool:
        if self._keys is None:
            self._keys = {(self._book.identify_label(label) or {}).get("key") for label in self._labels} - {None}
        return key in self._keys


def identity_function(settings) -> Callable:
    """A callable (gbif_usage_key, name) -> identity key, or None, safe across threads.

    Each thread keeps its own book, because a database connection belongs to the
    thread that opened it. Used where a computation, rather than a view, needs to
    know that two labels are one species: comparing a verifier with the field call,
    and counting a taxon's local rarity.
    """
    local = threading.local()

    def identify(gbif_usage_key, name) -> Optional[str]:
        book = getattr(local, "book", None)
        if book is None:
            book = TaxonBook.for_settings(settings)
            local.book = book
        ident = book.identify({"gbif_usage_key": gbif_usage_key, "common_name": name})
        return ident["key"] if ident else None

    return identify


def _det_for_key(key, label) -> dict:
    """A detection-shaped dictionary for a stored species key and its label."""
    key = None if key in (None, "") else str(key)
    usage = key if key is not None and key.isdigit() else None
    return {"gbif_usage_key": usage, "common_name": label if label else (None if usage else key)}


def trust_snapshot_by_identity(db, book: "TaxonBook") -> dict:
    """The per-species model-accuracy snapshot with one identity per species.

    The stored record keys accuracy by each label as written, so one species named
    two ways reads as two species, and an expert who relabels a model's class name
    to the same species's scientific name is counted as correcting an error. Here
    every reviewed detection is resolved to its identity first. A relabel to the
    same species is a confirmation of the species, because the model named it
    correctly and only the spelling changed. The raw labels behind each row are
    kept as provenance. Nothing is written.
    """
    from audtheia.analysis.model_trust import accuracy_table, model_rollups

    snapshot = db.model_trust_snapshot()
    if not snapshot or snapshot.get("empty"):
        return snapshot
    records = [dict(r) for r in db.reviewed_detection_records()]
    idents = []
    for rec in records:
        ident = book.identify(_det_for_key(rec.get("species_key"), rec.get("species_label")))
        target = None
        if rec.get("corrected_key") or rec.get("corrected_label"):
            target = book.identify(_det_for_key(rec.get("corrected_key"), rec.get("corrected_label")))
        idents.append((ident, target))
    for rec, (ident, target) in zip(records, idents):
        if ident is None:
            continue
        rec["species_key"] = ident["key"]
        rec["species_label"] = book.display(ident)
        if rec.get("verdict") == "relabel" and target is not None:
            if target["key"] == ident["key"]:
                rec["verdict"] = "confirm"
                rec["corrected_label"] = None
                rec["corrected_key"] = None
            else:
                rec["corrected_label"] = book.display(target)
    table = accuracy_table(records)
    by_key = {ident["key"]: ident for ident, _ in idents if ident is not None}
    for row in table:
        row["labels"] = book.labels(by_key.get(row["species_key"]))
        row["unresolved"] = bool((by_key.get(row["species_key"]) or {}).get("unresolved"))
    out = dict(snapshot)
    out["species"] = table
    out["models"] = model_rollups(table)
    out["identity"] = ("one identity per species: labels that resolve to one taxon are counted together, and a "
                       "relabel to the same species counts as a confirmation of the species")
    return out
