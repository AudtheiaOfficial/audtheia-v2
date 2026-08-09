"""Per-species model accuracy and Model trust.

Path: audtheia/analysis/model_trust.py

Two derived, inferred quantities that answer a question salience and the
longitudinal pass do not: how good is a given model, per species, judged against
expert review, and how much should a single detection be believed.

Both are inference, not measurement. They are computed from measured model
confidences and human expert verdicts, but they are themselves models of
reliability. Nothing here is ever written into the measured record: it does not
change a screening confidence, a salience value, or a longitudinal-pass result.
Every value produced carries the model version it belongs to, because a different
model yields different numbers and a trust figure with no model attached is
meaningless.

The mathematics, settled with the platform's author:

  Per-species accuracy of a model, from expert review. For a model version M and
  a species s, over every expert-reviewed detection where M predicted s, let c be
  the confirms, r the relabels (the expert changed it to another species) and x
  the rejects (no organism). The reviewed total is n = c + r + x.

      Acc(s, M) = (c + 1) / (n + 2)

  This is a Laplace-smoothed precision in [0, 1]. The +1 / +2 is a weak uniform
  Beta prior, so a single review does not read as 0 or 100 percent. When n = 0 the
  accuracy is NOT computed: the caller is told there are no expert reviews yet,
  never a false zero. A Wilson lower bound of c / n is offered as an optional,
  conservative secondary figure, never the primary number.

  Model trust, per event. For an event that model M labelled species s, with
  visual confidence C and acoustic confidence A (0 when that channel did not
  fire):

      D  = 1 - (1 - C)(1 - A)          the detection evidence salience already uses
      ET = D * Acc(s, M)               in [0, 1]

  How strongly it was detected, times how reliably this model gets this species
  right. D is imported from salience so it is the exact same quantity, and this
  module never alters it. When Acc(s, M) is not computable, ET is not computable
  either.

  Per-model rollups.

      micro (event-weighted)   = sum(c) / sum(n)          overall correctness
      macro (species-averaged) = mean over s of Acc(s, M) fair across species

  Micro weights every reviewed detection equally; macro weights every species
  equally and so exposes a model that is weak on a rare species it seldom
  predicts. Both are reported.

  From the relabel targets (what experts corrected s to) a small confusion view is
  built: which species M is confused with when it calls s.

Every function here is pure: plain rows in, tagged values out, no database calls
and no I/O. The aggregation that reads the record and applies the modality-to-
model and single-taxon event-level rules lives in the storage layer and feeds
this module the resolved review records it operates on.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

# D is imported, not re-derived, so Model trust shares salience's exact detection
# evidence and this module can never drift from it.
from audtheia.pipeline.salience import detection_evidence

# The weak uniform Beta prior behind the smoothed precision: one pseudo-confirm
# and one pseudo-reject, so a lone review is tempered toward one half rather than
# read as a certainty.
PRIOR_CONFIRMS = 1.0
PRIOR_TOTAL = 2.0

# The provenance tag every value here carries. It is inference, keyed to a model,
# and must never be mistaken for a measured or human-sourced fact.
PROVENANCE = "inferred"

# The verdict vocabulary this module counts, matching the observation_corrections
# CHECK constraint. A confirm is a true positive of the model's call; a relabel is
# the model naming the wrong species; a reject is the model calling an organism
# where there is none.
_VERDICTS = ("confirm", "relabel", "reject")


def laplace_accuracy(confirms: int, reviewed: int) -> Optional[float]:
    """Return the Laplace-smoothed per-species precision, or None when n = 0.

    ``confirms`` is c and ``reviewed`` is n = c + r + x. With no reviews the
    precision is not computable and this returns None, which the caller surfaces
    as "no expert reviews yet" rather than a false zero.
    """
    n = int(reviewed)
    if n <= 0:
        return None
    c = int(confirms)
    return (c + PRIOR_CONFIRMS) / (n + PRIOR_TOTAL)


def wilson_lower_bound(confirms: int, reviewed: int, z: float = 1.96) -> Optional[float]:
    """Return the Wilson score lower bound of c / n, or None when n = 0.

    A conservative secondary figure: the lower end of a 95 percent confidence
    interval on the raw precision c / n at the default z. It is offered beside the
    primary smoothed precision, never in place of it, because it answers a
    different, more cautious question: how bad could this model plausibly be on
    this species given how little it has been reviewed. Not computable at n = 0.
    """
    n = int(reviewed)
    if n <= 0:
        return None
    c = int(confirms)
    p = c / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = p + z2 / (2 * n)
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4 * n)) / n)
    lower = (centre - margin) / denom
    # Numerically clamp into [0, 1]; the formula can stray a hair outside it.
    return 0.0 if lower < 0.0 else 1.0 if lower > 1.0 else lower


def event_trust(c_eff: float, a_eff: float, accuracy: Optional[float]) -> Optional[float]:
    """Return Model trust ``D * Acc(s, M)`` in [0, 1], or None when not computable.

    ``c_eff`` and ``a_eff`` are the visual and acoustic detection confidences, each
    0 when that modality did not fire. ``accuracy`` is Acc(s, M) for the species
    this event was labelled and the model that labelled it. When ``accuracy`` is
    None (no expert reviews for that species under that model) Model trust is not
    computable and this returns None, never a fabricated number.
    """
    if accuracy is None:
        return None
    d = detection_evidence(c_eff, a_eff)
    et = d * float(accuracy)
    return 0.0 if et < 0.0 else 1.0 if et > 1.0 else et


def _species_row(key, label, model_version) -> dict:
    """A fresh per-species accumulator keyed to one model version."""
    return {
        "model_version": model_version,
        "species_key": key,
        "species_label": label,
        "confirms": 0,
        "relabels": 0,
        "rejects": 0,
        "reviewed": 0,
        # what M was corrected to when it called s, so the confusion is legible
        "confused_with": {},
    }


def accuracy_table(records: Iterable[dict]) -> list[dict]:
    """Aggregate resolved review records into a per-species accuracy table.

    Each record is one expert-reviewed detection, already resolved by the storage
    layer to the species the model predicted and the model version that predicted
    it. A record carries:

        model_version   the model that produced the call (None when a row never
                        recorded one, surfaced as an explicit unknown, not hidden)
        species_key     the stable species identity the model predicted
        species_label   the human-readable species name for display
        verdict         'confirm', 'relabel' or 'reject'
        corrected_label optional; the species an expert relabelled it to

    Records are grouped by (model_version, species_key). Each returned row carries
    c, r, x, n, the smoothed accuracy, the optional Wilson lower bound, the
    confusion counts, the model version tag, and the inference provenance tag. Rows
    are sorted by accuracy ascending so the fine-tuning targets sit at the top;
    ties break by fewer reviews first, then by label, so the order is stable.

    A record with no species identity contributes to nothing, because an accuracy
    per species needs a species. A record whose verdict is outside the vocabulary
    is ignored rather than trusted.
    """
    groups: dict = {}
    for rec in records:
        verdict = rec.get("verdict")
        if verdict not in _VERDICTS:
            continue
        key = rec.get("species_key")
        label = rec.get("species_label") or key
        if key in (None, ""):
            continue
        model_version = rec.get("model_version")
        group_key = (model_version, key)
        row = groups.get(group_key)
        if row is None:
            row = _species_row(key, label, model_version)
            groups[group_key] = row
        row["reviewed"] += 1
        if verdict == "confirm":
            row["confirms"] += 1
        elif verdict == "relabel":
            row["relabels"] += 1
            target = rec.get("corrected_label") or rec.get("corrected_key")
            if target:
                row["confused_with"][target] = row["confused_with"].get(target, 0) + 1
        elif verdict == "reject":
            row["rejects"] += 1

    table = []
    for row in groups.values():
        c = row["confirms"]
        n = row["reviewed"]
        table.append({
            "provenance": PROVENANCE,
            "model_version": row["model_version"],
            "species_key": row["species_key"],
            "species_label": row["species_label"],
            "confirms": c,
            "relabels": row["relabels"],
            "rejects": row["rejects"],
            "reviewed": n,
            "accuracy": laplace_accuracy(c, n),
            "wilson_lower": wilson_lower_bound(c, n),
            "confused_with": dict(row["confused_with"]),
        })

    # Accuracy is never None here (every row has n >= 1), but guard anyway so the
    # sort key is total. Low accuracy first, then fewer reviews, then label.
    table.sort(key=lambda r: (
        r["accuracy"] if r["accuracy"] is not None else 2.0,
        r["reviewed"],
        r["species_label"] or "",
    ))
    return table


def model_rollups(table: Iterable[dict]) -> dict:
    """Per-model micro and macro accuracy over an accuracy table.

    For each model version present in the table:

        micro = sum(c) / sum(n)              event-weighted overall correctness
        macro = mean over species of Acc     species-averaged, fair across species

    Returns a mapping of model version to a tagged rollup carrying micro, macro,
    the number of species and reviews behind them, and the inference provenance.
    A model with no reviewed species yields None for both, described plainly rather
    than as zero. The unknown-model bucket (a model version that was never
    recorded) is keyed under the empty string so it is visible and separable, never
    silently merged with a real model.
    """
    per_model: dict = {}
    for row in table:
        mv = row.get("model_version")
        bucket_key = mv if mv not in (None, "") else ""
        bucket = per_model.setdefault(bucket_key, {
            "model_version": mv,
            "sum_confirms": 0,
            "sum_reviewed": 0,
            "accuracies": [],
            "species": 0,
        })
        bucket["sum_confirms"] += int(row.get("confirms") or 0)
        bucket["sum_reviewed"] += int(row.get("reviewed") or 0)
        acc = row.get("accuracy")
        if acc is not None:
            bucket["accuracies"].append(float(acc))
        bucket["species"] += 1

    out: dict = {}
    for bucket_key, bucket in per_model.items():
        n = bucket["sum_reviewed"]
        accs = bucket["accuracies"]
        out[bucket_key] = {
            "provenance": PROVENANCE,
            "model_version": bucket["model_version"],
            "micro": (bucket["sum_confirms"] / n) if n > 0 else None,
            "macro": (sum(accs) / len(accs)) if accs else None,
            "species": bucket["species"],
            "reviewed": n,
        }
    return out


def accuracy_index(table: Iterable[dict]) -> dict:
    """Map (model_version, species_key) to its accuracy, for Model trust lookup.

    Model trust for a detection needs Acc(s, M) for that detection's species and
    model. This turns the accuracy table into that lookup. A pair absent from the
    index has no expert reviews, so its Model trust is not computable.
    """
    index: dict = {}
    for row in table:
        index[(row.get("model_version"), row.get("species_key"))] = row.get("accuracy")
    return index


def latest_verdicts(corrections: Iterable[dict]) -> dict:
    """The current expert verdict per target from an event's correction history.

    ``corrections`` is one event's correction rows, newest first, exactly as the
    storage layer returns them. Because a change of mind is a new row and the
    newest row for a target is the one that stands, this keeps only the first row
    seen for each target. The target key is the detection_id, with None meaning the
    verdict applies to the whole event.
    """
    latest: dict = {}
    for row in corrections:
        target = row.get("detection_id")
        if target not in latest:
            latest[target] = row
    return latest


def event_review_records(
    child_detections: Iterable[dict],
    corrections: Iterable[dict],
    *,
    screening_model_version: Optional[str],
    acoustic_model_version: Optional[str],
) -> list[dict]:
    """Resolve one event's reviewed detections into accuracy records.

    Applies the two settled mapping rules, so the pure math above never has to know
    them:

      Modality to model. A vision detection is keyed to the screening model
      version and an audio detection to the acoustic model version, because those
      are the models that actually produced each call.

      Event-level verdicts. A correction naming a specific detection judges that
      detection. A correction on the whole event (detection_id is None) names no
      species, so it is applied only when the event has exactly one child
      detection, where the target is unambiguous. On a multi-taxon event an
      event-level verdict is left uncounted rather than guessed onto a species,
      because a fabricated attribution is worse than a missing one. A
      detection-level verdict always takes precedence over an event-level one for
      the same detection.

    A detection with no species identity is skipped, since accuracy is per species.
    Returns the list of records ``accuracy_table`` consumes; an unreviewed
    detection contributes nothing.
    """
    detections = list(child_detections)
    latest = latest_verdicts(corrections)
    event_level = latest.get(None)
    single_taxon = len(detections) == 1

    records: list[dict] = []
    for det in detections:
        # Identify a taxon the same way the rest of the platform does (see
        # Database.taxon_event_counts): the backbone key when it resolved, then
        # the scientific name, then the model's own class label. A detection whose
        # only identity is a class label that never matched the backbone still
        # names a taxon and must be counted, or a whole class of reviewed
        # detections would silently fall out of the accuracy.
        key = det.get("gbif_usage_key") or det.get("scientific_name") or det.get("common_name")
        if key in (None, ""):
            continue
        label = det.get("scientific_name") or det.get("common_name") or key
        modality = det.get("modality") or "vision"
        model_version = (
            acoustic_model_version if modality == "audio" else screening_model_version
        )

        # Detection-level verdict wins; fall back to an event-level verdict only on
        # a single-taxon event.
        correction = latest.get(det.get("id"))
        if correction is None and single_taxon:
            correction = event_level
        if correction is None:
            continue

        verdict = correction.get("verdict")
        if verdict not in _VERDICTS:
            continue

        records.append({
            "model_version": model_version,
            "species_key": key,
            "species_label": label,
            "modality": modality,
            "verdict": verdict,
            "corrected_label": correction.get("corrected_scientific_name")
            or correction.get("corrected_common_name"),
            "corrected_key": correction.get("corrected_gbif_usage_key"),
        })
    return records
