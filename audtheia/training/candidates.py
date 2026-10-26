"""Which recorded events can become training examples, and why.

Path: audtheia/training/candidates.py

This module reads nothing from disk and writes nothing. It takes the evidence the
database gathered about each event, applies the rules below, and returns a plain
description of every candidate. The dataset preview, the counts a person sees and
the dataset builder all run through the same functions, so what the interface
shows and what is built cannot drift apart.

The three criteria

  Expert Confirmed. A person has settled what the event is. Either every
  identified animal in it has been confirmed or relabelled, or the event has been
  rejected as a false alarm. The label then comes from the expert, never from the
  model that made the original call.

  QC Passed. The field quality checks accepted the record.

  Verified. The desktop verifier has scored the event and opened its gate.

How an expert's verdict applies, matching the platform's model-trust rules

  A verdict that names one detection judges that detection. A verdict on the whole
  event names no species, so it applies only when the event holds exactly one
  detection. On an event with several detections it is left uncounted, because a
  label guessed onto the wrong species is worse than a missing one. A verdict on a
  detection always takes precedence over a verdict on the whole event.

Taxon identity follows the rest of the platform. The backbone key is used when the
name resolved, then the scientific name, then the model's own class label.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

__all__ = [
    "CRITERIA",
    "COMBINATIONS",
    "LABEL_POSITIVE",
    "LABEL_NEGATIVE",
    "LABEL_PARTIAL",
    "LABEL_MODEL",
    "LABEL_UNRESOLVED",
    "taxon_key",
    "name_key",
    "evaluate_candidate",
    "is_eligible",
    "required_criteria",
    "summarize",
]

CRITERIA = ("expert_confirmed", "qc_passed", "verified")

# The eight on/off combinations of the three criteria, in a stable order, each
# named by the criteria that are on. The summary reports every one of them so a
# person can see what each choice would keep before choosing it.
COMBINATIONS = tuple(
    "+".join(name for name, on in zip(CRITERIA, flags) if on) or "none"
    for flags in (
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (False, False, False),
    )
)

LABEL_POSITIVE = "positive"      # every identified animal confirmed or relabelled by an expert
LABEL_NEGATIVE = "negative"      # every identified animal rejected: a false alarm
LABEL_PARTIAL = "partial"        # some animals judged by an expert, some not
LABEL_MODEL = "model_labelled"   # no expert verdict applies, the label is the model's own call
LABEL_UNRESOLVED = "unresolved"  # nothing here names a taxon, so there is nothing to learn

_QC_PASSING = ("qc_passed", "verified")

STEP_REVIEW = "review"
STEP_VERIFY = "verify"
STEP_QC = "qc"


def taxon_key(gbif_usage_key: Any, name: Any) -> Optional[str]:
    """A stable identity for a taxon, preferring the backbone key.

    The key is namespaced so a backbone key can never collide with a name that
    happens to look like a number. Returns None when there is nothing to identify.
    """
    if gbif_usage_key not in (None, ""):
        return f"gbif:{str(gbif_usage_key).strip()}"
    cleaned = str(name or "").strip().lower()
    return f"name:{cleaned}" if cleaned else None


def name_key(name: Any) -> str:
    """A name reduced for comparison, so a class label and a stored name can match."""
    return " ".join(str(name or "").strip().lower().split())


def _child_identity(child: dict) -> Optional[dict]:
    """What one stored detection says it is, or None when it names nothing."""
    name = (child.get("scientific_name") or child.get("common_name") or "").strip()
    key = taxon_key(child.get("gbif_usage_key"), name)
    if key is None:
        return None
    return {
        "detection_id": child.get("id"),
        "class_key": key,
        "class_name": name or key,
        "gbif_usage_key": str(child["gbif_usage_key"]).strip() if child.get("gbif_usage_key") not in (None, "") else None,
        "match_names": sorted({name_key(child.get("scientific_name")), name_key(child.get("common_name"))} - {""}),
        "confidence": child.get("confidence"),
    }


def _latest_by_target(corrections: Iterable[dict]) -> dict:
    """The newest correction per target, where None stands for the whole event.

    The input is expected newest first, which is how the database returns it, so
    the first row seen for a target is the current one.
    """
    latest: dict = {}
    for row in corrections:
        target = row.get("detection_id")
        if target not in latest:
            latest[target] = row
    return latest


def _resolve_children(children_all: list, modality: str, corrections: Iterable[dict]) -> list:
    """Judge each detection of this modality by the expert verdict that applies to it."""
    latest = _latest_by_target(corrections)
    event_level = latest.get(None)
    single = len(children_all) == 1
    resolved = []
    for child in children_all:
        if (child.get("modality") or "vision") != modality:
            continue
        identity = _child_identity(child)
        if identity is None:
            continue
        correction = latest.get(child.get("id"))
        if correction is None and single:
            correction = event_level
        verdict = correction.get("verdict") if correction else None
        entry = dict(identity)
        entry["verdict"] = verdict if verdict in ("confirm", "relabel", "reject") else None
        entry["source"] = "model"
        if entry["verdict"] == "relabel":
            new_name = (correction.get("corrected_scientific_name") or correction.get("corrected_common_name") or "").strip()
            new_key = taxon_key(correction.get("corrected_gbif_usage_key"), new_name)
            if new_key is None:
                # A relabel that names nothing cannot be learned from. Treat the
                # detection as unjudged rather than inventing a class.
                entry["verdict"] = None
            else:
                entry["original_class_key"] = entry["class_key"]
                entry["class_key"] = new_key
                entry["class_name"] = new_name or new_key
                ucode = correction.get("corrected_gbif_usage_key")
                entry["gbif_usage_key"] = str(ucode).strip() if ucode not in (None, "") else None
                entry["source"] = "expert"
        elif entry["verdict"] == "confirm":
            entry["source"] = "expert"
        elif entry["verdict"] == "reject":
            entry["source"] = "expert"
        resolved.append(entry)
    return resolved


def evaluate_candidate(evidence: dict, modality: str) -> dict:
    """Describe one event as a training candidate.

    The evidence holds the observation row, every stored detection of the event of
    any modality, the newest correction per target, the verification row and the
    counts of frames an expert has judged. Returns a plain dictionary and changes
    nothing it was given.
    """
    obs = evidence.get("observation") or {}
    children_all = list(evidence.get("children") or [])
    verification = evidence.get("verification") or None
    frames = evidence.get("frames") or {}

    taxa = _resolve_children(children_all, modality, evidence.get("corrections") or [])
    positives = [t for t in taxa if t["verdict"] in ("confirm", "relabel")]
    rejected = [t for t in taxa if t["verdict"] == "reject"]
    unjudged = [t for t in taxa if t["verdict"] is None]

    if not taxa:
        label_state = LABEL_UNRESOLVED
    elif positives and not unjudged:
        label_state = LABEL_POSITIVE
    elif rejected and not positives and not unjudged:
        label_state = LABEL_NEGATIVE
    elif positives or rejected:
        label_state = LABEL_PARTIAL
    else:
        label_state = LABEL_MODEL

    expert_confirmed = label_state in (LABEL_POSITIVE, LABEL_NEGATIVE)
    qc_passed = obs.get("qc_state") in _QC_PASSING
    verified = bool(verification and verification.get("verified"))
    disagreed = bool(verification) and verification.get("rfdetr_agrees_with_field") == 0
    disagreement_resolved = disagreed and label_state == LABEL_POSITIVE

    missing = []
    if not expert_confirmed:
        missing.append(STEP_REVIEW)
    if not qc_passed:
        missing.append(STEP_QC)
    if not verified:
        missing.append(STEP_VERIFY)

    return {
        "observation_id": obs.get("id"),
        "event_name": obs.get("event_name"),
        "station_id": obs.get("station_id"),
        "first_seen": obs.get("first_seen"),
        "modality": modality,
        "qc_state": obs.get("qc_state"),
        "representative_frame": obs.get("representative_frame"),
        "audio_clip_path": obs.get("audio_clip_path"),
        "frame_count": obs.get("frame_count"),
        "audio_true_duration_seconds": obs.get("audio_true_duration_seconds"),
        "criteria": {
            "expert_confirmed": expert_confirmed,
            "qc_passed": qc_passed,
            "verified": verified,
        },
        "label_state": label_state,
        "taxa": taxa,
        "positive_taxa": positives,
        "disagreement_resolved": disagreement_resolved,
        "missing_steps": missing,
        "frame_verdicts": {
            "accurate": int(frames.get("accurate") or 0),
            "inaccurate": int(frames.get("inaccurate") or 0),
        },
    }


def required_criteria(selected: Optional[Iterable[str]]) -> frozenset:
    """The criteria a person switched on, checked against the known three.

    An unknown name raises ValueError so a typo in a request is reported rather than
    quietly ignored, which would keep more events than the person intended.
    """
    chosen = frozenset(selected or ())
    unknown = sorted(chosen - set(CRITERIA))
    if unknown:
        raise ValueError(f"unknown criteria {', '.join(unknown)}; the known ones are {', '.join(CRITERIA)}")
    return chosen


def is_eligible(candidate: dict, required: frozenset) -> bool:
    """Whether this event meets the criteria a person asked for.

    A rejected event is a person's decision that nothing is there, so it is
    eligible as a background example whatever the other two criteria say. Those two
    criteria exist to decide how far to trust a model's claim, and a rejection is
    not a model's claim. An event with no expert verdict at all, or only some, is
    eligible only when Expert Confirmed is switched off, and then only for the
    parts no expert has judged.
    """
    state = candidate["label_state"]
    if state == LABEL_UNRESOLVED:
        return False
    if state == LABEL_NEGATIVE:
        return True
    flags = candidate["criteria"]
    if "expert_confirmed" in required and not flags["expert_confirmed"]:
        return False
    if "qc_passed" in required and not flags["qc_passed"]:
        return False
    if "verified" in required and not flags["verified"]:
        return False
    return True


def _combination_name(flags: dict) -> str:
    return "+".join(name for name in CRITERIA if flags.get(name)) or "none"


def summarize(candidates: list, required: frozenset) -> dict:
    """Counts a person needs before building, from the same rows the builder uses.

    Everything here is a count of events. How many frames or windows each one
    yields is decided when the dataset is planned, because it depends on the frame
    rule, the review verdicts on frames and the files actually present.
    """
    per_criterion = {name: 0 for name in CRITERIA}
    combinations = {name: 0 for name in COMBINATIONS}
    by_class: dict = {}
    by_station: dict = {}
    states: dict = {}
    eligible = 0
    disagreement = 0
    for cand in candidates:
        flags = cand["criteria"]
        for name in CRITERIA:
            if flags[name]:
                per_criterion[name] += 1
        combinations[_combination_name(flags)] += 1
        states[cand["label_state"]] = states.get(cand["label_state"], 0) + 1
        if is_eligible(cand, required):
            eligible += 1
            station = cand.get("station_id") or "unknown"
            by_station[station] = by_station.get(station, 0) + 1
            if cand["label_state"] == LABEL_NEGATIVE:
                by_class["background"] = _bump(by_class.get("background"), "background")
            else:
                for taxon in cand["taxa"]:
                    if taxon["verdict"] == "reject":
                        continue
                    if cand["label_state"] == LABEL_MODEL or taxon["verdict"] is None:
                        # An unjudged call is the model's own, so it is counted
                        # under its own class and flagged as such.
                        by_class[taxon["class_key"]] = _bump(by_class.get(taxon["class_key"]), taxon["class_name"], model=True)
                    else:
                        by_class[taxon["class_key"]] = _bump(by_class.get(taxon["class_key"]), taxon["class_name"])
        if cand["disagreement_resolved"]:
            disagreement += 1
    return {
        "candidates": len(candidates),
        "eligible": eligible,
        "per_criterion": per_criterion,
        "combinations": combinations,
        "label_states": states,
        "disagreement_set": disagreement,
        "by_station": by_station,
        "by_class": by_class,
        "required": sorted(required),
    }


def _bump(entry: Optional[dict], name: str, model: bool = False) -> dict:
    entry = entry or {"class_name": name, "events": 0, "model_labelled_events": 0}
    entry["events"] += 1
    if model:
        entry["model_labelled_events"] += 1
    return entry
