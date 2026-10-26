"""Planning and writing a training dataset from reviewed events.

Path: audtheia/training/dataset.py

One plan describes a dataset completely, before anything is written. The preview a
person sees and the dataset that is built come from the same plan, so they cannot
differ. A plan is a pure read of the record and the stored media. Building then
writes a self-contained folder under the datasets directory and nothing else.

What this module never does

  It never writes, moves, renames or deletes anything under the data folder. Media is
  hard-linked or copied out of it. It never changes the database. It refuses to
  write if the datasets folder would sit inside the data folder.

The one file a plan may write is the cached roll-up answers under the datasets folder,
which is outside the data folder and can be deleted at any time.

The order of decisions

  1. Candidates. Events that meet the criteria a person switched on, from
     audtheia/training/candidates.py.
  2. Frames or windows. The frame rules in frames.py, or the window rules in
     windows.py, with the per-event cap and near-duplicate removal.
  3. Model-labelled content. Examples whose label is the model's own are capped by a
     setting, default none, and are only ever used for training.
  4. Class policy. Rare classes are excluded, kept with a warning, or rolled up to a
     genus or family, one class at a time, and the result is previewed.
  5. Split. Whole events go to training, validation or test, with a fixed seed.
  6. Background. Events an expert rejected become background or noise examples, up
     to a fraction of the positive examples in each split.

The manifest records every choice and a signature of the selection. Building the
same selection twice finds the first dataset and writes nothing.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from audtheia.training import candidates as cand_rules
from audtheia.training import formats, frames as frame_rules, windows as window_rules
from audtheia.training.splits import SPLITS, SplitError, assign_splits
from audtheia.training.taxon_rollup import LEVELS, RollupError, parent_for, rollup_targets

__all__ = [
    "DatasetError",
    "DatasetOptions",
    "RULES_VERSION",
    "options_from_request",
    "plan_dataset",
    "summarize_plan",
    "build_dataset",
    "dataset_manifest",
    "list_datasets",
]

# Bumped whenever a rule above changes what a given selection produces, so a dataset
# built under older rules is never mistaken for one built under the current rules.
RULES_VERSION = 1

BACKGROUND = "background"
CLASS_ACTIONS = ("exclude", "keep", "roll_up_genus", "roll_up_family")


class DatasetError(RuntimeError):
    """A dataset cannot be planned or built, with a reason a person can act on."""


@dataclass
class DatasetOptions:
    modality: str
    required: frozenset
    station_id: Optional[str]
    class_policy: dict
    roll_up_level: str
    split_mode: str
    seed: int
    fractions: dict
    min_events_for_heldout: int
    time_block_days: Optional[float]
    max_frames_per_event: int
    near_duplicate_hamming: int
    background_max_fraction: float
    model_labelled_max_fraction: float
    min_examples_per_class: int
    under_threshold_default: str
    engine_key: Optional[str] = None
    window_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    window_selection: str = "scored_with_energy_fallback"
    window_score_floor: float = 0.10
    max_windows_per_clip: int = 3
    noise_windows_per_rejected_clip: int = 1
    link_media: bool = True

    def public(self) -> dict:
        return {
            "modality": self.modality,
            "required_criteria": sorted(self.required),
            "station_id": self.station_id,
            "class_policy": dict(sorted(self.class_policy.items())),
            "roll_up_level": self.roll_up_level,
            "split": {
                "mode": self.split_mode,
                "seed": self.seed,
                "fractions": dict(self.fractions),
                "min_events_per_class_for_heldout": self.min_events_for_heldout,
                "time_block_days": self.time_block_days,
            },
            "max_frames_per_event": self.max_frames_per_event,
            "near_duplicate_hamming": self.near_duplicate_hamming,
            "background_max_fraction": self.background_max_fraction,
            "model_labelled_max_fraction": self.model_labelled_max_fraction,
            "min_examples_per_class": self.min_examples_per_class,
            "under_threshold_default": self.under_threshold_default,
            "engine": self.engine_key,
            "window_seconds": self.window_seconds,
            "sample_rate": self.sample_rate,
            "window_selection": self.window_selection,
            "window_score_floor": self.window_score_floor,
            "max_windows_per_clip": self.max_windows_per_clip,
            "noise_windows_per_rejected_clip": self.noise_windows_per_rejected_clip,
        }


_REQUEST_KEYS = {
    "modality", "required_criteria", "station_id", "class_policy", "roll_up_level", "split", "engine",
    "max_frames_per_event", "background_max_fraction", "model_labelled_max_fraction", "min_examples_per_class",
}
_SPLIT_KEYS = {"mode", "seed", "fractions", "min_events_per_class_for_heldout", "time_block_days"}


def _number(value, label: str, *, lo=None, hi=None, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetError(f"{label} must be a number")
    if integer and float(value) != int(value):
        raise DatasetError(f"{label} must be a whole number")
    if lo is not None and value < lo:
        raise DatasetError(f"{label} must be at least {lo}")
    if hi is not None and value > hi:
        raise DatasetError(f"{label} must be at most {hi}")
    return int(value) if integer else float(value)


def options_from_request(settings, request: dict) -> DatasetOptions:
    """Combine a request with the configured defaults into checked options.

    Unknown request fields are refused rather than ignored, so a misspelt choice
    cannot silently leave a default in place.
    """
    if not isinstance(request, dict):
        raise DatasetError("the request must be an object")
    unknown = sorted(set(request) - _REQUEST_KEYS)
    if unknown:
        raise DatasetError(f"unknown request fields: {', '.join(unknown)}")
    modality = request.get("modality")
    if modality not in ("vision", "audio"):
        raise DatasetError("modality must be 'vision' or 'audio'")

    config = settings.training_config()
    ds = config["dataset"]
    split_in = request.get("split") or {}
    if not isinstance(split_in, dict):
        raise DatasetError("split must be an object")
    bad = sorted(set(split_in) - _SPLIT_KEYS)
    if bad:
        raise DatasetError(f"unknown split fields: {', '.join(bad)}")
    split_cfg = ds["split"]

    try:
        required = cand_rules.required_criteria(
            request.get("required_criteria", ds["default_required_criteria"])
        )
    except ValueError as exc:
        raise DatasetError(str(exc)) from None

    policy = request.get("class_policy") or {}
    if not isinstance(policy, dict):
        raise DatasetError("class_policy must be an object")
    for key, action in policy.items():
        if action not in CLASS_ACTIONS:
            raise DatasetError(f"class_policy for {key} must be one of {', '.join(CLASS_ACTIONS)}")
    level = request.get("roll_up_level", "genus")
    if level not in LEVELS:
        raise DatasetError(f"roll_up_level must be one of {', '.join(LEVELS)}")

    fractions = dict(split_cfg["fractions"])
    fractions.update(split_in.get("fractions") or {})
    mode = split_in.get("mode", split_cfg["mode"])
    if mode not in ("event", "station", "time_block"):
        raise DatasetError("split mode must be event, station or time_block")
    days = split_in.get("time_block_days", split_cfg.get("time_block_days"))
    if days is not None:
        days = _number(days, "time_block_days", lo=0.0)
        if days <= 0:
            raise DatasetError("time_block_days must be greater than 0")

    options = DatasetOptions(
        modality=modality,
        required=required,
        station_id=request.get("station_id"),
        class_policy=dict(policy),
        roll_up_level=level,
        split_mode=mode,
        seed=_number(split_in.get("seed", split_cfg["seed"]), "seed", integer=True),
        fractions={k: _number(v, f"fraction {k}", lo=0.0, hi=1.0) for k, v in fractions.items()},
        min_events_for_heldout=_number(
            split_in.get("min_events_per_class_for_heldout", split_cfg["min_events_per_class_for_heldout"]),
            "min_events_per_class_for_heldout", lo=1, integer=True),
        time_block_days=days,
        max_frames_per_event=_number(request.get("max_frames_per_event", ds["max_frames_per_event"]),
                                     "max_frames_per_event", lo=1, integer=True),
        near_duplicate_hamming=_number(ds["near_duplicate_hamming"], "near_duplicate_hamming", lo=0, hi=64, integer=True),
        background_max_fraction=_number(request.get("background_max_fraction", ds["background_max_fraction"]),
                                        "background_max_fraction", lo=0.0, hi=1.0),
        model_labelled_max_fraction=_number(
            request.get("model_labelled_max_fraction", ds["model_labelled_max_fraction"]),
            "model_labelled_max_fraction", lo=0.0, hi=0.99),
        min_examples_per_class=_number(request.get("min_examples_per_class", ds["min_examples_per_class"]),
                                       "min_examples_per_class", lo=1, integer=True),
        under_threshold_default=ds["under_threshold_default"],
    )
    ac = ds["acoustic"]
    options.window_selection = ac["window_selection"]
    options.window_score_floor = float(ac["window_score_floor"])
    options.max_windows_per_clip = int(ac["max_windows_per_clip"])
    options.noise_windows_per_rejected_clip = int(ac["noise_windows_per_rejected_clip"])
    if modality == "audio":
        engines = {e["key"]: e for e in settings.training_acoustic_engines()}
        engine_key = request.get("engine")
        if engine_key is None:
            raise DatasetError("an acoustic dataset needs an engine, which sets the window length and sample rate")
        if engine_key not in engines:
            raise DatasetError(f"unknown acoustic engine {engine_key!r}; the known ones are {', '.join(sorted(engines))}")
        options.engine_key = engine_key
        options.window_seconds = float(engines[engine_key]["window_seconds"])
        options.sample_rate = int(engines[engine_key]["sample_rate"])
    return options


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_name(value: str) -> str:
    cleaned = "".join("_" if ch in '<>:"/\\|?*' or ord(ch) < 32 else ch for ch in (value or "")).strip().strip(". ")
    return cleaned or "event"


def _rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError:
        return str(path)


def _taxon_copy(taxon: dict) -> dict:
    return {
        "class_key": taxon["class_key"],
        "class_name": taxon["class_name"],
        "gbif_usage_key": taxon.get("gbif_usage_key"),
        "source": taxon.get("source", "model"),
    }


def _item_taxa(item: dict) -> list:
    if "boxes" in item:
        return [b["taxon"] for b in item["boxes"]]
    return list(item.get("labels") or [])


def _datasets_root(settings) -> Path:
    root = Path(settings.training_paths()["datasets_dir"]).resolve()
    data = Path(settings.path("data_dir")).resolve()
    if root == data or data in root.parents:
        raise DatasetError("the datasets folder must not be inside the data folder")
    return root


def default_rollup_lookup(settings) -> Callable:
    """Roll-up through the shipped backbone, with answers cached beside the datasets."""
    backbone = Path(settings.path("gbif_backbone_path"))
    cache = _datasets_root(settings) / ".rollup_cache.json"

    def lookup(class_keys, level):
        try:
            info = rollup_targets(backbone, class_keys, cache_path=cache)
        except RollupError as exc:
            raise DatasetError(str(exc)) from None
        return {key: parent_for(info.get(key), level) for key in class_keys}, info

    return lookup


# ---------------------------------------------------------------------------
# Gathering events
# ---------------------------------------------------------------------------


def _gather_vision(db, settings, options, candidates, notes, counters) -> list:
    repo_root = Path(settings.repo_root)
    data_dir = Path(settings.path("data_dir"))
    events = []
    for cand in candidates:
        try:
            event_dir = frame_rules.event_directory(repo_root, data_dir, cand.get("representative_frame"))
        except frame_rules.FrameSourceError as exc:
            counters["events_skipped"].append({"event": cand["event_name"], "reason": str(exc)})
            continue
        records = frame_rules.read_annotations(event_dir)
        if not records:
            counters["events_skipped"].append({"event": cand["event_name"], "reason": "the event has no saved frames"})
            continue
        verdicts = {r["frame_index"]: r["verdict"] for r in db.frame_reviews_for_observation(cand["observation_id"])}
        box_rows = db.frame_boxes_for_observation(cand["observation_id"])
        selection = frame_rules.select_frames(cand, records, verdicts, box_rows)
        thinned = frame_rules.thin_frames(
            selection["frames"], event_dir,
            max_frames=options.max_frames_per_event,
            near_duplicate_hamming=options.near_duplicate_hamming,
        )
        counters["missing_media"] += thinned["missing"]
        counters["near_duplicates"] += thinned["near_duplicates"]
        for reason, n in selection["skipped"].items():
            counters["frames_left_out"][reason] = counters["frames_left_out"].get(reason, 0) + n
        items = []
        for frame in thinned["frames"]:
            path = frame["path"]
            try:
                width, height = frame_rules.image_size(path)
            except Exception:  # noqa: BLE001 - an unreadable image is left out, never fatal
                counters["missing_media"] += 1
                continue
            boxes = []
            for box in frame["boxes"]:
                clamped = formats.clamp_box(box["xyxy"], width, height)
                if clamped is None:
                    continue
                boxes.append({"xyxy": [round(v, 2) for v in clamped], "taxon": _taxon_copy(box["taxon"]),
                              "box_source": box["box_source"]})
            if frame["role"] == "positive" and not boxes:
                continue
            index = int(frame["rec"]["index"])
            items.append({
                "id": f"{cand['observation_id']}:{index}",
                "event_id": cand["observation_id"],
                "role": frame["role"],
                "model_labelled": frame["model_labelled"],
                "box_edited": frame["box_edited"],
                "boxes": boxes,
                "path": path,
                "file_name": f"{_safe_name(cand['event_name'])}__{Path(frame['rec']['file']).name}",
                "width": width,
                "height": height,
                "frame_index": index,
            })
        if items:
            events.append(_event_record(cand, items))
        else:
            counters["events_skipped"].append({"event": cand["event_name"], "reason": "no frame qualified under the rules"})
    return events


def _event_record(cand: dict, items: list) -> dict:
    return {
        "id": cand["observation_id"],
        "event_name": cand["event_name"],
        "station_id": cand.get("station_id"),
        "first_seen": cand.get("first_seen"),
        "label_state": cand["label_state"],
        "criteria": dict(cand["criteria"]),
        "disagreement_resolved": cand["disagreement_resolved"],
        "items": items,
    }


def _gather_audio(db, settings, options, candidates, scorer, counters) -> list:
    repo_root = Path(settings.repo_root)
    data_dir = Path(settings.path("data_dir")).resolve()
    events = []
    for cand in candidates:
        clip_value = cand.get("audio_clip_path")
        clip = Path(clip_value) if clip_value else None
        if clip is None:
            counters["events_skipped"].append({"event": cand["event_name"], "reason": "the event has no stored clip"})
            continue
        clip = (clip if clip.is_absolute() else repo_root / clip).resolve()
        if data_dir != clip and data_dir not in clip.parents:
            counters["events_skipped"].append({"event": cand["event_name"], "reason": "the stored clip path is outside the data folder"})
            continue
        if not clip.is_file():
            counters["missing_media"] += 1
            counters["events_skipped"].append({"event": cand["event_name"], "reason": "the clip file is missing"})
            continue
        try:
            samples, rate = window_rules.read_wav(clip)
        except window_rules.AudioSourceError as exc:
            counters["events_skipped"].append({"event": cand["event_name"], "reason": str(exc)})
            continue
        duration = len(samples) / float(rate)
        items = []
        if cand["label_state"] == cand_rules.LABEL_NEGATIVE:
            rng = random.Random(f"{options.seed}:{cand['observation_id']}:noise")
            chosen = window_rules.noise_windows(duration, options.window_seconds,
                                                options.noise_windows_per_rejected_clip, rng)
            labels_for_event = []
        else:
            labels_for_event = [t for t in cand["taxa"] if t["verdict"] != "reject"]
            chosen = window_rules.select_windows(
                samples, rate,
                window_seconds=options.window_seconds,
                max_windows=options.max_windows_per_clip,
                selection=options.window_selection,
                scorer=scorer,
                score_floor=options.window_score_floor,
            )
        if not chosen:
            counters["events_skipped"].append({"event": cand["event_name"], "reason": "the clip is too short for one window"})
            continue
        for k, win in enumerate(chosen):
            _piece, padded = window_rules.cut_window(samples, rate, win["start"], options.window_seconds)
            del _piece
            labels = [_taxon_copy(t) for t in labels_for_event]
            role = "background" if not labels else "positive"
            items.append({
                "id": f"{cand['observation_id']}:w{k}",
                "event_id": cand["observation_id"],
                "role": role,
                "model_labelled": any(l["source"] != "expert" for l in labels),
                "labels": labels,
                "path": clip,
                "file_name": f"{_safe_name(cand['event_name'])}__{'n' if role == 'background' else 'w'}{k}.wav",
                "start": win["start"],
                "end": win["end"],
                "method": win["method"],
                "score": win["score"],
                "padded": bool(padded),
                "sample_rate": rate,
            })
        events.append(_event_record(cand, items))
    return events


# ---------------------------------------------------------------------------
# Model-labelled cap, class policy, split and background
# ---------------------------------------------------------------------------


def _expert_instance_count(events: list) -> int:
    total = 0
    for event in events:
        for item in event["items"]:
            if item["role"] == "positive" and not item["model_labelled"]:
                total += len(_item_taxa(item))
    return total


def _apply_model_label_cap(events: list, options: DatasetOptions, warnings: list, counters: dict) -> None:
    f = options.model_labelled_max_fraction
    expert = _expert_instance_count(events)
    allowed = int(math.floor((f / (1.0 - f)) * expert)) if f > 0 else 0
    rng = random.Random(f"{options.seed}:model_labelled")
    holders = sorted((e for e in events if any(i["model_labelled"] for i in e["items"])), key=lambda e: e["id"])
    rng.shuffle(holders)
    removed = 0
    for event in holders:
        kept = []
        for item in event["items"]:
            if not item["model_labelled"]:
                kept.append(item)
                continue
            size = len(_item_taxa(item))
            if allowed >= size:
                allowed -= size
                kept.append(item)
            else:
                removed += 1
        event["items"] = kept
    events[:] = [e for e in events if e["items"]]
    counters["model_labelled_removed"] = removed
    if f > 0 and any(i["model_labelled"] for e in events for i in e["items"]):
        warnings.append(
            "Some training examples are labelled by the model, not by a person. They are used for training only,"
            " never for validation or test, and each is flagged in the manifest."
        )


def _class_counts(events: list) -> dict:
    table: dict = {}
    for event in events:
        seen_in_event = set()
        for item in event["items"]:
            if item["role"] != "positive":
                continue
            for taxon in _item_taxa(item):
                row = table.setdefault(taxon["class_key"], {
                    "class_key": taxon["class_key"], "class_name": taxon["class_name"],
                    "gbif_usage_key": taxon.get("gbif_usage_key"),
                    "events": 0, "instances": 0, "model_labelled_instances": 0,
                })
                row["instances"] += 1
                if taxon.get("source") != "expert":
                    row["model_labelled_instances"] += 1
                elif taxon["class_key"] not in seen_in_event:
                    row["events"] += 1
                seen_in_event.add(taxon["class_key"])
    return table


def _plan_class_policy(events: list, options: DatasetOptions, rollup_lookup, warnings: list) -> dict:
    before = _class_counts(events)
    threshold = options.min_examples_per_class
    decisions: dict = {}
    to_roll: dict = {"genus": [], "family": []}
    for key, row in before.items():
        under = row["events"] < threshold
        requested = options.class_policy.get(key)
        action = requested
        if action is None and under:
            default = options.under_threshold_default
            if default == "exclude":
                action = "exclude"
            elif default == "roll_up":
                action = f"roll_up_{options.roll_up_level}"
            else:
                action = "keep"
        if action is None:
            action = "keep"
        decisions[key] = {"action": action, "under_threshold": under, "requested": requested is not None}
        if action.startswith("roll_up_"):
            to_roll[action.split("_", 2)[2]].append(key)

    rolled: dict = {}
    infos: dict = {}
    for level, keys in to_roll.items():
        if keys:
            parents, info = rollup_lookup(keys, level)
            infos.update(info)
            for key in keys:
                rolled[key] = parents.get(key)

    mapping: dict = {}
    for key, row in before.items():
        decision = decisions[key]
        action = decision["action"]
        if action == "exclude":
            mapping[key] = None
            decision["note"] = "excluded"
        elif action == "keep":
            mapping[key] = {"class_key": key, "class_name": row["class_name"], "gbif_usage_key": row["gbif_usage_key"]}
            if decision["under_threshold"]:
                decision["note"] = (f"kept with only {row['events']} expert-judged event(s), below the threshold of {threshold};"
                                    " results for this class will be unreliable")
        else:
            parent = rolled.get(key)
            if parent is None:
                reason = (infos.get(key) or {}).get("reason") or "the backbone gives no parent at that level"
                mapping[key] = {"class_key": key, "class_name": row["class_name"], "gbif_usage_key": row["gbif_usage_key"]}
                decision["note"] = f"could not be rolled up ({reason}), so it was kept as it is"
                decision["action"] = "keep"
                warnings.append(f"{row['class_name']} could not be rolled up: {reason}. It was kept as it is.")
            else:
                new_key = f"gbif:{parent['key']}"
                mapping[key] = {"class_key": new_key, "class_name": parent.get("name") or new_key,
                                "gbif_usage_key": str(parent["key"])}
                decision["note"] = f"rolled up into {parent.get('name') or new_key}"
    return {"mapping": mapping, "decisions": decisions, "before": before}


def _apply_class_mapping(events: list, mapping: dict) -> None:
    for event in events:
        kept = []
        for item in event["items"]:
            if item["role"] != "positive":
                kept.append(item)
                continue
            if "boxes" in item:
                boxes = []
                for box in item["boxes"]:
                    target = mapping.get(box["taxon"]["class_key"], "absent")
                    if target is None:
                        continue
                    if target != "absent":
                        box["taxon"] = {**box["taxon"], **target}
                    boxes.append(box)
                item["boxes"] = boxes
                if boxes:
                    kept.append(item)
            else:
                labels = []
                seen = set()
                for label in item["labels"]:
                    target = mapping.get(label["class_key"], "absent")
                    if target is None:
                        continue
                    if target != "absent":
                        label = {**label, **target}
                    if label["class_key"] not in seen:
                        seen.add(label["class_key"])
                        labels.append(label)
                item["labels"] = labels
                if labels:
                    kept.append(item)
        event["items"] = kept
    events[:] = [e for e in events if e["items"]]


def _primary_class(event: dict) -> str:
    counts: dict = {}
    for item in event["items"]:
        if item["role"] != "positive" or item["model_labelled"]:
            continue
        for taxon in _item_taxa(item):
            counts[taxon["class_key"]] = counts.get(taxon["class_key"], 0) + 1
    if not counts:
        return BACKGROUND
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _select_background(events: list, assignment: dict, options: DatasetOptions) -> dict:
    """Keep background examples up to a fraction of the positives in each split."""
    kept: dict = {}
    stats = {}
    for split in SPLITS:
        positives = sum(
            len(_item_taxa(i)) for e in events if assignment[e["id"]] == split
            for i in e["items"] if i["role"] == "positive"
        )
        cap = int(math.floor(options.background_max_fraction * positives))
        pool = sorted((e for e in events if assignment[e["id"]] == split
                       and any(i["role"] == "background" for i in e["items"])), key=lambda e: e["id"])
        queues = [[i for i in e["items"] if i["role"] == "background"] for e in pool]
        chosen = []
        position = 0
        while len(chosen) < cap and any(queues):
            for queue in queues:
                if position < len(queue) and len(chosen) < cap:
                    chosen.append(queue[position])
            position += 1
            if position > max((len(q) for q in queues), default=0):
                break
        kept[split] = {i["id"] for i in chosen}
        available = sum(len(q) for q in queues)
        stats[split] = {"available": available, "kept": len(chosen), "cap": cap}
    return {"kept": kept, "stats": stats}


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


def _signature(plan: dict) -> str:
    rows = []
    for split in SPLITS:
        for item in plan["items"][split]:
            row = [split, item["event_id"], item["file_name"], item["role"]]
            if "boxes" in item:
                row.append([[b["taxon"]["class_key"], b["xyxy"]] for b in item["boxes"]])
            else:
                row.append([[l["class_key"] for l in item["labels"]], item["start"], item["end"], item["method"]])
            rows.append(row)
    payload = {
        "rules": RULES_VERSION,
        "modality": plan["options"]["modality"],
        "criteria": plan["options"]["required_criteria"],
        "station": plan["options"]["station_id"],
        "split": plan["split"]["mode_used"],
        "seed": plan["options"]["split"]["seed"],
        "classes": [[c["class_key"], c["class_name"]] for c in plan["classes"]],
        "items": rows,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def plan_dataset(db, settings, options: DatasetOptions, *, rollup_lookup: Optional[Callable] = None,
                 scorer: Optional[Callable] = None) -> dict:
    """Work out exactly what a dataset would contain, writing nothing.

    Returns the full plan, including the media each example would be written from.
    Use summarize_plan for the compact form shown in the interface.
    """
    _datasets_root(settings)
    warnings: list = []
    counters = {"events_skipped": [], "missing_media": 0, "near_duplicates": 0,
                "frames_left_out": {}, "model_labelled_removed": 0}

    candidates = db.training_candidates(options.modality, options.station_id)
    eligible = [c for c in candidates if cand_rules.is_eligible(c, options.required)]
    if "expert_confirmed" not in options.required:
        warnings.append(
            "Expert Confirmed is switched off, so examples can carry the model's own label."
            " Without a person's judgement the model learns the previous model's mistakes."
            " Validation and test sets still come only from events an expert judged."
        )

    if options.modality == "vision":
        events = _gather_vision(db, settings, options, eligible, warnings, counters)
    else:
        events = _gather_audio(db, settings, options, eligible, scorer, counters)
        mismatched = sum(1 for e in events for i in e["items"] if i.get("sample_rate") != options.sample_rate)
        if mismatched:
            warnings.append(
                f"{mismatched} window(s) come from clips recorded at a sample rate other than the engine's"
                f" {options.sample_rate} Hz. They are written at their own rate, recorded in windows.csv, and the"
                " trainer resamples them when it loads them."
            )

    _apply_model_label_cap(events, options, warnings, counters)

    rollup = rollup_lookup or default_rollup_lookup(settings)
    policy = _plan_class_policy(events, options, rollup, warnings)
    _apply_class_mapping(events, policy["mapping"])

    units = []
    for event in events:
        primary = _primary_class(event)
        expert_content = any(i["role"] == "positive" and not i["model_labelled"] for i in event["items"])
        negative = event["label_state"] == cand_rules.LABEL_NEGATIVE
        units.append({
            "id": event["id"], "class_key": primary,
            "station_id": event["station_id"], "first_seen": event["first_seen"],
            "expert": bool(expert_content or negative),
        })
    try:
        divided = assign_splits(
            units, mode=options.split_mode, seed=options.seed, fractions=options.fractions,
            min_events_per_class_for_heldout=options.min_events_for_heldout,
            time_block_days=options.time_block_days,
        )
    except SplitError as exc:
        raise DatasetError(str(exc)) from None
    assignment = divided["assignment"]
    warnings.extend(divided["notes"])

    # Model-labelled examples are used for training only.
    dropped_for_split = 0
    for event in events:
        if assignment[event["id"]] != "train":
            before = len(event["items"])
            event["items"] = [i for i in event["items"] if not i["model_labelled"]]
            dropped_for_split += before - len(event["items"])
    events[:] = [e for e in events if e["items"]]
    if dropped_for_split:
        warnings.append(f"{dropped_for_split} model-labelled example(s) fell in events held out for validation or test and were removed.")

    background = _select_background(events, assignment, options)
    items_by_split = {s: [] for s in SPLITS}
    event_rows = []
    for event in sorted(events, key=lambda e: (str(e["first_seen"] or ""), e["id"])):
        split = assignment[event["id"]]
        kept_items = []
        for item in event["items"]:
            if item["role"] == "background" and item["id"] not in background["kept"][split]:
                continue
            kept_items.append(item)
        if not kept_items:
            continue
        items_by_split[split].extend(kept_items)
        event_rows.append({
            "event_id": event["id"], "event_name": event["event_name"], "station_id": event["station_id"],
            "first_seen": event["first_seen"], "split": split, "label_state": event["label_state"],
            "criteria": event["criteria"], "disagreement_resolved": event["disagreement_resolved"],
            "items": len(kept_items),
        })

    # Final classes and per-split counts, from what actually survived.
    final_counts: dict = {}
    per_split: dict = {s: {} for s in SPLITS}
    box_stats = {"model": 0, "expert_adjusted": 0, "expert_added": 0}
    class_names: dict = {}
    for split in SPLITS:
        for item in items_by_split[split]:
            if item["role"] != "positive":
                continue
            for taxon in _item_taxa(item):
                class_names[taxon["class_key"]] = (taxon["class_name"], taxon.get("gbif_usage_key"))
                row = per_split[split].setdefault(taxon["class_key"], {"events": set(), "instances": 0})
                row["events"].add(item["event_id"])
                row["instances"] += 1
            for box in item.get("boxes", []):
                box_stats[box["box_source"]] += 1
    untested = set(divided["untested_classes"])
    classes = []
    for key in sorted(class_names, key=lambda k: (class_names[k][0].lower(), k)):
        name, gbif = class_names[key]
        merged = sorted(
            ({"class_key": old, "class_name": policy["before"][old]["class_name"]}
             for old, tgt in policy["mapping"].items() if tgt and tgt["class_key"] == key and old != key),
            key=lambda m: m["class_key"])
        decision = policy["decisions"].get(key) or {}
        classes.append({
            "class_key": key, "class_name": name, "gbif_usage_key": gbif,
            "merged_from": merged,
            "counts": {s: {"events": len(per_split[s].get(key, {}).get("events", ())),
                           "instances": per_split[s].get(key, {}).get("instances", 0)} for s in SPLITS},
            "tested": key not in untested and bool(per_split["test"].get(key)),
            "note": decision.get("note"),
        })
    for index, cls in enumerate(classes):
        cls["category_id"] = index + 1
        cls["class_index"] = index
    class_by_key = {c["class_key"]: c for c in classes}

    for cls in classes:
        if not cls["tested"]:
            warnings.append(
                f"{cls['class_name']} is untested. Too few expert-judged events to hold one out, so it is"
                " trained on but gets no score."
            )
        if cls["note"] and cls["note"].startswith("kept with only"):
            warnings.append(f"{cls['class_name']} was {cls['note']}.")
    total_boxes = sum(box_stats.values())
    if options.modality == "vision":
        warnings.append(
            "Boxes are model-proposed and expert-verified. Where the model found nothing there is no box,"
            " so the trained model inherits the original model's blind spots."
            + (f" Of {total_boxes} boxes, {box_stats['expert_adjusted']} were adjusted and {box_stats['expert_added']}"
               " were added by an expert." if total_boxes else "")
        )
    else:
        multi = sum(1 for s in SPLITS for i in items_by_split[s] if len(i.get("labels", [])) > 1)
        if multi:
            warnings.append(f"{multi} window(s) carry more than one species label, and a window may contain only one of them.")
        if options.window_selection == "scored_with_energy_fallback" and not any(
                i.get("method") == window_rules.METHOD_SCORED for s in SPLITS for i in items_by_split[s]):
            warnings.append(
                "Windows were chosen by sound energy, not by a model, so they carry no model bias."
                " A loud sound that is not the target can win."
            )

    plan = {
        "options": options.public(),
        "items": items_by_split,
        "classes": classes,
        "class_by_key": class_by_key,
        "class_policy": {k: {kk: vv for kk, vv in v.items()} for k, v in policy["decisions"].items()},
        "classes_before": sorted(policy["before"].values(), key=lambda r: (r["class_name"].lower(), r["class_key"])),
        "events": event_rows,
        "split": {"mode_used": divided["mode_used"], "untested_classes": divided["untested_classes"],
                  "assignment": {e["event_id"]: e["split"] for e in event_rows}},
        "background": background["stats"],
        "box_stats": box_stats,
        "warnings": warnings,
        "skipped": {
            "events": counters["events_skipped"],
            "missing_media": counters["missing_media"],
            "near_duplicates": counters["near_duplicates"],
            "frames_left_out": counters["frames_left_out"],
            "model_labelled_removed": counters["model_labelled_removed"],
        },
        "candidates": len(candidates),
        "eligible": len(eligible),
    }
    plan["signature"] = _signature(plan)
    plan["dataset_id"] = f"{options.modality}-{plan['signature'][:16]}"
    return plan


def summarize_plan(plan: dict) -> dict:
    """The compact form of a plan shown before building, with no media attached."""
    totals = {s: {"events": 0, "examples": 0, "positive_instances": 0, "background": 0} for s in SPLITS}
    for row in plan["events"]:
        totals[row["split"]]["events"] += 1
    for split in SPLITS:
        for item in plan["items"][split]:
            totals[split]["examples"] += 1
            if item["role"] == "positive":
                totals[split]["positive_instances"] += len(_item_taxa(item))
            else:
                totals[split]["background"] += 1
    return {
        "dataset_id": plan["dataset_id"],
        "signature": plan["signature"],
        "options": plan["options"],
        "candidates": plan["candidates"],
        "eligible": plan["eligible"],
        "totals": totals,
        "classes": [
            {k: c[k] for k in ("class_key", "class_name", "gbif_usage_key", "merged_from", "counts", "tested", "note")}
            for c in plan["classes"]
        ],
        "classes_before": plan["classes_before"],
        "class_policy": plan["class_policy"],
        "split_mode_used": plan["split"]["mode_used"],
        "untested_classes": plan["split"]["untested_classes"],
        "background": plan["background"],
        "box_stats": plan["box_stats"],
        "warnings": plan["warnings"],
        "skipped": plan["skipped"],
        "events": plan["events"],
        "empty": not any(plan["items"][s] for s in SPLITS),
    }


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_datasets(settings) -> list:
    """Datasets already built, newest first, from their manifests."""
    root = _datasets_root(settings)
    found = []
    if root.is_dir():
        # The short summary file is read, not the manifest, which lists every example
        # and can be large. A folder without one is not a finished dataset.
        for summary in sorted(root.glob("*/summary.json")):
            if summary.parent.name.startswith("."):
                continue
            try:
                data = json.loads(summary.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            found.append({
                "dataset_id": data.get("dataset_id"), "modality": data.get("modality"),
                "created_at": data.get("created_at"), "signature": data.get("signature"),
                "totals": data.get("totals"), "path": str(summary.parent),
            })
    return sorted(found, key=lambda d: str(d.get("created_at") or ""), reverse=True)


def dataset_manifest(settings, dataset_id: str) -> Optional[dict]:
    if not dataset_id or Path(dataset_id).name != dataset_id or dataset_id.startswith("."):
        return None
    manifest = _datasets_root(settings) / dataset_id / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        return json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


_README = """# Training dataset

This folder is a self-contained training set written by Audtheia. Nothing in it
modifies the original record. Images and clips are linked or copied from the data
folder, and the data folder was only ever read.

Images and clips may be hard links to the originals, so never edit a file here in
place. Copy it elsewhere first.

A visual dataset is written twice over, as a COCO layout (train, valid and test folders, each
with _annotations.coco.json) and as a YOLO text layout under yolo/ with a data.yaml. Both
describe the same examples.

manifest.json lists every choice made, every example with the event it came from and
whether its label is an expert's or the model's, and a signature of the selection.
split.json lists which events are in training, validation and test. The split is by
event, never by frame or window, with the seed recorded.

Read the warnings in the manifest before training. In particular, boxes in a visual
dataset are the model's own, verified by an expert, and a model trained on them can
inherit the original model's blind spots. A class marked untested had too few
expert-judged events to hold one out and has no score.
"""


def build_dataset(db, settings, options: DatasetOptions, *, rollup_lookup=None, scorer=None) -> dict:
    """Write the planned dataset, or report the identical one that already exists."""
    plan = plan_dataset(db, settings, options, rollup_lookup=rollup_lookup, scorer=scorer)
    if not any(plan["items"][s] for s in SPLITS):
        raise DatasetError("nothing qualifies under these settings, so there is nothing to build")
    if not plan["items"]["train"]:
        raise DatasetError("no training examples remain after the split, so there is nothing to train on")
    root = _datasets_root(settings)
    target = root / plan["dataset_id"]
    if (target / "manifest.json").is_file():
        return {"already_exists": True, "dataset_id": plan["dataset_id"], "path": str(target),
                "summary": summarize_plan(plan)}
    partial = root / f".{plan['dataset_id']}.partial"
    root.mkdir(parents=True, exist_ok=True)
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir()
    try:
        written = _write_dataset(plan, options, settings, partial)
        if target.exists():
            shutil.rmtree(target)
        partial.rename(target)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return {"already_exists": False, "dataset_id": plan["dataset_id"], "path": str(target),
            "summary": summarize_plan(plan), "written": written}


def _write_dataset(plan: dict, options: DatasetOptions, settings, target: Path) -> dict:
    repo_root = Path(settings.repo_root)
    manifest_items = []
    link_kinds: dict = {}
    classes = plan["classes"]
    if options.modality == "vision":
        categories = [{"id": c["category_id"], "name": c["class_name"]} for c in classes]
        yolo_names = [c["class_name"] for c in classes]
        counts = {}
        for split in SPLITS:
            folder = target / formats.COCO_FOLDER_FOR_SPLIT[split]
            images = []
            for item in plan["items"][split]:
                kind = formats.place_file(item["path"], folder / item["file_name"], link=options.link_media)
                link_kinds[kind] = link_kinds.get(kind, 0) + 1
                images.append({
                    "file_name": item["file_name"], "width": item["width"], "height": item["height"],
                    "boxes": [{"category_id": plan["class_by_key"][b["taxon"]["class_key"]]["category_id"], "xyxy": b["xyxy"]}
                              for b in item["boxes"]],
                })
                manifest_items.append({
                    "split": split, "event_id": item["event_id"], "file": f"{formats.COCO_FOLDER_FOR_SPLIT[split]}/{item['file_name']}",
                    "source": _rel(item["path"], repo_root), "source_sha256": _sha256(item["path"]),
                    "role": item["role"], "model_labelled": item["model_labelled"], "box_expert_edited": item["box_edited"],
                    "frame_index": item["frame_index"],
                    "boxes": [{"class": b["taxon"]["class_name"], "box_source": b["box_source"], "xyxy": b["xyxy"]} for b in item["boxes"]],
                })
            counts[split] = formats.write_coco_split(folder, images, categories,
                                                     description=f"Audtheia {split} split, dataset {plan['dataset_id']}")
        yolo_images = {
            split: [{
                "source": item["path"], "file_name": item["file_name"], "width": item["width"], "height": item["height"],
                "boxes": [{"class_index": plan["class_by_key"][b["taxon"]["class_key"]]["class_index"], "xyxy": b["xyxy"]}
                          for b in item["boxes"]],
            } for item in plan["items"][split]]
            for split in SPLITS
        }
        yolo_counts = formats.write_yolo_dataset(target / "yolo", yolo_images, yolo_names, link=options.link_media)
        written = {"coco": counts, "yolo": yolo_counts, "yolo_class_names": yolo_names}
    else:
        rows = []
        cache_path = None
        cache_audio = None
        for split in SPLITS:
            for item in sorted(plan["items"][split], key=lambda i: (str(i["path"]), i["start"])):
                out = target / "windows" / split / item["file_name"]
                if cache_path != item["path"]:
                    cache_audio, _rate = window_rules.read_wav(item["path"])
                    cache_path = item["path"]
                piece, _padded = window_rules.cut_window(cache_audio, item["sample_rate"], item["start"], options.window_seconds)
                window_rules.write_wav_pcm16(piece, out, item["sample_rate"])
                labels = [l["class_name"] for l in item["labels"]]
                rows.append((split, item["file_name"], item["event_id"], ";".join(labels), item["start"], item["end"],
                             item["method"], int(item["padded"]), item["sample_rate"]))
                manifest_items.append({
                    "split": split, "event_id": item["event_id"], "file": f"windows/{split}/{item['file_name']}",
                    "source": _rel(item["path"], repo_root), "source_sha256": _sha256(item["path"]),
                    "role": item["role"], "model_labelled": item["model_labelled"],
                    "labels": labels, "start_seconds": item["start"], "end_seconds": item["end"],
                    "method": item["method"], "score": item["score"], "padded": item["padded"],
                    "sample_rate": item["sample_rate"],
                })
        header = "split,file,event_id,labels,start_seconds,end_seconds,method,padded,sample_rate\n"
        body = "".join(",".join(_csv(v) for v in row) + "\n" for row in rows)
        (target / "windows.csv").write_text(header + body, encoding="utf-8", newline="\n")
        written = {"windows": len(rows)}

    split_doc = {
        "mode": plan["split"]["mode_used"], "seed": options.seed, "fractions": options.fractions,
        "untested_classes": plan["split"]["untested_classes"],
        "assignment": plan["split"]["assignment"],
    }
    (target / "split.json").write_text(json.dumps(split_doc, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    (target / "classes.json").write_text(
        json.dumps([{k: c[k] for k in ("category_id", "class_index", "class_key", "class_name", "gbif_usage_key", "merged_from", "tested")}
                    for c in classes], indent=1) + "\n", encoding="utf-8", newline="\n")
    summary = summarize_plan(plan)
    manifest = {
        "dataset_id": plan["dataset_id"],
        "signature": plan["signature"],
        "rules_version": RULES_VERSION,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "modality": options.modality,
        "options": plan["options"],
        "totals": summary["totals"],
        "classes": summary["classes"],
        "class_policy": plan["class_policy"],
        "split": split_doc,
        "background": plan["background"],
        "box_stats": plan["box_stats"],
        "events": plan["events"],
        "warnings": plan["warnings"],
        "skipped": plan["skipped"],
        "media_placement": link_kinds,
        "software": {"python": sys.version.split()[0]},
        "provenance": (
            "Labels marked model_labelled are the model's own calls. Every other label comes from an expert's"
            " confirm or relabel verdict, or from a false-alarm rejection."
        ),
        "items": manifest_items,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    short = {k: manifest[k] for k in ("dataset_id", "modality", "created_at", "signature", "totals")}
    (target / "summary.json").write_text(json.dumps(short, indent=1) + "\n", encoding="utf-8", newline="\n")
    (target / "README.md").write_text(_README, encoding="utf-8", newline="\n")
    return written


def _csv(value) -> str:
    text = str(value)
    if any(ch in text for ch in ',"\n'):
        return '"' + text.replace('"', '""') + '"'
    return text
