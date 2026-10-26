"""Dividing events into training, validation and test sets without leakage.

Path: audtheia/training/splits.py

The unit that is divided is the event, never the frame or the window. Every frame
or window of an event goes to the same set, because neighbouring frames of one
animal are nearly identical and a model that has seen some of them has not been
tested fairly on the rest.

Only events an expert has judged may be placed in validation or test. A test set
labelled by the old model measures agreement with that model rather than accuracy,
so events whose labels are the model's own go to training only.

The same inputs and the same seed always give the same division. Each class is
shuffled with its own generator seeded from the seed and the class name, so adding
or removing one class never changes how another is divided.

Three modes are offered.

  event       Each class is divided on its own, so every class that has enough events
              appears in every set. This is the default.
  station     Whole stations are held out, which measures how the model does at a
              place it has never seen. It needs at least three stations.
  time_block  The latest events are held out, which measures how the model does on
              a later season. The block is a share of the events, or a number of days.

The same animal can still return as separate events minutes apart, and an event-wise
division cannot prevent that. The time block mode reduces it.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

__all__ = [
    "SPLITS",
    "SplitError",
    "assign_splits",
]

SPLITS = ("train", "validation", "test")


class SplitError(ValueError):
    """The requested division cannot be made, with a reason a person can act on."""


def _parse_time(value: str) -> datetime:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise SplitError(f"an event has a start time that cannot be read: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _held_out_counts(n: int, fractions: dict) -> tuple:
    """How many of n events go to validation and to test, leaving at least one for training."""
    want_val = _round_half_up(n * fractions["validation"]) if fractions["validation"] > 0 else 0
    want_test = _round_half_up(n * fractions["test"]) if fractions["test"] > 0 else 0
    if fractions["validation"] > 0:
        want_val = max(1, want_val)
    if fractions["test"] > 0:
        want_test = max(1, want_test)
    while want_val + want_test > n - 1:
        if want_test >= want_val and want_test > 0:
            want_test -= 1
        elif want_val > 0:
            want_val -= 1
        else:
            break
    return want_val, want_test


def _check_fractions(fractions: dict) -> dict:
    out = {name: float(fractions.get(name, 0.0)) for name in SPLITS}
    if any(v < 0 or v > 1 for v in out.values()):
        raise SplitError("every split fraction must be between 0 and 1")
    if abs(sum(out.values()) - 1.0) > 1e-6:
        raise SplitError("the split fractions must add up to 1")
    if out["train"] <= 0:
        raise SplitError("the training fraction must be greater than 0")
    return out


def _by_event_order(units: Iterable[dict]) -> list:
    return sorted(units, key=lambda u: (str(u.get("first_seen") or ""), str(u["id"])))


def _event_mode(units: list, fractions: dict, seed: int, min_events: int, assignment: dict, notes: list) -> set:
    groups: dict = {}
    for unit in _by_event_order(u for u in units if u["expert"]):
        groups.setdefault(unit["class_key"], []).append(unit)
    untested = set()
    for class_key in sorted(groups):
        members = list(groups[class_key])
        rng = random.Random(f"{seed}:{class_key}")
        rng.shuffle(members)
        if len(members) < min_events:
            for unit in members:
                assignment[unit["id"]] = "train"
            untested.add(class_key)
            continue
        n_val, n_test = _held_out_counts(len(members), fractions)
        for position, unit in enumerate(members):
            if position < n_test:
                assignment[unit["id"]] = "test"
            elif position < n_test + n_val:
                assignment[unit["id"]] = "validation"
            else:
                assignment[unit["id"]] = "train"
        if n_test == 0:
            untested.add(class_key)
    return untested


def _station_mode(units: list, fractions: dict, seed: int, assignment: dict, notes: list) -> Optional[set]:
    expert = [u for u in units if u["expert"]]
    stations = sorted({str(u.get("station_id") or "") for u in expert})
    if len(stations) < 3:
        return None
    rng = random.Random(f"{seed}:stations")
    order = list(stations)
    rng.shuffle(order)
    totals = {s: sum(1 for u in expert if str(u.get("station_id") or "") == s) for s in stations}
    total = sum(totals.values())
    held: dict = {}
    for split in ("test", "validation"):
        target = fractions[split] * total
        taken = 0
        if target <= 0:
            continue
        for station in list(order):
            if len(order) <= 1:
                break
            if taken >= target:
                break
            held[station] = split
            taken += totals[station]
            order.remove(station)
    for unit in expert:
        assignment[unit["id"]] = held.get(str(unit.get("station_id") or ""), "train")
    return set()


def _time_mode(units: list, fractions: dict, days: Optional[float], assignment: dict) -> set:
    expert = _by_event_order(u for u in units if u["expert"])
    if not expert:
        return set()
    if days:
        latest = max(_parse_time(u["first_seen"]) for u in expert)
        test_start = latest - timedelta(days=float(days))
        val_start = test_start - timedelta(days=float(days))
        for unit in expert:
            when = _parse_time(unit["first_seen"])
            if fractions["test"] > 0 and when > test_start:
                assignment[unit["id"]] = "test"
            elif fractions["validation"] > 0 and when > val_start:
                assignment[unit["id"]] = "validation"
            else:
                assignment[unit["id"]] = "train"
        return set()
    n_val, n_test = _held_out_counts(len(expert), fractions)
    for position, unit in enumerate(expert):
        if position >= len(expert) - n_test:
            assignment[unit["id"]] = "test"
        elif position >= len(expert) - n_test - n_val:
            assignment[unit["id"]] = "validation"
        else:
            assignment[unit["id"]] = "train"
    return set()


def assign_splits(
    units: list,
    *,
    mode: str,
    seed: int,
    fractions: dict,
    min_events_per_class_for_heldout: int = 3,
    time_block_days: Optional[float] = None,
) -> dict:
    """Place every event in training, validation or test.

    Each unit is a dictionary with an id, a class_key (use "background" for events
    that are false alarms), a station_id, a first_seen time, and expert, which is
    True when an expert judged the event. Returns the assignment, the mode actually
    used, the classes that could not be tested and plain-language notes about
    anything the request could not do exactly.
    """
    if mode not in ("event", "station", "time_block"):
        raise SplitError("the split mode must be event, station or time_block")
    fractions = _check_fractions(fractions)
    assignment: dict = {}
    notes: list = []
    used = mode

    if mode == "station":
        untested = _station_mode(units, fractions, seed, assignment, notes)
        if untested is None:
            notes.append(
                "Holding out whole stations needs at least three stations with expert-judged events,"
                " so the events were divided one by one instead."
            )
            used = "event"
            untested = _event_mode(units, fractions, seed, min_events_per_class_for_heldout, assignment, notes)
    elif mode == "time_block":
        untested = _time_mode(units, fractions, time_block_days, assignment)
    else:
        untested = _event_mode(units, fractions, seed, min_events_per_class_for_heldout, assignment, notes)

    for unit in units:
        assignment.setdefault(unit["id"], "train")
        if not unit["expert"]:
            assignment[unit["id"]] = "train"

    # A class is untested when none of its events reached the test set, whichever
    # mode did the dividing. It is still trained on, and results mark it as untested.
    tested_classes = {u["class_key"] for u in units if u["expert"] and assignment[u["id"]] == "test"}
    all_classes = {u["class_key"] for u in units if u["expert"] and u["class_key"] != "background"}
    untested = set(untested or ()) | (all_classes - tested_classes)
    untested.discard("background")

    return {
        "assignment": assignment,
        "mode_used": used,
        "untested_classes": sorted(untested),
        "notes": notes,
    }
