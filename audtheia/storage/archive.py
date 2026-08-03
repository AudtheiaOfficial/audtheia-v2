"""Archive captured media out of the active store, and optionally reclaim space.

Path: audtheia/storage/archive.py

The desktop is the authoritative archive and never deletes on its own. But a
scientist may want to move older captured frames onto separate storage and free
space on the working drive while keeping the observation record. This copies each
event's frames and a self-describing metadata sidecar to a chosen location and,
only when explicitly asked and only after the copy has been verified, removes
those frames from the active store. The database row, with its counts, taxa,
verdicts, and salience, is never touched, so the science record survives; only
the heavy pixels move.

Every deletion is guarded three ways: an event's frames are removed only after
its copy is confirmed present at the destination, only from a folder that
resolves to inside the configured detections directory, and never the database or
any other path. This is deliberately strict because a reclaim deletes real
captures.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger("audtheia.storage.archive")


class ArchiveError(RuntimeError):
    """An archive or reclaim could not proceed for a stated reason."""


def _within(child: Path, parent: Path) -> bool:
    """True when child is parent itself or sits inside it, both resolved."""
    child = child.resolve()
    parent = parent.resolve()
    return child == parent or parent in child.parents


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def _event_metadata(db, observation_id: str) -> dict:
    """A self-describing sidecar: the record and every claim about it."""
    obs = db.get_observation(observation_id) or {}
    meta = {
        "observation": obs,
        "child_detections": db.list_child_detections(observation_id),
        "environmental_readings": db.list_environmental_readings(observation_id),
        "verification": db.get_observation_verification(observation_id),
        "interpretations": db.list_interpretations(observation_id),
    }
    for name, fn in (
        ("corrections", "corrections_for_observation"),
        ("frame_reviews", "frame_reviews_for_observation"),
        ("skill_flags", "list_skill_flags"),
    ):
        getter = getattr(db, fn, None)
        if getter is not None:
            try:
                meta[name] = getter(observation_id)
            except Exception:  # noqa: BLE001 - an optional table absent is not fatal
                meta[name] = []
    return meta


def archive_events(
    db,
    settings,
    *,
    target_dir: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    station_id: Optional[str] = None,
    reclaim: bool = False,
) -> dict:
    """Copy each in-range event's frames and metadata to target_dir.

    When reclaim is set, the event's frames are then removed from the active
    detections directory, but only after the copy is confirmed at the
    destination. The database is never modified. Returns a summary.
    """
    visual_dir = Path(settings.path("detections_visual_dir")).resolve()
    if not (target_dir and str(target_dir).strip()):
        raise ArchiveError("a destination folder is required")
    target = Path(target_dir).expanduser().resolve()
    # The destination must not be inside the active detections directory, or a
    # copy would recurse into itself and a reclaim could delete what it just wrote.
    if _within(target, visual_dir) or _within(visual_dir, target):
        raise ArchiveError("the destination must be outside the captured-data folder")
    target.mkdir(parents=True, exist_ok=True)

    summary = {"events": 0, "archived": 0, "reclaimed": 0, "bytes_freed": 0,
               "skipped_no_frames": 0, "target": str(target)}

    for obs in db.list_observations(station_id=station_id, since=start, until=end):
        summary["events"] += 1
        event_name = obs.get("event_name")
        if not event_name:
            continue
        source = (visual_dir / event_name).resolve()
        # The source must be a real folder inside the detections directory.
        if not (source.is_dir() and _within(source, visual_dir) and source != visual_dir):
            summary["skipped_no_frames"] += 1
            continue

        dest = target / event_name
        try:
            shutil.copytree(source, dest, dirs_exist_ok=True)
            (dest / "metadata.json").write_text(
                json.dumps(_event_metadata(db, obs["id"]), indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as exc:
            raise ArchiveError(f"could not copy {event_name} to the destination: {exc}") from exc
        summary["archived"] += 1

        if reclaim:
            # Only delete once the copy is confirmed present at the destination,
            # and only a folder proven to be inside the detections directory.
            if not dest.is_dir():
                logger.error("archive of %s not found at destination; not reclaiming it", event_name)
                continue
            if not (_within(source, visual_dir) and source != visual_dir and source.is_dir()):
                continue
            freed = _dir_size(source)
            try:
                shutil.rmtree(source)
            except OSError as exc:
                logger.error("could not reclaim %s: %s", event_name, exc)
                continue
            summary["reclaimed"] += 1
            summary["bytes_freed"] += freed

    return summary
