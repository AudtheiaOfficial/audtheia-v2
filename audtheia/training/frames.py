"""Choosing which saved frames of an event become training examples.

Path: audtheia/training/frames.py

An event's frames are the images the field station saved while one animal was
tracked, with one line each in the event's annotation file naming the model's class
and box for that frame. Nothing in this module changes any of those files. It reads
them, applies the frame rules below, and returns the frames that qualify.

The rules, applied to each frame in this order

  1. A frame an expert marked inaccurate is left out. Inaccurate can mean a wrong box
     or a wrong species, not an empty frame, so it is never turned into a background
     example.
  2. In an event an expert rejected as a false alarm, every remaining frame is a
     background example with no box.
  3. In an event with one detection that an expert relabelled, every remaining frame
     takes the expert's species. The model's per-frame class is ignored, because the
     expert has said it is wrong.
  4. In an event an expert confirmed, a frame whose model class matches a confirmed
     species is kept, and so is a frame the expert marked accurate. A frame the expert
     has not looked at, whose class matches nothing confirmed, is left out as uncertain.
  5. Where no expert verdict applies, only frames marked accurate are kept, labelled
     with the model's own call and flagged as model-labelled.

Boxes. A frame's box is the model's own unless an expert has adjusted or removed it,
and an expert may add boxes the model missed. Corrections are read from the
append-only box table, so the saved annotation file is never altered.

Thinning. At most a fixed number of frames per event are kept, spread evenly along
the track, then near-duplicates are dropped with a small perceptual hash. The cap is
per event, so one animal that lingered counts like one that passed quickly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Optional

from audtheia.training.candidates import LABEL_NEGATIVE, name_key, taxon_key

__all__ = [
    "FrameSourceError",
    "event_directory",
    "read_annotations",
    "select_frames",
    "thin_frames",
    "average_hash",
    "hamming",
    "image_size",
]


class FrameSourceError(RuntimeError):
    """An event's frames cannot be read safely, with a reason a person can act on."""


def event_directory(repo_root, data_dir, representative_frame: str) -> Path:
    """The folder holding an event's frames, refusing anything outside the data folder.

    The stored path is trusted no further than any other input. A path that resolves
    outside the data directory is reported, never followed.
    """
    if not representative_frame:
        raise FrameSourceError("the event has no stored frame")
    rep = Path(representative_frame)
    absolute = (rep if rep.is_absolute() else Path(repo_root) / rep).resolve()
    root = Path(data_dir).resolve()
    if root != absolute and root not in absolute.parents:
        raise FrameSourceError("the stored frame path is outside the data folder")
    return absolute.parent


def read_annotations(event_dir: Path) -> list:
    """The per-frame annotation rows of an event, in frame order, skipping unreadable lines."""
    index_file = Path(event_dir) / "annotations.jsonl"
    if not index_file.is_file():
        return []
    rows = []
    for line in index_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict) or rec.get("file") in (None, "") or rec.get("index") is None:
            continue
        bbox = rec.get("bbox_xyxy") or []
        rec["_bbox"] = [float(v) for v in bbox] if len(bbox) == 4 else None
        rows.append(rec)
    rows.sort(key=lambda r: int(r["index"]))
    return rows


def _match_child(class_name: str, taxa: list) -> Optional[dict]:
    wanted = name_key(class_name)
    if not wanted:
        return None
    for taxon in taxa:
        # The name the model used is the name stored before any relabel, so a relabelled
        # detection is still found by what the model called it.
        names = set(taxon.get("match_names") or ())
        if wanted in names:
            return taxon
    return None


def _box_state(rows: Iterable[dict]) -> dict:
    state: dict = {}
    for row in rows:
        state[(int(row["frame_index"]), row["box_key"])] = row
    return state


def _boxes_for_frame(rec: dict, taxon: Optional[dict], state: dict) -> tuple:
    """The boxes a kept frame carries and whether an expert edited any of them."""
    index = int(rec["index"])
    boxes = []
    edited = False
    model_row = state.get((index, "model"))
    model_box = rec.get("_bbox")
    if taxon is not None and model_box is not None:
        action = model_row.get("action") if model_row else None
        if action == "remove":
            edited = True
        elif action == "adjust":
            boxes.append({"xyxy": [model_row["x1"], model_row["y1"], model_row["x2"], model_row["y2"]],
                          "taxon": taxon, "box_source": "expert_adjusted"})
            edited = True
        else:
            boxes.append({"xyxy": list(model_box), "taxon": taxon, "box_source": "model"})
    for (frame_index, box_key), row in sorted(state.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        if frame_index != index or box_key == "model" or row.get("action") != "add":
            continue
        key = taxon_key(row.get("corrected_gbif_usage_key"), row.get("corrected_scientific_name"))
        if key is None:
            continue
        gbif = row.get("corrected_gbif_usage_key")
        boxes.append({
            "xyxy": [row["x1"], row["y1"], row["x2"], row["y2"]],
            "taxon": {
                "class_key": key,
                "class_name": (row.get("corrected_scientific_name") or key),
                "gbif_usage_key": str(gbif).strip() if gbif not in (None, "") else None,
                "source": "expert",
            },
            "box_source": "expert_added",
        })
        edited = True
    return boxes, edited


def select_frames(candidate: dict, annotations: list, verdicts: dict, box_rows: Iterable[dict]) -> dict:
    """Apply the frame rules to one event.

    Returns the kept frames and a tally of why the others were left out. Each kept
    frame is a dictionary with its annotation row, its role (positive or background),
    its boxes and whether it is model-labelled. Files are not opened here.
    """
    taxa = candidate["taxa"]
    single = taxa[0] if len(taxa) == 1 else None
    negative_event = candidate["label_state"] == LABEL_NEGATIVE
    state = _box_state(box_rows)
    kept: list = []
    skipped: dict = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for rec in annotations:
        index = int(rec["index"])
        verdict = verdicts.get(index)
        if verdict == "inaccurate":
            skip("marked inaccurate by an expert")
            continue
        if negative_event:
            kept.append({"rec": rec, "role": "background", "boxes": [], "model_labelled": False, "box_edited": False})
            continue

        matched = _match_child(rec.get("class_name"), taxa)
        taxon = None
        if single is not None and single["verdict"] == "relabel":
            taxon = single
        elif matched is not None:
            taxon = matched
        elif single is not None and single["verdict"] == "confirm" and verdict == "accurate":
            taxon = single
        elif single is not None and single["verdict"] is None and verdict == "accurate":
            taxon = single

        if taxon is None:
            skip("class not confirmed for this frame")
            continue
        if taxon["verdict"] == "reject":
            skip("species rejected by an expert")
            continue
        if taxon["verdict"] is None and verdict != "accurate":
            skip("model label on a frame no expert marked accurate")
            continue

        boxes, edited = _boxes_for_frame(rec, taxon, state)
        if not boxes:
            skip("no usable box")
            continue
        kept.append({
            "rec": rec,
            "role": "positive",
            "boxes": boxes,
            "model_labelled": taxon["verdict"] is None,
            "box_edited": edited,
        })
    return {"frames": kept, "skipped": skipped}


def _spread(items: list, cap: int) -> list:
    """At most cap items spread evenly along the sequence, always including the first and last."""
    n = len(items)
    if cap <= 0 or n <= cap:
        return list(items)
    if cap == 1:
        return [items[n // 2]]
    positions = sorted({int(round(i * (n - 1) / (cap - 1))) for i in range(cap)})
    return [items[p] for p in positions]


def average_hash(path) -> int:
    """A 64 bit fingerprint of how an image looks, small changes giving a small distance."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow ships with the desktop
        raise FrameSourceError("preparing a dataset needs the Pillow library, which is part of the desktop requirements") from exc
    with Image.open(path) as img:
        small = img.convert("L").resize((8, 8))
        pixels = list(small.getdata())
    mean = sum(pixels) / 64.0
    value = 0
    for pixel in pixels:
        value = (value << 1) | (1 if pixel >= mean else 0)
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def image_size(path) -> tuple:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow ships with the desktop
        raise FrameSourceError("preparing a dataset needs the Pillow library, which is part of the desktop requirements") from exc
    with Image.open(path) as img:
        return img.width, img.height


def thin_frames(
    frames: list,
    event_dir: Path,
    *,
    max_frames: int,
    near_duplicate_hamming: int,
    hasher: Callable[[Path], int] = average_hash,
) -> dict:
    """Keep at most max_frames spread along the track, then drop near-duplicates.

    A frame whose file is missing or unreadable is dropped and counted, never fatal.
    """
    spread = _spread(frames, max_frames)
    kept: list = []
    hashes: list = []
    missing = 0
    duplicates = 0
    for frame in spread:
        path = Path(event_dir) / frame["rec"]["file"]
        try:
            resolved = path.resolve()
            if Path(event_dir).resolve() not in resolved.parents or not resolved.is_file():
                missing += 1
                continue
            value = hasher(resolved)
        except Exception:  # noqa: BLE001 - an unreadable image is skipped, never fatal
            missing += 1
            continue
        if near_duplicate_hamming > 0 and any(hamming(value, other) <= near_duplicate_hamming for other in hashes):
            duplicates += 1
            continue
        hashes.append(value)
        frame = dict(frame)
        frame["path"] = resolved
        kept.append(frame)
    return {"frames": kept, "missing": missing, "near_duplicates": duplicates}
