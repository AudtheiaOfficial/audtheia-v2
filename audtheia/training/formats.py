"""Writing a planned dataset in the layouts training tools read.

Path: audtheia/training/formats.py

Two layouts are written from the same description of a dataset.

  COCO. One folder per split holding the images and a file named
  _annotations.coco.json. This is the layout the detector trainer reads, with the
  validation folder named valid.

  YOLO text. An images folder and a labels folder per split, one text file per
  image with a line of class and normalised centre and size for each box, and a
  data.yaml naming the classes. This is for people who train a different detector
  elsewhere.

Images are hard-linked to the original when the two folders are on the same drive,
and copied otherwise. The original is never opened for writing, moved or deleted.
A box is clamped to its image so a box that ran past an edge cannot produce an
invalid annotation, and a box that collapses to nothing is dropped.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Iterable

__all__ = [
    "COCO_FOLDER_FOR_SPLIT",
    "place_file",
    "clamp_box",
    "write_coco_split",
    "write_yolo_dataset",
]

COCO_FOLDER_FOR_SPLIT = {"train": "train", "validation": "valid", "test": "test"}


def place_file(source: Path, target: Path, *, link: bool = True) -> str:
    """Put a file in the dataset, by hard link when possible, otherwise by copy.

    Returns "link" or "copy". The source is only ever read.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return "exists"
    if link:
        try:
            os.link(str(source), str(target))
            return "link"
        except (OSError, NotImplementedError):
            pass
    shutil.copy2(str(source), str(target))
    return "copy"


def clamp_box(xyxy: Iterable[float], width: int, height: int):
    """A box restricted to the image, or None when nothing of it is left."""
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0.0, min(x1, float(width)))
    x2 = max(0.0, min(x2, float(width)))
    y1 = max(0.0, min(y1, float(height)))
    y2 = max(0.0, min(y2, float(height)))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return x1, y1, x2, y2


def write_coco_split(split_dir: Path, images: list, categories: list, *, description: str = "") -> dict:
    """Write one split's annotation file next to its images.

    images is a list of dictionaries with file_name, width, height and boxes, where
    each box has category_id and xyxy. categories is a list with id and name. An image
    with no boxes is kept as a background example. Returns the counts written.
    """
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    coco_images = []
    coco_annotations = []
    next_annotation = 1
    dropped = 0
    for image_id, image in enumerate(sorted(images, key=lambda im: im["file_name"]), start=1):
        coco_images.append({
            "id": image_id,
            "file_name": image["file_name"],
            "width": int(image["width"]),
            "height": int(image["height"]),
        })
        for box in image.get("boxes", []):
            clamped = clamp_box(box["xyxy"], image["width"], image["height"])
            if clamped is None:
                dropped += 1
                continue
            x1, y1, x2, y2 = clamped
            w, h = x2 - x1, y2 - y1
            coco_annotations.append({
                "id": next_annotation,
                "image_id": image_id,
                "category_id": int(box["category_id"]),
                "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                "area": round(w * h, 2),
                "iscrowd": 0,
            })
            next_annotation += 1
    payload = {
        "info": {"description": description or "Audtheia training split"},
        "categories": [{"id": int(c["id"]), "name": c["name"], "supercategory": "organism"} for c in categories],
        "images": coco_images,
        "annotations": coco_annotations,
    }
    (split_dir / "_annotations.coco.json").write_text(
        json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return {"images": len(coco_images), "annotations": len(coco_annotations), "dropped_boxes": dropped}


def write_yolo_dataset(target: Path, images_by_split: dict, class_names: list, *, link: bool = True) -> dict:
    """Write the YOLO text layout.

    images_by_split maps a split name to a list of dictionaries with source (the
    image file), file_name, width, height and boxes, where each box has class_index,
    the zero based position in class_names, and xyxy. Returns counts per split.
    """
    target = Path(target)
    counts: dict = {}
    folder = {"train": "train", "validation": "val", "test": "test"}
    for split, images in images_by_split.items():
        name = folder[split]
        written = 0
        boxes_written = 0
        for image in sorted(images, key=lambda im: im["file_name"]):
            place_file(Path(image["source"]), target / "images" / name / image["file_name"], link=link)
            lines = []
            for box in image.get("boxes", []):
                clamped = clamp_box(box["xyxy"], image["width"], image["height"])
                if clamped is None:
                    continue
                x1, y1, x2, y2 = clamped
                cx = ((x1 + x2) / 2.0) / image["width"]
                cy = ((y1 + y2) / 2.0) / image["height"]
                w = (x2 - x1) / image["width"]
                h = (y2 - y1) / image["height"]
                lines.append(f"{int(box['class_index'])} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            label = target / "labels" / name / (Path(image["file_name"]).stem + ".txt")
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")
            written += 1
            boxes_written += len(lines)
        counts[split] = {"images": written, "boxes": boxes_written}
    names_block = "\n".join(f"  {i}: {json.dumps(n)}" for i, n in enumerate(class_names))
    (target / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n" + names_block + "\n",
        encoding="utf-8", newline="\n",
    )
    return counts
