#!/usr/bin/env python3
"""Build the desktop RF-DETR labels file from a Roboflow COCO export.

Background
----------
An RF-DETR ONNX export carries NO class-name metadata, and its ``labels`` head
is sized to ``max_category_id + 1`` using 1-based class IDs (index 0 is a
dummy/no-object slot). See roboflow/rf-detr#306. For the Official Porifera
Classifier v12 the head is 366 wide, but only ~199 of the IDs 1..365 are live
classes; the gaps are historical IDs left behind as classes were renamed and
consolidated across dataset versions. The website's alphabetical class list is
NOT in ID order, so the ONLY authoritative source for ``index -> name`` is the
``categories`` array inside the dataset's COCO ``_annotations.coco.json``.

This script reads that ``categories`` array and writes a sibling labels file
(``<model>.labels.json``) as an ``{id: name}`` map, which
``audtheia.pipeline.drivers._labels_from_file`` loads verbatim. The IDs are kept
exactly as Roboflow numbers them (1-based, gaps allowed) so they line up with the
raw argmax index the RF-DETR decoder emits. Nothing here runs at capture time --
this is a one-time, offline build step on the desktop.

Usage
-----
    python scripts/build_porifera_labels.py \
        --coco path/to/_annotations.coco.json \
        --model models/visual/porifera_rfdetr.onnx

If ``--out`` is omitted the labels file is written next to the model as
``<model-stem>.labels.json`` (the first place drivers.py looks). If ``onnx`` is
installed the head width is read from the model and cross-checked against the
category IDs; a mismatch is reported but does not abort, so you stay in control.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_categories(coco_path: Path) -> dict[int, str]:
    """Return an {id: name} map from a COCO annotations file's categories array.

    Every COCO split (train/valid/test) carries the same categories array, so
    any one of them is sufficient. A category whose name is empty or is the
    Roboflow supercategory root placeholder at id 0 is skipped, so the map holds
    only real, nameable classes.
    """
    try:
        data = json.loads(coco_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"cannot read COCO file {coco_path}: {exc}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{coco_path} is not valid JSON: {exc}")

    cats = data.get("categories")
    if not isinstance(cats, list) or not cats:
        raise SystemExit(
            f"{coco_path} has no 'categories' array; is this a COCO detection "
            f"annotations file (_annotations.coco.json)?"
        )

    mapping: dict[int, str] = {}
    for cat in cats:
        if not isinstance(cat, dict):
            continue
        cid = cat.get("id")
        name = cat.get("name")
        if cid is None or name is None:
            continue
        name = str(name).strip()
        if not name:
            continue
        # Roboflow COCO exports often include an id 0 whose name is the project's
        # supercategory root (e.g. the dataset name). RF-DETR reserves index 0 as
        # the no-object slot, so it never fires; drop it to avoid a misleading
        # label on the dummy channel.
        if int(cid) == 0:
            continue
        mapping[int(cid)] = name
    if not mapping:
        raise SystemExit(f"{coco_path} yielded no usable id->name entries.")
    return mapping


def read_head_width(model_path: Path) -> int | None:
    """Return the RF-DETR labels-head width from the ONNX, or None if unreadable.

    onnx is an optional dependency here; when it is missing the cross-check is
    simply skipped so the labels file can still be built on a machine without it.
    """
    try:
        import onnx  # noqa: PLC0415 - optional, imported lazily
    except ImportError:
        return None
    try:
        model = onnx.load(str(model_path), load_external_data=False)
    except Exception as exc:  # noqa: BLE001 - report and skip the check
        print(f"note: could not read {model_path} for the head-width check: {exc}")
        return None
    for out in model.graph.output:
        if out.name != "labels":
            continue
        dims = out.type.tensor_type.shape.dim
        if len(dims) >= 1:
            last = dims[-1]
            if last.HasField("dim_value") and last.dim_value > 0:
                return int(last.dim_value)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--coco",
        required=True,
        type=Path,
        help="path to a Roboflow COCO _annotations.coco.json (any split)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/visual/porifera_rfdetr.onnx"),
        help="path to the RF-DETR ONNX (used to locate the output and sanity-check)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output labels file (default: <model-stem>.labels.json beside the model)",
    )
    args = parser.parse_args(argv)

    mapping = load_categories(args.coco)
    ids = sorted(mapping)
    max_id = ids[-1]

    out_path = args.out or (args.model.parent / f"{args.model.stem}.labels.json")

    head = read_head_width(args.model)
    if head is not None:
        # RF-DETR uses 1-based IDs with a dummy at 0, so the head must be at least
        # max_id + 1. Equal is the healthy case; larger is fine (extra dead slots);
        # smaller means the export and this dataset version disagree.
        if max_id >= head:
            print(
                f"WARNING: category id {max_id} does not fit a {head}-wide head "
                f"(indices 0..{head - 1}). This COCO export likely does not match "
                f"the ONNX you have -- confirm you downloaded the SAME dataset "
                f"version as the model."
            )
        elif head != max_id + 1:
            print(
                f"note: head is {head} wide but the highest category id is {max_id} "
                f"(expected head {max_id + 1}). This is normal if the model's ID "
                f"space extends past the classes present in this split."
            )

    # Write as a string-keyed {id: name} map; drivers._labels_from_file coerces
    # the keys back to int. A stable key order keeps the file diff-friendly.
    payload = {str(cid): mapping[cid] for cid in ids}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"wrote {len(payload)} class names to {out_path}")
    print(f"id range: {ids[0]}..{max_id}  (head width: {head if head else 'unknown'})")
    print("sample:", ", ".join(f"{cid}={mapping[cid]!r}" for cid in ids[:5]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
