#!/usr/bin/env python3
"""Download the Porifera Classifier COCO export and build the desktop labels file.

This is a one-time, build-time helper (not a runtime dependency -- the offline
guarantee is untouched). It uses your Roboflow API key to download the dataset
that matches the RF-DETR ONNX you already have, then extracts the authoritative
``category_id -> name`` map from the COCO ``categories`` array and writes
``models/visual/porifera_rfdetr.labels.json`` beside the model. The RF-DETR
decoder in ``audtheia/pipeline/drivers.py`` loads that file verbatim; no code
change and no re-export of the model are needed.

Why the download: an RF-DETR ONNX carries NO class names, and its 366-wide head
uses 1-based IDs with a dummy at index 0 (roboflow/rf-detr#306). The only
trustworthy source for which ID is which species is the COCO annotations for the
SAME dataset version as the model (v12).

Usage (PowerShell / cmd), run from the repo root:

    pip install roboflow
    python scripts\\fetch_porifera_dataset.py --api-key YOUR_ROBOFLOW_KEY

The key can also come from the ROBOFLOW_API_KEY environment variable, so it never
has to be typed on the command line:

    $env:ROBOFLOW_API_KEY = "YOUR_ROBOFLOW_KEY"
    python scripts\\fetch_porifera_dataset.py

The heavy image download can be deleted afterward; only the labels JSON is kept.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

# Defaults for the Official Porifera Classifier. Overridable by flags so this
# script is not hardcoded to one deployment.
DEFAULT_WORKSPACE = "marine-sciences-research-station"
DEFAULT_PROJECT = "official-porifera-classifier-ju8er"
DEFAULT_VERSION = 12
DEFAULT_MODEL = Path("models/visual/porifera_rfdetr.onnx")


def _load_build_module():
    """Import build_porifera_labels.py (its category parser is reused here)."""
    here = Path(__file__).resolve().parent
    mod_path = here / "build_porifera_labels.py"
    spec = importlib.util.spec_from_file_location("build_porifera_labels", mod_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot locate {mod_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_coco_json(root: Path) -> Path | None:
    """Return any one _annotations.coco.json under the downloaded dataset root.

    Every split (train/valid/test) shares the same categories array, so the first
    one found is sufficient.
    """
    matches = sorted(root.rglob("_annotations.coco.json"))
    return matches[0] if matches else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ROBOFLOW_API_KEY"),
        help="Roboflow API key (or set ROBOFLOW_API_KEY). roboflow.com -> Settings -> API keys",
    )
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--version", type=int, default=DEFAULT_VERSION)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="path to the RF-DETR ONNX (used to name/locate the labels file and sanity-check)",
    )
    parser.add_argument(
        "--location",
        type=Path,
        default=Path("data/roboflow_porifera_coco"),
        help="where to download the dataset (images can be deleted afterward)",
    )
    args = parser.parse_args(argv)

    if not args.api_key:
        raise SystemExit(
            "no Roboflow API key. Pass --api-key or set ROBOFLOW_API_KEY. Find it at "
            "roboflow.com -> Settings -> API keys."
        )

    try:
        from roboflow import Roboflow  # noqa: PLC0415 - optional heavy dep
    except ImportError:
        raise SystemExit("roboflow is not installed. Run:  pip install roboflow")

    print(f"downloading {args.project}/{args.version} (COCO) ...")
    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)
    # download() writes into a version-named subfolder; location sets the parent.
    dataset = version.download("coco", location=str(args.location))

    dataset_root = Path(getattr(dataset, "location", args.location))
    coco = _find_coco_json(dataset_root) or _find_coco_json(args.location)
    if coco is None:
        raise SystemExit(
            f"downloaded, but no _annotations.coco.json was found under {dataset_root}. "
            f"Confirm the export format was 'coco'."
        )
    print(f"using categories from {coco}")

    build = _load_build_module()
    mapping = build.load_categories(coco)
    ids = sorted(mapping)
    out_path = args.model.parent / f"{args.model.stem}.labels.json"

    head = build.read_head_width(args.model)
    if head is not None and ids[-1] >= head:
        print(
            f"WARNING: category id {ids[-1]} does not fit a {head}-wide head. This "
            f"export may not match your ONNX -- confirm the model is also v{args.version}."
        )

    import json

    payload = {str(cid): mapping[cid] for cid in ids}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {len(payload)} class names to {out_path}")
    print(f"id range: {ids[0]}..{ids[-1]}  (head width: {head if head else 'unknown'})")
    print("sample:", ", ".join(f"{cid}={mapping[cid]!r}" for cid in ids[:5]))
    print(
        "\nDone. Start a desktop capture and the detection cards will show these "
        "names. You can delete the downloaded images under "
        f"{args.location} if you want the space back."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
