"""Packages for sharing a model or a dataset, made on this computer.

Path: audtheia/training/sharing.py

Nothing here sends anything anywhere. A model package is a zip of the model's own
folder: the model file, its labels, the PyTorch checkpoint or the trained layer,
metrics.json and MODEL_CARD.md, whose front matter a Hugging Face model page
reads. A dataset export is a zip of one layout of a built dataset, COCO with
images for Roboflow and most detection tools, YOLO text, or the sound windows
with their table, always with the class list and the manifest so the selection
and every label's origin travel with it. Packages are written beside the
datasets, outside the data folder, and an existing package is never overwritten.

A model whose licence limits use is packaged only after the person types the
same acknowledgement asked for when it is switched on.
"""

from __future__ import annotations

import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

from audtheia.training import registry

__all__ = ["SharingError", "DATASET_LAYOUTS", "packages_dir", "package_model", "export_dataset"]

DATASET_LAYOUTS = {
    "coco": {"modality": "vision", "folders": ("train", "valid", "test")},
    "yolo_txt": {"modality": "vision", "folders": ("yolo",)},
    "windows": {"modality": "audio", "folders": ("windows",)},
}
ALWAYS = ("classes.json", "manifest.json")


class SharingError(RuntimeError):
    """A package cannot be made, with a reason a person can act on."""

    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


def packages_dir(settings) -> Path:
    """Where packages are written: a packages folder beside the datasets folder."""
    datasets = Path(settings.training_paths()["datasets_dir"])
    if not datasets.is_absolute():
        datasets = Path(settings.repo_root) / datasets
    return datasets.parent / "packages"


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _zip(target: Path, files: list) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    stem, n = target.stem, 1
    while target.exists():
        n += 1
        target = target.with_name(f"{stem}-{n}{target.suffix}")
    partial = target.with_name("." + target.name + ".partial")
    try:
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for source, arcname in files:
                zf.write(source, arcname)
        os.replace(partial, target)
    except BaseException:
        try:
            partial.unlink()
        except OSError:
            pass
        raise
    return target


def package_model(settings, model: Mapping, *, acknowledgement: Optional[str] = None) -> dict:
    """Zip a registered model's folder."""
    from audtheia.training.deploy import ACKNOWLEDGEMENT, acknowledged

    if registry.requires_acknowledgement(model.get("license_spdx")) and not acknowledged(acknowledgement):
        raise SharingError(f"This model is licensed {model['license_spdx']}. Type \"{ACKNOWLEDGEMENT}\" to confirm"
                           " you will share it within those terms.", 428)
    folder = registry.model_folder(settings, model)
    if not (folder / model["model_file"]).is_file():
        raise SharingError(f"The model file is missing from {folder}.", 409)
    files = [(p, f"{model['name']}/{p.relative_to(folder).as_posix()}")
             for p in sorted(folder.rglob("*")) if p.is_file()]
    target = packages_dir(settings) / f"{model['name']}-{_stamp()}.zip"
    target = _zip(target, files)
    return {"path": str(target), "name": target.name, "files": [a for _, a in files],
            "bytes": target.stat().st_size}


def export_dataset(settings, dataset_dir: Path, layout: str) -> dict:
    """Zip one layout of a built dataset with its class list and manifest."""
    import json

    if layout not in DATASET_LAYOUTS:
        raise SharingError("The layout must be coco, yolo_txt or windows.")
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SharingError("That dataset was not found.", 404)
    modality = json.loads(manifest_path.read_text(encoding="utf-8")).get("modality")
    if modality != DATASET_LAYOUTS[layout]["modality"]:
        raise SharingError(f"A {modality} dataset has no {layout} layout.")
    files = []
    for name in ALWAYS:
        if (dataset_dir / name).is_file():
            files.append((dataset_dir / name, f"{dataset_dir.name}/{name}"))
    for folder in DATASET_LAYOUTS[layout]["folders"]:
        for p in sorted((dataset_dir / folder).rglob("*")):
            if p.is_file():
                files.append((p, f"{dataset_dir.name}/{p.relative_to(dataset_dir).as_posix()}"))
    if layout == "windows" and (dataset_dir / "windows.csv").is_file():
        files.append((dataset_dir / "windows.csv", f"{dataset_dir.name}/windows.csv"))
    target = packages_dir(settings) / f"{dataset_dir.name}-{layout}-{_stamp()}.zip"
    target = _zip(target, files)
    return {"path": str(target), "name": target.name, "files": len(files), "bytes": target.stat().st_size}
