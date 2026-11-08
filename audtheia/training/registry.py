"""Turning a finished run into a named, numbered model.

Path: audtheia/training/registry.py

A run that finished becomes a model only when a person adds it. Its files are
copied out of the run folder into a folder of its own under the models directory,
named Audtheia-<Name>-V<n>, with a model card, the test figures and the licence it
inherits. The number is issued by the database inside one write, so two models of
one name never share a number, a failed run never takes one, and a model folder is
never overwritten: a new fine-tune is a new version.

What travels with every model

  The model file and its labels, the PyTorch checkpoint for an image model so it
  can be trained further or uploaded elsewhere, the trained layer and calibration
  for a sound model, metrics.json with the test figures and the comparison with the
  model that was in use when it was scored, and MODEL_CARD.md. The licence is
  inherited from what the model was trained from. When it cannot be known, as for a
  person's own starting checkpoint, it is recorded as not confirmed rather than
  guessed.

Nothing here writes outside the models directory and the registry tables.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Mapping, Optional

from audtheia.storage.database import format_model_name, validate_model_user_name
from audtheia.training import evaluate as ev

__all__ = [
    "RegistryError",
    "MODEL_CARD",
    "METRICS",
    "NON_COMMERCIAL",
    "register_run",
    "model_folder",
    "licence_for",
    "requires_acknowledgement",
    "metrics_summary",
]

MODEL_CARD = "MODEL_CARD.md"
METRICS = "metrics.json"
NON_COMMERCIAL = ("CC-BY-NC-SA-4.0", "CC-BY-NC-4.0", "CC-BY-NC-ND-4.0")
_SPDX = {"CC BY-NC-SA 4.0": "CC-BY-NC-SA-4.0", "CC BY-NC 4.0": "CC-BY-NC-4.0", "CC BY 4.0": "CC-BY-4.0",
         "Apache-2.0": "Apache-2.0", "Apache 2.0": "Apache-2.0", "MIT": "MIT"}
_HF_LICENCE = {"Apache-2.0": "apache-2.0", "CC-BY-NC-SA-4.0": "cc-by-nc-sa-4.0", "CC-BY-NC-4.0": "cc-by-nc-4.0",
               "CC-BY-4.0": "cc-by-4.0", "MIT": "mit"}
_LOCK = threading.Lock()

VISION_FILES = (("export/model.onnx", "model.onnx", "model"), ("export/labels.json", "labels.json", "labels"),
                ("train_output/checkpoint_best_total.pth", "checkpoint.pth", "checkpoint"))
ACOUSTIC_FILES = (("export/model.tflite", "model.tflite", "model"), ("export/labels.txt", "labels.txt", "labels"),
                  ("export/labels.json", "labels.json", None), ("export/head.npz", "head.npz", "checkpoint"),
                  ("export/calibration.json", "calibration.json", None))


class RegistryError(RuntimeError):
    """A run cannot become a model, with a reason a person can act on. status is the HTTP status."""

    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


def _relative(settings, path: Path) -> str:
    root = Path(settings.repo_root).resolve()
    path = Path(path).resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def model_folder(settings, row: Mapping) -> Path:
    """The folder of a registered model, from its stored repository-relative path."""
    path = Path(row["folder"])
    return path if path.is_absolute() else Path(settings.repo_root) / path


def requires_acknowledgement(spdx: Optional[str]) -> bool:
    """Whether a licence limits use, so a person types an acknowledgement before using or sharing it."""
    return spdx in NON_COMMERCIAL


def licence_for(run: Mapping, config: Mapping) -> dict:
    """The licence a model inherits from what it was trained from."""
    if run["modality"] == "acoustic":
        base = config.get("base") or {}
        text = str(base.get("licence") or "").strip()
        spdx = _SPDX.get(text)
        label = base.get("label") or run.get("base_label")
        note = (f"The final layer was trained on top of the {label} base model, which is published under {text}."
                if text else f"The licence of the {label} base model is not recorded in the settings.")
        if spdx in NON_COMMERCIAL:
            note += (" The joined model may be used for non-commercial purposes only, with attribution, and shared"
                     " only under the same terms.")
        return {"license_spdx": spdx, "license_note": note, "base_license_note": text or None}
    start = config.get("start_from") or {"kind": "pretrained"}
    label = config.get("base_label") or run.get("base_label")
    if start.get("kind") == "checkpoint":
        name = Path(str(start.get("path") or "")).name
        return {"license_spdx": None, "base_license_note": None,
                "license_note": (f"Fine-tuned from your own checkpoint {name}. Its licence follows that checkpoint's"
                                 " terms, which Audtheia cannot know. Confirm them before sharing the model.")}
    if start.get("kind") == "scratch":
        return {"license_spdx": None, "base_license_note": None,
                "license_note": ("Trained from no pretrained weights with the RF-DETR training code, which is"
                                 " Apache-2.0. The weights carry the terms of the training images, which you state.")}
    return {"license_spdx": "Apache-2.0", "base_license_note": "Apache-2.0",
            "license_note": (f"Fine-tuned from the RF-DETR {label} pretrained weights, published under Apache-2.0."
                             " The terms of the training images are yours to state.")}


def metrics_summary(run_dir: Path, modality: str) -> dict:
    """The test figures kept with a model, or a plain statement that it was not tested."""
    from audtheia.training import acoustic as TA
    from audtheia.training import metrics as M

    record = ev.read_evaluation(run_dir)
    challenger = (record or {}).get("challenger")
    if challenger is None and modality == "acoustic":
        own = Path(run_dir) / "export" / TA.TEST_METRICS_NAME
        if own.is_file():
            challenger = json.loads(own.read_text(encoding="utf-8"))
    if not challenger:
        return {"tested": False, "headline": None, "comparison": None, "current_model": None, "box_origin": None,
                "note": ("This model has no test score. Its dataset held no test examples, or it has not been scored"
                         " on its test set yet. Score it in Results before relying on it.")}
    comparison = (record or {}).get("comparison") or {}
    return {
        "tested": True,
        "headline": M.headline(challenger),
        "test_counts": challenger.get("counts"),
        "class_names": challenger.get("class_names"),
        "comparison": {k: comparison.get(k) for k in ("rows", "shared_rows", "shared_classes", "not_in_current",
                                                     "note", "unknown_note")} if comparison else None,
        "current_model": _shareable_model((record or {}).get("current_model")),
        "box_origin": (record or {}).get("box_origin"),
        "scored_with": (record or {}).get("scored_with"),
        "note": None,
    }


def _shareable_model(entry: Optional[Mapping]) -> Optional[dict]:
    """The model a run was compared with, named by its file alone.

    The full path of that file belongs to this computer, and can name a person's
    account or drive, so it is kept in the run folder and never in a model's own
    folder, which is the one that is packaged and shared.
    """
    if not entry:
        return None
    out = {k: entry.get(k) for k in ("label", "version")}
    path = str(entry.get("path") or "").strip()
    out["file"] = path.replace("\\", "/").rsplit("/", 1)[-1] if path else None
    return out


def _fmt(value: Any) -> str:
    return "not defined" if value is None else (f"{value:.3f}" if isinstance(value, float) else str(value))


def model_card(row: Mapping, run: Mapping, metrics: Mapping, manifest: Mapping, licence: Mapping) -> str:
    """A model card readable on its own and ready for a Hugging Face model page."""
    hf = _HF_LICENCE.get(licence.get("license_spdx") or "", "other")
    lines = ["---", f"license: {hf}", "library_name: " + ("onnx" if row["modality"] == "vision" else "tflite"),
             "tags:", "- audtheia", "- biodiversity", "- " + ("object-detection" if row["modality"] == "vision"
                                                               else "audio-classification"),
             "---", "", f"# {row['name']}", "",
             f"{'An image detection' if row['modality'] == 'vision' else 'A sound classification'} model trained with"
             f" Audtheia from events that experts reviewed. Base: {row['base_label']}.", "",
             "## Licence", "", licence.get("license_note") or "Not recorded.", "",
             "## Training data", "",
             f"- Dataset {manifest.get('dataset_id')}, rules version {manifest.get('rules_version')}.",
             f"- Classes: {row['class_count']}.",
             "- Events are divided whole into training, validation and test, so no event appears in two of them.",
             "- The licence of the training data is: (state it here before sharing).", "",
             "## Test figures", ""]
    if metrics.get("tested"):
        for key, value in (metrics.get("headline") or {}).items():
            if key != "kind":
                lines.append(f"- {key}: {_fmt(value)}")
        if metrics.get("current_model"):
            lines.append(f"- Compared on the same test set with {metrics['current_model'].get('label')}; the figures"
                         " for both are in metrics.json.")
        if (metrics.get("box_origin") or {}).get("note"):
            lines.append(f"- {metrics['box_origin']['note']}")
    else:
        lines.append(metrics.get("note") or "Not tested.")
    lines += ["", "## Limits", "",
              "- A figure from one test split is an estimate; a class with few test events gives a rough one.",
              "- The same animal can appear in separate events close in time, which splitting by event does not"
              " remove.", "",
              f"Created {row['created_at']} from training run {run['id']}."]
    return "\n".join(lines) + "\n"


def register_run(settings, db, run_id: str, *, created_by: str) -> dict:
    """Copy a finished run's model into its own folder and issue its registry entry."""
    run = db.get_training_run(run_id)
    if run is None:
        raise RegistryError("There is no training run with that identifier.", 404)
    if run["status"] != "completed":
        raise RegistryError("Only a run that finished can become a model.", 409)
    if any(m["run_id"] == run_id for m in db.list_trained_models()):
        raise RegistryError("This run is already a model.", 409)
    run_dir = Path(settings.repo_root) / run["run_dir"] if not Path(run["run_dir"]).is_absolute() else Path(run["run_dir"])
    config = json.loads(run.get("config_json") or "{}")
    files = VISION_FILES if run["modality"] == "vision" else ACOUSTIC_FILES
    for source, _dest, role in files:
        if role in ("model", "labels") and not (run_dir / source).is_file():
            raise RegistryError(f"The run folder is missing {source}, so the model cannot be added.")
    dataset_dir = Path(settings.resolve_path(run["dataset_dir"]))
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8")) \
        if (dataset_dir / "manifest.json").is_file() else {}
    classes = json.loads((dataset_dir / "classes.json").read_text(encoding="utf-8")) \
        if (dataset_dir / "classes.json").is_file() else []
    user_name = validate_model_user_name(run["run_name"])
    licence = licence_for(run, config)
    metrics = metrics_summary(run_dir, run["modality"])
    models_dir = Path(settings.training_paths()["models_dir"])
    if not models_dir.is_absolute():
        models_dir = Path(settings.repo_root) / models_dir
    with _LOCK:
        name = format_model_name(user_name, db.next_model_version(user_name))
        target = models_dir / name
        if target.exists():
            raise RegistryError(f"A folder named {name} already exists in the models folder and is never overwritten.",
                                409)
        partial = models_dir / f".{name}.partial"
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        try:
            names = {}
            for source, dest, role in files:
                src = run_dir / source
                if src.is_file():
                    shutil.copy2(src, partial / dest)
                    if role:
                        names[role] = dest
            row = db.register_trained_model(
                user_name=user_name, modality=run["modality"], engine=run["engine"], run_id=run_id,
                folder=_relative(settings, target), model_file=names["model"], labels_file=names["labels"],
                checkpoint_file=names.get("checkpoint"), base_label=run["base_label"],
                metrics_json=json.dumps(metrics, default=float), dataset_signature=run["dataset_signature"],
                class_count=len(classes), created_by=created_by, **licence)
            if row["name"] != name:
                raise RegistryError("Another model of this name was added at the same moment. Try again.", 409)
            (partial / METRICS).write_text(json.dumps(metrics, indent=1, default=float) + "\n", encoding="utf-8",
                                           newline="\n")
            (partial / MODEL_CARD).write_text(model_card(row, run, metrics, manifest, licence), encoding="utf-8",
                                              newline="\n")
            os.replace(partial, target)
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise
    return row
