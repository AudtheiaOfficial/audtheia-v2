"""Switching a registered model on, and back off.

Path: audtheia/training/deploy.py

A model is switched on by writing it into one of three slots of the settings:

  desktop_verifier    the desktop verification model, desktop_models.visual_rfdetr
  desktop_screening   one station's desktop screening model, models.visual_desktop
  station_acoustic    one station's sound model, models.acoustic

The slot is written through the app's guarded settings writer, so the whole
configuration is validated before anything is saved, and the model slots are kept
in the local settings file: the committed settings never change because a model
was switched. Every activation and revert is appended to the activation log with
the slot exactly as it was before and after, so a revert restores it exactly. A
record already captured keeps the model version it was captured with.

A model whose licence limits use, such as a sound model built on BirdNET, is
switched on only after a person types the acknowledgement.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Callable, Mapping, Optional

from audtheia.training import registry

__all__ = ["DeployError", "SLOTS", "ACKNOWLEDGEMENT", "slot_entry", "activate", "revert", "slot_state"]

SLOTS = {
    "desktop_verifier": {"modality": "vision", "station": False, "label": "Desktop verification model"},
    "desktop_screening": {"modality": "vision", "station": True, "label": "Desktop screening model of a station"},
    "station_acoustic": {"modality": "acoustic", "station": True, "label": "Sound model of a station"},
}
ACKNOWLEDGEMENT = "I accept the licence terms"


class DeployError(RuntimeError):
    """A model cannot be switched on or off, with a reason a person can act on."""

    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


def _station(raw: dict, station_id: Optional[str]) -> dict:
    for station in raw.get("stations") or []:
        if station.get("station_id") == station_id:
            return station
    raise DeployError("There is no station with that identifier.", 404)


def slot_entry(raw: dict, slot: str, station_id: Optional[str]) -> dict:
    """The live dictionary of a slot inside a configuration, created empty when absent."""
    if slot not in SLOTS:
        raise DeployError("The slot must be the desktop verifier, a station's desktop screening model, or a"
                          " station's sound model.")
    if SLOTS[slot]["station"] and not station_id:
        raise DeployError("Choose the station this model is for.")
    if slot == "desktop_verifier":
        return raw.setdefault("desktop_models", {}).setdefault("visual_rfdetr", {})
    station = _station(raw, station_id)
    key = "visual_desktop" if slot == "desktop_screening" else "acoustic"
    return station.setdefault("models", {}).setdefault(key, {})


def _model_path(settings, model: Mapping, name: str) -> str:
    folder = registry.model_folder(settings, model)
    path = (folder / name).resolve()
    try:
        return path.relative_to(Path(settings.repo_root).resolve()).as_posix()
    except ValueError:
        return str(path)


def _new_entry(settings, model: Mapping, slot: str, previous: Mapping, run: Optional[Mapping]) -> dict:
    entry = dict(previous)
    entry["path"] = _model_path(settings, model, model["model_file"])
    entry["version"] = model["name"]
    entry["citation"] = (f"{model['name']}, trained with Audtheia from {model['base_label']}."
                         + (f" {model['license_note']}" if model.get("license_note") else ""))
    if slot == "station_acoustic":
        entry["labels_path"] = _model_path(settings, model, model["labels_file"])
        config = json.loads((run or {}).get("config_json") or "{}")
        if config.get("sample_rate"):
            entry["sample_rate"] = int(config["sample_rate"])
        if config.get("window_seconds"):
            entry["window_seconds"] = float(config["window_seconds"])
        entry["output_key"] = None
    return entry


def activate(settings, db, model_id: str, slot: str, *, station_id: Optional[str] = None,
             acknowledgement: Optional[str] = None, activated_by: str, persist: Callable[[dict], None]) -> dict:
    """Switch a registered model on in a slot. persist(draft) validates and saves the configuration."""
    model = db.get_trained_model(model_id)
    if model is None:
        raise DeployError("There is no model with that identifier.", 404)
    if SLOTS.get(slot, {}).get("modality") not in (None, model["modality"]):
        raise DeployError(f"A {model['modality']} model cannot go into the {SLOTS[slot]['label'].lower()}.")
    folder = registry.model_folder(settings, model)
    if not (folder / model["model_file"]).is_file():
        raise DeployError(f"The model file is missing from {folder}.", 409)
    acknowledged = registry.requires_acknowledgement(model.get("license_spdx"))
    if acknowledged and (acknowledgement or "").strip() != ACKNOWLEDGEMENT:
        raise DeployError(f"This model is licensed {model['license_spdx']}. Type \"{ACKNOWLEDGEMENT}\" to confirm"
                          " you will use it within those terms.", 428)
    draft = copy.deepcopy(settings.raw)
    entry = slot_entry(draft, slot, station_id)
    previous = copy.deepcopy(entry)
    new = _new_entry(settings, model, slot, previous, db.get_training_run(model["run_id"]))
    entry.clear()
    entry.update(new)
    persist(draft)
    return db.add_model_activation(model_id, slot=slot, action="activate", station_id=station_id,
                                   previous_json=json.dumps(previous, sort_keys=True),
                                   new_json=json.dumps(new, sort_keys=True), activated_by=activated_by,
                                   acknowledged_license=acknowledged)


def revert(settings, db, slot: str, *, station_id: Optional[str] = None, activated_by: str,
           persist: Callable[[dict], None]) -> dict:
    """Put a slot back exactly as it was before the change now in effect."""
    history = db.model_activations(slot, station_id)
    if not history:
        raise DeployError("Nothing was switched on in this slot from the Train tab, so there is nothing to revert.",
                          409)
    latest = history[0]
    if latest["action"] != "activate":
        raise DeployError("The last change in this slot was already a revert. Switch a model on to change it again.",
                          409)
    draft = copy.deepcopy(settings.raw)
    entry = slot_entry(draft, slot, station_id)
    current = copy.deepcopy(entry)
    if json.dumps(current, sort_keys=True) != latest["new_json"]:
        raise DeployError("This slot was changed elsewhere after the model was switched on, so it is not reverted"
                          " automatically. Check it in Settings.", 409)
    restored = json.loads(latest["previous_json"])
    entry.clear()
    entry.update(restored)
    persist(draft)
    return db.add_model_activation(latest["model_id"], slot=slot, action="revert", station_id=station_id,
                                   previous_json=json.dumps(current, sort_keys=True),
                                   new_json=json.dumps(restored, sort_keys=True), activated_by=activated_by)


def slot_state(settings, db, slot: str, station_id: Optional[str] = None) -> dict:
    """What a slot holds now, and whether a Train-tab change in it can be reverted."""
    entry = copy.deepcopy(slot_entry(copy.deepcopy(settings.raw), slot, station_id))
    history = db.model_activations(slot, station_id)
    latest = history[0] if history else None
    revertible = bool(latest and latest["action"] == "activate"
                      and json.dumps(entry, sort_keys=True) == latest["new_json"])
    return {"slot": slot, "station_id": station_id, "label": SLOTS[slot]["label"], "entry": entry,
            "model_id": latest["model_id"] if revertible else None,
            "revertible": revertible, "history": history}
