"""Which desktop model verifies a station's frames, decided in one place.

Path: audtheia/inference/visual_verifiers.py

The desktop can hold two visual verifiers: an RF-DETR detector in
desktop_models.visual_rfdetr, trained for one group of organisms such as the
Porifera classifier, and BioCLIP 2 in desktop_models.visual_bioclip, which names
organisms across the tree of life. A station may say which one checks its frames
with models.desktop_verifier. When it does not, RF-DETR is used if it is
configured and BioCLIP 2 otherwise, so a station that relied on RF-DETR before
keeps it.

The orchestrator that verifies events and the server that words verdicts both
ask this module, so a station's events are never checked by one model and
described as checked by another.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

__all__ = ["SLOTS", "choice", "configured", "build", "label_set", "label_sets", "version"]

SLOTS = ("visual_rfdetr", "visual_bioclip")


def _slot_ready(settings, slot: str) -> bool:
    if slot == "visual_bioclip":
        from audtheia.inference.bioclip_onnx import bioclip_configured

        return bioclip_configured(settings)
    entry = (settings.raw.get("desktop_models", {}) or {}).get(slot) or {}
    path = entry.get("path")
    if not path:
        return False
    resolved = Path(path) if Path(path).is_absolute() else Path(settings.repo_root) / path
    return resolved.is_file()


def choice(settings, station: Optional[dict]) -> Optional[str]:
    """The slot that verifies this station's frames, or None when none is ready.

    With no station, the first ready slot is given, for views that describe the
    desktop as a whole.
    """
    wanted = ((station or {}).get("models") or {}).get("desktop_verifier")
    if wanted in SLOTS:
        return wanted if _slot_ready(settings, wanted) else None
    for slot in SLOTS:
        if _slot_ready(settings, slot):
            return slot
    return None


def configured(settings, station: Optional[dict] = None) -> bool:
    return choice(settings, station) is not None


def version(settings, slot: Optional[str]) -> Optional[str]:
    if not slot:
        return None
    entry = (settings.raw.get("desktop_models", {}) or {}).get(slot) or {}
    if entry.get("version"):
        return entry["version"]
    if slot == "visual_bioclip":
        return "BioCLIP 2"
    path = entry.get("path")
    return Path(path).stem if path else None


def build(settings, station: Optional[dict]):
    """The verifier for this station's frames. Raises when the chosen model cannot load."""
    slot = choice(settings, station)
    if slot == "visual_bioclip":
        from audtheia.inference.bioclip_onnx import build_bioclip_verifier

        return build_bioclip_verifier(settings)
    if slot == "visual_rfdetr":
        from audtheia.inference.rfdetr_onnx import build_verifier

        return build_verifier(settings)
    raise RuntimeError("no desktop verification model is configured for this station")


def label_set(settings, slot: Optional[str]) -> tuple:
    """(version, sorted labels or None) of a verifier, for saying when a taxon is outside its labels."""
    if slot == "visual_bioclip":
        from audtheia.inference.bioclip_onnx import bioclip_label_set

        return bioclip_label_set(settings)
    if slot == "visual_rfdetr":
        from audtheia.inference.rfdetr_onnx import verifier_label_set

        return verifier_label_set(settings)
    return None, None


def label_sets(settings) -> dict:
    """{version: labels} for every visual verifier configured now, labels None when unreadable.

    A verdict names the model that wrote it, so a reader picks the label set by
    the verdict's own version rather than assuming one model.
    """
    out: dict = {}
    for slot in SLOTS:
        if not _slot_ready(settings, slot):
            continue
        try:
            found_version, labels = label_set(settings, slot)
        except Exception:  # noqa: BLE001 - a label set is a disclosure aid only
            found_version, labels = version(settings, slot), None
        out[found_version or version(settings, slot)] = labels
    return out
