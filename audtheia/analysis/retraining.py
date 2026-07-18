"""Desktop retraining exports.

Path: audtheia/analysis/retraining.py

A monitoring system earns its accuracy back over time by learning from the cases
it handled badly. This module gathers those cases out of the record and writes
them into a folder a person can feed straight into training, without hunting
through the database or the detection folders by hand.

Two exports are produced, one per modality.

  The visual export collects the frames behind weak or disputed detections and
  writes them with their bounding boxes in the annotation format a labelling
  tool reads directly. The boxes are the model's own guesses, so they are a
  starting point to correct rather than ground truth, and the export says so.

  The acoustic export collects the stored clips and sorts them into one folder
  per recognised label, which is the layout an acoustic classifier expects for
  training. The labels are the model's own guesses too.

What makes a case worth exporting:

  A low-confidence detection is one the model nearly missed. These are the
  highest-value training examples, and they are where an organism the model was
  never trained on tends to surface, wearing the name of whatever it most
  resembled.

  A disputed detection is one where the desktop verifier reached a different
  conclusion from the field station. Those disagreements are already recorded,
  and each one is a labelled example of a mistake worth correcting.

  A deferred record is one the field station could not classify at all.

Nothing here is a new measurement or an interpretation. Every row written is a
copy of a value already stored, and every export carries a manifest saying which
observation each file came from, so a corrected label can always be traced back
to the event that produced it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

__all__ = [
    "RetrainingExportError",
    "candidate_summary",
    "export_vision",
    "export_acoustic",
]

logger = logging.getLogger("audtheia.analysis.retraining")

# The default confidence below which a detection is treated as weak enough to be
# worth re-labelling. A deployment can pass its own value; this is only a
# starting point, chosen to sit above a typical screening floor so that genuinely
# marginal calls are caught without sweeping in every ordinary detection.
DEFAULT_CONFIDENCE_BELOW = 0.45

REASON_LOW_CONFIDENCE = "low_confidence"
REASON_DISAGREEMENT = "desktop_disagreed"
REASON_DEFERRED = "qc_deferred"


class RetrainingExportError(RuntimeError):
    """Raised when an export cannot be written, with a reason a person can act on."""


def _label_of(detection: dict) -> str:
    """The name a detection carries, preferring the scientific name when present."""
    return (detection.get("scientific_name") or detection.get("common_name") or "unlabelled").strip()


def _safe_name(value: str) -> str:
    """A version of a label safe to use as a folder or file name on any platform."""
    cleaned = "".join("_" if ch in '<>:"/\\|?*' else ch for ch in (value or "")).strip()
    cleaned = cleaned.strip(". ")
    return cleaned or "unlabelled"


def _exports_root(settings) -> Path:
    """Where exports are written.

    Read from the configuration when it names a folder, so a deployment can put
    exports on an external drive beside its data, and defaulting to a folder in
    the project otherwise, so an older configuration still works unchanged.
    """
    configured = (settings.raw.get("paths", {}) or {}).get("exports_dir") or "exports"
    return Path(settings.resolve_path(configured))


def _resolve_media(settings, stored_path: str) -> Optional[Path]:
    """Turn a stored media path into a real file, or nothing when it is missing.

    Paths are stored relative to the project, so they are resolved the same way
    every other configured path is. A file that has been moved or cleaned away is
    reported as missing rather than silently skipped.
    """
    if not stored_path:
        return None
    candidate = Path(stored_path)
    if not candidate.is_absolute():
        candidate = Path(settings.repo_root) / candidate
    return candidate if candidate.is_file() else None


def _verification_disagreed(db, observation_id: str) -> bool:
    verdict = db.get_observation_verification(observation_id)
    return bool(verdict) and verdict.get("rfdetr_agrees_with_field") == 0


def _collect(db, *, station_id, confidence_below, include_disagreements, include_deferred, modality):
    """Gather every candidate detection of one modality, with why it was chosen.

    Returns a list of (observation, detection, reasons) so the caller can both
    count candidates and export them from the same selection, which keeps the
    preview a person sees and the export they get in step with each other.
    """
    selected = []
    for obs in db.list_observations(station_id=station_id):
        deferred = include_deferred and obs.get("qc_state") == "qc_deferred"
        disagreed = include_disagreements and _verification_disagreed(db, obs["id"])
        for det in db.list_child_detections(obs["id"]):
            if (det.get("modality") or "vision") != modality:
                continue
            reasons = []
            confidence = det.get("confidence")
            if confidence is not None and confidence_below is not None and confidence < confidence_below:
                reasons.append(REASON_LOW_CONFIDENCE)
            if disagreed:
                reasons.append(REASON_DISAGREEMENT)
            if deferred:
                reasons.append(REASON_DEFERRED)
            if reasons:
                selected.append((obs, det, reasons))
    return selected


def candidate_summary(db, *, station_id=None, confidence_below=DEFAULT_CONFIDENCE_BELOW,
                      include_disagreements=True, include_deferred=True) -> dict:
    """How many detections would be exported, and why, without writing anything.

    This is what the interface shows before a person commits to an export, so the
    numbers below are produced by exactly the same selection the export uses.
    """
    out = {"confidence_below": confidence_below, "vision": {}, "acoustic": {}}
    for modality, key in (("vision", "vision"), ("audio", "acoustic")):
        rows = _collect(db, station_id=station_id, confidence_below=confidence_below,
                        include_disagreements=include_disagreements,
                        include_deferred=include_deferred, modality=modality)
        reasons: dict = {}
        labels: dict = {}
        for _obs, det, why in rows:
            for reason in why:
                reasons[reason] = reasons.get(reason, 0) + 1
            label = _label_of(det)
            labels[label] = labels.get(label, 0) + 1
        out[key] = {
            "detections": len(rows),
            "events": len({obs["id"] for obs, _d, _w in rows}),
            "by_reason": reasons,
            "by_label": labels,
        }
    return out


def _new_export_dir(settings, prefix: str) -> Path:
    """A fresh folder for one export, named by the moment it was written.

    The stamp counts in seconds, so two exports asked for in the same second
    would otherwise land on the same name. A short suffix is added in that case
    rather than failing or writing into the earlier folder.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = _exports_root(settings)
    target = root / f"{prefix}-{stamp}"
    suffix = 2
    while target.exists():
        target = root / f"{prefix}-{stamp}-{suffix}"
        suffix += 1
    target.mkdir(parents=True, exist_ok=False)
    return target


def _selection_signature(kind: str, rows) -> str:
    """A fingerprint of exactly which detections an export would contain.

    Two exports of the same detections produce the same fingerprint, which is how
    a repeat of an export that has already been written is recognised without
    comparing folders file by file.
    """
    ids = sorted(str(det.get("id") or "") for _obs, det, _why in rows)
    return hashlib.sha256(("|".join([kind] + ids)).encode("utf-8")).hexdigest()


def _existing_export(settings, signature: str) -> Optional[Path]:
    """An earlier export holding exactly these detections, if one is already there."""
    root = _exports_root(settings)
    if not root.is_dir():
        return None
    for marker in sorted(root.glob("*/selection.json")):
        try:
            recorded = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if recorded.get("signature") == signature:
            return marker.parent
    return None


def _write_selection(target: Path, kind: str, signature: str, rows, confidence_below) -> None:
    """Record what this package contains, so a later repeat can be recognised."""
    payload = {
        "kind": kind,
        "signature": signature,
        "detections": len(rows),
        "confidence_below": confidence_below,
        "written_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (target / "selection.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def _image_size(path: Path):
    """The pixel size of a stored frame, needed by the annotation format."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - pillow ships with the desktop
        raise RetrainingExportError(
            "reading image sizes needs the Pillow library, which is part of the desktop requirements"
        ) from exc
    with Image.open(path) as img:
        return img.width, img.height


def export_vision(db, settings, *, station_id=None, confidence_below=DEFAULT_CONFIDENCE_BELOW,
                  include_disagreements=True, include_deferred=True, force=False) -> dict:
    """Write the visual retraining package and return a summary of what it holds."""
    rows = _collect(db, station_id=station_id, confidence_below=confidence_below,
                    include_disagreements=include_disagreements,
                    include_deferred=include_deferred, modality="vision")
    if not rows:
        raise RetrainingExportError("no visual detections match these settings, so there is nothing to export")

    # Writing the same package twice leaves two folders holding identical work and
    # makes it unclear which one a person has already corrected, so a repeat is
    # reported instead, unless it is asked for deliberately.
    signature = _selection_signature("vision", rows)
    if not force:
        previous = _existing_export(settings, signature)
        if previous is not None:
            return {"kind": "vision", "already_exists": True, "path": str(previous),
                    "detections": len(rows), "confidence_below": confidence_below}

    target = _new_export_dir(settings, "vision-retraining")
    images_dir = target / "images"
    images_dir.mkdir()

    annotations = [("filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax")]
    manifest = [("image", "observation_id", "event_name", "station_id", "first_seen",
                 "label", "confidence", "reasons", "screening_model_version")]
    copied: dict = {}
    missing = 0

    for obs, det, reasons in rows:
        source = _resolve_media(settings, obs.get("representative_frame") or "")
        if source is None:
            missing += 1
            continue
        # One frame can carry several detections, so it is copied once and gets
        # one annotation row per detection on it.
        if source not in copied:
            name = f"{_safe_name(obs.get('event_name') or obs['id'])}__{source.name}"
            shutil.copy2(str(source), str(images_dir / name))
            copied[source] = name
        filename = copied[source]

        try:
            width, height = _image_size(source)
        except Exception:  # noqa: BLE001 - an unreadable frame is skipped, never fatal
            logger.warning("could not read the size of %s; skipping its annotation", source)
            continue

        x, y = det.get("bbox_x"), det.get("bbox_y")
        w, h = det.get("bbox_w"), det.get("bbox_h")
        if None not in (x, y, w, h):
            # Stored boxes are pixel coordinates. They are clamped to the frame so
            # a box that ran past an edge cannot produce an invalid annotation.
            xmin = max(0, min(int(round(x)), width))
            ymin = max(0, min(int(round(y)), height))
            xmax = max(xmin + 1, min(int(round(x + w)), width))
            ymax = max(ymin + 1, min(int(round(y + h)), height))
            annotations.append((filename, width, height, _label_of(det), xmin, ymin, xmax, ymax))

        manifest.append((filename, obs["id"], obs.get("event_name") or "", obs.get("station_id") or "",
                         obs.get("first_seen") or "", _label_of(det),
                         "" if det.get("confidence") is None else round(float(det["confidence"]), 4),
                         " ".join(reasons), obs.get("screening_model_version") or "not stated"))

    _write_csv(target / "annotations.csv", annotations)
    _write_csv(target / "manifest.csv", manifest)
    (target / "README.md").write_text(_VISION_README, encoding="utf-8", newline="\n")
    _write_selection(target, "vision", signature, rows, confidence_below)

    return {
        "kind": "vision",
        "already_exists": False,
        "path": str(target),
        "images": len(copied),
        "annotations": len(annotations) - 1,
        "detections": len(manifest) - 1,
        "missing_media": missing,
        "confidence_below": confidence_below,
    }


def export_acoustic(db, settings, *, station_id=None, confidence_below=DEFAULT_CONFIDENCE_BELOW,
                    include_disagreements=True, include_deferred=True, force=False) -> dict:
    """Write the acoustic retraining package and return a summary of what it holds."""
    rows = _collect(db, station_id=station_id, confidence_below=confidence_below,
                    include_disagreements=include_disagreements,
                    include_deferred=include_deferred, modality="audio")
    if not rows:
        raise RetrainingExportError("no acoustic detections match these settings, so there is nothing to export")

    signature = _selection_signature("acoustic", rows)
    if not force:
        previous = _existing_export(settings, signature)
        if previous is not None:
            return {"kind": "acoustic", "already_exists": True, "path": str(previous),
                    "detections": len(rows), "confidence_below": confidence_below}

    target = _new_export_dir(settings, "acoustic-retraining")
    clips_dir = target / "clips"
    clips_dir.mkdir()

    manifest = [("clip", "label", "observation_id", "event_name", "station_id", "first_seen",
                 "confidence", "reasons", "acoustic_model_version")]
    written = 0
    missing = 0
    labels: dict = {}

    for obs, det, reasons in rows:
        source = _resolve_media(settings, obs.get("audio_clip_path") or "")
        if source is None:
            missing += 1
            continue
        label = _label_of(det)
        # One folder per label is the layout an acoustic classifier trains from.
        folder = clips_dir / _safe_name(label)
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{_safe_name(obs.get('event_name') or obs['id'])}{source.suffix}"
        destination = folder / name
        if not destination.exists():
            shutil.copy2(str(source), str(destination))
            written += 1
        labels[label] = labels.get(label, 0) + 1
        manifest.append((str(Path("clips") / _safe_name(label) / name), label, obs["id"],
                         obs.get("event_name") or "", obs.get("station_id") or "",
                         obs.get("first_seen") or "",
                         "" if det.get("confidence") is None else round(float(det["confidence"]), 4),
                         " ".join(reasons), obs.get("acoustic_model_version") or "not stated"))

    _write_csv(target / "manifest.csv", manifest)
    (target / "README.md").write_text(_ACOUSTIC_README, encoding="utf-8", newline="\n")
    _write_selection(target, "acoustic", signature, rows, confidence_below)

    return {
        "kind": "acoustic",
        "already_exists": False,
        "path": str(target),
        "clips": written,
        "labels": len(labels),
        "detections": len(manifest) - 1,
        "missing_media": missing,
        "confidence_below": confidence_below,
    }


def _write_csv(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


_VISION_README = """# Visual retraining package

This folder holds the frames behind the detections your system handled least
well, together with the boxes it drew on them. It is a starting point for
correction, not a finished dataset.

## What is inside

- `images/` contains one copy of each frame, named after the event it came from.
- `annotations.csv` lists one row per detection: the file, its pixel size, the
  label the model chose, and the box corners. Most labelling tools import this
  format directly.
- `manifest.csv` records where every row came from: the observation, the station,
  the time, the confidence, and why the detection was selected.

## Read this before you train

The labels and boxes in here are the model's own guesses. They were exported
precisely because they are doubtful, so treat every one as a question. Correct
the wrong labels, fix the loose boxes, and delete anything that is not really
there. A detection that turns out to be an organism the model has never been
trained on is the most valuable item in this folder: give it a new label, and it
becomes something the next model can recognise.

## Suggested workflow

1. Create a project in your labelling tool and upload the `images/` folder.
2. Import `annotations.csv` so the existing boxes come in with the images.
3. Review every image. Correct labels, adjust boxes, add anything that was
   missed, and remove false detections.
4. Train a new version of the model from the corrected set.

## Putting a retrained model back to work

The desktop verification model is exported as ONNX and pointed at from the
settings, under the desktop model path.

The field station's screening model is different. After training, it has to be
compiled to the accelerator's own format before a station can run it. That
compile step runs on an x86 Linux machine with the accelerator's compiler
toolchain, and it is done once when you prepare the model, not by the station
while it is deployed. The station only ever loads the finished file.
"""


_ACOUSTIC_README = """# Acoustic retraining package

This folder holds the recorded clips behind the acoustic detections your system
was least sure about, sorted into one folder per label.

## What is inside

- `clips/` contains one folder per recognised label, holding the clips that were
  matched to it. This is the layout an acoustic classifier expects to train from,
  so you can point a trainer at this folder directly.
- `manifest.csv` records where every clip came from: the observation, the
  station, the time, the confidence, and why it was selected.

## Read this before you train

The folder names are the labels the model chose, and every clip in here was
exported because the match was weak. Listen before you train. Move a clip into
the right folder when it was filed wrongly, and delete clips that hold no call at
all. If a label is not an animal, for example an engine or another human sound,
keeping those clips in their own folder is still useful: a classifier that learns
what your site's background noise sounds like raises fewer false detections.

## Suggested workflow

1. Listen through each folder and correct anything filed in the wrong place.
2. Add more examples of your local species if you have them. A classifier
   improves fastest on the calls it hears most often at your own site.
3. Train a custom classifier from the corrected folders.
4. Place the finished classifier in the custom acoustic model folder and select
   it for the station in the settings.
"""
