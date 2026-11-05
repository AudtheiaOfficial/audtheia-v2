"""Scoring a trained model and the model in use on the same test set.

Path: audtheia/training/evaluate.py

A trained model is only worth comparing with the model it might replace when both
are scored on the same held-out events, by the same decoding and the same metric
code. This module does that on the desktop, after a run has finished.

  Images. Both models are run through the decoding the app itself uses during
  capture, at a low confidence floor so the full precision and recall curves can be
  drawn, and scored with metrics.evaluate_detections on the dataset's test split.

  Sounds. Both models are run through the station's own acoustic adapter on the
  dataset's test windows and scored with metrics.evaluate_acoustic. The trained
  model's own test figures, with its calibration, come from the run itself.

Classes are matched by taxon identity, through the same resolver the dataset
builder uses, so the model in use is credited with a species however its labels
spell it. A species the model in use cannot name counts as never found by it in the
first comparison, because that is what it would do in the field. The second
comparison keeps only the species both models know, which is the fair like-for-like
reading. Boxes the model in use draws for species outside the dataset are counted
and reported, never silently dropped. The app prints the numbers and their
differences; it never says which model is better.

Nothing here writes outside the run folder.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from audtheia.training import charts_data, metrics as M

__all__ = [
    "EvaluationError",
    "CONFIDENCE_FLOOR",
    "EVALUATION_DIR",
    "EVALUATION_FILE",
    "EXAMPLES_PER_KIND",
    "current_candidates",
    "class_mapping",
    "evaluate_vision",
    "evaluate_acoustic",
    "write_evaluation",
    "read_evaluation",
]

CONFIDENCE_FLOOR = 0.01
EVALUATION_DIR = "evaluation"
EVALUATION_FILE = "evaluation.json"
EXAMPLES_PER_KIND = 6
COCO_FILE = "_annotations.coco.json"
TEST_FOLDER = "test"
SCORED_WITH_VISION = (
    "Both models were run through the decoding the app uses during capture, keeping boxes from a confidence of"
    f" {CONFIDENCE_FLOOR} so the full curves can be drawn, and scored by the same code on the same test images."
)
SCORED_WITH_ACOUSTIC = (
    "Both models were run through the station's own acoustic adapter on the same test windows and scored by the same"
    " code."
)
BOX_ORIGIN_NOTE = (
    "{model} of the {total} test boxes were drawn by a model and only checked by an expert. When the model in use"
    " drew them, it is being scored against its own boxes, which favours it. Only boxes an expert drew or adjusted are"
    " free of that bias."
)
UNKNOWN_NOTE = (
    "A species the model in use cannot name counts as never found by it in the first comparison, because that is what it"
    " would do in the field. The second comparison keeps only the species both models know."
)


class EvaluationError(RuntimeError):
    """A model cannot be scored, with a reason a person can act on."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Which models are in use
# ---------------------------------------------------------------------------


def _resolved(settings, value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(settings.repo_root) / path


def current_candidates(settings, modality: str) -> list:
    """The models in use that a trained model can be compared with.

    For images these are the desktop verifier and each station's desktop screening
    model; for sounds, each station's acoustic model. A file used in several places
    is listed once with every place named. Each entry says whether its file exists.
    """
    found: dict = {}

    def add(cid, label, value, entry, station=None):
        path = _resolved(settings, value)
        key = str(path.resolve()) if path.exists() else str(path)
        if key in found:
            found[key]["used_by"].append(label)
            return
        found[key] = {"id": cid, "label": label, "used_by": [label], "path": str(path), "exists": path.exists(),
                      "version": (entry or {}).get("version"), "entry": dict(entry or {}), "station": station}

    if modality == "vision":
        desk = (settings.raw.get("desktop_models") or {}).get("visual_rfdetr") or {}
        if desk.get("path"):
            add("desktop:visual_rfdetr", "Desktop verifier", desk["path"], desk)
        for station in settings.stations():
            entry = settings.desktop_visual_model(station)
            if entry.get("path"):
                add(f"station:{station['station_id']}:visual_desktop",
                    f"{station.get('station_name')} screening model", entry["path"], entry)
    elif modality in ("acoustic", "audio"):
        for station in settings.stations():
            entry = (station.get("models") or {}).get("acoustic") or {}
            if entry.get("path"):
                add(f"station:{station['station_id']}:acoustic", f"{station.get('station_name')} sound model",
                    entry["path"], entry, station=station)
    else:
        raise EvaluationError("modality must be vision or acoustic")
    out = list(found.values())
    for row in out:
        row["label"] = ", ".join(row["used_by"])
    return out


# ---------------------------------------------------------------------------
# Matching a model's classes to the dataset's
# ---------------------------------------------------------------------------


def class_mapping(model_names: Mapping, dataset_keys: Sequence[str], resolver) -> dict:
    """For each model class index, the dataset class index it names, or None.

    Every model label goes through the same identity rules as the dataset, so a
    label and a dataset class match when they are the same taxon.
    """
    position = {key: i for i, key in enumerate(dataset_keys)}
    out = {}
    for index, label in model_names.items():
        text = str(label or "").strip()
        if not text:
            out[int(index)] = None
            continue
        from audtheia.training.candidates import taxon_key

        got = resolver.resolve(taxon_key(None, text), text, None)
        out[int(index)] = position.get(got["class_key"])
    return out


def _dataset_classes(dataset_dir: Path) -> list:
    path = dataset_dir / "classes.json"
    if not path.is_file():
        raise EvaluationError(f"the dataset at {dataset_dir} has no class list, so it cannot be scored")
    classes = json.loads(path.read_text(encoding="utf-8"))
    return sorted(classes, key=lambda c: int(c["class_index"]))


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def load_detector(path: Path, entry: Optional[Mapping] = None):
    """A detector over an ONNX model with the app's own decoding and a low confidence floor."""
    from audtheia.pipeline import drivers as D

    path = Path(path)
    if not path.is_file():
        raise EvaluationError(f"the model file was not found at {path}")
    if path.suffix.lower() == ".hef":
        raise EvaluationError("a compiled .hef runs only on a field station's accelerator and cannot be scored here")
    ort = D._import_onnxruntime()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rfdetr = D._looks_like_rfdetr(session)
    names = {} if rfdetr else D._class_names_from_session(session)
    if not names:
        names = D._class_names_from_config(dict(entry or {}), path)
    if not names and rfdetr:
        from audtheia.inference.rfdetr_onnx import _class_names_from_session as rf_names

        names = rf_names(session)
    if not names:
        raise EvaluationError(
            f"the model at {path} carries no class names and has no labels file beside it, so its boxes cannot be"
            " matched to species")
    width, height = D.RFDETR_INPUT_SIZE if rfdetr else D.DEFAULT_INPUT_SIZE
    shape = session.get_inputs()[0].shape
    if isinstance(shape, (list, tuple)) and len(shape) == 4:
        if isinstance(shape[3], int):
            width = shape[3]
        if isinstance(shape[2], int):
            height = shape[2]
    cls = D.OnnxRfDetrDetector if rfdetr else D.OnnxYoloDetector
    return cls(session, class_names=names, input_size=(width, height), conf_threshold=CONFIDENCE_FLOOR,
               iou_threshold=D.DEFAULT_NMS_IOU)


def _test_images(dataset_dir: Path, n_classes: int) -> tuple:
    coco_path = dataset_dir / TEST_FOLDER / COCO_FILE
    if not coco_path.is_file():
        raise EvaluationError("the dataset has no test split file, so nothing can be scored")
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    by_image: dict = {img["id"]: [] for img in coco.get("images") or []}
    for ann in coco.get("annotations") or []:
        x, y, w, h = (float(v) for v in ann["bbox"])
        by_image.setdefault(ann["image_id"], []).append((int(ann["category_id"]) - 1, [x, y, x + w, y + h]))
    files = {img["id"]: img["file_name"] for img in coco.get("images") or []}
    gt = {}
    for image_id, name in files.items():
        boxes = by_image.get(image_id) or []
        for cls, _ in boxes:
            if not (0 <= cls < n_classes):
                raise EvaluationError(f"the test split names a class outside the dataset's {n_classes} classes")
        gt[name] = boxes
    return gt


def box_origin(dataset_dir: Path) -> dict:
    """How many test boxes a model proposed and how many an expert drew or adjusted."""
    path = Path(dataset_dir) / "manifest.json"
    counts = {"model": 0, "expert": 0}
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for item in manifest.get("items") or []:
            if item.get("split") != "test":
                continue
            for box in item.get("boxes") or []:
                counts["model" if (box.get("box_source") or "model") == "model" else "expert"] += 1
    total = counts["model"] + counts["expert"]
    note = BOX_ORIGIN_NOTE.format(model=counts["model"], total=total) if counts["model"] else None
    return {"model_proposed": counts["model"], "expert_drawn": counts["expert"], "total": total, "note": note}


def _predict(detector, dataset_dir: Path, names: Sequence[str], mapping: Mapping, progress, label: str) -> tuple:
    import numpy as np
    from PIL import Image

    from audtheia.pipeline.monitor import Frame

    preds: dict = {}
    outside = 0
    total = len(names)
    for done, name in enumerate(names, start=1):
        with Image.open(dataset_dir / TEST_FOLDER / name) as img:
            image = np.asarray(img.convert("RGB"))
        rows = []
        for det in detector.detect(Frame(index=0, image=image, captured_at=_utc_now())):
            target = mapping.get(int(det.class_id))
            if target is None:
                outside += 1
                continue
            rows.append((int(target), float(det.confidence), [det.x1, det.y1, det.x2, det.y2]))
        preds[name] = rows
        if progress:
            progress(label, done, total)
    return preds, outside


def _known(mapping: Mapping) -> list:
    return sorted({v for v in mapping.values() if v is not None})


def _comparison(challenger, current, challenger_shared, current_shared, names, known, extra) -> dict:
    """The side-by-side table, with the species the model in use cannot name taken from its classes."""
    if current is None:
        table = charts_data.comparison_table(challenger, None)
    else:
        table = charts_data.comparison_table(challenger, current, challenger_shared=challenger_shared,
                                             current_shared=current_shared)
        known_names = [names[i] for i in known]
        table["not_in_current"] = [n for n in names if n not in set(known_names)]
        table["shared_classes"] = known_names
        table["not_in_challenger"] = []
        table["unknown_note"] = UNKNOWN_NOTE
    table.update(extra)
    return table


def evaluate_vision(dataset_dir, challenger_path, current: Optional[Mapping] = None, *, resolver,
                    load: Optional[Callable] = None, progress: Optional[Callable] = None,
                    options: Optional[Mapping] = None, seed: Any = 0) -> dict:
    """Score the trained detector, and the model in use when one is given, on the test images.

    current is an entry from current_candidates, or None when there is nothing to
    compare with. options is the evaluation block of the training settings: the IoU
    thresholds, the most boxes per image, the reporting confidence and how many
    example images of each kind to keep. load builds a detector from a path and a
    settings entry, and is replaced in tests.
    """
    options = dict(options or {})
    report_confidence = float(options.get("report_confidence", M.DEFAULT_REPORT_CONFIDENCE))
    score = {"iou_thresholds": tuple(options.get("iou_thresholds") or M.DEFAULT_IOU_THRESHOLDS),
             "max_detections": int(options.get("max_detections", M.DEFAULT_MAX_DETECTIONS)),
             "conf_threshold": report_confidence}
    per_kind = int(options.get("example_images_per_kind", EXAMPLES_PER_KIND))
    load = load or load_detector
    dataset_dir = Path(dataset_dir)
    classes = _dataset_classes(dataset_dir)
    names = [str(c["class_name"]) for c in classes]
    keys = [str(c["class_key"]) for c in classes]
    gt = _test_images(dataset_dir, len(names))
    if not gt:
        raise EvaluationError("the dataset's test split holds no images, so nothing can be scored")
    files = sorted(gt)

    detector = load(Path(challenger_path), None)
    try:
        own = {int(i): i for i in range(len(names))}
        pred, beyond = _predict(detector, dataset_dir, files, own, progress, "trained model")
    finally:
        detector.close()
    challenger = M.evaluate_detections(gt, pred, names, include_image_outcomes=True, **score)
    outcomes = challenger.pop("image_outcomes")
    chosen = M.select_examples(outcomes, per_kind, seed)
    examples = {kind: [{"file": f"{TEST_FOLDER}/{f}", "truth": [{"class": names[c], "xyxy": b} for c, b in gt[f]],
                        "predicted": [{"class": names[c], "score": s, "xyxy": b} for c, s, b in pred[f]
                                      if s >= report_confidence]} for f in files_]
                for kind, files_ in chosen.items()}

    record = {"kind": "detection", "created_at": _utc_now(), "dataset_dir": str(dataset_dir),
              "test_images": len(files), "scored_with": SCORED_WITH_VISION, "challenger": challenger,
              "examples": examples, "current_model": None, "current": None, "challenger_shared": None,
              "current_shared": None}
    origin = box_origin(dataset_dir)
    record["box_origin"] = origin
    extra = {"challenger_boxes_beyond_dataset": beyond, "box_origin": origin}
    if current is None:
        record["comparison"] = _comparison(challenger, None, None, None, names, [], extra)
        return record

    model = load(Path(current["path"]), current.get("entry"))
    try:
        mapping = class_mapping(model.class_names, keys, resolver)
        cur_pred, outside = _predict(model, dataset_dir, files, mapping, progress, "model in use")
    finally:
        model.close()
    current_metrics = M.evaluate_detections(gt, cur_pred, names, **score)
    known = _known(mapping)
    shared_c = shared_u = None
    if known:
        g, p, n = M.restrict_detection(gt, pred, names, known)
        shared_c = M.evaluate_detections(g, p, n, **score)
        g, p, n = M.restrict_detection(gt, cur_pred, names, known)
        shared_u = M.evaluate_detections(g, p, n, **score)
    extra.update({"current_boxes_outside_dataset": outside, "current_classes": len(mapping)})
    record.update({
        "current_model": {"label": current.get("label"), "path": current.get("path"), "version": current.get("version")},
        "current": current_metrics, "challenger_shared": shared_c, "current_shared": shared_u,
        "comparison": _comparison(challenger, current_metrics, shared_c, shared_u, names, known, extra),
    })
    return record


# ---------------------------------------------------------------------------
# Sounds
# ---------------------------------------------------------------------------


def _window_scores(model, dataset_dir: Path, items: Sequence[Mapping], mapping: Mapping, n_classes: int,
                   progress, label: str):
    import numpy as np

    from audtheia.training import windows as window_rules
    from audtheia.training.acoustic import fit_length, resample

    rate = int(model.sample_rate)
    length = int(round(float(model.window_seconds) * rate))
    out = np.zeros((len(items), n_classes), dtype=np.float64)
    for row, item in enumerate(items):
        raw, source_rate = window_rules.read_wav(dataset_dir / item["file"])
        audio = fit_length(resample(raw, source_rate, rate), length)
        for det in model.detect(audio, rate):
            target = mapping.get(int(det.class_id))
            if target is not None:
                out[row, target] = max(out[row, target], float(det.confidence))
        if progress:
            progress(label, row + 1, len(items))
    return out


def load_acoustic(path: Path, entry: Mapping, *, labels_path=None, settings=None, station=None):
    """The station's own acoustic adapter over a model file."""
    from audtheia.pipeline import acoustic as A

    if station is not None and settings is not None:
        return A.build_acoustic_model(station, settings)
    return A.TFLiteAcousticModel(model_path=Path(path), sample_rate=int(entry["sample_rate"]),
                                 window_seconds=float(entry["window_seconds"]), labels_path=labels_path)


def evaluate_acoustic(dataset_dir, export_dir, run_config: Mapping, current: Optional[Mapping] = None, *, resolver,
                      settings=None, load: Optional[Callable] = None, progress: Optional[Callable] = None) -> dict:
    """Score the trained sound model, and the model in use when one is given, on the test windows."""
    from audtheia.training import acoustic as TA

    load = load or load_acoustic
    dataset_dir = Path(dataset_dir)
    export_dir = Path(export_dir)
    names, _keys, _tested, splits = TA.dataset_windows(dataset_dir)
    keys = [str(c["class_key"]) for c in _dataset_classes(dataset_dir)]
    items = splits.get("test") or []
    threshold = float(run_config.get("report_confidence", M.DEFAULT_REPORT_CONFIDENCE))
    own_path = export_dir / TA.TEST_METRICS_NAME
    own = json.loads(own_path.read_text(encoding="utf-8")) if own_path.is_file() else None
    record = {"kind": "acoustic", "created_at": _utc_now(), "dataset_dir": str(dataset_dir),
              "test_windows": len(items), "scored_with": SCORED_WITH_ACOUSTIC, "challenger": own,
              "current_model": None, "current": None, "challenger_shared": None, "current_shared": None,
              "comparison": None, "examples": None}
    if not items:
        record["note"] = ("The dataset holds no test windows, so the model has no test score. Confirm more events,"
                          " at least three per species, and build the dataset again.")
        return record
    y = TA.label_matrix(items, len(names))
    model = load(export_dir / TA.EXPORT_NAME, {"sample_rate": run_config["sample_rate"],
                                               "window_seconds": run_config["window_seconds"]},
                 labels_path=export_dir / TA.LABELS_TEXT_NAME)
    try:
        own_scores = _window_scores(model, dataset_dir, items, {i: i for i in range(len(names))}, len(names),
                                    progress, "trained model")
    finally:
        model.close()
    challenger = M.evaluate_acoustic(y, own_scores, names, threshold=threshold)
    if record["challenger"] is None:
        record["challenger"] = challenger
    if current is None:
        record["comparison"] = _comparison(challenger, None, None, None, names, [], {})
        return record
    model = load(Path(current["path"]), current.get("entry") or {}, settings=settings, station=current.get("station"))
    try:
        mapping = class_mapping(model.class_names, keys, resolver)
        cur_scores = _window_scores(model, dataset_dir, items, mapping, len(names), progress, "model in use")
    finally:
        model.close()
    current_metrics = M.evaluate_acoustic(y, cur_scores, names, threshold=threshold)
    known = _known(mapping)
    shared_c = shared_u = None
    if known:
        a, b, n = M.restrict_acoustic(y, own_scores, names, known)
        shared_c = M.evaluate_acoustic(a, b, n, threshold=threshold)
        a, b, n = M.restrict_acoustic(y, cur_scores, names, known)
        shared_u = M.evaluate_acoustic(a, b, n, threshold=threshold)
    record.update({
        "current_model": {"label": current.get("label"), "path": current.get("path"), "version": current.get("version")},
        "current": current_metrics, "challenger_shared": shared_c, "current_shared": shared_u,
        "comparison": _comparison(challenger, current_metrics, shared_c, shared_u, names, known,
                                  {"current_classes": len(mapping)}),
    })
    return record


# ---------------------------------------------------------------------------
# Storing the result in the run folder
# ---------------------------------------------------------------------------


def write_evaluation(run_dir, record: Mapping) -> Path:
    """Write the evaluation and one CSV per chart beside it, inside the run folder only."""
    folder = Path(run_dir) / EVALUATION_DIR
    folder.mkdir(parents=True, exist_ok=True)
    body = json.dumps(record, indent=1, default=float, allow_nan=False) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=".evaluation.", suffix=".tmp", dir=str(folder))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(body)
        os.replace(tmp, folder / EVALUATION_FILE)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    if record.get("challenger"):
        bundle = charts_data.results_bundle(record["challenger"])
        charts_data.write_chart_csvs(bundle["charts"], folder / "csv")
    return folder / EVALUATION_FILE


def read_evaluation(run_dir) -> Optional[dict]:
    path = Path(run_dir) / EVALUATION_DIR / EVALUATION_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
