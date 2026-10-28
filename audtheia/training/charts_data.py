"""Chart series, tables and plain-language help for the Results sub-panel.

Path: audtheia/training/charts_data.py

Turns the figures produced by audtheia.training.metrics into exactly what the
interface draws and exports: for every chart its title, axes, series, points and
a plain-language help line, and for the same chart the rows of its CSV file.
Nothing here draws anything and nothing here picks a colour. Each series carries
a stable position so the interface can assign colours from the theme's
colour-blind-safe series tokens, and the saved PNG is produced from the same
drawing the person sees on screen.

Wording rule. The application prints numbers and their differences and never
the words better or improved, because a single test split cannot establish
that. The text in this module follows the same rule, and the tests scan it.

Standard library only.
"""

from __future__ import annotations

import csv
import io
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import metrics as M

CHARTS_DEFINITION_VERSION = 1

HONESTY_LINE = (
    "These figures come from events the model never saw in training, validation or "
    "early stopping. Frames of one event are never split across sets. A figure is "
    "an estimate from the events in the test set, and a class with only a few test "
    "events gives a rough one."
)

CALIBRATION_NOTE = (
    "Scores from these models are unitless, not probabilities. Per-class logistic "
    "calibration on the validation split, following Wood and Kahl (2024, Journal of "
    "Ornithology 165:777-782), turns a score into the probability that a call is "
    "present. The threshold shown for each class is the score at which that "
    "probability reaches the target. Thresholds are saved with the model and shown "
    "here. They are not applied automatically when the model runs, because the "
    "station's acoustic adapter has no threshold input."
)

CALIBRATION_STATUS_TEXT = {
    "ok": "Calibrated",
    "too_few_positives": "Not calibrated. Too few labelled examples of this class in the validation split.",
    "too_few_negatives": "Not calibrated. Too few windows without this class in the validation split.",
    "no_variation": "Not calibrated. Every validation score was the same.",
    "not_increasing": "Not calibrated. In the validation split the score did not rise with the presence of the class.",
}

# Plain-language help for every metric and chart. Keys are stable identifiers.
METRIC_HELP = {
    "map50": (
        "mAP at IoU 0.50",
        "Mean average precision when a prediction counts as correct if its box overlaps a "
        "labelled box by at least half (IoU 0.50). Averaged over classes that have labelled "
        "boxes in the test set. 1.0 means every box was found with no false alarms ranked above it.",
    ),
    "map50_95": (
        "mAP over IoU 0.50 to 0.95",
        "The COCO standard. Average precision averaged over ten overlap thresholds from 0.50 to "
        "0.95, so it rewards boxes that fit tightly as well as boxes that are in the right place.",
    ),
    "precision": (
        "Precision",
        "Of the boxes the model reported at or above the reporting confidence, the fraction that "
        "matched a labelled box of the same class. Not defined when nothing was reported.",
    ),
    "recall": (
        "Recall",
        "Of the labelled boxes, the fraction the model found at or above the reporting confidence "
        "with the right class.",
    ),
    "f1": (
        "F1",
        "A single figure that is high only when precision and recall are both high, computed as "
        "twice the matches divided by twice the matches plus the false alarms plus the misses.",
    ),
    "top1_accuracy_on_matched_boxes": (
        "Top-1 class accuracy on matched boxes",
        M.TOP1_DEFINITION,
    ),
    "cmap": (
        "Class-mean average precision",
        "Average precision of each class, averaged over classes that have at least one labelled "
        "window in the test set. Average precision summarises the whole precision-recall curve, so "
        "it does not depend on one threshold.",
    ),
    "macro_roc_auc": (
        "Macro ROC area",
        "For each class, the chance that a window containing it scores higher than a window "
        "without it, averaged over classes that have both kinds of window.",
    ),
    "top1_accuracy": (
        "Top-1 accuracy",
        M.ACOUSTIC_TOP1_DEFINITION,
    ),
    "macro_precision": (
        "Macro precision",
        "Precision of each class averaged over the classes that have labelled examples and made at "
        "least one call at the reporting threshold.",
    ),
    "macro_recall": (
        "Macro recall",
        "Recall of each class averaged over the classes that have labelled examples.",
    ),
    "macro_f1": (
        "Macro F1",
        "F1 of each class averaged over the classes that have labelled examples. A class that was "
        "never found counts as 0, so rare classes weigh as much as common ones.",
    ),
    "loss_by_epoch": (
        "Loss by epoch",
        "How far the model's output is from the labels, per pass over the training data. Training "
        "loss should fall. If validation loss starts to rise while training loss keeps falling the "
        "model is memorising the training events.",
    ),
    "map_by_epoch": (
        "Validation score by epoch",
        "The validation score after each epoch. The checkpoint with the highest value is the one "
        "that is kept.",
    ),
    "ap_by_iou": (
        "Average precision by overlap threshold",
        "Average precision over all classes at each box-overlap threshold. It falls as the threshold "
        "rises. A steep fall means the boxes are in the right place but do not fit tightly.",
    ),
    "f1_vs_confidence": (
        "F1 against confidence",
        "F1 at every confidence cut-off. The marker shows the cut-off with the highest F1 for the "
        "mean of all classes. Pick a cut-off near the top of the curve to balance false alarms and misses.",
    ),
    "precision_vs_confidence": (
        "Precision against confidence",
        "Precision at every confidence cut-off. It usually rises as the cut-off rises.",
    ),
    "recall_vs_confidence": (
        "Recall against confidence",
        "Recall at every confidence cut-off. It always falls as the cut-off rises.",
    ),
    "precision_recall": (
        "Precision-recall curve",
        "Precision against recall as the cut-off is lowered. A curve that stays near the top right "
        "corner is a model that finds most things without many false alarms. The area under it is "
        "the class's average precision.",
    ),
    "roc": (
        "ROC curve",
        "The share of windows containing a class that are found against the share of windows without "
        "it that are wrongly flagged, as the threshold is lowered. The diagonal is no skill.",
    ),
    "f1_vs_threshold": (
        "F1 against threshold",
        "F1 at every score threshold. The curve for a class peaks at the threshold that balances its "
        "false alarms and misses.",
    ),
    "per_class_ap": (
        "Average precision per class",
        "How well each class is found, from 0 to 1. A class with no bar has no labelled examples in the "
        "test set, so it was not tested.",
    ),
    "confusion_matrix": (
        "Confusion matrix",
        "Rows are the labelled truth, columns are what the model reported. The last row holds reports "
        "with nothing labelled there, the false alarms. The last column holds labelled items the model "
        "did not report, the misses. Counts on the diagonal are correct.",
    ),
    "reliability": (
        "Reliability diagram",
        "Scores are grouped into bins. For each bin the point shows the average score against the "
        "fraction of cases that were truly present. Points on the dotted diagonal mean a score of 0.8 "
        "is right about 80 percent of the time.",
    ),
    "calibration_threshold": (
        "Calibrated threshold",
        "The score at which the fitted probability that a call is present reaches the target. "
        "Calls scoring above it are expected to be right at least that often.",
    ),
}

# Fixed lists of the figures compared between two models.
_DETECTION_ROWS = ("map50", "map50_95", "macro_precision", "macro_recall", "macro_f1",
                   "top1_accuracy_on_matched_boxes")
_ACOUSTIC_ROWS = ("cmap", "macro_roc_auc", "macro_precision", "macro_recall", "macro_f1", "top1_accuracy")

COMPARISON_NOTE = (
    "Both models were scored on the same test set. The difference column is the challenger minus "
    "the model currently in use. Figures are printed as measured. The estimates rest on the number "
    "of test events, so small differences can be noise."
)

_ID_RE = re.compile(r"^[a-z0-9_]+$")


class ChartsError(ValueError):
    """The metrics handed to this module were not in the expected shape."""


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _help(key: str) -> str:
    return METRIC_HELP[key][1]


def _title(key: str) -> str:
    return METRIC_HELP[key][0]


def _check_chart_id(chart_id: str) -> str:
    if not _ID_RE.match(chart_id):
        raise ChartsError(f"chart id {chart_id!r} must be lower-case letters, digits and underscores")
    return chart_id


def _series(index: int, key: str, label: str, points: Sequence, *, role: str = "class",
            dashed: bool = False, visible: bool = True) -> dict:
    return {"index": index, "key": key, "label": label, "role": role, "dashed": dashed,
            "default_visible": visible, "points": [[x, y] for x, y in points]}


def _line_chart(chart_id: str, help_key: str, x_label: str, y_label: str, series: list, *,
                x_range: tuple = (0.0, 1.0), y_range: tuple | None = (0.0, 1.0), title: str | None = None,
                markers: list | None = None, note: str | None = None, omitted: int = 0,
                extra: Mapping | None = None) -> dict:
    chart = {
        "id": _check_chart_id(chart_id),
        "kind": "line",
        "title": title or _title(help_key),
        "help": _help(help_key),
        "x": {"label": x_label, "min": x_range[0], "max": x_range[1]},
        "y": {"label": y_label, "min": None if y_range is None else y_range[0],
              "max": None if y_range is None else y_range[1]},
        "series": series,
        "markers": markers or [],
        "note": note,
        "series_omitted": omitted,
    }
    if extra:
        chart.update(extra)
    chart["csv"] = {
        "columns": ["series", "x", "y"],
        "rows": [[s["label"], x, y] for s in series for x, y in s["points"]],
    }
    return chart


def _bar_chart(chart_id: str, help_key: str, categories: Sequence[str], series: list, y_label: str, *,
               note: str | None = None, omitted: int = 0) -> dict:
    chart = {
        "id": _check_chart_id(chart_id),
        "kind": "bar",
        "title": _title(help_key),
        "help": _help(help_key),
        "categories": list(categories),
        "y": {"label": y_label, "min": 0.0, "max": 1.0},
        "series": series,
        "note": note,
        "series_omitted": omitted,
    }
    chart["csv"] = {
        "columns": ["class"] + [s["label"] for s in series],
        "rows": [[cat] + [s["values"][i] for s in series] for i, cat in enumerate(categories)],
    }
    return chart


def _matrix_chart(chart_id: str, help_key: str, labels: Sequence[str], matrix: Sequence[Sequence[int]], *,
                  row_title: str, col_title: str, note: str | None = None) -> dict:
    fractions = []
    for row in matrix:
        total = sum(row)
        fractions.append([(v / total if total else None) for v in row])
    chart = {
        "id": _check_chart_id(chart_id),
        "kind": "matrix",
        "title": _title(help_key),
        "help": _help(help_key),
        "row_labels": list(labels),
        "col_labels": list(labels),
        "row_title": row_title,
        "col_title": col_title,
        "values": [list(r) for r in matrix],
        "row_fractions": fractions,
        "note": note,
    }
    chart["csv"] = {
        "columns": [row_title, col_title, "count", "fraction of row"],
        "rows": [[labels[i], labels[j], matrix[i][j], fractions[i][j]]
                 for i in range(len(labels)) for j in range(len(labels))],
    }
    return chart


def _order_by_support(support: Sequence[int]) -> list:
    """Class indices, the best supported first, ties by index."""
    return sorted(range(len(support)), key=lambda c: (-support[c], c))


def _visible_set(support: Sequence[int], max_visible: int) -> set:
    return {c for c in _order_by_support(support)[:max_visible] if support[c] > 0}


def _pairs(xs: Sequence, ys: Sequence) -> list:
    return list(zip(xs, ys))


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

_PROGRESS_LOSS = (("train_loss", "Training loss"), ("val_loss", "Validation loss"))
_PROGRESS_SCORE = (("map50", "mAP at IoU 0.50"), ("map50_95", "mAP over IoU 0.50 to 0.95"),
                   ("cmap", "Class-mean average precision"), ("macro_f1", "Macro F1"))


def training_curve_charts(progress_rows: Sequence[Mapping]) -> list:
    """Loss and validation-score charts from the worker's progress rows.

    Each row is a mapping with an integer epoch and any of train_loss, val_loss,
    map50, map50_95, cmap, macro_f1. A missing or null value leaves a gap. If an
    epoch appears more than once, as after a resumed run, the last row wins.
    """
    by_epoch: dict = {}
    for row in progress_rows or []:
        try:
            epoch = int(row["epoch"])
        except (KeyError, TypeError, ValueError):
            raise ChartsError("every progress row needs a whole-number epoch")
        by_epoch[epoch] = row
    epochs = sorted(by_epoch)
    charts = []

    def values(key: str) -> list:
        out = []
        for e in epochs:
            v = by_epoch[e].get(key)
            out.append(_clean(v))
        return out

    loss_series = []
    for key, label in _PROGRESS_LOSS:
        vals = values(key)
        if any(v is not None for v in vals):
            loss_series.append(_series(len(loss_series), key, label, _pairs(epochs, vals), role="metric"))
    if loss_series:
        charts.append(_line_chart("loss_by_epoch", "loss_by_epoch", "Epoch", "Loss", loss_series,
                                  x_range=(epochs[0], epochs[-1]), y_range=None))
    score_series = []
    for key, label in _PROGRESS_SCORE:
        vals = values(key)
        if any(v is not None for v in vals):
            score_series.append(_series(len(score_series), key, label, _pairs(epochs, vals), role="metric"))
    if score_series:
        charts.append(_line_chart("map_by_epoch", "map_by_epoch", "Epoch", "Score", score_series,
                                  x_range=(epochs[0], epochs[-1])))
    return charts


def _clean(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ---------------------------------------------------------------------------
# Detection charts
# ---------------------------------------------------------------------------


def detection_charts(metrics: Mapping, *, max_class_series: int = 40, max_visible_classes: int = 8) -> list:
    """Every chart of a vision model's Results view, in display order."""
    _require_kind(metrics, "detection")
    names = metrics["class_names"]
    per_class = metrics["per_class"]
    support = [p["n_ground_truth"] for p in per_class]
    with_gt = [c for c in _order_by_support(support) if support[c] > 0]
    shown = with_gt[:max_class_series]
    omitted = len(with_gt) - len(shown)
    visible = _visible_set(support, max_visible_classes)
    curves = metrics["curves"]
    grid = curves["confidence_grid"]
    note_omit = (f"{omitted} further classes are in the table but not drawn here." if omitted else None)
    charts = []

    iou_pts = [(row["iou"], row["ap"]) for row in metrics["ap_by_iou"]]
    charts.append(_line_chart(
        "ap_by_iou", "ap_by_iou", "IoU threshold", "Average precision",
        [_series(0, "all", "All classes", iou_pts, role="macro")],
        x_range=(min(x for x, _ in iou_pts), max(x for x, _ in iou_pts)),
    ))

    def curve_chart(chart_id: str, key: str, y_label: str) -> dict:
        series = [_series(0, "macro", "All classes (mean)", _pairs(grid, curves["macro"][key]), role="macro")]
        for rank, c in enumerate(shown):
            ys = curves["per_class"][c][key]
            if ys is None:
                continue
            series.append(_series(rank + 1, f"class_{c}", names[c], _pairs(grid, ys),
                                  visible=c in visible))
        markers = []
        if key == "f1" and metrics["f1_optimal"]["macro_confidence"] is not None:
            markers.append({"series": "macro", "x": metrics["f1_optimal"]["macro_confidence"],
                            "y": metrics["f1_optimal"]["macro_f1"], "label": "Highest mean F1"})
        return _line_chart(chart_id, chart_id, "Confidence", y_label, series, markers=markers,
                           note=note_omit, omitted=omitted)

    charts.append(curve_chart("f1_vs_confidence", "f1", "F1"))
    charts.append(curve_chart("precision_vs_confidence", "precision", "Precision"))
    charts.append(curve_chart("recall_vs_confidence", "recall", "Recall"))

    pr = curves.get("precision_recall")
    if pr:
        rp = pr["recall_points"]
        series = []
        stacked = [e["precision"] for e in pr["per_class"] if e["precision"] is not None]
        if stacked:
            mean = [sum(col) / len(col) for col in zip(*stacked)]
            series.append(_series(0, "macro", "All classes (mean)", _pairs(rp, mean), role="macro"))
        for rank, c in enumerate(shown):
            ys = pr["per_class"][c]["precision"]
            if ys is not None:
                series.append(_series(rank + 1, f"class_{c}", names[c], _pairs(rp, ys), visible=c in visible))
        charts.append(_line_chart("precision_recall", "precision_recall", "Recall", "Precision", series,
                                  note="Computed at IoU 0.50 with the 101-point interpolation used for mAP."
                                       + (" " + note_omit if note_omit else ""), omitted=omitted))

    standard = metrics["settings"].get("standard_iou_range", False)
    tested = [c for c in range(len(names)) if support[c] > 0]
    ap50 = [per_class[c]["ap50"] for c in tested]
    apall = [per_class[c]["ap50_95"] for c in tested]
    series = []
    if 0.5 in metrics["settings"]["iou_thresholds"]:
        series.append({"index": 0, "key": "ap50", "label": "AP at IoU 0.50", "values": ap50})
    series.append({"index": len(series), "key": "ap50_95",
                   "label": "AP over IoU 0.50 to 0.95" if standard else "AP over the chosen IoU thresholds",
                   "values": apall})
    charts.append(_bar_chart("per_class_ap", "per_class_ap", [names[c] for c in tested], series,
                             "Average precision",
                             note=("Classes with no labelled boxes in the test set are not shown, "
                                   "because they were not tested."), omitted=0))
    cm = metrics["confusion_matrix"]
    charts.append(_matrix_chart(
        "confusion_matrix", "confusion_matrix", cm["labels"], cm["matrix"],
        row_title="Labelled truth", col_title="Model report",
        note=(f"Counted at confidence {cm['confidence']:g} and IoU {cm['iou']:g}. "
              "A box of the right class counts on the diagonal. A box in the right place with the wrong "
              "class is off the diagonal. The last row is false alarms and the last column is misses."),
    ))
    return charts


# ---------------------------------------------------------------------------
# Acoustic charts
# ---------------------------------------------------------------------------


def acoustic_charts(metrics: Mapping, *, max_class_series: int = 40, max_visible_classes: int = 8) -> list:
    """Every chart of a sound model's Results view, in display order."""
    _require_kind(metrics, "acoustic")
    names = metrics["class_names"]
    per_class = metrics["per_class"]
    support = [p["n_positive"] for p in per_class]
    with_pos = [c for c in _order_by_support(support) if support[c] > 0]
    shown = with_pos[:max_class_series]
    omitted = len(with_pos) - len(shown)
    visible = _visible_set(support, max_visible_classes)
    note_omit = (f"{omitted} further classes are in the table but not drawn here." if omitted else None)
    curves = metrics["curves"]
    grid = curves["threshold_grid"]
    lo, hi = grid[0], grid[-1]
    charts = []

    series = []
    for rank, c in enumerate(shown):
        pr = curves["per_class"][c]["precision_recall"]
        if pr:
            series.append(_series(rank, f"class_{c}", names[c], _pairs(pr["recall"], pr["precision"]),
                                  visible=c in visible))
    charts.append(_line_chart("precision_recall", "precision_recall", "Recall", "Precision", series,
                              note=note_omit, omitted=omitted))

    series = []
    for rank, c in enumerate(shown):
        roc = curves["per_class"][c]["roc"]
        if roc:
            series.append(_series(rank + 1, f"class_{c}", names[c], _pairs(roc["fpr"], roc["tpr"]),
                                  visible=c in visible))
    series.insert(0, _series(0, "no_skill", "No skill", [(0.0, 0.0), (1.0, 1.0)], role="reference", dashed=True))
    charts.append(_line_chart("roc", "roc", "False positive rate", "True positive rate", series,
                              note=note_omit, omitted=omitted))

    series = [_series(0, "macro", "Mean over classes", _pairs(grid, curves["macro_f1"]), role="macro")]
    for rank, c in enumerate(shown):
        f1 = curves["per_class"][c]["f1"]
        if f1:
            series.append(_series(rank + 1, f"class_{c}", names[c], _pairs(grid, f1), visible=c in visible))
    sh = metrics["f1_optimal"]["shared_threshold"]
    markers = ([{"series": "macro", "x": sh["threshold"], "y": sh["macro_f1"], "label": "Highest mean F1"}]
               if sh["threshold"] is not None else [])
    charts.append(_line_chart("f1_vs_threshold", "f1_vs_threshold", "Score threshold", "F1", series,
                              x_range=(lo, hi), markers=markers, note=note_omit, omitted=omitted))

    tested = [c for c in range(len(names)) if support[c] > 0]
    ap_vals = [per_class[c]["ap"] for c in tested]
    auc_vals = [per_class[c]["roc_auc"] for c in tested]
    charts.append(_bar_chart("per_class_ap", "per_class_ap", [names[c] for c in tested], [
        {"index": 0, "key": "ap", "label": "Average precision", "values": ap_vals},
        {"index": 1, "key": "roc_auc", "label": "ROC area", "values": auc_vals},
    ], "Score", note="Classes with no labelled windows in the test set are not shown, because they were not tested."))

    cm = metrics["confusion_matrix"]
    charts.append(_matrix_chart(
        "confusion_matrix", "confusion_matrix", cm["labels"], cm["matrix"],
        row_title="Labelled truth", col_title="Model report",
        note=(f"Each window is assigned the class with the highest score if that score reaches "
              f"{cm['threshold']:g}, and otherwise to background. The last row is windows with no "
              "labelled species, the last column is windows reported as background."),
    ))

    rel = metrics.get("reliability") or {}
    series = [_series(0, "perfect", "Perfect calibration", [(0.0, 0.0), (1.0, 1.0)], role="reference", dashed=True)]
    counts = {}
    notes = []
    for idx, (key, label) in enumerate((("raw", "Raw score"), ("calibrated", "Calibrated probability")), start=1):
        block = rel.get(key)
        if block and block["n"]:
            pts = [(b["mean_predicted"], b["fraction_positive"]) for b in block["bins"] if b["count"]]
            series.append(_series(idx, key, label, pts, role="metric"))
            counts[key] = [b["count"] for b in block["bins"]]
            if block.get("expected_calibration_error") is not None:
                notes.append(f"{label}: expected calibration error {block['expected_calibration_error']:.3f}.")
    charts.append(_line_chart("reliability", "reliability", "Mean predicted value in the bin",
                              "Fraction truly present", series, note=" ".join(notes) or None,
                              extra={"bin_counts": counts}))
    return charts


def _require_kind(metrics: Mapping, kind: str) -> None:
    if not isinstance(metrics, Mapping) or metrics.get("kind") != kind:
        raise ChartsError(f"expected {kind} metrics")


# ---------------------------------------------------------------------------
# Tables and cards
# ---------------------------------------------------------------------------


def headline_cards(metrics: Mapping) -> list:
    """The few figures shown at the top of Results, each with its help line."""
    h = M.headline(metrics)
    keys = _DETECTION_ROWS if h["kind"] == "detection" else _ACOUSTIC_ROWS
    cards = []
    for key in keys:
        label, text = METRIC_HELP[key]
        cards.append({"key": key, "label": label, "help": text, "value": h.get(key)})
    return cards


def class_table(metrics: Mapping) -> dict:
    """One row per class with its counts and figures. Undefined figures are None."""
    kind = metrics.get("kind")
    if kind == "detection":
        columns = [
            ("class_name", "Class", "The class name."),
            ("n_ground_truth", "Test boxes", "Labelled boxes of this class in the test set."),
            ("ap50", "AP at IoU 0.50", _help("map50")),
            ("ap50_95", "AP over IoU range", _help("map50_95")),
            ("precision", "Precision", _help("precision")),
            ("recall", "Recall", _help("recall")),
            ("f1", "F1", _help("f1")),
        ]
        rows = metrics["per_class"]
        tested = [p["n_ground_truth"] > 0 for p in rows]
    elif kind == "acoustic":
        columns = [
            ("class_name", "Class", "The class name."),
            ("n_positive", "Test windows", "Windows labelled with this class in the test set."),
            ("ap", "Average precision", _help("cmap")),
            ("roc_auc", "ROC area", _help("macro_roc_auc")),
            ("precision", "Precision", _help("precision")),
            ("recall", "Recall", _help("recall")),
            ("f1", "F1", _help("f1")),
        ]
        rows = metrics["per_class"]
        tested = [p["n_positive"] > 0 for p in rows]
    else:
        raise ChartsError(f"unknown metrics kind {kind!r}")
    out_rows = []
    for row, is_tested in zip(rows, tested):
        entry = {key: row.get(key) for key, _, _ in columns}
        entry["tested"] = is_tested
        entry["status"] = "" if is_tested else "Not tested. No labelled examples in the test set."
        out_rows.append(entry)
    return {"columns": [{"key": k, "label": l, "help": h} for k, l, h in columns], "rows": out_rows}


def calibration_table(metrics: Mapping) -> dict | None:
    """Per-class calibrated thresholds of a sound model, in plain language."""
    _require_kind(metrics, "acoustic")
    cals = metrics.get("calibration")
    if not cals:
        return None
    rows = []
    for cal in cals:
        rows.append({
            "class_name": cal.get("class_name"),
            "status": cal["status"],
            "status_text": CALIBRATION_STATUS_TEXT.get(cal["status"], cal["status"]),
            "target_probability": cal["target_probability"],
            "threshold": cal["threshold"],
            "threshold_within_observed_scores": cal["threshold_within_observed_scores"],
            "n_positive": cal["n_positive"],
            "n_negative": cal["n_negative"],
            "slope": cal["a"],
            "intercept": cal["b"],
        })
    return {"note": CALIBRATION_NOTE, "help": _help("calibration_threshold"), "rows": rows}


def comparison_table(
    challenger: Mapping,
    current: Mapping | None,
    *,
    challenger_shared: Mapping | None = None,
    current_shared: Mapping | None = None,
) -> dict:
    """Both models on the same test set, side by side, with differences.

    challenger_shared and current_shared are the same two models scored again on
    only the classes they share, which the caller makes with
    metrics.restrict_detection or metrics.restrict_acoustic. The table never says
    which model is better, it prints the numbers and challenger minus current.
    """
    kind = challenger.get("kind")
    keys = _DETECTION_ROWS if kind == "detection" else _ACOUSTIC_ROWS if kind == "acoustic" else None
    if keys is None:
        raise ChartsError(f"unknown metrics kind {kind!r}")
    if current is not None and current.get("kind") != kind:
        raise ChartsError("the two models were scored with different kinds of metrics")
    if (challenger_shared is None) != (current_shared is None):
        raise ChartsError("give both shared-class results or neither")

    def rows_for(a: Mapping, b: Mapping | None) -> list:
        ha = M.headline(a)
        hb = M.headline(b) if b is not None else {}
        out = []
        for key in keys:
            va, vb = ha.get(key), hb.get(key)
            diff = (va - vb) if (va is not None and vb is not None) else None
            label, text = METRIC_HELP[key]
            out.append({"metric": key, "label": label, "help": text, "challenger": va, "current": vb,
                        "difference": diff})
        return out

    cn = list(challenger["class_names"])
    if current is None:
        return {"note": "No model is currently in use for this slot, so there is nothing to compare with.",
                "rows": rows_for(challenger, None), "not_in_current": cn, "not_in_challenger": [],
                "shared_classes": [], "shared_rows": None}
    un = list(current["class_names"])
    shared = [n for n in cn if n in set(un)]
    return {
        "note": COMPARISON_NOTE,
        "rows": rows_for(challenger, current),
        "not_in_current": [n for n in cn if n not in set(un)],
        "not_in_challenger": [n for n in un if n not in set(cn)],
        "shared_classes": shared,
        "shared_rows": rows_for(challenger_shared, current_shared) if challenger_shared is not None else None,
        "test_counts": {"challenger": _test_counts(challenger), "current": _test_counts(current)},
    }


def _test_counts(metrics: Mapping) -> dict:
    return dict(metrics["counts"])


def results_bundle(metrics: Mapping, progress_rows: Sequence[Mapping] | None = None, **chart_options) -> dict:
    """Everything the Results sub-panel needs for one model."""
    kind = metrics.get("kind")
    if kind == "detection":
        charts = detection_charts(metrics, **chart_options)
    elif kind == "acoustic":
        charts = acoustic_charts(metrics, **chart_options)
    else:
        raise ChartsError(f"unknown metrics kind {kind!r}")
    if progress_rows:
        charts = training_curve_charts(progress_rows) + charts
    return {
        "definition_version": CHARTS_DEFINITION_VERSION,
        "kind": kind,
        "honesty_line": HONESTY_LINE,
        "cards": headline_cards(metrics),
        "charts": charts,
        "class_table": class_table(metrics),
        "calibration": calibration_table(metrics) if kind == "acoustic" else None,
    }


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def chart_csv_text(chart: Mapping) -> str:
    """The chart's data as CSV text, one header row then one row per point."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(chart["csv"]["columns"])
    for row in chart["csv"]["rows"]:
        writer.writerow([_cell(v) for v in row])
    return buf.getvalue()


def write_chart_csvs(charts: Sequence[Mapping], folder: Path | str) -> dict:
    """Write one UTF-8 CSV per chart into folder, named <chart id>.csv.

    The byte-order mark lets spreadsheet programs open accented species names
    correctly. Returns {chart id: file name}. An existing file of the same name
    is replaced, because a chart file is a copy of figures stored elsewhere.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = {}
    for chart in charts:
        chart_id = _check_chart_id(chart["id"])
        name = f"{chart_id}.csv"
        with open(folder / name, "w", encoding="utf-8-sig", newline="") as handle:
            handle.write(chart_csv_text(chart))
        written[chart_id] = name
    return written
