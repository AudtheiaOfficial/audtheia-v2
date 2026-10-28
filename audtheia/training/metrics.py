"""Evaluation metrics for the Train tab.

Path: audtheia/training/metrics.py

Audtheia's own implementations of the standard measures used to judge a trained
model on its held-out test split, written with NumPy only so they run on any
computer that can run the application, with or without the training libraries.
The test suite checks every figure against an independent reference
implementation (pycocotools, faster-coco-eval, scikit-learn and supervision) on
fixtures with known answers. The reference libraries are never imported here.

Two families are provided.

Detection (vision). COCO-style average precision, computed exactly as the COCO
evaluation does: per image and class the detections are sorted by score and cut
to the maximum per image, each is matched greedily to the unmatched ground-truth
box with the highest overlap that reaches the threshold, and precision is
interpolated at 101 recall points. From the same matching come precision, recall
and F1 against confidence, a confusion matrix with a background row and column,
and top-1 class accuracy on matched boxes.

Acoustic. Average precision per class as the step-wise area under the
precision-recall curve (the definition scikit-learn uses), class-mean average
precision, ROC area per class and macro-averaged, precision, recall and F1 at a
chosen and at the F1-optimal threshold, a top-1 confusion matrix with a
background row and column, reliability bins, and per-class logistic calibration
of scores following Wood and Kahl (2024, Journal of Ornithology 165:777-782):
a logistic regression of whether a call is present on the logit of the score,
solved for the score that reaches a chosen probability.

Every figure returned is a plain Python value that survives JSON. A value that
cannot be defined, for example the recall of a class with no ground truth, is
None and never NaN. These are model outputs, never measurements of the
environment.
"""

from __future__ import annotations

import math
import random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

METRICS_DEFINITION_VERSION = 1

DEFAULT_IOU_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
DEFAULT_MAX_DETECTIONS = 100
DEFAULT_REPORT_CONFIDENCE = 0.30
DEFAULT_MATCH_IOU = 0.50
RECALL_POINTS = 101
CONFIDENCE_GRID_POINTS = 101
CALIBRATION_BINS = 10
CURVE_MAX_POINTS = 201

TOP1_DEFINITION = (
    "Top-1 class accuracy on matched boxes. Among ground-truth boxes that a "
    "prediction overlaps at the matching overlap threshold and at or above the "
    "reporting confidence, the fraction whose matched prediction carries the "
    "right class. It is not average precision and must not be read as such."
)

ACOUSTIC_TOP1_DEFINITION = (
    "Top-1 accuracy. Among sound windows that contain at least one labelled "
    "species, the fraction whose highest-scoring class is one of the labelled "
    "species."
)

_EPS = float(np.spacing(1))


class MetricsError(ValueError):
    """The inputs to a metric were malformed. The message says what and where."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _num(value: Any) -> float | None:
    """A JSON-safe float, or None when the value is missing, NaN or infinite."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _num_list(values: Iterable[Any]) -> list:
    return [_num(v) for v in values]


def _check_thresholds(iou_thresholds: Sequence[float]) -> list:
    values = [float(t) for t in iou_thresholds]
    if not values:
        raise MetricsError("iou_thresholds must not be empty")
    for t in values:
        if not (0.0 < t <= 1.0) or not math.isfinite(t):
            raise MetricsError(f"iou threshold {t!r} must be above 0 and at most 1")
    if len(set(values)) != len(values):
        raise MetricsError("iou_thresholds must not repeat a value")
    return values


def _check_unit(name: str, value: float) -> float:
    v = float(value)
    if not (0.0 <= v <= 1.0) or not math.isfinite(v):
        raise MetricsError(f"{name} must be between 0 and 1, got {value!r}")
    return v


def _safe_div(num: float, den: float) -> float | None:
    return float(num) / float(den) if den else None


def _f1_from_counts(tp: int, fp: int, fn: int) -> float | None:
    """F1 as 2TP / (2TP + FP + FN). None when the class has no positives at all."""
    if tp + fn == 0:
        return None
    return 2.0 * tp / (2.0 * tp + fp + fn)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _logit(p: np.ndarray, eps: float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def downsample_points(n: int, max_points: int) -> list:
    """Indices of at most max_points evenly spread positions out of n, ends kept."""
    if n <= max_points or max_points < 2:
        return list(range(n))
    idx = np.unique(np.round(np.linspace(0, n - 1, max_points)).astype(int))
    return [int(i) for i in idx]


# ---------------------------------------------------------------------------
# Detection input handling
# ---------------------------------------------------------------------------


def _as_box(box: Any, where: str) -> np.ndarray:
    arr = np.asarray(box, dtype=np.float64).reshape(-1)
    if arr.shape != (4,) or not np.all(np.isfinite(arr)):
        raise MetricsError(f"{where}: a box needs four finite numbers x1, y1, x2, y2")
    return arr


def _parse_detection_inputs(gt: Mapping, pred: Mapping, n_classes: int):
    """Normalise to per-image, per-class arrays. Raises on anything malformed."""
    if not isinstance(gt, Mapping) or not isinstance(pred, Mapping):
        raise MetricsError("ground truth and predictions must each map image id to a list")
    unknown = [k for k in pred if k not in gt]
    if unknown:
        raise MetricsError(
            f"predictions for image {unknown[0]!r} have no entry in the ground truth; "
            "list every evaluated image in the ground truth, with an empty list for "
            "an image that has no boxes"
        )
    images = list(gt.keys())
    parsed = []
    for image_id in images:
        gt_by_class: dict = {}
        for k, item in enumerate(gt[image_id] or []):
            try:
                cls, box = item
            except (TypeError, ValueError):
                raise MetricsError(f"ground truth {image_id!r}[{k}] must be (class, box)")
            cls = _class_index(cls, n_classes, f"ground truth {image_id!r}[{k}]")
            b = _as_box(box, f"ground truth {image_id!r}[{k}]")
            if b[2] <= b[0] or b[3] <= b[1]:
                raise MetricsError(f"ground truth {image_id!r}[{k}] has no area")
            gt_by_class.setdefault(cls, []).append(b)
        det_by_class: dict = {}
        for k, item in enumerate(pred.get(image_id) or []):
            try:
                cls, score, box = item
            except (TypeError, ValueError):
                raise MetricsError(f"prediction {image_id!r}[{k}] must be (class, score, box)")
            cls = _class_index(cls, n_classes, f"prediction {image_id!r}[{k}]")
            score = float(score)
            if not math.isfinite(score):
                raise MetricsError(f"prediction {image_id!r}[{k}] has a score that is not finite")
            b = _as_box(box, f"prediction {image_id!r}[{k}]")
            det_by_class.setdefault(cls, []).append((score, b))
        parsed.append((image_id, gt_by_class, det_by_class))
    return parsed


def _class_index(value: Any, n_classes: int, where: str) -> int:
    try:
        if isinstance(value, bool) or int(value) != value:
            raise ValueError
        cls = int(value)
    except (TypeError, ValueError):
        raise MetricsError(f"{where}: class must be a whole number")
    if not (0 <= cls < n_classes):
        raise MetricsError(f"{where}: class {cls} is outside 0..{n_classes - 1}")
    return cls


def _iou_matrix(dt: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Intersection over union, shape (len(dt), len(gt)), boxes as x1 y1 x2 y2."""
    if len(dt) == 0 or len(gt) == 0:
        return np.zeros((len(dt), len(gt)), dtype=np.float64)
    d_area = (dt[:, 2] - dt[:, 0]) * (dt[:, 3] - dt[:, 1])
    g_area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    iw = np.minimum(dt[:, None, 2], gt[None, :, 2]) - np.maximum(dt[:, None, 0], gt[None, :, 0])
    ih = np.minimum(dt[:, None, 3], gt[None, :, 3]) - np.maximum(dt[:, None, 1], gt[None, :, 1])
    inter = np.clip(iw, 0.0, None) * np.clip(ih, 0.0, None)
    union = d_area[:, None] + g_area[None, :] - inter
    out = np.zeros_like(inter)
    np.divide(inter, union, out=out, where=union > 0)
    return out


def _match_in_score_order(iou: np.ndarray, threshold: float) -> np.ndarray:
    """COCO matching. Detections arrive in score order. Each takes the free
    ground-truth box with the highest IoU that reaches the threshold, the last
    one when several tie. Returns the matched ground-truth index per detection,
    or -1."""
    n_det, n_gt = iou.shape
    match = np.full(n_det, -1, dtype=np.int64)
    if n_det == 0 or n_gt == 0:
        return match
    taken = np.zeros(n_gt, dtype=bool)
    bar = min(threshold, 1.0 - 1e-10)
    for d in range(n_det):
        row = iou[d]
        cand = (row >= bar) & ~taken
        if not cand.any():
            continue
        masked = np.where(cand, row, -1.0)
        best = masked.max()
        m = n_gt - 1 - int(np.argmax(masked[::-1] == best))
        taken[m] = True
        match[d] = m
    return match


def _interpolated_precision(scores: np.ndarray, matched: np.ndarray, n_gt: int):
    """COCO precision at 101 recall points for each threshold row of matched.

    scores: (D,) all detections of one class. matched: (T, D) booleans.
    Returns q of shape (T, 101).
    """
    n_thr = matched.shape[0]
    q = np.zeros((n_thr, RECALL_POINTS), dtype=np.float64)
    n_det = scores.shape[0]
    if n_det == 0 or n_gt == 0:
        return q
    order = np.argsort(-scores, kind="mergesort")
    tp = matched[:, order]
    tp_sum = np.cumsum(tp, axis=1, dtype=np.float64)
    fp_sum = np.cumsum(~tp, axis=1, dtype=np.float64)
    recall_points = np.linspace(0.0, 1.0, RECALL_POINTS)
    for t in range(n_thr):
        rc = tp_sum[t] / n_gt
        pr = tp_sum[t] / (fp_sum[t] + tp_sum[t] + _EPS)
        pr = np.maximum.accumulate(pr[::-1])[::-1]
        inds = np.searchsorted(rc, recall_points, side="left")
        ok = inds < n_det
        q[t, ok] = pr[inds[ok]]
    return q


# ---------------------------------------------------------------------------
# Detection evaluation
# ---------------------------------------------------------------------------


def evaluate_detections(
    gt: Mapping,
    pred: Mapping,
    class_names: Sequence[str],
    *,
    iou_thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
    conf_threshold: float = DEFAULT_REPORT_CONFIDENCE,
    match_iou: float = DEFAULT_MATCH_IOU,
    confidence_grid_points: int = CONFIDENCE_GRID_POINTS,
    include_image_outcomes: bool = False,
) -> dict:
    """Score detections against ground truth.

    gt maps an image id to a list of (class_index, [x1, y1, x2, y2]). pred maps
    an image id to a list of (class_index, score, [x1, y1, x2, y2]). Every image
    that was evaluated must appear as a key of gt, with an empty list when it has
    no ground-truth boxes. class_index points into class_names.
    """
    names = [str(n) for n in class_names]
    n_classes = len(names)
    if n_classes < 1:
        raise MetricsError("class_names must not be empty")
    thresholds = _check_thresholds(iou_thresholds)
    max_det = int(max_detections)
    if max_det < 1:
        raise MetricsError("max_detections must be at least 1")
    conf = _check_unit("conf_threshold", conf_threshold)
    match_iou = float(match_iou)
    if not (0.0 < match_iou <= 1.0):
        raise MetricsError("match_iou must be above 0 and at most 1")
    if int(confidence_grid_points) < 2:
        raise MetricsError("confidence_grid_points must be at least 2")

    parsed = _parse_detection_inputs(gt, pred, n_classes)

    all_thr = list(thresholds)
    if match_iou in all_thr:
        match_idx = all_thr.index(match_iou)
    else:
        all_thr.append(match_iou)
        match_idx = len(all_thr) - 1
    n_thr = len(all_thr)

    # Evaluate every (image, class) cell once, keep what later stages need.
    cells: dict = {}
    per_class_scores: list = [[] for _ in range(n_classes)]
    per_class_matched: list = [[] for _ in range(n_classes)]
    n_gt = [0] * n_classes
    n_pred = [0] * n_classes
    n_pred_used = [0] * n_classes
    for img_i, (_image_id, gt_by_class, det_by_class) in enumerate(parsed):
        for cls in set(gt_by_class) | set(det_by_class):
            g = np.array(gt_by_class.get(cls, []), dtype=np.float64).reshape(-1, 4)
            dets = det_by_class.get(cls, [])
            n_gt[cls] += len(g)
            n_pred[cls] += len(dets)
            if dets:
                sc = np.array([d[0] for d in dets], dtype=np.float64)
                bx = np.array([d[1] for d in dets], dtype=np.float64).reshape(-1, 4)
                order = np.argsort(-sc, kind="mergesort")[:max_det]
                sc, bx = sc[order], bx[order]
            else:
                sc = np.zeros(0, dtype=np.float64)
                bx = np.zeros((0, 4), dtype=np.float64)
            iou = _iou_matrix(bx, g)
            matches = np.stack([_match_in_score_order(iou, t) for t in all_thr]) if len(sc) else \
                np.zeros((n_thr, 0), dtype=np.int64)
            n_pred_used[cls] += len(sc)
            if len(sc):
                per_class_scores[cls].append(sc)
                per_class_matched[cls].append(matches >= 0)
            cells[(img_i, cls)] = (g, sc, bx, matches[match_idx] if len(sc) else np.zeros(0, dtype=np.int64))

    # Average precision per class and threshold.
    n_requested = len(thresholds)
    class_q = []
    ap_matrix = np.full((n_requested, n_classes), np.nan)
    pr_iou50 = {}
    for c in range(n_classes):
        scores = np.concatenate(per_class_scores[c]) if per_class_scores[c] else np.zeros(0)
        matched = np.concatenate(per_class_matched[c], axis=1) if per_class_matched[c] else \
            np.zeros((n_thr, 0), dtype=bool)
        if n_gt[c] > 0:
            q = _interpolated_precision(scores, matched, n_gt[c])
            ap_matrix[:, c] = q[:n_requested].mean(axis=1)
            if 0.5 in thresholds:
                pr_iou50[c] = q[thresholds.index(0.5)]
        class_q.append((scores, matched))
    valid = ~np.isnan(ap_matrix)
    if valid.any():
        ap_by_iou = [float(np.nanmean(ap_matrix[t])) if valid[t].any() else None for t in range(n_requested)]
        map_all = float(ap_matrix[valid].mean())
    else:
        ap_by_iou = [None] * n_requested
        map_all = None
    map50 = ap_by_iou[thresholds.index(0.5)] if 0.5 in thresholds else None

    # Confidence statistics at the matching threshold, class by class.
    # Each grid value is the correctly rounded decimal i / (N - 1), so 0.70 is the
    # number a person types as 0.70 and a score of exactly 0.70 is at or above it.
    grid = np.arange(int(confidence_grid_points)) / (int(confidence_grid_points) - 1)
    class_curves = []
    for c in range(n_classes):
        scores, matched = class_q[c]
        class_curves.append(_confidence_curves(scores, matched[match_idx] if matched.size else
                                               np.zeros(0, dtype=bool), n_gt[c], grid))

    # Confusion matrix, top-1 accuracy, per-image outcomes at the reporting confidence.
    matrix = np.zeros((n_classes + 1, n_classes + 1), dtype=np.int64)
    tp_c = [0] * n_classes
    kept_c = [0] * n_classes
    outcomes: dict = {}
    for img_i, (image_id, gt_by_class, det_by_class) in enumerate(parsed):
        classes_here = sorted(set(gt_by_class) | set(det_by_class))
        gt_cls, gt_box, det_cls, det_box = [], [], [], []
        gt_matched_flat: list = []
        det_matched_flat: list = []
        tp_img = 0
        for cls in classes_here:
            g, sc, bx, mg = cells[(img_i, cls)]
            n_keep = int(np.count_nonzero(sc >= conf))
            taken = np.zeros(len(g), dtype=bool)
            for d in range(n_keep):
                if mg[d] >= 0:
                    taken[mg[d]] = True
            n_tp = int(taken.sum())
            tp_img += n_tp
            tp_c[cls] += n_tp
            kept_c[cls] += n_keep
            matrix[cls, cls] += n_tp
            for i in range(len(g)):
                gt_cls.append(cls)
                gt_box.append(g[i])
                gt_matched_flat.append(bool(taken[i]))
            det_hit = np.zeros(n_keep, dtype=bool)
            det_hit[:] = mg[:n_keep] >= 0
            for d in range(n_keep):
                det_cls.append(cls)
                det_box.append(bx[d])
                det_matched_flat.append(bool(det_hit[d]))
        n_gt_img = len(gt_cls)
        n_det_img = len(det_cls)
        free_g = [i for i in range(n_gt_img) if not gt_matched_flat[i]]
        free_d = [j for j in range(n_det_img) if not det_matched_flat[j]]
        g_used = set()
        d_used = set()
        if free_g and free_d:
            iou = _iou_matrix(np.array([det_box[j] for j in free_d]), np.array([gt_box[i] for i in free_g]))
            bar = min(match_iou, 1.0 - 1e-10)
            dj, gi = np.nonzero(iou >= bar)
            if dj.size:
                order = np.lexsort((dj, gi, -iou[dj, gi]))
                for k in order:
                    di, gj = int(dj[k]), int(gi[k])
                    if di in d_used or gj in g_used:
                        continue
                    d_used.add(di)
                    g_used.add(gj)
                    matrix[gt_cls[free_g[gj]], det_cls[free_d[di]]] += 1
        for k, i in enumerate(free_g):
            if k not in g_used:
                matrix[gt_cls[i], n_classes] += 1
        for k, j in enumerate(free_d):
            if k not in d_used:
                matrix[n_classes, det_cls[j]] += 1
        if include_image_outcomes:
            outcomes[str(image_id)] = {
                "tp": tp_img,
                "fp": n_det_img - tp_img,
                "fn": n_gt_img - tp_img,
            }

    diag = int(sum(matrix[c, c] for c in range(n_classes)))
    confused = int(matrix[:n_classes, :n_classes].sum()) - diag
    matched_total = diag + confused
    top1 = {
        "value": _safe_div(diag, matched_total),
        "matched": matched_total,
        "correct": diag,
        "definition": TOP1_DEFINITION,
    }

    # Per-class table at the reporting confidence.
    per_class = []
    for c in range(n_classes):
        tp, fp = tp_c[c], kept_c[c] - tp_c[c]
        fn = n_gt[c] - tp if n_gt[c] > 0 else 0
        has_gt = n_gt[c] > 0
        ap50 = None
        ap_all = None
        if has_gt:
            ap_all = _num(np.mean(ap_matrix[:, c]))
            if 0.5 in thresholds:
                ap50 = _num(ap_matrix[thresholds.index(0.5), c])
        per_class.append({
            "class_index": c,
            "class_name": names[c],
            "n_ground_truth": n_gt[c],
            "n_predictions": n_pred[c],
            "ap50": ap50,
            "ap50_95": ap_all,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": _safe_div(tp, tp + fp),
            "recall": _safe_div(tp, n_gt[c]) if has_gt else None,
            "f1": _f1_from_counts(tp, fp, fn) if has_gt else None,
        })

    with_gt = [c for c in range(n_classes) if n_gt[c] > 0]
    micro_tp = sum(tp_c[c] for c in with_gt)
    micro_fp = sum(kept_c[c] - tp_c[c] for c in range(n_classes))
    micro_fn = sum(n_gt[c] - tp_c[c] for c in with_gt)
    prec_defined = [r["precision"] for r in per_class if r["n_ground_truth"] > 0 and r["precision"] is not None]
    at_conf = {
        "confidence": conf,
        "macro_precision": _num(np.mean(prec_defined)) if prec_defined else None,
        "macro_recall": _num(np.mean([per_class[c]["recall"] for c in with_gt])) if with_gt else None,
        "macro_f1": _num(np.mean([per_class[c]["f1"] for c in with_gt])) if with_gt else None,
        "micro_precision": _safe_div(micro_tp, micro_tp + micro_fp),
        "micro_recall": _safe_div(micro_tp, micro_tp + micro_fn),
        "micro_f1": _f1_from_counts(micro_tp, micro_fp, micro_fn),
    }

    # Pooled curves.
    macro_curve = _macro_curves(class_curves, with_gt, grid)
    micro_curve = _micro_curves(class_curves, with_gt, n_gt, grid)
    best = int(np.argmax(np.array([v if v is not None else -1.0 for v in macro_curve["f1"]])[::-1]))
    best_idx = len(grid) - 1 - best
    f1_opt = {
        "macro_confidence": _num(grid[best_idx]) if with_gt else None,
        "macro_f1": macro_curve["f1"][best_idx] if with_gt else None,
        "per_class": [
            class_curves[c]["f1_optimal"] if n_gt[c] > 0 else None for c in range(n_classes)
        ],
    }

    result = {
        "kind": "detection",
        "definition_version": METRICS_DEFINITION_VERSION,
        "settings": {
            "iou_thresholds": thresholds,
            "max_detections": max_det,
            "report_confidence": conf,
            "match_iou": match_iou,
            "recall_points": RECALL_POINTS,
            "standard_iou_range": [round(t, 10) for t in thresholds] == list(DEFAULT_IOU_THRESHOLDS),
        },
        "counts": {
            "images": len(parsed),
            "ground_truth_boxes": int(sum(n_gt)),
            "predictions": int(sum(n_pred)),
            "predictions_evaluated": int(sum(n_pred_used)),
            "classes": n_classes,
            "classes_with_ground_truth": len(with_gt),
        },
        "class_names": names,
        "map50": map50,
        "map50_95": map_all,
        "ap_by_iou": [{"iou": thresholds[t], "ap": ap_by_iou[t]} for t in range(n_requested)],
        "per_class": per_class,
        "at_confidence": at_conf,
        "top1_accuracy_on_matched_boxes": top1,
        "confusion_matrix": {
            "labels": names + ["background"],
            "matrix": matrix.tolist(),
            "rows": "ground truth",
            "columns": "prediction",
            "confidence": conf,
            "iou": match_iou,
        },
        "curves": {
            "confidence_grid": _num_list(grid),
            "per_class": [
                {
                    "class_index": c,
                    "class_name": names[c],
                    "precision": class_curves[c]["precision"],
                    "recall": class_curves[c]["recall"],
                    "f1": class_curves[c]["f1"],
                } if n_gt[c] > 0 else {"class_index": c, "class_name": names[c],
                                       "precision": None, "recall": None, "f1": None}
                for c in range(n_classes)
            ],
            "macro": macro_curve,
            "micro": micro_curve,
            "precision_recall": {
                "recall_points": _num_list(np.linspace(0.0, 1.0, RECALL_POINTS)),
                "iou": 0.5,
                "per_class": [
                    {"class_index": c, "class_name": names[c],
                     "precision": _num_list(pr_iou50[c]) if c in pr_iou50 else None}
                    for c in range(n_classes)
                ],
            } if 0.5 in thresholds else None,
        },
        "f1_optimal": f1_opt,
    }
    if include_image_outcomes:
        result["image_outcomes"] = outcomes
    return result


def _confidence_curves(scores: np.ndarray, matched50: np.ndarray, n_gt: int, grid: np.ndarray) -> dict:
    """Precision, recall and F1 of one class at each confidence in grid, plus the
    F1-optimal confidence. matched50 holds the match flag of each detection at
    the matching IoU. Because matching proceeds in score order, the detections at
    or above any confidence are matched exactly as they are in the full run."""
    n = scores.shape[0]
    out: dict = {"precision": [None] * len(grid), "recall": [None] * len(grid), "f1": [None] * len(grid),
                 "f1_optimal": None, "_tp": None, "_kept": None}
    order = np.argsort(-scores, kind="mergesort")
    s_desc = scores[order]
    cum_tp = np.cumsum(matched50[order], dtype=np.int64)
    asc = s_desc[::-1]
    kept = n - np.searchsorted(asc, grid, side="left")
    if n:
        tp = np.where(kept > 0, cum_tp[np.clip(kept - 1, 0, n - 1)], 0)
    else:
        tp = np.zeros(len(grid), dtype=np.int64)
    out["_tp"], out["_kept"] = tp, kept
    if n_gt == 0:
        return out
    fp = kept - tp
    fn = n_gt - tp
    out["precision"] = [_safe_div(int(tp[i]), int(kept[i])) for i in range(len(grid))]
    out["recall"] = [_safe_div(int(tp[i]), n_gt) for i in range(len(grid))]
    out["f1"] = [_f1_from_counts(int(tp[i]), int(fp[i]), int(fn[i])) for i in range(len(grid))]
    if n:
        # distinct score values, each with every detection at or above it
        last = np.r_[np.nonzero(np.diff(s_desc))[0], n - 1]
        tp_k = cum_tp[last]
        kept_k = last + 1
        f1_k = 2.0 * tp_k / (kept_k + n_gt)
        k = int(np.argmax(f1_k))
        out["f1_optimal"] = {
            "confidence": _num(s_desc[last[k]]),
            "f1": _num(f1_k[k]),
            "precision": _num(tp_k[k] / kept_k[k]),
            "recall": _num(tp_k[k] / n_gt),
        }
    else:
        out["f1_optimal"] = {"confidence": None, "f1": 0.0, "precision": None, "recall": 0.0}
    return out


def _macro_curves(class_curves: list, with_gt: list, grid: np.ndarray) -> dict:
    """Mean over classes that have ground truth. Precision averages only the
    classes that made a prediction at that confidence."""
    p, r, f = [], [], []
    for i in range(len(grid)):
        ps = [class_curves[c]["precision"][i] for c in with_gt if class_curves[c]["precision"][i] is not None]
        rs = [class_curves[c]["recall"][i] for c in with_gt]
        fs = [class_curves[c]["f1"][i] for c in with_gt]
        p.append(_num(np.mean(ps)) if ps else None)
        r.append(_num(np.mean(rs)) if rs else None)
        f.append(_num(np.mean(fs)) if fs else None)
    return {"precision": p, "recall": r, "f1": f}


def _micro_curves(class_curves: list, with_gt: list, n_gt: list, grid: np.ndarray) -> dict:
    """Pooled counts over every class. A prediction of a class that has no
    ground truth is a false alarm and is counted."""
    total_gt = sum(n_gt[c] for c in with_gt)
    p, r, f = [], [], []
    for i in range(len(grid)):
        tp = sum(int(class_curves[c]["_tp"][i]) for c in range(len(class_curves)))
        kept = sum(int(class_curves[c]["_kept"][i]) for c in range(len(class_curves)))
        p.append(_safe_div(tp, kept))
        r.append(_safe_div(tp, total_gt))
        f.append(_f1_from_counts(tp, kept - tp, total_gt - tp))
    return {"precision": p, "recall": r, "f1": f}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def detection_options(evaluation: Mapping | None) -> dict:
    """Keyword arguments for evaluate_detections from the evaluation block of the
    settings. A missing key gets the documented default."""
    evaluation = evaluation or {}
    return {
        "iou_thresholds": list(evaluation.get("iou_thresholds", DEFAULT_IOU_THRESHOLDS)),
        "max_detections": int(evaluation.get("max_detections", DEFAULT_MAX_DETECTIONS)),
        "conf_threshold": float(evaluation.get("report_confidence", DEFAULT_REPORT_CONFIDENCE)),
    }


def acoustic_options(evaluation: Mapping | None) -> dict:
    """Keyword arguments for evaluate_acoustic from the evaluation block of the settings."""
    evaluation = evaluation or {}
    return {
        "threshold": float(evaluation.get("report_confidence", DEFAULT_REPORT_CONFIDENCE)),
        "calibration_bins": int(evaluation.get("calibration_bins", CALIBRATION_BINS)),
    }


# ---------------------------------------------------------------------------
# Selecting images to show, restricting to shared classes, headline figures
# ---------------------------------------------------------------------------


def select_examples(image_outcomes: Mapping[str, Mapping[str, int]], per_kind: int, seed: Any) -> dict:
    """Choose images to show as correct, false alarm and missed.

    image_outcomes is the image_outcomes entry of evaluate_detections. An image is
    correct when it has at least one true positive and no false alarm or miss, a
    false alarm when it has any unmatched prediction, and missed when it has any
    unmatched ground-truth box. One image can be both a false alarm and missed.
    The choice is a seeded sample, so the same inputs always give the same images.
    """
    per_kind = max(0, int(per_kind))
    kinds = {
        "correct": lambda o: o["tp"] > 0 and o["fp"] == 0 and o["fn"] == 0,
        "false_alarm": lambda o: o["fp"] > 0,
        "missed": lambda o: o["fn"] > 0,
    }
    out = {}
    for kind, test in kinds.items():
        pool = sorted(i for i, o in image_outcomes.items() if test(o))
        rng = random.Random(f"{seed}:{kind}")
        chosen = pool if len(pool) <= per_kind else rng.sample(pool, per_kind)
        out[kind] = sorted(chosen)
    return out


def restrict_detection(gt: Mapping, pred: Mapping, class_names: Sequence[str], keep: Sequence[int]):
    """Keep only the listed classes, renumbered 0..k-1 in the order given.

    Used to score two models on the classes they share. Returns (gt, pred,
    class_names) in the same shapes evaluate_detections takes.
    """
    keep = [int(k) for k in keep]
    if len(set(keep)) != len(keep):
        raise MetricsError("keep must not repeat a class")
    names = [str(n) for n in class_names]
    for k in keep:
        if not (0 <= k < len(names)):
            raise MetricsError(f"class {k} is outside 0..{len(names) - 1}")
    remap = {old: new for new, old in enumerate(keep)}
    new_gt = {img: [(remap[c], b) for c, b in (items or []) if c in remap] for img, items in gt.items()}
    new_pred = {img: [(remap[c], s, b) for c, s, b in (items or []) if c in remap] for img, items in pred.items()}
    return new_gt, new_pred, [names[k] for k in keep]


def restrict_acoustic(y_true: Any, scores: Any, class_names: Sequence[str], keep: Sequence[int]):
    """Keep only the listed class columns, in the order given."""
    keep = [int(k) for k in keep]
    y, s, names = _check_acoustic(y_true, scores, class_names)
    for k in keep:
        if not (0 <= k < len(names)):
            raise MetricsError(f"class {k} is outside 0..{len(names) - 1}")
    return y[:, keep], s[:, keep], [names[k] for k in keep]


def headline(metrics: Mapping) -> dict:
    """The few figures kept with a registered model. All are plain numbers or None."""
    kind = metrics.get("kind")
    if kind == "detection":
        at = metrics["at_confidence"]
        return {
            "kind": kind,
            "map50": metrics["map50"],
            "map50_95": metrics["map50_95"],
            "report_confidence": at["confidence"],
            "macro_precision": at["macro_precision"],
            "macro_recall": at["macro_recall"],
            "macro_f1": at["macro_f1"],
            "top1_accuracy_on_matched_boxes": metrics["top1_accuracy_on_matched_boxes"]["value"],
        }
    if kind == "acoustic":
        at = metrics["at_threshold"]
        return {
            "kind": kind,
            "cmap": metrics["cmap"],
            "macro_roc_auc": metrics["macro_roc_auc"],
            "threshold": at["threshold"],
            "macro_precision": at["macro_precision"],
            "macro_recall": at["macro_recall"],
            "macro_f1": at["macro_f1"],
            "top1_accuracy": metrics["top1_accuracy"]["value"],
        }
    raise MetricsError(f"unknown metrics kind {kind!r}")


# ---------------------------------------------------------------------------
# Acoustic curves
# ---------------------------------------------------------------------------


def binary_curve(y: np.ndarray, s: np.ndarray):
    """Cumulative false and true positives at each distinct score, highest first.

    Returns (fps, tps, thresholds). A prediction is positive when score >= its
    threshold, so tied scores enter together.
    """
    y = np.asarray(y).astype(np.int64).reshape(-1)
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    if y.shape != s.shape:
        raise MetricsError("labels and scores must have the same length")
    if y.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    order = np.argsort(-s, kind="mergesort")
    s_o, y_o = s[order], y[order]
    distinct = np.nonzero(np.diff(s_o))[0]
    idx = np.r_[distinct, y_o.size - 1]
    tps = np.cumsum(y_o)[idx].astype(np.float64)
    fps = (1 + idx - tps).astype(np.float64)
    return fps, tps, s_o[idx]


def average_precision(y: np.ndarray, s: np.ndarray) -> float | None:
    """Step-wise area under the precision-recall curve, the sum over thresholds of
    the gain in recall times the precision at that threshold. None without positives."""
    fps, tps, _ = binary_curve(y, s)
    n_pos = float(np.asarray(y).sum())
    if n_pos == 0 or tps.size == 0:
        return None
    precision = tps / (tps + fps)
    recall = tps / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_points(y: np.ndarray, s: np.ndarray):
    """ROC curve as (false positive rate, true positive rate), from (0, 0)."""
    fps, tps, thr = binary_curve(y, s)
    y = np.asarray(y)
    n_pos = float(y.sum())
    n_neg = float(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    return np.r_[0.0, fps / n_neg], np.r_[0.0, tps / n_pos], thr


def roc_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    """Area under the ROC curve by the trapezoid rule. None unless the class has
    both positives and negatives."""
    pts = roc_points(y, s)
    if pts is None:
        return None
    fpr, tpr, _ = pts
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))


# ---------------------------------------------------------------------------
# Calibration (Wood and Kahl 2024)
# ---------------------------------------------------------------------------

CALIBRATION_STATUS_OK = "ok"


def _calibration_feature(scores: np.ndarray, feature: str, eps: float) -> tuple:
    if feature not in ("auto", "logit", "raw"):
        raise MetricsError("feature must be auto, logit or raw")
    in_unit = bool(scores.size) and bool(np.all((scores >= 0.0) & (scores <= 1.0)))
    chosen = ("logit" if in_unit else "raw") if feature == "auto" else feature
    if chosen == "logit" and not in_unit:
        raise MetricsError("the logit feature needs scores between 0 and 1")
    x = _logit(scores, eps) if chosen == "logit" else np.asarray(scores, dtype=np.float64)
    return chosen, x


def fit_calibration(
    scores: Any,
    y: Any,
    *,
    target_probability: float = 0.90,
    feature: str = "auto",
    min_positives: int = 5,
    min_negatives: int = 5,
    eps: float = 1e-6,
    max_iterations: int = 100,
) -> dict:
    """Logistic calibration of one class's scores.

    Fits P(present) = sigmoid(a * x + b) where x is the logit of the score (or the
    raw score when scores are not between 0 and 1) by Newton's method, with
    Platt's smoothed targets so that perfectly separable scores still give a
    finite answer. Then solves for the raw score at which the fitted probability
    reaches target_probability, the per-class threshold. The scores come from a
    validation split, never from the test split.

    The status is "ok" or says why no calibration was possible: too few positives
    or negatives, no variation in the scores, or a fit whose probability does not
    rise with the score.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    t = np.asarray(y).astype(np.int64).reshape(-1)
    if s.shape != t.shape:
        raise MetricsError("scores and labels must have the same length")
    if not np.all(np.isfinite(s)):
        raise MetricsError("scores must be finite")
    target = float(target_probability)
    if not (0.0 < target < 1.0):
        raise MetricsError("target_probability must be between 0 and 1, exclusive")
    n_pos = int(t.sum())
    n_neg = int(t.size - n_pos)
    out = {
        "status": CALIBRATION_STATUS_OK,
        "a": None,
        "b": None,
        "feature": None,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "target_probability": target,
        "threshold": None,
        "threshold_within_observed_scores": None,
        "iterations": 0,
        "converged": None,
    }
    if n_pos < int(min_positives):
        out["status"] = "too_few_positives"
        return out
    if n_neg < int(min_negatives):
        out["status"] = "too_few_negatives"
        return out
    chosen, x = _calibration_feature(s, feature, eps)
    out["feature"] = chosen
    if float(np.ptp(x)) == 0.0:
        out["status"] = "no_variation"
        return out

    t_pos = (n_pos + 1.0) / (n_pos + 2.0)
    t_neg = 1.0 / (n_neg + 2.0)
    target_vec = np.where(t == 1, t_pos, t_neg)
    # Fit on the centred feature, which keeps the slope and the intercept from
    # trading off against each other, then move the intercept back.
    x_mean = float(x.mean())
    xc = x - x_mean

    def nll(a_: float, b_: float) -> float:
        z = a_ * xc + b_
        return float(np.sum(np.logaddexp(0.0, z) - target_vec * z))

    a, b = 0.0, math.log((n_pos + 1.0) / (n_neg + 1.0))
    current = nll(a, b)
    converged = False
    iterations = 0
    for iterations in range(1, int(max_iterations) + 1):
        p = _sigmoid(a * xc + b)
        resid = p - target_vec
        grad = np.array([np.sum(resid * xc), np.sum(resid)])
        if float(np.max(np.abs(grad))) < 1e-10:
            converged = True
            break
        w = p * (1.0 - p)
        hess = np.array([[np.sum(w * xc * xc), np.sum(w * xc)], [np.sum(w * xc), np.sum(w)]])
        hess += np.eye(2) * 1e-12
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        scale = 1.0
        improved = False
        while scale > 1e-12:
            cand = nll(a - scale * step[0], b - scale * step[1])
            if cand <= current - 1e-4 * scale * float(grad @ step):
                improved = True
                break
            scale *= 0.5
        if not improved:
            converged = float(np.max(np.abs(grad))) < 1e-5
            break
        a -= scale * step[0]
        b -= scale * step[1]
        gain = current - cand
        current = cand
        # The objective is a sum over every window, so it cannot be resolved below a
        # few parts in 1e13 of its size. Once it stops improving at that level and the
        # gradient is already tiny the fit has reached the limit of floating point.
        if gain <= 1e-13 * (1.0 + abs(current)) and float(np.max(np.abs(grad))) < 1e-5:
            converged = True
            break
    b = b - a * x_mean
    out["a"], out["b"] = float(a), float(b)
    out["iterations"] = iterations
    out["converged"] = bool(converged)
    if not (a > 0.0):
        out["status"] = "not_increasing"
        return out
    x_star = (math.log(target / (1.0 - target)) - b) / a
    threshold = float(_sigmoid(np.array([x_star]))[0]) if chosen == "logit" else float(x_star)
    out["threshold"] = threshold
    out["threshold_within_observed_scores"] = bool(float(s.min()) <= threshold <= float(s.max()))
    return out


def calibrated_probability(scores: Any, calibration: Mapping, eps: float = 1e-6) -> np.ndarray | None:
    """Apply a fitted calibration. None when the calibration did not succeed."""
    if calibration.get("status") != CALIBRATION_STATUS_OK or calibration.get("a") is None:
        return None
    s = np.asarray(scores, dtype=np.float64)
    x = _logit(s, eps) if calibration["feature"] == "logit" else s
    return _sigmoid(calibration["a"] * x + calibration["b"])


def fit_calibrations(y_true: Any, scores: Any, class_names: Sequence[str], **kwargs) -> list:
    """fit_calibration for every class column, each tagged with its class."""
    y, s, names = _check_acoustic(y_true, scores, class_names)
    out = []
    for c, name in enumerate(names):
        cal = fit_calibration(s[:, c], y[:, c], **kwargs)
        cal["class_index"] = c
        cal["class_name"] = name
        out.append(cal)
    return out


def reliability_bins(probabilities: Any, outcomes: Any, n_bins: int = CALIBRATION_BINS) -> dict:
    """Reliability diagram figures. Equal-width probability bins, a value equal to
    an upper edge belonging to the bin it closes. Also the expected calibration
    error, the count-weighted mean gap between predicted and observed frequency."""
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    y = np.asarray(outcomes).astype(np.int64).reshape(-1)
    n_bins = int(n_bins)
    if n_bins < 1:
        raise MetricsError("n_bins must be at least 1")
    if p.shape != y.shape:
        raise MetricsError("probabilities and outcomes must have the same length")
    if p.size and (not np.all(np.isfinite(p)) or p.min() < 0.0 or p.max() > 1.0):
        raise MetricsError("probabilities must be between 0 and 1")
    edges = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)
    ids = np.searchsorted(edges[1:-1], p)
    bins = []
    ece = 0.0
    for k in range(n_bins):
        mask = ids == k
        count = int(mask.sum())
        entry = {
            "lower": k / n_bins,
            "upper": (k + 1) / n_bins,
            "count": count,
            "mean_predicted": float(p[mask].mean()) if count else None,
            "fraction_positive": float(y[mask].mean()) if count else None,
        }
        if count:
            ece += count / p.size * abs(entry["fraction_positive"] - entry["mean_predicted"])
        bins.append(entry)
    return {"bins": bins, "n": int(p.size), "expected_calibration_error": _num(ece) if p.size else None}


# ---------------------------------------------------------------------------
# Acoustic evaluation
# ---------------------------------------------------------------------------


def _check_acoustic(y_true: Any, scores: Any, class_names: Sequence[str]):
    names = [str(n) for n in class_names]
    try:
        y = np.asarray(y_true)
        s = np.asarray(scores, dtype=np.float64)
    except (TypeError, ValueError):
        raise MetricsError("labels and scores must be numeric arrays")
    if y.ndim != 2 or s.shape != y.shape:
        raise MetricsError("labels and scores must be two-dimensional arrays of the same shape, windows by classes")
    if y.shape[1] != len(names):
        raise MetricsError("the number of class names must match the number of columns")
    if y.shape[0] < 1 or y.shape[1] < 1:
        raise MetricsError("there must be at least one window and one class")
    if not np.all(np.isfinite(s)):
        raise MetricsError("scores must be finite")
    if not np.all((y == 0) | (y == 1)):
        raise MetricsError("labels must be 0 or 1")
    return y.astype(np.int64), s, names


def _counts_at(y: np.ndarray, s: np.ndarray, thr: float) -> tuple:
    pred = s >= thr
    tp = int(np.count_nonzero(pred & (y == 1)))
    fp = int(np.count_nonzero(pred & (y == 0)))
    fn = int(np.count_nonzero(~pred & (y == 1)))
    return tp, fp, fn


def evaluate_acoustic(
    y_true: Any,
    scores: Any,
    class_names: Sequence[str],
    *,
    threshold: float = DEFAULT_REPORT_CONFIDENCE,
    calibrations: Sequence[Mapping] | None = None,
    calibration_bins: int = CALIBRATION_BINS,
    threshold_grid_points: int = CONFIDENCE_GRID_POINTS,
    max_curve_points: int = CURVE_MAX_POINTS,
) -> dict:
    """Score a sound classifier.

    y_true is a windows-by-classes array of 0 and 1. A window with no 1 is a
    background window. scores is the model's score for every window and class.
    calibrations, when given, are the per-class results of fit_calibrations on the
    validation split, used for the reliability figures.
    """
    y, s, names = _check_acoustic(y_true, scores, class_names)
    n_win, n_cls = y.shape
    thr = float(threshold)
    if not math.isfinite(thr):
        raise MetricsError("threshold must be finite")
    if calibrations is not None and len(calibrations) != n_cls:
        raise MetricsError("calibrations must have one entry per class")
    if int(threshold_grid_points) < 2:
        raise MetricsError("threshold_grid_points must be at least 2")

    in_unit = bool(np.all((s >= 0.0) & (s <= 1.0)))
    lo, hi = (0.0, 1.0) if in_unit else (float(s.min()), float(s.max()))
    if hi <= lo:
        hi = lo + 1.0
    if in_unit:
        grid = np.arange(int(threshold_grid_points)) / (int(threshold_grid_points) - 1)
    else:
        grid = np.linspace(lo, hi, int(threshold_grid_points))

    n_pos = y.sum(axis=0)
    per_class, curve_rows = [], []
    for c in range(n_cls):
        yc, sc = y[:, c], s[:, c]
        P = int(n_pos[c])
        N = n_win - P
        tp, fp, fn = _counts_at(yc, sc, thr)
        row = {
            "class_index": c,
            "class_name": names[c],
            "n_positive": P,
            "n_negative": N,
            "ap": None,
            "roc_auc": None,
            "tp": tp, "fp": fp, "fn": fn if P else 0,
            "precision": _safe_div(tp, tp + fp),
            "recall": _safe_div(tp, P),
            "f1": _f1_from_counts(tp, fp, fn),
            "f1_optimal": None,
        }
        curves = {"class_index": c, "class_name": names[c], "precision_recall": None, "roc": None, "f1": None}
        if P > 0:
            fps, tps, thr_desc = binary_curve(yc, sc)
            precision = tps / (tps + fps)
            recall = tps / P
            row["ap"] = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
            f1_k = 2.0 * tps / (tps + fps + P)
            k = int(np.argmax(f1_k))
            row["f1_optimal"] = {
                "threshold": _num(thr_desc[k]),
                "f1": _num(f1_k[k]),
                "precision": _num(precision[k]),
                "recall": _num(recall[k]),
            }
            keep = downsample_points(len(thr_desc), max_curve_points)
            curves["precision_recall"] = {
                "recall": _num_list(recall[keep]),
                "precision": _num_list(precision[keep]),
                "thresholds": _num_list(thr_desc[keep]),
            }
            pos_sorted = np.sort(sc[yc == 1])
            neg_sorted = np.sort(sc[yc == 0])
            tp_g = P - np.searchsorted(pos_sorted, grid, side="left")
            fp_g = N - np.searchsorted(neg_sorted, grid, side="left")
            curves["f1"] = [_f1_from_counts(int(tp_g[i]), int(fp_g[i]), int(P - tp_g[i])) for i in range(len(grid))]
            if N > 0:
                roc = roc_points(yc, sc)
                fpr, tpr, _ = roc
                row["roc_auc"] = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))
                keep = downsample_points(len(fpr), max_curve_points)
                curves["roc"] = {"fpr": _num_list(fpr[keep]), "tpr": _num_list(tpr[keep])}
        per_class.append(row)
        curve_rows.append(curves)

    with_pos = [c for c in range(n_cls) if n_pos[c] > 0]
    aps = [per_class[c]["ap"] for c in with_pos]
    aucs = [per_class[c]["roc_auc"] for c in with_pos if per_class[c]["roc_auc"] is not None]
    prec_defined = [per_class[c]["precision"] for c in with_pos if per_class[c]["precision"] is not None]
    at_threshold = {
        "threshold": thr,
        "macro_precision": _num(np.mean(prec_defined)) if prec_defined else None,
        "macro_recall": _num(np.mean([per_class[c]["recall"] for c in with_pos])) if with_pos else None,
        "macro_f1": _num(np.mean([per_class[c]["f1"] for c in with_pos])) if with_pos else None,
    }
    macro_f1_grid = []
    for i in range(len(grid)):
        vals = [curve_rows[c]["f1"][i] for c in with_pos]
        macro_f1_grid.append(_num(np.mean(vals)) if vals else None)
    if with_pos:
        arr = np.array([v if v is not None else -1.0 for v in macro_f1_grid])
        best = len(grid) - 1 - int(np.argmax(arr[::-1]))
        shared = {"threshold": _num(grid[best]), "macro_f1": macro_f1_grid[best]}
        opt_vals = [per_class[c]["f1_optimal"]["f1"] for c in with_pos]
        per_class_opt_macro = _num(np.mean(opt_vals))
    else:
        shared = {"threshold": None, "macro_f1": None}
        per_class_opt_macro = None

    # Top-1 accuracy and confusion matrix with a background row and column.
    top_idx = np.argmax(s, axis=1)
    top_val = s[np.arange(n_win), top_idx]
    has_label = y.sum(axis=1) > 0
    correct = int(np.count_nonzero(has_label & (y[np.arange(n_win), top_idx] == 1)))
    n_labelled = int(has_label.sum())
    matrix = np.zeros((n_cls + 1, n_cls + 1), dtype=np.int64)
    for w in range(n_win):
        pred = int(top_idx[w]) if top_val[w] >= thr else n_cls
        if has_label[w]:
            if pred < n_cls and y[w, pred] == 1:
                truth = pred
            else:
                truth = int(np.nonzero(y[w])[0][0])
        else:
            truth = n_cls
        matrix[truth, pred] += 1

    # Reliability, raw scores and calibrated probabilities.
    reliability = {"raw": None, "calibrated": None}
    if in_unit:
        reliability["raw"] = reliability_bins(s.ravel(), y.ravel(), calibration_bins)
    if calibrations is not None:
        cols_p, cols_y = [], []
        for c in range(n_cls):
            prob = calibrated_probability(s[:, c], calibrations[c])
            if prob is not None:
                cols_p.append(prob)
                cols_y.append(y[:, c])
        if cols_p:
            reliability["calibrated"] = reliability_bins(np.concatenate(cols_p), np.concatenate(cols_y),
                                                         calibration_bins)

    return {
        "kind": "acoustic",
        "definition_version": METRICS_DEFINITION_VERSION,
        "settings": {
            "threshold": thr,
            "calibration_bins": int(calibration_bins),
            "scores_between_0_and_1": in_unit,
        },
        "counts": {
            "windows": n_win,
            "classes": n_cls,
            "classes_with_positives": len(with_pos),
            "labelled_windows": n_labelled,
            "background_windows": n_win - n_labelled,
        },
        "class_names": names,
        "cmap": _num(np.mean(aps)) if aps else None,
        "macro_roc_auc": _num(np.mean(aucs)) if aucs else None,
        "top1_accuracy": {
            "value": _safe_div(correct, n_labelled),
            "windows": n_labelled,
            "correct": correct,
            "definition": ACOUSTIC_TOP1_DEFINITION,
        },
        "per_class": per_class,
        "at_threshold": at_threshold,
        "f1_optimal": {
            "per_class_macro_f1": per_class_opt_macro,
            "shared_threshold": shared,
        },
        "confusion_matrix": {
            "labels": names + ["background"],
            "matrix": matrix.tolist(),
            "rows": "label",
            "columns": "prediction",
            "threshold": thr,
        },
        "curves": {
            "threshold_grid": _num_list(grid),
            "macro_f1": macro_f1_grid,
            "per_class": curve_rows,
        },
        "reliability": reliability,
        "calibration": [dict(c) for c in calibrations] if calibrations is not None else None,
    }
