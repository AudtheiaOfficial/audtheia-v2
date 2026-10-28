"""Verification for the evaluation metrics of the Train tab.

Path: tests/test_training_metrics.py

Two kinds of check. The first needs nothing beyond NumPy and the project: small
cases whose answers are worked out by hand, edge cases, invariants such as every
ground-truth box being counted exactly once in the confusion matrix, repeatable
output that survives JSON, and malformed input being refused with a clear
message.

The second compares Audtheia's own metric code with independent reference
implementations on seeded fixtures. Detection average precision, the precision
curve and the per-class figures are compared with pycocotools and with
faster-coco-eval, the confusion matrix with supervision, and the acoustic
average precision, ROC area, curves, F1, reliability bins and logistic
calibration with scikit-learn. A reference library that is not installed is
reported as skipped, never as passed, so a computer without them still runs the
first kind in full. The reference libraries are test-time tools only and the
application never imports them.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import math
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from audtheia.training import metrics as M  # noqa: E402

CHECKS = {"passed": 0, "failed": 0, "skipped": 0}


def check(label: str, condition: bool) -> None:
    CHECKS["passed" if condition else "failed"] += 1
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")


def skip(label: str, why: str) -> None:
    CHECKS["skipped"] += 1
    print(f"  SKIP  {label} ({why})")


def close(a, b, tol=1e-9) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol


def raises(fn, exc=Exception) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


def silent(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


# Reference libraries, each optional.
try:
    from pycocotools.coco import COCO as PyCOCO  # noqa: E402
    from pycocotools.cocoeval import COCOeval as PyCOCOeval  # noqa: E402
    HAVE_PYCOCO = True
except Exception:  # noqa: BLE001
    HAVE_PYCOCO = False
try:
    from faster_coco_eval import COCO as FastCOCO, COCOeval_faster  # noqa: E402
    HAVE_FAST = True
except Exception:  # noqa: BLE001
    HAVE_FAST = False
try:
    from sklearn import metrics as skm  # noqa: E402
    from sklearn.calibration import calibration_curve  # noqa: E402
    from sklearn.linear_model import LogisticRegression  # noqa: E402
    HAVE_SK = True
except Exception:  # noqa: BLE001
    HAVE_SK = False
try:
    import supervision as sv  # noqa: E402
    HAVE_SV = True
except Exception:  # noqa: BLE001
    HAVE_SV = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_detection_case(seed, n_images=40, n_classes=4, ties=True, absent_gt_class=None, empty_images=3):
    """Seeded ground truth and predictions with misses, wrong classes, duplicates,
    false alarms, background-only images and tied scores."""
    rng = np.random.RandomState(seed)
    gt, pred = {}, {}
    for i in range(n_images):
        items, dets = [], []
        n_gt = 0 if i < empty_images else int(rng.randint(0, 6))
        for _ in range(n_gt):
            cls = int(rng.randint(0, n_classes))
            if cls == absent_gt_class:
                cls = (cls + 1) % n_classes
            x1, y1 = int(rng.randint(0, 500)), int(rng.randint(0, 340))
            w, h = int(rng.randint(20, 140)), int(rng.randint(20, 140))
            box = [x1, y1, x1 + w, y1 + h]
            items.append((cls, box))
            if rng.rand() < 0.85:
                jit = rng.randint(-8, 9, size=4)
                dcls = cls if rng.rand() < 0.85 else int(rng.randint(0, n_classes))
                dbox = [box[0] + int(jit[0]), box[1] + int(jit[1]), box[2] + int(jit[2]), box[3] + int(jit[3])]
                dets.append((dcls, float(rng.uniform(0.2, 1.0)), dbox))
                if rng.rand() < 0.2:
                    dets.append((dcls, float(rng.uniform(0.1, 0.9)), [dbox[0] + 1, dbox[1], dbox[2], dbox[3]]))
        for _ in range(int(rng.randint(0, 4))):
            x1, y1 = int(rng.randint(0, 500)), int(rng.randint(0, 340))
            dets.append((int(rng.randint(0, n_classes)), float(rng.uniform(0.05, 0.9)),
                         [x1, y1, x1 + int(rng.randint(15, 120)), y1 + int(rng.randint(15, 120))]))
        if ties:
            dets = [(c, round(s, 2), b) for c, s, b in dets]
        gt[i + 1] = items
        pred[i + 1] = dets
    return gt, pred


def to_coco(gt, pred, names, empty_detections=False):
    images = [{"id": i, "width": 700, "height": 500} for i in gt]
    cats = [{"id": k + 1, "name": n} for k, n in enumerate(names)]
    anns, aid = [], 1
    for i, items in gt.items():
        for cls, b in items:
            w, h = b[2] - b[0], b[3] - b[1]
            anns.append({"id": aid, "image_id": i, "category_id": cls + 1, "bbox": [b[0], b[1], w, h],
                         "area": w * h, "iscrowd": 0})
            aid += 1
    dets = []
    for i, items in pred.items():
        for cls, s, b in items:
            dets.append({"image_id": i, "category_id": cls + 1, "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                         "score": s})
    return images, cats, anns, dets


def run_reference(gt, pred, names, thresholds, max_det, which):
    """Reference precision array of shape (T, R, K) at the given maximum detections."""
    images, cats, anns, dets = to_coco(gt, pred, names)
    cls = PyCOCO if which == "pycocotools" else FastCOCO
    ev_cls = PyCOCOeval if which == "pycocotools" else COCOeval_faster
    g = cls()
    g.dataset = {"images": images, "categories": cats, "annotations": anns}
    silent(g.createIndex)
    if dets:
        d = silent(g.loadRes, dets)
    else:
        d = cls()
        d.dataset = {"images": images, "categories": cats, "annotations": []}
        silent(d.createIndex)
    ev = ev_cls(g, d, "bbox")
    ev.params.imgIds = sorted(gt.keys())
    ev.params.iouThrs = np.array(thresholds, dtype=np.float64)
    ev.params.maxDets = [1, 2, max_det] if max_det > 2 else [1, 2, 3]
    silent(ev.evaluate)
    silent(ev.accumulate)
    return np.array(ev.eval["precision"])[:, :, :, 0, 2]


def compare_to_reference(label, gt, pred, names, thresholds=M.DEFAULT_IOU_THRESHOLDS, max_det=100):
    mine = M.evaluate_detections(gt, pred, names, iou_thresholds=thresholds, max_detections=max_det)
    for which, have in (("pycocotools", HAVE_PYCOCO), ("faster-coco-eval", HAVE_FAST)):
        if not have:
            skip(f"{label} against {which}", "not installed")
            continue
        prec = run_reference(gt, pred, names, thresholds, max_det, "pycocotools" if which == "pycocotools" else "fast")
        valid = prec > -1
        ref_map = float(prec[valid].mean()) if valid.any() else None
        check(f"{label}: mAP over thresholds equals {which}", close(mine["map50_95"], ref_map, 1e-9))
        ok = True
        for t in range(len(thresholds)):
            v = prec[t] > -1
            ref_t = float(prec[t][v].mean()) if v.any() else None
            ok &= close(mine["ap_by_iou"][t]["ap"], ref_t, 1e-9)
        check(f"{label}: AP at every IoU threshold equals {which}", ok)
        ok = True
        for k in range(len(names)):
            col = prec[:, :, k]
            if (col > -1).any():
                ok &= close(mine["per_class"][k]["ap50_95"], float(col.mean()), 1e-9)
                if 0.5 in thresholds:
                    ok &= close(mine["per_class"][k]["ap50"], float(col[list(thresholds).index(0.5)].mean()), 1e-9)
            else:
                ok &= mine["per_class"][k]["ap50_95"] is None and mine["per_class"][k]["ap50"] is None
        check(f"{label}: per-class AP equals {which}", ok)
        if 0.5 in thresholds and mine["curves"]["precision_recall"]:
            ok = True
            for k in range(len(names)):
                ref_curve = prec[list(thresholds).index(0.5), :, k]
                mine_curve = mine["curves"]["precision_recall"]["per_class"][k]["precision"]
                if (ref_curve > -1).any():
                    ok &= mine_curve is not None and np.allclose(mine_curve, ref_curve, atol=1e-9, rtol=0)
                else:
                    ok &= mine_curve is None
            check(f"{label}: 101-point precision curve equals {which}", ok)
    return mine


# ---------------------------------------------------------------------------
# Detection: hand-computed cases
# ---------------------------------------------------------------------------


def test_detection_hand():
    print("detection, worked by hand")
    names = ["a"]
    # Two ground-truth boxes, detections TP 0.9, FP 0.8, TP 0.7.
    gt = {1: [(0, [0, 0, 10, 10]), (0, [100, 100, 110, 110])]}
    pred = {1: [(0, 0.9, [0, 0, 10, 10]), (0, 0.8, [200, 200, 210, 210]), (0, 0.7, [100, 100, 110, 110])]}
    r = M.evaluate_detections(gt, pred, names)
    expected = (51 * 1.0 + 50 * (2.0 / 3.0)) / 101.0
    check("AP of TP, FP, TP with two ground-truth boxes is 84.33 percent of the 101-point rule",
          close(r["map50"], expected, 1e-12))
    check("every threshold gives the same AP when overlaps are exact", close(r["map50_95"], expected, 1e-12))
    pc = r["per_class"][0]
    check("at confidence 0.30 the counts are TP 2, FP 1, FN 0", (pc["tp"], pc["fp"], pc["fn"]) == (2, 1, 0))
    check("precision 2/3, recall 1, F1 0.8", close(pc["precision"], 2 / 3) and close(pc["recall"], 1.0)
          and close(pc["f1"], 0.8))
    check("the F1-optimal confidence is the lowest score that keeps both true positives",
          close(r["f1_optimal"]["per_class"][0]["confidence"], 0.7) and close(r["f1_optimal"]["per_class"][0]["f1"], 0.8))

    # Duplicate detections: the second on the same box is a false positive.
    gt = {1: [(0, [0, 0, 10, 10])]}
    pred = {1: [(0, 0.9, [0, 0, 10, 10]), (0, 0.8, [0, 0, 10, 10])]}
    r = M.evaluate_detections(gt, pred, names)
    pc = r["per_class"][0]
    check("a duplicate detection counts as a false positive", (pc["tp"], pc["fp"], pc["fn"]) == (1, 1, 0))
    check("AP is still 1 because the duplicate ranks below the true positive", close(r["map50"], 1.0, 1e-12))

    # IoU exactly on a threshold. Box areas 16 and 8 with the smaller inside: IoU 0.5.
    gt = {1: [(0, [0, 0, 4, 4])]}
    pred = {1: [(0, 0.9, [0, 0, 4, 2])]}
    r = M.evaluate_detections(gt, pred, names)
    check("IoU exactly 0.5 matches at 0.5", close(r["map50"], 1.0))
    check("and fails at 0.55, so mAP over 0.50 to 0.95 is 0.1", close(r["map50_95"], 0.1, 1e-12))
    check("the AP is reported per threshold", close(r["ap_by_iou"][0]["ap"], 1.0, 1e-12) and r["ap_by_iou"][1]["ap"] == 0.0)

    # No detections at all.
    gt = {1: [(0, [0, 0, 10, 10])], 2: []}
    r = M.evaluate_detections(gt, {}, ["a", "b"])
    check("no detections gives AP 0 for a class with ground truth and None for one without",
          r["per_class"][0]["ap50"] == 0.0 and r["per_class"][1]["ap50"] is None)
    check("no detections gives mAP 0", r["map50"] == 0.0 and r["map50_95"] == 0.0)
    pc = r["per_class"][0]
    check("recall 0 and precision undefined", pc["recall"] == 0.0 and pc["precision"] is None and pc["f1"] == 0.0)
    check("the missed box lands in the background column",
          r["confusion_matrix"]["matrix"] == [[0, 0, 1], [0, 0, 0], [0, 0, 0]])

    # No ground truth at all.
    gt = {1: [], 2: []}
    pred = {1: [(0, 0.9, [0, 0, 10, 10])]}
    r = M.evaluate_detections(gt, pred, ["a"])
    check("with no ground truth anywhere mAP is None, never 0 or NaN", r["map50"] is None and r["map50_95"] is None)
    check("the prediction is a false alarm in the background row",
          r["confusion_matrix"]["matrix"] == [[0, 0], [1, 0]])
    check("recall, F1 and AP are all undefined", r["per_class"][0]["recall"] is None and r["per_class"][0]["f1"] is None)
    check("the F1-optimal macro confidence is None", r["f1_optimal"]["macro_confidence"] is None)
    r = M.evaluate_detections({}, {}, ["a"])
    check("an empty evaluation gives None rather than failing", r["map50"] is None and r["counts"]["images"] == 0)

    # A predicted class with no ground truth is skipped in mAP but is a false alarm.
    gt = {1: [(0, [0, 0, 10, 10])]}
    pred = {1: [(0, 0.9, [0, 0, 10, 10]), (1, 0.95, [50, 50, 60, 60])]}
    r = M.evaluate_detections(gt, pred, ["a", "b"])
    check("a class without ground truth does not enter mAP", close(r["map50"], 1.0))
    check("its prediction counts as a false positive in the pooled figures",
          close(r["at_confidence"]["micro_precision"], 0.5))
    check("macro precision averages only classes with ground truth", close(r["at_confidence"]["macro_precision"], 1.0))
    grid = r["curves"]["confidence_grid"]
    i30 = min(range(len(grid)), key=lambda i: abs(grid[i] - 0.3))
    mi = r["curves"]["micro"]
    check("the pooled curve also counts the false alarm of a class without ground truth",
          close(mi["precision"][i30], 0.5) and close(mi["recall"][i30], 1.0) and close(mi["f1"][i30], 2 / 3, 1e-12))
    check("the macro curve does not", close(r["curves"]["macro"]["precision"][i30], 1.0))

    # Truncation to the maximum detections per image and class.
    gt = {1: [(0, [0, 0, 10, 10]), (0, [100, 100, 110, 110])]}
    pred = {1: [(0, 0.9, [300, 300, 310, 310]), (0, 0.8, [0, 0, 10, 10]), (0, 0.7, [100, 100, 110, 110])]}
    r1 = M.evaluate_detections(gt, pred, ["a"], max_detections=1)
    check("with a maximum of one detection only the top-scoring false positive is kept",
          r1["map50"] == 0.0 and r1["counts"]["predictions_evaluated"] == 1 and r1["counts"]["predictions"] == 3)
    r3 = M.evaluate_detections(gt, pred, ["a"], max_detections=3)
    check("raising the maximum keeps all three", r3["counts"]["predictions_evaluated"] == 3 and r3["map50"] > 0.5)

    # A tied-IoU match takes the later ground-truth box, as the COCO evaluation does.
    gt = {1: [(0, [0, 0, 10, 10]), (0, [0, 0, 10, 10])]}
    pred = {1: [(0, 0.9, [0, 0, 10, 10]), (0, 0.8, [0, 0, 10, 10])]}
    r = M.evaluate_detections(gt, pred, ["a"])
    check("two identical boxes each take one ground-truth box", r["per_class"][0]["tp"] == 2 and close(r["map50"], 1.0, 1e-12))

    # Equal overlap with two boxes: the COCO rule takes the later box, which leaves the
    # exact prediction that follows without a free box.
    gt = {1: [(0, [0, 0, 10, 10]), (0, [10, 0, 20, 10])]}
    pred = {1: [(0, 0.9, [5, 0, 15, 10]), (0, 0.8, [10, 0, 20, 10])]}
    r = M.evaluate_detections(gt, pred, ["a"], iou_thresholds=[0.3])
    pc = r["per_class"][0]
    check("a prediction equally close to two boxes takes the later one",
          (pc["tp"], pc["fp"], pc["fn"]) == (1, 1, 1) and close(r["map50_95"], 51 / 101, 1e-12))

    # A score exactly on a grid value is at or above it: the grid holds decimals as typed.
    gt = {1: [(0, [0, 0, 10, 10]), (0, [100, 100, 110, 110])]}
    pred = {1: [(0, 0.9, [0, 0, 10, 10]), (0, 0.8, [200, 200, 210, 210]), (0, 0.7, [100, 100, 110, 110])]}
    r = M.evaluate_detections(gt, pred, ["a"], conf_threshold=0.7)
    grid = r["curves"]["confidence_grid"]
    check("the confidence grid holds the decimals a person types", all(grid[i] == i / 100 for i in range(101)))
    check("a score of exactly 0.70 counts at confidence 0.70",
          r["per_class"][0]["tp"] == 2 and r["curves"]["per_class"][0]["f1"][70] == r["per_class"][0]["f1"])

    # Two confidences with the same best F1: the higher confidence is reported.
    gt = {1: [(0, [0, 0, 10, 10]), (0, [100, 0, 110, 10])]}
    pred = {1: [(0, 0.9, [0, 0, 10, 10]), (0, 0.8, [300, 0, 310, 10]), (0, 0.7, [400, 0, 410, 10]),
                (0, 0.6, [100, 0, 110, 10])]}
    r = M.evaluate_detections(gt, pred, ["a"])
    fo = r["f1_optimal"]["per_class"][0]
    check("when two confidences give the same best F1 the higher one is reported",
          close(fo["confidence"], 0.9) and close(fo["f1"], 2 / 3, 1e-12))


def test_confusion_hand():
    print("confusion matrix, worked by hand")
    names = ["a", "b", "c"]
    gt = {1: [(0, [0, 0, 10, 10]), (1, [100, 0, 110, 10]), (2, [200, 0, 210, 10])]}
    pred = {1: [
        (0, 0.9, [0, 0, 10, 10]),        # correct
        (0, 0.8, [100, 0, 110, 10]),     # right place, wrong class: b taken for a
        (2, 0.7, [400, 0, 410, 10]),     # false alarm of class c
        (1, 0.1, [200, 0, 210, 10]),     # below the confidence, ignored
    ]}
    r = M.evaluate_detections(gt, pred, names, conf_threshold=0.3, include_image_outcomes=True)
    expected = [
        [1, 0, 0, 0],   # a: found
        [1, 0, 0, 0],   # b: reported as a
        [0, 0, 0, 1],   # c: missed
        [0, 0, 1, 0],   # background: a false alarm of class c
    ]
    check("rows are the ground truth, columns the prediction, with a background row and column",
          r["confusion_matrix"]["matrix"] == expected)
    check("labels end with background", r["confusion_matrix"]["labels"] == ["a", "b", "c", "background"])
    t1 = r["top1_accuracy_on_matched_boxes"]
    check("top-1 accuracy on matched boxes is 1 of 2", t1["matched"] == 2 and t1["correct"] == 1 and close(t1["value"], 0.5))
    check("the definition is carried with the figure", "not average precision" in t1["definition"])
    o = r["image_outcomes"]["1"]
    check("image outcomes count the true positive, two unmatched predictions and two unmatched boxes",
          (o["tp"], o["fp"], o["fn"]) == (1, 2, 2))
    check("the diagonal equals the true positives of the per-class table",
          all(r["confusion_matrix"]["matrix"][c][c] == r["per_class"][c]["tp"] for c in range(3)))

    # Two wrongly labelled predictions over one box: the better overlap is the confusion.
    gt = {1: [(0, [0, 0, 100, 100])]}
    pred = {1: [(1, 0.9, [0, 0, 100, 60]), (2, 0.8, [0, 0, 100, 90])]}
    r = M.evaluate_detections(gt, pred, names)
    check("of two wrongly labelled predictions over one box the better overlap is the confusion "
          "and the other a false alarm",
          r["confusion_matrix"]["matrix"] == [[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]])

    # The documented difference from tools that match by overlap alone.
    gt = {1: [(0, [0, 0, 100, 100]), (0, [0, 70, 100, 170])]}
    # X overlaps both boxes, Y overlaps only the first more strongly.
    pred = {1: [(0, 0.9, [0, 30, 100, 130]), (0, 0.8, [0, 0, 100, 100])]}
    iou = M._iou_matrix(np.array([[0, 30, 100, 130]], float), np.array([[0, 0, 100, 100], [0, 70, 100, 170]], float))
    check("setup: X overlaps ground-truth box 1 at 0.54 and box 2 at 0.43 so the pair differ in order",
          close(iou[0, 0], 7000 / 13000, 1e-9) and iou[0, 1] > 0)
    r = M.evaluate_detections(gt, pred, ["a"], conf_threshold=0.3, match_iou=0.4)
    pc = r["per_class"][0]
    check("matching follows score order as the COCO evaluation does, so the higher-scored X takes box 1",
          (pc["tp"], pc["fp"], pc["fn"]) == (1, 1, 1))


def test_validation_and_hygiene():
    print("input checks and output hygiene")
    names = ["a", "b"]
    ok_gt = {1: [(0, [0, 0, 10, 10])]}
    check("a prediction for an image missing from the ground truth is refused",
          raises(lambda: M.evaluate_detections(ok_gt, {2: []}, names), M.MetricsError))
    check("a class outside the list is refused",
          raises(lambda: M.evaluate_detections({1: [(5, [0, 0, 1, 1])]}, {}, names), M.MetricsError))
    check("a box with no area in the ground truth is refused",
          raises(lambda: M.evaluate_detections({1: [(0, [5, 5, 5, 9])]}, {}, names), M.MetricsError))
    check("a non-finite box is refused",
          raises(lambda: M.evaluate_detections({1: [(0, [0, 0, float("nan"), 9])]}, {}, names), M.MetricsError))
    check("a non-finite score is refused",
          raises(lambda: M.evaluate_detections(ok_gt, {1: [(0, float("inf"), [0, 0, 1, 1])]}, names), M.MetricsError))
    check("a malformed ground-truth entry is refused",
          raises(lambda: M.evaluate_detections({1: [(0,)]}, {}, names), M.MetricsError))
    check("a malformed prediction entry is refused",
          raises(lambda: M.evaluate_detections(ok_gt, {1: [(0, 0.5)]}, names), M.MetricsError))
    check("a fractional class number is refused",
          raises(lambda: M.evaluate_detections({1: [(0.5, [0, 0, 1, 1])]}, {}, names), M.MetricsError))
    check("an empty class list is refused", raises(lambda: M.evaluate_detections(ok_gt, {}, []), M.MetricsError))
    check("an empty threshold list is refused",
          raises(lambda: M.evaluate_detections(ok_gt, {}, names, iou_thresholds=[]), M.MetricsError))
    check("a repeated threshold is refused",
          raises(lambda: M.evaluate_detections(ok_gt, {}, names, iou_thresholds=[0.5, 0.5]), M.MetricsError))
    check("a threshold of zero is refused",
          raises(lambda: M.evaluate_detections(ok_gt, {}, names, iou_thresholds=[0.0]), M.MetricsError))
    check("a confidence above 1 is refused",
          raises(lambda: M.evaluate_detections(ok_gt, {}, names, conf_threshold=1.5), M.MetricsError))
    check("a maximum of zero detections is refused",
          raises(lambda: M.evaluate_detections(ok_gt, {}, names, max_detections=0), M.MetricsError))

    gt, pred = make_detection_case(7)
    before = copy.deepcopy((gt, pred))
    a = M.evaluate_detections(gt, pred, ["a", "b", "c", "d"], include_image_outcomes=True)
    check("the inputs are not modified", (gt, pred) == before)
    b = M.evaluate_detections(gt, pred, ["a", "b", "c", "d"], include_image_outcomes=True)
    check("the same input gives the same output", a == b)
    text = json.dumps(a, allow_nan=False)
    check("the result survives JSON with no NaN or infinity", json.loads(text) == a)

    # Order independence with distinct scores.
    gt, pred = make_detection_case(11, ties=False)
    base = M.evaluate_detections(gt, pred, ["a", "b", "c", "d"])
    rng = np.random.RandomState(3)
    shuffled_pred = {}
    for img in rng.permutation(list(pred.keys())):
        items = list(pred[int(img)])
        rng.shuffle(items)
        shuffled_pred[int(img)] = items
    shuffled_gt = {int(img): list(gt[int(img)]) for img in rng.permutation(list(gt.keys()))}
    other = M.evaluate_detections(shuffled_gt, shuffled_pred, ["a", "b", "c", "d"])
    same = (close(base["map50"], other["map50"]) and close(base["map50_95"], other["map50_95"])
            and base["confusion_matrix"]["matrix"] == other["confusion_matrix"]["matrix"]
            and base["per_class"] == other["per_class"])
    check("image order and prediction order do not change the result when scores are distinct", same)


def test_detection_invariants():
    print("detection invariants on seeded fixtures")
    for seed in (1, 2, 3):
        gt, pred = make_detection_case(seed)
        names = ["a", "b", "c", "d"]
        r = M.evaluate_detections(gt, pred, names, conf_threshold=0.305)
        mat = np.array(r["confusion_matrix"]["matrix"])
        n_gt = [sum(1 for items in gt.values() for c, _ in items if c == k) for k in range(4)]
        check(f"seed {seed}: each ground-truth box appears exactly once across its matrix row",
              all(int(mat[k].sum()) == n_gt[k] for k in range(4)))
        kept = [0] * 4
        for items in pred.values():
            by = {}
            for c, s, b in items:
                by.setdefault(c, []).append(s)
            for c, ss in by.items():
                kept[c] += sum(1 for s in sorted(ss, reverse=True)[:100] if s >= 0.305)
        check(f"seed {seed}: each kept prediction appears exactly once across its matrix column",
              all(int(mat[:, k].sum()) == kept[k] for k in range(4)))
        check(f"seed {seed}: the diagonal equals the per-class true positives",
              all(int(mat[k, k]) == r["per_class"][k]["tp"] for k in range(4)))
        check(f"seed {seed}: tp + fn equals the ground-truth count",
              all(p["tp"] + p["fn"] == p["n_ground_truth"] for p in r["per_class"]))
        grid = r["curves"]["confidence_grid"]
        idx = min(range(len(grid)), key=lambda i: abs(grid[i] - 0.3))
        check(f"seed {seed}: the grid contains 0.30 for the reporting figure", abs(grid[idx] - 0.3) < 1e-12)
        r30 = M.evaluate_detections(gt, pred, names, conf_threshold=0.3)
        ok = True
        for k in range(4):
            row = r30["per_class"][k]
            cur = r30["curves"]["per_class"][k]
            if row["n_ground_truth"] > 0:
                ok &= close(row["precision"], cur["precision"][idx]) and close(row["recall"], cur["recall"][idx])
                ok &= close(row["f1"], cur["f1"][idx])
        check(f"seed {seed}: the table at 0.30 equals the curve at 0.30", ok)
        best = r["f1_optimal"]["per_class"]
        ok = True
        for k in range(4):
            if best[k] is None:
                continue
            ok &= best[k]["f1"] >= max(v for v in r["curves"]["per_class"][k]["f1"] if v is not None) - 1e-12
        check(f"seed {seed}: the F1-optimal figure is at least the best F1 on the grid", ok)
        check(f"seed {seed}: mAP lies between 0 and 1", 0.0 <= r["map50_95"] <= r["map50"] <= 1.0)


def test_detection_against_references():
    print("detection against pycocotools and faster-coco-eval")
    names = ["a", "b", "c", "d"]
    for seed in (1, 2, 3, 4):
        gt, pred = make_detection_case(seed)
        compare_to_reference(f"seed {seed} with tied scores", gt, pred, names)
    gt, pred = make_detection_case(5, ties=False)
    compare_to_reference("seed 5 with distinct scores", gt, pred, names)
    gt, pred = make_detection_case(6, absent_gt_class=3)
    mine = compare_to_reference("seed 6 with one class absent from the ground truth", gt, pred, names)
    check("the class absent from the ground truth has no AP", mine["per_class"][3]["ap50"] is None)
    gt, pred = make_detection_case(8, n_images=25)
    compare_to_reference("a maximum of 5 detections", gt, pred, names, max_det=5)
    compare_to_reference("a custom threshold list", gt, pred, names, thresholds=[0.5, 0.6, 0.75, 0.9])
    gt = {1: [(0, [0, 0, 4, 4])], 2: [(1, [0, 0, 8, 8])]}
    pred = {1: [(0, 0.9, [0, 0, 4, 2])], 2: [(1, 0.8, [0, 0, 8, 4])]}
    compare_to_reference("boxes with IoU exactly 0.5", gt, pred, ["a", "b"])
    gt, pred = make_detection_case(9, n_images=10)
    compare_to_reference("no detections", gt, {k: [] for k in gt}, names)
    pred_only = {k: v for k, v in make_detection_case(10, n_images=6)[1].items()}
    compare_to_reference("no ground truth", {k: [] for k in pred_only}, pred_only, names)
    gt, pred = make_detection_case(12, n_images=30, empty_images=30)
    compare_to_reference("background-only images with false alarms", gt, pred, names)


def test_confusion_against_supervision():
    print("confusion matrix against supervision")
    if not HAVE_SV:
        skip("confusion matrix against supervision", "not installed")
        return
    names = ["a", "b", "c"]
    rng = np.random.RandomState(21)
    agree = 0
    total = 6
    for case in range(total):
        # Boxes in separate grid cells so no two predictions compete for one box.
        gt, pred = {}, {}
        for img in range(1, 9):
            items, dets = [], []
            cells = rng.permutation(12)[:7]
            for n, cell in enumerate(cells):
                x1, y1 = (cell % 4) * 150, (cell // 4) * 150
                box = [x1, y1, x1 + 100, y1 + 100]
                kind = rng.randint(0, 5)
                cls = int(rng.randint(0, 3))
                if kind == 0:      # found with the right class
                    items.append((cls, box))
                    dets.append((cls, round(float(rng.uniform(0.4, 1.0)), 2), [x1 + 3, y1 - 2, x1 + 98, y1 + 101]))
                elif kind == 1:    # found with the wrong class
                    items.append((cls, box))
                    dets.append(((cls + 1) % 3, round(float(rng.uniform(0.4, 1.0)), 2), [x1 + 1, y1 + 1, x1 + 99, y1 + 99]))
                elif kind == 2:    # missed
                    items.append((cls, box))
                elif kind == 3:    # false alarm
                    dets.append((cls, round(float(rng.uniform(0.4, 1.0)), 2), box))
                else:              # found but under the confidence
                    items.append((cls, box))
                    dets.append((cls, 0.12, [x1 + 2, y1 + 2, x1 + 98, y1 + 98]))
            gt[img], pred[img] = items, dets
        mine = M.evaluate_detections(gt, pred, names, conf_threshold=0.305, match_iou=0.5)["confusion_matrix"]["matrix"]

        def to_sv(per_image, with_conf):
            out = []
            for img in sorted(per_image):
                it = per_image[img]
                if with_conf:
                    xyxy = np.array([b for _, _, b in it], dtype=float).reshape(-1, 4)
                    out.append(sv.Detections(xyxy=xyxy, class_id=np.array([c for c, _, _ in it], dtype=int),
                                             confidence=np.array([s for _, s, _ in it], dtype=float)))
                else:
                    xyxy = np.array([b for _, b in it], dtype=float).reshape(-1, 4)
                    out.append(sv.Detections(xyxy=xyxy, class_id=np.array([c for c, _ in it], dtype=int)))
            return out

        cm = sv.ConfusionMatrix.from_detections(to_sv(pred, True), to_sv(gt, False), classes=names,
                                                conf_threshold=0.305, iou_threshold=0.5)
        agree += int(np.array_equal(np.array(mine), np.array(cm.matrix)))
    check("where no predictions compete for one box the matrix equals supervision's, cell for cell",
          agree == total)

    # Hand case for the documented difference.
    gt = {1: [(0, [0, 0, 100, 100]), (0, [0, 70, 100, 170])]}
    pred = {1: [(0, 0.9, [0, 30, 100, 130]), (0, 0.8, [0, 0, 100, 100])]}
    mine = M.evaluate_detections(gt, pred, ["a"], conf_threshold=0.3, match_iou=0.4)["confusion_matrix"]["matrix"]
    g = sv.Detections(xyxy=np.array([[0, 0, 100, 100], [0, 70, 100, 170]], float), class_id=np.array([0, 0]))
    p = sv.Detections(xyxy=np.array([[0, 30, 100, 130], [0, 0, 100, 100]], float), class_id=np.array([0, 0]),
                      confidence=np.array([0.9, 0.8]))
    theirs = sv.ConfusionMatrix.from_detections([p], [g], classes=["a"], conf_threshold=0.3,
                                                iou_threshold=0.4).matrix
    check("with competing predictions the score-ordered rule finds 1 true positive, 1 false alarm and 1 miss",
          mine == [[1, 1], [1, 0]])
    # Tools that order matches by overlap alone can give another answer here, and different
    # versions of supervision differ, so the installed version's answer is printed, not asserted.
    print(f"  INFO  supervision {getattr(sv, '__version__', '?')} gives {theirs.astype(int).tolist()} "
          f"for the same case, ours gives {mine}")


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def test_detection_helpers():
    print("examples, restriction and headline figures")
    gt, pred = make_detection_case(31)
    names = ["a", "b", "c", "d"]
    r = M.evaluate_detections(gt, pred, names, include_image_outcomes=True)
    ex = M.select_examples(r["image_outcomes"], 4, seed=20260101)
    check("examples come in three kinds with at most the requested number",
          set(ex) == {"correct", "false_alarm", "missed"} and all(len(v) <= 4 for v in ex.values()))
    out = r["image_outcomes"]
    check("every chosen image matches its kind",
          all(out[i]["tp"] > 0 and out[i]["fp"] == 0 and out[i]["fn"] == 0 for i in ex["correct"])
          and all(out[i]["fp"] > 0 for i in ex["false_alarm"]) and all(out[i]["fn"] > 0 for i in ex["missed"]))
    check("the same seed gives the same images", ex == M.select_examples(r["image_outcomes"], 4, seed=20260101))
    check("a different seed can give different images",
          ex != M.select_examples(r["image_outcomes"], 4, seed=7) or len(out) < 12)
    check("zero examples gives empty lists", M.select_examples(out, 0, 1) == {"correct": [], "false_alarm": [], "missed": []})

    g2, p2, n2 = M.restrict_detection(gt, pred, names, [2, 0])
    check("restriction renumbers classes in the order given", n2 == ["c", "a"])
    check("restriction drops other classes",
          all(c in (0, 1) for items in g2.values() for c, _ in items)
          and all(c in (0, 1) for items in p2.values() for c, _, _ in items))
    r2 = M.evaluate_detections(g2, p2, n2)
    check("the AP of a kept class is unchanged by dropping the others (matching is per class)",
          close(r2["per_class"][0]["ap50"], r["per_class"][2]["ap50"])
          and close(r2["per_class"][1]["ap50"], r["per_class"][0]["ap50"]))
    check("restriction refuses a repeated or unknown class",
          raises(lambda: M.restrict_detection(gt, pred, names, [1, 1]), M.MetricsError)
          and raises(lambda: M.restrict_detection(gt, pred, names, [9]), M.MetricsError))
    h = M.headline(r)
    check("the headline carries the standard figures",
          set(h) >= {"map50", "map50_95", "macro_precision", "macro_recall", "macro_f1",
                     "top1_accuracy_on_matched_boxes"} and h["kind"] == "detection")
    check("downsampling keeps both ends and respects the cap",
          M.downsample_points(1000, 50)[0] == 0 and M.downsample_points(1000, 50)[-1] == 999
          and len(M.downsample_points(1000, 50)) <= 50 and M.downsample_points(10, 50) == list(range(10)))


# ---------------------------------------------------------------------------
# Acoustic
# ---------------------------------------------------------------------------


def make_acoustic_case(seed, n=320, n_cls=6, decimals=3):
    rng = np.random.RandomState(seed)
    y = np.zeros((n, n_cls), dtype=int)
    for w in range(n):
        if rng.rand() < 0.8:
            y[w, int(rng.randint(0, n_cls - 1))] = 1       # last class never has a positive
            if rng.rand() < 0.05:
                y[w, int(rng.randint(0, n_cls - 1))] = 1
    s = np.zeros((n, n_cls))
    for c in range(n_cls):
        strength = rng.uniform(0.8, 2.5)
        raw = rng.normal(-1.5, 1.0, size=n) + strength * y[:, c] * rng.uniform(0.5, 1.5, size=n)
        s[:, c] = 1.0 / (1.0 + np.exp(-raw))
    if decimals is not None:
        s = np.round(s, decimals)
    return y, s


def test_acoustic_hand():
    print("acoustic, worked by hand")
    names = ["a", "b"]
    y = np.array([[1, 0], [1, 0], [0, 1], [0, 0], [0, 1]])
    s = np.array([[0.9, 0.2], [0.4, 0.3], [0.1, 0.8], [0.7, 0.6], [0.2, 0.3]])
    r = M.evaluate_acoustic(y, s, names, threshold=0.5)
    # Class a: scores 0.9(+) 0.7(-) 0.4(+) 0.2 0.1(-): AP = 1*0.5 + (2/3)*0.5
    check("average precision of class a", close(r["per_class"][0]["ap"], 0.5 + (2 / 3) * 0.5, 1e-12))
    check("average precision of class b", close(r["per_class"][1]["ap"], 0.5 * 1.0 + 0.5 * 0.5, 1e-12))
    check("class-mean average precision is the mean over classes with positives",
          close(r["cmap"], (r["per_class"][0]["ap"] + r["per_class"][1]["ap"]) / 2))
    check("ROC area of class a is 5/6", close(r["per_class"][0]["roc_auc"], 5 / 6))
    pc = r["per_class"][0]
    check("class a at threshold 0.5 has TP 1, FP 1, FN 1", (pc["tp"], pc["fp"], pc["fn"]) == (1, 1, 1))
    check("precision 0.5, recall 0.5, F1 0.5", close(pc["precision"], 0.5) and close(pc["recall"], 0.5) and close(pc["f1"], 0.5))
    check("the F1-optimal threshold of class a is 0.4 with F1 0.8 (both positives, one false alarm)",
          close(pc["f1_optimal"]["threshold"], 0.4) and close(pc["f1_optimal"]["f1"], 0.8, 1e-12))
    y3 = np.array([[1], [0], [0], [1]])
    s3 = np.array([[0.9], [0.8], [0.7], [0.6]])
    fo = M.evaluate_acoustic(y3, s3, ["a"])["per_class"][0]["f1_optimal"]
    check("when two thresholds give the same best F1 the higher one is reported",
          close(fo["threshold"], 0.9) and close(fo["f1"], 2 / 3, 1e-12))
    t1 = r["top1_accuracy"]
    check("top-1 accuracy over the four labelled windows is 4 of 4, whatever the threshold",
          t1["windows"] == 4 and t1["correct"] == 4 and close(t1["value"], 1.0))
    check("the background window is counted apart", r["counts"]["background_windows"] == 1)
    expected = [
        [1, 0, 1],   # truth a: window 0 found, window 1 below threshold so background
        [0, 1, 1],   # truth b: window 2 found, window 4 below threshold so background
        [1, 0, 0],   # background window 3: top score 0.7 is class a
    ]
    check("confusion matrix with background row and column", r["confusion_matrix"]["matrix"] == expected)

    # A class that never fires has no precision, and the macro precision leaves it out.
    yq = np.array([[1, 0], [0, 1]])
    sq = np.array([[0.2, 0.1], [0.1, 0.9]])
    r = M.evaluate_acoustic(yq, sq, names, threshold=0.5)
    check("a class that never fires has recall 0 and no precision",
          r["per_class"][0]["precision"] is None and r["per_class"][0]["recall"] == 0.0)
    check("macro precision leaves it out while macro recall and F1 keep it",
          close(r["at_threshold"]["macro_precision"], 1.0) and close(r["at_threshold"]["macro_recall"], 0.5)
          and close(r["at_threshold"]["macro_f1"], 0.5))

    # A class without positives is skipped, never turned into zero.
    y2 = np.array([[1, 0], [0, 0], [1, 0]])
    s2 = np.array([[0.9, 0.5], [0.1, 0.4], [0.8, 0.6]])
    r = M.evaluate_acoustic(y2, s2, names)
    check("a class without positives has no AP, AUC or F1",
          r["per_class"][1]["ap"] is None and r["per_class"][1]["roc_auc"] is None and r["per_class"][1]["f1"] is None)
    check("cmAP and macro AUC skip it", close(r["cmap"], 1.0) and close(r["macro_roc_auc"], 1.0))
    check("a class with every window positive has an AP but no ROC area",
          M.roc_auc(np.ones(5), np.arange(5)) is None and close(M.average_precision(np.ones(5), np.arange(5)), 1.0))
    check("average precision with no positives is None", M.average_precision(np.zeros(4), np.arange(4)) is None)

    # Tied scores enter together.
    check("tied scores enter the curve together",
          close(M.average_precision(np.array([1, 0, 1, 0]), np.array([0.5, 0.5, 0.5, 0.5])), 0.5))
    # Raw scores outside 0..1 still work, and the grid follows their range.
    r = M.evaluate_acoustic(y, s * 10 - 3, names, threshold=2.0)
    check("the threshold grid for scores in 0..1 holds decimals as typed",
          all(M.evaluate_acoustic(y, s, names)["curves"]["threshold_grid"][i] == i / 100 for i in range(101)))
    check("scores outside 0..1 are accepted and flagged", r["settings"]["scores_between_0_and_1"] is False
          and r["reliability"]["raw"] is None and r["curves"]["threshold_grid"][0] < 0)

    # Validation.
    check("a shape mismatch is refused", raises(lambda: M.evaluate_acoustic(y, s[:, :1], names), M.MetricsError))
    check("a name count mismatch is refused", raises(lambda: M.evaluate_acoustic(y, s, ["a"]), M.MetricsError))
    bad = y.copy()
    bad[0, 0] = 2
    check("a label other than 0 or 1 is refused", raises(lambda: M.evaluate_acoustic(bad, s, names), M.MetricsError))
    nan = s.copy()
    nan[0, 0] = float("nan")
    check("a NaN score is refused", raises(lambda: M.evaluate_acoustic(y, nan, names), M.MetricsError))
    check("an empty set is refused", raises(lambda: M.evaluate_acoustic(np.zeros((0, 2)), np.zeros((0, 2)), names),
                                            M.MetricsError))
    check("calibrations must match the class count",
          raises(lambda: M.evaluate_acoustic(y, s, names, calibrations=[{}]), M.MetricsError))

    yy, ss = make_acoustic_case(5)
    names6 = [f"c{i}" for i in range(6)]
    cals = M.fit_calibrations(yy, ss, names6)
    full = M.evaluate_acoustic(yy, ss, names6, calibrations=cals)
    check("the acoustic result survives JSON with no NaN", json.loads(json.dumps(full, allow_nan=False)) == full)
    check("the same input gives the same output", full == M.evaluate_acoustic(yy, ss, names6, calibrations=cals))
    check("the headline carries the standard figures",
          set(M.headline(full)) >= {"cmap", "macro_roc_auc", "macro_f1", "top1_accuracy"})
    g, sq, nm = M.restrict_acoustic(yy, ss, names6, [3, 1])
    check("column restriction keeps the order given", nm == ["c3", "c1"] and np.array_equal(g[:, 0], yy[:, 3]))
    check("downsampled curves respect the cap",
          all(len(c["precision_recall"]["recall"]) <= M.CURVE_MAX_POINTS
              for c in full["curves"]["per_class"] if c["precision_recall"]))
    rowsum = np.array(full["confusion_matrix"]["matrix"]).sum()
    check("every window appears once in the confusion matrix", int(rowsum) == yy.shape[0])


def test_acoustic_against_sklearn():
    print("acoustic against scikit-learn")
    if not HAVE_SK:
        skip("acoustic against scikit-learn", "not installed")
        return
    for seed, decimals in ((1, 3), (2, 2), (3, None)):
        y, s = make_acoustic_case(seed, decimals=decimals)
        names = [f"c{i}" for i in range(6)]
        r = M.evaluate_acoustic(y, s, names, threshold=0.4)
        tag = f"seed {seed}" + (" with tied scores" if decimals else "")
        ok_ap = ok_auc = ok_f1 = ok_opt = ok_pr = ok_roc = True
        aps, aucs, ref_f1, ref_prec, ref_rec, ref_opt = [], [], [], [], [], []
        for c in range(6):
            P = int(y[:, c].sum())
            row = r["per_class"][c]
            if P == 0:
                ok_ap &= row["ap"] is None and row["roc_auc"] is None
                continue
            ref_ap = skm.average_precision_score(y[:, c], s[:, c])
            ok_ap &= close(row["ap"], ref_ap, 1e-12)
            aps.append(ref_ap)
            ref_auc = skm.roc_auc_score(y[:, c], s[:, c])
            ok_auc &= close(row["roc_auc"], ref_auc, 1e-12)
            aucs.append(ref_auc)
            pred = (s[:, c] >= 0.4).astype(int)
            ok_f1 &= close(row["f1"], skm.f1_score(y[:, c], pred, zero_division=0), 1e-12)
            ref_f1.append(skm.f1_score(y[:, c], pred, zero_division=0))
            ref_rec.append(skm.recall_score(y[:, c], pred, zero_division=0))
            if pred.sum():
                ref_prec.append(skm.precision_score(y[:, c], pred, zero_division=0))
                ok_f1 &= close(row["precision"], skm.precision_score(y[:, c], pred, zero_division=0), 1e-12)
            ok_f1 &= close(row["recall"], skm.recall_score(y[:, c], pred, zero_division=0), 1e-12)
            # F1-optimal against a brute-force search over every distinct score.
            best_f1, best_thr = -1.0, None
            for t in sorted(set(s[:, c].tolist()), reverse=True):
                f = skm.f1_score(y[:, c], (s[:, c] >= t).astype(int), zero_division=0)
                if f > best_f1 + 1e-15:
                    best_f1, best_thr = f, t
            ok_opt &= close(row["f1_optimal"]["f1"], best_f1, 1e-12) and close(row["f1_optimal"]["threshold"], best_thr)
            ref_opt.append(best_f1)
            # Curves against scikit-learn's own, truncated where recall first reaches 1.
            prec, rec, thr = skm.precision_recall_curve(y[:, c], s[:, c])
            fps, tps, th = M.binary_curve(y[:, c], s[:, c])
            mp, mr = tps / (tps + fps), tps / P
            # Older scikit-learn stops at the first point with full recall, newer keeps every
            # threshold. Either way the points it returns must equal ours from the top.
            cut = int(np.argmax(tps == P)) + 1
            n_sk = len(thr)
            ok_pr &= n_sk >= cut and n_sk <= len(th)
            ok_pr &= np.allclose(mp[:n_sk], prec[:-1][::-1], atol=1e-12) and np.allclose(mr[:n_sk], rec[:-1][::-1], atol=1e-12)
            ok_pr &= np.allclose(th[:n_sk], thr[::-1], atol=1e-12)
            if P < len(y):
                fpr_r, tpr_r, _ = skm.roc_curve(y[:, c], s[:, c], drop_intermediate=False)
                fpr_m, tpr_m, _ = M.roc_points(y[:, c], s[:, c])
                ok_roc &= np.allclose(fpr_m, fpr_r, atol=1e-12) and np.allclose(tpr_m, tpr_r, atol=1e-12)
        check(f"{tag}: average precision per class equals scikit-learn", ok_ap)
        check(f"{tag}: ROC area per class equals scikit-learn", ok_auc)
        check(f"{tag}: cmAP is the mean of scikit-learn's per-class AP", close(r["cmap"], float(np.mean(aps)), 1e-12))
        check(f"{tag}: macro ROC area is the mean of scikit-learn's per-class AUC", close(r["macro_roc_auc"], float(np.mean(aucs)), 1e-12))
        check(f"{tag}: precision, recall and F1 at the threshold equal scikit-learn", ok_f1)
        at = r["at_threshold"]
        check(f"{tag}: macro F1, recall and precision are means over classes with positives",
              close(at["macro_f1"], float(np.mean(ref_f1)), 1e-12) and close(at["macro_recall"], float(np.mean(ref_rec)), 1e-12)
              and close(at["macro_precision"], float(np.mean(ref_prec)), 1e-12))
        check(f"{tag}: the per-class optimal macro F1 is the mean of the brute-force optima",
              close(r["f1_optimal"]["per_class_macro_f1"], float(np.mean(ref_opt)), 1e-12))
        check(f"{tag}: the F1-optimal threshold and F1 equal a brute-force search", ok_opt)
        check(f"{tag}: the precision-recall curve equals scikit-learn's", ok_pr)
        check(f"{tag}: the ROC curve equals scikit-learn's", ok_roc)
        # F1 curve on the grid.
        grid = r["curves"]["threshold_grid"]
        ok = True
        for c in range(6):
            f = r["curves"]["per_class"][c]["f1"]
            if int(y[:, c].sum()) == 0:
                ok &= f is None
                continue
            for i in (0, 17, 50, 83, 100):
                ok &= close(f[i], skm.f1_score(y[:, c], (s[:, c] >= grid[i]).astype(int), zero_division=0), 1e-12)
        check(f"{tag}: the F1 curve on the threshold grid equals scikit-learn", ok)
        # Top-1 against a direct count.
        top = s.argmax(axis=1)
        lab = y.sum(axis=1) > 0
        ref = float(np.mean(y[np.arange(len(y)), top][lab]))
        check(f"{tag}: top-1 accuracy equals a direct count", close(r["top1_accuracy"]["value"], ref, 1e-12))

    # Reliability bins against scikit-learn's calibration curve.
    rng = np.random.RandomState(4)
    p = np.round(rng.beta(2, 5, size=500), 3)
    yy = (rng.rand(500) < p * 1.3).astype(int)
    rel = M.reliability_bins(p, yy, 10)
    nonempty = [b for b in rel["bins"] if b["count"]]
    pt, pp = calibration_curve(yy, p, n_bins=10, strategy="uniform")
    check("reliability bins equal scikit-learn's calibration curve",
          len(nonempty) == len(pt) and np.allclose([b["fraction_positive"] for b in nonempty], pt, atol=1e-12)
          and np.allclose([b["mean_predicted"] for b in nonempty], pp, atol=1e-12))
    edge = np.array([0.0, 0.1, 0.2, 0.5, 0.9, 1.0, 0.1, 0.3])
    ye = np.array([0, 1, 0, 1, 1, 1, 0, 1])
    rel = M.reliability_bins(edge, ye, 10)
    pt, pp = calibration_curve(ye, edge, n_bins=10, strategy="uniform")
    ne = [b for b in rel["bins"] if b["count"]]
    check("values exactly on a bin edge fall in the same bin as scikit-learn puts them",
          len(ne) == len(pt) and np.allclose([b["fraction_positive"] for b in ne], pt, atol=1e-12))
    check("the bins cover all values", sum(b["count"] for b in rel["bins"]) == len(edge))
    ece = sum(b["count"] / 500 * abs(b["fraction_positive"] - b["mean_predicted"]) for b in
              M.reliability_bins(p, yy, 10)["bins"] if b["count"])
    check("the expected calibration error is the count-weighted gap",
          close(M.reliability_bins(p, yy, 10)["expected_calibration_error"], ece, 1e-12))
    check("probabilities outside 0..1 are refused", raises(lambda: M.reliability_bins([1.2], [1], 10), M.MetricsError))


def test_calibration():
    print("calibration")
    rng = np.random.RandomState(8)
    n = 400
    y = (rng.rand(n) < 0.35).astype(int)
    logit_score = rng.normal(-1.0, 1.0, n) + 2.2 * y
    s = 1.0 / (1.0 + np.exp(-logit_score))
    cal = M.fit_calibration(s, y, target_probability=0.9)
    check("a calibration fits", cal["status"] == "ok" and cal["converged"] is True and cal["a"] > 0)
    check("it used the logit of the score", cal["feature"] == "logit")
    p_at = M.calibrated_probability([cal["threshold"]], cal)[0]
    check("the calibrated probability at the threshold is the target", close(p_at, 0.9, 1e-9))
    pr = M.calibrated_probability(np.linspace(0.01, 0.99, 50), cal)
    check("calibrated probability rises with the score and stays inside 0..1",
          bool(np.all(np.diff(pr) > 0)) and pr.min() > 0 and pr.max() < 1)
    check("the threshold is inside the observed score range here", cal["threshold_within_observed_scores"] is True)
    cal2 = M.fit_calibration(s, y, target_probability=0.9999)
    inside = float(s.min()) <= cal2["threshold"] <= float(s.max())
    check("the flag for a threshold outside the observed scores matches the scores",
          cal2["status"] == "ok" and cal2["threshold_within_observed_scores"] is inside)

    if HAVE_SK:
        x = np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6)))
        P, N = int(y.sum()), int(len(y) - y.sum())
        tp, tn = (P + 1) / (P + 2), 1 / (N + 2)
        X2 = np.r_[x, x].reshape(-1, 1)
        y2 = np.r_[np.ones(n), np.zeros(n)]
        w2 = np.r_[np.where(y == 1, tp, tn), 1 - np.where(y == 1, tp, tn)]
        try:
            lr = LogisticRegression(C=np.inf, solver="lbfgs", tol=1e-14, max_iter=100000)
        except Exception:  # noqa: BLE001
            lr = LogisticRegression(penalty=None, solver="lbfgs", tol=1e-14, max_iter=100000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr.fit(X2, y2, sample_weight=w2)
        check("slope equals scikit-learn's unpenalised weighted fit", close(cal["a"], lr.coef_[0][0], 1e-5))
        check("intercept equals scikit-learn's unpenalised weighted fit", close(cal["b"], lr.intercept_[0], 1e-5))
    else:
        skip("calibration against scikit-learn", "not installed")

    # A battery of hard cases: heavy overlap, near-perfect separation, extreme scores.
    rng2 = np.random.RandomState(0)
    all_converged, all_increasing = True, True
    sk_ok = True
    for trial in range(60):
        n2 = int(rng2.randint(30, 260))
        P2 = int(rng2.randint(5, n2 - 5))
        y3 = np.r_[np.ones(P2), np.zeros(n2 - P2)].astype(int)
        kind = trial % 4
        if kind == 0:
            s3 = 1.0 / (1.0 + np.exp(-(rng2.normal(0.0, 3.0, n2) + 1.0 * y3)))
        elif kind == 1:
            s3 = np.where(y3 == 1, rng2.uniform(0.999, 1, n2), rng2.uniform(0, 0.001, n2))
        elif kind == 2:
            s3 = np.where(y3 == 1, rng2.uniform(0.5, 1, n2), rng2.uniform(0, 0.9, n2))
        else:
            s3 = np.where(y3 == 1, rng2.uniform(0.9999, 1, n2), rng2.uniform(0.9998, 0.99995, n2))
        c3 = M.fit_calibration(s3, y3)
        all_converged &= c3["status"] in ("ok", "not_increasing") and c3["converged"] is True
        all_increasing &= c3["status"] == "ok"
        if HAVE_SK and kind != 1:
            x3 = np.log(np.clip(s3, 1e-6, 1 - 1e-6) / (1 - np.clip(s3, 1e-6, 1 - 1e-6)))
            t_p, t_n = (P2 + 1) / (P2 + 2), 1 / (n2 - P2 + 2)
            w3 = np.where(y3 == 1, t_p, t_n)
            try:
                lr3 = LogisticRegression(C=np.inf, solver="lbfgs", tol=1e-14, max_iter=100000)
            except Exception:  # noqa: BLE001
                lr3 = LogisticRegression(penalty=None, solver="lbfgs", tol=1e-14, max_iter=100000)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lr3.fit(np.r_[x3, x3].reshape(-1, 1), np.r_[np.ones(n2), np.zeros(n2)],
                        sample_weight=np.r_[w3, 1 - w3])
            # compare the fitted probabilities rather than the raw coefficients, which trade off
            # against each other when the feature is nearly constant
            p_mine = 1.0 / (1.0 + np.exp(-(c3["a"] * x3 + c3["b"])))
            p_sk = lr3.predict_proba(x3.reshape(-1, 1))[:, 1]
            sk_ok &= bool(np.max(np.abs(p_mine - p_sk)) < 1e-4)
    check("every calibration in a battery of hard cases converges", all_converged)
    check("each of them has a probability that rises with the score", all_increasing)
    if HAVE_SK:
        check("their fitted probabilities equal scikit-learn's unpenalised weighted fit", sk_ok)

    # Perfectly separable scores still give a finite answer.
    ys = np.array([0] * 20 + [1] * 20)
    ss = np.r_[np.linspace(0.05, 0.4, 20), np.linspace(0.6, 0.95, 20)]
    cs = M.fit_calibration(ss, ys, target_probability=0.9)
    check("separable data gives a finite slope and intercept",
          cs["status"] == "ok" and math.isfinite(cs["a"]) and math.isfinite(cs["b"]) and cs["a"] > 0)
    check("and a threshold between the two groups' extremes or beyond", cs["threshold"] is not None)

    # Statuses.
    check("too few positives is reported", M.fit_calibration(s[:30], np.r_[np.ones(2), np.zeros(28)])["status"] == "too_few_positives")
    check("too few negatives is reported", M.fit_calibration(s[:30], np.r_[np.zeros(2), np.ones(28)])["status"] == "too_few_negatives")
    check("constant scores are reported", M.fit_calibration(np.full(40, 0.5), ys)["status"] == "no_variation")
    inv = M.fit_calibration(1.0 - ss, ys)
    check("a fit whose probability falls as the score rises has no threshold",
          inv["status"] == "not_increasing" and inv["threshold"] is None)
    check("no calibrated probability is produced without a successful fit",
          M.calibrated_probability([0.5], inv) is None)
    raw = M.fit_calibration(ss * 10 - 2, ys)
    check("scores outside 0..1 use the raw score", raw["feature"] == "raw" and raw["status"] == "ok")
    check("forcing the logit on such scores is refused",
          raises(lambda: M.fit_calibration(ss * 10 - 2, ys, feature="logit"), M.MetricsError))
    check("a target of 1 is refused", raises(lambda: M.fit_calibration(s, y, target_probability=1.0), M.MetricsError))
    check("a length mismatch is refused", raises(lambda: M.fit_calibration(s, y[:-1]), M.MetricsError))
    check("the calibration survives JSON", json.loads(json.dumps(cal, allow_nan=False)) == cal)
    yy, s6 = make_acoustic_case(5)
    cals = M.fit_calibrations(yy, s6, [f"c{i}" for i in range(6)])
    check("a class with no positives is reported, not dropped",
          cals[5]["status"] == "too_few_positives" and cals[5]["class_name"] == "c5" and len(cals) == 6)
    rel = M.evaluate_acoustic(yy, s6, [f"c{i}" for i in range(6)], calibrations=cals)["reliability"]
    check("calibrated reliability bins are produced from the classes that calibrated",
          rel["calibrated"] is not None and rel["calibrated"]["n"] > 0)


def test_options():
    print("settings")
    from audtheia.config import DEFAULT_TRAINING  # noqa: E402
    ev = DEFAULT_TRAINING["evaluation"]
    d = M.detection_options(ev)
    check("the documented settings give the documented detection arguments",
          d == {"iou_thresholds": list(M.DEFAULT_IOU_THRESHOLDS), "max_detections": 100, "conf_threshold": 0.30})
    a = M.acoustic_options(ev)
    check("and the acoustic ones", a == {"threshold": 0.30, "calibration_bins": 10})
    check("a missing block gets the same defaults", M.detection_options(None) == d and M.acoustic_options({}) == a)
    check("a changed setting is passed through",
          M.detection_options({"max_detections": 7, "report_confidence": 0.5})["max_detections"] == 7
          and M.detection_options({"report_confidence": 0.5})["conf_threshold"] == 0.5
          and M.acoustic_options({"calibration_bins": 5})["calibration_bins"] == 5)
    gt, pred = make_detection_case(3, n_images=10)
    r = M.evaluate_detections(gt, pred, ["a", "b", "c", "d"], **d)
    check("the arguments are accepted and recorded", r["settings"]["max_detections"] == 100
          and r["settings"]["report_confidence"] == 0.30 and r["settings"]["standard_iou_range"] is True)
    y, s = make_acoustic_case(1)
    ra = M.evaluate_acoustic(y, s, [f"c{i}" for i in range(6)], **a)
    check("the acoustic arguments are accepted and recorded", ra["settings"]["threshold"] == 0.30
          and ra["settings"]["calibration_bins"] == 10)
    check("the default acoustic threshold is the reporting confidence",
          M.evaluate_acoustic(y, s, [f"c{i}" for i in range(6)])["settings"]["threshold"] == M.DEFAULT_REPORT_CONFIDENCE)


def main() -> int:
    test_detection_hand()
    test_confusion_hand()
    test_validation_and_hygiene()
    test_detection_invariants()
    test_detection_against_references()
    test_confusion_against_supervision()
    test_detection_helpers()
    test_acoustic_hand()
    test_acoustic_against_sklearn()
    test_calibration()
    test_options()
    skipped = CHECKS["skipped"]
    if skipped:
        print(f"\nSkipped {skipped} comparison(s) because a reference library is not installed.")
    print(f"\nResult: {CHECKS['passed']} passed, {CHECKS['failed']} failed")
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
