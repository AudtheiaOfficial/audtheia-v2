"""Verification for the chart data and tables of the Train tab.

Path: tests/test_training_charts.py

Checks that stored metrics are turned into exactly the series, matrices and
tables the interface draws and exports. Worked cases are compared with figures
computed by hand, every chart's CSV rows are compared with its points, long
class lists are capped without losing the table, the challenger-versus-current
table carries numbers and differences and never a verdict, and the two modules
import nothing beyond the standard library and NumPy.

Uses the standard library, NumPy and the project itself.
"""

from __future__ import annotations

import ast
import csv
import io
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from audtheia.training import charts_data as C  # noqa: E402
from audtheia.training import metrics as M  # noqa: E402
from test_training_metrics import make_acoustic_case, make_detection_case  # noqa: E402

CHECKS = {"passed": 0, "failed": 0}


def check(label: str, condition: bool) -> None:
    CHECKS["passed" if condition else "failed"] += 1
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")


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


def by_id(charts: list) -> dict:
    return {c["id"]: c for c in charts}


def all_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from all_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from all_strings(v)


FORBIDDEN = ("better", "improved", "improvement", "worse", "outperform", "superior", "inferior", "winner")


def has_verdict_words(obj) -> bool:
    text = " ".join(all_strings(obj)).lower()
    return any(w in text for w in FORBIDDEN)


# ---------------------------------------------------------------------------


def hand_detection():
    gt = {1: [(0, [0, 0, 10, 10]), (0, [100, 100, 110, 110])]}
    pred = {1: [(0, 0.9, [0, 0, 10, 10]), (0, 0.8, [200, 200, 210, 210]), (0, 0.7, [100, 100, 110, 110])]}
    return M.evaluate_detections(gt, pred, ["a"])


def test_detection_charts():
    print("detection charts")
    r = hand_detection()
    charts = C.detection_charts(r)
    ids = [c["id"] for c in charts]
    check("the detection charts come in a fixed order",
          ids == ["ap_by_iou", "f1_vs_confidence", "precision_vs_confidence", "recall_vs_confidence",
                  "precision_recall", "per_class_ap", "confusion_matrix"])
    ch = by_id(charts)
    expected_ap = (51 * 1.0 + 50 * (2.0 / 3.0)) / 101.0
    check("average precision by IoU carries every threshold with its value",
          [p[0] for p in ch["ap_by_iou"]["series"][0]["points"]] == list(M.DEFAULT_IOU_THRESHOLDS)
          and all(close(p[1], expected_ap, 1e-12) for p in ch["ap_by_iou"]["series"][0]["points"]))
    f1 = ch["f1_vs_confidence"]
    macro = f1["series"][0]
    grid = [p[0] for p in macro["points"]]
    i70 = min(range(len(grid)), key=lambda i: abs(grid[i] - 0.7))
    i75 = min(range(len(grid)), key=lambda i: abs(grid[i] - 0.75))
    i85 = min(range(len(grid)), key=lambda i: abs(grid[i] - 0.85))
    check("F1 against confidence is 0.8 at 0.70, 0.5 at 0.75 and 2/3 at 0.85",
          close(macro["points"][i70][1], 0.8, 1e-12) and close(macro["points"][i75][1], 0.5, 1e-12)
          and close(macro["points"][i85][1], 2 / 3, 1e-12))
    check("a score of exactly 0.70 counts at the grid value 0.70", grid[i70] == 0.7)
    check("the highest mean F1 is marked on the curve",
          len(f1["markers"]) == 1 and close(f1["markers"][0]["x"], 0.7) and close(f1["markers"][0]["y"], 0.8, 1e-12))
    check("the macro line is first, drawn as the mean, and visible",
          macro["role"] == "macro" and macro["default_visible"] and macro["index"] == 0)
    check("a class series follows with a stable position",
          f1["series"][1]["key"] == "class_0" and f1["series"][1]["index"] == 1)
    pr = ch["precision_recall"]
    check("the precision-recall chart has 101 points per series and a mean line",
          all(len(s["points"]) == 101 for s in pr["series"]) and pr["series"][0]["role"] == "macro")
    check("its first point is full precision at recall 0",
          close(pr["series"][0]["points"][0][1], 1.0) and pr["series"][0]["points"][0][0] == 0.0)
    bar = ch["per_class_ap"]
    check("the class bars carry both averages", bar["categories"] == ["a"] and len(bar["series"]) == 2
          and close(bar["series"][0]["values"][0], expected_ap, 1e-12))
    cm = ch["confusion_matrix"]
    check("the confusion matrix chart has the labels, counts and row fractions",
          cm["row_labels"] == ["a", "background"] and cm["values"] == [[2, 0], [1, 0]]
          and cm["row_fractions"][0] == [1.0, 0.0] and cm["row_fractions"][1] == [1.0, 0.0])
    check("rows and columns are titled", cm["row_title"] == "Labelled truth" and cm["col_title"] == "Model report")

    for c in charts:
        ok = {"id", "kind", "title", "help", "csv"} <= set(c) and c["help"] and c["title"]
        check(f"{c['id']}: has an id, kind, title, help line and CSV", bool(ok))
    check("every series in a chart has its own position",
          all(len({s['index'] for s in c['series']}) == len(c['series']) for c in charts if c["kind"] != "matrix"))
    check("the whole bundle survives JSON with no NaN", json.loads(json.dumps(charts, allow_nan=False)) == charts)
    check("the same metrics give the same charts", charts == C.detection_charts(r))
    check("no colour appears in the data, only a position",
          "#" not in json.dumps(charts) and "color" not in json.dumps(charts).lower())
    check("no chart text says which model is better", not has_verdict_words(charts))


def test_many_classes():
    print("long class lists")
    n_cls = 60
    gt, pred = make_detection_case(2, n_images=120, n_classes=n_cls)
    names = [f"class {i:02d}" for i in range(n_cls)]
    r = M.evaluate_detections(gt, pred, names)
    with_gt = [p for p in r["per_class"] if p["n_ground_truth"] > 0]
    charts = by_id(C.detection_charts(r, max_class_series=10, max_visible_classes=4))
    f1 = charts["f1_vs_confidence"]
    check("a line chart draws the mean and at most the requested number of classes", len(f1["series"]) <= 11)
    check("it says how many classes it left out", f1["series_omitted"] == len(with_gt) - 10 and "further classes" in (f1["note"] or ""))
    visible = [s for s in f1["series"] if s["role"] == "class" and s["default_visible"]]
    check("only the best-supported classes are visible at first", len(visible) <= 4)
    order = sorted(range(n_cls), key=lambda c: (-r["per_class"][c]["n_ground_truth"], c))[:4]
    check("and they are the ones with the most test boxes",
          {s["key"] for s in visible} <= {f"class_{c}" for c in order})
    check("the bar chart still shows every tested class",
          len(charts["per_class_ap"]["categories"]) == len(with_gt))
    table = C.class_table(r)
    check("the table lists all classes", len(table["rows"]) == n_cls)
    untested = [row for row in table["rows"] if not row["tested"]]
    check("a class with no test boxes is marked not tested",
          all("Not tested" in row["status"] for row in untested) and
          all(row["tested"] for row in table["rows"] if row["n_ground_truth"] > 0))
    check("the confusion matrix is complete however many classes",
          len(charts["confusion_matrix"]["values"]) == n_cls + 1)


def test_curve_series_values():
    print("curve series against the metrics")
    gt, pred = make_detection_case(12, n_images=40)
    names = ["a", "b", "c", "d"]
    r = M.evaluate_detections(gt, pred, names)
    charts = by_id(C.detection_charts(r))
    pr = charts["precision_recall"]
    macro = pr["series"][0]
    classes = [s for s in pr["series"] if s["role"] == "class"]
    check("every class with test boxes has a precision-recall series", len(classes) == sum(
        1 for p in r["per_class"] if p["n_ground_truth"] > 0))
    ok = True
    for k in (0, 10, 50, 77, 100):
        mean = sum(s["points"][k][1] for s in classes) / len(classes)
        ok &= close(macro["points"][k][1], mean, 1e-12)
    check("the mean precision-recall line is the mean of the class lines at each recall", ok)
    ok = True
    for s in classes:
        c = int(s["key"].split("_")[1])
        ok &= [p[1] for p in s["points"]] == r["curves"]["precision_recall"]["per_class"][c]["precision"]
    check("each class line is the stored 101-point precision", ok)
    f1 = charts["f1_vs_confidence"]
    ok = all([p[1] for p in s["points"]] == (r["curves"]["macro"]["f1"] if s["key"] == "macro"
                                              else r["curves"]["per_class"][int(s["key"].split("_")[1])]["f1"])
             for s in f1["series"])
    check("the F1 lines are the stored curves", ok)
    ap = charts["ap_by_iou"]
    rows = ap["csv"]["rows"]
    check("CSV rows follow the order of the points, series by series",
          [row[1] for row in rows] == [p[0] for p in ap["series"][0]["points"]]
          and [row[2] for row in rows] == [p[1] for p in ap["series"][0]["points"]]
          and all(row[0] == "All classes" for row in rows))
    check("the x values rise along every line",
          all(all(a[0] <= b[0] for a, b in zip(s["points"], s["points"][1:])) for c in charts.values()
              if c["kind"] == "line" for s in c["series"]))


def test_csv():
    print("CSV")
    r = hand_detection()
    charts = C.detection_charts(r)
    for c in charts:
        text = C.chart_csv_text(c)
        rows = list(csv.reader(io.StringIO(text)))
        check(f"{c['id']}: header plus one row per record", rows[0] == c["csv"]["columns"]
              and len(rows) == 1 + len(c["csv"]["rows"]))
    ap = by_id(charts)["ap_by_iou"]
    rows = list(csv.reader(io.StringIO(C.chart_csv_text(ap))))
    check("floating-point values round-trip exactly through the CSV text",
          all(float(rows[i + 1][2]) == ap["csv"]["rows"][i][2] for i in range(len(rows) - 1)))
    check("a missing value is an empty cell",
          C.chart_csv_text({"csv": {"columns": ["a", "b"], "rows": [["x", None], ["y", 1.5]]}}) == "a,b\nx,\ny,1.5\n")
    check("a boolean is written as true or false",
          C.chart_csv_text({"csv": {"columns": ["a"], "rows": [[True], [False]]}}) == "a\ntrue\nfalse\n")

    # Species names with accents and commas.
    gt = {1: [(0, [0, 0, 10, 10]), (1, [50, 50, 60, 60])]}
    pred = {1: [(0, 0.9, [0, 0, 10, 10]), (1, 0.8, [50, 50, 60, 60])]}
    r2 = M.evaluate_detections(gt, pred, ["Dysidea etheria, form \u00f1", "Aplysina cauliformis"])
    with tempfile.TemporaryDirectory() as td:
        folder = Path(td) / "charts" / "nested"
        written = C.write_chart_csvs(C.detection_charts(r2), folder)
        files = sorted(p.name for p in folder.iterdir())
        check("one file per chart named by its id", files == sorted(f"{k}.csv" for k in written)
              and written["per_class_ap"] == "per_class_ap.csv")
        raw = (folder / "per_class_ap.csv").read_bytes()
        check("the file starts with a byte-order mark so spreadsheets read accents", raw.startswith(b"\xef\xbb\xbf"))
        check("the file holds exactly the chart's CSV text",
              raw.decode("utf-8-sig") == C.chart_csv_text(by_id(C.detection_charts(r2))["per_class_ap"]))
        back = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
        check("accents and commas in a name survive", back[1][0] == "Dysidea etheria, form \u00f1")
        written2 = C.write_chart_csvs(C.detection_charts(r2), folder)
        check("writing again replaces the files without error", written2 == written)
        check("an invalid chart id is refused",
              raises(lambda: C.write_chart_csvs([{"id": "../x", "csv": {"columns": [], "rows": []}}], folder),
                     C.ChartsError))


def test_training_curves():
    print("training curves")
    rows = [
        {"epoch": 1, "train_loss": 1.0, "val_loss": 1.2, "map50": 0.1},
        {"epoch": 2, "train_loss": 0.8, "val_loss": None, "map50": 0.3, "map50_95": 0.2},
        {"epoch": 3, "train_loss": 0.7, "val_loss": 0.9, "map50": 0.4},
        {"epoch": 3, "train_loss": 0.65, "val_loss": 0.85, "map50": 0.45},     # resumed run repeats an epoch
    ]
    charts = by_id(C.training_curve_charts(rows))
    check("a loss chart and a score chart are produced", set(charts) == {"loss_by_epoch", "map_by_epoch"})
    loss = charts["loss_by_epoch"]
    train = [s for s in loss["series"] if s["key"] == "train_loss"][0]
    val = [s for s in loss["series"] if s["key"] == "val_loss"][0]
    check("the last row of a repeated epoch wins", train["points"][-1] == [3, 0.65])
    check("a missing value is a gap, not a zero", val["points"][1] == [2, None])
    check("epochs are in order", [p[0] for p in train["points"]] == [1, 2, 3])
    score = charts["map_by_epoch"]
    check("only the scores present are drawn", {s["key"] for s in score["series"]} == {"map50", "map50_95"})
    check("a series missing an early epoch has a gap there",
          [s for s in score["series"] if s["key"] == "map50_95"][0]["points"][0] == [1, None])
    check("no rows gives no charts", C.training_curve_charts([]) == [] and C.training_curve_charts(None) == [])
    check("a row without an epoch is refused", raises(lambda: C.training_curve_charts([{"train_loss": 1}]), C.ChartsError))
    check("a non-finite value is a gap",
          C.training_curve_charts([{"epoch": 1, "train_loss": float("nan")}, {"epoch": 2, "train_loss": 0.5}])[0]
          ["series"][0]["points"][0] == [1, None])
    check("a bool is not a number",
          C.training_curve_charts([{"epoch": 1, "train_loss": True}, {"epoch": 2, "train_loss": 0.5}])[0]
          ["series"][0]["points"][0] == [1, None])


def test_tables_and_cards():
    print("tables and cards")
    gt, pred = make_detection_case(4)
    r = M.evaluate_detections(gt, pred, ["a", "b", "c", "d"])
    cards = C.headline_cards(r)
    check("six headline cards for a vision model, each with a help line",
          [c["key"] for c in cards] == ["map50", "map50_95", "macro_precision", "macro_recall", "macro_f1",
                                        "top1_accuracy_on_matched_boxes"] and all(c["help"] for c in cards))
    h = M.headline(r)
    check("card values are the headline figures", all(c["value"] == h[c["key"]] for c in cards))
    top = [c for c in cards if c["key"] == "top1_accuracy_on_matched_boxes"][0]
    check("the accuracy card carries its exact definition and says it is not mAP",
          "not average precision" in top["help"] and "matching overlap threshold" in top["help"])
    t = C.class_table(r)
    check("the class table has a column for each figure",
          [c["key"] for c in t["columns"]] == ["class_name", "n_ground_truth", "ap50", "ap50_95", "precision",
                                              "recall", "f1"])
    check("every column has a help line", all(c["help"] for c in t["columns"]))
    check("there is one row per class", [row["class_name"] for row in t["rows"]] == ["a", "b", "c", "d"])
    y, s = make_acoustic_case(1)
    ra = M.evaluate_acoustic(y, s, [f"c{i}" for i in range(6)])
    ta = C.class_table(ra)
    check("the acoustic table is keyed by test windows", ta["columns"][1]["key"] == "n_positive")
    check("the class with no positives is marked not tested", ta["rows"][5]["tested"] is False
          and "Not tested" in ta["rows"][5]["status"])
    check("an unknown kind is refused", raises(lambda: C.class_table({"kind": "x"}), C.ChartsError))
    check("charts refuse the wrong kind", raises(lambda: C.detection_charts(ra), C.ChartsError)
          and raises(lambda: C.acoustic_charts(r), C.ChartsError))


def test_acoustic_charts():
    print("acoustic charts")
    y, s = make_acoustic_case(3)
    names = [f"c{i}" for i in range(6)]
    cals = M.fit_calibrations(y, s, names)
    r = M.evaluate_acoustic(y, s, names, calibrations=cals)
    charts = C.acoustic_charts(r)
    check("the acoustic charts come in a fixed order",
          [c["id"] for c in charts] == ["precision_recall", "roc", "f1_vs_threshold", "per_class_ap",
                                       "confusion_matrix", "reliability"])
    ch = by_id(charts)
    roc = ch["roc"]
    check("the ROC chart has a dashed no-skill reference line",
          roc["series"][0]["key"] == "no_skill" and roc["series"][0]["dashed"] and roc["series"][0]["role"] == "reference")
    check("the class with no positives has no curve", all(s_["key"] != "class_5" for s_ in ch["precision_recall"]["series"]))
    rel = ch["reliability"]
    keys = [s_["key"] for s_ in rel["series"]]
    check("reliability shows the diagonal, the raw score and the calibrated probability",
          keys == ["perfect", "raw", "calibrated"] and rel["series"][0]["dashed"])
    check("the bin counts are kept for the interface", set(rel["bin_counts"]) == {"raw", "calibrated"}
          and sum(rel["bin_counts"]["raw"]) == y.size)
    check("the expected calibration error is stated", "expected calibration error" in (rel["note"] or ""))
    f1 = ch["f1_vs_threshold"]
    check("F1 against threshold marks the best shared threshold", len(f1["markers"]) == 1)
    check("its axis follows the grid", f1["x"]["min"] == r["curves"]["threshold_grid"][0]
          and f1["x"]["max"] == r["curves"]["threshold_grid"][-1])
    cmx = ch["confusion_matrix"]
    check("every window is in the acoustic confusion chart", sum(sum(row) for row in cmx["values"]) == y.shape[0])
    check("the bars have AP and ROC area for the classes with positives",
          ch["per_class_ap"]["categories"] == names[:5] and len(ch["per_class_ap"]["series"]) == 2)
    for c in charts:
        rows = list(csv.reader(io.StringIO(C.chart_csv_text(c))))
        check(f"{c['id']}: CSV has a header plus one row per record", len(rows) == 1 + len(c["csv"]["rows"]))
    check("the acoustic charts survive JSON", json.loads(json.dumps(charts, allow_nan=False)) == charts)
    r_nocal = M.evaluate_acoustic(y, s, names)
    rel2 = by_id(C.acoustic_charts(r_nocal))["reliability"]
    check("without calibrations only the raw score is drawn", [s_["key"] for s_ in rel2["series"]] == ["perfect", "raw"])
    r_out = M.evaluate_acoustic(y, s * 10 - 3, names, threshold=2.0)
    rel3 = by_id(C.acoustic_charts(r_out))["reliability"]
    check("scores outside 0 to 1 leave only the reference line", [s_["key"] for s_ in rel3["series"]] == ["perfect"])

    table = C.calibration_table(r)
    check("the calibration table has one row per class with plain-language status",
          len(table["rows"]) == 6 and table["rows"][5]["status"] == "too_few_positives"
          and "Too few" in table["rows"][5]["status_text"] and table["rows"][0]["status_text"] == "Calibrated")
    check("it states that scores are not probabilities and thresholds are not applied automatically",
          "not probabilities" in table["note"] and "not applied automatically" in table["note"])
    check("a result without calibrations has no calibration table", C.calibration_table(r_nocal) is None)


def test_comparison():
    print("challenger against current")
    names = ["a", "b", "c", "d"]
    gt, pred = make_detection_case(5)
    challenger = M.evaluate_detections(gt, pred, names)
    gt2, pred2, names2 = M.restrict_detection(gt, pred, names, [0, 1, 2])
    thinned = {img: items[::2] for img, items in pred2.items()}      # a weaker model: every other report
    current = M.evaluate_detections(gt2, thinned, names2)
    cs_gt, cs_pred, cs_names = M.restrict_detection(gt, pred, names, [0, 1, 2])
    challenger_shared = M.evaluate_detections(cs_gt, cs_pred, cs_names)
    table = C.comparison_table(challenger, current, challenger_shared=challenger_shared, current_shared=current)
    check("every row carries the challenger figure, the current figure and challenger minus current",
          all(close(row["difference"], row["challenger"] - row["current"], 1e-15)
              for row in table["rows"] if row["challenger"] is not None and row["current"] is not None))
    check("the rows are the vision headline figures",
          [r_["metric"] for r_ in table["rows"]] == ["map50", "map50_95", "macro_precision", "macro_recall",
                                                    "macro_f1", "top1_accuracy_on_matched_boxes"])
    check("classes the current model does not know are named", table["not_in_current"] == ["d"])
    check("classes only the current model knows are named", table["not_in_challenger"] == [])
    check("the shared classes are listed", table["shared_classes"] == ["a", "b", "c"])
    check("a second set of rows is restricted to the shared classes",
          table["shared_rows"] is not None and len(table["shared_rows"]) == 6)
    hs, hc = M.headline(challenger_shared), M.headline(current)
    check("the shared rows hold the challenger and the current model on the shared classes, each in its own column",
          all(row["challenger"] == hs[row["metric"]] and row["current"] == hc[row["metric"]]
              for row in table["shared_rows"]))
    check("the two models differ, so the columns cannot be mistaken for each other",
          any(row["challenger"] != row["current"] for row in table["shared_rows"]))
    check("the table prints numbers and never a verdict", not has_verdict_words(table))
    check("it states the sign convention and that small differences can be noise",
          "challenger minus" in table["note"] and "noise" in table["note"])
    none = C.comparison_table(challenger, None)
    check("with no current model there is nothing to compare and the rows still print",
          none["rows"][0]["current"] is None and none["rows"][0]["difference"] is None
          and none["not_in_current"] == names and "nothing to compare" in none["note"])
    check("one shared-class result without the other is refused",
          raises(lambda: C.comparison_table(challenger, current, challenger_shared=challenger_shared), C.ChartsError))
    y, s = make_acoustic_case(1)
    ra = M.evaluate_acoustic(y, s, [f"c{i}" for i in range(6)])
    check("two different kinds are refused", raises(lambda: C.comparison_table(challenger, ra), C.ChartsError))
    ta = C.comparison_table(ra, ra)
    check("an acoustic comparison uses the acoustic figures and equal models differ by zero",
          [r_["metric"] for r_ in ta["rows"]][:2] == ["cmap", "macro_roc_auc"]
          and all(r_["difference"] == 0 for r_ in ta["rows"] if r_["challenger"] is not None))
    check("a figure missing on one side gives no difference",
          C.comparison_table(M.evaluate_detections({1: []}, {}, ["a"]), challenger)["rows"][0]["difference"] is None)


def test_bundle_and_hygiene():
    print("bundle and hygiene")
    gt, pred = make_detection_case(6)
    r = M.evaluate_detections(gt, pred, ["a", "b", "c", "d"])
    progress = [{"epoch": e, "train_loss": 1.0 / e, "map50": 0.1 * e} for e in range(1, 5)]
    b = C.results_bundle(r, progress)
    check("the bundle has cards, charts, the class table and the honesty line",
          set(b) == {"definition_version", "kind", "honesty_line", "cards", "charts", "class_table", "calibration"})
    check("training curves come first", [c["id"] for c in b["charts"]][:2] == ["loss_by_epoch", "map_by_epoch"])
    check("the honesty line says the events were never seen in training", "never saw" in b["honesty_line"])
    check("a vision bundle has no calibration block", b["calibration"] is None)
    check("the bundle survives JSON", json.loads(json.dumps(b, allow_nan=False)) == b)
    check("no verdict words anywhere in the bundle or the help text",
          not has_verdict_words(b) and not has_verdict_words(C.METRIC_HELP) and not has_verdict_words(C.COMPARISON_NOTE)
          and not has_verdict_words(C.CALIBRATION_NOTE) and not has_verdict_words(C.HONESTY_LINE))
    y, s = make_acoustic_case(2)
    names = [f"c{i}" for i in range(6)]
    ra = M.evaluate_acoustic(y, s, names, calibrations=M.fit_calibrations(y, s, names))
    ba = C.results_bundle(ra)
    check("an acoustic bundle carries the calibration table", ba["calibration"] is not None and ba["kind"] == "acoustic")
    check("an unknown kind is refused", raises(lambda: C.results_bundle({"kind": "x"}), C.ChartsError))
    check("chart options pass through", len(by_id(C.results_bundle(r, max_class_series=1)["charts"])["f1_vs_confidence"]["series"]) <= 2)

    for name in ("metrics.py", "charts_data.py"):
        src = (REPO / "audtheia" / "training" / name).read_text(encoding="utf-8")
        check(f"{name}: ASCII only, no em dash", all(ord(ch) < 128 for ch in src))
        tree = ast.parse(src)
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                mods.add((node.module or "").split(".")[0])
        allowed = {"__future__", "math", "random", "typing", "numpy"} if name == "metrics.py" else \
            {"__future__", "csv", "io", "math", "re", "pathlib", "typing"}
        check(f"{name}: imports only {sorted(allowed - {'__future__'})}", mods <= allowed)
    check("every metric the interface shows has a label and a help line",
          all(len(v) == 2 and v[0] and v[1] for v in C.METRIC_HELP.values()))
    needed = {"map50", "map50_95", "precision", "recall", "f1", "top1_accuracy_on_matched_boxes", "cmap",
              "macro_roc_auc", "top1_accuracy", "confusion_matrix", "reliability", "roc", "precision_recall",
              "f1_vs_confidence", "f1_vs_threshold", "per_class_ap", "calibration_threshold", "loss_by_epoch",
              "map_by_epoch", "ap_by_iou", "precision_vs_confidence", "recall_vs_confidence", "macro_precision",
              "macro_recall", "macro_f1"}
    check("help exists for every chart and figure", needed <= set(C.METRIC_HELP))


def main() -> int:
    test_detection_charts()
    test_many_classes()
    test_curve_series_values()
    test_csv()
    test_training_curves()
    test_tables_and_cards()
    test_acoustic_charts()
    test_comparison()
    test_bundle_and_hygiene()
    print(f"\nResult: {CHECKS['passed']} passed, {CHECKS['failed']} failed")
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
