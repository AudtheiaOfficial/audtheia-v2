"""Verification for scoring a trained model against the model in use.

Path: tests/test_training_evaluate.py

Builds a real dataset in a temporary folder, then scores scripted detectors and
sound models through the real evaluation code: a trained model that finds every
box, and a model in use that names one species under another spelling, does not
know another, and draws boxes for a species outside the dataset. Proves the
classes are matched by taxon identity, that a species the model in use cannot
name is listed and counts as missed, that the like-for-like row keeps only the
shared species, that out-of-dataset boxes are counted, that the share of
model-proposed test boxes is stated, that results are written only inside the run
folder, and that the endpoints refuse what they should.

Uses the standard library, NumPy and Pillow, plus the project itself.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from training_fixture import World, fake_rollup, make_backbone  # noqa: E402
from audtheia.pipeline.monitor import RawDetection  # noqa: E402
from audtheia.storage.database import TrainingRun, new_id, utc_now_iso  # noqa: E402
from audtheia.training import dataset as ds  # noqa: E402
from audtheia.training import evaluate as ev  # noqa: E402
from audtheia.training import taxon_identity as ti  # noqa: E402

CHECKS = {"passed": 0, "failed": 0}
SPECIES = [("Aplysina fistularis", 100), ("Poecile carolinensis", 200), ("mystery-bird", None)]


def check(label: str, condition: bool) -> None:
    CHECKS["passed" if condition else "failed"] += 1
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")


def digest(arr) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


class FakeDetector:
    """Returns the boxes it was given for each picture, under its own class numbers."""

    def __init__(self, class_names: dict, answers: dict):
        self.class_names = dict(class_names)
        self.answers = answers
        self.closed = False

    def detect(self, frame):
        return [RawDetection(x1=b[0], y1=b[1], x2=b[2], y2=b[3], confidence=s, class_id=c,
                             class_name=self.class_names.get(c, str(c)))
                for c, s, b in self.answers.get(digest(frame.image), [])]

    def close(self):
        self.closed = True


def build_vision(root: Path):
    w = World(root)
    make_backbone(Path(w.settings.path("gbif_backbone_path")))
    for name, key in SPECIES:
        for i in range(5):
            e = w.vision_event([(name, key)], station=f"S{i % 2}", first_seen=f"2026-0{i + 1}-0{1 + len(name) % 7}T10:00:00Z")
            w.confirm(e)
    req = {"modality": "vision", "required_criteria": ["expert_confirmed"], "min_examples_per_class": 1}
    built = ds.build_dataset(w.db, w.settings, ds.options_from_request(w.settings, req), rollup_lookup=fake_rollup({}))
    return w, Path(built["path"])


def truth(dataset_dir: Path) -> dict:
    coco = json.loads((dataset_dir / "test" / "_annotations.coco.json").read_text(encoding="utf-8"))
    files = {img["id"]: img["file_name"] for img in coco["images"]}
    out = {}
    for img_id, name in files.items():
        with Image.open(dataset_dir / "test" / name) as im:
            key = digest(np.asarray(im.convert("RGB")))
        out[key] = [(a["category_id"] - 1, [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2],
                                            a["bbox"][1] + a["bbox"][3]]) for a in coco["annotations"]
                    if a["image_id"] == img_id]
    return out


def test_vision(root: Path) -> None:
    print("\n[1] Images: matching by identity and the two comparisons")
    w, d = build_vision(root)
    classes = sorted(json.loads((d / "classes.json").read_text(encoding="utf-8")), key=lambda c: c["class_index"])
    names = [c["class_name"] for c in classes]
    idx = {c["class_key"]: c["class_index"] for c in classes}
    gt = truth(d)
    check("the dataset has a test split for every class", len(gt) >= 3 and len(names) == 3)
    perfect = {k: [(c, 0.9, b) for c, b in v] for k, v in gt.items()}
    n = len(names)
    for k in list(perfect)[:2]:
        perfect[k] = perfect[k] + [(n, 0.95, [0, 0, 5, 5])]
    # The model in use names Poecile through a BirdNET-style label, knows the sponge,
    # does not know the mystery bird, and draws extra boxes for a species outside the dataset.
    cur_names = {0: "Verongia fistularis", 1: "Poecile carolinensis_Carolina Chickadee", 2: "Dup name"}
    remap = {idx["gbif:100"]: 0, idx["gbif:200"]: 1}
    current_answers = {}
    for k, v in gt.items():
        rows = [(remap[c], 0.8, b) for c, b in v if c in remap]
        rows.append((2, 0.7, [1, 1, 9, 9]))
        current_answers[k] = rows
    made = []

    def load(path, entry):
        det = FakeDetector({i: names[i] for i in range(n)}, perfect) if path.name == "challenger.onnx" \
            else FakeDetector(cur_names, current_answers)
        made.append(det)
        return det

    calls = []
    with ti.Resolver(Path(w.settings.path("gbif_backbone_path"))) as r:
        rec = ev.evaluate_vision(d, d / "challenger.onnx", {"id": "x", "label": "In use", "path": str(d / "current.onnx"),
                                                           "entry": {}, "version": "V9"},
                                 resolver=r, load=load, progress=lambda *a: calls.append(a),
                                 options={"report_confidence": 0.3, "example_images_per_kind": 2}, seed=1)
    comp = rec["comparison"]
    rows = {row["metric"]: row for row in comp["rows"]}
    check("the trained model that finds every box scores 1 at IoU 0.50", rows["map50"]["challenger"] == 1.0)
    check("a species the model in use cannot name is listed", comp["not_in_current"] == ["mystery-bird"])
    check("a synonym and a BirdNET-style label are matched to their dataset classes by identity",
          sorted(comp["shared_classes"]) == sorted(["Aplysina fistularis", "Poecile carolinensis"]))
    check("in the first comparison the unknown species counts as missed, so recall falls below 1",
          rows["macro_recall"]["current"] is not None and rows["macro_recall"]["current"] < 1.0)
    shared = {row["metric"]: row for row in comp["shared_rows"]}
    check("on the shared species alone the model in use is scored on what it knows",
          shared["macro_recall"]["current"] == 1.0)
    check("boxes for a species outside the dataset are counted, not dropped silently",
          comp["current_boxes_outside_dataset"] == len(gt))
    check("a trained model's box on a column beyond its classes is counted", comp["challenger_boxes_beyond_dataset"] == 2)
    check("the differences are challenger minus current",
          abs(rows["macro_recall"]["difference"] - (rows["macro_recall"]["challenger"] - rows["macro_recall"]["current"])) < 1e-12)
    check("the comparison never words a verdict",
          not any(w_ in json.dumps(comp).lower() for w_ in (" better", "improved", " worse")))
    check("the share of model-proposed test boxes is stated with its caveat",
          rec["box_origin"]["model_proposed"] == rec["box_origin"]["total"] > 0 and "favours it" in rec["box_origin"]["note"])
    n_test_boxes = len(json.loads((d / "test" / "_annotations.coco.json").read_text(encoding="utf-8"))["annotations"])
    check("only the test split's boxes are counted for that caveat", rec["box_origin"]["total"] == n_test_boxes)
    check("the examples are test images with their truth and predictions",
          all(x["file"].startswith("test/") and x["truth"] for x in rec["examples"]["correct"])
          and len(rec["examples"]["correct"]) <= 2)
    check("every detector that was opened was closed", made and all(m.closed for m in made))
    check("progress is reported for both models",
          {c[0] for c in calls} == {"trained model", "model in use"})
    check("the model in use is recorded with its version", rec["current_model"]["version"] == "V9")

    print("\n[2] Images: with nothing to compare with")
    with ti.Resolver(None) as r:
        alone = ev.evaluate_vision(d, d / "challenger.onnx", None, resolver=r, load=load, options={})
    check("the table says there is nothing to compare with", alone["current"] is None
          and "nothing to compare" in alone["comparison"]["note"])

    print("\n[3] Storing the result")
    run_dir = root / "run"
    data_before = sorted(p.as_posix() for p in Path(w.settings.path("data_dir")).rglob("*"))
    path = ev.write_evaluation(run_dir, rec)
    check("the evaluation is written inside the run folder", path == run_dir / "evaluation" / "evaluation.json")
    check("it reads back identically", ev.read_evaluation(run_dir)["comparison"] == json.loads(json.dumps(comp)))
    check("a CSV is written for each chart", len(list((run_dir / "evaluation" / "csv").glob("*.csv"))) >= 4)
    check("the data folder is untouched", data_before == sorted(p.as_posix() for p in Path(w.settings.path("data_dir")).rglob("*")))
    (run_dir / "evaluation" / "evaluation.json").write_text("{broken", encoding="utf-8")
    check("an unreadable evaluation reads as none", ev.read_evaluation(run_dir) is None)
    check("no temporary file is left behind", not list((run_dir / "evaluation").glob(".evaluation.*")))
    try:
        ev.evaluate_vision(root / "nowhere", d / "c.onnx", None, resolver=ti.Resolver(None), load=load)
        refused = False
    except ev.EvaluationError:
        refused = True
    check("a folder that is not a dataset is refused with a reason", refused)
    return w, d, rec


def test_candidates(w: World) -> None:
    print("\n[4] Which models are in use")
    raw = w.settings.raw
    model = Path(w.repo) / "models" / "visual" / "in_use.onnx"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"x")
    raw.setdefault("desktop_models", {})["visual_rfdetr"] = {"path": "models/visual/in_use.onnx", "version": "V1"}
    for s in raw["stations"]:
        s.setdefault("models", {})["visual_desktop"] = {"path": "models/visual/in_use.onnx", "version": "V1"}
    found = ev.current_candidates(w.settings, "vision")
    check("one file used in several places is listed once, naming every place",
          len(found) == 1 and found[0]["label"].startswith("Desktop verifier") and found[0]["exists"])
    raw["desktop_models"]["visual_rfdetr"]["path"] = "models/visual/missing.onnx"
    found = ev.current_candidates(w.settings, "vision")
    check("a configured file that is missing is listed as missing",
          any(not c["exists"] for c in found) and any(c["exists"] for c in found))
    check("an unknown modality is refused", _raises(lambda: ev.current_candidates(w.settings, "smell")))


def _raises(fn) -> bool:
    try:
        fn()
    except ev.EvaluationError:
        return True
    return False


class FakeSound:
    def __init__(self, names: dict, scorer):
        self.class_names = dict(names)
        self.sample_rate = 24000
        self.window_seconds = 3.0
        self.scorer = scorer
        self.closed = False

    def detect(self, samples, rate):
        from audtheia.pipeline.acoustic import AcousticDetection

        assert len(samples) == int(self.window_seconds * self.sample_rate)
        return [AcousticDetection(class_id=c, class_name=self.class_names[c], confidence=s)
                for c, s in self.scorer(samples).items()]

    def close(self):
        self.closed = True


def test_acoustic(root: Path) -> None:
    print("\n[5] Sounds: both models on the same test windows")
    w = World(root)
    make_backbone(Path(w.settings.path("gbif_backbone_path")))
    for name, key in [("Poecile carolinensis_Carolina Chickadee", None), ("Aplysina fistularis", 100)]:
        for i in range(5):
            e = w.audio_event([(name, key)], station=f"S{i % 2}", first_seen=f"2026-0{i + 1}-0{1 + len(name) % 7}T06:00:00Z")
            w.confirm(e)
    req = {"modality": "audio", "engine": "acoustic_head_a", "required_criteria": ["expert_confirmed"],
           "min_examples_per_class": 1}
    built = ds.build_dataset(w.db, w.settings, ds.options_from_request(w.settings, req), rollup_lookup=fake_rollup({}))
    d = Path(built["path"])
    names = [c["class_name"] for c in sorted(json.loads((d / "classes.json").read_text(encoding="utf-8")),
                                            key=lambda c: c["class_index"])]
    check("the BirdNET label became the accepted name in the dataset", "Poecile carolinensis" in names)
    export = root / "export"
    export.mkdir()
    made = []

    def load(path, entry, **kw):
        if kw.get("station") is None and path.parent == export:
            m = FakeSound({i: n for i, n in enumerate(names)}, lambda x: {0: 0.9, 1: 0.1})
        else:
            m = FakeSound({0: "Poecile carolinensis_Carolina Chickadee", 1: "Some other bird_Other",
                           2: "Poecile carolinensis"}, lambda x: {0: 0.6, 1: 0.99, 2: 0.2})
        made.append(m)
        return m

    with ti.Resolver(Path(w.settings.path("gbif_backbone_path"))) as r:
        rec = ev.evaluate_acoustic(d, export, {"sample_rate": 24000, "window_seconds": 3.0, "report_confidence": 0.5},
                                   {"id": "s", "label": "Reef sound model", "path": str(root / "birdnet.tflite"),
                                    "entry": {}, "station": None}, resolver=r, load=load)
    comp = rec["comparison"]
    check("the test windows were scored", rec["test_windows"] > 0 and rec["challenger"]["kind"] == "acoustic")
    check("the trained model's own figures are used when the run left none", rec["challenger"]["class_names"] == names)
    check("the species the model in use cannot name is listed", comp["not_in_current"] == ["Aplysina fistularis"])
    check("its BirdNET label is matched to the dataset's class", comp["shared_classes"] == ["Poecile carolinensis"])
    shared = {row["metric"]: row for row in comp["shared_rows"]}
    check("two labels for one species keep the higher score, so every window passes the threshold",
          shared["macro_recall"]["current"] == 1.0)
    check("every sound model opened was closed", all(m.closed for m in made) and len(made) == 2)
    (export / "test_metrics.json").write_text(json.dumps(dict(rec["challenger"], marker="from the run")), encoding="utf-8")
    with ti.Resolver(None) as r:
        rec2 = ev.evaluate_acoustic(d, export, {"sample_rate": 24000, "window_seconds": 3.0}, None, resolver=r, load=load)
    check("the run's own test figures, with calibration, are kept when present", rec2["challenger"].get("marker") == "from the run")


def test_endpoints(root: Path) -> None:
    print("\n[6] The endpoints")
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        print("  SKIP  the web framework is not installed in this interpreter")
        return
    from audtheia.app import server as srv
    from audtheia.training import runs as R

    w, d, rec = build_vision(root)[0], None, None
    w2, d = w, build_vision(root / "b")[1]
    manager = R.RunManager(w.settings, w.db)
    rid = new_id()
    run_dir = manager.run_dir(rid)
    run_dir.mkdir(parents=True)
    rel = lambda p: str(Path(p).resolve().relative_to(Path(w.repo).resolve())) if Path(w.repo).resolve() in Path(p).resolve().parents else str(p)  # noqa: E731
    w.db.create_training_run(TrainingRun(
        id=rid, run_name="T", planned_model_name="Audtheia-T-V1", modality="vision", engine="vision_detector",
        mode="gentle", status="completed", dataset_signature="s", dataset_dir=rel(d), run_dir=rel(run_dir), seed=1,
        base_label="b", config_json="{}", created_by="tester", created_at=utc_now_iso()))
    queued = new_id()
    w.db.create_training_run(TrainingRun(
        id=queued, run_name="Q", planned_model_name="Audtheia-Q-V1", modality="vision", engine="vision_detector",
        mode="gentle", status="failed", dataset_signature="s", dataset_dir=rel(d), run_dir=rel(manager.run_dir(queued)),
        seed=1, base_label="b", config_json="{}", created_by="tester", created_at=utc_now_iso()))
    client = TestClient(srv.create_app(w.settings, w.db))
    res = client.get(f"/api/train/runs/{rid}/results")
    body = res.json()
    check("results of a finished run that is not scored yet say so", res.status_code == 200 and body["evaluated"] is False)
    check("the dataset's split and identity are reported", body["dataset"]["dataset_id"] == d.name)
    check("results of a run that did not finish are refused", client.get(f"/api/train/runs/{queued}/results").status_code == 409)
    check("scoring is a route of its own, not mistaken for pause or resume",
          client.post(f"/api/train/runs/{queued}/evaluate", json={}).status_code == 409)
    check("an unknown model in use is refused",
          client.post(f"/api/train/runs/{rid}/evaluate", json={"current_id": "nope"}).status_code == 422)
    ev.write_evaluation(run_dir, json.loads(json.dumps(_record(d))))
    body = client.get(f"/api/train/runs/{rid}/results").json()
    check("after scoring, the results carry the charts, the table and the comparison",
          body["evaluated"] and body["results"]["charts"] and body["comparison"]["rows"])
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
    ok = client.post(f"/api/train/runs/{rid}/charts", json={"charts": [{"id": "per_class_ap", "png": base64.b64encode(png).decode()}]})
    check("a chart picture is stored in the run folder", ok.status_code == 200
          and (run_dir / "evaluation" / "png" / "per_class_ap.png").read_bytes() == png)
    bad = client.post(f"/api/train/runs/{rid}/charts", json={"charts": [{"id": "x", "png": base64.b64encode(b"GIF89a").decode()}]})
    check("a picture that is not a PNG is refused", bad.status_code == 422)
    evil = client.post(f"/api/train/runs/{rid}/charts", json={"charts": [{"id": "../../escape", "png": base64.b64encode(png).decode()}]})
    check("a chart name that would leave the folder is refused", evil.status_code == 422 and not (run_dir.parent / "escape.png").exists())
    check("an empty upload is refused", client.post(f"/api/train/runs/{rid}/charts", json={"charts": []}).status_code == 422)
    first = sorted((d / "test").glob("*.jpg"))[0].name
    check("a test image of the run's dataset is served", client.get(f"/api/train/runs/{rid}/example", params={"file": f"test/{first}"}).status_code == 200)
    check("a file outside the test split is refused",
          client.get(f"/api/train/runs/{rid}/example", params={"file": "manifest.json"}).status_code == 404)
    check("a path that climbs out of the dataset is refused",
          client.get(f"/api/train/runs/{rid}/example", params={"file": "test/../../../config/settings.json"}).status_code == 404)
    w.settings.raw["node"]["role"] = "pi"
    check("a field station refuses the results", client.get(f"/api/train/runs/{rid}/results").status_code == 403)
    w.settings.raw["node"]["role"] = "desktop"


def _record(d: Path) -> dict:
    classes = sorted(json.loads((d / "classes.json").read_text(encoding="utf-8")), key=lambda c: c["class_index"])
    names = [c["class_name"] for c in classes]
    gt = {}
    coco = json.loads((d / "test" / "_annotations.coco.json").read_text(encoding="utf-8"))
    for img in coco["images"]:
        gt[img["file_name"]] = [(a["category_id"] - 1, [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2],
                                                       a["bbox"][1] + a["bbox"][3]]) for a in coco["annotations"]
                                if a["image_id"] == img["id"]]
    pred = {k: [(c, 0.9, b) for c, b in v] for k, v in gt.items()}
    from audtheia.training import metrics as M
    from audtheia.training import charts_data

    m = M.evaluate_detections(gt, pred, names)
    return {"kind": "detection", "challenger": m, "current": None, "comparison": charts_data.comparison_table(m, None),
            "examples": {"correct": [], "false_alarm": [], "missed": []}}


def test_text() -> None:
    print("\n[7] The module text")
    text = (REPO / "audtheia" / "training" / "evaluate.py").read_text(encoding="utf-8")
    check("evaluate.py is ASCII only", text.isascii())
    check("no runtime is imported at module level",
          all(m not in text.split('"""', 2)[2].split("\ndef ")[0] for m in ("import onnxruntime", "import numpy", "import tensorflow")))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="audtheia-evaluate-") as tmp:
        root = Path(tmp)
        w, _d, _rec = test_vision(root / "v")
        test_candidates(w)
        test_acoustic(root / "a")
        test_endpoints(root / "e")
    test_text()
    print(f"\nResult: {CHECKS['passed']} passed, {CHECKS['failed']} failed")
    return 0 if CHECKS["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
