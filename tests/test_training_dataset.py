"""Verification for planning and building a training dataset.

Path: tests/test_training_dataset.py

Drives the planner and builder end to end against a small record built in a
temporary folder through the real storage layer. Proves that the preview and the
build come from one plan, that nothing in the data folder or the database is ever
changed, that events are never split across training, validation and test, that
rejected events become capped background, that model-labelled examples reach
training only and only up to their cap, that a rare class can be left out, kept
or merged into its genus or family, that an untested class is marked, that the
same selection builds once and is then recognised, that a failed build leaves
nothing half written, and that an acoustic dataset is cut to the engine's window.

Uses the standard library, NumPy and Pillow, plus the project itself.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from training_fixture import World, fake_rollup  # noqa: E402
from audtheia.training import dataset as ds  # noqa: E402
from audtheia.training import formats  # noqa: E402

CHECKS = {"passed": 0, "failed": 0}
A = ("Aplysina fistularis", 100)
B = ("Aplysina lacunosa", 102)
C = ("Poecile carolinensis", 200)
ROLL = {
    "gbif:102": {"genus": ("20", "Aplysina"), "family": ("10", "Aplysinidae")},
    "gbif:100": {"genus": ("20", "Aplysina"), "family": ("10", "Aplysinidae")},
}


def check(label: str, condition: bool) -> None:
    CHECKS["passed" if condition else "failed"] += 1
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")


def raises(fn, exc=Exception) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


def tree_digest(root: Path) -> dict:
    out = {}
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            out[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def db_digest(world: World) -> dict:
    out = {}
    with world.db.connect() as conn:
        for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall():
            rows = conn.execute(f'SELECT * FROM "{name}"').fetchall()
            out[name] = (len(rows), hashlib.sha256(repr(sorted(repr(tuple(r)) for r in rows)).encode("utf-8")).hexdigest())
    return out


def options(w: World, **request):
    request.setdefault("modality", "vision")
    return ds.options_from_request(w.settings, request)


def plan(w: World, table=None, **request):
    return ds.plan_dataset(w.db, w.settings, options(w, **request), rollup_lookup=fake_rollup(table or {}))


def build(w: World, table=None, **request):
    return ds.build_dataset(w.db, w.settings, options(w, **request), rollup_lookup=fake_rollup(table or {}))


def confirmed(w: World, species, count, *, prefix, frames=6, **extra):
    made = []
    for i in range(count):
        ev = w.vision_event([species], station=f"{prefix}{i % 3}", first_seen=f"2026-03-{(i % 27) + 1:02d}T{8 + i % 10:02d}:00:00Z", frames=frames, **extra)
        w.confirm(ev)
        made.append(ev)
    return made


def standard_world(root: Path) -> World:
    w = World(root)
    confirmed(w, A, 10, prefix="S")
    confirmed(w, C, 6, prefix="S")
    for i in range(2):
        r = w.vision_event([A], station="S0", first_seen=f"2026-04-{i + 1:02d}T09:00:00Z", frames=6)
        w.reject(r)
    return w


# ---------------------------------------------------------------------------


def test_options(work: Path) -> None:
    print("\n[1] Checking a request before anything is planned")
    w = World(work / "options")
    ok = options(w)
    check("a bare request takes the configured defaults", ok.min_examples_per_class == 5 and ok.split_mode == "event" and ok.seed == 20260101)
    check("the default criteria are all three", ok.required == frozenset({"expert_confirmed", "qc_passed", "verified"}))
    check("an empty list of criteria means none, not the defaults", options(w, required_criteria=[]).required == frozenset())
    check("a request can narrow the criteria", options(w, required_criteria=["expert_confirmed"]).required == frozenset({"expert_confirmed"}))
    check("a misspelt field is refused rather than ignored", raises(lambda: options(w, minimum_examples=3), ds.DatasetError))
    check("an unknown data type is refused", raises(lambda: ds.options_from_request(w.settings, {"modality": "video"}), ds.DatasetError))
    check("an unknown criterion is refused", raises(lambda: options(w, required_criteria=["trusted"]), ds.DatasetError))
    check("an unknown class decision is refused", raises(lambda: options(w, class_policy={"gbif:1": "delete"}), ds.DatasetError))
    check("a fraction above one is refused", raises(lambda: options(w, background_max_fraction=1.5), ds.DatasetError))
    check("a non-numeric threshold is refused", raises(lambda: options(w, min_examples_per_class="five"), ds.DatasetError))
    check("a threshold below one is refused", raises(lambda: options(w, min_examples_per_class=0), ds.DatasetError))
    check("an unknown split field is refused", raises(lambda: options(w, split={"shuffle": True}), ds.DatasetError))
    check("an unknown split mode is refused", raises(lambda: options(w, split={"mode": "random"}), ds.DatasetError))
    check("an acoustic request without an engine is refused", raises(lambda: ds.options_from_request(w.settings, {"modality": "audio"}), ds.DatasetError))
    check("an unknown acoustic engine is refused", raises(lambda: ds.options_from_request(w.settings, {"modality": "audio", "engine": "nope"}), ds.DatasetError))
    audio = ds.options_from_request(w.settings, {"modality": "audio", "engine": "acoustic_head_b"})
    check("an engine sets the window length and sample rate", audio.window_seconds == 5.0 and audio.sample_rate == 32000)
    check("a request that is not an object is refused", raises(lambda: ds.options_from_request(w.settings, ["vision"]), ds.DatasetError))


def test_plan_and_build(work: Path) -> None:
    print("\n[2] One plan for the preview and the build, and nothing changed")
    w = standard_world(work / "std")
    data_before, db_before = tree_digest(w.data), db_digest(w)

    first = plan(w)
    second = plan(w)
    summary = ds.summarize_plan(first)
    check("planning twice gives the same signature", first["signature"] == second["signature"])
    check("planning gives the same dataset id twice", first["dataset_id"] == second["dataset_id"] and first["dataset_id"].startswith("vision-"))
    check("planning writes nothing to the data folder", tree_digest(w.data) == data_before)
    check("planning writes nothing to the database", db_digest(w) == db_before)
    check("planning writes no dataset", not (Path(w.settings.training_paths()["datasets_dir"]) / first["dataset_id"]).exists())

    t = summary["totals"]
    check("ninety-six animal examples are planned", sum(x["positive_instances"] for x in t.values()) == 96)
    check("the ten and six event classes divide six, two, two and four, one, one",
          (t["train"]["events"], t["validation"]["events"], t["test"]["events"]) == (6 + 4 + 2, 2 + 1, 2 + 1))
    check("every event is in one split only", len(first["events"]) == len({e["event_id"] for e in first["events"]}) == 18)
    in_split = {s: {i["event_id"] for i in first["items"][s]} for s in ds.SPLITS}
    check("no event has examples in two splits", not (in_split["train"] & in_split["validation"]) and not (in_split["train"] & in_split["test"]) and not (in_split["validation"] & in_split["test"]))
    check("every example sits in the split its event was assigned", all(first["split"]["assignment"][i["event_id"]] == s for s in ds.SPLITS for i in first["items"][s]))
    bg = summary["background"]
    check("background is capped at a tenth of the animals in each split", bg["train"]["kept"] == 6 and bg["train"]["available"] == 12 and bg["train"]["cap"] == 6)
    check("a split with no false alarms keeps no background", bg["validation"]["kept"] == 0 and bg["test"]["kept"] == 0)
    check("both classes are scored", all(c["tested"] for c in summary["classes"]) and summary["untested_classes"] == [])
    check("the preview reports the same counts the plan holds", summary["totals"]["train"]["examples"] == len(first["items"]["train"]))
    check("model boxes are counted", summary["box_stats"]["model"] == 96 and summary["box_stats"]["expert_adjusted"] == 0)
    check("the model-box caveat is among the warnings", any("blind spots" in x for x in summary["warnings"]))

    built = build(w)
    folder = Path(built["path"])
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    check("the build reports a new dataset", built["already_exists"] is False and built["dataset_id"] == first["dataset_id"])
    check("the dataset folder sits under the datasets directory", folder.parent == Path(w.settings.training_paths()["datasets_dir"]).resolve())
    check("the build writes nothing to the data folder", tree_digest(w.data) == data_before)
    check("the build changes nothing in the database", db_digest(w) == db_before)
    check("the manifest holds one entry per example", len(manifest["items"]) == sum(x["examples"] for x in summary["totals"].values()))
    check("the manifest totals match the preview", manifest["totals"] == summary["totals"])
    check("every manifest file exists in the dataset", all((folder / i["file"]).is_file() for i in manifest["items"]))
    check("every manifest entry names its source and a checksum of it", all(i["source"] and len(i["source_sha256"]) == 64 for i in manifest["items"]))
    check("a placed file is identical to its source", all(hashlib.sha256((folder / i["file"]).read_bytes()).hexdigest() == i["source_sha256"] for i in manifest["items"]))
    check("the manifest records how media was placed", set(manifest["media_placement"]) <= {"link", "copy"})
    check("the manifest records the options and the seed", manifest["options"]["split"]["seed"] == 20260101 and manifest["rules_version"] == ds.RULES_VERSION)
    split_doc = json.loads((folder / "split.json").read_text(encoding="utf-8"))
    check("the split file assigns every event once", len(split_doc["assignment"]) == 18 and set(split_doc["assignment"].values()) <= set(ds.SPLITS))
    check("the manifest keeps expert and model labels apart", all(i["model_labelled"] is False for i in manifest["items"]))
    check("the README is written", (folder / "README.md").is_file() and "never edit" in (folder / "README.md").read_text(encoding="utf-8").lower())
    names = [p.name for p in folder.iterdir()]
    check("a visual dataset has the COCO folders and the YOLO layout", {"train", "valid", "test", "yolo", "manifest.json", "split.json", "classes.json"} <= set(names))

    for split, name in (("train", "train"), ("validation", "valid"), ("test", "test")):
        coco = json.loads((folder / name / "_annotations.coco.json").read_text(encoding="utf-8"))
        files = {p.name for p in (folder / name).iterdir() if p.suffix == ".jpg"}
        check(f"{name}: every listed image is present and no extra image is", {i["file_name"] for i in coco["images"]} == files)
        ids = {i["id"] for i in coco["images"]}
        check(f"{name}: every box points at a real image and a real category", all(a["image_id"] in ids and a["category_id"] in {c["id"] for c in coco["categories"]} for a in coco["annotations"]))
        check(f"{name}: every box lies inside its image", all(a["bbox"][0] >= 0 and a["bbox"][1] >= 0 and a["bbox"][0] + a["bbox"][2] <= 64 and a["bbox"][1] + a["bbox"][3] <= 48 for a in coco["annotations"]))
        check(f"{name}: the COCO count matches the plan", len(coco["images"]) == summary["totals"][split]["examples"])
    classes = json.loads((folder / "classes.json").read_text(encoding="utf-8"))
    check("categories are numbered from one in a stable order", [c["category_id"] for c in classes] == [1, 2] and [c["class_name"] for c in classes] == sorted(c["class_name"] for c in classes))
    label_files = list((folder / "yolo" / "labels" / "train").glob("*.txt"))
    check("the YOLO layout has a label file for every image", len(label_files) == len(list((folder / "yolo" / "images" / "train").glob("*.jpg"))) == summary["totals"]["train"]["examples"])
    check("background examples have empty label files", sum(1 for f in label_files if f.read_text(encoding="utf-8") == "") == 6)

    before = tree_digest(folder)
    stamp = {p: p.stat().st_mtime_ns for p in folder.rglob("*") if p.is_file()}
    again = build(w)
    check("building the same selection again finds the first dataset", again["already_exists"] is True and again["path"] == built["path"])
    check("it writes nothing the second time", tree_digest(folder) == before and {p: p.stat().st_mtime_ns for p in folder.rglob("*") if p.is_file()} == stamp)
    listed = ds.list_datasets(w.settings)
    check("the dataset is listed", [d["dataset_id"] for d in listed] == [built["dataset_id"]])
    check("its manifest can be read back", ds.dataset_manifest(w.settings, built["dataset_id"])["signature"] == first["signature"])
    check("a path-like dataset id reads nothing", ds.dataset_manifest(w.settings, "../x") is None and ds.dataset_manifest(w.settings, ".hidden") is None)

    other_seed = plan(w, split={"seed": 99})
    check("a different seed gives a different dataset id", other_seed["dataset_id"] != first["dataset_id"])
    narrower = plan(w, required_criteria=["expert_confirmed"])
    check("the same events under the same checks keep the same id", narrower["signature"] != "" and plan(w, required_criteria=["expert_confirmed"])["signature"] == narrower["signature"])


def test_frame_rules_end_to_end(work: Path) -> None:
    print("\n[3] Frame rules applied to real event folders")
    w = World(work / "frames")
    evs = confirmed(w, A, 6, prefix="F")
    for ev in evs:
        w.frame(ev, 0, "inaccurate"); w.frame(ev, 1, "inaccurate")
    built = build(w)
    manifest = json.loads((Path(built["path"]) / "manifest.json").read_text(encoding="utf-8"))
    check("frames an expert marked inaccurate are not in the dataset", all(i["frame_index"] not in (0, 1) for i in manifest["items"]))
    check("the other frames are", len(manifest["items"]) == 6 * 4)
    check("the skipped frames are counted with their reason", built["summary"]["skipped"]["frames_left_out"].get("marked inaccurate by an expert") == 12)

    w2 = World(work / "cap")
    for i in range(6):
        e = w2.vision_event([A], station=f"F{i % 3}", first_seen=f"2026-03-{i + 1:02d}T10:00:00Z", frames=30); w2.confirm(e)
    p = plan(w2)
    per_event = {}
    for s in ds.SPLITS:
        for item in p["items"][s]:
            per_event[item["event_id"]] = per_event.get(item["event_id"], 0) + 1
    check("no event contributes more than the cap", max(per_event.values()) == 10 and set(per_event.values()) == {10})
    check("a lower cap is honoured", max(_counts(plan(w2, max_frames_per_event=3)).values()) == 3)

    w3 = World(work / "dups")
    for i in range(6):
        e = w3.vision_event([A], station=f"F{i % 3}", first_seen=f"2026-03-{i + 1:02d}T10:00:00Z", frames=6, identical_frames=True); w3.confirm(e)
    p3 = plan(w3)
    check("identical frames collapse to one example per event", set(_counts(p3).values()) == {1} and p3["skipped"]["near_duplicates"] == 6 * 5)

    w4 = World(work / "missing")
    confirmed(w4, A, 5, prefix="M")
    gone = w4.vision_event([A], station="M0", first_seen="2026-03-20T10:00:00Z", no_frames_on_disk=True); w4.confirm(gone)
    p4 = plan(w4)
    check("an event whose images are gone is left out with a reason, not a crash", gone["event"] in [e["event"] for e in p4["skipped"]["events"]])
    check("missing files are counted", p4["skipped"]["missing_media"] >= 6)
    check("the other events are unaffected", len(p4["events"]) == 5)

    w5 = World(work / "escape")
    good = confirmed(w5, A, 5, prefix="E")
    bad = w5.vision_event([A], station="E0", first_seen="2026-03-21T10:00:00Z"); w5.confirm(bad)
    with w5.db.connect() as conn:
        conn.execute("UPDATE observations SET representative_frame = ? WHERE id = ?", ("../../outside/ev/frame_000000.jpg", bad["id"]))
    p5 = plan(w5)
    check("a stored path outside the data folder is never followed", bad["event"] in [e["event"] for e in p5["skipped"]["events"]] and len(p5["events"]) == 5)

    w6 = World(work / "mixed")
    mixed = w6.vision_event([A, C], station="X0", frames=8, class_names=[A[0], C[0]])
    w6.confirm(mixed, detection=mixed["children"][0]); w6.confirm(mixed, detection=mixed["children"][1])
    p6 = plan(w6)
    labels = {}
    for item in p6["items"]["train"] + p6["items"]["validation"] + p6["items"]["test"]:
        for b in item["boxes"]:
            labels.setdefault(item["frame_index"] % 2, set()).add(b["taxon"]["class_name"])
    check("in a mixed event each frame carries the species the model saw in it", labels == {0: {A[0]}, 1: {C[0]}})

    w7 = World(work / "relabel")
    for i in range(5):
        e = w7.vision_event([A], station=f"R{i % 3}", first_seen=f"2026-03-{i + 1:02d}T10:00:00Z"); w7.relabel(e, B[0], B[1])
    p7 = plan(w7)
    names = {b["taxon"]["class_name"] for s in ds.SPLITS for i in p7["items"][s] for b in i["boxes"]}
    check("a relabelled event is learned under the expert's species only", names == {B[0]} and [c["class_name"] for c in p7["classes"]] == [B[0]])
    check("relabelled boxes are labelled by an expert, not the model", all(b["taxon"]["source"] == "expert" for s in ds.SPLITS for i in p7["items"][s] for b in i["boxes"]))


def test_box_corrections(work: Path) -> None:
    print("\n[3b] Expert box corrections reach the dataset")
    w = World(work / "boxes")
    evs = confirmed(w, A, 6, prefix="B")
    target = evs[0]
    w.db.add_frame_box_correction(target["id"], 0, action="adjust", corrector="tester", x1=5, y1=5, x2=20, y2=20, frame_width=64, frame_height=48)
    w.db.add_frame_box_correction(target["id"], 1, action="remove", corrector="tester")
    w.db.add_frame_box_correction(target["id"], 2, action="add", corrector="tester", x1=30, y1=10, x2=50, y2=40, frame_width=64, frame_height=48,
                                  corrected_scientific_name=C[0], corrected_gbif_usage_key=str(C[1]))

    def boxes_of(p, index):
        for sp in ds.SPLITS:
            for item in p["items"][sp]:
                if item["event_id"] == target["id"] and item["frame_index"] == index:
                    return item["boxes"]
        return None

    p = plan(w)
    zero = boxes_of(p, 0)
    check("an adjusted box replaces the model's box", zero is not None and zero[0]["xyxy"] == [5.0, 5.0, 20.0, 20.0] and zero[0]["box_source"] == "expert_adjusted")
    check("a frame whose only box an expert removed is left out", boxes_of(p, 1) is None)
    check("the removal is counted among the frames left out", p["skipped"]["frames_left_out"].get("no usable box") == 1)
    two = boxes_of(p, 2)
    check("an added box sits beside the model's box", two is not None and [b["box_source"] for b in two] == ["model", "expert_added"])
    check("the added box carries the expert's species", two[1]["taxon"]["class_key"] == "gbif:200" and two[1]["taxon"]["source"] == "expert")
    check("the box counts report the edits", p["box_stats"]["expert_adjusted"] == 1 and p["box_stats"]["expert_added"] == 1)
    check("the added species becomes a class of its own", {c["class_key"] for c in p["classes"]} == {"gbif:100", "gbif:200"})

    w.db.add_frame_box_correction(target["id"], 0, action="cleared", corrector="tester")
    back = boxes_of(plan(w), 0)
    check("clearing a correction restores the model's box", back is not None and back[0]["box_source"] == "model" and back[0]["xyxy"] == [10.0, 8.0, 40.0, 30.0])

    data_before = tree_digest(w.data)
    built = build(w)
    manifest = json.loads((Path(built["path"]) / "manifest.json").read_text(encoding="utf-8"))
    edited = [i for i in manifest["items"] if i["event_id"] == target["id"] and i["box_expert_edited"]]
    check("the manifest flags examples whose boxes an expert edited", len(edited) >= 1)
    check("the manifest names each box's source", any(b["box_source"] == "expert_added" for i in manifest["items"] for b in i["boxes"]))
    check("the original annotation files are untouched", tree_digest(w.data) == data_before)


def _counts(p: dict) -> dict:
    per_event = {}
    for s in ds.SPLITS:
        for item in p["items"][s]:
            per_event[item["event_id"]] = per_event.get(item["event_id"], 0) + 1
    return per_event


def test_model_labelled(work: Path) -> None:
    print("\n[4] Examples that carry the model's own label")
    w = World(work / "model")
    confirmed(w, A, 4, prefix="K")
    for i in range(6):
        e = w.vision_event([A], station=f"K{i % 3}", first_seen=f"2026-05-{i + 1:02d}T10:00:00Z")
        for f in range(6):
            w.frame(e, f, "accurate")
    off = plan(w, required_criteria=[])
    check("with the cap at zero no model-labelled example is used", all(not i["model_labelled"] for s in ds.SPLITS for i in off["items"][s]))
    check("turning Expert Confirmed off is warned about", any("Expert Confirmed is switched off" in x for x in off["warnings"]))

    on = plan(w, required_criteria=[], model_labelled_max_fraction=0.5)
    flagged = [(s, i) for s in ds.SPLITS for i in on["items"][s] if i["model_labelled"]]
    expert_count = sum(len(i["boxes"]) for s in ds.SPLITS for i in on["items"][s] if not i["model_labelled"] and i["role"] == "positive")
    check("model-labelled examples are used once the cap allows", len(flagged) > 0)
    check("they only ever reach training", all(s == "train" for s, _ in flagged))
    check("they stay within the cap", len(flagged) <= expert_count)
    check("they are flagged in the manifest", all(i["model_labelled"] is True for _, i in flagged))
    check("their use is warned about", any("labelled by the model" in x for x in on["warnings"]))
    check("held-out sets hold only expert-judged examples", all(not i["model_labelled"] for s in ("validation", "test") for i in on["items"][s]))
    stricter = plan(w, required_criteria=[], model_labelled_max_fraction=0.2)
    n_strict = sum(1 for s in ds.SPLITS for i in stricter["items"][s] if i["model_labelled"])
    check("a smaller cap admits fewer", n_strict < len(flagged))
    check("with the default checks the unjudged events are not candidates at all", plan(w)["eligible"] == 4)

    mixed = World(work / "model_mixed")
    for i in range(12):
        e = mixed.vision_event([A, C], station=f"K{i % 3}", first_seen=f"2026-06-{i + 1:02d}T10:00:00Z", frames=6, class_names=[A[0], C[0]])
        mixed.confirm(e, detection=e["children"][0])
        for f in (1, 3, 5):
            mixed.frame(e, f, "accurate")
    pm = plan(mixed, required_criteria=[], model_labelled_max_fraction=0.9)
    held_out_events = {e["event_id"] for e in pm["events"] if e["split"] in ("validation", "test")}
    model_items = [(s_, i) for s_ in ds.SPLITS for i in pm["items"][s_] if i["model_labelled"]]
    check("a partly judged event gives expert examples and model-labelled ones", len(model_items) > 0 and any(not i["model_labelled"] for i in pm["items"]["train"]))
    check("some of those events were held out for validation or test", len(held_out_events) > 0)
    check("a held-out event never carries a model-labelled example", all(s_ == "train" for s_, _ in model_items) and all(not i["model_labelled"] for s_ in ("validation", "test") for i in pm["items"][s_]))
    check("the removed examples are reported", any("fell in events held out for validation or test" in x for x in pm["warnings"]))


def test_class_policy(work: Path) -> None:
    print("\n[5] Rare classes, merging and untested classes")
    w = World(work / "policy")
    confirmed(w, A, 8, prefix="P")
    confirmed(w, B, 3, prefix="Q")
    p = plan(w, ROLL)
    rare = p["class_policy"]["gbif:102"]
    check("a class under the threshold is kept with a warning by default", rare["action"] == "keep" and rare["under_threshold"] is True)
    check("the warning names the class and the shortfall", any("Aplysina lacunosa was kept with only 3" in x for x in p["warnings"]))
    check("a class at the threshold is not flagged", p["class_policy"]["gbif:100"]["under_threshold"] is False)

    left = plan(w, ROLL, class_policy={"gbif:102": "exclude"})
    check("a class can be left out", [c["class_name"] for c in left["classes"]] == [A[0]])
    check("leaving it out removes its events", len(left["events"]) == 8)

    genus = plan(w, ROLL, class_policy={"gbif:102": "roll_up_genus"})
    names = {c["class_name"]: c for c in genus["classes"]}
    check("a rare class can merge into its genus", set(names) == {A[0], "Aplysina"})
    check("the merged class lists what it took in", names["Aplysina"]["merged_from"] == [{"class_key": "gbif:102", "class_name": B[0]}])
    check("the genus carries its taxonomic key", names["Aplysina"]["gbif_usage_key"] == "20")
    check("the merge is described in the class decisions", genus["class_policy"]["gbif:102"]["note"] == "rolled up into Aplysina")

    both = plan(w, ROLL, class_policy={"gbif:102": "roll_up_genus", "gbif:100": "roll_up_genus"})
    check("when both merge into one genus there is one class", [c["class_name"] for c in both["classes"]] == ["Aplysina"])
    check("that class holds the events of both", sum(c["counts"][s]["events"] for c in both["classes"] for s in ds.SPLITS) == 11)
    family = plan(w, ROLL, class_policy={"gbif:102": "roll_up_family"})
    check("a rare class can merge into its family", "Aplysinidae" in {c["class_name"] for c in family["classes"]})

    nothing = plan(w, {}, class_policy={"gbif:102": "roll_up_genus"})
    check("a class that cannot be placed stays as it was", B[0] in {c["class_name"] for c in nothing["classes"]})
    check("the failure is explained", any("could not be rolled up" in x for x in nothing["warnings"]))

    rolled_default = ds.options_from_request(w.settings, {"modality": "vision"})
    w.settings.raw["training"]["dataset"]["under_threshold_default"] = "exclude"
    excluded = ds.plan_dataset(w.db, w.settings, ds.options_from_request(w.settings, {"modality": "vision"}), rollup_lookup=fake_rollup(ROLL))
    check("the configured default can leave rare classes out", [c["class_name"] for c in excluded["classes"]] == [A[0]])
    w.settings.raw["training"]["dataset"]["under_threshold_default"] = "roll_up"
    merged = ds.plan_dataset(w.db, w.settings, ds.options_from_request(w.settings, {"modality": "vision"}), rollup_lookup=fake_rollup(ROLL))
    check("the configured default can merge rare classes", "Aplysina" in {c["class_name"] for c in merged["classes"]})
    w.settings.raw["training"]["dataset"]["under_threshold_default"] = "keep_with_warning"

    few = World(work / "few")
    confirmed(few, A, 8, prefix="U")
    confirmed(few, C, 2, prefix="V")
    pf = plan(few, ROLL, min_examples_per_class=2)
    classes = {c["class_name"]: c for c in pf["classes"]}
    check("a class with two events is trained on but not scored", classes[C[0]]["tested"] is False and classes[A[0]]["tested"] is True)
    check("it is reported as untested in the plan", len(pf["split"]["untested_classes"]) == 1)
    check("its untested state is warned about", any("Poecile carolinensis is untested" in x for x in pf["warnings"]))
    check("all its events are in training", all(pf["split"]["assignment"][e["event_id"]] == "train" for e in pf["events"] if e["event_name"].startswith("V")))

    station = plan(w, ROLL, split={"mode": "station"})
    check("holding out whole stations falls back, saying so, with too few stations", station["split"]["mode_used"] in ("event", "station"))
    stations = World(work / "stations")
    for i in range(12):
        e = stations.vision_event([A], station=f"T{i % 4}", first_seen=f"2026-03-{i + 1:02d}T10:00:00Z"); stations.confirm(e)
    ps = plan(stations, split={"mode": "station"})
    held = {}
    for e in ps["events"]:
        held.setdefault(e["station_id"], set()).add(e["split"])
    check("with four stations whole stations are held out", ps["split"]["mode_used"] == "station" and all(len(v) == 1 for v in held.values()) and "test" in {next(iter(v)) for v in held.values()})
    only = plan(stations, station_id=stations.stations["T0"])
    check("a station filter keeps only that station's events", {e["station_id"] for e in only["events"]} == {stations.stations["T0"]})


def test_safety(work: Path) -> None:
    print("\n[6] Failing safely")
    w = standard_world(work / "safe")
    inside = World(work / "inside")
    confirmed(inside, A, 5, prefix="I")
    inside.settings.raw["training"]["paths"]["datasets_dir"] = "data/train_out"
    check("a datasets folder inside the data folder is refused", raises(lambda: ds.plan_dataset(inside.db, inside.settings, ds.options_from_request(inside.settings, {"modality": "vision"}), rollup_lookup=fake_rollup({})), ds.DatasetError))
    check("it is refused for a build too", raises(lambda: ds.build_dataset(inside.db, inside.settings, ds.options_from_request(inside.settings, {"modality": "vision"}), rollup_lookup=fake_rollup({})), ds.DatasetError))
    check("nothing was created inside the data folder", not (inside.data / "train_out").exists())

    empty = World(work / "empty")
    confirmed(empty, A, 3, prefix="Z", qc="qc_pending")
    summary = ds.summarize_plan(plan(empty))
    check("a selection that matches nothing previews as empty", summary["empty"] is True and summary["eligible"] == 0)
    check("building an empty selection is refused in plain words", raises(lambda: build(empty), ds.DatasetError))
    check("nothing is created for it", not any(Path(empty.settings.training_paths()["datasets_dir"]).glob("vision-*")))

    real_place = formats.place_file
    calls = {"n": 0}

    def flaky(source, target, *, link=True):
        calls["n"] += 1
        if calls["n"] == 7:
            raise OSError("disk full")
        return real_place(source, target, link=link)

    formats.place_file = flaky
    try:
        failed = raises(lambda: build(w), OSError)
    finally:
        formats.place_file = real_place
    root = Path(w.settings.training_paths()["datasets_dir"])
    check("a failure part way through is reported", failed)
    check("it leaves no finished dataset", not list(root.glob("vision-*")))
    check("it leaves no half-written folder behind", not [p for p in root.iterdir() if p.name.endswith(".partial")])
    check("an unfinished folder is never listed as a dataset", ds.list_datasets(w.settings) == [])
    again = build(w)
    check("building again afterwards succeeds", again["already_exists"] is False and (Path(again["path"]) / "manifest.json").is_file())

    (root / ".stale.partial").mkdir()
    (root / ".stale.partial" / "manifest.json").write_text("{}", encoding="utf-8")
    (root / ".stale.partial" / "summary.json").write_text("{}", encoding="utf-8")
    (root / "vision-unfinished").mkdir()
    (root / "vision-unfinished" / "manifest.json").write_text("{}", encoding="utf-8")
    check("a stale partial folder is not listed either", [d["dataset_id"] for d in ds.list_datasets(w.settings)] == [again["dataset_id"]])
    check("a folder without its summary is not a finished dataset", all(d["dataset_id"] != "vision-unfinished" for d in ds.list_datasets(w.settings)))
    check("a finished dataset has a short summary beside its manifest", (Path(again["path"]) / "summary.json").is_file() and (Path(again["path"]) / "summary.json").stat().st_size < (Path(again["path"]) / "manifest.json").stat().st_size)


def test_audio(work: Path) -> None:
    print("\n[7] An acoustic dataset")
    w = World(work / "audio")
    for i in range(6):
        e = w.audio_event([C], station=f"A{i % 3}", first_seen=f"2026-03-{i + 1:02d}T06:00:00Z"); w.confirm(e)
    for i in range(2):
        e = w.audio_event([C], station="A0", first_seen=f"2026-04-{i + 1:02d}T06:00:00Z", burst=None); w.reject(e)
    data_before, db_before = tree_digest(w.data), db_digest(w)
    p = ds.plan_dataset(w.db, w.settings, ds.options_from_request(w.settings, {"modality": "audio", "engine": "acoustic_head_a", "background_max_fraction": 1.0}), rollup_lookup=fake_rollup({}))
    s = ds.summarize_plan(p)
    check("an acoustic plan finds the events", s["eligible"] == 8)
    items = [i for sp in ds.SPLITS for i in p["items"][sp]]
    positives = [i for i in items if i["role"] == "positive"]
    check("each confirmed clip gives between one and three windows", all(1 <= sum(1 for i in positives if i["event_id"] == e["event_id"]) <= 3 for e in p["events"] if e["label_state"] == "positive"))
    check("each window is the engine's three seconds", all(abs((i["end"] - i["start"]) - 3.0) < 1e-6 for i in items))
    check("the window around the sound is chosen", all(i["start"] <= 4.0 and i["end"] >= 5.5 for i in positives[:6] if i["method"] == "energy"))
    check("a rejected clip gives a noise window with no species", any(i["role"] == "background" and not i["labels"] and i["method"] == "noise" for i in items))
    check("windows chosen by energy are said to be", any("sound energy" in x for x in p["warnings"]))
    check("a clip recorded at another rate than the engine's is warned about", any("sample rate other than" in x for x in p["warnings"]))
    check("planning an acoustic dataset changed nothing", tree_digest(w.data) == data_before and db_digest(w) == db_before)

    built = ds.build_dataset(w.db, w.settings, ds.options_from_request(w.settings, {"modality": "audio", "engine": "acoustic_head_a", "background_max_fraction": 1.0}), rollup_lookup=fake_rollup({}))
    folder = Path(built["path"])
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    import wave
    lengths = set()
    for item in manifest["items"]:
        with wave.open(str(folder / item["file"]), "rb") as h:
            lengths.add((h.getnframes(), h.getframerate()))
    check("every written window is exactly three seconds at the clip's own rate", lengths == {(3 * 24000, 24000)})
    rows = (folder / "windows.csv").read_text(encoding="utf-8").strip().splitlines()
    check("the window table lists every window with its source times", len(rows) - 1 == len(manifest["items"]) and rows[0].startswith("split,file,event_id,labels,start_seconds"))
    check("an acoustic dataset has window folders per split", (folder / "windows" / "train").is_dir())
    check("building an acoustic dataset changed nothing", tree_digest(w.data) == data_before and db_digest(w) == db_before)
    check("no window is in a different split from its event", all(manifest["split"]["assignment"][i["event_id"]] == i["split"] for i in manifest["items"]))
    again = ds.build_dataset(w.db, w.settings, ds.options_from_request(w.settings, {"modality": "audio", "engine": "acoustic_head_a", "background_max_fraction": 1.0}), rollup_lookup=fake_rollup({}))
    check("an acoustic dataset is recognised on a second build", again["already_exists"] is True)
    other = ds.plan_dataset(w.db, w.settings, ds.options_from_request(w.settings, {"modality": "audio", "engine": "acoustic_head_b", "background_max_fraction": 1.0}), rollup_lookup=fake_rollup({}))
    check("a different engine is a different dataset", other["dataset_id"] != p["dataset_id"])

    clipless = World(work / "clipless")
    ev = clipless.audio_event([C], station="A0")
    clipless.confirm(ev)
    ev["path"].unlink()
    skipped = ds.plan_dataset(clipless.db, clipless.settings, ds.options_from_request(clipless.settings, {"modality": "audio", "engine": "acoustic_head_a"}), rollup_lookup=fake_rollup({}))
    check("a missing clip is left out with a reason", skipped["skipped"]["events"][0]["reason"] == "the clip file is missing" and skipped["skipped"]["missing_media"] == 1)
    short = World(work / "short")
    ev = short.audio_event([C], station="A0", seconds=1.0, burst=None)
    short.confirm(ev)
    p2 = ds.plan_dataset(short.db, short.settings, ds.options_from_request(short.settings, {"modality": "audio", "engine": "acoustic_head_a"}), rollup_lookup=fake_rollup({}))
    check("a clip too short for one window is left out with a reason", "too short" in p2["skipped"]["events"][0]["reason"])


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        test_options(work)
        test_plan_and_build(work)
        test_frame_rules_end_to_end(work)
        test_box_corrections(work)
        test_model_labelled(work)
        test_class_policy(work)
        test_safety(work)
        test_audio(work)
    print(f"\nResult: {CHECKS['passed']} passed, {CHECKS['failed']} failed")
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
