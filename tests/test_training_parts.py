"""Verification for the parts a training dataset is assembled from.

Path: tests/test_training_parts.py

Checks, one part at a time and without a database, the rules that decide which
frames and sound windows become examples, how events are divided into training,
validation and test, how files are placed and written, and how a rare class is
merged into its genus or family through the taxonomic backbone. The backbone
checks use a small file in the shipped layout, and also prove the lookup only
reads it.

Uses the standard library, NumPy and Pillow, plus the project itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from training_fixture import make_backbone  # noqa: E402
from audtheia.training import formats, frames, splits, taxon_rollup, windows  # noqa: E402
from audtheia.training.candidates import LABEL_MODEL, LABEL_NEGATIVE, LABEL_POSITIVE  # noqa: E402

CHECKS = {"passed": 0, "failed": 0}


def check(label: str, condition: bool) -> None:
    CHECKS["passed" if condition else "failed"] += 1
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")


def raises(fn, exc=Exception) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------


def make_units(counts: dict, stations=("S1", "S2", "S3", "S4")) -> list:
    units = []
    n = 0
    for class_key, how_many in counts.items():
        for i in range(how_many):
            n += 1
            units.append({"id": f"{class_key}-{i:02d}", "class_key": class_key,
                          "station_id": stations[n % len(stations)],
                          "first_seen": f"2026-01-{(n % 28) + 1:02d}T10:00:00Z", "expert": True})
    return units


def test_splits() -> None:
    print("\n[1] Dividing events without leakage")
    fractions = {"train": 0.70, "validation": 0.15, "test": 0.15}
    units = make_units({"gbif:1": 10, "gbif:2": 20, "gbif:3": 2, "background": 4})
    first = splits.assign_splits(units, mode="event", seed=7, fractions=fractions, min_events_per_class_for_heldout=3)
    again = splits.assign_splits(units, mode="event", seed=7, fractions=fractions, min_events_per_class_for_heldout=3)
    other = splits.assign_splits(units, mode="event", seed=8, fractions=fractions, min_events_per_class_for_heldout=3)
    a = first["assignment"]

    check("every event is placed exactly once", sorted(a) == sorted(u["id"] for u in units))
    check("every placement is a known split", set(a.values()) <= set(splits.SPLITS))
    check("the same seed gives the same division", first == again)
    check("a different seed gives a different division", other["assignment"] != a)

    def count(cls, split):
        return sum(1 for u in units if u["class_key"] == cls and a[u["id"]] == split)

    check("ten events divide into six, two and two", (count("gbif:1", "train"), count("gbif:1", "validation"), count("gbif:1", "test")) == (6, 2, 2))
    check("twenty events divide into fourteen, three and three", (count("gbif:2", "train"), count("gbif:2", "validation"), count("gbif:2", "test")) == (14, 3, 3))
    check("a class with too few events is trained on only", count("gbif:3", "train") == 2)
    check("that class is reported as untested", "gbif:3" in first["untested_classes"])
    check("a class with enough events is not reported as untested", "gbif:1" not in first["untested_classes"] and "gbif:2" not in first["untested_classes"])
    check("false alarms are divided too but never reported as a class", "background" not in first["untested_classes"])

    smaller = [u for u in units if u["class_key"] != "gbif:2"]
    reduced = splits.assign_splits(smaller, mode="event", seed=7, fractions=fractions, min_events_per_class_for_heldout=3)
    check("removing one class never changes how another is divided",
          all(reduced["assignment"][u["id"]] == a[u["id"]] for u in smaller))

    mixed = [dict(u) for u in units]
    mixed[0]["expert"] = False
    placed = splits.assign_splits(mixed, mode="event", seed=7, fractions=fractions, min_events_per_class_for_heldout=3)
    check("an event labelled by the model goes to training only", placed["assignment"][mixed[0]["id"]] == "train")

    check("fractions that do not add up are refused", raises(lambda: splits.assign_splits(units, mode="event", seed=1, fractions={"train": 0.5, "validation": 0.1, "test": 0.1}), splits.SplitError))
    check("a zero training fraction is refused", raises(lambda: splits.assign_splits(units, mode="event", seed=1, fractions={"train": 0.0, "validation": 0.5, "test": 0.5}), splits.SplitError))
    check("an unknown mode is refused", raises(lambda: splits.assign_splits(units, mode="shuffle", seed=1, fractions=fractions), splits.SplitError))

    station = splits.assign_splits(units, mode="station", seed=7, fractions=fractions)
    held = {}
    for u in units:
        held.setdefault(u["station_id"], set()).add(station["assignment"][u["id"]])
    check("station mode used the station division", station["mode_used"] == "station")
    check("no station is split across sets", all(len(v) == 1 for v in held.values()))
    check("station mode holds something out for test", "test" in station["assignment"].values())
    few = splits.assign_splits(make_units({"gbif:1": 12}, stations=("S1", "S2")), mode="station", seed=7, fractions=fractions)
    check("two stations are too few to hold one out, so events are divided one by one", few["mode_used"] == "event")
    check("the fallback is explained in plain words", any("three stations" in n for n in few["notes"]))

    dated = [{"id": f"e{i:02d}", "class_key": "gbif:1", "station_id": "S1", "first_seen": f"2026-02-{i + 1:02d}T00:00:00Z", "expert": True} for i in range(20)]
    timed = splits.assign_splits(dated, mode="time_block", seed=1, fractions=fractions)
    latest = sorted(dated, key=lambda u: u["first_seen"])[-3:]
    check("time mode holds the newest events out for test", all(timed["assignment"][u["id"]] == "test" for u in latest))
    check("time mode keeps the oldest events for training", timed["assignment"]["e00"] == "train")
    by_days = splits.assign_splits(dated, mode="time_block", seed=1, fractions=fractions, time_block_days=2)
    check("a block of days holds out the events inside the last days", by_days["assignment"]["e19"] == "test" and by_days["assignment"]["e00"] == "train")


# ---------------------------------------------------------------------------


def write_event(folder: Path, rows: list, images: bool = True, identical: bool = False) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for r in rows:
        if images:
            rng = np.random.RandomState(1 if identical else int(r["index"]) + 100)
            Image.fromarray(rng.randint(0, 255, size=(48, 64, 3), dtype=np.uint8)).save(folder / r["file"], quality=95)
    (folder / "annotations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def taxon(key, name, verdict, source=None, names=()):
    return {"class_key": key, "class_name": name, "gbif_usage_key": key.split(":")[1] if key.startswith("gbif:") else None,
            "verdict": verdict, "source": source or ("model" if verdict is None else "expert"), "match_names": list(names) or [name.lower()]}


def cand(taxa, state):
    return {"taxa": taxa, "label_state": state, "observation_id": "o1", "event_name": "ev"}


def rows_for(names):
    return [{"index": i, "file": f"frame_{i:06d}.jpg", "class_name": n, "bbox_xyxy": [10, 8, 40, 30], "_bbox": [10.0, 8.0, 40.0, 30.0]} for i, n in enumerate(names)]


def test_frames(work: Path) -> None:
    print("\n[2] Which frames become examples")
    data = work / "data"
    inside = data / "detections" / "visual" / "ev" / "frame_000000.jpg"
    check("an event folder inside the data folder is accepted", frames.event_directory(work, data, "data/detections/visual/ev/frame_000000.jpg") == inside.parent.resolve())
    check("a stored path that escapes the data folder is refused", raises(lambda: frames.event_directory(work, data, "../outside/ev/frame_000000.jpg"), frames.FrameSourceError))
    check("an absolute path elsewhere is refused", raises(lambda: frames.event_directory(work, data, str(work.parent / "x" / "f.jpg")), frames.FrameSourceError))
    check("an event with no stored frame is refused", raises(lambda: frames.event_directory(work, data, ""), frames.FrameSourceError))

    folder = work / "ann"
    write_event(folder, rows_for(["a", "a"]), images=False)
    with open(folder / "annotations.jsonl", "a", encoding="utf-8") as fh:
        fh.write("not json\n\n" + json.dumps({"index": 9, "file": ""}) + "\n" + json.dumps({"index": 5, "file": "frame_000005.jpg", "class_name": "a", "bbox_xyxy": [1, 2, 3, 4]}) + "\n")
    read = frames.read_annotations(folder)
    check("unreadable and incomplete annotation lines are skipped", [r["index"] for r in read] == [0, 1, 5])
    check("a missing annotation file reads as no frames", frames.read_annotations(work / "nowhere") == [])

    a = taxon("gbif:1", "Species A", "confirm")
    ann = rows_for(["Species A", "Species A", "other", "Species A"])
    sel = frames.select_frames(cand([a], LABEL_POSITIVE), ann, {1: "inaccurate"}, [])
    kept = [f["rec"]["index"] for f in sel["frames"]]
    check("a frame marked inaccurate is left out", 1 not in kept)
    check("it is not turned into a background example", all(f["role"] == "positive" for f in sel["frames"]))
    check("a confirmed event keeps frames whose class matches", kept == [0, 3])
    check("a frame of another class nobody looked at is left out as uncertain", 2 not in kept)
    sel = frames.select_frames(cand([a], LABEL_POSITIVE), ann, {2: "accurate"}, [])
    check("a frame an expert marked accurate is kept even if the model's class differs", 2 in [f["rec"]["index"] for f in sel["frames"]])

    gone = taxon("gbif:9", "Rejected sp", "reject")
    sel = frames.select_frames(cand([gone], LABEL_NEGATIVE), ann, {0: "inaccurate"}, [])
    check("a rejected event makes every other frame a background example", {f["rec"]["index"] for f in sel["frames"]} == {1, 2, 3} and all(f["role"] == "background" and f["boxes"] == [] for f in sel["frames"]))

    rel = taxon("gbif:2", "Expert species", "relabel", names=["species a"])
    sel = frames.select_frames(cand([rel], LABEL_POSITIVE), ann, {}, [])
    check("a relabelled single detection gives every frame the expert's species", len(sel["frames"]) == 4 and all(b["taxon"]["class_key"] == "gbif:2" for f in sel["frames"] for b in f["boxes"]))

    model = taxon("gbif:1", "Species A", None)
    sel = frames.select_frames(cand([model], LABEL_MODEL), ann, {0: "accurate", 3: "accurate"}, [])
    check("without a verdict only frames marked accurate are kept", [f["rec"]["index"] for f in sel["frames"]] == [0, 3])
    check("those frames are flagged as model-labelled", all(f["model_labelled"] for f in sel["frames"]))
    check("unreviewed model-labelled frames are left out", frames.select_frames(cand([model], LABEL_MODEL), ann, {}, [])["frames"] == [])

    box = lambda **k: dict({"frame_index": 0, "box_key": "model", "action": "adjust", "x1": 5, "y1": 5, "x2": 20, "y2": 20}, **k)
    sel = frames.select_frames(cand([a], LABEL_POSITIVE), rows_for(["Species A"]), {}, [box()])
    check("an adjusted box replaces the model's box", sel["frames"][0]["boxes"][0]["xyxy"] == [5, 5, 20, 20] and sel["frames"][0]["boxes"][0]["box_source"] == "expert_adjusted")
    check("an adjusted frame is flagged as expert edited", sel["frames"][0]["box_edited"] is True)
    sel = frames.select_frames(cand([a], LABEL_POSITIVE), rows_for(["Species A"]), {}, [box(action="remove")])
    check("a removed box leaves the frame with no usable box, so it is left out", sel["frames"] == [] and sel["skipped"].get("no usable box") == 1)
    added = {"frame_index": 0, "box_key": "extra1", "action": "add", "x1": 30, "y1": 10, "x2": 50, "y2": 40,
             "corrected_scientific_name": "Added species", "corrected_gbif_usage_key": "77"}
    sel = frames.select_frames(cand([a], LABEL_POSITIVE), rows_for(["Species A"]), {}, [added])
    boxes = sel["frames"][0]["boxes"]
    check("an added box brings its own species", [b["taxon"]["class_key"] for b in boxes] == ["gbif:1", "gbif:77"])
    check("an added box is marked as the expert's", boxes[1]["box_source"] == "expert_added" and boxes[1]["taxon"]["source"] == "expert")
    later = [box(action="adjust"), box(action="cleared")]
    sel = frames.select_frames(cand([a], LABEL_POSITIVE), rows_for(["Species A"]), {}, later)
    check("a cleared correction restores the model's box", sel["frames"][0]["boxes"][0]["box_source"] == "model")

    big = work / "big"
    many = rows_for(["Species A"] * 30)
    write_event(big, many)
    sel = frames.select_frames(cand([a], LABEL_POSITIVE), many, {}, [])
    thin = frames.thin_frames(sel["frames"], big, max_frames=10, near_duplicate_hamming=4)
    idx = [f["rec"]["index"] for f in thin["frames"]]
    check("thirty frames are thinned to the cap", len(idx) == 10)
    check("the thinning always keeps the first and last frames", idx[0] == 0 and idx[-1] == 29)
    check("the thinning is spread along the track", all(b > a_ for a_, b in zip(idx, idx[1:])) and max(b - a_ for a_, b in zip(idx, idx[1:])) <= 4)
    check("thinning twice gives the same frames", [f["rec"]["index"] for f in frames.thin_frames(sel["frames"], big, max_frames=10, near_duplicate_hamming=4)["frames"]] == idx)

    same = work / "same"
    six = rows_for(["Species A"] * 6)
    write_event(same, six, identical=True)
    sel = frames.select_frames(cand([a], LABEL_POSITIVE), six, {}, [])
    thin = frames.thin_frames(sel["frames"], same, max_frames=10, near_duplicate_hamming=4)
    check("identical frames collapse to one", len(thin["frames"]) == 1 and thin["near_duplicates"] == 5)
    thin = frames.thin_frames(sel["frames"], same, max_frames=10, near_duplicate_hamming=0)
    check("with the check off, identical frames all stay", len(thin["frames"]) == 6)

    sparse = work / "sparse"
    write_event(sparse, six)
    (sparse / "frame_000002.jpg").unlink()
    (sparse / "frame_000004.jpg").write_bytes(b"not an image")
    sel = frames.select_frames(cand([a], LABEL_POSITIVE), six, {}, [])
    thin = frames.thin_frames(sel["frames"], sparse, max_frames=10, near_duplicate_hamming=4)
    check("a missing or unreadable image is dropped and counted, never fatal", len(thin["frames"]) == 4 and thin["missing"] == 2)
    evil = [{"index": 0, "file": "../ann/annotations.jsonl", "class_name": "Species A", "_bbox": [1, 2, 3, 4]}]
    sel = frames.select_frames(cand([a], LABEL_POSITIVE), evil, {}, [])
    thin = frames.thin_frames(sel["frames"], sparse, max_frames=10, near_duplicate_hamming=4)
    check("an annotation naming a file outside the event folder is refused", thin["frames"] == [] and thin["missing"] == 1)


# ---------------------------------------------------------------------------


def tone_clip(seconds=8.0, rate=24000, burst=(4.0, 5.5)):
    rng = np.random.RandomState(3)
    t = np.arange(int(seconds * rate)) / float(rate)
    sig = 0.01 * rng.randn(len(t))
    if burst:
        m = (t >= burst[0]) & (t < burst[1])
        sig[m] += 0.5 * np.sin(2 * np.pi * 2000.0 * t[m])
    return sig.astype(np.float32), rate


def test_windows(work: Path) -> None:
    print("\n[3] Which sound windows become examples")
    sig, rate = tone_clip()
    path = work / "clip.wav"
    windows.write_wav_pcm16(sig, path, rate)
    back, back_rate = windows.read_wav(path)
    check("a clip written and read back keeps its rate and length", back_rate == rate and len(back) == len(sig))
    check("a clip written and read back keeps its sound to 16 bit precision", float(np.max(np.abs(back - sig))) < 1e-3)

    picked = windows.select_windows(back, rate, window_seconds=3.0, max_windows=3, selection="energy_only")
    check("the loudest window is the one holding the sound", len(picked) >= 1 and picked[0]["start"] <= 4.0 and picked[0]["end"] >= 5.5)
    check("windows chosen by energy say so", all(w["method"] == windows.METHOD_ENERGY for w in picked))
    check("windows never overlap", all(b["start"] >= a["end"] - 1e-6 for a, b in zip(picked, picked[1:])))
    check("no more windows than asked for", len(picked) <= 3)
    check("every window is the full length", all(abs((w["end"] - w["start"]) - 3.0) < 1e-6 for w in picked))
    check("choosing windows twice gives the same windows", picked == windows.select_windows(back, rate, window_seconds=3.0, max_windows=3, selection="energy_only"))

    centre = windows.select_windows(back, rate, window_seconds=3.0, max_windows=3, selection="clip_centre")
    check("the centre option takes one window from the middle", len(centre) == 1 and abs(centre[0]["start"] - 2.5) < 1e-6 and centre[0]["method"] == windows.METHOD_CENTRE)

    def scorer(samples, r, starts, window):
        return [1.0 if abs(s - 0.5) < 1e-6 else 0.0 for s in starts]

    scored = windows.select_windows(back, rate, window_seconds=3.0, max_windows=1, selection="scored_with_energy_fallback", scorer=scorer)
    check("a scoring model's best window is chosen when one is supplied", scored[0]["start"] == 0.5 and scored[0]["method"] == windows.METHOD_SCORED)
    weak = windows.select_windows(back, rate, window_seconds=3.0, max_windows=1, selection="scored_with_energy_fallback", scorer=lambda s, r, st, w: [0.0] * len(st))
    check("when the scorer finds nothing above the floor, energy takes over", weak[0]["method"] == windows.METHOD_ENERGY)
    nothing = windows.select_windows(back, rate, window_seconds=3.0, max_windows=1, selection="scored_with_energy_fallback")
    check("with no scorer the windows fall back to energy", nothing[0]["method"] == windows.METHOD_ENERGY)

    silent = np.zeros(8 * rate, dtype=np.float32)
    quiet = windows.select_windows(silent, rate, window_seconds=3.0, max_windows=2, selection="energy_only")
    check("a clip with no sound falls back to the middle, never to nothing", len(quiet) == 1 and quiet[0]["method"] == windows.METHOD_CENTRE)
    short = windows.select_windows(back[: int(1.0 * rate)], rate, window_seconds=3.0, max_windows=2)
    check("a clip under half a window yields nothing", short == [])
    near = windows.select_windows(back[: int(2.0 * rate)], rate, window_seconds=3.0, max_windows=2)
    check("a clip over half a window yields one padded window", len(near) == 1)
    piece, padded = windows.cut_window(back[: int(2.0 * rate)], rate, 0.0, 3.0)
    check("a short window is padded with silence to full length", padded is True and len(piece) == 3 * rate and float(np.abs(piece[2 * rate:]).max()) == 0.0)
    piece, padded = windows.cut_window(back, rate, 1.0, 3.0)
    check("a window inside the clip is not padded", padded is False and len(piece) == 3 * rate)

    rng_a, rng_b = random.Random("n"), random.Random("n")
    noise_a = windows.noise_windows(8.0, 3.0, 2, rng_a)
    check("noise windows are reproducible", noise_a == windows.noise_windows(8.0, 3.0, 2, rng_b))
    check("noise windows do not overlap and fit in the clip", all(w["end"] <= 8.0 + 1e-6 for w in noise_a) and (len(noise_a) < 2 or noise_a[1]["start"] - noise_a[0]["start"] >= 3.0))
    check("a clip under half a window gives no noise window", windows.noise_windows(1.0, 3.0, 1, random.Random(1)) == [])

    bad = work / "bad.wav"
    bad.write_bytes(b"RIFF0000")
    check("a clip that is not a WAV file is reported, not crashed on", raises(lambda: windows.read_wav(bad), windows.AudioSourceError))
    before = sha(path)
    windows.select_windows(back, rate, window_seconds=3.0, max_windows=3)
    check("reading and choosing windows never changes the clip", sha(path) == before)


# ---------------------------------------------------------------------------


def test_formats(work: Path) -> None:
    print("\n[4] Placing files and writing the two layouts")
    src = work / "src.jpg"
    Image.fromarray(np.random.RandomState(1).randint(0, 255, (48, 64, 3), dtype=np.uint8)).save(src)
    original = sha(src)
    linked = work / "out" / "linked.jpg"
    kind = formats.place_file(src, linked, link=True)
    check("a file is hard-linked when the disk allows it, otherwise copied", kind in ("link", "copy"))
    check("the placed file has the same bytes", sha(linked) == original)
    copied = work / "out" / "copied.jpg"
    check("copying can be asked for", formats.place_file(src, copied, link=False) == "copy" and os.stat(copied).st_ino != os.stat(src).st_ino)
    check("placing over an existing file leaves it alone", formats.place_file(src, linked, link=False) == "exists")
    check("the original is untouched by placing", sha(src) == original)

    check("a box is clamped to the image", formats.clamp_box([-5, -5, 70, 60], 64, 48) == (0.0, 0.0, 64.0, 48.0))
    check("a reversed box is put right", formats.clamp_box([40, 30, 10, 8], 64, 48) == (10.0, 8.0, 40.0, 30.0))
    check("a box with no area left is dropped", formats.clamp_box([70, 10, 80, 20], 64, 48) is None)

    images = [{"file_name": "b.jpg", "width": 64, "height": 48, "boxes": [{"category_id": 2, "xyxy": [10, 8, 40, 30]}]},
              {"file_name": "a.jpg", "width": 64, "height": 48, "boxes": []},
              {"file_name": "c.jpg", "width": 64, "height": 48, "boxes": [{"category_id": 1, "xyxy": [70, 70, 80, 80]}]}]
    counts = formats.write_coco_split(work / "coco" / "train", images, [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}], description="d")
    doc = json.loads((work / "coco" / "train" / "_annotations.coco.json").read_text(encoding="utf-8"))
    check("the annotation file lists every image in a stable order", [i["file_name"] for i in doc["images"]] == ["a.jpg", "b.jpg", "c.jpg"])
    check("an image with no boxes stays as a background example", doc["images"][0]["id"] == 1 and not [a for a in doc["annotations"] if a["image_id"] == 1])
    check("a box with nothing left inside the image is dropped and counted", counts["dropped_boxes"] == 1 and len(doc["annotations"]) == 1)
    ann = doc["annotations"][0]
    check("a box is written as left, top, width, height", ann["bbox"] == [10.0, 8.0, 30.0, 22.0] and ann["area"] == 660.0)
    check("categories carry the ids and names given", [(c["id"], c["name"]) for c in doc["categories"]] == [(1, "A"), (2, "B")])

    yolo = formats.write_yolo_dataset(work / "yolo", {"train": [{"source": src, "file_name": "a.jpg", "width": 64, "height": 48, "boxes": [{"class_index": 1, "xyxy": [16, 12, 48, 36]}]},
                                                                 {"source": src, "file_name": "bg.jpg", "width": 64, "height": 48, "boxes": []}],
                                                      "validation": [], "test": []}, ["A", "B"])
    label = (work / "yolo" / "labels" / "train" / "a.txt").read_text(encoding="utf-8").strip().split()
    check("a YOLO line is class then centre, width and height as fractions", label[0] == "1" and [round(float(v), 4) for v in label[1:]] == [0.5, 0.5, 0.5, 0.5])
    check("a background image gets an empty label file", (work / "yolo" / "labels" / "train" / "bg.txt").read_text(encoding="utf-8") == "")
    check("the YOLO layout writes a data file naming the classes", '0: "A"' in (work / "yolo" / "data.yaml").read_text(encoding="utf-8") and yolo["train"]["images"] == 2)
    check("the validation folder is called val in the YOLO layout", "val: images/val" in (work / "yolo" / "data.yaml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------


def test_rollup(work: Path) -> None:
    print("\n[5] Merging a rare class into its genus or family")
    backbone = make_backbone(work / "backbone")
    before = {p.name: sha(p) for p in backbone.iterdir()}
    cache = work / "cache" / "rollup.json"
    info = taxon_rollup.rollup_targets(backbone, ["gbif:100", "gbif:101", "name:poecile carolinensis", "gbif:20", "gbif:10", "name:dup name", "gbif:999", "name:unknown thing", "gbif:5"], cache_path=cache)

    import sqlite3
    with sqlite3.connect(backbone / "index.db") as probe:
        indexed = {r[0] for r in probe.execute("SELECT usage_key FROM taxon_index")}
    check("the test index holds species only, like the shipped one", "20" not in indexed and "10" not in indexed and "100" in indexed)
    check("a species finds its genus", taxon_rollup.parent_for(info["gbif:100"], "genus") == {"key": "20", "name": "Aplysina"})
    check("a species finds its family", taxon_rollup.parent_for(info["gbif:100"], "family") == {"key": "10", "name": "Aplysinidae"})
    check("a synonym resolves to the accepted taxon first", info["gbif:101"]["accepted_key"] == "100" and taxon_rollup.parent_for(info["gbif:101"], "genus")["key"] == "20")
    check("a class known only by name is found when the name is unambiguous", taxon_rollup.parent_for(info["name:poecile carolinensis"], "genus")["name"] == "Poecile")
    check("a genus rolled to genus stays itself", taxon_rollup.parent_for(info["gbif:20"], "genus") == {"key": "20", "name": "Aplysina"})
    check("a genus rolled to family finds its family", taxon_rollup.parent_for(info["gbif:20"], "family")["name"] == "Aplysinidae")
    check("a family rolled to family stays itself", taxon_rollup.parent_for(info["gbif:10"], "family") == {"key": "10", "name": "Aplysinidae"})
    check("a family has no genus to merge into", taxon_rollup.parent_for(info["gbif:10"], "genus") is None)
    check("a name shared by two taxa is not guessed", "reason" in info["name:dup name"] and taxon_rollup.parent_for(info["name:dup name"], "genus") is None)
    check("an unknown key says why", "reason" in info["gbif:999"] and taxon_rollup.parent_for(info["gbif:999"], "genus") is None)
    check("an unknown name says why", "reason" in info["name:unknown thing"])
    check("a taxon above family is never merged upward by guesswork", taxon_rollup.parent_for(info["gbif:5"], "family") is None and taxon_rollup.parent_for(info["gbif:5"], "genus") is None)
    check("an unknown level is refused", raises(lambda: taxon_rollup.parent_for(info["gbif:100"], "order"), ValueError))
    check("the backbone files are exactly as they were", {p.name: sha(p) for p in backbone.iterdir()} == before)
    check("the answers were cached for next time", cache.is_file())

    real_scan = taxon_rollup._scan
    taxon_rollup._scan = lambda *a, **k: (_ for _ in ()).throw(AssertionError("scanned again"))
    try:
        again = taxon_rollup.rollup_targets(backbone, ["gbif:100", "gbif:101"], cache_path=cache)
        check("a second lookup is answered from the cache without reading the backbone again", taxon_rollup.parent_for(again["gbif:100"], "genus")["key"] == "20")
    finally:
        taxon_rollup._scan = real_scan

    (backbone / "simple.txt").write_text((backbone / "simple.txt").read_text(encoding="utf-8") + "\n", encoding="utf-8")
    scanned = []
    taxon_rollup._scan = lambda *a, **k: scanned.append(1) or real_scan(*a, **k)
    try:
        taxon_rollup.rollup_targets(backbone, ["gbif:100"], cache_path=cache)
        check("a changed backbone invalidates the cache", len(scanned) >= 1)
    finally:
        taxon_rollup._scan = real_scan

    check("a missing index is reported in plain words", raises(lambda: taxon_rollup.rollup_targets(work / "nowhere", ["gbif:1"]), taxon_rollup.RollupError))
    (backbone / "simple.txt").unlink()
    check("a missing full backbone is reported in plain words", raises(lambda: taxon_rollup.rollup_targets(backbone, ["gbif:777"]), taxon_rollup.RollupError))


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        test_splits()
        test_frames(work)
        test_windows(work)
        test_formats(work)
        test_rollup(work)
    print(f"\nResult: {CHECKS['passed']} passed, {CHECKS['failed']} failed")
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
