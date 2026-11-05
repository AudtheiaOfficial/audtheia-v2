"""Verification for giving every label one taxon identity.

Path: tests/test_training_identity.py

Proves the resolution order (the key in the record, an exact scientific name, the
scientific part of a BirdNET label, then the expert label map), that synonyms are
followed to the accepted taxon, that an ambiguous or unknown name is never guessed,
that a missing index degrades to the keys and the map with a stated warning, that
the label map is written atomically with only valid entries, and that a dataset
merges one species named three ways into one class while listing what it could not
resolve. Runs against a small backbone in the shipped layout in a temporary folder.

Uses the standard library, NumPy and Pillow, plus the project itself.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from training_fixture import World, fake_rollup, make_backbone  # noqa: E402
from audtheia.training import dataset as ds  # noqa: E402
from audtheia.training import taxon_identity as ti  # noqa: E402

CHECKS = {"passed": 0, "failed": 0}


def check(label: str, condition: bool) -> None:
    CHECKS["passed" if condition else "failed"] += 1
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")


def raises(fn, exc=Exception, text=None) -> bool:
    try:
        fn()
    except exc as err:
        return text is None or text in str(err)
    return False


def resolver(root: Path, label_map=None) -> ti.Resolver:
    return ti.Resolver(make_backbone(root / "backbone"), label_map or {})


def test_order(root: Path) -> None:
    print("\n[1] The resolution order")
    with resolver(root) as r:
        got = r.resolve("gbif:200", "anything", "200")
        check("a key in the record wins and names the accepted taxon",
              got["class_key"] == "gbif:200" and got["class_name"] == "Poecile carolinensis" and got["method"] == "record_key")
        got = r.resolve("gbif:101", "Verongia fistularis", "101")
        check("a synonym key is followed to the accepted taxon",
              got["class_key"] == "gbif:100" and got["class_name"] == "Aplysina fistularis"
              and got["method"] == "record_key_synonym" and got["gbif_usage_key"] == "100")
        got = r.resolve("gbif:20", "Aplysina", "20")
        check("a key the index does not hold, such as a genus, is kept exactly as given",
              got["class_key"] == "gbif:20" and got["class_name"] == "Aplysina" and got["gbif_usage_key"] == "20")
        got = r.resolve("name:poecile carolinensis", "  poecile   CAROLINENSIS ", None)
        check("an exact scientific name resolves regardless of case and spacing",
              got["class_key"] == "gbif:200" and got["method"] == "backbone_name")
        got = r.resolve("name:verongia fistularis", "Verongia fistularis", None)
        check("a synonym name is followed to the accepted taxon",
              got["class_key"] == "gbif:100" and got["method"] == "backbone_name_synonym")
        got = r.resolve("name:x", "Poecile carolinensis_Carolina Chickadee", None)
        check("the scientific part of a BirdNET label resolves",
              got["class_key"] == "gbif:200" and got["method"] == "birdnet_label"
              and got["label"] == "Poecile carolinensis_Carolina Chickadee")
        got = r.resolve("name:x", "Poecile carolinensis_Carolina_Chickadee", None)
        check("a BirdNET label is split at its first underscore only",
              got["class_key"] == "gbif:200" and got["method"] == "birdnet_label")
        got = r.resolve("name:dup name", "Dup name", None)
        check("a name carried by two accepted taxa is never guessed",
              got["method"] == "unresolved" and "ambiguous" in got["reason"] and got["class_key"] == "name:dup name")
        got = r.resolve("name:northern-cardinal", "northern-cardinal", None)
        check("a model class name with no map entry stays its own class with a reason",
              got["method"] == "unresolved" and got["class_key"] == "name:northern-cardinal"
              and got["gbif_usage_key"] is None and "no scientific name" in got["reason"])
        got = r.resolve("name:human vocal_human vocal", "Human vocal_Human vocal", None)
        check("a BirdNET label whose scientific part is not a taxon stays unresolved, both reasons given",
              got["method"] == "unresolved" and "its scientific part" in got["reason"])
        got = r.resolve("name:poecile", "Poecile", None)
        check("a genus name is not matched, because the index holds species only",
              got["method"] == "unresolved")

    lm = {"northern-cardinal": {"action": "map", "scientific_name": "Poecile carolinensis", "gbif_usage_key": "200"},
          "aaptos sp.2": {"action": "keep"},
          "poecile carolinensis": {"action": "map", "scientific_name": "Aplysina lacunosa", "gbif_usage_key": "102"},
          "genus label": {"action": "map", "scientific_name": "Aplysina", "gbif_usage_key": "20"},
          "syn label": {"action": "map", "scientific_name": "Verongia fistularis", "gbif_usage_key": "101"}}
    with resolver(root / "m", lm) as r:
        got = r.resolve("name:northern-cardinal", "Northern-Cardinal", None)
        check("the label map resolves a model class name, matched without regard to case",
              got["class_key"] == "gbif:200" and got["method"] == "label_map" and got["class_name"] == "Poecile carolinensis")
        got = r.resolve("name:aaptos sp.2", "Aaptos sp.2", None)
        check("a label an expert chose to keep stays its own class and is not listed as unresolved",
              got["method"] == "kept_as_label" and got["class_key"] == "name:aaptos sp.2")
        got = r.resolve("name:poecile carolinensis", "Poecile carolinensis", None)
        check("an exact backbone name is settled before the map, so a map entry cannot override it",
              got["class_key"] == "gbif:200" and got["method"] == "backbone_name")
        got = r.resolve("gbif:200", "Poecile carolinensis", "200")
        check("a key in the record is settled before the map", got["method"] == "record_key")
        got = r.resolve("name:genus label", "genus label", None)
        check("a mapped key the index does not hold is kept as the expert gave it",
              got["class_key"] == "gbif:20" and got["class_name"] == "Aplysina")
        got = r.resolve("name:syn label", "syn label", None)
        check("a mapped synonym key is followed to the accepted taxon",
              got["class_key"] == "gbif:100" and got["class_name"] == "Aplysina fistularis")


def test_no_index(root: Path) -> None:
    print("\n[2] Without the taxonomic index")
    lm = {"northern-cardinal": {"action": "map", "scientific_name": "Poecile carolinensis", "gbif_usage_key": "200"}}
    with ti.Resolver(root / "nowhere", lm) as r:
        check("the index is reported missing", r.index_available is False)
        got = r.resolve("gbif:200", "Poecile carolinensis", "200")
        check("a key in the record is still used", got["class_key"] == "gbif:200" and got["class_name"] == "Poecile carolinensis")
        got = r.resolve("name:poecile carolinensis", "Poecile carolinensis", None)
        check("a bare name is left unresolved and the reason says the index is missing",
              got["method"] == "unresolved" and "not available" in got["reason"])
        got = r.resolve("name:northern-cardinal", "northern-cardinal", None)
        check("the label map still applies", got["class_key"] == "gbif:200" and got["method"] == "label_map")
    bad = root / "badindex"
    bad.mkdir(parents=True)
    (bad / ti.INDEX_FILE).write_bytes(b"not a database")
    with ti.Resolver(bad) as r:
        check("an unreadable index is treated as missing, not as an error", r.index_available is False)


def test_classes(root: Path) -> None:
    print("\n[3] Resolving a set of classes")
    lm = {"northern-cardinal": {"action": "map", "scientific_name": "Poecile carolinensis", "gbif_usage_key": "200"}}
    rows = [
        {"class_key": "name:northern-cardinal", "class_name": "northern-cardinal", "gbif_usage_key": None},
        {"class_key": "gbif:200", "class_name": "Poecile carolinensis", "gbif_usage_key": "200"},
        {"class_key": "name:poecile carolinensis_carolina chickadee", "class_name": "Poecile carolinensis_Carolina Chickadee",
         "gbif_usage_key": None},
        {"class_key": "name:mystery", "class_name": "mystery", "gbif_usage_key": None},
        {"class_key": "gbif:102", "class_name": "Aplysina lacunosa", "gbif_usage_key": "102"},
        {"class_key": "background", "class_name": "background", "gbif_usage_key": None},
    ]
    with resolver(root, lm) as r:
        out = ti.resolve_classes(rows, r)
    rep = out["report"]
    check("one species named three ways becomes one class", len(rep["merged"]) == 1
          and rep["merged"][0]["class_key"] == "gbif:200" and len(rep["merged"][0]["labels"]) == 3)
    check("the mapping lists only classes whose identity changes",
          set(out["mapping"]) == {"name:northern-cardinal", "name:poecile carolinensis_carolina chickadee"})
    check("every changed class points at the accepted key and name",
          all(v == {"class_key": "gbif:200", "class_name": "Poecile carolinensis", "gbif_usage_key": "200"}
              for v in out["mapping"].values()))
    check("the unresolved label is listed with its reason",
          [u["label"] for u in rep["unresolved"]] == ["mystery"] and rep["unresolved"][0]["reason"])
    check("background is never resolved as a taxon", all(c["class_key"] != "background" for c in rep["classes"]))
    check("the report carries a plain description of every method", set(rep["methods"]) == set(ti.METHODS))
    check("a class that needs no change is not in the mapping", "gbif:102" not in out["mapping"])
    with resolver(root / "r2") as r:
        out = ti.resolve_classes([{"class_key": "gbif:101", "class_name": "Verongia fistularis", "gbif_usage_key": "101"}], r)
    check("a synonym key alone is still remapped to the accepted key",
          out["mapping"] == {"gbif:101": {"class_key": "gbif:100", "class_name": "Aplysina fistularis", "gbif_usage_key": "100"}})


def test_map_file(root: Path) -> None:
    print("\n[4] The label map file")
    path = root / "maps" / ti.LABEL_MAP_FILE
    check("a missing map reads as empty", ti.load_label_map(path) == {})
    entry = ti.save_label_entry(path, "  Northern-Cardinal ", action="map", scientific_name="Cardinalis cardinalis",
                                gbif_usage_key="2490384", corrector="tester")
    check("a mapping is written with the person's verdict and data_source",
          entry["data_source"] == "human_expert" and entry["corrector"] == "tester" and entry["label"] == "Northern-Cardinal")
    loaded = ti.load_label_map(path)
    check("the map is read back keyed by the reduced label",
          set(loaded) == {"northern-cardinal"} and loaded["northern-cardinal"]["gbif_usage_key"] == "2490384")
    ti.save_label_entry(path, "Aaptos sp.2", action="keep")
    check("a keep decision is stored beside it", ti.load_label_map(path)["aaptos sp.2"]["action"] == "keep")
    ti.save_label_entry(path, "northern-cardinal", action="map", scientific_name="Cardinalis cardinalis",
                        gbif_usage_key="2490384")
    check("deciding a label again replaces the old entry, never duplicates it", len(ti.load_label_map(path)) == 2)
    check("a mapping without a key is refused",
          raises(lambda: ti.save_label_entry(path, "x", action="map", scientific_name="Y y"), ti.IdentityError, "both"))
    check("a key that is not a number is refused",
          raises(lambda: ti.save_label_entry(path, "x", action="map", scientific_name="Y y", gbif_usage_key="12a"),
                 ti.IdentityError, "number"))
    check("an unknown action is refused", raises(lambda: ti.save_label_entry(path, "x", action="merge"), ti.IdentityError))
    check("an empty label is refused", raises(lambda: ti.save_label_entry(path, "   ", action="keep"), ti.IdentityError))
    check("a refused entry leaves the file untouched", len(ti.load_label_map(path)) == 2)
    check("no temporary file is left behind", sorted(p.name for p in path.parent.iterdir()) == [ti.LABEL_MAP_FILE])
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["entries"]["broken"] = {"action": "map"}
    raw["entries"]["odd"] = "text"
    path.write_text(json.dumps(raw), encoding="utf-8")
    check("malformed entries are ignored when read", set(ti.load_label_map(path)) == {"northern-cardinal", "aaptos sp.2"})
    check("removing a decision works and reports it", ti.remove_label_entry(path, "AAPTOS SP.2") is True
          and "aaptos sp.2" not in ti.load_label_map(path))
    check("removing an absent decision reports nothing removed", ti.remove_label_entry(path, "nothing") is False)
    path.write_text("{not json", encoding="utf-8")
    check("an unreadable map reads as empty rather than failing a build", ti.load_label_map(path) == {})


def _world(root: Path) -> World:
    w = World(root)
    make_backbone(Path(w.settings.path("gbif_backbone_path")))
    return w


def test_dataset(root: Path) -> None:
    print("\n[5] A dataset merges one species named three ways")
    w = _world(root / "v")
    for i in range(3):
        ev = w.vision_event([("northern-cardinal", None)], station=f"S{i}", first_seen=f"2026-03-0{i + 1}T10:00:00Z")
        w.confirm(ev)
    for i in range(3):
        ev = w.vision_event([("Poecile carolinensis", None)], station=f"S{i}", first_seen=f"2026-04-0{i + 1}T10:00:00Z")
        w.confirm(ev)
    for i in range(3):
        ev = w.vision_event([("chickadee-slug", None)], station=f"S{i}", first_seen=f"2026-05-0{i + 1}T10:00:00Z")
        w.relabel(ev, "Poecile carolinensis", gbif=200)
    for i in range(2):
        ev = w.vision_event([("mystery-bird", None)], station=f"S{i}", first_seen=f"2026-06-0{i + 1}T10:00:00Z")
        w.confirm(ev)
    req = {"modality": "vision", "required_criteria": ["expert_confirmed"], "min_examples_per_class": 1}

    def plan():
        return ds.plan_dataset(w.db, w.settings, ds.options_from_request(w.settings, req), rollup_lookup=fake_rollup({}))

    first = ds.summarize_plan(plan())
    keys = sorted(c["class_key"] for c in first["classes"])
    check("the name and the expert relabel are one class before any map entry",
          "gbif:200" in keys and "name:poecile carolinensis" not in keys)
    check("the model class names stay separate and are listed as unresolved",
          sorted(u["label"] for u in first["identity"]["unresolved"]) == ["mystery-bird", "northern-cardinal"])
    check("the preview warns about the unresolved labels",
          any("could not be matched to the backbone" in x and "northern-cardinal" in x for x in first["warnings"]))
    check("the preview names the merge it made",
          any("same taxon and were merged" in x for x in first["warnings"]))
    ti.save_label_entry(ds.label_map_path(w.settings), "northern-cardinal", action="map",
                        scientific_name="Poecile carolinensis", gbif_usage_key="200")
    ti.save_label_entry(ds.label_map_path(w.settings), "mystery-bird", action="keep")
    second = ds.summarize_plan(plan())
    cls = {c["class_key"]: c for c in second["classes"]}
    check("after the expert maps the label, all three names are one class",
          sorted(cls) == ["gbif:200", "name:mystery-bird"])
    before = {c["class_key"]: c for c in second["classes_before"]}
    check("the merged class counts every event once", before["gbif:200"]["events"] == 9)
    check("the merged class carries the accepted name and key",
          cls["gbif:200"]["class_name"] == "Poecile carolinensis" and cls["gbif:200"]["gbif_usage_key"] == "200")
    check("nothing is left unresolved once every label is decided", second["identity"]["unresolved"] == []
          and [k["label"] for k in second["identity"]["kept_as_label"]] == ["mystery-bird"])
    check("an expert's decision changes the dataset's identity, so the old dataset is never reused",
          first["signature"] != second["signature"])
    data_before = sorted(p.as_posix() for p in Path(w.settings.path("data_dir")).rglob("*"))
    built = ds.build_dataset(w.db, w.settings, ds.options_from_request(w.settings, req), rollup_lookup=fake_rollup({}))
    manifest = json.loads((Path(built["path"]) / "manifest.json").read_text(encoding="utf-8"))
    check("the manifest records how each class was identified",
          [c["class_key"] for c in manifest["identity"]["merged"]] == ["gbif:200"])
    classes = json.loads((Path(built["path"]) / "classes.json").read_text(encoding="utf-8"))
    check("the written class list has one entry per taxon", sorted(c["class_key"] for c in classes) == ["gbif:200", "name:mystery-bird"])
    check("the data folder is untouched", data_before == sorted(p.as_posix() for p in Path(w.settings.path("data_dir")).rglob("*")))
    check("the label map lives beside the datasets, outside the data folder",
          Path(w.settings.path("data_dir")).resolve() not in ds.label_map_path(w.settings).resolve().parents)

    print("\n[6] An acoustic dataset resolves BirdNET labels")
    a = _world(root / "a")
    for i in range(3):
        ev = a.audio_event([("Poecile carolinensis_Carolina Chickadee", None)], station=f"S{i}",
                           first_seen=f"2026-03-0{i + 1}T06:00:00Z")
        a.confirm(ev)
    ev = a.audio_event([("Poecile carolinensis_Carolina Chickadee", None), ("Poecile carolinensis", 200)],
                       station="S0", first_seen="2026-03-09T06:00:00Z")
    a.confirm(ev, detection=ev["children"][0], modality="audio")
    a.confirm(ev, detection=ev["children"][1], modality="audio")
    req = {"modality": "audio", "engine": "acoustic_head_a", "required_criteria": ["expert_confirmed"],
           "min_examples_per_class": 1}
    summary = ds.summarize_plan(ds.plan_dataset(a.db, a.settings, ds.options_from_request(a.settings, req),
                                                rollup_lookup=fake_rollup({})))
    check("BirdNET's joined label becomes the accepted scientific name and key",
          [(c["class_key"], c["class_name"]) for c in summary["classes"]] == [("gbif:200", "Poecile carolinensis")])
    check("two labels for one species in one window count as one label, not two",
          all(len(set(t)) == len(t) for t in [[l["class_key"] for l in i.get("labels", [])]
                                               for i in ds.plan_dataset(a.db, a.settings, ds.options_from_request(a.settings, req),
                                                                         rollup_lookup=fake_rollup({}))["items"]["train"]]))
    check("the event with both labels is counted once", {c["class_key"]: c for c in summary["classes_before"]}["gbif:200"]["events"] == 4)

    print("\n[7] Without the index the build still works and says why names were not resolved")
    n = World(root / "n")
    ev = n.vision_event([("Poecile carolinensis", None)])
    n.confirm(ev)
    s = ds.summarize_plan(ds.plan_dataset(n.db, n.settings, ds.options_from_request(n.settings, {
        "modality": "vision", "required_criteria": ["expert_confirmed"], "min_examples_per_class": 1}),
        rollup_lookup=fake_rollup({})))
    check("a warning says the taxonomic index was not found", any("taxonomic index was not found" in x for x in s["warnings"]))
    check("the report says the index is unavailable", s["identity"]["index_available"] is False)


def test_endpoints(root: Path) -> None:
    print("\n[9] The endpoints")
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        print("  SKIP  the web framework is not installed in this interpreter")
        return
    from audtheia.app import server as srv
    w = _world(root)
    for i in range(2):
        ev = w.vision_event([("northern-cardinal", None)], station=f"S{i}", first_seen=f"2026-03-0{i + 1}T10:00:00Z")
        w.confirm(ev)
    ev = w.vision_event([("Poecile carolinensis", 200)], first_seen="2026-03-05T10:00:00Z")
    w.confirm(ev)
    ev = w.vision_event([("northern-cardinal", None), ("Poecile carolinensis", 200)], first_seen="2026-03-06T10:00:00Z")
    w.confirm(ev, detection=ev["children"][0], modality="vision")
    w.confirm(ev, detection=ev["children"][1], modality="vision")
    client = TestClient(srv.create_app(w.settings, w.db))
    summary = client.get("/api/train/candidates/summary", params={"modality": "vision"}).json()
    check("the counts before building already use one identity per species",
          set(summary["by_class"]) == {"name:northern-cardinal", "gbif:200"})
    post = lambda body: client.post("/api/train/identity/labels", json=body)  # noqa: E731
    check("an unknown backbone key is refused with a plain reason",
          post({"label": "northern-cardinal", "action": "map", "gbif_usage_key": "999999"}).status_code == 422)
    check("a mapping without a chosen taxon is refused",
          post({"label": "northern-cardinal", "action": "map"}).status_code == 422)
    check("an unknown action is refused", post({"label": "northern-cardinal", "action": "merge"}).status_code == 422)
    check("an empty label is refused", post({"label": " ", "action": "keep"}).status_code == 422)
    check("no map file is written by a refused request", not ds.label_map_path(w.settings).exists())
    r = post({"label": "northern-cardinal", "action": "map", "gbif_usage_key": "101", "corrector": "Reviewer"})
    entry = r.json().get("entry") or {}
    check("a synonym key is stored as the accepted taxon",
          r.status_code == 200 and entry.get("gbif_usage_key") == "100" and entry.get("scientific_name") == "Aplysina fistularis")
    r = post({"label": "northern-cardinal", "action": "map", "gbif_usage_key": "200"})
    check("a valid mapping is stored with the accepted name",
          r.status_code == 200 and r.json()["entry"]["scientific_name"] == "Poecile carolinensis"
          and r.json()["entry"]["corrector"] == "expert")
    summary = client.get("/api/train/candidates/summary", params={"modality": "vision"}).json()
    check("after mapping, the counts show one class with every event",
          list(summary["by_class"]) == ["gbif:200"] and summary["by_class"]["gbif:200"]["events"] == 4)
    preview = client.post("/api/train/dataset/preview", json={"modality": "vision", "required_criteria": ["expert_confirmed"],
                                                               "min_examples_per_class": 1}).json()
    check("the preview shows the mapping in its identity report",
          [c["class_key"] for c in preview["identity"]["merged"]] == ["gbif:200"] and preview["identity"]["unresolved"] == [])
    r = post({"label": "northern-cardinal", "action": "clear"})
    check("clearing a decision removes it", r.status_code == 200 and r.json()["removed"] is True)
    check("clearing it again reports nothing removed", post({"label": "northern-cardinal", "action": "clear"}).json()["removed"] is False)
    check("keeping a label is stored", post({"label": "northern-cardinal", "action": "keep"}).json()["entry"]["action"] == "keep")
    w.settings.raw["node"]["role"] = "pi"
    r = client.post("/api/train/identity/labels", json={"label": "x", "action": "keep"})
    check("a field station refuses to record label decisions", r.status_code == 403 and "desktop" in r.json()["detail"])
    w.settings.raw["node"]["role"] = "desktop"


def test_text() -> None:
    print("\n[8] The module text")
    text = (REPO / "audtheia" / "training" / "taxon_identity.py").read_text(encoding="utf-8")
    check("taxon_identity.py is ASCII only", text.isascii())
    check("it imports nothing beyond the standard library",
          all(m not in text for m in ("import numpy", "import torch", "import tensorflow", "from audtheia")))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="audtheia-identity-") as tmp:
        root = Path(tmp)
        test_order(root / "o")
        test_no_index(root / "n")
        test_classes(root / "c")
        test_map_file(root / "f")
        test_dataset(root / "d")
        test_endpoints(root / "e")
    test_text()
    print(f"\nResult: {CHECKS['passed']} passed, {CHECKS['failed']} failed")
    return 0 if CHECKS["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
