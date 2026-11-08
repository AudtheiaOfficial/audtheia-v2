"""Verification for the model registry, switching models on and off, and sharing.

Path: tests/test_training_models.py

Drives the real endpoints against a record in a temporary repository whose
settings sit at config/settings.json, so the local settings file is used exactly
as on a real installation. Proves that a finished run becomes the next numbered
version of its name with its files, metrics and model card, that a folder is
never overwritten and a run is never added twice, that the licence is inherited
and a limiting licence needs the typed acknowledgement, that switching a model on
writes only the local settings file and leaves the committed one byte for byte
the same, also across a later settings save, that a revert restores the slot
exactly and is refused when the slot was changed elsewhere, that packages hold
the listed files and are never overwritten, and that the Model filter and the
card versions read the stored provenance.

Uses the standard library, NumPy and Pillow, plus the project itself.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from training_fixture import World, fake_rollup  # noqa: E402
from audtheia.config import load_settings  # noqa: E402
from audtheia.storage.database import TrainingRun, new_id, utc_now_iso  # noqa: E402
from audtheia.training import dataset as ds  # noqa: E402
from audtheia.training import deploy, registry  # noqa: E402
from audtheia.training import runs as R  # noqa: E402

CHECKS = {"passed": 0, "failed": 0}
ACK = deploy.ACKNOWLEDGEMENT


def check(label: str, condition: bool) -> None:
    CHECKS["passed" if condition else "failed"] += 1
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(w: World, p: Path) -> str:
    return Path(p).resolve().relative_to(w.repo.resolve()).as_posix()


def make_run(w: World, name: str, modality: str, dataset_dir: Path, *, status="completed", config=None,
             files=True) -> str:
    manager = R.RunManager(w.settings, w.db)
    rid = new_id()
    run_dir = manager.run_dir(rid)
    run_dir.mkdir(parents=True)
    if files and modality == "vision":
        (run_dir / "export").mkdir()
        (run_dir / "export" / "model.onnx").write_bytes(b"onnx-" + name.encode())
        (run_dir / "export" / "labels.json").write_text('{"0": {"name": "Aplysina fistularis"}}', encoding="utf-8")
        (run_dir / "train_output").mkdir()
        (run_dir / "train_output" / "checkpoint_best_total.pth").write_bytes(b"pth")
    if files and modality == "acoustic":
        (run_dir / "export").mkdir()
        (run_dir / "export" / "model.tflite").write_bytes(b"tflite")
        (run_dir / "export" / "labels.txt").write_text("Poecile carolinensis\n", encoding="utf-8")
        (run_dir / "export" / "head.npz").write_bytes(b"npz")
        (run_dir / "export" / "calibration.json").write_text("{}", encoding="utf-8")
    cfg = config or ({"base_label": "Nano", "start_from": {"kind": "pretrained"}} if modality == "vision" else
                     {"base_label": "BirdNET 2.4", "sample_rate": 48000, "window_seconds": 3.0,
                      "base": {"label": "BirdNET 2.4", "licence": "CC BY-NC-SA 4.0"}})
    w.db.create_training_run(TrainingRun(
        id=rid, run_name=name, planned_model_name=f"Audtheia-{name}-V1", modality=modality,
        engine="vision_detector" if modality == "vision" else "acoustic_head", mode="gentle", status=status,
        dataset_signature="s", dataset_dir=rel(w, dataset_dir), run_dir=rel(w, run_dir), seed=1,
        base_label=cfg["base_label"], config_json=json.dumps(cfg), created_by="tester", created_at=utc_now_iso()))
    return rid


def main() -> int:
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        print("  SKIP  the web framework is not installed in this interpreter")
        return 0
    from audtheia.app import server as srv

    with tempfile.TemporaryDirectory(prefix="audtheia-models-") as tmp:
        w = World(Path(tmp))
        for i in range(2):
            e = w.vision_event([("Aplysina fistularis", 100)], first_seen=f"2026-03-0{i + 1}T10:00:00Z")
            w.confirm(e)
        built = ds.build_dataset(w.db, w.settings, ds.options_from_request(w.settings, {
            "modality": "vision", "required_criteria": ["expert_confirmed"], "min_examples_per_class": 1}),
            rollup_lookup=fake_rollup({}))
        d = Path(built["path"])
        settings_file = w.settings_path
        w.settings = load_settings(settings_file)
        client = TestClient(srv.create_app(w.settings, w.db))
        tracked_before = sha(settings_file)
        local = settings_file.parent / "settings.local.json"

        print("\n[1] A finished run becomes a numbered model")
        failed = make_run(w, "Reef", "vision", d, status="failed")
        check("a run that did not finish is refused",
              client.post(f"/api/train/runs/{failed}/register", json={}).status_code == 409)
        r1 = make_run(w, "Reef", "vision", d)
        res = client.post(f"/api/train/runs/{r1}/register", json={})
        m1 = res.json().get("model") or {}
        check("it is issued as the first version of its name", res.status_code == 201 and m1.get("name") == "Audtheia-Reef-V1")
        folder = w.repo / m1["folder"]
        check("its folder holds the model, labels, checkpoint, metrics and card",
              sorted(p.name for p in folder.iterdir()) == ["MODEL_CARD.md", "checkpoint.pth", "labels.json", "metrics.json",
                                                         "model.onnx"])
        card = (folder / "MODEL_CARD.md").read_text(encoding="utf-8")
        check("the model card carries the inherited licence in its front matter",
              card.startswith("---\nlicense: apache-2.0") and "Audtheia-Reef-V1" in card)
        check("a model with no test score says so", m1["metrics"]["tested"] is False and "no test score" in m1["metrics"]["note"])
        check("the licence comes from the pretrained weights", m1["license_spdx"] == "Apache-2.0")
        check("the same run is never added twice", client.post(f"/api/train/runs/{r1}/register", json={}).status_code == 409)
        r2 = make_run(w, "reef", "vision", d)
        m2 = client.post(f"/api/train/runs/{r2}/register", json={}).json()["model"]
        check("a second run of the same name, in any letter case, is the next version", m2["name"] == "Audtheia-reef-V2")
        r3 = make_run(w, "Reef", "vision", d)
        (Path(w.settings.training_paths()["models_dir"]) / "Audtheia-Reef-V3").mkdir(parents=True)
        res = client.post(f"/api/train/runs/{r3}/register", json={})
        check("an existing folder is never overwritten", res.status_code == 409 and "never overwritten" in res.json()["detail"])
        check("and no number is spent on it", len(w.db.list_trained_models()) == 2)
        scored = make_run(w, "Scored", "vision", d)
        ev_dir = w.repo / w.db.get_training_run(scored)["run_dir"] / "evaluation"
        ev_dir.mkdir(parents=True)
        private = str(w.repo / "Private Folder" / "models" / "visual" / "in_use.onnx")
        challenger = {"kind": "detection", "map50": 0.5, "map50_95": 0.3, "counts": {}, "class_names": ["A"],
                      "top1_accuracy_on_matched_boxes": {"value": 0.9},
                      "at_confidence": {"confidence": 0.3, "macro_precision": 0.4, "macro_recall": 0.5, "macro_f1": 0.45}}
        (ev_dir / "evaluation.json").write_text(json.dumps({
            "challenger": challenger, "comparison": {"rows": []}, "box_origin": None,
            "current_model": {"label": "Reef screening model", "path": private, "version": "V12"}}), encoding="utf-8")
        ms = client.post(f"/api/train/runs/{scored}/register", json={}).json()["model"]
        kept = (w.repo / ms["folder"] / "metrics.json").read_text(encoding="utf-8")
        check("the model it was compared with is named by its file, never by this computer's path",
              "Private Folder" not in kept and json.loads(kept)["current_model"] ==
              {"label": "Reef screening model", "version": "V12", "file": "in_use.onnx"})
        card = (w.repo / ms["folder"] / "MODEL_CARD.md").read_text(encoding="utf-8")
        check("the card reads correctly and carries no path of this computer",
              "\nAn image detection model trained with Audtheia" in card and "Private Folder" not in card)
        bare = make_run(w, "Bare", "vision", d, files=False)
        check("a run whose model file is missing is refused",
              client.post(f"/api/train/runs/{bare}/register", json={}).status_code == 422)
        own = make_run(w, "Own", "vision", d, config={"base_label": "Nano", "start_from": {"kind": "checkpoint", "path": "x/bird_weights.pt"}})
        m_own = client.post(f"/api/train/runs/{own}/register", json={}).json()["model"]
        check("a model trained from a person's own checkpoint has its licence recorded as not confirmed",
              m_own["license_spdx"] is None and "bird_weights.pt" in m_own["license_note"])
        snd = make_run(w, "Dawn", "acoustic", d)
        m_snd = client.post(f"/api/train/runs/{snd}/register", json={}).json()["model"]
        check("a sound model built on BirdNET inherits its licence and needs the acknowledgement",
              m_snd["license_spdx"] == "CC-BY-NC-SA-4.0" and m_snd["needs_acknowledgement"])
        listing = client.get("/api/train/models").json()
        check("the registry lists the models and the runs not yet added",
              len(listing["models"]) == 5 and {r["id"] for r in listing["unregistered_runs"]} >= {r3, bare})
        check("adding models left the committed settings unchanged", sha(settings_file) == tracked_before)

        print("\n[2] Switching a model on writes only the local settings file")
        act = lambda mid, body: client.post(f"/api/train/models/{mid}/activate", json=body)  # noqa: E731
        res = act(m1["id"], {"slot": "desktop_verifier"})
        check("the model is switched on as the desktop verifier", res.status_code == 200)
        check("the committed settings file is byte for byte the same", sha(settings_file) == tracked_before)
        overrides = json.loads(local.read_text(encoding="utf-8"))["overrides"]
        check("the local file holds the slot's path and version",
              overrides.get("desktop_models.visual_rfdetr.version") == "Audtheia-Reef-V1"
              and overrides.get("desktop_models.visual_rfdetr.path", "").endswith("Audtheia-Reef-V1/model.onnx"))
        fresh = load_settings(settings_file)
        check("a fresh load of the settings uses the model",
              fresh.raw["desktop_models"]["visual_rfdetr"]["version"] == "Audtheia-Reef-V1")
        row = res.json()["activation"]
        check("the activation is logged with the slot before and after, as the expert's action",
              row["data_source"] == "human_expert" and json.loads(row["new_json"])["version"] == "Audtheia-Reef-V1"
              and "version" in json.loads(row["previous_json"]) or row["previous_json"] == "{}")
        sid = w.settings.stations()[0]["station_id"]
        check("an image model cannot go into a sound slot",
              act(m1["id"], {"slot": "station_acoustic", "station_id": sid}).status_code == 422)
        check("a station slot needs a station", act(m1["id"], {"slot": "desktop_screening"}).status_code == 422)
        check("an unknown station is refused", act(m1["id"], {"slot": "desktop_screening", "station_id": "nope"}).status_code == 404)
        check("an unknown slot is refused", act(m1["id"], {"slot": "everywhere"}).status_code == 422)
        res = act(m_snd["id"], {"slot": "station_acoustic", "station_id": sid})
        check("a sound model with a limiting licence needs the typed acknowledgement",
              res.status_code == 428 and ACK in res.json()["detail"])
        check("a sentence that is not the acknowledgement is refused",
              act(m_snd["id"], {"slot": "station_acoustic", "station_id": sid,
                                "acknowledgement": "I accept the terms"}).status_code == 428)
        check("the acknowledgement is read whatever the case, spacing or full stop, and in American spelling",
              all(deploy.acknowledged(t) for t in (ACK, "I accept the license terms", "  i ACCEPT the License terms. "))
              and not deploy.acknowledged("I do not accept the licence terms"))
        res = act(m_snd["id"], {"slot": "station_acoustic", "station_id": sid,
                                "acknowledgement": "I accept the license terms"})
        entry = load_settings(settings_file).station(sid)["models"]["acoustic"]
        check("the running application holds the new model at once, so the next capture start uses it",
              w.settings.station(sid)["models"]["acoustic"]["version"] == "Audtheia-Dawn-V1"
              and "next time that station's capture is started" in res.json()["note"])
        check("with it, in either spelling, the station's sound model is set with its labels and audio shape",
              res.status_code == 200 and entry["version"] == "Audtheia-Dawn-V1" and entry["labels_path"].endswith("labels.txt")
              and entry["sample_rate"] == 48000 and entry["window_seconds"] == 3.0)
        check("and the acknowledgement is recorded", res.json()["activation"]["acknowledged_license"] == 1)
        check("the committed settings file is still the same", sha(settings_file) == tracked_before)
        theme = client.post("/api/settings/update", json={"changes": [{"scope": "global", "field": "ui_theme", "value": "forest"}]})
        check("a later settings save keeps the switched-on models in the local file",
              theme.status_code == 200
              and load_settings(settings_file).raw["desktop_models"]["visual_rfdetr"]["version"] == "Audtheia-Reef-V1")
        tracked = json.loads(settings_file.read_text(encoding="utf-8"))
        check("and still writes no model version into the committed file",
              tracked["desktop_models"]["visual_rfdetr"].get("version") != "Audtheia-Reef-V1"
              and tracked["ui"]["theme"] == "forest")

        print("\n[3] Putting the previous model back")
        before = json.loads(row["previous_json"])
        res = client.post("/api/train/slots/revert", json={"slot": "desktop_verifier"})
        now = load_settings(settings_file).raw["desktop_models"]["visual_rfdetr"]
        check("the slot is restored exactly", res.status_code == 200 and now == before)
        check("a second revert is refused", client.post("/api/train/slots/revert", json={"slot": "desktop_verifier"}).status_code == 409)
        slot = [s for s in client.get("/api/train/models").json()["slots"] if s["slot"] == "desktop_verifier"][0]
        check("the slot is no longer offered for revert", slot["revertible"] is False and len(slot["history"]) == 2)
        act(m2["id"], {"slot": "desktop_verifier"})
        client.post("/api/settings/update", json={"changes": [{"scope": "global", "field": "visual_rfdetr_version", "value": "hand edit"}]})
        res = client.post("/api/train/slots/revert", json={"slot": "desktop_verifier"})
        check("a slot changed elsewhere is not reverted blindly", res.status_code == 409 and "changed elsewhere" in res.json()["detail"])
        check("nothing switched on here means nothing to revert",
              client.post("/api/train/slots/revert", json={"slot": "desktop_screening", "station_id": sid}).status_code == 409)
        original = json.loads(json.dumps(w.settings.station(sid)["models"]["visual_desktop"]))
        screening = {"slot": "desktop_screening", "station_id": sid}
        act(m1["id"], screening)
        act(m2["id"], screening)
        first = client.post("/api/train/slots/revert", json=screening)
        check("one press undoes the latest change only",
              first.status_code == 200 and w.settings.station(sid)["models"]["visual_desktop"]["version"] == "Audtheia-Reef-V1")
        view = [s_ for s_ in client.get("/api/train/models").json()["slots"]
                if s_["slot"] == "desktop_screening" and s_["station_id"] == sid][0]
        check("after it, the earlier change is still offered to put back", view["revertible"] is True)
        second = client.post("/api/train/slots/revert", json=screening)
        check("a second press returns the slot to how it was before either change",
              second.status_code == 200 and w.settings.station(sid)["models"]["visual_desktop"] == original
              and load_settings(settings_file).station(sid)["models"]["visual_desktop"] == original)
        check("with every change undone, a further press is refused",
              client.post("/api/train/slots/revert", json=screening).status_code == 409)
        check("the undo history pairs each revert with the change it undid",
              deploy.undoable([{"action": "revert"}, {"action": "activate", "id": 2}, {"action": "activate", "id": 1}])["id"] == 1
              and deploy.undoable([{"action": "activate", "id": 3}, {"action": "revert"}, {"action": "activate", "id": 1}])["id"] == 3
              and deploy.undoable([{"action": "revert"}, {"action": "revert"}, {"action": "activate"}, {"action": "activate"}]) is None)

        print("\n[4] Packages")
        res = client.post(f"/api/train/models/{m1['id']}/package", json={})
        pkg = res.json()
        names = zipfile.ZipFile(pkg["path"]).namelist()
        check("a model package holds the model folder", res.status_code == 201 and
              sorted(names) == sorted(f"Audtheia-Reef-V1/{n}" for n in ("MODEL_CARD.md", "checkpoint.pth", "labels.json",
                                                                       "metrics.json", "model.onnx")))
        again = client.post(f"/api/train/models/{m1['id']}/package", json={}).json()
        check("a second package never overwrites the first", again["path"] != pkg["path"] and Path(pkg["path"]).is_file())
        check("a limiting licence needs the acknowledgement to package",
              client.post(f"/api/train/models/{m_snd['id']}/package", json={}).status_code == 428
              and client.post(f"/api/train/models/{m_snd['id']}/package",
                              json={"acknowledgement": "I accept the license terms"}).status_code == 201)
        dl = client.get(f"/api/train/packages/{pkg['name']}")
        check("a package can be downloaded", dl.status_code == 200 and dl.content[:2] == b"PK")
        check("a name outside the packages folder is refused", client.get("/api/train/packages/..%2Fsettings.json").status_code == 404)
        exp = client.post(f"/api/train/dataset/{d.name}/export/coco").json()
        enames = zipfile.ZipFile(exp["path"]).namelist()
        check("a COCO export holds the splits, the class list and the manifest",
              f"{d.name}/manifest.json" in enames and f"{d.name}/classes.json" in enames
              and any(n.endswith("_annotations.coco.json") for n in enames))
        check("an image dataset has no sound layout", client.post(f"/api/train/dataset/{d.name}/export/windows").status_code == 422)
        check("an unknown dataset is refused", client.post("/api/train/dataset/vision-0000000000000000/export/coco").status_code == 404)
        check("a malformed dataset name is refused", client.post("/api/train/dataset/vision-zz/export/coco").status_code == 404)
        check("packages sit outside the data folder",
              Path(w.settings.path("data_dir")).resolve() not in Path(pkg["path"]).resolve().parents)

        print("\n[5] The Model filter and card versions")
        with w.db.connect() as conn:
            ids = [r[0] for r in conn.execute("SELECT id FROM observations WHERE trigger_source = 'vision' ORDER BY id")]
            conn.execute("UPDATE observations SET screening_model_version = 'Audtheia-Reef-V1' WHERE id = ?", (ids[0],))
        models = client.get("/api/detections/models").json()
        check("the filter lists the stored versions and the events without one",
              {m["value"] for m in models} == {"Audtheia-Reef-V1", "__not_recorded__"})
        one = client.get("/api/detections", params={"model": "Audtheia-Reef-V1"}).json()
        check("filtering by a version returns only its events", one["total"] == 1 and one["items"][0]["id"] == ids[0])
        none = client.get("/api/detections", params={"model": "__not_recorded__"}).json()
        check("events without a version stay reachable", none["total"] == len(ids) - 1)
        check("without a filter every event is listed", client.get("/api/detections").json()["total"] == len(ids))
        check("the sound list has its own filter", client.get("/api/audio/models").status_code == 200)
        check("adding and switching models never changed a stored version",
              client.get("/api/detections", params={"model": "Audtheia-Reef-V2"}).json()["total"] == 0)
        check("the committed settings differ only by the theme saved through the settings page",
              {k for k in tracked if tracked[k] != json.loads(settings_file.read_text(encoding="utf-8"))[k]} == set())

        print("\n[6] Removing the files of a run that did not become a model")
        spare = make_run(w, "Spare", "vision", d)
        spare_dir = w.repo / w.db.get_training_run(spare)["run_dir"]
        kept_files = {"run.json": "{}", "status.json": "{}", "log.txt": "Starting the run.\n",
                      "progress.jsonl": '{"epoch": 1}\n', "train_output/metrics.csv": "epoch,loss\n1,2\n",
                      "evaluation/evaluation.json": "{}", "evaluation/png/loss.png": "png"}
        for name, text in kept_files.items():
            (spare_dir / name).parent.mkdir(parents=True, exist_ok=True)
            (spare_dir / name).write_text(text, encoding="utf-8")
        (spare_dir / "train_output" / "last.ckpt").write_bytes(b"x" * 5000)
        (spare_dir / "checkpoints").mkdir()
        (spare_dir / "checkpoints" / "epoch1.pth").write_bytes(b"y" * 3000)
        outside = w.root / "outside"
        outside.mkdir()
        (outside / "precious.txt").write_text("keep me", encoding="utf-8")
        linked = False
        try:
            (spare_dir / "link_out").symlink_to(outside, target_is_directory=True)
            linked = True
        except (OSError, NotImplementedError):
            pass
        doomed = [p for p in spare_dir.rglob("*") if p.is_file() and not p.is_symlink()
                  and p.relative_to(spare_dir).as_posix() not in kept_files]
        doomed_bytes = sum(p.stat().st_size for p in doomed)
        row_before = w.db.get_training_run(spare)
        listed = {r["id"]: r for r in client.get("/api/train/runs").json()["runs"]}
        check("the Runs view shows how much space a run takes and offers removal",
              listed[spare]["files"]["removable"] is True and listed[spare]["files"]["bytes"] >= doomed_bytes)
        check("a run added as a model names its model and is not offered removal",
              listed[r1]["files"]["model"] == "Audtheia-Reef-V1" and listed[r1]["files"]["removable"] is False)
        refused = client.post(f"/api/train/runs/{r1}/remove-files")
        check("removing the files of a run added as a model is refused", refused.status_code == 409
              and (w.repo / w.db.get_training_run(r1)["run_dir"] / "export" / "model.onnx").is_file())
        res = client.post(f"/api/train/runs/{spare}/remove-files")
        body = res.json()
        check("the large files are removed and the space freed is reported",
              res.status_code == 200 and body["removed"]["freed_bytes"] == doomed_bytes
              and not any(p.exists() for p in doomed))
        check("the record, settings, log, figures by epoch and test scores are kept",
              all((spare_dir / n).read_text(encoding="utf-8") == t for n, t in kept_files.items()))
        check("emptied folders are tidied away and a note says when and how much",
              not (spare_dir / "export").exists() and not (spare_dir / "checkpoints").exists()
              and json.loads((spare_dir / R.REMOVED_FILE).read_text(encoding="utf-8"))["freed_bytes"] == doomed_bytes)
        check("the database record of the run is unchanged", w.db.get_training_run(spare) == row_before)
        if linked:
            check("a link inside the run folder is removed without touching what it points to",
                  not (spare_dir / "link_out").exists() and (outside / "precious.txt").read_text(encoding="utf-8") == "keep me")
        check("removing twice is refused", client.post(f"/api/train/runs/{spare}/remove-files").status_code == 409)
        after = {r["id"]: r for r in client.get("/api/train/runs").json()["runs"]}[spare]["files"]
        check("the Runs view then says the files were removed",
              after["removed"]["freed_bytes"] == doomed_bytes and after["removable"] is False)
        results = client.get(f"/api/train/runs/{spare}/results").json()
        check("Results still opens and says the files were removed", results["files_removed"] is not None)
        check("such a run cannot be scored again",
              client.post(f"/api/train/runs/{spare}/evaluate", json={}).status_code == 409)
        check("or added as a model", client.post(f"/api/train/runs/{spare}/register", json={}).status_code == 409)
        check("and it is no longer offered in the Models list",
              spare not in {r["id"] for r in client.get("/api/train/models").json()["unregistered_runs"]})
        failed_run = make_run(w, "Broken", "vision", d, status="failed")
        check("a failed run's files can be removed",
              client.post(f"/api/train/runs/{failed_run}/remove-files").status_code == 200)
        paused = make_run(w, "Held", "vision", d, status="paused")
        check("a paused run's files are kept until it is cancelled",
              client.post(f"/api/train/runs/{paused}/remove-files").status_code == 409
              and (w.repo / w.db.get_training_run(paused)["run_dir"] / "export" / "model.onnx").is_file())
        check("an unknown action is a plain 404", client.post(f"/api/train/runs/{spare}/delete").status_code == 404)
        if linked:
            stray = make_run(w, "Stray", "vision", d)
            stray_dir = w.repo / w.db.get_training_run(stray)["run_dir"]
            elsewhere = w.root / "elsewhere"
            stray_dir.rename(elsewhere)
            stray_dir.symlink_to(elsewhere, target_is_directory=True)
            check("a run folder that is a link to somewhere else is never emptied",
                  client.post(f"/api/train/runs/{stray}/remove-files").status_code in (404, 409)
                  and (elsewhere / "export" / "model.onnx").is_file())
        check("the interface offers the removal with a second press to confirm",
              "Remove its files" in client.get("/app.js").text and "Press again to remove" in client.get("/app.js").text)

        print("\n[7] Putting a model back after Audtheia restarts")
        sid2 = w.settings.stations()[1]["station_id"]
        published = json.loads(settings_file.read_text(encoding="utf-8"))
        for st in published["stations"]:
            if st["station_id"] == sid2:
                # A published shape that already matches the trained model, written as whole numbers,
                # so the switched-on values stay in the committed file and come back as 3, not 3.0.
                st["models"]["acoustic"].update({"sample_rate": 48000, "window_seconds": 3})
        settings_file.write_text(json.dumps(published, indent=2), encoding="utf-8")
        first_app = TestClient(srv.create_app(load_settings(settings_file), w.db))
        sound2 = {"slot": "station_acoustic", "station_id": sid2}
        on = first_app.post(f"/api/train/models/{m_snd['id']}/activate", json=dict(sound2, acknowledgement=ACK))
        before_sound = json.loads(w.db.model_activations("station_acoustic", sid2)[0]["previous_json"])
        restarted = TestClient(srv.create_app(load_settings(settings_file), w.db))
        reread = load_settings(settings_file).station(sid2)["models"]["acoustic"]
        check("after a restart the window length is read back as the whole number 3",
              on.status_code == 200 and reread["window_seconds"] == 3 and isinstance(reread["window_seconds"], int))
        view = [s_ for s_ in restarted.get("/api/train/models").json()["slots"]
                if s_["slot"] == "station_acoustic" and s_["station_id"] == sid2][0]
        check("the model switched on before the restart is still offered to put back", view["revertible"] is True)
        back = restarted.post("/api/train/slots/revert", json=sound2)
        check("and putting it back restores the slot as it was",
              back.status_code == 200 and load_settings(settings_file).station(sid2)["models"]["acoustic"] == before_sound)

    for name in ("registry.py", "deploy.py", "sharing.py"):
        text = (REPO / "audtheia" / "training" / name).read_text(encoding="utf-8")
        check(f"{name} is ASCII only", text.isascii())
    print(f"\nResult: {CHECKS['passed']} passed, {CHECKS['failed']} failed")
    return 0 if CHECKS["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
