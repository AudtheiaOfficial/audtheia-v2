"""Request-level verification for the Train tab's dataset endpoints.

Path: tests/test_training_endpoints.py

Builds the real FastAPI application over a small record in a temporary folder and
drives it through the in-process test client. Proves the status, counts, event
list, preview, build and detail endpoints answer correctly; that bad requests get
a plain 422 rather than a crash; that preview and build refuse to run anywhere but
the desktop; that the same selection builds once; that nothing in the data folder
or the database changes; and that the page and script that serve the tab are
delivered with the Train item in place and no other panel disturbed.

Needs the web framework and its test client, which the interface checks use too.
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

from training_fixture import World  # noqa: E402
from audtheia.app import server as srv  # noqa: E402

CHECKS = {"passed": 0, "failed": 0}
A = ("Aplysina fistularis", 100)
C = ("Poecile carolinensis", 200)


def check(label: str, condition: bool) -> None:
    CHECKS["passed" if condition else "failed"] += 1
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")


def tree_digest(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(root).rglob("*")) if p.is_file()}


def db_digest(world: World) -> dict:
    out = {}
    with world.db.connect() as conn:
        for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall():
            rows = conn.execute(f'SELECT * FROM "{name}"').fetchall()
            out[name] = (len(rows), hashlib.sha256(repr(sorted(repr(tuple(r)) for r in rows)).encode("utf-8")).hexdigest())
    return out


def run() -> None:
    from fastapi.testclient import TestClient

    with tempfile.TemporaryDirectory() as td:
        w = World(Path(td))
        for i in range(10):
            e = w.vision_event([A], station=f"S{i % 3}", first_seen=f"2026-03-{i + 1:02d}T10:00:00Z"); w.confirm(e)
        for i in range(6):
            e = w.vision_event([C], station=f"S{i % 3}", first_seen=f"2026-04-{i + 1:02d}T10:00:00Z"); w.confirm(e)
        pending = w.vision_event([A], station="S0", first_seen="2026-05-01T10:00:00Z", qc="qc_pending"); w.confirm(pending)
        for i in range(2):
            e = w.audio_event([C], station="S1", first_seen=f"2026-05-{i + 2:02d}T10:00:00Z"); w.confirm(e)

        client = TestClient(srv.create_app(w.settings, w.db))
        data_before, db_before = tree_digest(w.data), db_digest(w)

        r = client.get("/api/train/status")
        body = r.json()
        check("status answers", r.status_code == 200 and body["is_desktop"] is True)
        check("status carries the configured defaults", body["dataset_defaults"]["min_examples_per_class"] == 5)
        check("status lists the acoustic engines", [e["key"] for e in body["acoustic_engines"]] == ["acoustic_head_a", "acoustic_head_b"])
        check("status lists no dataset before one is built", body["datasets"] == [])

        r = client.get("/api/train/candidates/summary?modality=vision")
        s = r.json()
        check("the counts answer with the default criteria", r.status_code == 200 and s["candidates"] == 17 and s["eligible"] == 16)
        check("the counts say how many events each criterion covers", s["per_criterion"] == {"expert_confirmed": 17, "qc_passed": 16, "verified": 17})
        s2 = client.get("/api/train/candidates/summary?modality=vision&criteria=expert_confirmed").json()
        check("the counts follow the chosen criteria", s2["eligible"] == 17 and s2["required"] == ["expert_confirmed"])
        s3 = client.get("/api/train/candidates/summary?modality=vision&criteria=none").json()
        check("the word none means no criteria", s3["required"] == [] and s3["eligible"] == 17)
        check("an unknown criterion gets a plain 422", client.get("/api/train/candidates/summary?criteria=trusted").status_code == 422)
        check("an unknown data type gets a plain 422", client.get("/api/train/candidates/summary?modality=video").status_code == 422)
        check("audio events are counted separately", client.get("/api/train/candidates/summary?modality=audio").json()["candidates"] == 2)

        lst = client.get("/api/train/candidates?modality=vision&limit=5").json()
        check("the event list is paged", lst["total"] == 17 and len(lst["events"]) == 5)
        row = lst["events"][0]
        check("each listed event carries its criteria and missing steps", set(row) >= {"criteria", "missing_steps", "label_state", "eligible", "taxa", "event_name"})
        only = client.get("/api/train/candidates?modality=vision&only_eligible=true&limit=100").json()
        check("only qualifying events can be asked for", only["total"] == 16 and all(e["eligible"] for e in only["events"]))
        waiting = [e for e in client.get("/api/train/candidates?modality=vision&limit=100").json()["events"] if not e["eligible"]]
        check("an event short of a step says which", len(waiting) == 1 and waiting[0]["missing_steps"] == ["qc"])
        check("a bad page size gets a plain 422", client.get("/api/train/candidates?limit=0").status_code == 422)

        r = client.post("/api/train/dataset/preview", json={"modality": "vision"})
        preview = r.json()
        check("a preview answers with the plan", r.status_code == 200 and preview["dataset_id"].startswith("vision-"))
        check("the preview reports the split", {k: v["events"] for k, v in preview["totals"].items()} == {"train": 10, "validation": 3, "test": 3})
        check("the preview carries no media or file paths", "path" not in json.dumps(preview["events"]) and "samples" not in json.dumps(preview))
        check("the preview wrote no dataset", client.get("/api/train/status").json()["datasets"] == [])
        check("the preview changed nothing", tree_digest(w.data) == data_before and db_digest(w) == db_before)
        check("a misspelt field gets a plain 422", client.post("/api/train/dataset/preview", json={"modality": "vision", "minimum": 3}).status_code == 422)
        check("an acoustic preview needs an engine", client.post("/api/train/dataset/preview", json={"modality": "audio"}).status_code == 422)
        check("a missing data type gets a plain 422", client.post("/api/train/dataset/preview", json={}).status_code == 422)
        bad = client.post("/api/train/dataset/preview", json={"modality": "vision", "class_policy": {"gbif:1": "delete"}})
        check("a bad class decision gets a plain 422 that says why", bad.status_code == 422 and "class_policy" in bad.json()["detail"])

        first = client.post("/api/train/dataset/build", json={"modality": "vision"}).json()
        check("a build writes the dataset the preview described", first["already_exists"] is False and first["dataset_id"] == preview["dataset_id"])
        check("the build reports where it wrote", Path(first["path"]).is_dir())
        check("the build changed nothing in the record", tree_digest(w.data) == data_before and db_digest(w) == db_before)
        second = client.post("/api/train/dataset/build", json={"modality": "vision"}).json()
        check("the same selection builds once", second["already_exists"] is True and second["path"] == first["path"])
        listed = client.get("/api/train/status").json()["datasets"]
        check("status then lists the dataset", [d["dataset_id"] for d in listed] == [first["dataset_id"]])

        detail = client.get(f"/api/train/dataset/{first['dataset_id']}")
        check("a built dataset's manifest can be fetched", detail.status_code == 200 and detail.json()["dataset_id"] == first["dataset_id"])
        check("the fetched manifest leaves out the per-example list but counts it", "items" not in detail.json() and detail.json()["item_count"] > 0)
        check("an unknown dataset is a plain 404", client.get("/api/train/dataset/vision-0000000000000000").status_code == 404)
        check("a hidden dataset name is a plain 404", client.get("/api/train/dataset/.rollup_cache.json").status_code == 404)

        empty = client.post("/api/train/dataset/build", json={"modality": "audio", "engine": "acoustic_head_a", "required_criteria": ["expert_confirmed"], "station_id": w.stations["S0"]})
        check("building a selection that matches nothing is a plain 422", empty.status_code == 422 and "nothing" in empty.json()["detail"])

        audio = client.post("/api/train/dataset/preview", json={"modality": "audio", "engine": "acoustic_head_a"})
        check("an acoustic preview answers", audio.status_code == 200 and audio.json()["options"]["window_seconds"] == 3.0)

        w.settings.raw["node"]["role"] = "pi"
        guarded_p = client.post("/api/train/dataset/preview", json={"modality": "vision"})
        guarded_b = client.post("/api/train/dataset/build", json={"modality": "vision"})
        check("preview is refused away from the desktop", guarded_p.status_code == 403 and "desktop" in guarded_p.json()["detail"])
        check("build is refused away from the desktop", guarded_b.status_code == 403)
        check("status says it is not the desktop", client.get("/api/train/status").json()["is_desktop"] is False)
        w.settings.raw["node"]["role"] = "desktop"

        page = client.get("/")
        script = client.get("/app.js")
        style = client.get("/style.css")
        check("the page is served", page.status_code == 200 and 'data-panel="train"' in page.text and 'id="panel-train"' in page.text)
        check("the Train item sits after Brain in the sidebar", page.text.index('data-panel="brain"') < page.text.index('data-panel="train"') < page.text.index('data-panel="gps"'))
        check("the script is served with the Train view", script.status_code == 200 and "train-dataset" in script.text and "loadTrainDataset" in script.text)
        check("the stylesheet is served with the Train styles", style.status_code == 200 and ".train-controls" in style.text)
        check("the Brain panel and its sub-panels are still present", all(k in page.text for k in ('id="subpanel-brain-models"', 'id="subpanel-brain-learning"', 'id="subpanel-brain-skills"')))
        for tab in ("detections", "audio", "brain", "train", "gps", "sensors", "analytics", "reports", "settings"):
            check(f"the {tab} panel exists", f'id="panel-{tab}"' in page.text)


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  {type(exc).__name__}: {exc}")
        CHECKS["failed"] += 1
        import traceback
        traceback.print_exc()
    print(f"\nResult: {CHECKS['passed']} passed, {CHECKS['failed']} failed")
    sys.exit(1 if CHECKS["failed"] else 0)
