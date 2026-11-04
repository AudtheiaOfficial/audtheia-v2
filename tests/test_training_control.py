"""Verification for the Train view's hardware, install and run endpoints.

Path: tests/test_training_control.py

Builds the real application over a small record in a temporary folder. The
package installer, the speed measurement and the process launcher are replaced
with stand-ins, so nothing is downloaded and no training process is started,
while everything else is real: the settings, the database, the dataset builder,
the safety checks and the run records. Checks that the hardware card answers,
that installing needs the size first and an explicit confirmation, that the
speed measurement runs in the training environment and gives a time and the
disk space, that a run cannot start on battery, during capture without the
person's choice, or without the components, and that pause, resume and cancel
reach the run. Also checks that a field station refuses every one of these, and
that nothing in the record or the data folder changes. For sound, checks that
the sound components install through the same endpoints, that the estimate says
when the speed will be measured, that an engine with no base model yet is refused,
and that a sound run starts on the sound engine with the dataset's engine.

Uses the standard library, NumPy and Pillow, plus the project itself.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from training_fixture import World  # noqa: E402
from test_training_components import FakePip, FakePopen, sizer  # noqa: E402
from audtheia.app import server as srv  # noqa: E402
from audtheia.training import components as C  # noqa: E402
from audtheia.training import hardware as H  # noqa: E402
from audtheia.training import runs as R  # noqa: E402

CHECKS = {"passed": 0, "failed": 0}
A = ("Aplysina fistularis", 100)
B = ("Aplysina lacunosa", 102)


def check(label: str, condition: bool) -> None:
    CHECKS["passed" if condition else "failed"] += 1
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")


def tree_digest(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


class QuietLauncher(R.ProcessLauncher):
    def __init__(self):
        self.launched = []

    def launch(self, python, run_dir, code_root, *, resume, gentle):
        self.launched.append({"python": python, "run_dir": str(run_dir), "resume": resume, "gentle": gentle})
        return 4321

    def stop(self, pid):
        return True


def wait_job(client, url, key="job", timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        body = client.get(url).json()
        job = body.get(key)
        if job and job.get("status") in ("done", "error"):
            return body
        time.sleep(0.05)
    return client.get(url).json()


def laptop_probe(on_ac=True):
    def probe(path=".", *, sources=None, include_torch=False):
        return {"platform": "windows", "machine": "AMD64", "python": "3.13.1", "notes": [],
                "cpu": {"name": "13th Gen Intel(R) Core(TM) i7-1355U", "logical_cores": 12},
                "memory": {"total_gb": 16.0, "available_gb": 9.0, "source": "test"},
                "disk": {"path": str(path), "total_gb": 500.0, "free_gb": 120.0},
                "power": {"on_ac": on_ac, "has_battery": True, "battery_percent": 80.0, "source": "test"},
                "gpus": [{"vendor": "intel", "name": "Intel(R) Iris(R) Xe Graphics", "memory_gb": None,
                          "driver": None, "cuda": None, "compute_capability": None}],
                "thermal": {"cpu_signal": False, "gpu_signal": False,
                            "sample": {"cpu_throttle_count": None, "gpu_thermal_slowdown": None}}}
    return probe


class Busy:
    def running(self):
        return True


def run() -> None:
    from fastapi.testclient import TestClient

    real_probe = H.probe
    H.probe = laptop_probe(True)
    try:
        with tempfile.TemporaryDirectory() as td:
            w = World(Path(td))
            for i in range(8):
                e = w.vision_event([A], station=f"S{i % 3}", first_seen=f"2026-03-{i + 1:02d}T10:00:00Z"); w.confirm(e)
            for i in range(8):
                e = w.vision_event([B], station=f"S{i % 3}", first_seen=f"2026-04-{i + 1:02d}T10:00:00Z"); w.confirm(e)
            app = srv.create_app(w.settings, w.db)
            client = TestClient(app)
            launcher = QuietLauncher()
            app.state.training["components"] = C.Components(w.settings, base_python="/usr/bin/python-base",
                                                            runner=FakePip(), sizer=sizer, popen=FakePopen())
            app.state.training["runs"] = R.RunManager(w.settings, w.db, launcher=launcher)
            data_before = tree_digest(w.data)

            r = client.get("/api/train/hardware")
            hw = r.json()
            check("the hardware card answers", r.status_code == 200 and hw["hardware"]["cpu"]["logical_cores"] == 12)
            check("it chooses the processor build for Iris Xe graphics", hw["choice"]["vendor_key"] == "cpu"
                  and hw["choice"]["url"] == "https://download.pytorch.org/whl/cpu")
            check("it describes the computer in plain lines", any("i7-1355U" in line for line in hw["lines"]))
            check("it says a run could start now", hw["start_check"]["ok"] is True)
            check("it carries the safety limits from the settings", hw["limits"]["min_free_disk_gb"] == 5.0)

            r = client.get("/api/train/components")
            check("the components start uninstalled", r.status_code == 200 and r.json()["state"] == "not_installed")
            r = client.post("/api/train/components/install", json={"engine": "vision_detector", "confirm": True})
            check("installing before the size is known is refused", r.status_code == 409)
            r = client.post("/api/train/components/plan", json={"engine": "vision_detector"})
            check("working out the size starts in the background", r.status_code == 202)
            body = wait_job(client, "/api/train/components", key="plan_job")
            plan = body["plan_job"]["result"]
            check("the plan reports the measured download", body["plan_job"]["status"] == "done"
                  and plan["download_bytes"] == 132_900_000 and plan["vendor_key"] == "cpu")
            r = client.post("/api/train/components/install", json={"engine": "vision_detector"})
            check("installing without confirmation is refused", r.status_code == 400)
            r = client.post("/api/train/components/install", json={"engine": "vision_detector", "confirm": True})
            check("installing with confirmation starts", r.status_code == 202)
            end = time.time() + 20
            while time.time() < end and client.get("/api/train/components").json()["state"] != "ready":
                time.sleep(0.05)
            check("the components become ready", client.get("/api/train/components").json()["state"] == "ready")
            log = client.get("/api/train/components/log").json()
            check("the install log can be read", "Installed and checked." in log["text"] and log["next_offset"] > 0)
            tail = client.get(f"/api/train/components/log?offset={log['next_offset']}").json()
            check("reading on from the end gives nothing new", tail["text"] == "")
            check("an unknown engine is refused", client.get("/api/train/components?engine=nope").status_code == 404)

            r = client.post("/api/train/runs", json={"dataset_id": "vision-nothing", "name": "Reef"})
            check("a run needs a built dataset", r.status_code == 404)
            ds = client.post("/api/train/dataset/build", json={"modality": "vision"}).json()
            req = {"dataset_id": ds["dataset_id"], "name": "Reef", "base": "nano", "mode": "gentle"}

            seen = {}

            def fake_measure(args, timeout, pythonpath=None):
                seen.update(args=list(args), pythonpath=pythonpath)
                return 0, 'AUDTHEIA_BENCHMARK {"base_model": "nano", "device": "cpu", "parameters_mb": 116.2, ' \
                          '"seconds_per_train_image": 2.0, "seconds_per_eval_image": 0.5}\n', ""

            real_measure = C.run_capture
            C.run_capture = fake_measure
            try:
                r = client.post("/api/train/estimate", json={"dataset_id": ds["dataset_id"], "base": "nano"})
                check("measuring the speed starts in the background", r.status_code == 202)
                est = wait_job(client, "/api/train/estimate")["job"]
            finally:
                C.run_capture = real_measure
            check("the measurement runs Audtheia's module in the training environment, with the code folder "
                  "on its search path", seen["args"][1:4] == ["-m", "audtheia.training.vision", "benchmark"]
                  and "vision" in seen["args"][0] and seen["pythonpath"] == str(R.CODE_ROOT))
            check("Gentle is measured as it trains: one image at a time with checkpointing",
                  seen["args"][seen["args"].index("--micro-batch") + 1] == "1" and "--checkpointing" in seen["args"])
            result = est["result"]
    # 100 epochs keep the last checkpoint and ten numbered ones, each four times
            # the weights, plus four times the weights in light files and the export.
            check("the estimate gives a time and the disk space from the model's size",
                  est["status"] == "done" and result["estimate"]["hours"] > 0
                  and abs(result["estimate"]["disk_gb"] * 1024.0 - (11 * 4 + 4) * 116.2) < 0.01)

            H.probe = laptop_probe(False)
            r = client.post("/api/train/runs", json=req)
            check("a run does not start on battery, and says why", r.status_code == 409 and "charger" in r.json()["detail"])
            H.probe = laptop_probe(True)

            srv._CAPTURE_JOBS["test-station"] = Busy()
            try:
                r = client.post("/api/train/runs", json=req)
                check("during capture a run needs the person's choice", r.status_code == 409
                      and "Continue anyway" in r.json()["detail"])
                r = client.post("/api/train/runs", json=dict(req, continue_while_capturing=True))
                check("Continue anyway starts it", r.status_code == 201)
            finally:
                srv._CAPTURE_JOBS.pop("test-station", None)
            run = r.json()["run"]
            check("the run is recorded as Gentle on the Nano size", run["mode"] == "gentle" and run["base_label"] == "Nano"
                  and run["planned_model_name"] == "Audtheia-Reef-V1" and run["status"] == "preparing")
            check("the worker was launched with the training environment's interpreter, gently",
                  launcher.launched[-1]["gentle"] is True and "vision" in launcher.launched[-1]["python"])
            check("the run folder holds the configuration", (Path(launcher.launched[-1]["run_dir"]) / "run.json").is_file())
            r = client.post("/api/train/runs", json=dict(req, name="Second", continue_while_capturing=True))
            check("a second run cannot start while one is active", r.status_code == 409)
            r = client.post("/api/train/runs", json=dict(req, name="bad name"))
            check("a model name with a space is refused", r.status_code in (409, 422))

            listed = client.get("/api/train/runs").json()["runs"]
            check("the run is listed", [x["id"] for x in listed] == [run["id"]])
            check("an active run carries what its worker last reported", listed[0]["live"]["state"] == "preparing")
            detail = client.get(f"/api/train/runs/{run['id']}").json()
            check("its detail carries the record, progress and live state", detail["run"]["id"] == run["id"]
                  and detail["progress"] == [] and "state" in detail["live"])
            check("its log can be read", client.get(f"/api/train/runs/{run['id']}/log").status_code == 200)
            r = client.post(f"/api/train/runs/{run['id']}/pause")
            check("Pause is accepted for an active run", r.status_code == 200)
            r = client.post(f"/api/train/runs/{run['id']}/resume")
            check("Resume is refused while it is still active", r.status_code == 409)
            r = client.post("/api/train/components/remove", json={"engine": "vision_detector", "confirm": True})
            check("the components cannot be removed while a run is active", r.status_code == 409)
            r = client.post(f"/api/train/runs/{run['id']}/cancel")
            check("Cancel is accepted", r.status_code == 200)
            check("an unknown action is refused", client.post(f"/api/train/runs/{run['id']}/explode").status_code == 404)
            check("a malformed run id is refused", client.get("/api/train/runs/..%2F..%2Fdatabase").status_code == 404
                  and client.get("/api/train/runs/not-a-run").status_code == 404)
            check("the record and the data folder were not changed by any of this", tree_digest(w.data) == data_before)

            w.db.update_training_run(run["id"], status="cancelled")
            r = client.post("/api/train/components/remove", json={"engine": "vision_detector"})
            check("removing needs confirmation", r.status_code == 400)
            r = client.post("/api/train/components/remove", json={"engine": "vision_detector", "confirm": True})
            check("removing then deletes the environment", r.status_code == 200 and r.json()["removed"] is True
                  and client.get("/api/train/components").json()["state"] == "not_installed")
            r = client.post("/api/train/estimate", json={"dataset_id": ds["dataset_id"], "base": "nano"})
            check("an estimate needs the components", r.status_code == 409)

            w.settings.raw["node"]["role"] = "field"
            refused = [client.get("/api/train/hardware").status_code, client.get("/api/train/components").status_code,
                       client.post("/api/train/runs", json=req).status_code, client.get("/api/train/runs").status_code,
                       client.post("/api/train/components/plan", json={}).status_code]
            check("a field station refuses every training control", refused == [403] * 5)
    finally:
        H.probe = real_probe


def run_sound() -> None:
    from fastapi.testclient import TestClient

    print("sound training")
    real_probe = H.probe
    H.probe = laptop_probe(True)
    try:
        with tempfile.TemporaryDirectory() as td:
            w = World(Path(td))
            for i in range(6):
                e = w.audio_event([A], station=f"S{i % 3}", first_seen=f"2026-03-{i + 1:02d}T06:00:00Z"); w.confirm(e)
                e = w.audio_event([B], station=f"S{i % 3}", first_seen=f"2026-04-{i + 1:02d}T06:00:00Z"); w.confirm(e)
            app = srv.create_app(w.settings, w.db)
            client = TestClient(app)
            launcher = QuietLauncher()
            comps = C.Components(w.settings, base_python="/usr/bin/python-base", runner=FakePip(), sizer=sizer,
                                 popen=FakePopen(), windows=False)
            app.state.training["components"] = comps
            app.state.training["runs"] = R.RunManager(w.settings, w.db, launcher=launcher)
            status = client.get("/api/train/status").json()
            check("the status carries the sound defaults and each engine's base model",
                  status["acoustic_defaults"]["batch_size"] == 32
                  and status["acoustic_engines"][0]["base"]["label"] == "BirdNET 2.4"
                  and status["acoustic_engines"][1]["base"] is None)
            ds_a = client.post("/api/train/dataset/build", json={"modality": "audio", "engine": "acoustic_head_a",
                                                                 "required_criteria": ["expert_confirmed"]}).json()
            ds_b = client.post("/api/train/dataset/build", json={"modality": "audio", "engine": "acoustic_head_b",
                                                                 "required_criteria": ["expert_confirmed"]}).json()
            req = {"dataset_id": ds_a["dataset_id"], "name": "Dawn", "mode": "gentle"}
            r = client.post("/api/train/runs", json=req)
            check("a sound run needs the sound components", r.status_code == 409 and "sound" in r.json()["detail"])
            client.post("/api/train/components/plan", json={"engine": "acoustic_head"})
            wait_job(client, "/api/train/components?engine=acoustic_head", key="plan_job")
            r = client.post("/api/train/components/install", json={"engine": "acoustic_head", "confirm": True})
            C.wait_for_job(comps, "acoustic_head", 20)
            check("the sound components install through the same endpoints", r.status_code == 202
                  and client.get("/api/train/components?engine=acoustic_head").json()["state"] == "ready")
            r = client.post("/api/train/estimate", json={"dataset_id": ds_a["dataset_id"], "mode": "gentle"})
            est = wait_job(client, "/api/train/estimate")["job"]
            check("before the base model is downloaded the estimate gives the disk and says when the speed is measured",
                  r.status_code == 202 and est["status"] == "done" and est["result"]["estimate"]["hours"] is None
                  and est["result"]["estimate"]["disk_gb"] > 0.2 and "124.5 MB" in est["result"]["note"])
            r = client.post("/api/train/runs", json=dict(req, engine="vision_detector"))
            check("the image engine refuses a sound dataset", r.status_code == 422)
            r = client.post("/api/train/runs", json={"dataset_id": ds_b["dataset_id"], "name": "Reef", "mode": "gentle"})
            check("an engine with no base model yet is refused in plain words", r.status_code == 422
                  and "no base model in this version yet" in r.json()["detail"])
            r = client.post("/api/train/runs", json=dict(req, overrides={"dropout": 2}))
            check("a sound setting out of range is refused", r.status_code == 422)
            r = client.post("/api/train/runs", json=req)
            run = r.json()["run"]
            cfg = json.loads((Path(launcher.launched[-1]["run_dir"]) / "run.json").read_text())
            check("a sound run starts on the sound engine in the sound environment, gently",
                  r.status_code == 201 and run["modality"] == "acoustic" and run["engine"] == "acoustic_head"
                  and run["base_label"] == "BirdNET 2.4" and run["planned_model_name"] == "Audtheia-Dawn-V1"
                  and launcher.launched[-1]["gentle"] is True and "acoustic" in launcher.launched[-1]["python"])
            check("its configuration names the engine of the dataset and keeps the base model in the environment",
                  cfg["engine_key"] == "acoustic_head_a" and cfg["embedding_batch"] == 1
                  and Path(cfg["bases_dir"]) == comps.env_dir("acoustic_head") / "bases")
            check("the one-run-at-a-time rule covers both engines",
                  client.post("/api/train/runs", json=dict(req, name="Again")).status_code == 409)
            client.post(f"/api/train/runs/{run['id']}/cancel")
    finally:
        H.probe = real_probe


def main() -> int:
    run()
    run_sound()
    print(f"\nResult: {CHECKS['passed']} passed, {CHECKS['failed']} failed")
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
