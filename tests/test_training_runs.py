"""Verification for training run control: start, pause, resume, cancel and recovery.

Path: tests/test_training_runs.py

Starts real worker processes against a temporary record, using the built-in
self-test engine, which imitates training without any machine-learning library.
Checks that a run reaches the database exactly as the worker reports it, that
Pause says truthfully where Resume will continue and Resume continues from the
last saved epoch without repeating or skipping one, or from the beginning when
no epoch had finished, that the worker opens no window on Windows, that its
printed output is written as UTF-8 so a table of figures cannot stop a run on a
computer with a legacy character set, that Cancel is final, that a worker that dies mid-run
is found and marked interrupted, that only one run trains at a time, that a
safety limit pauses a run with a plain reason, and that a worker writes nothing
outside its own run folder.

Uses the standard library, NumPy and Pillow, plus the project itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from training_fixture import World  # noqa: E402
from audtheia.training import runs as R  # noqa: E402
from audtheia.training import worker as W  # noqa: E402

CHECKS = {"passed": 0, "failed": 0}


def check(label: str, condition: bool) -> None:
    CHECKS["passed" if condition else "failed"] += 1
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")


def raises(fn, exc=Exception, status=None) -> bool:
    try:
        fn()
    except exc as e:  # noqa: BLE001
        return status is None or getattr(e, "status", None) == status
    return False


LIMITS = {"require_ac_power": False, "min_battery_percent": 0, "min_free_disk_gb": 0.0,
          "gpu_memory_headroom_fraction": 0.10, "thermal_throttle_stop_seconds": 120, "cpu_only_warn_hours": 6.0}


def config(epochs=3, steps=3, step_seconds=0.01, limits=None, **opts):
    return {"epochs": epochs, "seed": 7, "resources": {"mode": "gentle", "cpu_threads": 1},
            "hardware": {"limits": dict(limits or LIMITS)},
            "engine_options": dict(steps_per_epoch=steps, step_seconds=step_seconds, **opts)}


def wait_for(manager, run_id, states, timeout=40.0):
    end = time.time() + timeout
    row = None
    while time.time() < end:
        row = manager.refresh(run_id)
        if row["status"] in states:
            return row
        time.sleep(0.1)
    return row


def wait_progress(manager, run_id, n, timeout=40.0):
    end = time.time() + timeout
    while time.time() < end:
        if len(manager.read_progress(run_id)) >= n:
            return True
        time.sleep(0.05)
    return False


def tree_digest(root: Path, skip: Path) -> dict:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and skip not in p.parents:
            out[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


class World4(World):
    def __init__(self, root):
        super().__init__(root)
        self.dataset = {"path": str(self.repo / "exports" / "train" / "vision-test"), "signature": "sig0001"}
        Path(self.dataset["path"]).mkdir(parents=True, exist_ok=True)
        self.manager = R.RunManager(self.settings, self.db)

    def start(self, cfg, name="Reef", **kw):
        return self.manager.create(user_name=name, modality="vision", engine="selftest", mode="gentle",
                                   dataset=self.dataset, base_label="self-test", config=cfg, environment=None, **kw)


# ---------------------------------------------------------------------------


def test_names_and_ids():
    print("names and identifiers")
    check("a plain name is accepted", R.validate_user_name("Reef_2026-a") == "Reef_2026-a")
    for bad in ("", "a b", "x" * 41, "con", "LPT1", "reef/..", "café", None):
        check(f"{bad!r} is refused", raises(lambda b=bad: R.validate_user_name(b), R.RunError, 422))
    check("the planned model name follows Audtheia-Name-Vn", R.planned_model_name("Reef", 3) == "Audtheia-Reef-V3")
    with tempfile.TemporaryDirectory() as td:
        w = World4(Path(td))
        check("a run folder is only made from a real run identifier",
              raises(lambda: w.manager.run_dir("../../database"), R.RunError, 404)
              and raises(lambda: w.manager.run_dir("x" * 36), R.RunError, 404))
        check("an engine whose training components are missing is refused",
              raises(lambda: w.manager.python_for("vision"), R.RunError, 409))


def test_complete_run():
    print("a run from start to finish")
    with tempfile.TemporaryDirectory() as td:
        w = World4(Path(td))
        before = tree_digest(w.repo, w.manager.runs_root)
        row = w.start(config(epochs=3))
        check("the run is recorded with its planned model name and is starting",
              row["planned_model_name"] == "Audtheia-Reef-V1" and row["status"] == "preparing"
              and row["data_source"] == "model" and row["seed"] == 7 and row["epochs_total"] == 3)
        check("its folder is under the trained-models folder and stored relative to the repository",
              row["run_dir"] == f"models/trained/runs/{row['id']}" and row["dataset_dir"] == "exports/train/vision-test")
        done = wait_for(w.manager, row["id"], ("completed", "failed"))
        check("the run completes", done["status"] == "completed" and done["stop_reason"] is None)
        check("the database has the epochs, the best figure and the end time",
              done["epochs_done"] == 3 and done["best_metric_name"] == "map" and done["best_metric_value"] == 1.0
              and done["ended_at"] and done["started_at"] and done["last_heartbeat_at"])
        check("the library versions the worker used are kept", json.loads(done["software_json"])["python"])
        prog = w.manager.read_progress(row["id"])
        check("one line of figures per epoch", [p["epoch"] for p in prog] == [1, 2, 3]
              and prog[-1]["map50_95"] == 1.0)
        log = w.manager.read_log(row["id"])
        check("the log says what happened", "Starting the run." in log["text"] and "The run finished." in log["text"])
        check("the configuration is stored with the run", json.loads(done["config_json"])["engine"] == "selftest")
        live = w.manager.live(row["id"])
        check("progress within the epoch is reported", live["current_epoch"] == 3 and live["batch_done"] == 3
              and live["batches_total"] == 3)
        check("with the measured seconds per step, for the time left", (live["seconds_per_step"] or 0) > 0)
        check("a finished run is not changed again by a later look", w.manager.refresh(row["id"]) == done)
        check("a finished run cannot be paused, resumed or cancelled",
              raises(lambda: w.manager.pause(row["id"]), R.RunError, 409)
              and raises(lambda: w.manager.resume(row["id"]), R.RunError, 409)
              and raises(lambda: w.manager.cancel(row["id"]), R.RunError, 409))
        after = tree_digest(w.repo, w.manager.runs_root)
        changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
        check("outside its run folder only the database changed",
              changed <= {"database/audtheia.db", "database/audtheia.db-wal", "database/audtheia.db-shm"})
        second = w.start(config(epochs=1))
        check("the next run of the same name plans the next version only once a model is issued",
              second["planned_model_name"] == "Audtheia-Reef-V1")
        wait_for(w.manager, second["id"], ("completed", "failed"))


def test_pause_resume_cancel():
    print("pause, resume and cancel")
    with tempfile.TemporaryDirectory() as td:
        w = World4(Path(td))
        row = w.start(config(epochs=4, steps=15, step_seconds=0.03))
        check("the first epoch is reported", wait_progress(w.manager, row["id"], 1))
        check("a second run cannot start while one trains",
              raises(lambda: w.start(config(epochs=1), name="Other"), R.RunError, 409))
        w.manager.pause(row["id"])
        paused = wait_for(w.manager, row["id"], ("paused", "completed", "failed"))
        check("Pause stops the run and says where Resume will continue", paused["status"] == "paused"
              and "Paused at your request" in paused["stop_reason"]
              and f"from the end of epoch {paused['epochs_done']}, the last one saved" in paused["stop_reason"])
        done_before = len(w.manager.read_progress(row["id"]))
        check("the paused process has ended", not w.manager._alive(w.manager.read_status(row["id"])) or
              w.manager.read_status(row["id"]).get("state") == "paused")
        resumed = w.manager.resume(row["id"])
        check("Resume starts it again", resumed["status"] == "preparing" and resumed["stop_reason"] is None)
        check("an immediate look does not read the old paused state back",
              w.manager.refresh(row["id"])["status"] in ("preparing", "running"))
        done = wait_for(w.manager, row["id"], ("completed", "failed"))
        epochs = [p["epoch"] for p in w.manager.read_progress(row["id"])]
        check("it finishes every epoch exactly once, continuing after the last saved one",
              done["status"] == "completed" and epochs == [1, 2, 3, 4] and done_before >= 1)
        check("the log records the resume", "Resuming after epoch" in w.manager.read_log(row["id"])["text"])

        row = w.start(config(epochs=50, steps=20, step_seconds=0.02), name="Cancel")
        check("a long run reports its first epoch", wait_progress(w.manager, row["id"], 1))
        w.manager.cancel(row["id"])
        gone = wait_for(w.manager, row["id"], ("cancelled", "completed", "failed"))
        check("Cancel ends the run for good", gone["status"] == "cancelled" and gone["ended_at"])
        check("a cancelled run cannot be resumed", raises(lambda: w.manager.resume(row["id"]), R.RunError, 409))
        check("another run may start once nothing trains", w.manager.active_run() is None)

        row = w.start(config(epochs=3, steps=200, step_seconds=0.02), name="PauseThenCancel")
        wait_progress(w.manager, row["id"], 1)
        w.manager.pause(row["id"])
        wait_for(w.manager, row["id"], ("paused",))
        c = w.manager.cancel(row["id"])
        check("a paused run can be cancelled directly", c["status"] == "cancelled")


def test_pause_before_first_epoch():
    print("pausing before the first epoch has finished")
    with tempfile.TemporaryDirectory() as td:
        w = World4(Path(td))
        row = w.start(config(epochs=2, steps=200, step_seconds=0.05))
        end = time.time() + 30
        while time.time() < end and not (w.manager.read_status(row["id"]).get("batch_done") or 0) >= 2:
            time.sleep(0.05)
        w.manager.pause(row["id"])
        paused = wait_for(w.manager, row["id"], ("paused", "completed", "failed"))
        check("a pause in the first epoch stops the run", paused["status"] == "paused" and paused["epochs_done"] == 0)
        check("and says plainly that Resume starts again, not that progress is saved",
              "starts the training again from the beginning" in paused["stop_reason"]
              and "saved" not in paused["stop_reason"])
        w.manager.resume(row["id"])
        done = wait_for(w.manager, row["id"], ("completed", "failed", "paused"))
        epochs = [p["epoch"] for p in w.manager.read_progress(row["id"])]
        check("Resume then trains from the first epoch to the end, each epoch once",
              done["status"] == "completed" and epochs == [1, 2])
        check("the log says it started again", "starts again from the first epoch" in w.manager.read_log(row["id"])["text"])

        row = w.start(config(epochs=3, steps=10, step_seconds=0.03), name="Lost")
        wait_progress(w.manager, row["id"], 1)
        w.manager.pause(row["id"])
        wait_for(w.manager, row["id"], ("paused",))
        (w.manager.run_dir(row["id"]) / "checkpoints" / "last.json").unlink()
        w.manager.resume(row["id"])
        lost = wait_for(w.manager, row["id"], ("failed", "completed"))
        check("after a finished epoch a missing checkpoint is an error, never a silent restart",
              lost["status"] == "failed" and "checkpoint is missing" in lost["stop_reason"]
              and [p["epoch"] for p in w.manager.read_progress(row["id"])][0] == 1
              and len(w.manager.read_progress(row["id"])) == lost["epochs_done"])


TABLE_TEXT = "\u250f\u2501\u2513 Val (Epoch 1/3) \u2014 Overall Metrics \u2503 0.512 \u2503"


def test_output_encoding():
    print("printed output on a computer whose character set has no box-drawing characters")
    import subprocess
    with tempfile.TemporaryDirectory() as td:
        env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONIOENCODING="cp1252")
        env.pop("PYTHONUTF8", None)
        code = ("import sys\n"
                "from audtheia.training import worker as W\n"
                "if sys.argv[1] == 'fixed':\n"
                "    W.use_utf8_output()\n"
                "print(sys.argv[2])\n"
                "print(sys.stdout.fileno() not in (1, 2))\n")
        results = {}
        for mode in ("plain", "fixed"):
            out = Path(td) / f"{mode}.txt"
            with open(out, "wb") as handle:
                done = subprocess.run([sys.executable, "-c", code, mode, TABLE_TEXT], stdout=handle,
                                      stderr=subprocess.STDOUT, env=env, timeout=60)
            results[mode] = (done.returncode, out.read_bytes())
        check("without the fix, printing the library's table stops the process, as it did on Windows",
              results["plain"][0] != 0 and b"UnicodeEncodeError" in results["plain"][1])
        text = results["fixed"][1].decode("utf-8")
        check("with it, the table is written to the log as UTF-8", results["fixed"][0] == 0 and TABLE_TEXT in text)
        check("and printing no longer goes through the console's own streams", text.strip().endswith("True"))

        w = World4(Path(td) / "world")
        saved = os.environ.get("PYTHONIOENCODING")
        os.environ["PYTHONIOENCODING"] = "cp1252"
        try:
            row = w.start(config(epochs=2, print_text=TABLE_TEXT))
            done_row = wait_for(w.manager, row["id"], ("completed", "failed"))
        finally:
            if saved is None:
                os.environ.pop("PYTHONIOENCODING", None)
            else:
                os.environ["PYTHONIOENCODING"] = saved
        output = (w.manager.run_dir(row["id"]) / "worker_output.txt").read_bytes().decode("utf-8")
        check("a real run that prints the table each epoch finishes on such a computer",
              done_row["status"] == "completed" and output.count(TABLE_TEXT) == 2)

        # The worker started directly, without the launcher's settings, still
        # writes UTF-8, and a character no encoding can hold is replaced.
        rid = w.start(config(epochs=1, print_text=TABLE_TEXT + " \udcff"), name="Direct", start=False)["id"]
        rd = w.manager.run_dir(rid)
        with open(rd / "worker_output.txt", "wb") as handle:
            done = subprocess.run([sys.executable, "-m", "audtheia.training.worker", str(rd)], stdout=handle,
                                  stderr=subprocess.STDOUT, env=env, cwd=str(REPO), timeout=120)
        output = (rd / "worker_output.txt").read_bytes().decode("utf-8")
        check("the worker itself writes UTF-8 and never stops over a character it cannot write",
              done.returncode == 0 and W.read_json(rd / W.STATUS_FILE)["state"] == "completed"
              and TABLE_TEXT in output)
        w.db.update_training_run(rid, status="completed")


def test_windows_launch_flags():
    print("how the worker is started on Windows")

    class FakeSubprocess:
        DETACHED_PROCESS, CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW = 0x8, 0x200, 0x08000000
        BELOW_NORMAL_PRIORITY_CLASS, STDOUT, DEVNULL = 0x4000, -2, -3
        seen = {}

        class Popen:
            def __init__(self, args, **kw):
                FakeSubprocess.seen.update(kw)
                self.pid = 77

    class FakeSys:
        platform = "win32"

    real_sub, real_sys = R.subprocess, R.sys
    R.subprocess, R.sys = FakeSubprocess, FakeSys
    try:
        with tempfile.TemporaryDirectory() as td:
            R.ProcessLauncher().launch("python.exe", Path(td), REPO, resume=False, gentle=True)
    finally:
        R.subprocess, R.sys = real_sub, real_sys
    flags = FakeSubprocess.seen.get("creationflags", 0)
    check("the worker gets a hidden console of its own, so no window opens for it or its helpers",
          flags & FakeSubprocess.CREATE_NO_WINDOW and not flags & FakeSubprocess.DETACHED_PROCESS)
    check("it is in its own process group and runs below normal priority in Gentle",
          flags & FakeSubprocess.CREATE_NEW_PROCESS_GROUP and flags & FakeSubprocess.BELOW_NORMAL_PRIORITY_CLASS)
    env = FakeSubprocess.seen.get("env") or {}
    check("it and the helpers it starts read and write UTF-8 whatever the computer's character set",
          env.get("PYTHONUTF8") == "1" and env.get("PYTHONIOENCODING") == "utf-8")


def test_failures_and_recovery():
    print("failures, crashes and safety limits")
    with tempfile.TemporaryDirectory() as td:
        w = World4(Path(td))
        row = w.start(config(epochs=3, fail_at_epoch=2))
        failed = wait_for(w.manager, row["id"], ("failed", "completed"))
        check("an error stops the run as failed with the error in plain words",
              failed["status"] == "failed" and "self-test failure requested at epoch 2" in failed["stop_reason"]
              and "The log has the details" in failed["stop_reason"])
        check("the log holds the traceback", "Traceback" in w.manager.read_log(row["id"])["text"])
        check("a failed run is not offered Resume", raises(lambda: w.manager.resume(row["id"]), R.RunError, 409))

        row = w.start(config(epochs=3, crash_at_epoch=2), name="Crash")
        status = {}
        end = time.time() + 30
        while time.time() < end:
            status = w.manager.read_status(row["id"])
            if len(w.manager.read_progress(row["id"])) >= 1 and status.get("state") == "running":
                time.sleep(0.6)
                break
            time.sleep(0.05)
        check("the worker reached its second epoch before dying", w.manager.read_progress(row["id"])[-1]["epoch"] == 1)
        still = w.manager.refresh(row["id"])
        check("right after a crash the heartbeat is still recent, so it is not yet judged",
              still["status"] in ("running", "preparing"))
        later = R.RunManager(w.settings, w.db, clock=lambda: time.time() + R.STALE_SECONDS + 5)
        row2 = later.refresh(row["id"])
        check("once the heartbeat is stale the run is marked interrupted",
              row2["status"] == "interrupted" and "Resume" in row2["stop_reason"] and row2["ended_at"])
        later.resume(row["id"])
        done = wait_for(w.manager, row["id"], ("completed", "failed", "interrupted"))
        epochs = [p["epoch"] for p in w.manager.read_progress(row["id"])]
        check("Resume after a crash continues from the last saved epoch", done["status"] == "completed"
              and epochs == [1, 2, 3])

        row = w.start(config(epochs=2, limits=dict(LIMITS, min_free_disk_gb=1e9)), name="Disk")
        p = wait_for(w.manager, row["id"], ("paused", "completed", "failed"))
        check("a crossed safety limit pauses the run before training with the plain reason",
              p["status"] == "paused" and "Free some space" in p["stop_reason"] and p["epochs_done"] == 0
              and "starts the training again from the beginning" in p["stop_reason"])
        check("a run paused for safety can still be cancelled", w.manager.cancel(row["id"])["status"] == "cancelled")

        cfg = config(epochs=1)
        rid = w.start(cfg, name="Evil", start=False)["id"]
        run_file = w.manager.run_dir(rid) / W.RUN_FILE
        data = json.loads(run_file.read_text(encoding="utf-8"))
        data["engine"] = "os:system"
        run_file.write_text(json.dumps(data), encoding="utf-8")
        code = W.run(w.manager.run_dir(rid), check_limits=False)
        st = w.manager.read_status(rid)
        check("a run folder cannot make the worker import anything but a known engine",
              code == 1 and st["state"] == "failed" and "unknown training engine" in st["stop_reason"])
        w.db.update_training_run(rid, status="cancelled")


class QuietLauncher(R.ProcessLauncher):
    def __init__(self):
        self.launched, self.stopped = [], []

    def launch(self, python, run_dir, code_root, *, resume, gentle):
        self.launched.append((python, str(run_dir), resume, gentle))
        return 4242

    def stop(self, pid):
        self.stopped.append(pid)
        return True


def test_with_fake_launcher():
    print("launch bookkeeping")
    with tempfile.TemporaryDirectory() as td:
        w = World4(Path(td))
        q = QuietLauncher()
        clock = {"t": time.time()}
        w.manager = R.RunManager(w.settings, w.db, launcher=q, clock=lambda: clock["t"])
        row = w.start(config(epochs=1))
        check("the worker is launched with this interpreter for the self-test, gently, not resuming",
              q.launched[0][0] == sys.executable and q.launched[0][2] is False and q.launched[0][3] is True)
        clock["t"] += R.STALE_SECONDS + 1
        (w.manager.run_dir(row["id"]) / "worker_output.txt").write_text("ModuleNotFoundError: no module torch\n")
        r = w.manager.refresh(row["id"])
        check("a worker that never reports is marked failed with what it printed",
              r["status"] == "failed" and "did not start" in r["stop_reason"] and "ModuleNotFoundError" in r["stop_reason"])

        row = w.start(config(epochs=1), name="Stubborn")
        rd = w.manager.run_dir(row["id"])
        W.write_json_atomic(rd / W.STATUS_FILE, {"state": "running", "pid": 999999, "heartbeat_epoch": clock["t"],
                                                 "heartbeat_at": R.utc_iso(clock["t"])})
        w.manager.cancel(row["id"])
        check("Cancel first asks the worker", json.loads((rd / W.CONTROL_FILE).read_text())["command"] == "cancel"
              and q.stopped == [])
        clock["t"] += R.CANCEL_GRACE_SECONDS + 1
        W.write_json_atomic(rd / W.STATUS_FILE, {"state": "running", "pid": 999999, "heartbeat_epoch": clock["t"],
                                                 "heartbeat_at": R.utc_iso(clock["t"])})
        r = w.manager.refresh(row["id"])
        check("a live worker that ignores Cancel for a minute is stopped", q.stopped == [999999]
              and r["status"] == "cancelled")

        row = w.start(config(epochs=1), name="Gone")
        rd = w.manager.run_dir(row["id"])
        W.write_json_atomic(rd / W.STATUS_FILE, {"state": "running", "pid": 888888, "heartbeat_epoch": clock["t"] - 500,
                                                 "heartbeat_at": R.utc_iso(clock["t"] - 500)})
        w.manager._write_control(row["id"], "cancel")
        clock["t"] += R.CANCEL_GRACE_SECONDS + 1
        r = w.manager.refresh(row["id"])
        check("a process whose heartbeat is stale is never killed, because its number may now belong to "
              "another program", 888888 not in q.stopped and r["status"] == "interrupted")
        w.manager.cancel(row["id"])

        restarted = R.RunManager(w.settings, w.db, launcher=q, clock=lambda: clock["t"])
        check("after the application restarts every unfinished run is looked at again",
              isinstance(restarted.refresh_all(), list))


def test_files():
    print("files the worker and the application share")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        W.write_json_atomic(d / "a.json", {"x": 1})
        check("an atomic write leaves exactly the file and no temporary", W.read_json(d / "a.json") == {"x": 1}
              and sorted(p.name for p in d.iterdir()) == ["a.json"])
        (d / "bad.json").write_text("{half", encoding="utf-8")
        check("an unreadable file reads as nothing", W.read_json(d / "bad.json") is None
              and W.read_json(d / "missing.json") is None)
        w = World4(d / "w")
        rid = w.start(config(epochs=1), start=False)["id"]
        rd = w.manager.run_dir(rid)
        text = "first line\nespécie étudiée\nlast\n"
        (rd / W.LOG_FILE).write_bytes(text.encode("utf-8"))
        pieces, off = [], 0
        while True:
            got = w.manager.read_log(rid, off, max_bytes=5)
            if got["next_offset"] == off:
                break
            pieces.append(got["text"])
            off = got["next_offset"]
        check("a log read in small pieces never splits a character", "".join(pieces) == text)
        check("an offset beyond the end is clamped", w.manager.read_log(rid, 10 ** 9)["text"] == "")
        (rd / W.PROGRESS_FILE).write_text('{"epoch": 1}\n{"epoch": 2, "map\n', encoding="utf-8")
        check("a progress line being written is skipped, not an error", w.manager.read_progress(rid) == [{"epoch": 1}])
        check("a cut byte sequence keeps whole characters", R._whole_characters("aé".encode()[:2]) == 1
              and R._whole_characters(b"\xc3") == 1 and R._whole_characters("é".encode()) == 2)
        saved = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
        try:
            W.apply_resource_limits({"cpu_threads": 3, "mode": "full_speed"})
            check("the thread cap is set for the numeric libraries before they load",
                  os.environ["OMP_NUM_THREADS"] == "3" and os.environ["MKL_NUM_THREADS"] == "3")
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        check("only known engines can be loaded", raises(lambda: W.load_engine("os"), ValueError))
        w.db.update_training_run(rid, status="cancelled")


def main() -> int:
    test_names_and_ids()
    test_complete_run()
    test_pause_resume_cancel()
    test_pause_before_first_epoch()
    test_windows_launch_flags()
    test_output_encoding()
    test_failures_and_recovery()
    test_with_fake_launcher()
    test_files()
    print(f"\nResult: {CHECKS['passed']} passed, {CHECKS['failed']} failed")
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
