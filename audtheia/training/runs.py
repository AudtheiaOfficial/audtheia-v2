"""Starting, watching and controlling training runs, from the application side.

Path: audtheia/training/runs.py

A run is a folder under the trained-models directory, a row in training_runs
and, while it trains, a separate worker process (see worker.py). This module
creates the folder and the row, starts the worker detached from the
application so that closing the window does not stop a long run, and turns what
the worker writes into the database record the interface shows.

The worker never touches the database; only this module does, and only the
training_runs table. Pause and Cancel are requests written into the run folder.
A run whose heartbeat stops is marked interrupted the next time it is looked at,
and Resume continues it from its last saved epoch.

Only one run may train at a time, so a laptop is never asked to do two at once.

Standard library only.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from audtheia.training import worker as W

CODE_ROOT = Path(__file__).resolve().parents[2]   # the folder that holds the audtheia package

STALE_SECONDS = 120.0          # no heartbeat for this long means the worker has gone
CANCEL_GRACE_SECONDS = 60.0    # a worker that ignores Cancel this long is stopped
LOG_CHUNK_BYTES = 65536

ACTIVE = ("queued", "preparing", "running")
RESUMABLE = ("paused", "interrupted")
FINAL = ("completed", "cancelled")

NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
RESERVED_NAMES = {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(1, 10)} | {f"lpt{i}" for i in range(1, 10)}
RUN_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class RunError(ValueError):
    """A run request that cannot be carried out. status is the HTTP status to answer with."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def validate_user_name(name: Any) -> str:
    """The word a person chose for their model, checked so it is a safe folder name everywhere."""
    text = str(name or "").strip()
    if not NAME_RE.match(text):
        raise RunError("A model name uses 1 to 40 letters, digits, hyphens or underscores, with no spaces.", 422)
    if text.lower() in RESERVED_NAMES:
        raise RunError(f"{text} is a name Windows reserves for devices. Please choose another.", 422)
    return text


def planned_model_name(user_name: str, version: int) -> str:
    return f"Audtheia-{user_name}-V{int(version)}"


def _whole_characters(chunk: bytes) -> int:
    """Length of the longest start of chunk that does not end inside a UTF-8 character.

    A log read in pieces must not split a character such as an accented species
    name across two reads. If the piece holds nothing but the start of one
    character, all of it is returned so a reader can never stall.
    """
    n = len(chunk)
    i, seen = n - 1, 0
    while i >= 0 and seen < 3 and (chunk[i] & 0xC0) == 0x80:
        i -= 1
        seen += 1
    if i < 0 or chunk[i] < 0x80:
        return n
    lead = chunk[i]
    need = 2 if lead < 0xE0 else 3 if lead < 0xF0 else 4
    keep = n if n - i >= need else i
    return keep if keep > 0 else n


def utc_iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ProcessLauncher:
    """Starts a worker detached from the application. Tests replace it."""

    def launch(self, python: str, run_dir: Path, code_root: Path, *, resume: bool, gentle: bool) -> int:
        args = [str(python), "-m", "audtheia.training.worker", str(run_dir)]
        if resume:
            args.append("--resume")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(code_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["PYTHONUNBUFFERED"] = "1"
        # UTF-8 everywhere in the training process and the helpers it starts, so
        # printed figures and species names with accents are written and read
        # the same on every computer.
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        out = open(Path(run_dir) / "worker_output.txt", "ab")  # noqa: SIM115 - handed to the child
        kwargs: dict = {"cwd": str(code_root), "env": env, "stdout": out, "stderr": subprocess.STDOUT,
                        "stdin": subprocess.DEVNULL}
        if sys.platform.startswith("win"):
            flags = 0
            # CREATE_NO_WINDOW gives the worker a console of its own that is never
            # shown, so it outlives the application and the helper processes it
            # starts for loading images inherit that hidden console. With
            # DETACHED_PROCESS the worker had no console, and Windows opened a
            # visible window for each helper.
            for name in ("CREATE_NEW_PROCESS_GROUP", "CREATE_NO_WINDOW"):
                flags |= getattr(subprocess, name, 0)
            if gentle:
                flags |= getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
            kwargs["creationflags"] = flags
        else:
            kwargs["start_new_session"] = True
        try:
            proc = subprocess.Popen(args, **kwargs)  # noqa: S603 - fixed module entry point
        finally:
            out.close()
        return proc.pid

    def stop(self, pid: int) -> bool:
        """Forcefully end a worker that ignored Cancel. True if a signal was sent."""
        try:
            if sys.platform.startswith("win"):
                done = subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", "/F"], capture_output=True,
                                      timeout=15, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                return done.returncode == 0
            os.kill(int(pid), signal.SIGTERM)
            return True
        except Exception:  # noqa: BLE001
            return False


class RunManager:
    """The application's view of training runs."""

    def __init__(self, settings: Any, database: Any, *, launcher: ProcessLauncher | None = None,
                 clock: Callable[[], float] = time.time, created_by: str = "expert"):
        self.settings = settings
        self.db = database
        self.launcher = launcher or ProcessLauncher()
        self.clock = clock
        self.created_by = created_by
        paths = settings.training_paths()
        self.models_dir = Path(paths["models_dir"])
        self.envs_dir = Path(paths["envs_dir"])
        self.runs_root = self.models_dir / "runs"
        self.repo_root = Path(getattr(settings, "repo_root", CODE_ROOT))

    # -- locations ---------------------------------------------------------

    def run_dir(self, run_id: str) -> Path:
        if not RUN_ID_RE.match(str(run_id)):
            raise RunError("That is not a run identifier.", 404)
        return self.runs_root / run_id

    def python_for(self, environment: str | None) -> str:
        """The interpreter that runs an engine: the training environment's, or this one for the self-test."""
        if environment is None:
            return sys.executable
        env_dir = self.envs_dir / environment
        exe = env_dir / ("Scripts/python.exe" if sys.platform.startswith("win") else "bin/python")
        if not exe.is_file():
            raise RunError("The training components are not installed yet. Install them from the Train view first.",
                           409)
        return str(exe)

    def _relative(self, path: Path) -> str:
        try:
            return Path(path).resolve().relative_to(self.repo_root.resolve()).as_posix()
        except ValueError:
            return Path(path).resolve().as_posix()

    # -- reading the worker's files ---------------------------------------

    def read_status(self, run_id: str) -> dict:
        return W.read_json(self.run_dir(run_id) / W.STATUS_FILE) or {}

    def read_progress(self, run_id: str) -> list:
        path = self.run_dir(run_id) / W.PROGRESS_FILE
        rows = []
        try:
            text = path.read_text(encoding="utf-8")
        except (FileNotFoundError, PermissionError, OSError):
            return rows
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue          # a line being written at this instant is read next time
        return rows

    def read_log(self, run_id: str, offset: int = 0, max_bytes: int = LOG_CHUNK_BYTES) -> dict:
        path = self.run_dir(run_id) / W.LOG_FILE
        try:
            size = path.stat().st_size
        except OSError:
            return {"text": "", "next_offset": 0, "size": 0}
        offset = max(0, min(int(offset), size))
        with open(path, "rb") as handle:
            handle.seek(offset)
            chunk = handle.read(max(1, int(max_bytes)))
        cut = _whole_characters(chunk)
        text = chunk[:cut].decode("utf-8", errors="replace")
        return {"text": text, "next_offset": offset + cut, "size": size}

    def _write_control(self, run_id: str, command: str) -> None:
        W.write_json_atomic(self.run_dir(run_id) / W.CONTROL_FILE,
                            {"command": command, "issued_at": utc_iso(self.clock()), "issued_epoch": self.clock()})

    # -- record keeping ----------------------------------------------------

    def _alive(self, status: Mapping) -> bool:
        beat = status.get("heartbeat_epoch")
        return isinstance(beat, (int, float)) and (self.clock() - float(beat)) < STALE_SECONDS

    def refresh(self, run_id: str) -> dict:
        """Bring the database row up to date with the run folder and return it."""
        row = self.db.get_training_run(run_id)
        if row is None:
            raise RunError("There is no training run with that identifier.", 404)
        if row["status"] in FINAL:
            return row
        status = self.read_status(run_id)
        state = status.get("state")
        updates: dict = {}
        for key in ("epochs_total", "epochs_done", "best_metric_name", "best_metric_value", "started_at", "ended_at"):
            if status.get(key) is not None and status.get(key) != row.get(key):
                updates[key] = status[key]
        if status.get("heartbeat_at") and status.get("heartbeat_at") != row.get("last_heartbeat_at"):
            updates["last_heartbeat_at"] = status["heartbeat_at"]
        if status.get("software") and not row.get("software_json"):
            updates["software_json"] = json.dumps(status["software"], sort_keys=True)
        alive = self._alive(status)
        new_status = row["status"]
        reason = row.get("stop_reason")
        if state in ("completed", "cancelled", "failed", "paused"):
            new_status = state
            reason = status.get("stop_reason") if state != "completed" else None
        elif state in ("preparing", "running"):
            if alive:
                new_status = state
                reason = None
            elif status.get("pid") is None:
                new_status = "failed"
                tail = self._worker_output_tail(run_id)
                reason = ("The training process did not start." + (f" It said: {tail}" if tail else ""))
                if row["status"] != "failed":
                    updates["ended_at"] = utc_iso(self.clock())
            else:
                new_status = "interrupted"
                reason = ("The training stopped unexpectedly, for example because the computer was switched off "
                          "or went to sleep. Your progress up to the last saved epoch is kept; press Resume.")
                if row["status"] != "interrupted":
                    updates["ended_at"] = utc_iso(self.clock())
        elif row["status"] in ("queued", "preparing"):
            started = row.get("last_heartbeat_at") or row.get("created_at")
            if self._older_than(started, STALE_SECONDS):
                new_status = "failed"
                tail = self._worker_output_tail(run_id)
                reason = ("The training process did not start." + (f" It said: {tail}" if tail else ""))
        if new_status != row["status"]:
            updates["status"] = new_status
        if reason != row.get("stop_reason"):
            updates["stop_reason"] = reason
        # a worker that ignores Cancel is stopped, but only while its heartbeat proves it is ours and alive
        control = W.read_json(self.run_dir(run_id) / W.CONTROL_FILE) or {}
        if control.get("command") == "cancel" and alive and new_status in ACTIVE \
                and self.clock() - float(control.get("issued_epoch") or self.clock()) > CANCEL_GRACE_SECONDS:
            if status.get("pid") and self.launcher.stop(int(status["pid"])):
                updates["status"] = "cancelled"
                updates["stop_reason"] = "Cancelled at your request. The training process was stopped."
                updates["ended_at"] = utc_iso(self.clock())
        if updates:
            row = self.db.update_training_run(run_id, **updates)
        return row

    def _older_than(self, iso: str | None, seconds: float) -> bool:
        if not iso:
            return True
        try:
            when = datetime.strptime(iso[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            return True
        return self.clock() - when > seconds

    def _worker_output_tail(self, run_id: str, chars: int = 400) -> str:
        try:
            text = (self.run_dir(run_id) / "worker_output.txt").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return " ".join(text.strip().split())[-chars:]

    def refresh_all(self) -> list:
        """Called when the application starts: re-attach to every run that was not finished."""
        out = []
        for row in self.db.list_training_runs():
            if row["status"] in ACTIVE or row["status"] == "paused":
                try:
                    out.append(self.refresh(row["id"]))
                except RunError:
                    continue
        return out

    def active_run(self) -> dict | None:
        for row in self.db.list_training_runs():
            if row["status"] in ACTIVE:
                row = self.refresh(row["id"])
                if row["status"] in ACTIVE:
                    return row
        return None

    # -- actions -------------------------------------------------------------

    def create(self, *, user_name: str, modality: str, engine: str, mode: str, dataset: Mapping,
               base_label: str, config: Mapping, environment: str | None, hardware_info: Mapping | None = None,
               parent_model_id: str | None = None, start: bool = True) -> dict:
        """Record a run, write its folder and, unless start is False, launch it.

        config is the engine's complete configuration and must include epochs,
        seed, resources and hardware limits. dataset is the summary of a built
        dataset with its id, signature and path.
        """
        name = validate_user_name(user_name)
        if mode not in ("full_speed", "gentle"):
            raise RunError("The mode is Full speed or Gentle.", 422)
        if self.active_run() is not None:
            raise RunError("A training run is already in progress. Pause or cancel it before starting another.", 409)
        python = self.python_for(environment)
        version = self.db.next_model_version(name)
        run_id = str(uuid.uuid4())
        run_dir = self.run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=False)
        full_config = dict(config)
        full_config.update({"run_id": run_id, "engine": engine, "mode": mode, "modality": modality,
                            "dataset_dir": str(Path(dataset["path"]).resolve()),
                            "dataset_signature": dataset["signature"], "environment": environment})
        W.write_json_atomic(run_dir / W.RUN_FILE, full_config)
        W.write_json_atomic(run_dir / W.CONTROL_FILE, {"command": "run", "issued_at": utc_iso(self.clock()),
                                                       "issued_epoch": self.clock()})
        from audtheia.storage.database import TrainingRun  # noqa: PLC0415

        now = utc_iso(self.clock())
        run = TrainingRun(
            id=run_id, run_name=name, planned_model_name=planned_model_name(name, version), modality=modality,
            engine=engine, mode=mode, status="queued", dataset_signature=str(dataset["signature"]),
            dataset_dir=self._relative(Path(dataset["path"])), run_dir=self._relative(run_dir),
            seed=int(full_config.get("seed", 0)), base_label=str(base_label),
            config_json=json.dumps(full_config, sort_keys=True), created_by=self.created_by, created_at=now,
            parent_model_id=parent_model_id, epochs_total=full_config.get("epochs"),
            hardware_json=json.dumps(hardware_info, sort_keys=True) if hardware_info else None,
        )
        self.db.create_training_run(run)
        if start:
            self._launch(run_id, python, resume=False, gentle=mode == "gentle")
        return self.db.get_training_run(run_id)

    def _launch(self, run_id: str, python: str, *, resume: bool, gentle: bool) -> None:
        # The run folder still holds the last worker's status, for example
        # "paused". It is reset first, so the run reads as starting until the new
        # worker reports, and a worker that never starts is recognised as such.
        status_path = self.run_dir(run_id) / W.STATUS_FILE
        previous = W.read_json(status_path) or {}
        now = self.clock()
        previous.update({"state": "preparing", "pid": None, "heartbeat_epoch": now, "heartbeat_at": utc_iso(now),
                         "stop_code": None, "stop_reason": None})
        W.write_json_atomic(status_path, previous)
        try:
            self.launcher.launch(python, self.run_dir(run_id), CODE_ROOT, resume=resume, gentle=gentle)
        except Exception as exc:  # noqa: BLE001
            self.db.update_training_run(run_id, status="failed",
                                        stop_reason=f"The training process could not be started: {exc}")
            raise RunError(f"The training process could not be started: {exc}", 500) from exc
        self.db.update_training_run(run_id, status="preparing", stop_reason=None,
                                    last_heartbeat_at=utc_iso(self.clock()))

    def pause(self, run_id: str) -> dict:
        row = self.refresh(run_id)
        if row["status"] not in ACTIVE:
            raise RunError(f"The run is {row['status']}, so it cannot be paused.", 409)
        self._write_control(run_id, "pause")
        return row

    def resume(self, run_id: str) -> dict:
        row = self.refresh(run_id)
        if row["status"] not in RESUMABLE:
            raise RunError(f"The run is {row['status']}, so it cannot be resumed.", 409)
        if self.active_run() is not None:
            raise RunError("Another training run is in progress. Pause or cancel it first.", 409)
        config = W.read_json(self.run_dir(run_id) / W.RUN_FILE) or {}
        python = self.python_for(config.get("environment"))
        self._write_control(run_id, "run")
        self._launch(run_id, python, resume=True, gentle=row["mode"] == "gentle")
        return self.db.get_training_run(run_id)

    def cancel(self, run_id: str) -> dict:
        row = self.refresh(run_id)
        if row["status"] in FINAL:
            raise RunError(f"The run is already {row['status']}.", 409)
        self._write_control(run_id, "cancel")
        if row["status"] not in ACTIVE:
            row = self.db.update_training_run(run_id, status="cancelled", stop_reason="Cancelled at your request.",
                                              ended_at=utc_iso(self.clock()))
        return row

    def detail(self, run_id: str) -> dict:
        row = self.refresh(run_id)
        status = self.read_status(run_id)
        return {"run": row, "progress": self.read_progress(run_id),
                "live": self.live(run_id, status)}

    LIVE_FIELDS = ("state", "epochs_done", "epochs_total", "best_metric_name", "best_metric_value", "heartbeat_at",
                   "stop_code", "current_epoch", "batch_done", "batches_total", "seconds_per_step", "stage")

    def live(self, run_id: str, status: Mapping | None = None) -> dict:
        """What the worker last reported, including progress within the current epoch."""
        status = self.read_status(run_id) if status is None else status
        return {k: status.get(k) for k in self.LIVE_FIELDS}
