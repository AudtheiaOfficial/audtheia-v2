"""The training worker: one process per run, separate from the application.

Path: audtheia/training/worker.py

Started by the application as

    <training environment python> -m audtheia.training.worker <run folder> [--resume]

It reads the run configuration the application wrote into the run folder, runs
the training engine, and reports only through files in that same folder:

    run.json        the configuration, written once by the application
    control.json    written by the application: run, pause or cancel
    status.json     written here: state, epoch, best figure, heartbeat
    progress.jsonl  written here: one line of figures per finished epoch
    log.txt         written here: what happened, in plain words
    checkpoints/    written by the engine, to resume from

The worker never opens the database and never writes outside its run folder, so
stopping it at any instant, by Pause, Cancel, a crash or a power cut, cannot
damage the record or the data folder. A pause saves progress and ends the
process; Resume starts a new process from the last saved epoch.

While it runs it checks, about every half minute, that the charger is still
connected, that the battery and the free disk are above their limits and that
the computer is not slowing itself down because of heat, and pauses with a plain
reason when one of them fails. It writes a heartbeat every few seconds so the
application can tell a run that is working from one whose process has gone.

Standard library only, apart from the engine it is asked to run.
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from audtheia.training import hardware

RUN_FILE = "run.json"
CONTROL_FILE = "control.json"
STATUS_FILE = "status.json"
PROGRESS_FILE = "progress.jsonl"
LOG_FILE = "log.txt"
CHECKPOINT_DIR = "checkpoints"

HEARTBEAT_SECONDS = 5.0          # how often the heartbeat is refreshed
LIMIT_CHECK_SECONDS = 30.0       # how often power, disk and heat are read
BATCH_NOTE_SECONDS = 3.0         # how often progress within an epoch is noted

# Engine keys the worker knows how to start. A run configuration can only name
# one of these, so a run folder can never make the worker import anything else.
ENGINES = {
    "selftest": "audtheia.training.worker:SelfTestEngine",
    "vision_detector": "audtheia.training.vision:VisionEngine",
}

STATES = ("preparing", "running", "paused", "completed", "failed", "cancelled")


class PauseRequested(Exception):
    """Stop now with progress saved. code is a short reason key, reason is plain words."""

    def __init__(self, code: str, reason: str):
        super().__init__(reason)
        self.code = code
        self.reason = reason


class CancelRequested(Exception):
    """Stop now and keep nothing to resume from."""


def utc_iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json_atomic(path: Path, data: Any, attempts: int = 40) -> None:
    """Write JSON so a reader never sees half a file.

    The text goes to a temporary file in the same folder, which then replaces the
    target in one step. On Windows that replacement fails while another process
    has the target open, so it is retried for up to about two seconds.
    """
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
    for i in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.05)


def read_json(path: Path, attempts: int = 5) -> Any:
    """Read a JSON file written by write_json_atomic. None if it is missing or unreadable."""
    for i in range(attempts):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (PermissionError, json.JSONDecodeError, UnicodeDecodeError):
            if i == attempts - 1:
                return None
            time.sleep(0.05)
        except OSError:
            return None
    return None


class RunContext:
    """What an engine uses to report and to obey. Every write stays in the run folder."""

    def __init__(self, run_dir: Path, config: Mapping, *, resume: bool = False,
                 clock: Callable[[], float] = time.time, monotonic: Callable[[], float] = time.monotonic,
                 sources: hardware.SystemSources | None = None, check_limits: bool = True):
        self.run_dir = Path(run_dir)
        self.config = config
        self.resume = resume
        self.clock = clock
        self.monotonic = monotonic
        self.sources = sources
        self.limits = dict((config.get("hardware") or {}).get("limits") or {})
        self.device = config.get("device", "cpu")
        self.check_limits_enabled = check_limits
        self.checkpoint_dir = self.run_dir / CHECKPOINT_DIR
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._status: dict = read_json(self.run_dir / STATUS_FILE) or {}
        self._lock = threading.Lock()
        self._last_limit_check = -1e18
        self._thermal = hardware.ThermalWatch()
        self._stop_heartbeat = threading.Event()
        self._thread: threading.Thread | None = None

    # -- writing -----------------------------------------------------------

    def log(self, message: str) -> None:
        line = f"{utc_iso(self.clock())}  {message}\n"
        with self._lock:
            with open(self.run_dir / LOG_FILE, "a", encoding="utf-8") as handle:
                handle.write(line)

    def set_state(self, state: str, **fields: Any) -> None:
        if state not in STATES:
            raise ValueError(f"unknown state {state}")
        with self._lock:
            self._status.update(fields)
            self._status["state"] = state
            self._status["pid"] = os.getpid()
            self._beat_locked()

    def update(self, **fields: Any) -> None:
        with self._lock:
            self._status.update(fields)
            self._beat_locked()

    def _beat_locked(self) -> None:
        now = self.clock()
        self._status["heartbeat_epoch"] = now
        self._status["heartbeat_at"] = utc_iso(now)
        write_json_atomic(self.run_dir / STATUS_FILE, self._status)

    def heartbeat(self) -> None:
        with self._lock:
            self._beat_locked()

    def start_heartbeat(self) -> None:
        def loop() -> None:
            while not self._stop_heartbeat.wait(HEARTBEAT_SECONDS):
                try:
                    self.heartbeat()
                except Exception:  # noqa: BLE001
                    pass

        self._thread = threading.Thread(target=loop, name="audtheia-heartbeat", daemon=True)
        self._thread.start()

    def stop_heartbeat(self) -> None:
        self._stop_heartbeat.set()
        if self._thread is not None:
            self._thread.join(timeout=HEARTBEAT_SECONDS + 1)

    @property
    def status(self) -> dict:
        with self._lock:
            return dict(self._status)

    # -- obeying -----------------------------------------------------------

    def command(self) -> str:
        control = read_json(self.run_dir / CONTROL_FILE) or {}
        return str(control.get("command") or "run")

    def step(self) -> None:
        """Call often. Raises PauseRequested or CancelRequested when the run must stop."""
        cmd = self.command()
        if cmd == "cancel":
            raise CancelRequested()
        if cmd == "pause":
            raise PauseRequested("paused", "Paused at your request.")
        if not self.check_limits_enabled:
            return
        now = self.monotonic()
        if now - self._last_limit_check < LIMIT_CHECK_SECONDS:
            return
        self._last_limit_check = now
        self.check_safety(starting=False)

    @property
    def epochs_done(self) -> int:
        """Epochs finished and saved so far, including those before a pause."""
        return int(self.status.get("epochs_done") or 0)

    def resume_note(self) -> str:
        """What Resume will do, said when a run pauses. Training is saved at the
        end of each epoch, so the epoch in progress is done again."""
        done = self.epochs_done
        if done == 0:
            return "No epoch had finished yet, so Resume starts the training again from the beginning."
        return (f"Resume continues from the end of epoch {done}, the last one saved, "
                "and repeats the epoch that was in progress.")

    def nothing_to_resume(self) -> None:
        """For an engine asked to resume that finds no saved epoch. Before the
        first epoch finishes there is nothing to lose, so it starts again; after
        that a missing checkpoint is an error, because the finished epochs are
        already recorded and would be counted twice."""
        if self.epochs_done:
            raise RuntimeError(f"epoch {self.epochs_done} was recorded but its saved checkpoint is missing, "
                               "so the run cannot continue")
        self.log("No epoch had finished before the pause, so the training starts again from the first epoch.")

    def check_safety(self, *, starting: bool) -> list:
        """Read power, disk and heat and pause if a limit is crossed. Returns the warnings."""
        state = hardware.live_state(self.run_dir, sources=self.sources, has_nvidia=self.device == "cuda")
        seconds = self._thermal.update(state["thermal"]["sample"], self.monotonic())
        gpu_used = hardware.gpu_memory_used_fraction(self.sources) if self.device == "cuda" else None
        verdict = hardware.check_limits(state, self.limits, starting=starting, thermal_seconds=seconds,
                                        gpu_memory_used_fraction=gpu_used)
        if verdict["stop"]:
            first = verdict["stop"][0]
            raise PauseRequested(first["code"], first["message"])
        return verdict["warnings"]

    def batch(self, epoch: int, done: int, total: int | None) -> None:
        """Note how far through the current epoch the run is, at most every few seconds,
        with the measured seconds per step so the interface can say how long is left."""
        now = self.monotonic()
        if getattr(self, "_speed_epoch", None) != epoch:
            self._speed_epoch, self._speed_start, self._speed_first = epoch, now, int(done)
        if done != total and now - getattr(self, "_last_batch_note", -1e18) < BATCH_NOTE_SECONDS:
            return
        self._last_batch_note = now
        steps = int(done) - self._speed_first
        per_step = (now - self._speed_start) / steps if steps > 0 else None
        fields: dict = {"current_epoch": int(epoch), "batch_done": int(done),
                        "batches_total": int(total) if total else None}
        if per_step is not None:
            fields["seconds_per_step"] = round(per_step, 3)
        self.update(**fields)

    def end_epoch(self, epoch: int, figures: Mapping[str, Any], *, best: tuple | None = None) -> None:
        """Record one finished epoch, then check whether to carry on."""
        self.record_epoch(epoch, figures, best=best)
        self.step()

    def record_epoch(self, epoch: int, figures: Mapping[str, Any], *, best: tuple | None = None) -> None:
        """Record one finished epoch without checking control. For engines whose
        checkpoint for the epoch is written after they report it."""
        row = {"epoch": int(epoch), "at": utc_iso(self.clock())}
        for key, value in figures.items():
            row[key] = value if isinstance(value, (int, float, str, type(None))) else str(value)
        with self._lock:
            with open(self.run_dir / PROGRESS_FILE, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        fields: dict = {"epochs_done": int(epoch)}
        if best is not None:
            fields["best_metric_name"], fields["best_metric_value"] = best
        self.update(**fields)


# ---------------------------------------------------------------------------
# A tiny engine that needs nothing installed
# ---------------------------------------------------------------------------


class SelfTestEngine:
    """Imitates training without any machine-learning library.

    Used to prove pause, resume, cancel, the safety checks and crash recovery,
    and as a quick check that run control works on a computer. Each epoch takes
    a set number of short steps, writes a checkpoint and a line of figures.
    """

    def __init__(self, config: Mapping):
        opts = dict(config.get("engine_options") or {})
        self.epochs = int(config.get("epochs", 3))
        self.steps = int(opts.get("steps_per_epoch", 5))
        self.step_seconds = float(opts.get("step_seconds", 0.01))
        self.fail_at_epoch = opts.get("fail_at_epoch")
        self.crash_at_epoch = opts.get("crash_at_epoch")
        self.crash_only_first_time = bool(opts.get("crash_only_first_time", True))
        self.print_text = opts.get("print_text")

    def run(self, ctx: RunContext) -> dict:
        start = 1
        ckpt = ctx.checkpoint_dir / "last.json"
        if ctx.resume:
            saved = read_json(ckpt)
            if saved is None:
                ctx.nothing_to_resume()
            else:
                start = int(saved["epoch"]) + 1
                ctx.log(f"Resuming after epoch {saved['epoch']}.")
        ctx.set_state("running", epochs_total=self.epochs)
        best = None
        for epoch in range(start, self.epochs + 1):
            for k in range(self.steps):
                time.sleep(self.step_seconds)
                ctx.batch(epoch, k + 1, self.steps)
                ctx.step()
            if self.fail_at_epoch is not None and epoch == int(self.fail_at_epoch):
                raise RuntimeError(f"self-test failure requested at epoch {epoch}")
            if self.crash_at_epoch is not None and epoch == int(self.crash_at_epoch) \
                    and not (self.crash_only_first_time and ctx.resume):
                os._exit(3)                                   # imitate a power cut: no clean-up at all
            if self.print_text:
                print(self.print_text, flush=True)          # as a library prints its table of figures
            score = round(epoch / self.epochs, 6)
            write_json_atomic(ckpt, {"epoch": epoch, "score": score})
            best = ("map", score)
            ctx.end_epoch(epoch, {"train_loss": round(1.0 / (epoch + 1), 6), "map50_95": score}, best=best)
        return {"best_metric_name": best[0] if best else None, "best_metric_value": best[1] if best else None}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def load_engine(key: str):
    if key not in ENGINES:
        raise ValueError(f"unknown training engine {key!r}")
    module_name, class_name = ENGINES[key].split(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def use_spawned_loaders() -> None:
    """Start data-loading helper processes fresh instead of copying this one.

    The worker runs a heartbeat thread. Copying a process that has threads, which
    is what Linux does by default, can leave the copy waiting forever on a lock
    the thread held at that instant. Starting helpers fresh is what Windows and
    macOS always do, so every platform behaves the same.
    """
    import multiprocessing  # noqa: PLC0415

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except (RuntimeError, ValueError):
        pass


def apply_resource_limits(plan: Mapping) -> None:
    """Cap processor threads before any numeric library loads, and lower the
    priority in Gentle so the computer stays responsive."""
    threads = plan.get("cpu_threads")
    if threads:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS"):
            os.environ[var] = str(int(threads))
    if plan.get("mode") == "gentle":
        try:
            if sys.platform.startswith("win"):
                import ctypes  # noqa: PLC0415
                below_normal = 0x00004000
                ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), below_normal)
            else:
                os.nice(10)
        except Exception:  # noqa: BLE001
            pass


def software_versions() -> dict:
    """The versions of the libraries this run actually loaded, from their package metadata."""
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    out = {"python": sys.version.split()[0], "platform": sys.platform}
    for module, dist in (("numpy", "numpy"), ("torch", "torch"), ("torchvision", "torchvision"),
                         ("rfdetr", "rfdetr"), ("onnx", "onnx"), ("onnxruntime", "onnxruntime"),
                         ("pytorch_lightning", "pytorch-lightning")):
        if module in sys.modules:
            try:
                out[module] = version(dist)
            except PackageNotFoundError:
                out[module] = "unknown"
    return out


def run(run_dir: Path, *, resume: bool = False, check_limits: bool = True,
        sources: hardware.SystemSources | None = None) -> int:
    run_dir = Path(run_dir).resolve()
    config = read_json(run_dir / RUN_FILE)
    if not isinstance(config, dict):
        print(f"no readable {RUN_FILE} in {run_dir}", file=sys.stderr)
        return 2
    apply_resource_limits(config.get("resources") or {})
    use_spawned_loaders()
    ctx = RunContext(run_dir, config, resume=resume, sources=sources, check_limits=check_limits)
    started = ctx.clock()
    fields = {"epochs_total": config.get("epochs"), "stop_code": None, "stop_reason": None, "ended_at": None}
    if not ctx.status.get("started_at"):
        fields["started_at"] = utc_iso(started)
    ctx.set_state("preparing", **fields)
    ctx.log("Resuming the run." if resume else "Starting the run.")
    ctx.start_heartbeat()
    code = 0
    try:
        if check_limits:
            for warning in ctx.check_safety(starting=True):
                ctx.log(f"Note: {warning['message']}")
        engine_cls = load_engine(str(config.get("engine")))
        engine = engine_cls(config)
        result = engine.run(ctx) or {}
        ctx.update(software=software_versions())
        ctx.set_state("completed", ended_at=utc_iso(ctx.clock()), result=result)
        ctx.log("The run finished.")
    except PauseRequested as stop:
        reason = f"{stop.reason} {ctx.resume_note()}"
        ctx.set_state("paused", stop_code=stop.code, stop_reason=reason, software=software_versions())
        ctx.log(reason)
    except CancelRequested:
        ctx.set_state("cancelled", ended_at=utc_iso(ctx.clock()), stop_code="cancelled",
                      stop_reason="Cancelled at your request.")
        ctx.log("Cancelled at your request.")
    except Exception as exc:  # noqa: BLE001
        reason = (f"The training stopped because of an error: {type(exc).__name__}: {exc}. "
                  "The log has the details.")
        ctx.log("Error:\n" + traceback.format_exc())
        try:
            ctx.set_state("failed", ended_at=utc_iso(ctx.clock()), stop_code="error", stop_reason=reason,
                          software=software_versions())
        except Exception:  # noqa: BLE001
            pass
        code = 1
    finally:
        ctx.stop_heartbeat()
    return code


def use_utf8_output() -> None:
    """Send this process's printed output to its log as UTF-8, whatever the
    computer's own character set.

    On Windows the output file is written in the computer's legacy character
    set, which has no box-drawing characters, so the training library's table of
    figures stopped the run with a UnicodeEncodeError. Each stream is replaced by
    a UTF-8 writer on a copy of the same file, which also keeps that library from
    drawing through the Windows console instead of the file. A character that
    still cannot be written is replaced, never raised.
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        try:
            stream.flush()
            target = os.dup(stream.fileno())
        except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
            continue
        setattr(sys, name, io.TextIOWrapper(io.FileIO(target, "w", closefd=True), encoding="utf-8",
                                            errors="replace", line_buffering=True, write_through=True))


def main(argv: list | None = None) -> int:
    use_utf8_output()
    parser = argparse.ArgumentParser(prog="python -m audtheia.training.worker")
    parser.add_argument("run_dir")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    return run(Path(args.run_dir), resume=args.resume)


if __name__ == "__main__":
    sys.exit(main())
