"""Optional training components: what would be installed, its real size, install and removal.

Path: audtheia/training/components.py

The training libraries are large and are never part of the everyday
application. They go, only when a person presses Install, into a separate
environment under the training environments folder (.venv-train by default),
one per engine, with its own Python interpreter. The application's own
environment is never changed, and removing the components deletes that one
folder.

Before anything is installed the planner asks pip, in dry-run mode, exactly
which files it would download for this computer, and reads each file's size from
the server, so the interface can show the real download size instead of a
guess. Dry-run mode downloads only package indexes and changes nothing.

The PyTorch build is chosen from the settings by the hardware probe: a CUDA
build for a supported NVIDIA card, the XPU build for Intel Arc graphics, ROCm for
an AMD card on Linux, and otherwise the build for the processor. A download
location missing from the settings stops the plan with a plain reason.

Standard library only.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from audtheia.training import hardware

# PyTorch and its image companion always come from the PyTorch location chosen
# for the computer. Every other package comes from the standard package index.
TORCH_PACKAGES = ("torch", "torchvision")

# How many file-size requests go to the package servers at once while planning.
SIZE_REQUESTS_AT_ONCE = 8

# Which environment folder each engine key installs into.
ENVIRONMENT_FOR_ENGINE = {"vision_detector": "vision"}

MARKER_INSTALLING = "INSTALLING"
COMPONENT_FILE = "component.json"
SELF_TEST_SNIPPET = (
    "import json, time, sys\n"
    "import torch, torchvision\n"
    "info = {'python': sys.version.split()[0], 'torch': torch.__version__, 'torchvision': torchvision.__version__}\n"
    "import importlib\n"
    "import importlib.metadata as md\n"
    "for name in sys.argv[1:]:\n"
    "    importlib.import_module(name)\n"
    "    try: info[name] = md.version(name)\n"
    "    except Exception: info[name] = 'unknown'\n"
    "dev = 'cpu'\n"
    "if torch.cuda.is_available(): dev = 'cuda'\n"
    "elif hasattr(torch, 'xpu') and torch.xpu.is_available(): dev = 'xpu'\n"
    "elif torch.backends.mps.is_available(): dev = 'mps'\n"
    "a = torch.randn(256, 256, device=dev)\n"
    "t = time.perf_counter()\n"
    "for _ in range(20): a = (a @ a).tanh()\n"
    "float(a.sum())\n"
    "info['device'] = dev\n"
    "info['probe_seconds'] = round(time.perf_counter() - t, 4)\n"
    "print('AUDTHEIA_SELFTEST ' + json.dumps(info))\n"
)


class ComponentError(ValueError):
    """A plan or install that cannot go ahead, in plain words. status is the HTTP status."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def env_python(env_dir: Path) -> Path:
    return Path(env_dir) / ("Scripts/python.exe" if sys.platform.startswith("win") else "bin/python")


def _creation_flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform.startswith("win") else 0


def run_capture(args: Sequence[str], timeout: float, pythonpath: str | None = None) -> tuple:
    """Run a command and return (return code, standard output, standard error). Never raises.

    pythonpath puts a folder first on the module search path of the command, so
    a training environment can run Audtheia's own modules without installing them.
    """
    env = None
    if pythonpath:
        env = dict(os.environ)
        env["PYTHONPATH"] = pythonpath + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    try:
        done = subprocess.run(list(args), capture_output=True, text=True, timeout=timeout,
                              creationflags=_creation_flags(), env=env)
        return done.returncode, done.stdout or "", done.stderr or ""
    except subprocess.TimeoutExpired:
        return None, "", "timed out"
    except Exception as exc:  # noqa: BLE001
        return None, "", str(exc)


def content_length(url: str, timeout: float = 20.0) -> int | None:
    """The size the server reports for a file, read with a HEAD request. None if not reported."""
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "audtheia-install-planner"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - https package hosts only
            value = resp.headers.get("Content-Length")
            return int(value) if value and value.isdigit() else None
    except Exception:  # noqa: BLE001
        return None


class Components:
    """Plans, installs, checks and removes the training environment of an engine."""

    def __init__(self, settings: Any, *, base_python: str | None = None,
                 runner: Callable[..., tuple] = run_capture, sizer: Callable[[str], int | None] = content_length,
                 popen: Callable[..., Any] = subprocess.Popen):
        self.settings = settings
        cfg = settings.training_config()
        self.install_cfg = cfg.get("install") or {}
        self.envs_dir = Path(settings.training_paths()["envs_dir"])
        self.base_python = base_python or sys.executable
        self.runner = runner
        self.sizer = sizer
        self.popen = popen
        self._lock = threading.Lock()
        self._jobs: dict = {}

    # -- where things are ----------------------------------------------------

    def environment(self, engine: str) -> str:
        if engine not in ENVIRONMENT_FOR_ENGINE:
            raise ComponentError(f"There are no installable components for the engine {engine!r}.", 404)
        return ENVIRONMENT_FOR_ENGINE[engine]

    def env_dir(self, engine: str) -> Path:
        return self.envs_dir / self.environment(engine)

    def log_path(self, engine: str) -> Path:
        return self.envs_dir / f"{self.environment(engine)}.install.log"

    def packages(self, engine: str) -> list:
        env = self.environment(engine)
        names = list((self.install_cfg.get("packages") or {}).get(env) or [])
        if not names:
            raise ComponentError(f"No packages are listed for {env} under training.install.packages.", 422)
        return names

    # -- status --------------------------------------------------------------

    def status(self, engine: str) -> dict:
        env_dir = self.env_dir(engine)
        info = {"engine": engine, "environment": self.environment(engine), "path": str(env_dir),
                "state": "not_installed", "component": None, "size_gb": None,
                "job": self.job(engine)}
        if not env_dir.exists():
            return info
        if (env_dir / MARKER_INSTALLING).exists():
            info["state"] = "installing" if info["job"] and info["job"]["state"] == "running" else "incomplete"
        elif (env_dir / COMPONENT_FILE).is_file() and env_python(env_dir).is_file():
            try:
                info["component"] = json.loads((env_dir / COMPONENT_FILE).read_text(encoding="utf-8"))
                info["state"] = "ready"
            except (OSError, json.JSONDecodeError):
                info["state"] = "incomplete"
        else:
            info["state"] = "incomplete"
        return info

    def folder_size_gb(self, engine: str) -> float | None:
        env_dir = self.env_dir(engine)
        if not env_dir.exists():
            return None
        total = 0
        for root, _dirs, files in os.walk(env_dir):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return round(total / hardware.GIB, 2)

    # -- planning ------------------------------------------------------------

    def choose(self, hw: Mapping) -> dict:
        indexes = self.install_cfg.get("torch_index_by_vendor") or {}
        return hardware.choose_torch_index(hw, indexes)

    def _dry_run(self, args: list, label: str) -> list:
        with tempfile.TemporaryDirectory(prefix="audtheia-plan-") as td:
            report = Path(td) / "report.json"
            code, _out, err = self.runner(
                [self.base_python, "-m", "pip", "install", "--dry-run", "--ignore-installed", "--quiet",
                 "--disable-pip-version-check", "--report", str(report), *args], timeout=600)
            if code != 0 or not report.is_file():
                tail = " ".join(err.strip().split())[-500:]
                raise ComponentError(f"pip could not work out what to download for {label}. "
                                     f"Check the internet connection. pip said: {tail or 'nothing'}", 502)
            data = json.loads(report.read_text(encoding="utf-8"))
        items = []
        for entry in data.get("install") or []:
            meta = entry.get("metadata") or {}
            dl = entry.get("download_info") or {}
            items.append({"name": meta.get("name"), "version": meta.get("version"), "url": dl.get("url")})
        return items

    def plan(self, engine: str, hw: Mapping) -> dict:
        """What Install would download, and its measured size. Changes nothing."""
        choice = self.choose(hw)
        if not choice.get("url"):
            raise ComponentError(choice["reason"], 422)
        packages = self.packages(engine)
        torch_items = self._dry_run(["--index-url", choice["url"], *TORCH_PACKAGES], "PyTorch")
        pins = []
        for item in torch_items:
            if item["name"] and item["name"].lower() in TORCH_PACKAGES:
                pins.append(f"{item['name']}=={item['version']}")
        if len(pins) != len(TORCH_PACKAGES):
            raise ComponentError("pip did not offer PyTorch for this computer from the chosen location.", 502)
        everything = self._dry_run(["--extra-index-url", choice["url"], *pins, *packages], "the training libraries")
        urls = [item["url"] for item in everything]
        with ThreadPoolExecutor(max_workers=SIZE_REQUESTS_AT_ONCE) as pool:
            sizes = list(pool.map(lambda u: self.sizer(u) if u else None, urls))
        files = [dict(item, size_bytes=size) for item, size in zip(everything, sizes)]
        total = sum(s for s in sizes if s is not None)
        unknown = sum(1 for s in sizes if s is None)
        return {
            "engine": engine,
            "environment": self.environment(engine),
            "path": str(self.env_dir(engine)),
            "vendor_key": choice["vendor_key"],
            "device": choice["device"],
            "reason": choice["reason"],
            "index_url": choice["url"],
            "torch_pins": pins,
            "packages": packages,
            "files": files,
            "download_bytes": total,
            "download_gb": round(total / hardware.GIB, 3),
            "files_without_size": unknown,
            "internet_needed": "Once, to download these files. Training itself works offline.",
            "installed_size_note": "The installed size is larger than the download and is measured after installing.",
        }

    # -- installing ----------------------------------------------------------

    def job(self, engine: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(engine)
            return dict(job) if job else None

    def start_install(self, engine: str, plan: Mapping, *, confirmed: bool) -> dict:
        """Install in the background. Only with confirmed True, which the interface
        sends after the person has seen the plan and pressed the download button."""
        if not confirmed:
            raise ComponentError("Installing needs your confirmation after you have seen the download size.", 400)
        if plan.get("engine") != engine or not plan.get("torch_pins") or not plan.get("index_url"):
            raise ComponentError("The install plan is missing or is for another engine. Plan again first.", 400)
        current = self.status(engine)
        if current["state"] == "ready":
            raise ComponentError("The training components are already installed.", 409)
        if current["state"] == "incomplete":
            raise ComponentError("An earlier install did not finish. Remove it first, then install again.", 409)
        with self._lock:
            job = self._jobs.get(engine)
            if job and job["state"] == "running":
                raise ComponentError("An install is already running.", 409)
            self._jobs[engine] = {"state": "running", "started_at": utc_now(), "message": "Starting.",
                                  "ended_at": None}
        thread = threading.Thread(target=self._install, args=(engine, dict(plan)), daemon=True,
                                  name=f"audtheia-install-{engine}")
        thread.start()
        return self.job(engine)

    def _set_job(self, engine: str, **fields: Any) -> None:
        with self._lock:
            self._jobs.setdefault(engine, {}).update(fields)

    def _log(self, engine: str, text: str) -> None:
        path = self.log_path(engine)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(text if text.endswith("\n") else text + "\n")

    def _stream(self, engine: str, args: list) -> int:
        self._log(engine, "$ " + " ".join(str(a) for a in args))
        proc = self.popen(list(args), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          creationflags=_creation_flags())
        for line in proc.stdout:
            self._log(engine, line.rstrip("\n"))
        return proc.wait()

    def _install(self, engine: str, plan: Mapping) -> None:
        env_dir = self.env_dir(engine)
        try:
            env_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            self._set_job(engine, state="failed", ended_at=utc_now(),
                          message="The environment folder already exists. Remove it first.")
            return
        (env_dir / MARKER_INSTALLING).write_text(utc_now(), encoding="utf-8")
        self._log(engine, f"Installing the {self.environment(engine)} training components into {env_dir}.")
        try:
            self._set_job(engine, message="Creating the separate environment.")
            if self._stream(engine, [self.base_python, "-m", "venv", str(env_dir)]) != 0:
                raise ComponentError("Python could not create the environment.", 500)
            py = str(env_python(env_dir))
            self._set_job(engine, message="Downloading and installing PyTorch.")
            if self._stream(engine, [py, "-m", "pip", "install", "--disable-pip-version-check", "--index-url",
                                     plan["index_url"], *plan["torch_pins"]]) != 0:
                raise ComponentError("PyTorch did not install. The install log has pip's explanation.", 500)
            self._set_job(engine, message="Downloading and installing the training libraries.")
            if self._stream(engine, [py, "-m", "pip", "install", "--disable-pip-version-check", "--extra-index-url",
                                     plan["index_url"], *plan["torch_pins"], *plan["packages"]]) != 0:
                raise ComponentError("The training libraries did not install. The install log has pip's explanation.",
                                     500)
            self._set_job(engine, message="Checking that everything works.")
            test = self.self_test(engine)
            if not test["ok"]:
                raise ComponentError(f"The installed libraries did not pass the check: {test['error']}", 500)
            component = {
                "engine": engine, "environment": self.environment(engine), "installed_at": utc_now(),
                "vendor_key": plan.get("vendor_key"), "index_url": plan["index_url"],
                "torch_pins": plan["torch_pins"], "packages": plan["packages"], "self_test": test["info"],
            }
            (env_dir / COMPONENT_FILE).write_text(json.dumps(component, indent=1, sort_keys=True), encoding="utf-8")
            (env_dir / MARKER_INSTALLING).unlink()
            self._log(engine, "Installed and checked.")
            self._set_job(engine, state="completed", ended_at=utc_now(), message="Installed and checked.")
        except Exception as exc:  # noqa: BLE001
            self._log(engine, f"Stopped: {exc}")
            self._set_job(engine, state="failed", ended_at=utc_now(), message=str(exc))

    def self_test(self, engine: str) -> dict:
        """Import the libraries in the environment and run a one-second calculation."""
        py = env_python(self.env_dir(engine))
        extra = [p.split("[")[0].split("=")[0].replace("-", "_") for p in self.packages(engine)]
        code, out, err = self.runner([str(py), "-c", SELF_TEST_SNIPPET, *extra], timeout=300)
        for line in out.splitlines():
            if line.startswith("AUDTHEIA_SELFTEST "):
                try:
                    return {"ok": code == 0, "info": json.loads(line.split(" ", 1)[1]), "error": None}
                except json.JSONDecodeError:
                    break
        return {"ok": False, "info": None, "error": " ".join((err or out).strip().split())[-400:] or "no output"}

    # -- removing ------------------------------------------------------------

    def remove(self, engine: str, *, run_active: bool = False) -> dict:
        """Delete the environment folder of an engine, and nothing else."""
        if run_active:
            raise ComponentError("A training run is using the components. Pause or cancel it first.", 409)
        job = self.job(engine)
        if job and job["state"] == "running":
            raise ComponentError("An install is running. Wait for it to finish first.", 409)
        env_dir = self.env_dir(engine).resolve()
        root = self.envs_dir.resolve()
        if env_dir.parent != root:
            raise ComponentError("The environment is not inside the training environments folder.", 400)
        if not env_dir.exists():
            return {"removed": False, "path": str(env_dir)}
        if not (env_dir / "pyvenv.cfg").is_file() and not (env_dir / MARKER_INSTALLING).is_file():
            raise ComponentError("That folder is not a training environment, so it was left alone.", 409)
        def _clear_read_only(func, path, _exc):
            os.chmod(path, 0o700)
            func(path)

        if sys.version_info >= (3, 12):
            shutil.rmtree(env_dir, onexc=_clear_read_only)
        else:
            shutil.rmtree(env_dir, onerror=_clear_read_only)
        with self._lock:
            self._jobs.pop(engine, None)
        return {"removed": True, "path": str(env_dir)}


def wait_for_job(components: Components, engine: str, timeout: float) -> dict | None:
    """Block until an install finishes. Used by tests and scripts."""
    end = time.time() + timeout
    while time.time() < end:
        job = components.job(engine)
        if job and job["state"] != "running":
            return job
        time.sleep(0.1)
    return components.job(engine)
