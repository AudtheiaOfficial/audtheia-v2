"""The Prepare step that sets up BioCLIP 2 as a desktop verifier.

Path: audtheia/app/bioclip_prepare.py

BioCLIP 2 is published for PyTorch only. The desktop runs ONNX Runtime and does
not carry PyTorch, but the Train tab's image environment does. Prepare uses that
environment once:

  1. It checks that the image training components are installed.
  2. It installs open_clip into that environment if it is missing, at a pinned
     version, together with onnxscript for the exporter.
  3. It writes the candidate names: every species in Perch 2.0's list, every
     species resolved in this record, and every station's target species.
  4. It runs audtheia.training.bioclip_export in that environment on the
     processor, which downloads the weights once from the authors, exports the
     image encoder, checks it against PyTorch and embeds the names.
  5. It fills in desktop_models.visual_bioclip on this computer only.

Everything the step does is written to prepare.log beside its output, so a
failure can be read and explained. Run Prepare again after adding target
species or new species to the record; the names are written afresh each time.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("audtheia.app.bioclip_prepare")

__all__ = ["OUT_DIR", "OPEN_CLIP_PIN", "PrepareJob", "candidate_names"]

OUT_DIR = "models/visual/bioclip2"
OPEN_CLIP_PIN = "open_clip_torch==4.2.2"
EXTRA_PACKAGES = ("onnxscript",)
VERSION = "BioCLIP 2"
CITATION = ("Gu et al., BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning, "
            "NeurIPS 2025, arXiv:2505.23883; weights imageomics/bioclip-2, MIT licence")
_SPECIES = re.compile(r"^[A-Z][a-z]+ [a-z][a-z-]+( [a-z][a-z-]+)?$")


def _species_from_perch(repo_root: Path) -> list:
    path = Path(repo_root) / "audtheia" / "resources" / "perch" / "labels.csv"
    try:
        with open(path, encoding="utf-8", newline="") as handle:
            rows = [r[0].strip() for r in csv.reader(handle) if r][1:]
    except OSError:
        return []
    return [name for name in rows if _SPECIES.match(name)]


def candidate_names(settings, db) -> list:
    """The species BioCLIP 2 chooses among, in a stable order, without duplicates."""
    names: list = []
    for station in settings.stations():
        for item in station.get("target_species") or []:
            name = item if isinstance(item, str) else (item or {}).get("scientific_name")
            if name and str(name).strip():
                names.append(str(name).strip())
    try:
        from audtheia.analysis.taxa import TaxonBook

        with TaxonBook.for_settings(settings, db) as book:
            for obs in db.list_observations():
                for det in db.list_child_detections(obs["id"]):
                    ident = book.identify(det)
                    if ident and ident.get("resolved") and ident.get("scientific_name"):
                        names.append(ident["scientific_name"])
                    elif ident and _SPECIES.match(str(ident.get("label") or "")):
                        # A field call already written as a binomial is a candidate
                        # even when the index cannot confirm it.
                        names.append(str(ident["label"]))
    except Exception as exc:  # noqa: BLE001 - the record's names are an addition, never a blocker
        logger.warning("could not read species from the record: %s", exc)
    names.extend(_species_from_perch(Path(settings.repo_root)))
    return list(dict.fromkeys(n for n in names if _SPECIES.match(n) or " " in n))


def _flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform.startswith("win") else 0


class PrepareJob:
    """Runs Prepare in the background and reports where it has got to."""

    def __init__(self, settings, db, persist: Callable, *, components=None, popen=subprocess.Popen):
        self._settings = settings
        self._db = db
        self._persist = persist
        self._components = components
        self._popen = popen
        self._lock = threading.Lock()
        self._job: Optional[dict] = None

    # -- where things are ----------------------------------------------------

    def out_dir(self) -> Path:
        return Path(self._settings.repo_root) / OUT_DIR

    def log_path(self) -> Path:
        return self.out_dir() / "prepare.log"

    def _comps(self):
        if self._components is None:
            from audtheia.training.components import Components

            self._components = Components(self._settings)
        return self._components

    def status(self) -> dict:
        from audtheia.inference.bioclip_onnx import bioclip_configured

        try:
            env = self._comps().status("vision_detector")
            env_state = env.get("state")
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            env_state, env = "unknown", {"error": str(exc)}
        sidecar = None
        try:
            sidecar = json.loads((self.out_dir() / "bioclip2.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        with self._lock:
            job = dict(self._job) if self._job else None
        return {"vision_environment": env_state, "prepared": bioclip_configured(self._settings),
                "sidecar": sidecar, "job": job, "log_path": str(self.log_path()),
                "version": VERSION, "citation": CITATION, "licence": "MIT"}

    # -- running -------------------------------------------------------------

    def start(self, *, wait: bool = False) -> dict:
        with self._lock:
            if self._job and self._job.get("state") == "running":
                raise RuntimeError("Prepare is already running")
            self._job = {"state": "running", "step": "Starting.", "started_at": time.time(), "message": None}
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()
        if wait:
            thread.join()
        return self.status()

    def _step(self, text: str) -> None:
        with self._lock:
            if self._job is not None:
                self._job["step"] = text
        self._log(text)

    def _log(self, text: str) -> None:
        try:
            self.out_dir().mkdir(parents=True, exist_ok=True)
            with open(self.log_path(), "a", encoding="utf-8") as handle:
                handle.write(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + text.rstrip() + "\n")
        except OSError:
            pass

    def _finish(self, state: str, message: str) -> None:
        with self._lock:
            if self._job is not None:
                self._job.update(state=state, message=message, finished_at=time.time())
        self._log(f"{state.upper()}: {message}")

    def _run(self) -> None:
        try:
            self._prepare()
        except Exception as exc:  # noqa: BLE001 - every failure is reported, never raised into a thread
            logger.warning("Prepare failed: %s", exc)
            self._finish("error", str(exc))

    def _run_logged(self, args: list, *, timeout: float, env: Optional[dict] = None) -> int:
        """Run a command in the image environment, copying its output into the log line by line."""
        self._log("RUN " + " ".join(str(a) for a in args))
        proc = self._popen(args, cwd=str(self._settings.repo_root), env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True, encoding="utf-8",
                           errors="replace", creationflags=_flags())
        started = time.time()
        for line in proc.stdout:
            self._log(line)
            if line.startswith("PROGRESS "):
                self._step(line[len("PROGRESS "):].strip().capitalize() + ".")
            if time.time() - started > timeout:
                proc.kill()
                raise RuntimeError("the step took too long and was stopped")
        return proc.wait()

    def _prepare(self) -> None:
        from audtheia.training.components import env_python

        self._log("=" * 60)
        self._step("Checking the Train tab's image environment.")
        comps = self._comps()
        state = comps.status("vision_detector").get("state")
        if state != "ready":
            raise RuntimeError("the image training components are not installed. Install them in the Train tab, "
                               "under Installing the training components, then press Prepare again")
        python = str(env_python(comps.env_dir("vision_detector")))

        env = dict(os.environ)
        env["PYTHONPATH"] = str(self._settings.repo_root) + (os.pathsep + env["PYTHONPATH"]
                                                            if env.get("PYTHONPATH") else "")
        env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                    "CUDA_VISIBLE_DEVICES": "", "HF_HUB_DISABLE_TELEMETRY": "1"})

        self._step("Checking for open_clip in the image environment.")
        if self._run_logged([python, "-c", "import open_clip, onnxscript"], timeout=120, env=env) != 0:
            self._step("Installing open_clip into the image environment (needs the internet once).")
            code = self._run_logged([python, "-m", "pip", "install", OPEN_CLIP_PIN, *EXTRA_PACKAGES],
                                    timeout=3600, env=env)
            if code != 0:
                raise RuntimeError("open_clip could not be installed into the image environment; see prepare.log")

        self._step("Writing the candidate species names.")
        names = candidate_names(self._settings, self._db)
        if not names:
            raise RuntimeError("there are no candidate species names to embed")
        out = self.out_dir()
        out.mkdir(parents=True, exist_ok=True)
        names_file = out / "names.json"
        names_file.write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
        self._log(f"{len(names)} candidate names")

        self._step("Exporting BioCLIP 2 (the first run downloads its weights, about 1.7 GB).")
        code = self._run_logged([python, "-m", "audtheia.training.bioclip_export", "--out", str(out),
                                 "--names", str(names_file)], timeout=7200, env=env)
        if code != 0:
            raise RuntimeError("the export did not finish; see prepare.log beside the model for the reason")

        self._step("Setting up the verifier on this computer.")
        slot = self._settings.raw.setdefault("desktop_models", {}).setdefault("visual_bioclip", {})
        slot.update({"path": f"{OUT_DIR}/bioclip2_visual.onnx", "embeddings_path": f"{OUT_DIR}/text_embeddings.npy",
                     "labels_path": f"{OUT_DIR}/labels.json", "version": VERSION, "citation": CITATION})
        self._persist(self._settings)
        self._finish("done", f"BioCLIP 2 is set up on this computer, choosing among {len(names)} species.")
