"""In-app download of the desktop models the manifest offers.

Path: audtheia/app/model_downloads.py

Some desktop models are fetched once from a public address rather than shipped,
because of their size and their authors' licences. config/model_sources.json is
the single record of where each comes from, its exact size and its SHA-256. An
entry marked in_app_download can be fetched from Brain, Models and Memory, so a
person never needs a terminal.

A download is written beside its destination under a temporary name, checked
against the recorded size and checksum, and only then moved into place, so a
broken or tampered transfer never becomes a model. A file already at the
destination whose checksum matches is used as it is, with nothing downloaded.
Once the file is in place the model's slot is filled in, on this computer only:
the slot is a local-only model slot, so the committed configuration does not
change.

Only addresses written in the manifest are ever fetched, and only into the
repository's models folder.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("audtheia.app.model_downloads")

__all__ = ["DownloadError", "downloadable", "status_of", "DownloadJobs", "sha256_of"]

CHUNK = 1024 * 1024


class DownloadError(RuntimeError):
    """A model could not be downloaded or did not match its record."""


def _manifest(repo_root: Path) -> dict:
    try:
        data = json.loads((Path(repo_root) / "config" / "model_sources.json").read_text(encoding="utf-8"))
        return data.get("models", {}) or {}
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}


def downloadable(repo_root: Path) -> dict:
    """The manifest entries offered for download inside the application."""
    return {key: entry for key, entry in _manifest(repo_root).items()
            if isinstance(entry, dict) and entry.get("in_app_download") and entry.get("slot")}


def _destination(repo_root: Path, entry: dict) -> Path:
    """The entry's destination, refused unless it lies inside the models folder."""
    root = Path(repo_root).resolve()
    target = (root / str(entry.get("destination") or "")).resolve()
    models = (root / "models").resolve()
    if models not in target.parents:
        raise DownloadError("the download destination is outside the models folder")
    return target


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def status_of(settings, key: str, entry: dict) -> dict:
    """What a person needs to see about one downloadable model, without hashing it."""
    repo_root = Path(settings.repo_root)
    try:
        target = _destination(repo_root, entry)
        present = target.is_file()
        size = target.stat().st_size if present else None
    except DownloadError:
        target, present, size = None, False, None
    slot = (settings.raw.get("desktop_models", {}) or {}).get(entry["slot"]) or {}
    configured_path = slot.get("path")
    configured = False
    if configured_path:
        resolved = Path(configured_path)
        resolved = resolved if resolved.is_absolute() else repo_root / resolved
        configured = resolved.is_file()
    return {
        "key": key,
        "slot": entry["slot"],
        "version": entry.get("version"),
        "license": entry.get("license"),
        "citation": entry.get("citation"),
        "url": entry.get("url"),
        "size_bytes": entry.get("size_bytes"),
        "sha256": entry.get("sha256"),
        "destination": entry.get("destination"),
        "file_present": present,
        "file_size_bytes": size,
        "configured": configured,
        "configured_path": configured_path,
    }


class DownloadJobs:
    """Background downloads, one at a time per model, with readable progress.

    persist is called with the settings after a slot is filled in, so the server
    can save the change through its own guarded path.
    """

    def __init__(self, settings, persist: Callable, opener: Optional[Callable] = None):
        self._settings = settings
        self._persist = persist
        self._open = opener or urllib.request.urlopen
        self._lock = threading.Lock()
        self._jobs: dict = {}

    def job(self, key: str) -> Optional[dict]:
        with self._lock:
            found = self._jobs.get(key)
            return dict(found) if found else None

    def start(self, key: str, *, wait: bool = False) -> dict:
        entries = downloadable(Path(self._settings.repo_root))
        if key not in entries:
            raise DownloadError(f"{key} is not a model that can be downloaded here")
        entry = entries[key]
        with self._lock:
            current = self._jobs.get(key)
            if current and current.get("state") == "running":
                raise DownloadError("this model is already downloading")
            self._jobs[key] = {"key": key, "state": "running", "received_bytes": 0,
                               "total_bytes": entry.get("size_bytes"), "message": "Starting.",
                               "started_at": time.time()}
        thread = threading.Thread(target=self._run, args=(key, entry), daemon=True)
        thread.start()
        if wait:
            thread.join()
        return self.job(key)

    def _update(self, key: str, **changes) -> None:
        with self._lock:
            self._jobs.setdefault(key, {}).update(changes)

    def _run(self, key: str, entry: dict) -> None:
        try:
            target = _destination(Path(self._settings.repo_root), entry)
            expected = (entry.get("sha256") or "").lower() or None
            if target.is_file() and expected and sha256_of(target) == expected:
                message = "Already on this computer and its checksum matches, so nothing was downloaded."
            else:
                self._download(key, entry, target, expected)
                message = "Downloaded and checked against its recorded checksum."
            self._configure(entry)
            self._update(key, state="done", message=message + " The model is set up on this computer.")
        except Exception as exc:  # noqa: BLE001 - every failure is reported, never raised into a thread
            logger.warning("downloading %s failed: %s", key, exc)
            self._update(key, state="error", message=str(exc))

    def _download(self, key: str, entry: dict, target: Path, expected: Optional[str]) -> None:
        url = str(entry.get("url") or "")
        if not url:
            raise DownloadError("the manifest gives no download address for this model")
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        digest = hashlib.sha256()
        received = 0
        request = urllib.request.Request(url, headers={"User-Agent": "Audtheia"})
        try:
            with self._open(request) as response, open(partial, "wb") as out:
                total = entry.get("size_bytes") or int(response.headers.get("Content-Length") or 0) or None
                self._update(key, total_bytes=total, message="Downloading.")
                for block in iter(lambda: response.read(CHUNK), b""):
                    out.write(block)
                    digest.update(block)
                    received += len(block)
                    self._update(key, received_bytes=received)
            size = entry.get("size_bytes")
            if size and received != int(size):
                raise DownloadError(f"the download is {received} bytes but the record says {size}; it was not used")
            if expected and digest.hexdigest() != expected:
                raise DownloadError("the downloaded file's checksum does not match the record, so it was not used")
            os.replace(partial, target)
        finally:
            if partial.exists():
                try:
                    partial.unlink()
                except OSError:
                    pass

    def _configure(self, entry: dict) -> None:
        """Fill in the model's slot on this computer and save it through the server."""
        slot = self._settings.raw.setdefault("desktop_models", {}).setdefault(entry["slot"], {})
        slot["path"] = entry.get("destination")
        if entry.get("labels_path"):
            slot["labels_path"] = entry["labels_path"]
        slot["version"] = entry.get("version")
        slot["citation"] = entry.get("citation")
        self._persist(self._settings)
