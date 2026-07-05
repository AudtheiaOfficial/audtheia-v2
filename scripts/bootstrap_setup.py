#!/usr/bin/env python3
"""Audtheia desktop setup.

Path: scripts/bootstrap_setup.py

This is the one command a person runs first on a fresh desktop computer. In a
single, repeatable step it brings a clean machine to a runnable state:

  1. Checks the Python version is new enough for the pinned dependencies.
  2. Creates an isolated virtual environment for Audtheia, so its packages never
     collide with anything else installed on the computer.
  3. Installs the pinned desktop dependencies into that environment.
  4. Creates and initializes the local database, once, leaving an existing one
     untouched.
  5. Creates the local secrets file from its template when it is missing, so
     there is a clear place to put credentials, and never overwrites one that
     already holds real values.
  6. Downloads the base models and the taxonomic backbone the desktop needs, so
     that after setup the system runs with no internet at all.

It is safe to run more than once. Every step checks for what it would create and
skips the work when it is already done, so a repeat run repairs a partial setup
rather than duplicating or breaking anything.

It uses only the Python standard library, so it runs on a brand-new machine
before any dependency is installed, and it behaves the same on Windows and on
Raspberry Pi OS Bookworm 64-bit. The thin setup.sh and setup.bat wrappers simply
find a suitable Python and hand control to this file.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixed facts about the layout and the environment.
# ---------------------------------------------------------------------------

# The repository root is the parent of the scripts directory this file lives in.
# Every path below is resolved against it, so setup works no matter which folder
# it is launched from.
REPO_ROOT = Path(__file__).resolve().parent.parent

# The isolated environment lives at the repository root as .venv, the name the
# project's ignore rules already exclude from version control and the launcher
# expects. Keeping it at the root, rather than inside the importable package,
# keeps it out of import paths and test discovery.
VENV_DIR = REPO_ROOT / ".venv"

REQUIREMENTS_FILE = REPO_ROOT / "requirements.txt"
MODEL_SOURCES_FILE = REPO_ROOT / "config" / "model_sources.json"
SECRETS_FILE = REPO_ROOT / "config" / "secrets.json"
SECRETS_TEMPLATE = REPO_ROOT / "config" / "secrets.example.json"

# The pinned dependencies require this Python or newer. The version is checked
# up front so a machine with an older Python gets a clear message instead of a
# confusing failure deep inside a package install.
MIN_PYTHON = (3, 11)

# The generative model runtime publishes no ready-to-install package on the main
# index, so it is installed from the maintainer's prebuilt processor-only wheel
# index and its failure is not allowed to stop the rest of setup. On a Raspberry
# Pi the system's own wheel index supplies the matching build.
LLAMA_PACKAGE_NAME = "llama-cpp-python"
LLAMA_CPU_WHEEL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cpu"

# A courteous identifier for the model downloads, since some hosts refuse a
# request that does not name a client.
HTTP_USER_AGENT = "Audtheia-Setup/1.0"

# How much of a file to read at a time while downloading, one mebibyte, so a
# large model transfers with flat memory use.
DOWNLOAD_CHUNK_BYTES = 1024 * 1024


# ---------------------------------------------------------------------------
# Small console helpers, so the run reads clearly at a glance.
# ---------------------------------------------------------------------------


def _step(message: str) -> None:
    print(f"\n==> {message}", flush=True)


def _info(message: str) -> None:
    print(f"    {message}", flush=True)


def _warn(message: str) -> None:
    print(f"    warning: {message}", flush=True)


class SetupError(Exception):
    """A step failed in a way that should stop setup with a clear message."""


# ---------------------------------------------------------------------------
# Stage 1: Python version.
# ---------------------------------------------------------------------------


def check_python_version() -> None:
    _step("Checking the Python version")
    if sys.version_info < MIN_PYTHON:
        have = ".".join(str(p) for p in sys.version_info[:3])
        want = ".".join(str(p) for p in MIN_PYTHON)
        raise SetupError(
            f"Audtheia needs Python {want} or newer, but this is Python {have}. "
            f"Install a newer Python and run setup again. On Raspberry Pi OS "
            f"Bookworm 64-bit the system Python already meets this."
        )
    _info(f"Python {'.'.join(str(p) for p in sys.version_info[:3])} is new enough.")


# ---------------------------------------------------------------------------
# Stage 2: the isolated environment.
# ---------------------------------------------------------------------------


def venv_python() -> Path:
    """The path to the Python interpreter inside the virtual environment."""
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def ensure_venv() -> Path:
    _step("Preparing the isolated environment")
    interpreter = venv_python()
    if interpreter.exists():
        _info(f"Reusing the environment already present at {_show(VENV_DIR)}.")
        return interpreter

    try:
        subprocess.run(
            [sys.executable, "-m", "venv", str(VENV_DIR)],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise SetupError(
            "Could not create the isolated environment. On Debian and Raspberry "
            "Pi OS this usually means the 'python3-venv' package is missing; "
            "install it and run setup again."
        ) from exc

    if not interpreter.exists():
        raise SetupError(
            f"The environment was created but its Python was not found at "
            f"{interpreter}. Remove {_show(VENV_DIR)} and run setup again."
        )
    _info(f"Created a fresh environment at {_show(VENV_DIR)}.")
    return interpreter


# ---------------------------------------------------------------------------
# Stage 3: dependencies.
# ---------------------------------------------------------------------------


def _read_requirement_specs() -> list[str]:
    """The pinned package specifications, read from requirements.txt.

    Comment lines and blank lines are ignored, so the file stays the single,
    readable home for the version pins while setup installs exactly what it
    lists.
    """
    if not REQUIREMENTS_FILE.exists():
        raise SetupError(f"requirements file not found at {REQUIREMENTS_FILE}")
    specs: list[str] = []
    for line in REQUIREMENTS_FILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        specs.append(stripped)
    return specs


def _split_llama(specs: list[str]) -> tuple[list[str], str | None]:
    """Separate the generative-runtime pin from the rest.

    That one package installs from a different index and is allowed to fail, so
    it is handled on its own after the others.
    """
    others: list[str] = []
    llama: str | None = None
    for spec in specs:
        name = spec.replace("=", " ").replace(">", " ").replace("<", " ").split()[0].lower()
        if name == LLAMA_PACKAGE_NAME:
            llama = spec
        else:
            others.append(spec)
    return others, llama


def install_dependencies(interpreter: Path) -> None:
    _step("Installing the desktop dependencies")

    # A current pip resolves the pinned wheels cleanly; upgrading it first avoids
    # a stale resolver on an older base image. A failure here is not fatal.
    try:
        subprocess.run(
            [str(interpreter), "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
        )
    except subprocess.CalledProcessError:
        _warn("could not upgrade pip; continuing with the version already present.")

    specs = _read_requirement_specs()
    others, llama = _split_llama(specs)

    if others:
        _info("Installing: " + ", ".join(others))
        try:
            subprocess.run(
                [str(interpreter), "-m", "pip", "install", *others],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise SetupError(
                "Installing the core dependencies failed. The message above says "
                "which package and why. Fix that and run setup again."
            ) from exc
    _info("Core dependencies are installed.")

    if llama is not None:
        _install_llama(interpreter, llama)


def _install_llama(interpreter: Path, spec: str) -> None:
    _info(f"Installing the generative model runtime ({spec}) from its prebuilt wheel index.")
    try:
        subprocess.run(
            [
                str(interpreter),
                "-m",
                "pip",
                "install",
                spec,
                "--extra-index-url",
                LLAMA_CPU_WHEEL_INDEX,
            ],
            check=True,
        )
        _info("Generative model runtime installed.")
    except subprocess.CalledProcessError:
        _warn(
            "the generative model runtime did not install. The desktop still "
            "runs, but the longitudinal analysis pass will wait until it is "
            "present. A prebuilt wheel needs a matching Python; otherwise "
            "installing it from source needs a C or C++ build toolchain. You can "
            "install it later into the environment without redoing the rest of "
            "setup."
        )


# ---------------------------------------------------------------------------
# Stage 4: the database.
# ---------------------------------------------------------------------------


# Run inside the environment so it reads the same validated configuration the
# rest of the system does. It creates the database only when one is not already
# initialized, so a repeat run never disturbs collected data.
_DB_INIT_CODE = r"""
import sqlite3
from pathlib import Path
from audtheia.config import load_settings
from audtheia.storage.database import Database

settings = load_settings()
db_path = settings.db_path()
Path(db_path).parent.mkdir(parents=True, exist_ok=True)

already = False
if Path(db_path).exists():
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='observations'"
        ).fetchone()
        already = row is not None
    finally:
        conn.close()

if already:
    print("READY: the database is already initialized; leaving it untouched.")
else:
    Database(db_path, **settings.database_kwargs()).initialize_schema(settings.schema_path())
    print("READY: the database was created and initialized.")
"""


def initialize_database(interpreter: Path) -> None:
    _step("Initializing the database")
    try:
        result = subprocess.run(
            [str(interpreter), "-c", _DB_INIT_CODE],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise SetupError(
            "Initializing the database failed:\n" + (exc.stderr or exc.stdout or "")
        ) from exc
    for line in result.stdout.splitlines():
        if line.startswith("READY:"):
            _info(line[len("READY:"):].strip())


# ---------------------------------------------------------------------------
# Stage 5: the secrets file.
# ---------------------------------------------------------------------------


def ensure_secrets() -> None:
    _step("Preparing the local credentials file")
    if SECRETS_FILE.exists():
        _info("A credentials file already exists; leaving it as it is.")
        return
    if not SECRETS_TEMPLATE.exists():
        _warn(
            f"no credentials template at {_show(SECRETS_TEMPLATE)}; skipping. "
            f"Create {_show(SECRETS_FILE)} by hand when you need credentials."
        )
        return
    shutil.copyfile(SECRETS_TEMPLATE, SECRETS_FILE)
    _info(
        f"Created {_show(SECRETS_FILE)} from the template. Fill in your GBIF and "
        f"IUCN credentials there before fetching per-species data."
    )


# ---------------------------------------------------------------------------
# Stage 6: the base models and the taxonomic backbone.
# ---------------------------------------------------------------------------


def _load_manifest() -> dict:
    if not MODEL_SOURCES_FILE.exists():
        raise SetupError(f"model sources manifest not found at {MODEL_SOURCES_FILE}")
    try:
        data = json.loads(MODEL_SOURCES_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SetupError(f"model sources manifest is not valid JSON: {exc}") from None
    models = data.get("models")
    if not isinstance(models, dict):
        raise SetupError("model sources manifest has no 'models' object")
    return models


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _already_present(destination: Path, expected_sha: str | None) -> bool:
    """Whether the destination already holds the wanted file.

    With a checksum, the file must match it; without one, an existing file is
    trusted so a large model is not downloaded again on every run.
    """
    if not destination.exists():
        return False
    if expected_sha:
        return _sha256_of(destination) == expected_sha.lower()
    return True


def _download_to(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": HTTP_USER_AGENT})
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), suffix=".partial")
    tmp_path = Path(tmp_name)
    try:
        with urllib.request.urlopen(request) as response, os.fdopen(fd, "wb") as out:
            shutil.copyfileobj(response, out, length=DOWNLOAD_CHUNK_BYTES)
        tmp_path.replace(target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _finalize_gzip(downloaded_gz: Path, destination: Path) -> None:
    """Decompress a downloaded .gz into its destination, then drop the archive."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(downloaded_gz, "rb") as src, destination.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=DOWNLOAD_CHUNK_BYTES)
    downloaded_gz.unlink(missing_ok=True)


def _fetch_one(name: str, entry: dict) -> str:
    """Fetch one manifest entry. Returns one of: fetched, present, pending, failed."""
    destination = (REPO_ROOT / entry["destination"]).resolve()
    url = (entry.get("url") or "").strip()
    expected_sha = entry.get("sha256")
    kind = entry.get("kind", "file")

    if _already_present(destination, expected_sha):
        _info(f"{name}: already in place at {_show(destination)}.")
        return "present"

    if not url:
        _info(f"{name}: no download source set. Place it at {_show(destination)} by hand.")
        note = entry.get("note")
        if note:
            _info(f"    {note}")
        return "pending"

    _info(f"{name}: downloading from {url}")
    try:
        if kind == "gzip":
            archive = destination.parent / (destination.name + ".gz")
            _download_to(url, archive)
            _finalize_gzip(archive, destination)
        else:
            _download_to(url, destination)
    except (urllib.error.URLError, OSError) as exc:
        _warn(f"{name}: download failed ({exc}). Setup will continue; re-run to retry.")
        return "failed"

    if expected_sha:
        actual = _sha256_of(destination)
        if actual != expected_sha.lower():
            destination.unlink(missing_ok=True)
            _warn(f"{name}: checksum did not match; the file was removed. Re-run to retry.")
            return "failed"

    _info(f"{name}: saved to {_show(destination)}.")
    return "fetched"


def fetch_models(include_full: bool) -> dict[str, list[str]]:
    scope = "all base models (including the field-station models to stage)" if include_full else "the essential base models"
    _step(f"Fetching {scope}")
    models = _load_manifest()
    outcomes: dict[str, list[str]] = {"fetched": [], "present": [], "pending": [], "failed": []}
    for name, entry in models.items():
        tier = entry.get("tier", "essential")
        if tier == "full" and not include_full:
            _info(f"{name}: skipped (fetched only in full mode).")
            continue
        result = _fetch_one(name, entry)
        outcomes[result].append(name)
    return outcomes


# ---------------------------------------------------------------------------
# Presentation.
# ---------------------------------------------------------------------------


def _show(path: Path) -> str:
    """A path shown relative to the repository when it sits inside it."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _print_summary(interpreter: Path, model_outcomes: dict[str, list[str]] | None) -> None:
    _step("Setup summary")
    _info(f"Environment: {_show(VENV_DIR)}")
    if model_outcomes is not None:
        for label, key in (
            ("Downloaded", "fetched"),
            ("Already present", "present"),
            ("Waiting to be placed by hand", "pending"),
            ("Failed, re-run to retry", "failed"),
        ):
            names = model_outcomes.get(key) or []
            if names:
                _info(f"{label}: {', '.join(names)}")

    print("")
    _info("Next steps:")
    _info("  1. Put your GBIF and IUCN credentials in config/secrets.json.")
    _info("  2. Place any model listed above as waiting to be placed by hand.")
    interp = _show(interpreter)
    _info(f"  3. Start the desktop application: {interp} -m audtheia.app.server")
    _info("     then open the local address it prints in a web browser.")


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="setup",
        description=(
            "Set up the Audtheia desktop: an isolated environment, the pinned "
            "dependencies, the database, the credentials file, and the base "
            "models. Safe to run more than once."
        ),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also download the field-station models the desktop stages for the Pi.",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Do everything except downloading models.",
    )
    parser.add_argument(
        "--deps-only",
        action="store_true",
        help="Only create the environment and install dependencies.",
    )
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="Only download models. Combine with --full to include the staged models.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    print("Audtheia desktop setup")
    print(f"Repository: {REPO_ROOT}")

    try:
        if args.models_only:
            if args.skip_models:
                _warn("--models-only and --skip-models together leave nothing to do.")
                return 0
            outcomes = fetch_models(include_full=args.full)
            _print_summary(venv_python(), outcomes)
            return 0

        check_python_version()
        interpreter = ensure_venv()
        install_dependencies(interpreter)

        if args.deps_only:
            _print_summary(interpreter, None)
            return 0

        initialize_database(interpreter)
        ensure_secrets()

        outcomes = None
        if not args.skip_models:
            outcomes = fetch_models(include_full=args.full)

        _print_summary(interpreter, outcomes)
        return 0

    except SetupError as exc:
        print(f"\nSetup stopped: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nSetup interrupted. It is safe to run again.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
