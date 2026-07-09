#!/usr/bin/env python3
"""Run an Audtheia station on this desktop, with no field hardware.

Path: scripts/bootstrap_run_desktop.py

This is the one command for the hardware-free path. It captures from an ordinary
webcam, network stream, or video file, quality-controls and verifies each
observation, runs the longitudinal dream pass on its schedule, generates reports,
and serves the web interface, all on one computer. It simply finds the isolated
environment that setup created and hands off to the desktop orchestrator, so it
uses only the Python standard library and behaves the same on Windows, macOS, and
Linux. Any arguments are passed straight through (for example --once to run over a
video file and exit, or --station-id to pick a specific station).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = REPO_ROOT / ".venv"


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def main(argv: list) -> int:
    interpreter = venv_python()
    if not interpreter.exists():
        print(
            "The Audtheia environment is not set up yet. Run setup first "
            "(scripts/setup.sh on Linux, macOS, or Raspberry Pi OS, or scripts\\setup.bat "
            "on Windows), then run this again.",
            file=sys.stderr,
        )
        return 1

    command = [str(interpreter), "-m", "audtheia.app.orchestrator", *argv]
    try:
        return subprocess.call(command, cwd=str(REPO_ROOT))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
