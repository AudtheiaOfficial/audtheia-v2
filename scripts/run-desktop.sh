#!/usr/bin/env bash
# Run an Audtheia station on this desktop (Linux, macOS, Raspberry Pi OS), with
# no field hardware. Any arguments pass straight through to the orchestrator.
HERE="$(cd "$(dirname "$0")" && pwd)"
exec python3 "${HERE}/bootstrap_run_desktop.py" "$@"
