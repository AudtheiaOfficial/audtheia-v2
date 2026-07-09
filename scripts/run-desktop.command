#!/usr/bin/env bash
# Double-clickable macOS launcher for the hardware-free desktop station.
HERE="$(cd "$(dirname "$0")" && pwd)"
exec python3 "${HERE}/bootstrap_run_desktop.py" "$@"
