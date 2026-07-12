#!/usr/bin/env sh
# Audtheia desktop app-window launcher for macOS.
#
# Double-click this in Finder to start Audtheia in its own desktop window instead
# of a browser tab. It finds a suitable Python and hands control to the launcher
# with the window option.
#
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "Python 3 was not found. Run the desktop setup first." >&2
  exit 1
fi

exec "$PY" "$SCRIPT_DIR/bootstrap_start.py" --window "$@"
