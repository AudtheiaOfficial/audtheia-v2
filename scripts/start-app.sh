#!/usr/bin/env sh
# Audtheia desktop app-window launcher for Linux and macOS.
#
# Start Audtheia in its own desktop window instead of a browser tab. This is a
# thin wrapper that finds a suitable Python and hands control to the launcher
# with the window option. Any extra options are forwarded, for example:
#
#   ./start-app.sh                open Audtheia in a desktop window
#   ./start-app.sh --no-browser   do not fall back to the browser
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
