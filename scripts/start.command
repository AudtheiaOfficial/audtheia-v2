#!/usr/bin/env sh
# Audtheia desktop launcher for macOS.
#
# This file can be double-clicked in Finder to start Audtheia. It finds a
# suitable Python and hands control to the launcher, which starts the
# application and opens the interface.
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

exec "$PY" "$SCRIPT_DIR/bootstrap_start.py" "$@"
