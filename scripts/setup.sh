#!/usr/bin/env sh
# Audtheia desktop setup for Linux, macOS, and Raspberry Pi OS Bookworm.
#
# This is a thin wrapper. It finds a suitable Python and hands control to the
# cross-platform bootstrap, which does all of the real work and behaves the same
# on every operating system. Any options you pass are forwarded, for example:
#
#   ./setup.sh              install everything and fetch the essential models
#   ./setup.sh --full       also fetch the field-station models to stage
#   ./setup.sh --skip-models  set up the environment and database only
#
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "Python 3 was not found. Install Python 3.11 or newer and run setup again." >&2
  exit 1
fi

exec "$PY" "$SCRIPT_DIR/bootstrap_setup.py" "$@"
