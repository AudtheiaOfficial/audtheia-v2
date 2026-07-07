#!/usr/bin/env sh
# Audtheia per-species reference fetch for Linux, macOS, and Raspberry Pi OS.
#
# This is a thin wrapper. It prefers the environment that setup created (which
# has the network library installed) and hands control to the fetcher. Options
# are forwarded, for example:
#
#   ./fetch-species-data.sh                          fetch every station's target species
#   ./fetch-species-data.sh --species "Panthera leo" fetch one species by name
#   ./fetch-species-data.sh --refresh                update species already on file
#
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python"

if [ -x "$VENV_PY" ]; then
  PY="$VENV_PY"
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "Python 3 was not found. Run the desktop setup first." >&2
  exit 1
fi

exec "$PY" "$SCRIPT_DIR/bootstrap_fetch_species.py" "$@"
