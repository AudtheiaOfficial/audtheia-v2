#!/usr/bin/env sh
# Audtheia field-station provisioning for Linux, macOS, and Raspberry Pi OS desktop.
#
# This is a thin wrapper. It finds a suitable Python and hands control to the
# provisioning orchestrator, which connects to a Pi over SSH and stands the
# station up. Options are forwarded, for example:
#
#   ./connect-pi.sh --station-id <id> --host <pi-address> --user <pi-user>
#   ./connect-pi.sh --station-id <id> --dry-run     preview without contacting a Pi
#
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "Python 3 was not found. Install Python 3.11 or newer and try again." >&2
  exit 1
fi

exec "$PY" "$SCRIPT_DIR/bootstrap_setup_pi.py" "$@"
