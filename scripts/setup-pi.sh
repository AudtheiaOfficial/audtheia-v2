#!/usr/bin/env bash
# Audtheia field-station setup, run on the Raspberry Pi.
#
# The desktop sends this script to the Pi along with the application code, the
# station's configuration, its models, and its hotspot key, then runs it here. It
# unpacks the code, creates an isolated environment that can still see the
# system's hardware packages, initializes the local store, brings up the network
# hotspot and local-name discovery, and installs the service that keeps the
# station running across reboots.
#
# It is safe to run more than once: each step checks for what it would create and
# leaves an existing one in place. The steps that need system privileges are
# best-effort and never abort the run, so the core of the station is always
# configured and any system step that needs attention is reported clearly.

set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

echo "==> Audtheia field station setup"
echo "    working in ${HERE}"

# -- 1. Unpack the application code -----------------------------------------
if [ -f audtheia-code.tar.gz ]; then
  tar xzf audtheia-code.tar.gz || { echo "    could not unpack the code archive" >&2; exit 1; }
  echo "    code unpacked"
fi

if [ ! -d audtheia ]; then
  echo "    the application code is missing; nothing to set up" >&2
  exit 1
fi

# -- 2. Place configuration, credentials, and runtime directories -----------
mkdir -p config data database reports
if [ -f settings.json ]; then
  mv -f settings.json config/settings.json
  echo "    station configuration placed"
fi
if [ -f secrets.json ]; then
  mv -f secrets.json config/secrets.json
  chmod 600 config/secrets.json
  echo "    station credentials placed"
fi

# -- 3. Isolated environment with access to the system hardware packages -----
# The accelerator runtime and the camera stack are installed system-wide through
# the operating system's own packages; --system-site-packages lets the isolated
# environment import them while still keeping Audtheia's own packages separate.
if [ ! -x .venv/bin/python ]; then
  python3 -m venv --system-site-packages .venv || {
    echo "    could not create the environment; on Raspberry Pi OS install python3-venv" >&2
    exit 1
  }
  echo "    environment created"
else
  echo "    environment already present"
fi
VP="${HERE}/.venv/bin/python"
"$VP" -m pip install --upgrade pip >/dev/null 2>&1 || true

# -- 4. Field dependencies ---------------------------------------------------
# The field runner with no hardware drivers installed needs only the standard
# library, so a station reaches a running state immediately. The hardware driver
# stack (the array maths, the ByteTrack tracker, the acoustic model runtime, and
# the accelerator and camera bindings) installs together with the drivers for the
# specific hardware a station carries, described in the hardware guide. Nothing is
# pinned blindly here for hardware this script cannot see.

# -- 5. Initialize the local store (idempotent) ------------------------------
"$VP" - <<'PY'
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
        already = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='observations'"
        ).fetchone() is not None
    finally:
        conn.close()

if already:
    print("    database already initialized")
else:
    Database(db_path, **settings.database_kwargs()).initialize_schema(settings.schema_path())
    print("    database initialized")
PY

# -- 6. Read the station's network identity from its configuration -----------
{
  read -r STATION_NAME
  read -r HOTSPOT_SSID
  read -r PI_HOSTNAME
} < <("$VP" - <<'PY'
from audtheia.config import load_settings

settings = load_settings()
station = settings.active_station() or settings.stations()[0]
name = station["station_name"]
net = settings.raw.get("network", {})
ssid = net.get("hotspot_ssid_pattern", "AUDTHEIA-{station_name}").replace("{station_name}", name)
host = "".join(c if (c.isalnum() or c == "-") else "-" for c in name).strip("-").lower() or "audtheia"
print(name)
print(ssid)
print(host)
PY
)
HOTSPOT_PW="$("$VP" -c "from audtheia.config import load_settings as L; print(L().secrets.get('hotspot_password',''))")"

# -- 7. Network hotspot and local-name discovery -----------------------------
if command -v nmcli >/dev/null 2>&1 && [ -n "${HOTSPOT_PW}" ]; then
  if ! nmcli -t -f NAME connection show 2>/dev/null | grep -qx "audtheia-hotspot"; then
    sudo nmcli connection add type wifi ifname "*" con-name audtheia-hotspot autoconnect yes ssid "${HOTSPOT_SSID}" >/dev/null 2>&1 \
      && sudo nmcli connection modify audtheia-hotspot 802-11-wireless.mode ap 802-11-wireless.band bg ipv4.method shared >/dev/null 2>&1 \
      && sudo nmcli connection modify audtheia-hotspot wifi-sec.key-mgmt wpa-psk wifi-sec.psk "${HOTSPOT_PW}" >/dev/null 2>&1 \
      && echo "    hotspot '${HOTSPOT_SSID}' configured" \
      || echo "    hotspot could not be configured automatically; configure Wi-Fi access on the Pi"
  else
    echo "    hotspot already configured"
  fi
  sudo nmcli connection up audtheia-hotspot >/dev/null 2>&1 || true
else
  echo "    hotspot not configured (network manager or hotspot key not available)"
fi

sudo hostnamectl set-hostname "${PI_HOSTNAME}" >/dev/null 2>&1 \
  && echo "    station reachable as ${PI_HOSTNAME}.local" \
  || echo "    could not set the station hostname automatically"
sudo systemctl enable --now avahi-daemon >/dev/null 2>&1 || true

# The vision model runs from a compiled .hef placed at the path the station
# configuration names under models/visual/pi. Compiling a trained model to that
# format is an x86 Linux workflow done ahead of deployment, described in the
# custom-models guide; standing a station up here does not perform it.

# -- 8. Install and enable the boot service ----------------------------------
SERVICE_PATH="/etc/systemd/system/audtheia-field.service"
if sudo tee "${SERVICE_PATH}" >/dev/null <<UNIT
[Unit]
Description=Audtheia field station
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=${HERE}
ExecStart=${VP} -m audtheia.pipeline
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
then
  sudo systemctl daemon-reload >/dev/null 2>&1 || true
  sudo systemctl enable --now audtheia-field.service >/dev/null 2>&1 \
    && echo "    boot service installed and started" \
    || echo "    boot service installed; start it with: sudo systemctl start audtheia-field"
else
  echo "    could not install the boot service automatically (needs administrator rights)"
fi

# -- 9. Verify the field runner starts ---------------------------------------
echo "==> Verifying the field runner"
if "$VP" -m audtheia.pipeline --once; then
  echo "    the field runner ran a quality-control sweep"
else
  echo "    the field runner did not start cleanly; check the messages above" >&2
fi

echo "==> Field station setup complete for ${STATION_NAME}"
