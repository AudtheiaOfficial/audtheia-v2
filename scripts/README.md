# scripts

Setup, provisioning, and launch scripts.

| File | Runs on | Role |
|------|---------|------|
| setup.sh | Desktop hub (Linux, macOS, Raspberry Pi OS) | Thin wrapper that runs the desktop setup bootstrap. |
| setup.bat | Desktop hub (Windows) | Thin wrapper that runs the desktop setup bootstrap. |
| bootstrap_setup.py | Desktop hub | The desktop setup: an isolated environment, the pinned dependencies, the database, the credentials file, and the base models. Standard library only. |
| connect-pi.sh | Desktop hub (Linux, macOS, Raspberry Pi OS) | Thin wrapper that runs the field-station provisioning orchestrator. |
| connect-pi.bat | Desktop hub (Windows) | Thin wrapper that runs the field-station provisioning orchestrator. |
| bootstrap_setup_pi.py | Desktop hub | Provisions a Raspberry Pi field station over SSH: sends the code, the station's configuration and models, and runs the Pi-side setup. Standard library only. |
| setup-pi.sh | Raspberry Pi field station | Sent to the Pi and run there: unpacks the code, creates the environment, initializes the store, configures the hotspot, and installs the boot service. |
| fetch-species-data.sh | Desktop hub (Linux, macOS, Raspberry Pi OS) | Thin wrapper that runs the species reference fetch. |
| fetch-species-data.bat | Desktop hub (Windows) | Thin wrapper that runs the species reference fetch. |
| bootstrap_fetch_species.py | Desktop hub | Fetches each target species' GBIF taxonomy and occurrence count and its IUCN Red List status, once, and caches it locally for offline use. |
| start.sh | Desktop hub | A one-command launcher, with a system-tray launcher for Windows, macOS, and Linux. |

Set up the desktop once on a fresh machine:

```
./setup.sh          # Linux, macOS, Raspberry Pi OS
setup.bat           # Windows
```

Then stand up a field station. Flash a Raspberry Pi with Raspberry Pi OS using
Raspberry Pi Imager, with SSH and Wi-Fi enabled, and boot it on the same network.
Then, from the desktop:

```
./connect-pi.sh --station-id <id> --host <pi-address> --user <pi-user>
connect-pi.bat --station-id <id> --host <pi-address> --user <pi-user>
```

Add `--dry-run` to preview every action without contacting a Pi. The desktop
setup download sources live in `config/model_sources.json`.
