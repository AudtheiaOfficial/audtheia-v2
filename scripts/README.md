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
| start.sh | Desktop hub (Linux, macOS) | Thin wrapper that runs the launcher. |
| start.command | Desktop hub (macOS) | Double-clickable launcher for Finder. |
| start.bat | Desktop hub (Windows) | Double-clickable launcher. |
| start-app.sh | Desktop hub (Linux, macOS) | Thin wrapper that opens Audtheia in its own desktop window. |
| start-app.command | Desktop hub (macOS) | Double-clickable app-window launcher for Finder. |
| start-app.bat | Desktop hub (Windows) | Double-clickable app-window launcher. |
| bootstrap_start.py | Desktop hub | Starts the application, waits until it answers, shows the local address, and offers to open the browser; an optional desktop app window and an optional system-tray icon with Open and Quit are available. |

Set up the desktop once on a fresh machine:

```
./setup.sh          # Linux, macOS, Raspberry Pi OS
setup.bat           # Windows
```

Antivirus note (Windows): the setup installs matplotlib as a dependency of the
object tracker, and some antivirus tools raise a heuristic false positive on one
of its compiled files (`matplotlib\ft2font...pyd`), quarantining it. This is a
known false alarm on a legitimate, widely used library, not a real threat. If it
happens, allow or restore that file, or add a Windows Defender exclusion for the
`.venv` folder before running setup:

```
Add-MpPreference -ExclusionPath "C:\path\to\audtheia-v2\.venv"
```

Then run setup (or reinstall) so the file stays in place and the detection
tracker can load.

Then stand up a field station. Flash a Raspberry Pi with Raspberry Pi OS using
Raspberry Pi Imager, with SSH and Wi-Fi enabled, and boot it on the same network.
Then, from the desktop:

```
./connect-pi.sh --station-id <id> --host <pi-address> --user <pi-user>
connect-pi.bat --station-id <id> --host <pi-address> --user <pi-user>
```

Add `--dry-run` to preview every action without contacting a Pi. The desktop
setup download sources live in `config/model_sources.json`.

Launch the desktop application:

```
./start.sh          # Linux, macOS: opens in your browser
start.bat           # Windows: opens in your browser
./start-app.sh      # Linux, macOS: opens in its own desktop window
start-app.bat       # Windows: opens in its own desktop window
```

The window option draws Audtheia in its own application window using the
operating system's own web view, so it runs like an app rather than a browser
tab. It needs one optional package (`pywebview`); the launcher offers to install
it the first time and falls back to the browser if it is declined or unavailable.
The same option is available on the plain launcher as `start.bat --window` (or
`./start.sh --window`).
