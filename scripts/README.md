# scripts

Setup and launch scripts.

| File | Runs on | Role |
|------|---------|------|
| setup.sh | Desktop hub (Linux, macOS, Raspberry Pi OS) | Thin wrapper that runs the setup bootstrap. |
| setup.bat | Desktop hub (Windows) | Thin wrapper that runs the setup bootstrap. |
| bootstrap_setup.py | Desktop hub | The setup itself: an isolated environment, the pinned dependencies, the database, the credentials file, and the base models. Standard library only, so it runs on a brand-new machine. |
| setup-pi.sh | Pushed to the field station | Configures a Raspberry Pi field station remotely over SSH from the desktop application. |
| fetch-species-data.sh | Desktop hub | Fetches the per-species occurrence and conservation data under your own credentials, with a documented path to refresh it later. |
| start.sh | Desktop hub | A one-command launcher, with a system-tray launcher for Windows, macOS, and Linux. |

Run setup once on a fresh machine:

```
./setup.sh          # Linux, macOS, Raspberry Pi OS
setup.bat           # Windows
```

Useful options, forwarded to the bootstrap: `--full` also downloads the
field-station models the desktop stages for a Pi, `--skip-models` sets up the
environment and database without downloading models, `--deps-only` installs just
the dependencies, and `--models-only` fetches models into an existing setup. The
download sources live in `config/model_sources.json`.
