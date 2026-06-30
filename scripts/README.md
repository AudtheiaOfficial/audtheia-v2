# scripts

Setup and launch scripts.

| File | Runs on | Role |
|------|---------|------|
| setup.sh | Desktop hub | Installs the desktop dependencies and downloads the base models. |
| setup-pi.sh | Pushed to the field station | Configures a Raspberry Pi field station remotely over SSH from the desktop application. |
| fetch-species-data.sh | Desktop hub | Fetches the per-species occurrence and conservation data under your own credentials, with a documented path to refresh it later. |
| start.sh | Desktop hub | A one-command launcher, with a system-tray launcher for Windows, macOS, and Linux. |
