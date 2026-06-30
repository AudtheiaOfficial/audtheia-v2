# audtheia/app

The web interface: a FastAPI backend and a self-contained frontend. It serves the
live feed, history, analytics, reports, and settings.

| File | Runs on | Role |
|------|---------|------|
| server.py | Desktop hub | The backend and all of its endpoints. |
| static/index.html | Desktop hub | The single-page application shell. |
| static/style.css | Desktop hub | The visual design, including light and dark themes. |
| static/app.js | Desktop hub | All of the frontend logic. |

The interface is served locally: on the desktop at localhost on port 8000, and on
a field station's own hotspot. Every asset is bundled with the application, so the
interface works with no internet connection and no external content delivery
network.
