# `audtheia/app/` — Web interface (FastAPI backend + bundled frontend)

| File / dir | Builds in | Role |
|---|---|---|
| `server.py` | Session 11 | FastAPI backend and all API endpoints. |
| `static/index.html` | Session 12 | Single-page app shell. |
| `static/style.css` | Session 13 | Design system (dark/light). |
| `static/app.js` | Session 14 | All frontend logic. |

Served locally — desktop at `localhost:8000`, Pi hotspot at `audtheia.local`. All assets bundled locally; **no CDN dependency**. See `audtheia-v2-master-concept.md` §5 for the full panel/navigation spec (Detections, Audio, Brain, GPS, Analytics, Reports, Settings).
