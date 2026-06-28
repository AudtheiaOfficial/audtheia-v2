# `audtheia/pipeline/` — Field station runtime (runs on the Pi)

| File | Builds in | Role |
|---|---|---|
| `monitor.py` | Session 4 | Camera → YOLO11 (`.hef`) on the Hailo NPU → ByteTrack event aggregation (Pi CPU) → trigger on track close. |
| `acoustic.py` | Session 5 | Swappable audio model (BirdNET / marine PAM); triggered capture **and** independent acoustic trigger; event-gated capture window. |
| `environment.py` | Session 6 | GPS (+ UTC clock) + water/air/soil sensors, read on trigger. |

Nothing in this folder runs a language model. See `audtheia-v2-master-concept.md` §3–4.
