# `models/acoustic/` — Swappable acoustic detection models

- **`birdnet/`** — BirdNET-Analyzer **full** model (e.g. `BirdNET_GLOBAL_6K.tflite`). Terrestrial/coastal-avian default. Runs on the Pi CPU.
- **`marine/`** — Underwater passive-acoustic-monitoring (PAM) model slot (cetaceans, fish choruses, snapping-shrimp soundscape indices). BirdNET is in-air avian and not usable on a submerged hydrophone. Default model selection is an open research item (O1, → Session 5).
- **`custom/`** — User fine-tuned classifiers for local/regional species (e.g. Puerto Rico, Haiti terrestrial species).

Model selection is a `settings.json` value — swapping requires no code change. See `audtheia-v2-master-concept.md` §7.
