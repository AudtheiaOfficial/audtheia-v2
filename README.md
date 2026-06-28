# Audtheia V2

**A fully offline environmental-intelligence platform for biodiversity monitoring, Marine Protected Area designation support, and longitudinal pattern discovery.**

Audtheia V2 runs on an ordinary personal computer (the desktop hub) and connects to one or more Raspberry Pi 5 + AI HAT+ 2 field stations deployed in marine, terrestrial, estuarine, or freshwater sites. Once a station is set up, **no internet connection is required at runtime.** A camera or a hydrophone, a field station, and the desktop app are enough to run a rigorous, structured monitoring pipeline anywhere, including the remote, hard-to-reach places where continuous observation is least practical and most needed.

> **Status:** pre-release and in active development. The database foundation (`audtheia/storage/schema.sql`) is complete and verified.

---

## Why it exists

Some of the ecosystems most worth protecting are the hardest to watch. Field sites can be remote, difficult to access, or demand around-the-clock observation that no small team can sustain in person. Audtheia was built to close that gap: to give a single researcher, a classroom, a small NGO, or a protected-area manager the kind of continuous, defensible monitoring that normally takes a full team on site.

It does that without asking you to give up control of your data. Everything runs on hardware you own. Observations live in a local database on your own desktop; nothing is sent to a cloud service, and no credentials leave your machine. The entire pipeline (detection models, analysis code, the database schema, and the desktop application) is released under the MIT license, so anyone can deploy it, adapt it to their own species and sites, or build on it without restriction.

## How it works

**Detection is the trigger for everything.** A field station's camera streams continuously into a custom-trained YOLO11 model running on its Hailo NPU, which screens every frame it can. Nothing is captured on a timer. When the model sees something, or when the acoustic stream *hears* something, that moment becomes an **event**:

- **ByteTrack** collapses the detected animal across consecutive frames into one event, never one row per frame.
- The event simultaneously captures audio, reads GPS and every configured environmental sensor, and queries the offline GBIF taxonomic backbone.
- One provenance-tagged, multimodal observation is written to the field station's local database.

The desktop hub does the heavy, high-accuracy work. It re-verifies every detection with a second, larger model (RF-DETR via ONNX Runtime), owns all ecological interpretation, runs a longitudinal pattern-discovery pass over the verified record, generates reports, and holds the authoritative database that every field station syncs up into. **Only reports and the pattern pass run on a schedule;** everything else is driven by what actually happens in front of the sensor.

The split is deliberate. The field station spends energy *in proportion to ecological events* instead of burning compute around the clock, and its two processors never contend: vision stays on the NPU, while tracking, the acoustic model, and sensor orchestration stay on the CPU.

## The scientific guarantee

What makes Audtheia's data trustworthy is not a metaphor; it is bookkeeping, enforced at the database level.

Every value the system ever stores carries a `data_source` tag and a quality-control status. A number a sensor measured, a value looked up from a reference database, a model's classification, an interpretation inferred downstream, and a candidate pattern proposed by the longitudinal pass all remain **permanently distinguishable**. The schema's `CHECK` constraints make it structurally impossible for the system to blur measured fact with inference, or to coin a status term it was never given. Marine sensor channels additionally carry the standard QARTOD quality flags used in oceanographic data.

Interpretation and pattern discovery live on the far side of that firewall. The field station only validates, completes, and consolidates what it measured; it never writes free-form ecological claims as data. Ecological role, rarity, and behavioral context are added afterward, on the desktop, and labeled as inferred. The longitudinal pass surfaces **candidate hypotheses**, each carrying an effect size and the exact span of data behind it, and each traceable back to the verified observations that produced it. They are offered as questions worth investigating, never as finished findings.

The goal is a research-grade, FAIR-defensible record a scientist can stand behind.

## Configurable for your ecosystem

Audtheia ships with a marine-sponge (Porifera) detection model as a working reference implementation, but nothing about the platform is sponge-specific. The detection model, the acoustic model, the environmental-sensor set, and the species reference data are all **configuration, not code**: point `settings.json` at your own custom-trained models and your target species, and the same pipeline runs for birds, fish, reef invertebrates, or whatever your site requires, with no code change between a marine and a terrestrial deployment.

## Hardware

- Raspberry Pi 5 (8GB; 16GB optional) + Raspberry Pi AI HAT+ 2 (Hailo-10H NPU, 40 TOPS, 8GB onboard RAM)
- Pi Camera Module 3 Wide
- **Marine:** Aquarian H2d hydrophone element + preamp into a USB audio ADC. **Terrestrial:** standard microphone + BirdNET.
- u-blox M10 GPS dongle (also the authoritative UTC clock source)
- Deployment-specific environmental sensors (water, or air/soil/light), toggled in `settings.json` (no code change)

Full parts list, wiring, and power-budget methodology live in `docs/hardware.md` (in progress).

## Repository structure

```
audtheia-v2/
├── audtheia/
│   ├── pipeline/   ← FIELD STATION runtime (Pi): monitor.py · acoustic.py · environment.py
│   ├── analysis/   ← observation.py (Pi, deterministic QC) · verify.py (desktop, RF-DETR) · dream.py (desktop)
│   ├── reports/    ← generate.py (PDF + CSV)
│   ├── storage/    ← schema.sql · database.py
│   └── app/        ← server.py (FastAPI) · static/ (index.html · style.css · app.js)
├── models/         ← visual/ (pi · desktop) · acoustic/ (birdnet · marine · custom) · llm/
├── data/           ← rolling buffer / detections / gps (gitignored)
├── database/       ← audtheia.db (gitignored)
├── reports/        ← generated PDFs/CSVs (gitignored)
├── config/         ← settings.json + README
├── scripts/        ← setup.sh · setup-pi.sh · fetch-species-data.sh · start.sh
├── tests/          ← unit + integration + mocked-hardware smoke tests
└── docs/           ← hardware.md · custom-models.md · dream-pass.md
```

## Relationship to Audtheia V1

Audtheia V2 builds on the earlier Audtheia monitoring platform, [Audtheia V1](https://audtheiaofficial.github.io/audtheia-environmental-monitoring), which remains available and unchanged. V2 is a fresh, fully offline redesign and lives in its own repository. If you worked with V1, the marine-sponge detection model that shipped with it carries forward here as the default high-accuracy verification model, so your earlier work continues to apply.

## License

Released under the MIT license. © 2026 Andy Portalatin. See [`LICENSE`](LICENSE).
