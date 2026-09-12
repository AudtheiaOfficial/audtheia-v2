# Audtheia V2

<div align="center">

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21940755-blue?style=flat)](https://doi.org/10.5281/zenodo.21940755)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-1f6feb?style=flat)](#system-requirements)
[![Runtime](https://img.shields.io/badge/Runtime-100%25%20offline-2ea44f?style=flat)](#why-it-exists)
[![Edge AI](https://img.shields.io/badge/Edge%20AI-Raspberry%20Pi%205%20%2B%20Hailo-c51a4a?style=flat&logo=raspberrypi&logoColor=white)](docs/hardware.md)
[![Version](https://img.shields.io/badge/version-2.0.0-blue?style=flat)](CHANGELOG.md)
[![Documentation](https://img.shields.io/badge/docs-latest-8250df?style=flat)](docs/)
[![GitHub Stars](https://img.shields.io/github/stars/AudtheiaOfficial/audtheia-v2?style=flat&logo=github&cacheSeconds=3600)](https://github.com/AudtheiaOfficial/audtheia-v2/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/AudtheiaOfficial/audtheia-v2?style=flat&logo=github&cacheSeconds=3600)](https://github.com/AudtheiaOfficial/audtheia-v2/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/AudtheiaOfficial/audtheia-v2?style=flat&logo=github&cacheSeconds=3600)](https://github.com/AudtheiaOfficial/audtheia-v2/issues)

</div>


<div align="center">

<img src="assets/audtheia-demo.gif" width="820" alt="Audtheia V2 in action: the live Detections feed of Carolina Chickadee events, expert review, the acoustic audit, the Brain panel, and theme options">

### Fully offline environmental intelligence for the field

</div>

**A fully offline environmental-intelligence platform for biodiversity monitoring, Marine Protected Area designation support, and longitudinal pattern discovery.**

Audtheia V2 runs on an ordinary personal computer and, when you want a field deployment, connects to one or more Raspberry Pi 5 field stations with a Hailo AI accelerator, deployed in marine, terrestrial, estuarine, or freshwater sites. Once set up, **no internet connection is required at runtime.** A camera or a hydrophone, a field station, and the desktop application are enough to run a rigorous, structured monitoring pipeline anywhere, including the remote, hard-to-reach places where continuous observation is least practical and most needed.

You can also run the entire platform on a single computer with no field hardware at all. That makes it easy to try, to teach with, to compare methods, and to analyze recorded footage, before or instead of building a station.

---

## Two ways to run Audtheia

Audtheia runs the same pipeline in two deployments. Choose the one that fits what you have.

| | Desktop hardware-free mode | Field station |
|---|---|---|
| Where it runs | Any personal computer | A Raspberry Pi 5 in the field, plus your computer as the hub |
| What it watches | A webcam, a network or web-page stream, or a video file | A dedicated camera and, for marine sites, a hydrophone, at a fixed location |
| Detection model | An RF-DETR model (Apache-2.0) run through ONNX Runtime | A detector you train and compile to the accelerator |
| Power | Wall power | Solar and battery, or wall power near shore |
| Best for | Trying the system, teaching, comparing methods, analyzing recordings | Real longitudinal field studies and Marine Protected Area work |

In both modes, detection triggers a complete multimodal observation, quality control finalizes it, the desktop verifies and runs a longitudinal analysis over the record, and reports are generated with every value labeled by its provenance. The field station simply adds real sensors and real, unattended deployment.

## Project status

**Working now, on any computer.** The desktop hardware-free mode captures live from a webcam, a network or web-page stream, or a video file, and runs the full pipeline: detection, object tracking, one-event-per-observation capture, desktop verification, quality control, the longitudinal pass, expert review, and provenance-labeled PDF and CSV reports. This path is covered by the automated test suite.

**Next steps, validating on hardware.** As of today, the field-station capture layer, live audio from a microphone or hydrophone, an I2C environmental sensor bank, a satellite receiver, and on-accelerator detection on the Hailo NPU, are fully implemented against the documented device protocols and unit-tested for scripted hardware backends. It is now entering on-device validation on physical Raspberry Pi 5 and AI HAT+ 2 hardware. Until that validation completes, treat the field-station capture path as functionally complete but not yet field-proven. Field trials and contributions are warmly welcome; see [Contributing to the field station](#contributing-to-the-field-station) and [`docs/field-drivers.md`](docs/field-drivers.md).

**A suggested direction.** Alexandra Pierre, PhD has suggested a future path toward open, multilingual models, so that reports and community-facing educational fact sheets can reach local communities in their own languages, such as Spanish and French, rather than in English alone.

## Why it exists

Many ecosystems most worth protecting are significantly challenging to observe. Traditional field surveys are limited by what a person can see and how long they can stay, for instance: many species are cryptic, nocturnal, or rare; dense habitats and low light conceal them; skilled fieldwork is costly and hard to sustain; and even careful observers introduce bias. Continuous audio-visual monitoring closes these gaps, watching field sites around the clock and listening to animal calls, catching the crepuscular and nocturnal activity that timed surveys miss, so that even elusive species are detected and counted more reliably, and without the disturbance a human presence might bring. Audtheia was built to put that capability in the hands of a single researcher, a classroom, a small organization, or a protected-area manager, work that would otherwise take a full team on site.

It does that without asking you to give up control of your data. Everything runs on hardware you own. Observations live in a local database on your own computer; nothing is sent to a cloud service, and no credentials leave your machine. The detection code, the analysis code, the database schema, and the desktop application are released under the MIT license, so anyone can deploy it, adapt it to their own species and sites, or build on it.

## Who Audtheia is for

Audtheia is built for people who need a defensible ecological record but may not have a data-engineering team behind them. Everything a study needs is reachable from the interface, so no programming is required to run it. It is also an accessible way for anyone curious about environmental monitoring in the age of edge AI to learn by doing, from a first detection to a finished, provenance-labeled report. The platform makes three distinct subfields of AI tangible and, crucially, keeps them separable and labeled: computer vision, bioacoustics and audio recognition, and a language model (LLM) that, by deliberate design, is constrained to never generate new facts or invent data.

- **Field researchers and graduate students** running longitudinal studies at sites that are remote, hard to reach, or simply too demanding to watch in person around the clock.
- **Marine Protected Area managers** who need continuous, provenance-labeled evidence to support designation, evaluation, reporting, and habitat preservation.
- **Small conservation organizations** that want research-grade monitoring without a cloud subscription or a standing compute budget.
- **Educators and classrooms** using the desktop hardware-free mode to teach visual and acoustic detection, data evaluation, ethical AI practices, and environmental monitoring practices.
- **Anyone re-analyzing recorded footage**, comparing methods, or building a baseline before committing to field hardware.

Typical uses include coral-reef and benthic monitoring, marine megafauna and fish surveys, bird phenology and acoustic monitoring, estuarine and freshwater biodiversity work, and endangered-species presence tracking with conservation status attached to each observation.

## How it works

**Detection is the trigger for everything.** A camera streams continuously into a computer vision model, which screens every frame it can. Nothing is captured on a timer. When the model sees something, or when the acoustic stream hears something, that moment becomes an **event**:

- an object tracker collapses the detected animal across consecutive frames into one event, never one row per frame;
- the event simultaneously captures audio, reads the GPS and every configured environmental sensor, and queries the offline GBIF taxonomic backbone;
- one provenance-tagged, multimodal observation is written to the local database.

To keep storage bounded and a station's power and memory footprint low, an event is saved as the salient frames the model detected rather than as continuous video, and the interface reconstructs a short clip of the event by replaying those frames.

The desktop hub does the heavy, high-accuracy work: it re-verifies every detection with a second, larger model (RF-DETR through ONNX Runtime), owns all ecological interpretation, runs a longitudinal pattern-discovery pass over the verified record, generates reports, and holds the authoritative database. **Only reports and the pattern pass run on a schedule;** everything else is driven by what actually happens in front of the sensor.

## System requirements

**Desktop hub (required).** Any reasonably modern computer running Windows, macOS, or Linux, with Python 3.11 or newer. No GPU is required, the desktop verifier runs on ONNX Runtime and works on the CPU, though a supported GPU speeds it up. Allow several gigabytes of free disk for the offline GBIF taxonomic backbone and the models, which setup downloads once. After setup, no internet connection is needed to run the platform; a network is used only to reach a field station and, at setup time, to fetch species reference data.

**Field station (optional).** A Raspberry Pi 5 running Raspberry Pi OS (64-bit, Bookworm or newer) with the Raspberry Pi AI HAT+ 2 and its Hailo accelerator, a camera, and, for underwater sites, a hydrophone, plus the environmental sensors your site needs. Field stations run on solar and battery, or on wall power near shore. The exact parts, reference builds, and the power-and-solar budget method are in the [hardware guide](docs/hardware.md).

## Setup Guide

The full step-by-step instructions, with screenshots and short demos, live in the **[Setup Guide](docs/setup-guide.md)**: installing Audtheia, running desktop capture with no hardware, connecting a Raspberry Pi field station, preparing species reference data, and using the interface. No programming experience is required.

A one-minute start:

```
git clone https://github.com/AudtheiaOfficial/audtheia-v2.git
cd audtheia-v2
```

Then follow the [Setup Guide](docs/setup-guide.md).

## Repository structure

```
audtheia-v2/
├── audtheia/
│   ├── config.py            validating settings loader
│   ├── pipeline/            capture runtime: monitor, acoustic, environment, composer, drivers (desktop capture)
│   ├── analysis/            observation (deterministic QC), verify (RF-DETR), dream (longitudinal pass)
│   ├── inference/           concrete desktop model adapters (RF-DETR ONNX verifier)
│   ├── reports/             provenance-labeled PDF and CSV
│   ├── storage/             schema.sql, database.py
│   └── app/                 server (FastAPI) + static interface, orchestrator (desktop station)
├── models/                  visual, acoustic, and language models (fetched or placed, not committed)
├── config/                  settings.json, model_sources.json, secrets template, README
├── scripts/                 setup, connect-pi, fetch-species-data, start, run-desktop (each .sh/.command/.bat)
├── tests/                   unit and integration tests, all on mocked hardware
├── docs/                    hardware, custom-models, dream-pass guides
└── requirements.txt · requirements-dev.txt · requirements-pi.txt · LICENSE
```

## Documentation

- [Setup guide](docs/setup-guide.md): installing and running Audtheia, desktop and field-station, step by step with screenshots and demos.
- [Hardware guide](docs/hardware.md): the parts, the reference builds, the power-and-solar budget method, and the camera anti-fouling plan.
- [Custom models guide](docs/custom-models.md): training your own detectors, compiling a field model to the accelerator, exporting the desktop model, and training acoustic classifiers.
- [Dream pass guide](docs/dream-pass.md): how the longitudinal analysis works and how to read its candidate hypotheses.

## Applications and impact

Because it turns continuous observation into a structured, provenance-labeled record, Audtheia is suited to work that traditional surveys struggle to sustain.

- **Marine Protected Area monitoring and compliance.** Continuous, defensible baselines and time series that support designation, evaluation, and reporting against conservation frameworks, with every figure traceable to the detections behind it.
- **Research on elusive, nocturnal, and rare species.** Around-the-clock audio-visual capture surfaces the crepuscular and nocturnal activity and hard-to-see taxa that timed transects miss, and paired environmental sensors relate that activity to the conditions driving it.
- **Education and edge-AI training.** A hands-on way to learn environmental monitoring and on-device AI from end to end, from a first detection to a finished report, with no programming required.
- **Expert-in-the-loop science and FAIR data.** Reviewers confirm, reject, or relabel detections to fine-tune models and audit results; captured frames export as ready-to-correct training packages; and the record follows FAIR and comparable scientific guidelines (Findable, Accessible, Interoperable, and Reusable), feeding both report generation and the longitudinal pass that surfaces trends, correlations, and co-occurrences across a site over time.

## Relationship to Audtheia V1

Audtheia V2 builds on the earlier Audtheia monitoring platform, [Audtheia V1](https://audtheiaofficial.github.io/audtheia-environmental-monitoring/index.html), which remains available and unchanged. V1 is a cloud-connected system that processes video in real time while specialized AI agents perform deep ecological analysis in parallel, storing research-grade observations and generating professional PDF reports; its [website](https://audtheiaofficial.github.io/audtheia-environmental-monitoring/index.html) hosts a live demo, a researcher dashboard, and full documentation. V2 is a fresh, fully offline redesign in its own repository. The marine-sponge detection model from V1, the [Official Porifera Classifier](https://universe.roboflow.com/marine-sciences-research-station/official-porifera-classifier-ju8er) on Roboflow Universe, carries forward here as the default desktop verification model, so earlier work continues to apply.

## Contributing to the field station

Audtheia's field-station architecture is complete and the desktop platform is ready to pair with it; what remains is validating the hardware driver layer on physical devices and running real deployments. If you work with the Raspberry Pi 5 and the AI HAT+ 2, with environmental sensors, hydrophones, or GPS receivers, or you are planning a marine, terrestrial, estuarine, or freshwater deployment, your help would be genuinely valuable and gratefully credited.

Especially welcome:

- **On-device validation of the field drivers.** The live audio source, the I2C environmental sensor bank, the NMEA satellite receiver, and the Hailo accelerator detector each plug into a stable seam and need confirming against real hardware. The drivers and a step-by-step validation checklist are in [`docs/field-drivers.md`](docs/field-drivers.md); the code is [`audtheia/pipeline/field_drivers.py`](audtheia/pipeline/field_drivers.py).
- **Building and compiling a detector to the accelerator**, following the [custom models guide](docs/custom-models.md), on the [reference hardware](docs/hardware.md).
- **Real field trials and the notes that come out of them**, following the [field deployment checklist](docs/field-deployment-checklist.md).

The best place to start is a [GitHub issue or discussion](https://github.com/AudtheiaOfficial/audtheia-v2/issues): say hello, and we will find the right first step together. Please also read [`CONTRIBUTING.md`](CONTRIBUTING.md) for the design rules a contribution keeps.

## Citation

If you use Audtheia V2 in your research, please cite it. A machine-readable [`CITATION.cff`](CITATION.cff) is included; GitHub renders a "Cite this repository" button from it. The archived v2.0.0 release is citable with the DOI [10.5281/zenodo.21940755](https://doi.org/10.5281/zenodo.21940755).

```bibtex
@software{audtheia_v2,
  author  = {Portalatin, Andy},
  title   = {Audtheia V2: A Fully Offline Environmental-Intelligence Platform},
  year    = {2026},
  version = {2.0.0},
  doi     = {10.5281/zenodo.21940755},
  publisher = {Zenodo},
  url     = {https://github.com/AudtheiaOfficial/audtheia-v2}
}
```

## License

Released under the MIT license. Copyright 2026 Andy Portalatin. See [LICENSE](LICENSE).

## Acknowledgments

Audtheia V2 stands on open scientific data and open-source software.

- **[GBIF](https://www.gbif.org/)**, the Global Biodiversity Information Facility, for the taxonomic backbone and occurrence data, used under CC BY 4.0.
- **[IUCN Red List](https://www.iucnredlist.org/)** for conservation-status data, fetched under the user's own credentials.
- **[BirdNET](https://github.com/birdnet-team/BirdNET-Analyzer)** for avian and terrestrial acoustic recognition. Its analyzer code is MIT-licensed; its models are provided under CC BY-NC-SA 4.0, under which research and educational use is treated as non-commercial.
- **[RF-DETR](https://github.com/roboflow/rf-detr)** (Apache-2.0), the transformer detection architecture used for high-accuracy desktop verification through ONNX Runtime.
- **[ONNX Runtime](https://onnxruntime.ai/)**, **[llama.cpp](https://github.com/ggml-org/llama.cpp)** for on-device language-model inference, and **[FastAPI](https://fastapi.tiangolo.com/)** and **[Uvicorn](https://www.uvicorn.org/)** for the local application server.

The direction of building Audtheia as a fully offline, freely usable educational tool was encouraged by the guidance of Alexandra Pierre, PhD, whose support is gratefully acknowledged.

Thanks to the marine biology, acoustic monitoring, and conservation communities whose feedback shaped the platform.

**Demo footage.** The wildlife footage in these demonstrations is shown for illustration, with gratitude to the videographers. The Carolina Chickadee footage in the overview animation is by [Navarre's Wild Shots](https://www.youtube.com/watch?v=I4OUdQR3oTQ); the desktop hardware-free demonstration uses *The Giant Barrel Sponge (Xestospongia muta) Spawning* by [Shane Wever](https://www.youtube.com/@shanewever).

## Contact

Questions, bug reports, and feature requests go through **[GitHub Issues](https://github.com/AudtheiaOfficial/audtheia-v2/issues)**. Please search existing issues before opening a new one, and see [CONTRIBUTING.md](CONTRIBUTING.md) for how to report effectively.

---

<div align="center">

Made with 💚 for the scientific community.

</div>
