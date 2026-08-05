# Changelog

All notable changes to Audtheia V2 are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - Unreleased

Audtheia V2 is a fresh, fully offline redesign of the Audtheia platform in its own
repository. Where V1 relied on cloud workflows and external services, V2 runs the
entire pipeline on hardware the user owns, with no internet connection required at
runtime. The date above is set when the first release is tagged.

### Added

- **Fully offline architecture.** The complete capture-to-report pipeline runs on
  a personal computer, optionally connected to Raspberry Pi field stations, with
  no runtime cloud dependency and no credentials leaving the machine.
- **Event-driven capture.** Detection is the trigger for everything. A detection
  or an acoustic onset becomes an event; an object tracker collapses it across
  frames into one observation that simultaneously captures audio, reads GPS and
  configured sensors, and queries the offline GBIF taxonomic backbone.
- **The provenance firewall.** Every stored value carries a provenance tag and a
  quality-control status, keeping measured, referenced, inferred, and
  pattern-derived values permanently distinguishable at the database level.
- **Two deployments from one pipeline.** A desktop hardware-free mode that
  captures from a webcam, stream, or video file, and a field-station mode that
  adds real sensors and unattended deployment.
- **Desktop verification.** A second, higher-accuracy detection model
  (RF-DETR through ONNX Runtime) re-verifies each detection on the desktop.
- **The longitudinal pass.** A scheduled pass over the confirmed record that
  proposes candidate patterns (trend, correlation, and co-occurrence) as
  hypotheses, always labeled as inference and never as measurement.
- **The Brain interface.** Model and memory management, learning and auditing,
  and reusable skills that add site-specific rules, across three sub-panels.
- **Acoustic pipeline.** Terrestrial and avian recognition on the field station,
  with a slot for an underwater passive-acoustic model.
- **On-device language-model analysis** of each observation, running locally.
- **Species reference data.** Guided taxonomic index building and per-species
  fetching of GBIF taxonomy and occurrence counts and IUCN Red List status, cached
  locally with fetch dates recorded.
- **Provenance-labeled reports** in PDF and CSV, with a cover page, executive
  summary, embedded charts, and a provenance panel.
- **Local storage tools**, including a configurable data directory and a safe
  export-then-reclaim flow.
- **A local overrides layer** that relocates machine-specific absolute paths out
  of the tracked configuration so ordinary use does not leak a local path.

### Changed

- Replaced V1's cloud and workflow stack (external detection workflows, a
  workflow-automation analyst, a hosted database, and external mapping and climate
  services) with a self-contained local application backed by SQLite and a
  FastAPI server serving a bundled interface.

### Notes

- The marine-sponge detection model from V1 carries forward as the default desktop
  verification model.
