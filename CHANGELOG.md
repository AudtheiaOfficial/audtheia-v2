# Changelog

All notable changes to Audtheia V2 are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Train tab groundwork.** Four new tables for the coming Train tab, added
  without changing any existing table. `frame_box_correction` holds an expert's
  corrections to boxes on saved frames as an append-only record, so the captured
  annotations are never altered. `training_runs`, `trained_models` and
  `model_activation_log` record what was trained, the numbered model versions
  that resulted, and every time one was switched on or reverted. All three are
  marked as model output or expert action, never as measurement. A new
  `training` block in the settings, with validation and sensible defaults, and
  `scripts/migrate_add_training.py` for adding the tables to an existing
  database explicitly. A settings file without the block still loads.
- **Train tab: Dataset.** A new Train item in the sidebar with a Dataset view that
  turns the events experts have judged into a training dataset. Choose which of
  three checks an event must have passed (Expert Confirmed, QC Passed, Verified);
  the counts show how many events qualify under every combination before
  anything is written. A preview reports exactly what would be built, with the
  split into training, validation and test, the classes and how often each is
  represented, the background and model-labelled share, and a plain warning for
  anything the person should read first. Rare classes can be kept with a warning,
  left out, or merged into their genus or family through the taxonomic backbone
  that ships with Audtheia, class by class. Building writes a new folder under
  the datasets directory with a COCO layout and a YOLO text layout for visual
  data, or WAV windows with a table for acoustic data, plus a manifest recording
  every choice, the seed, each example's source and whether its label is an
  expert's or the model's. The record and the data folder are only ever read.
  The same selection builds once and is then recognised, and a failed build
  leaves nothing half written.
- **Dataset rules.** Events are divided as whole events, so no event contributes
  to more than one of training, validation and test. Frames an expert marked
  Inaccurate are never used. An event an expert rejected becomes background, up
  to a cap. A relabelled event is learned under the expert's species. Examples
  that carry the model's own label are used for training only, up to a cap that
  defaults to none. A class with too few expert-judged events to hold one out is
  trained on and marked untested.

### Changed

- **Per-frame review wording.** The two verdict buttons in the Detections frame
  review now read Accurate and Inaccurate, and the matching summary, help text,
  and "mark all" confirmation use the same words. The stored verdicts
  (`accurate`, `inaccurate`, `cleared`) and every summary figure are unchanged,
  so existing reviews read exactly as before. The summary metric keeps its
  statistical name, Frame precision: the share of judged frames marked Accurate.
- **Sub-panel switching.** Choosing a sub-panel now changes only the sub-panels of
  the panel it belongs to, and each panel remembers its own choice, so Brain and
  Train cannot hide each other's content. A choice remembered before this change
  still applies to Brain.

## [2.0.0] - 2026-08-15

Audtheia V2 is a fresh, fully offline redesign of the Audtheia platform in its own
repository. Where V1 relied on cloud workflows and external services, V2 runs the
entire pipeline on hardware the user owns, with no internet connection required at
runtime.

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
