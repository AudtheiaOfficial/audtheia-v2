# Audtheia V2 Glossary

A plain-language dictionary of the terms used across the Audtheia V2 master concept and platform. Definitions are grouped alphabetically. Where a term is a piece of recorded data, its provenance is noted, because Audtheia's central guarantee is that measured facts and downstream inferences stay permanently distinguishable.

## A

**Acoustic detection.** One of the two co-equal triggers for an observation. An onset in the audio stream (a bird call, a cetacean vocalization) starts a complete multimodal record, capturing the visual frames and sensor readings for the same window. Recorded with `trigger_source = audio`.

**AI HAT+ 2.** The Raspberry Pi accelerator board built around the Hailo-10H (40 TOPS, 8 GB onboard). In Audtheia it runs continuous vision only; its on-accelerator language-model capability is intentionally left unused on the field hot path.

**Analytics panel.** The interface section that presents biodiversity metrics and longitudinal trends derived from the observation record.

**Append-only archive.** The rule that the authoritative observation store is only ever added to, never overwritten or silently pruned. It protects longitudinal patterns, which are correlations across many records that destructive edits would corrupt.

**Audio windowing.** The rule for how long an acoustic clip is. Audio is captured from a continuous ring buffer so a clip can include pre-roll and post-roll around the event, bounded by a configurable cap. Whenever a clip is capped, the true event duration is still recorded, so there is never silent truncation.

**Authoritative store.** The desktop-held, long-term database that is the single source of truth for the full longitudinal record. The field station holds only a rolling buffer of recent data until it syncs to the desktop.

## B

**Baseline gist.** The compact running statistics (per period, per species, per signal) that the dream pass consolidates during its NREM-A phase. The regularity is extracted into this durable baseline before anything is pruned, so patterns are preserved.

**Benjamini-Hochberg q-value.** The false-discovery-rate control reported alongside each candidate pattern from the dream pass, so that patterns surfaced by generative search are not mistaken for established findings.

**Biomimetic heuristic.** The organizing analogy that structures the software (field station as cerebellum, desktop as cerebrum, the observation as episodic memory). It names event-driven sparsity at the system level and is a design guide only; it does not imply neuromorphic hardware, and the scientific guarantee remains the provenance system, not the metaphor.

**BirdNET-Analyzer.** The default terrestrial and coastal-avian acoustic model, run in full (not Lite) on the Pi CPU. It is swappable and supports custom training for local species; it is unsuitable underwater, where a marine model is used instead.

**Brain panel.** The interface section covering three areas: Models and Memory, Learning and Fine-tuning and Auditing, and Skills.

**ByteTrack.** The multi-object tracker that associates a detected animal across consecutive frames into a single event with a persistent track ID. It runs on the Pi CPU downstream of YOLO11 and prevents one lingering animal from being counted as hundreds of separate detections.

## C

**Candidate pattern.** The output of the dream pass: a hypothesis, tagged `dream`, reported with an effect size and the data span behind it, and never presented as an established fact.

**Cerebellum (field analogy).** The field station's role in the biomimetic heuristic: fast, reflexive detection, capture, validation, and forward-modeling, without deliberation or claim generation.

**Cerebrum (desktop analogy).** The desktop's role: high-accuracy verification (occipital), reports (prefrontal), and the dream pass (limbic consolidation and pattern extraction).

**Channel manifest.** The declared list, in `settings.json`, of which sensor channels a station has. The field QC engine compares each captured record against it to decide what is complete, missing, or in error.

**Controlled vocabulary (missing data).** A fixed set of status terms used to describe why a data point is absent (for example, not attempted, attempted and failed, not applicable), so a gap is always explained rather than left as an ambiguous blank.

**Cross-modal source attribution.** Any claim that a specific sound was produced by a specific visually detected animal. This is an inference, generated downstream and labeled `llm_inferred`, never asserted at capture; capture only records measured spatiotemporal co-occurrence.

## D

**data_source.** The provenance tag carried by every value written to the database, marking whether it is measured, referenced, model-inferred, human-expert, or pattern-derived. It is the technical heart of the measured-versus-inferred firewall.

**Detection as trigger.** The core architectural rule that nothing runs on a schedule except reports and dream passes; a vision or audio detection is what starts a complete observation. This is what makes the system energy-proportional.

**Detections panel.** The interface section showing the live visual feed and the full detection history by date, site, species, and confidence, including each detection's desktop verification result.

**Downscale (NREM-B).** The dream pass phase that prunes the derived dream working memory toward a bounded, salience-ranked set, keeping the strongest candidates. It never touches the authoritative observation archive.

**Dream pass.** The desktop-side longitudinal analysis whose purpose is to surface patterns humans cannot see. It runs as a two-phase NREM to REM cycle over the confirmed record and emits labeled candidate patterns, never authoritative facts.

## E

**Energy-proportional operation.** The property that energy spent scales with ecological events rather than wall-clock time, because detection triggers all work. It is what makes single-digit-watt, solar-capable, cloud-free field deployment feasible.

**Epoch, cycle, and pass.** The work-based budget units of the dream pass. An epoch is one NREM consolidation batch, a cycle is one full NREM to REM traversal (the commit unit), and a pass is all the cycles needed to clear the backlog since the last pass. Cancellation means pausing after the current cycle, never mid-write.

**Event memory.** The model of an observation as one multimodal record that binds every sensor stream captured in a single spatiotemporal window, mirroring how an organism binds several senses into one episodic memory.

## F

**Feature embedding.** An optional, settings-gated, off-by-default numeric fingerprint of a detection frame, forwarded to the desktop to enable similarity clustering and novelty detection. It is reserved for power- and storage-rich deployments because it inflates the buffer and sync payload.

**Field QC engine.** The deterministic predict, compare, correct loop on the Pi CPU that makes every observation complete and well-formed. It runs no language model on the hot path, assigns controlled-vocabulary status and QARTOD flags, computes provisional salience, and never writes free-form ecological claims. Records it cannot classify are stamped `qc_deferred`.

**Field station.** A deployed Raspberry Pi 5 plus AI HAT+ 2 with sensors, camera, and (for marine work) a hydrophone. It captures and validates observations and holds a rolling buffer until it syncs to the desktop.

## G

**GBIF taxonomic backbone.** The Global Biodiversity Information Facility taxonomy (kingdom down to species, with usage keys) shipped with the repository and queried at capture. Occurrence and IUCN data are fetched per target species at setup under the user's own credentials, so no bulk database is redistributed.

**Grey matter and white matter.** The design split between modules that transform data (monitor, QC, verify, dream) and the layer that transports it (the sync protocol and the observation schema). Transport must be lossless and low-latency, so it cannot tolerate type coercion, silent truncation, or lossy compression.

## H

**Hailo NPU and .hef.** The neural accelerator on the AI HAT+ 2 and the compiled model format it runs. Audtheia's custom YOLO11 is compiled to a `.hef` file (via the Hailo Dataflow Compiler, targeting the Hailo-10H) for continuous on-device vision.

**Hydrophone.** The correct underwater microphone for marine deployments (an analog element plus preamp into a USB audio ADC the Pi reads live), used instead of a condenser mic. Deploy-and-recover recorders are avoided because they do not stream live and so cannot fit the detection-triggered design.

## I

**IUCN Red List status.** The conservation-status reference for a species, fetched under the user's own credentials at setup and cached, with a snapshot date stamped on every dependent record. It is referenced data, not a measurement.

## L

**llm_inferred.** The provenance tag for downstream interpretation such as ecological role, rarity, anomaly flags, cross-modal source attribution, and skill outputs. It is always generated on the desktop, never at field capture, and always labeled as inference.

## M

**Measured versus inferred firewall.** The overarching guarantee that measured observations and downstream inferences are stored so they remain permanently distinguishable. It is enforced by the `data_source` tag and the field tier's refusal to write interpretive claims as data.

**Model trust.** A per-species reliability figure for a detection model, computed as a Laplace-smoothed precision from confirmed reviews and combined with detection evidence. It tells the user how much to trust a given model's call for a given species.

**Multi-station.** The capability for one desktop to pull from and unify the records of several field stations, merged by `station_id`, with no coordination needed because each station only writes its own observations.

## N

**NREM to REM cycle.** The mandatory two-phase structure of each dream pass. NREM consolidates regularity into the baseline and then downscales the working memory; REM then runs generative recombination over the compressed substrate. The order is fixed and REM is gated on desktop verification.

## O

**Observation.** The atomic unit of the record: one multimodal event memory, identified by a UUID and a human-legible name, carrying its `trigger_source`, representative frame, frame count, first and last seen times, duration, sensor readings, and provenance tags.

**Occipital gate.** The rule that NREM consolidation uses all synced observations, but REM generative integration runs only over observations the verification step has cleared. It keeps every candidate pattern anchored to confirmed detections.

## P

**Provenance.** The recorded origin and status of every value, spanning the `data_source` tag, missing-data vocabulary, QARTOD flags, and model and snapshot version stamps. Provenance, not the biomimetic metaphor, is Audtheia's scientific guarantee.

## Q

**QARTOD flags.** The IOOS Quality Assurance of Real-Time Oceanographic Data flag scale (1 pass, 2 not evaluated, 3 suspect, 4 fail, 9 missing) applied to marine sensor channels, chosen over a homegrown scheme so the output is reviewer-defensible.

**qc_state and qc_deferred.** The record-level quality-control status and the specific stamp for a record the deterministic field engine cannot classify, which is then deferred to the desktop where interpretation lives.

## R

**Representative frame.** The single highest-confidence frame chosen to stand in for a whole tracked event, stored with the observation instead of every positive frame.

**Reports panel.** The interface section that generates scheduled PDF and CSV outputs in which every value is labeled by its provenance.

**RF-DETR.** The high-accuracy transformer detection model run on the desktop through ONNX Runtime to verify saved detection frames and add the desktop's interpretive analysis. It runs on the desktop because its ONNX graph cannot currently be compiled to the Hailo NPU.

**Rolling-buffer policy.** The field storage rule that the Pi never silently evicts data. A high-water threshold triggers a storage panel and sync prompts; cleaning only ever removes already-synced records, and at a hard ceiling the station alerts and can pause capture rather than drop unsynced data.

## S

**Salience score.** A normalized 0 to 1 novelty measure combining detection confidence with a QARTOD-gated anomaly signal and a site-relative rarity proxy. The schema reserves provisional (Pi) and authoritative (desktop) slots and retains the ingredients, so the exact formula can evolve without schema changes.

**Skills.** Modular analysis units carrying a field-skill type tag that enforces the measured-versus-inferred firewall: deterministic, rule-based skills that are verifiable against the measurement (for example, a thermal-stress flag) may run in the field, while open-ended interpretive skills run downstream on the desktop.

**station_telemetry.** The schema table recording station health, observation effort, and optional energy figures (energy populated only when a meter is configured), kept separate from the observation data itself.

## T

**trigger_source.** The field on every observation recording what started it: `vision`, `audio`, or `sensor`. It makes the origin of each event explicit and queryable.

**Time base (UTC and GPS).** The rule that all timestamps are stored in UTC, sourced authoritatively from GPS when a fix is available, held by the Pi's battery-backed clock between fixes, and cross-checked against the desktop on connect. Local time is a display setting only.

## U

**UUID.** The globally unique identifier generated on the Pi at the moment of capture and used as every observation's primary key. UUIDs are collision-proof across unlimited independent stations, so no central coordination is needed.

## V

**Verification step (verify.py).** The desktop stage that re-scores field detections with RF-DETR, validates the field station's annotations, and adds the desktop's own interpretive points (ecological role, rarity, and the rest), all tagged as inference. It also acts as the occipital gate for the dream pass.

## Y

**YOLO11.** The custom-trained screening model compiled to `.hef` that runs continuously on the Hailo NPU, checking every frame up to NPU throughput. Because it is trained on the user's own target species it carries real species knowledge and is not a generic detector.
