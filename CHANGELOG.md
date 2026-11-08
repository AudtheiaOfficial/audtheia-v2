# Changelog

All notable changes to Audtheia V2 are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2026-11-05

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
- **Train tab: Dataset.** A new Train item in the sidebar with a Dataset view
  that turns the events experts have judged into a training dataset. Choose
  which of three checks an event must have passed (Expert Confirmed, QC Passed,
  Verified), with the first two chosen at the start; the counts show how many
  events qualify under every combination before anything is written, and when
  nothing qualifies the preview names the ticked check that no event has passed.
  A preview reports exactly what would be built, with the split into training,
  validation and test, the classes and how often each is represented, the
  background and model-labelled share, and a plain warning for anything the
  person should read first. Rare classes can be kept with a warning, left out,
  or merged into their genus or family through the taxonomic backbone that ships
  with Audtheia, class by class. Building writes a new folder under the datasets
  directory with a COCO layout and a YOLO text layout for visual data, or WAV
  windows with a table for acoustic data, plus a manifest recording every
  choice, the seed, each example's source and whether its label is an expert's
  or the model's. The record and the data folder are only ever read. The same
  selection builds once and is then recognised, and a failed build leaves
  nothing half written.
- **Dataset rules.** Events are divided as whole events, so no event contributes
  to more than one of training, validation and test. Frames an expert marked
  Inaccurate are never used. An event an expert rejected becomes background, up
  to a cap. A relabelled event is learned under the expert's species. Examples
  that carry the model's own label are used for training only, up to a cap that
  defaults to none. A class with too few expert-judged events to hold one out is
  trained on and marked untested.
- **Evaluation measures.** The standard measures for judging a trained model on
  events it never saw, written into Audtheia itself so they run without the
  training libraries. For vision: mAP at IoU 0.50 and over 0.50 to 0.95, computed
  as the COCO evaluation computes it; average precision per class; precision,
  recall and F1 at a chosen confidence and at the confidence that gives the
  highest F1; a confusion matrix with a background row and column; and top-1
  class accuracy on matched boxes, defined exactly and never presented as if it
  were mAP. For sound: average precision per class and averaged over classes;
  ROC area; precision, recall and F1 at a chosen and at the best threshold; a
  confusion matrix with a background row and column; reliability bins; and
  per-class logistic calibration of scores following Wood and Kahl (2024), which
  gives each class a threshold. A figure that cannot be defined, such as the
  recall of a class with no test examples, is reported as not defined and never
  as zero. Every measure is checked in the test suite against an independent
  reference implementation (pycocotools, faster-coco-eval, supervision and
  scikit-learn), which are used by the tests only.
- **Results figures.** The stored measures are turned into the series, matrices
  and tables the Results view will draw, each with a plain-language help line and
  a CSV of the same numbers. Classes with no test examples are marked not tested.
  A challenger-versus-current table prints both models on the same test set with
  their differences, restricted to the shared classes as well, and never states
  which model is better.
- **Train tab: training components.** The training libraries, PyTorch and the
  RF-DETR training library for images, and TensorFlow and the TFLite runtime for
  sounds, are installed only on request into their own folder beside the
  application, one per engine, so the everyday application and the field station
  never carry them. Before anything is downloaded, Audtheia asks pip in dry-run
  mode exactly which files this computer needs and reads each file's size from
  the package servers, and the button shows that real size. The PyTorch build is
  chosen for the computer: CUDA 13 or CUDA 12.6 for an NVIDIA card depending on
  its driver, XPU for Intel Arc graphics, ROCm for an AMD card on Linux, and the
  processor build otherwise, including for Intel Iris Xe graphics, which PyTorch
  cannot train on. On Windows, an installation folder too long for the deepest
  library files is refused before downloading, with the fix. After installing,
  the libraries are loaded and checked before they are marked ready. A failed
  install can be removed in one step, and removal deletes only that folder.
- **Train tab: Train and Runs.** A new Train view describes this computer in
  plain words, measures how fast it trains the chosen size, estimates the least
  time and the disk space a run will take, and starts a run from a built dataset
  with a chosen model size, model name, starting point and speed. The name is
  checked as it is typed and shows the resulting Audtheia-Name-V1. An ONNX file
  is refused as a starting point, because it cannot be trained further. The Runs
  view follows a run every few seconds with its epochs, best validation score,
  figures per epoch and log, without losing its place, and offers Pause, Resume
  and Cancel. Training is saved at the end of each epoch, so Resume continues
  from the last finished epoch, or starts again from the beginning when none had
  finished, and the pause message says which. A run trains in its own process
  with no window of its own and writes its output as UTF-8, so the training
  library's tables of figures print on any computer. Closing the window does not
  stop a run, and a run whose process disappears, for example because the
  computer was switched off, is marked interrupted and resumes from its last
  saved epoch. Only one run trains at a time.
- **Train tab: sound models.** The Train view now trains sounds as well as
  images. For a sound dataset, a fixed base sound model summarises every window
  and Audtheia trains only a small final layer that turns that summary into the
  dataset's species, with the binary cross-entropy of each species, dropout and
  early stopping on the validation score. The terrestrial and avian engine uses
  BirdNET 2.4, downloaded once from BirdNET's Zenodo record into the sound
  training environment and checked against its description in the settings
  before use. Its licence, CC BY-NC-SA 4.0, is shown before a run starts. The
  windows are brought to the engine's sample rate, each species is calibrated on
  the validation windows with a per-species threshold that is recorded but not
  applied at the station, and the test windows are scored. The base model and
  the trained layer are joined into one TFLite file that takes raw audio and
  gives one score per species before the sigmoid, so a station's acoustic
  adapter loads it unchanged. Before a run is called finished, that file is
  loaded through the station's own adapter and its scores are compared with the
  trained layer's. A run resumes after its last finished epoch with exactly the
  weights it would have had without the pause. The marine and general engine
  has no base model yet and says so.
- **Gentle and Full speed.** Gentle uses half the processor threads at a lower
  priority, feeds one image at a time and adds the learning signal up to the
  same batch, so it trains with exactly the settings Full speed uses: the same
  seed, epochs, batch, image size, precision, learning rates and model size. A
  check refuses to start a run if the two would differ in any of those. Gentle
  is the default on a computer without a supported graphics card.
- **Training safety.** A run will not start on battery power and pauses, keeping
  every finished epoch, when the charger is removed, the battery falls below its
  limit, free disk space falls below its limit, the graphics card is almost out
  of memory, or the computer reports that it has been slowing down because of
  heat for two minutes. Where the computer does not report heat, the Train view
  says so instead of implying protection. Training only reads the dataset folder
  and writes only inside its own run folder, never the record or the data
  folder. While capture runs on the same computer, a run starts only after the
  person chooses to continue.
- **One identity per species in every dataset.** The same organism can reach the
  record under several labels: a model's own class name, BirdNET's joined
  scientific and common name, or the accepted name an expert chose. Before any
  class is counted, every label is matched to the shipped taxonomic backbone, by
  the key the record carries, an exact scientific name, or the scientific part of
  a BirdNET label, with synonyms followed to the accepted taxon, and the classes
  that are one taxon are merged. A label no rule can read, such as a model's
  class name, is listed in the preview with the reason, for an expert to map
  once to its species through the backbone search or to keep as its own class,
  for a morphotype or an informal name. Nothing is merged by guessing, an
  ambiguous name is never resolved, and the record is never changed: the
  decisions are kept in a small label map beside the datasets, each marked as an
  expert's. The counts before building use the same identities, and the dataset
  manifest records how every class was identified.
- **Train tab: Results.** A finished run is scored on its test set, beside a
  model already in use that the person chooses, by the same decoding the app
  uses during capture and the same metric code, on the same images or sound
  windows. Classes are matched by taxon identity, so the model in use is credited
  with a species however its labels spell it. A first comparison covers every
  class in the test set, with the species the model in use cannot name listed
  and counted as missed, as they would be in the field, and a second covers only
  the species both models know. Boxes the model in use draws for species outside
  the dataset are counted and reported. The share of test boxes that a model
  proposed is stated, with the caveat that a model scored against boxes it drew
  itself is favoured. The app prints the figures and their differences and never
  says which model is better. The panel shows the headline figures, every chart
  of the evaluation, the figures for each class, calibrated thresholds for sound
  models, and example test images with their truth and predicted boxes. A run
  whose dataset held no test examples is said to have no results, and is not
  offered for scoring; its run log and epoch table say plainly when there were
  no validation or test examples, so a falling training loss is not mistaken for
  a model ready for the field.
- **Charts in every theme, saved as pictures and tables.** Charts are drawn in
  the browser from a colour-blind-safe series palette defined for each of the
  seven themes, each colour at least 3:1 against the surface and the page. Each
  chart has Save PNG and Save CSV, series that can be switched on and off, and
  the pictures are saved into the run folder from the same drawing shown on the
  screen. The evaluation and one CSV per chart are written inside the run folder
  only.
- **Box tool.** In the Detections frame review, each frame has a Boxes button.
  An expert can redraw or remove the model's box, or draw a box the model missed
  with its taxon chosen from the backbone search, and move or delete that box
  later. Each change is appended as the expert's own record and is used only
  when building training datasets, where the frame is marked as edited and the
  box is counted as expert-drawn. The captured frame, its annotation file and the
  model's detections never change.
- **Large allocation units are named before installing.** A drive formatted with
  large blocks gives every file at least one whole block, so the training
  libraries, which are mostly small files, can take many times their real size.
  The install plan now reads the block size of the drive the environments go to
  and, above 64 KB, says plainly how much more space they could take and how to
  choose a folder on another drive, before anything is downloaded.
- **Train tab: Models.** A finished run becomes a model when a person adds it, as
  the next numbered version of its name, Audtheia-Name-Vn, in a folder of its own
  under models/trained with the model file, its labels, the PyTorch checkpoint or
  the trained sound layer and calibration, the test figures and a model card. A
  folder is never overwritten and a run is never added twice. The licence is
  inherited from what the model was trained from, and recorded as not confirmed
  when it cannot be known, such as for a person's own starting checkpoint.
- **Switching a model on, and back.** A model can be switched on as the desktop
  verification model, a station's desktop screening model, or a station's sound
  model. The change is saved through the guarded settings writer into this
  computer's local settings file, so the committed settings file stays byte for
  byte the same, and every switch is logged with the slot before and after so
  Put back the previous model restores it exactly. A slot changed elsewhere in the
  meantime is not reverted blindly. A model with a non-commercial licence, such as
  a sound model built on BirdNET, asks for a typed acknowledgement first. For a
  field station, the sound model and its labels can be sent with the configuration
  to a Pi that has been connected once.
- **Model labels and the Model filter.** Detection cards say which models
  screened and verified each event, and Audio cards which model heard it, read
  from the versions stored with the event, with "model version not recorded" where
  none was stored. Detections and Audio gain a Model filter listing the versions
  actually present, including events without one. Reports and CSV exports already
  carry these versions, so a switched-on trained model is named in them by its
  registry name.
- **Sharing packages.** A model package zips the model's folder, whose model card
  front matter a Hugging Face model page reads. A dataset can be exported as COCO
  with images, YOLO text, or sound windows, each with its class list and manifest.
  Packages are written beside the datasets and never overwritten, and nothing is
  uploaded by Audtheia.
- **Training guide.** docs/training.md describes the Train tab end to end,
  including what data a model needs, the identity rules, the comparison, switching
  models on, and how to share with Hugging Face and Roboflow. The glossary, the
  custom models guide and the README point to it.
- **Removing a run's files.** A run that finished, was cancelled or failed, and
  was not added as a model, can have its files removed from the Runs view to free
  the disk, after a second press to confirm. Only the checkpoints, the exported
  model and the training library's output are deleted. The record, the run
  settings, the log, the figures by epoch and any test scores are kept, with a
  note of when the files were removed and how much space that freed. Such a run
  can no longer be scored again or added as a model, and the Runs view shows how
  much space each run takes.

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
- **Number fields follow the theme.** Number inputs, such as the dataset and
  training settings, now use the theme's colours and borders like every other
  field, instead of the browser's plain white box.
- **Sound run cards** name the base model as such, Mode and base model, rather
  than a model size.
- **Saving settings that change nothing committed leaves the file untouched.** A
  save whose committed content is unchanged no longer rewrites or reformats
  config/settings.json.
- **Model slots live in the local settings file.** Any member of the desktop
  verification, desktop screening or station sound model slots that differs from
  the committed value is kept in config/settings.local.json, so which model a
  computer uses is a property of that computer. A slot already published in the
  committed file is read exactly as before.
- **Scenario settings files no longer read the machine's local overrides** unless
  they sit in a config folder, so two temporary configurations named
  settings.json in one folder cannot share a stale local file. This only affects
  test configurations; the application's own settings file is unchanged.
- **The test suite refuses to run in a working installation.** tests/run_all.py
  stops, naming what it found, in a folder that holds a database, a local
  settings file or a machine-specific training path, because several tests build
  their configurations from config/settings.json and one of them removes a
  training environment it installed. The shared test fixture also keeps every
  training folder inside its own temporary repository, whatever the copied
  configuration says. Run the suite in a fresh clone, as CONTRIBUTING.md and the
  setup guide describe.
- **Tests close every database they open**, so the temporary folders they use can
  be deleted on Windows, and remove every scenario file they write, so a full run
  leaves the checkout exactly as it found it.
- **The training run check compares folder contents the same way on every
  system.** It listed changed files with the system's own path separator, so on
  Windows the database was reported as an unexpected change. Paths are now
  compared in the forward-slash form on every system.
- **A failing check file shows every failed check.** tests/run_all.py now prints
  each failed check of that file first, wherever it appears in the output,
  followed by the end of the output, instead of only the last twenty lines.
- **The Zenodo record names its version.** .zenodo.json carries the release
  version, so the archived record is labelled 2.1.0.
- **Results match the model in use by species for datasets built earlier.** A
  dataset built before species identities were resolved names some classes by a
  plain label. Each dataset class is now resolved by the same rules as the model's
  labels, so a species the model in use knows is no longer counted as one it
  cannot name. A species two classes of one dataset both resolve to is left
  unmatched rather than credited to either.
- **A model's own folder names the model it was compared with by its file.** The
  full path of that file belongs to the computer that scored it and can name an
  account or a drive, so it stays in the run folder and is no longer copied into
  metrics.json, which is packaged and shared. The model card also reads "An image
  detection model".
- **Citation points to every version.** The README badge, the citation text,
  the BibTeX entry and CITATION.cff now carry the concept DOI
  10.5281/zenodo.21940754, which always resolves to the latest archived release.
  CITATION.cff also lists the 2.0.0 version DOI and the release date.
- **Dataset rules version 2.** Datasets built from now on carry rules version 2,
  because merging labels of one taxon can change what a selection holds. A
  dataset built before stays valid and is recognised as built under version 1.
- **PyTorch download locations.** The training install settings now hold the
  official PyTorch download locations instead of empty values, which by design
  blocked installing.
- **The image training library is pinned.** The training install now asks for
  RF-DETR 1.12.0, the release Audtheia's image training was verified with on
  Windows, so every installation trains with the same library. A different
  release, installed by hand, still runs with a note in the run's log.

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
