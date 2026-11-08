# Training your own models

The Train tab turns the events experts have reviewed into training datasets, trains image and sound models on this computer, scores them on events they never saw, and switches a chosen model on where it should be used. Everything runs offline except the one-time download of the training libraries and, for sound models, the base model. Nothing is uploaded anywhere.

The tab has five sections: Dataset, Train, Runs, Results and Models.

## What you need first

- **Reviewed events.** A model learns from an expert's judgement, not from the model that made the original call. Confirm, relabel or reject events in Detections and Audio first.
- **Enough events per species.** Events are divided whole into training, validation and test, so no event appears in two of them. A species needs at least three expert-judged events before one can be held out for validation and one for test. With fewer, the species is still trained on but has no test score, and the run cannot be compared with anything. Five or more events per species give a first meaningful figure; more give a firmer one.
- **The species index.** Run the species data setup once so labels can be matched to the GBIF backbone.

## Dataset

1. Choose images or sounds, and optionally a station.
2. Choose which checks an event must have passed: Expert Confirmed, QC Passed and Verified. The counts show what every combination would keep before anything is written. Expert Confirmed and QC Passed are chosen at the start.
3. Press **Preview dataset**. The preview reports the split, the classes and how often each is represented, background examples, and anything to read first.
4. **One identity per species.** Every label is matched to the backbone: by the key stored with the record, an exact scientific name, or the scientific part of a BirdNET label, following synonyms to the accepted name. Labels that are one species become one class. A label no rule can read, such as a model's own class name like `northern-cardinal`, is listed with the reason. Press **Map to a species** and choose it from the backbone search, or **Keep as its own class** for a morphotype or an informal name. Your decision is kept in `taxon_label_map.json` beside the datasets and can be undone; the record itself never changes.
5. **Rare classes.** For each class below the threshold choose to keep it with a warning, leave it out, or merge it into its genus or family through the backbone.
6. Press **Build dataset**. A folder is written under the datasets directory with a COCO layout and a YOLO text layout for images, or WAV windows with a table for sounds, plus a manifest recording every choice, the seed and the origin of every label. Building the same selection again finds the first dataset.

**Boxes.** Boxes on saved frames are the model's own unless an expert changes them. In Detections, open an event's frames and press **Boxes** on a frame to redraw or remove the model's box, or to draw a box the model missed with its species. These changes are your own record, used only when building datasets, and the captured frame never changes. A dataset built afterwards uses them and counts them as expert-drawn.

## Installing the training components

The first time, the Train section offers to install the training libraries into a separate folder, one for images and one for sounds, set by `training.paths.envs_dir`. The plan shows the exact packages and their measured download size before anything is downloaded, and needs the internet once.

The plan also reads the allocation unit of the drive. A drive formatted with large blocks, such as some external drives, gives every file at least one whole block, and these libraries are tens of thousands of small files, so they can take tens of gigabytes more than their real size. When the plan warns about this, set `training.paths.envs_dir` to a folder on a drive with normal blocks, for example `C:\atrain`, before installing.

## Training

- **Name.** The model is named `Audtheia-<Name>-V<n>`; a new run of the same name becomes the next version and never overwrites an earlier one.
- **Gentle or Full speed.** Gentle uses fewer processor threads at a lower priority so the computer stays usable and cooler; the learning itself is the same.
- **Safety.** A run does not start on battery power, and pauses, keeping every finished epoch, when the charger is removed, the battery or free disk falls below its limit, the graphics card is nearly out of memory, or the computer reports heat slowdown for two minutes.
- **Runs.** Follow progress, pause, resume or cancel. A run keeps going if the window is closed. When a dataset has no validation or test examples, the log and the epoch table say so plainly: such a run checks that training works and is not a model for the field.
- **Removing a run's files.** A run that finished, was cancelled or failed, and was not added as a model, shows how much disk it takes and offers **Remove its files**. Press it twice to confirm. The checkpoints and the exported model are deleted; the record, the log, the figures by epoch and any test scores stay. Such a run can no longer be scored again or added as a model. A run added as a model keeps its files, and its model lives in its own folder.
- **Training library.** Image training installs RF-DETR 1.12.0, the release it was verified with, so every installation trains the same way.
- **Sound models** train a final layer on top of BirdNET 2.4, downloaded once from Zenodo. BirdNET is licensed CC BY-NC-SA 4.0, so a sound model built on it may be used for non-commercial purposes only, with attribution, and shared only under the same terms.

## Results

Choose a finished run, choose the model already in use to compare it with, and press **Score on the test set**. Both models are run through the decoding the app uses during capture, on the same test images or sound windows, and scored by the same code.

- Classes are matched by species identity, so the model in use is credited with a species however its labels spell it.
- The first comparison covers every class in the test set. A species the model in use cannot name counts as never found by it, as it would be in the field, and is listed. The second comparison covers only the species both models know, which is the like-for-like reading.
- When the test boxes were drawn by a model, the results say so: a model scored against boxes it drew itself is favoured. Only boxes an expert drew or adjusted are free of that bias.
- The app prints the figures and their differences and never says which model is better. A figure from one test split is an estimate.
- Every chart has **Save PNG** and **Save CSV**, and the pictures and tables are also saved in the run folder.

## Models

- **Add as a model.** A finished run becomes a model only when you add it. Its folder under `models/trained` holds the model file, its labels, the PyTorch checkpoint (images) or the trained layer and calibration (sounds), `metrics.json` and `MODEL_CARD.md`. The licence is inherited from what it was trained from; when it cannot be known, as for your own starting checkpoint, it is recorded as not confirmed.
- **Switch on.** Choose where the model is used: the desktop verification model, a station's desktop screening model, or a station's sound model. The choice is saved in this computer's local settings file, `config/settings.local.json`, so the committed `config/settings.json` never changes. It is used the next time Audtheia starts. A model with a limiting licence asks you to type the acknowledgement first.
- **Put back the previous model.** Restores the slot exactly as it was before. It is refused if the slot was changed elsewhere in the meantime, so nothing is overwritten blindly.
- **Field stations.** For a station's sound model, **Send to the field station** sends the configuration and the model files to a Pi that has been connected once; the station uses them after it restarts. A field station screens images with a model compiled for its accelerator (`.hef`), described in the [custom models guide](custom-models.md), so an image model switched on here is used for desktop capture only.
- **Records keep their model.** Every event stores the versions of the models that produced it. Cards show "Screened by", "Verified by" and "Heard by", the **Model** filter in Detections and Audio lists the versions present, and reports and CSV exports carry the same columns. Switching a model never changes a record already captured.

## Sharing

Packages are zip files written beside the datasets, in a `packages` folder; nothing is uploaded by Audtheia.

- **Model package.** The model's folder, with `MODEL_CARD.md`, whose front matter a Hugging Face model page reads. To publish it, create a model repository on Hugging Face and upload the folder's files; state the licence of your training data in the card first.
- **Roboflow.** A dataset export in the COCO layout includes the images and can be imported into a Roboflow project. Roboflow's custom-weights upload accepts RF-DETR detection weights with model types `rfdetr-nano` through `rfdetr-2xlarge`; the package's `checkpoint.pth` is the file to give it, following [Roboflow's upload steps](https://docs.roboflow.com/models/model-weights/upload-custom-weights).
- **iNaturalist** does not take uploaded models; its identification model is trained by iNaturalist from community observations. What you can contribute there are observations with photos or sounds, from your own iNaturalist account.

## Where things are

| What | Where |
|------|-------|
| Datasets and the label map | `exports/train/` |
| Packages | `exports/packages/` |
| Training runs | `models/trained/runs/` |
| Models | `models/trained/Audtheia-<Name>-V<n>/` |
| Training libraries | `training.paths.envs_dir` |
| Switched-on models | `config/settings.local.json` |

None of these are committed to the repository.
