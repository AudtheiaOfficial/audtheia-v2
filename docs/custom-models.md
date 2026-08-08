# Custom Models Guide

Audtheia ships with a working reference model so you can see it run, but nothing about the platform is tied to one species or one place. The detection model, the acoustic model, and the species reference data are all configuration, not code: you point the settings file at your own models and your own target species, and the same pipeline runs for sponges, birds, fish, or reef invertebrates with no change to the software. This guide explains the models Audtheia uses, how to train your own, and how to prepare each one for the place it runs.

## The models, and where each runs

Audtheia uses a small set of models, and it matters which processor each one runs on, because that is what keeps a field station fast and low power.

| Model | Runs on | Format | Role |
|---|---|---|---|
| Field screening detector | The Hailo accelerator on a Pi station | A compiled `.hef` file | Watches every frame and flags what to record |
| Desktop verifier | Your computer | An RF-DETR model in ONNX | Re-scores each event at high accuracy and opens the analysis gate |
| Desktop-mode screening detector | Your computer (hardware-free mode) | An ONNX detector | Watches frames when you run with no field hardware |
| Acoustic model | The Pi's processor | BirdNET, or a marine model | Listens for sound events, and can open an observation on its own |

Every one of these is named in the settings file with a path, a version, and a citation, so a report can always disclose exactly which model produced a result. You never edit code to change a model; you change the path.

## Why the vision work is split in two

The most important thing to understand before training a detector is why the field station and the desktop use different vision models.

The field station screens with a YOLO family model because YOLO is the proven, supported path to the Hailo accelerator: it compiles cleanly to the accelerator's format and runs fast at very low power, which is what makes continuous, all day, solar capable detection possible. The desktop then re-verifies with RF-DETR, a transformer detector that is more accurate but which cannot currently be compiled to the Hailo accelerator, because its network graph fails the accelerator's compiler at the translation stage, a limitation confirmed across toolchains. So the design is deliberate: the field station does fast, cheap screening, and the desktop does slow, accurate verification, and neither compromises the other. When you train your own detector you are really training two models on the same species, a YOLO for the field and an RF-DETR for the desktop, though you can start with just one depending on how you deploy.

## Training a detector for your species

The starting point for either model is a labeled image set of your target species, which you can build and annotate in a tool such as Roboflow or any labeling platform you prefer. From that one dataset you train the detectors you need.

- For a **field station**, train a **YOLO** model, then compile it to the accelerator format as described below. YOLO models from Ultralytics are licensed AGPL-3.0, so a model you train is yours to use, and its license is your responsibility if you distribute the model or run a public networked service; it does not affect Audtheia's own license, and it never touches the data you collect, which is always yours to publish.
- For the **desktop verifier** and for the **desktop hardware-free mode**, train or fine-tune an **RF-DETR** model, which is Apache-2.0 licensed and therefore free of copyleft obligations, and export it to ONNX.

You do not have to train both at once. A desktop-only user needs only the RF-DETR path; a full field deployment uses the YOLO field model plus the RF-DETR desktop verifier.

## Preparing the field model: compiling YOLO to the accelerator format

The Hailo accelerator does not run a YOLO model directly. The model must be compiled ahead of time into the accelerator's own `.hef` executable format, and this is the one genuinely advanced step in the whole platform. It is worth being clear about what it is and is not.

**This is a build-time step, done once before deployment, not something the Pi or the application ever does at runtime.** A field station only ever loads a finished `.hef`; it never compiles one. You can compile the file yourself on a capable computer, or use one that has already been compiled for your species.

The compile has firm requirements:

- It runs only on an **x86-64 Linux** machine. It cannot be done on a Raspberry Pi or any ARM device, and it is not part of installing or running Audtheia.
- It uses the **Hailo Dataflow Compiler**, downloaded from the Hailo Developer Zone, matched to the accelerator. For the AI HAT+ 2's Hailo-10H, target the Hailo-10H architecture specifically, since the compiler distinguishes accelerator families.
- It is resource intensive, especially the calibration stage, so a machine with **32 GB of RAM or more** is recommended to avoid running out of memory during quantization.
- It needs a small **calibration image set**, a representative sample of frames, which the compiler uses to convert the model to the accelerator's efficient integer arithmetic.

The path, at a high level, is:

1. Train your YOLO model on your species.
2. Export it to ONNX.
3. Run the Hailo Dataflow Compiler on the x86-64 Linux machine: it parses the ONNX, optimizes and quantizes it against your calibration images, and compiles it, targeting the Hailo-10H architecture.
4. The result is a single `.hef` file.
5. Copy that `.hef` to the station and set its path in the settings file under the station's field model.

Ultralytics publishes a Hailo export integration that wraps much of this, and the Hailo Model Zoo provides reference flows and pre-built examples, both of which are good starting points. If compiling is beyond your setup, a `.hef` compiled for your species by someone else drops straight in; only the file matters to the station, not how it was made.

## Preparing the desktop model: from a Roboflow project to a labeled detector

The desktop verifier and the desktop hardware-free detector both load an RF-DETR model in ONNX, with a small labels file beside it. No special compilation is needed; the model runs through ONNX Runtime on your computer's own processor, so there is no accelerator, no compiler, and no calibration step. If your model lives in a Roboflow project, the full path from project to working, named detector is below, in order.

Two different things are downloaded from Roboflow, and confusing them is the most common mistake. The trained **model weights** become the `.onnx` the app loads. The **dataset in COCO format** supplies the class names. Downloading only the dataset gives you no model; downloading only the weights gives you a model that labels detections by number. You need both.

Have three things ready: your Roboflow **API key** (Account, then Roboflow Keys), the **version number** you trained, and the **RF-DETR size** you trained (Nano, Small, Medium, Base, or Large).

### 1. Download the trained weights

`version(N).model` is a property, so there are no parentheses after `model`:

```
pip install roboflow
python -c "from roboflow import Roboflow; Roboflow(api_key='YOUR_KEY').workspace('YOUR_WORKSPACE').project('YOUR_PROJECT').version(N).model.download()"
```

This writes `weights.pt` into the current folder. Downloading weights is a paid Roboflow feature; if it is refused, open that version on the Roboflow website and use **Download Weights** instead, or use the checkpoint from wherever you trained the model.

### 2. Export the weights to ONNX

```
pip install "rfdetr[onnxexport]"
python scripts/export_rfdetr_onnx.py "path/to/weights.pt"
```

This writes `weights.onnx` next to the checkpoint. The script assumes RF-DETR **Small**; if you trained a different size, open `scripts/export_rfdetr_onnx.py` and change `RFDETRSmall` to your size before running it. Move the `.onnx` into `models/visual/` and give it a clear name, for example `bird_rfdetr.onnx`. The `.onnx` is the only file the app loads at run time; the `.pt` checkpoint is just the source, and you can keep it elsewhere as a backup.

### 3. Give the model its class names

An RF-DETR ONNX carries no class names, so Audtheia reads them from a file named after the model with a `.labels.json` suffix, placed beside it: `bird_rfdetr.onnx` pairs with `bird_rfdetr.labels.json` (the loader also accepts `.labels.txt` or `.names`). Without it, detections are still recorded but labeled by numeric id, which is exactly what you see if this step is skipped. Download the dataset in COCO format and build the file from its class list:

```
python -c "from roboflow import Roboflow; Roboflow(api_key='YOUR_KEY').workspace('YOUR_WORKSPACE').project('YOUR_PROJECT').version(N).download('coco')"
```

That creates a folder holding `train/`, `valid/`, and `test/`, each with an `_annotations.coco.json`. Build the labels from any one of them, using the real path it wrote (not a placeholder), and point `--model` at your renamed `.onnx`:

```
python scripts/build_porifera_labels.py --coco "path/to/train/_annotations.coco.json" --model "models/visual/bird_rfdetr.onnx"
```

It writes `bird_rfdetr.labels.json` beside the model. The script's name is historical; it works for any species. It warns if the category ids do not fit the model's head, which means the dataset version and the model version differ. The dataset folder is only needed for this step and can be deleted afterward.

Two properties of RF-DETR make the names non-obvious, and both are handled for you. First, RF-DETR uses **1-based** class ids with a reserved dummy slot at index 0, so a detection reported as class 21 corresponds to the annotation category whose id is 21. Second, RF-DETR sizes its class head to the highest category id it was trained against plus one, not to the number of classes you actually use, so a model with 130 live classes can export a much wider head (the reference Porifera model's head is 366 wide, ids 1 to 130 in use and the rest inert). The authoritative id-to-name mapping is therefore the `categories` array in the model's own COCO training data, which is exactly what the step above reads.

### 4. Point Audtheia at the model

In Settings, set the station's **Desktop screening model** and, under **Model paths**, the **verification model**, both to `models/visual/bird_rfdetr.onnx`. The labels file is found automatically from its matching name; you never enter its path. Restart capture so the model reloads, and detections now carry species names. The reference Porifera model already ships with its own `porifera_rfdetr.labels.json`, so keeping that model's filename gives correct names with no further work.

## Preparing the desktop hardware-free detector

The hardware-free desktop mode screens frames with an ONNX detector too. Export your trained detector to ONNX and set the station's Desktop screening model path to it. The screening step accepts either a **YOLO** or an **RF-DETR** export and detects which from the model's own outputs, so the same RF-DETR file you use to verify can also screen; there is no need for a separate YOLO model on the desktop. No model file ships with the repository, so this path is always your own exported file.

## Acoustic models

The audio component is model agnostic and chosen in the settings file.

- **Terrestrial and coastal-avian sites** use BirdNET, running on the Pi's processor. You can train a custom classifier for local species, which is well supported: collect local recordings, organize them into one folder per species, run BirdNET's training, and drop the resulting classifier into the custom acoustic models folder, then select it in the settings. BirdNET is licensed CC BY-NC-SA, which permits non-commercial use with attribution and share-alike, so keep that in mind if your work is commercial.
- **Underwater sites** need a marine passive-acoustic model, because BirdNET is trained on in-air bird sound and is not useful on a submerged hydrophone. A marine model goes in the marine acoustic models folder and is selected in the settings. Choosing the best current open marine model for your target sounds, whether cetaceans, fish choruses, or a snapping-shrimp soundscape, is an active area, so pick and validate one for your site.

## Pointing Audtheia at your models

Every model is registered in the settings file, never in code. Each entry carries three things:

- a **path** to the model file,
- a **version** string, so you can tell which iteration of a model produced a result, and
- a **citation**, so a report can credit the model's source.

Because these travel with every observation and every report, your outputs always disclose which model, at which version, produced each detection. That is part of what makes Audtheia's records defensible: a reader can always trace a result back to the exact model behind it.

## Where to go next

- To build the physical station that runs a field model, see the [hardware guide](hardware.md).
- To set up the desktop and run either mode, see the [README](../README.md).
- To understand how the desktop turns verified detections into longitudinal insight, see the [dream pass guide](dream-pass.md).

## References

- Hailo Dataflow Compiler workflow (ONNX to HEF, x86-64 Linux, calibration and memory needs): https://developer.ridgerun.com/wiki/index.php/Hailo/Hailo-8/AI_Software_and_Tools/Hailo_Dataflow_Compiler
- Ultralytics Hailo export integration: https://docs.ultralytics.com/integrations/hailo
- Hailo Model Zoo (reference building and evaluation flows): https://github.com/hailo-ai/hailo_model_zoo
- RF-DETR (Apache-2.0, ONNX export): https://github.com/roboflow/rf-detr
