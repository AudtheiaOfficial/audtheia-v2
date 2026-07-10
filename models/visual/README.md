# models/visual

The visual detection models. Which ones you need depends on how you run Audtheia. Every model is registered in the settings file with a path, a version, and a citation, so a report can always disclose which model produced a result. Model files are not committed to the repository because of their size (see .gitignore); setup downloads them, or you place them here by hand.

## Field station (Raspberry Pi)

- **pi**: the continuous screening model, a custom-trained YOLO model compiled to the field accelerator's `.hef` format. It runs on the Hailo NPU and checks every frame. Compiling a model to that format is an advanced, build-time workflow on an x86-64 Linux machine, documented in [docs/custom-models.md](../../docs/custom-models.md); it is never done on the Pi, and it is not needed to run a station, which only ever loads a finished `.hef`. YOLO models from Ultralytics are AGPL-3.0 licensed, so a detector you train is yours to use, and its license is your responsibility only if you distribute the model or run a public networked service. It does not affect Audtheia's own license, and it never touches the data you collect, which is always yours to publish.

## Desktop hub

- **desktop**: the high-accuracy verification model, RF-DETR run through ONNX Runtime on your computer. It re-scores each event and opens the analysis gate. RF-DETR cannot compile to the field accelerator, which is exactly why verification runs on the desktop rather than on the NPU. RF-DETR is Apache-2.0 licensed. The marine-sponge classifier that shipped with the earlier Audtheia platform carries forward here as the default verifier.

## Desktop hardware-free mode

The desktop folder also holds the screening detector for running with no field hardware at all, an RF-DETR model exported to ONNX. The reference build ships one pretrained on the common object set, licensed Apache-2.0, so a fresh install detects out of the box against a webcam, a stream, or a video file. Point it at your own species model to specialize it. See [docs/custom-models.md](../../docs/custom-models.md) for how to prepare either model, and the main [README](../../README.md) for how the two modes differ.
