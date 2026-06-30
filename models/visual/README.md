# models/visual

The two-tier visual detection models.

- pi: the continuous detection model, a custom-trained YOLO11 compiled for the
  field accelerator. It runs on the Hailo NPU at the field station. Compiling a
  model to that format is an advanced workflow on x86 Linux and is documented in
  docs/custom-models.md; it is not needed to run a station.
- desktop: the high-accuracy verification model, RF-DETR through ONNX Runtime. It
  runs on the desktop hub. RF-DETR cannot compile to the field accelerator's
  format, which is why verification runs on the desktop rather than on the NPU.

The marine-sponge classifier that shipped with the earlier Audtheia platform
carries forward here as the default desktop verification model.
