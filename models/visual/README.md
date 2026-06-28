# `models/visual/` — Visual detection models (two-tier)

- **`pi/`** — `yolo11_porifera.hef` (or your target-species equivalent). Custom-trained YOLO11, compiled to `.hef`, runs continuously on the Hailo-10H NPU. The ONNX→HEF compile step is an x86-Linux advanced workflow targeting the Hailo-10H architecture — see `docs/custom-models.md` (Session 23). **Not a runtime dependency** on the Pi.
- **`desktop/`** — `porifera_rfdetr.onnx` (or your target-species equivalent). RF-DETR via ONNX Runtime, high-accuracy verification. RF-DETR cannot compile to `.hef` (Hailo Dataflow Compiler parser failure on its ONNX graph) — it runs on the desktop, never the NPU.

The V1 Porifera classifier (RF-DETR Medium, v12) carries forward as the default desktop verification model. See `audtheia-v2-master-concept.md` §7, §11.
