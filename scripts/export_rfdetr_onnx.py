"""Export a trained RF-DETR checkpoint to ONNX for Audtheia's desktop screening slot.

Audtheia never runs a model in the cloud, so a model trained on Roboflow (or
anywhere) has to become a local ONNX file before a station can use it. This turns
a checkpoint (weights.pt / .pth) into that ONNX, entirely on your computer.

Run it from PowerShell, after installing the exporter once:

    pip install "rfdetr[onnx]"
    python scripts/export_rfdetr_onnx.py "C:\\Users\\you\\Downloads\\weights.pt"

It writes <weights>.onnx next to your checkpoint. Then:
  1. copy that .onnx into models/visual/
  2. add a labels file beside it (see the note printed at the end)
  3. open the station, set it as the Desktop screening model.

If your model is not the "Small" size, change RFDETRSmall below to the size you
trained (RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRBase, or RFDETRLarge).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("Give the path to your weights file, for example:")
        print('  python scripts/export_rfdetr_onnx.py "C:\\Users\\you\\Downloads\\weights.pt"')
        return 1

    checkpoint = Path(sys.argv[1].strip().strip('"'))
    if not checkpoint.is_file():
        print(f"No file found at: {checkpoint}")
        return 1

    try:
        from rfdetr import RFDETRSmall  # change the size here if you trained a different one
    except ImportError:
        print('The exporter is not installed yet. Run this first, then try again:')
        print('  pip install "rfdetr[onnx]"')
        return 1

    print(f"Loading {checkpoint.name} ...")
    model = RFDETRSmall(pretrain_weights=str(checkpoint))

    print("Exporting to ONNX. This can take a minute, and prints a lot of progress ...")
    model.export()

    # rfdetr writes inference_model.onnx into an output directory under the current
    # folder; find it wherever it landed rather than assuming one fixed path.
    produced = None
    for candidate in Path(".").rglob("inference_model.onnx"):
        produced = candidate
        break
    if produced is None:
        print("Export finished but inference_model.onnx was not found. Look for an")
        print("'output' folder created next to where you ran this command.")
        return 1

    destination = checkpoint.with_suffix(".onnx")
    shutil.copyfile(produced, destination)
    print("")
    print(f"Done. Your model is here: {destination}")
    print("")
    print("Next steps:")
    print(f"  1. Copy {destination.name} into your Audtheia models/visual/ folder.")
    print("  2. Beside it, create a labels file named the same with .labels.json,")
    print("     for example bird_rfdetr.labels.json, holding your class names in")
    print('     training order, like: {"0": "Blue Jay", "1": "American Robin"}')
    print("     (get the exact names and order from your Roboflow project's")
    print("     Classes and Tags page). Without it, detections show a number, not a name.")
    print("  3. In Audtheia, open the station and set this file as the Desktop")
    print("     screening model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
