"""Build an Audtheia labels file from a YOLO/Roboflow data.yaml.

Audtheia reads class names from a file named exactly like the model with
'.labels.json' added (bird_classifier.onnx -> bird_classifier.labels.json), in the
same folder. It does NOT read data.yaml. This converts one into the other, in the
right order.

The one subtlety: RF-DETR numbers its classes from 1 (index 0 is reserved), while a
YOLO data.yaml lists them from 0. So a name that sits third in data.yaml is class 3
to the model, not 2. This shifts by one by default to match, which is why blue-jay
(third alphabetically) lines up with the class 3 you saw. If the names come out off
by one for your model, re-run with:  --base 0

Usage (from PowerShell):
    python scripts/build_labels_from_yaml.py "<path\\to\\data.yaml>" "<path\\to\\model.onnx>"

Options:
    --base 0   use 0-based numbering (for a YOLO model, not RF-DETR)
    --raw      keep the exact dataset names (blue-jay) instead of prettifying (Blue Jay)

It writes <model>.labels.json next to the model and prints the mapping so you can
confirm the numbers are right before you run capture.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load_names(yaml_path: Path) -> list:
    """Read the ordered class-name list from a data.yaml, with or without PyYAML."""
    text = yaml_path.read_text(encoding="utf-8")
    try:
        import yaml  # present if you installed the training tools; optional here
        data = yaml.safe_load(text)
        names = data.get("names") if isinstance(data, dict) else None
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
        if isinstance(names, list):
            return [str(n) for n in names]
    except Exception:
        pass

    # Fallback parser: handle both "names: [a, b]" and a block of "- a" lines.
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("names:"):
            continue
        rest = stripped[len("names:"):].strip()
        if rest.startswith("["):
            inner = rest.strip("[]")
            return [x.strip().strip("'\"") for x in inner.split(",") if x.strip()]
        collected = []
        for follow in lines[i + 1:]:
            f = follow.strip()
            if f.startswith("- "):
                collected.append(f[2:].strip().strip("'\""))
            elif f and not f.startswith("#"):
                break
        return collected
    return []


def prettify(name: str) -> str:
    return name.replace("-", " ").replace("_", " ").strip().title()


def main() -> int:
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    base = 1
    if "--base" in sys.argv:
        try:
            base = int(sys.argv[sys.argv.index("--base") + 1])
        except (IndexError, ValueError):
            base = 1
    raw = "--raw" in sys.argv

    if len(positional) < 2:
        print('Usage: python scripts/build_labels_from_yaml.py "<data.yaml>" "<model.onnx>" [--base 0] [--raw]')
        return 1

    yaml_path = Path(positional[0].strip().strip('"'))
    model_path = Path(positional[1].strip().strip('"'))
    if not yaml_path.is_file():
        print(f"No data.yaml found at: {yaml_path}")
        return 1

    names = load_names(yaml_path)
    if not names:
        print("Could not read a names list from that data.yaml.")
        return 1

    mapping = {str(i + base): (n if raw else prettify(n)) for i, n in enumerate(names)}

    out_path = model_path.with_name(model_path.stem + ".labels.json")
    out_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(mapping)} labels to: {out_path}")
    print(f"Numbering starts at {base} (use --base 0 if your names come out off by one).")
    # Show a couple of anchors so the numbering can be sanity-checked at a glance.
    for key, value in list(mapping.items())[:4]:
        print(f"  {key} -> {value}")
    blue = next((k for k, v in mapping.items() if v.lower().replace(" ", "-") == "blue-jay"), None)
    if blue:
        print(f"  ...")
        print(f"  {blue} -> {mapping[blue]}   (this should match the number your model showed for the blue jay)")
    print("")
    print("Now restart capture on the station and re-run a clip; boxes will show names.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
