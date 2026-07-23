#!/usr/bin/env python3
"""Download the BirdNET acoustic model and place it for Audtheia.

BirdNET (Kahl et al., 2021) is the terrestrial/avian acoustic classifier. Its
model file is not bundled with Audtheia, so this one-time helper downloads the
BirdNET GLOBAL 6K V2.4 audio classifier and drops it into
`models/acoustic/birdnet/`. A station points its single acoustic model at the
downloaded file, so this is one concrete model a person may choose; the platform
itself names no model family. Labels are handled too: if a BirdNET labels file is
already beside the model it is kept; otherwise pass one with `--labels`.

This downloads only from the public URL below (a stable, widely-mirrored copy of
the upstream BirdNET V2.4 model) and writes only into the destination folder. It
is offline afterward, and nothing here runs at capture time.

Usage (from the repo root, using the app's environment):
    .venv\\Scripts\\python.exe scripts/fetch_birdnet.py
    .venv\\Scripts\\python.exe scripts/fetch_birdnet.py --labels path\\to\\Labels_en_us.txt

After it finishes it prints the exact `path` and `labels_path` to set on the
station's flat `models.acoustic` block in `config/settings.json`, or through
Settings, Models, in the interface.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# The BirdNET GLOBAL 6K V2.4 *audio* classifier (3-second windows at 48 kHz).
# Not the "MData" geo meta-model, which filters species by location rather than
# classifying sound. ~25 MB. Overridable with --model-url.
DEFAULT_MODEL_URL = (
    "https://raw.githubusercontent.com/mcguirepr89/BirdNET-Pi/main/model/"
    "BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite"
)
DEFAULT_MODEL_NAME = "BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite"
EXPECTED_MIN_BYTES = 20_000_000  # the real file is ~25 MB; a smaller file is an error page
# TFLite files are FlatBuffers carrying the "TFL3" file identifier at byte 4.
TFLITE_MAGIC = b"TFL3"


def _download(url: str, dest: Path) -> None:
    """Stream a URL to a file, showing simple progress."""
    print(f"downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp:  # noqa: S310 - fixed, documented model URL
        total = int(resp.headers.get("Content-Length") or 0)
        read = 0
        chunk = 1024 * 256
        with tmp.open("wb") as fh:
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                fh.write(block)
                read += len(block)
                if total:
                    pct = read * 100 // total
                    print(f"\r  {read // 1024:>8} KB / {total // 1024} KB ({pct}%)", end="", flush=True)
    print()
    tmp.replace(dest)


def _verify_tflite(path: Path) -> None:
    size = path.stat().st_size
    if size < EXPECTED_MIN_BYTES:
        raise SystemExit(
            f"downloaded file is only {size} bytes, which is too small to be the "
            f"model (expected ~25 MB). The URL may have returned an error page. "
            f"Delete {path} and try again, or pass --model-url."
        )
    with path.open("rb") as fh:
        head = fh.read(8)
    if TFLITE_MAGIC not in head:
        raise SystemExit(
            f"{path.name} does not look like a TFLite model (missing the 'TFL3' "
            f"identifier). Delete it and re-run, or pass a correct --model-url."
        )


def _find_existing_labels(dest_dir: Path) -> Path | None:
    """Return a BirdNET labels file already sitting beside the model, if any."""
    candidates = sorted(dest_dir.glob("*Labels*.txt")) + sorted(dest_dir.glob("*labels*.txt"))
    return candidates[0] if candidates else None


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-url", default=DEFAULT_MODEL_URL)
    parser.add_argument(
        "--dest",
        type=Path,
        default=repo_root / "models" / "acoustic" / "birdnet",
        help="folder to place the model + labels (default: models/acoustic/birdnet)",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="path to a BirdNET labels file (one label per line) to copy in; "
        "optional if one is already in the destination folder",
    )
    parser.add_argument("--force", action="store_true", help="re-download even if the model exists")
    args = parser.parse_args(argv)

    dest_dir: Path = args.dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    model_path = dest_dir / DEFAULT_MODEL_NAME

    if model_path.exists() and not args.force:
        print(f"model already present: {model_path} (use --force to re-download)")
    else:
        _download(args.model_url, model_path)
    _verify_tflite(model_path)
    print(f"model ready: {model_path} ({model_path.stat().st_size // (1024 * 1024)} MB)")

    if args.labels is not None:
        if not args.labels.is_file():
            raise SystemExit(f"labels file not found: {args.labels}")
        import shutil

        target = dest_dir / args.labels.name
        shutil.copyfile(args.labels, target)
        labels_path = target
        print(f"labels copied: {labels_path}")
    else:
        labels_path = _find_existing_labels(dest_dir)
        if labels_path is None:
            print(
                "\nNOTE: no labels file was found in the destination. BirdNET needs "
                "one (one species per line) for detections to be named. Re-run with "
                "--labels <file>, or copy a 'BirdNET_..._Labels_*.txt' into the folder."
            )
        else:
            print(f"using existing labels: {labels_path}")

    def _rel(p: Path) -> str:
        try:
            return p.resolve().relative_to(repo_root).as_posix()
        except ValueError:
            return str(p)

    print("\nSet these on the station's models.acoustic block in config/settings.json,")
    print("or through Settings, Models, on the station you will run:")
    print(f'  "path": "{_rel(model_path)}"')
    if labels_path is not None:
        print(f'  "labels_path": "{_rel(labels_path)}"')
    print('  and set "sample_rate" to the rate this model expects (for example 48000).')
    return 0


if __name__ == "__main__":
    sys.exit(main())
