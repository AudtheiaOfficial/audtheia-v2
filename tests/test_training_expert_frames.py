"""Frames with an expert's boxes are kept when an event is thinned for training.

Path: tests/test_training_expert_frames.py

Training keeps at most a set number of frames per event. Spreading them evenly
along a long track passed over most of the frames whose boxes an expert had
redrawn, removed or added, so the most trustworthy labels in the event were
rarely used. These checks prove that:

  - Every frame an expert edited is kept when they fit within the limit, with
    the remaining places spread along the rest of the track.
  - When more frames were edited than the limit allows, the kept frames are all
    edited ones, spread along the track.
  - Where an edited frame and an unedited one look alike, the edited one is kept.
  - With no edits, the frames kept are exactly those chosen before.
  - Kept frames come back in track order.

Frames are tiny files in a temporary folder; nothing in the repository is read
or written.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from audtheia.training import frames as fr  # noqa: E402

_checks = 0


def check(label: str, condition: bool) -> None:
    global _checks
    if not condition:
        raise AssertionError(label)
    _checks += 1
    print(f"  ok  {label}")


def track(folder: Path, n: int, edited=()) -> list:
    out = []
    for i in range(n):
        name = f"frame_{i:06d}.jpg"
        (folder / name).write_bytes(b"x")
        out.append({"rec": {"index": i, "file": name}, "role": "positive", "boxes": [],
                    "model_labelled": False, "box_edited": i in set(edited)})
    return out


def distinct(path) -> int:
    """Every frame looks different."""
    return int(Path(path).stem.split("_")[1]) * 1_000_003


def indices(result) -> list:
    return [int(f["rec"]["index"]) for f in result["frames"]]


def run() -> None:
    with tempfile.TemporaryDirectory() as td:
        folder = Path(td)
        frames = track(folder, 1263, edited=(345, 346, 347, 700, 1001))

        kept = indices(fr.thin_frames(frames, folder, max_frames=10, near_duplicate_hamming=4, hasher=distinct))
        check("ten frames are kept", len(kept) == 10)
        check("every edited frame is kept", {345, 346, 347, 700, 1001} <= set(kept))
        check("kept frames come back in track order", kept == sorted(kept))
        check("the other places are spread along the track, from its first frame to its last",
              0 in kept and 1262 in kept)

        many = track(folder, 1263, edited=range(300, 320))
        kept = indices(fr.thin_frames(many, folder, max_frames=10, near_duplicate_hamming=4, hasher=distinct))
        check("with more edited frames than places, every kept frame is an edited one",
              len(kept) == 10 and all(300 <= i < 320 for i in kept))
        check("and they are spread along the edited stretch", 300 in kept and 319 in kept)

        plain = track(folder, 1263)
        before = [int(f["rec"]["index"]) for f in fr._spread(plain, 10)]
        kept = indices(fr.thin_frames(plain, folder, max_frames=10, near_duplicate_hamming=4, hasher=distinct))
        check("with no edits, the frames kept are exactly those chosen before", kept == before)

        # Frame 0 is an evenly spread pick and frame 5 an edited one; they look
        # alike, so only one is kept, and it is the expert's.
        alike = track(folder, 50, edited=(5,))

        def lookalike(path) -> int:
            index = int(Path(path).stem.split("_")[1])
            return 0 if index in (0, 5) else index * 1_000_003

        kept = indices(fr.thin_frames(alike, folder, max_frames=10, near_duplicate_hamming=4, hasher=lookalike))
        check("where an edited frame and an unedited one look alike, the edited one is kept",
              5 in kept and 0 not in kept)

        check("with no limit, every frame is kept",
              len(fr.thin_frames(plain[:30], folder, max_frames=0, near_duplicate_hamming=0, hasher=distinct)["frames"]) == 30)

    print(f"\nAll {_checks} expert-frame checks passed.")


if __name__ == "__main__":
    try:
        run()
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
