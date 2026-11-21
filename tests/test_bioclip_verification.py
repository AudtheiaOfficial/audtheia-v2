"""Checks for BioCLIP 2 as a desktop visual verifier and for its Prepare step.

Path: tests/test_bioclip_verification.py

Runs in a temporary repository with its own settings, database, data folder and
models folder, so nothing here can reach a real installation's data, database,
reports, models or training environments. A scripted encoder stands in for
BioCLIP 2's exported image encoder: it embeds a picture as its average colour,
and each "species" is a colour, so the right answer is known exactly. It needs
neither ONNX nor ONNX Runtime, so the checks run wherever the suite runs. Proves:

  - The organism is cropped from the box saved with each frame, resized and
    normalised as the sidecar says, and matched against every candidate name
    with a softmax over the model's logit scale.
  - The verifier is built only when its encoder, name embeddings and names are
    all present, and refuses names and embeddings that do not belong together.
  - Each station's frames are checked by the model it chose, RF-DETR by default
    when both are set, and an unknown choice is refused by the configuration.
  - A visual event checked by BioCLIP 2 is verified when it names the field
    species, and is worded as a two-model check by BioCLIP 2.
  - Prepare refuses to run without the image training environment, and, with a
    scripted environment, installs what is missing, writes the candidate names,
    runs the export, logs every step and sets the verifier up through the save
    path.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from training_fixture import World  # noqa: E402
from audtheia.analysis import taxa, verdicts  # noqa: E402
from audtheia.analysis.verify import VerifyEngine  # noqa: E402
from audtheia.app.bioclip_prepare import OUT_DIR, PrepareJob, candidate_names  # noqa: E402
from audtheia.app.orchestrator import NullInterpreter  # noqa: E402
from audtheia.config import ConfigError, _validate  # noqa: E402
from audtheia.inference import bioclip_onnx as bio  # noqa: E402
from audtheia.inference import visual_verifiers  # noqa: E402

_checks = 0
SIZE = 8
RED, BLUE = (220, 30, 30), (30, 30, 220)


def check(label: str, condition: bool) -> None:
    global _checks
    if not condition:
        raise AssertionError(label)
    _checks += 1
    print(f"  ok  {label}")


class ColourSession:
    """Stands in for the exported encoder: a picture's embedding is its mean colour."""

    class _Input:
        name = "pixels"

    def get_inputs(self):
        return [self._Input()]

    def run(self, names, feeds):
        return [np.asarray(feeds["pixels"], dtype=np.float32).mean(axis=(2, 3))]


def prepared_files(root: Path, names: list, vectors: list) -> None:
    out = root / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "bioclip2_visual.onnx").write_bytes(b"stand-in encoder")
    emb = np.asarray(vectors, dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.save(out / "text_embeddings.npy", emb)
    (out / "labels.json").write_text(json.dumps(names), encoding="utf-8")
    (out / "bioclip2.json").write_text(json.dumps({"image_size": SIZE, "mean": [0.0, 0.0, 0.0],
                                                   "std": [1.0, 1.0, 1.0], "logit_scale": 100.0,
                                                   "names": len(names), "parity_max_abs_diff": 1e-6}),
                                       encoding="utf-8")


def configure(settings) -> None:
    settings.raw["desktop_models"]["visual_bioclip"] = {
        "path": f"{OUT_DIR}/bioclip2_visual.onnx", "embeddings_path": f"{OUT_DIR}/text_embeddings.npy",
        "labels_path": f"{OUT_DIR}/labels.json", "version": "BioCLIP 2", "citation": None}


def paint_frame(path: Path, box, inside, outside, size=(64, 48)) -> None:
    from PIL import Image

    image = Image.new("RGB", size, outside)
    x1, y1, x2, y2 = (int(v) for v in box)
    image.paste(Image.new("RGB", (x2 - x1, y2 - y1), inside), (x1, y1))
    image.save(path, quality=100)


def test_adapter(w: World) -> None:
    print("\n[1] The adapter crops, normalises and matches each frame")
    names = ["Poecile carolinensis", "Cardinalis cardinalis", "Viridis viridis"]
    prepared_files(w.repo, names, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    check("an unconfigured verifier is not built", bio.build_bioclip_verifier(w.settings) is None)
    configure(w.settings)
    check("a prepared verifier is reported as configured", bio.bioclip_configured(w.settings))
    verifier = bio.build_bioclip_verifier(w.settings, session=ColourSession())
    folder = w.root / "event"
    folder.mkdir()
    box = [10.0, 8.0, 40.0, 30.0]
    paint_frame(folder / "frame_000000.jpg", box, RED, BLUE)
    (folder / "annotations.jsonl").write_text(json.dumps(
        {"index": 0, "file": "frame_000000.jpg", "bbox_xyxy": box, "confidence": 0.8}) + "\n", encoding="utf-8")
    paint_frame(folder / "frame_000001.jpg", [0, 0, 4, 4], RED, BLUE)
    result = verifier.verify_frames([folder / "frame_000000.jpg", folder / "frame_000001.jpg",
                                     folder / "missing.jpg"])
    check("the organism is cropped from its saved box before matching",
          result[0]["scientific_name"] == "Poecile carolinensis")
    check("a frame with no saved box is matched whole", result[1]["scientific_name"] == "Cardinalis cardinalis")
    check("an unreadable frame supports no taxon", result[2]["scientific_name"] is None)
    check("the confidence is a softmax over every candidate name", 0.9 < result[0]["confidence"] <= 1.0)
    version, labels = bio.bioclip_label_set(w.settings)
    check("its label set is the candidate names", version == "BioCLIP 2" and labels == sorted(names))
    try:
        bio.BioCLIPVerifier(None, np.zeros((2, 3), dtype=np.float32), ["a b"], {}, version="x")
        refused = False
    except bio.BioCLIPError:
        refused = True
    check("names and embeddings of different lengths are refused", refused)


def test_choice(w: World) -> None:
    print("\n[2] Each station's frames are checked by the model it chose")
    station = {"models": {}}
    check("with no visual verifier there is no choice", visual_verifiers.choice(w.settings, station) is None)
    configure(w.settings)
    prepared_files(w.repo, ["Poecile carolinensis"], [[1, 0, 0]])
    check("BioCLIP 2 is used when it is the only one", visual_verifiers.choice(w.settings, station) == "visual_bioclip")
    rf = w.repo / "models" / "visual" / "desktop" / "rf.onnx"
    rf.parent.mkdir(parents=True, exist_ok=True)
    rf.write_bytes(b"stand-in")
    w.settings.raw["desktop_models"]["visual_rfdetr"]["path"] = "models/visual/desktop/rf.onnx"
    check("RF-DETR stays the default when both are set", visual_verifiers.choice(w.settings, station) == "visual_rfdetr")
    check("a station that chose BioCLIP 2 gets it",
          visual_verifiers.choice(w.settings, {"models": {"desktop_verifier": "visual_bioclip"}}) == "visual_bioclip")
    w.settings.raw["desktop_models"]["visual_rfdetr"]["path"] = None
    raw = json.loads(json.dumps(w.settings.raw))
    raw["stations"][0]["models"]["desktop_verifier"] = "something_else"
    try:
        _validate(raw)
        refused = False
    except ConfigError:
        refused = True
    check("an unknown verifier choice is refused by the configuration", refused)


def test_engine(w: World) -> None:
    print("\n[3] A visual event checked by BioCLIP 2")
    configure(w.settings)
    prepared_files(w.repo, ["Poecile carolinensis", "Cardinalis cardinalis"], [[1, 0, 0], [0, 0, 1]])
    ev = w.vision_event([("Poecile carolinensis", None)], frames=4, verified=None)
    for i in range(4):
        paint_frame(Path(ev["dir"]) / f"frame_{i:06d}.jpg", [10, 8, 40, 30], RED, BLUE)
    engine = VerifyEngine(settings=w.settings, db=w.db,
                          verifier=bio.build_bioclip_verifier(w.settings, session=ColourSession()),
                          interpreter=NullInterpreter(), identify=taxa.identity_function(w.settings))
    engine.process(ev["id"])
    v = w.db.get_observation_verification(ev["id"])
    check("the event is verified when BioCLIP 2 names the field species",
          v["verified"] == 1 and v["rfdetr_version"] == "BioCLIP 2" and v["rfdetr_scientific_name"] == "Poecile carolinensis")
    text = verdicts.describe(w.db.get_observation(ev["id"]), v, children=w.db.list_child_detections(ev["id"]))
    check("it is worded as a two-model check by BioCLIP 2, counted in frames",
          text.startswith("verified by the two-model check: BioCLIP 2 recognized Poecile carolinensis in 4 of 4 frames"))


class FakeComponents:
    def __init__(self, root: Path, state: str):
        self.root, self.state = root, state

    def status(self, engine):
        return {"state": self.state}

    def env_dir(self, engine):
        return self.root / "env"


class FakeProc:
    def __init__(self, lines, code):
        self.stdout, self._code = iter(lines), code

    def wait(self):
        return self._code

    def kill(self):
        pass


def test_prepare(w: World) -> None:
    print("\n[4] Prepare")
    saved = []
    job = PrepareJob(w.settings, w.db, lambda s: saved.append(dict(s.raw["desktop_models"]["visual_bioclip"])),
                     components=FakeComponents(w.root, "not_installed"))
    done = job.start(wait=True)
    check("without the image environment Prepare stops and says where to install it",
          done["job"]["state"] == "error" and "Train tab" in done["job"]["message"])

    resources = w.repo / "audtheia" / "resources" / "perch"
    resources.mkdir(parents=True, exist_ok=True)
    (resources / "labels.csv").write_text("inat2024_fsd50k\nAbavorana luctuosa\nSpeech\nPoecile carolinensis\n",
                                          encoding="utf-8")
    w.vision_event([("Cardinalis cardinalis", None)])
    w.settings.raw["stations"][0]["target_species"] = ["Sitta carolinensis"]
    names = candidate_names(w.settings, w.db)
    check("the candidates are target species, the record's species and Perch's species, without sound classes",
          names[:2] == ["Sitta carolinensis", "Cardinalis cardinalis"]
          and "Abavorana luctuosa" in names and "Poecile carolinensis" in names and "Speech" not in names)

    calls = []

    def popen(args, **kwargs):
        calls.append(list(args))
        if "-c" in args:
            return FakeProc(["ModuleNotFoundError: No module named 'open_clip'\n"], 1)
        if "pip" in args:
            return FakeProc(["Successfully installed open_clip_torch-4.2.2\n"], 0)
        out = Path(args[args.index("--out") + 1])
        prepared_files(w.repo, json.loads(Path(args[args.index("--names") + 1]).read_text(encoding="utf-8")),
                       [[1, 0, 0]] * len(names))
        check("the export runs on the processor and can import Audtheia",
              kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "" and str(w.repo) in kwargs["env"]["PYTHONPATH"]
              and out == w.repo / OUT_DIR)
        return FakeProc(["PROGRESS loading the model\n", "PROGRESS done\n"], 0)

    job = PrepareJob(w.settings, w.db, lambda s: saved.append(dict(s.raw["desktop_models"]["visual_bioclip"])),
                     components=FakeComponents(w.root, "ready"), popen=popen)
    done = job.start(wait=True)
    check("Prepare installs open_clip when it is missing, then exports",
          done["job"]["state"] == "done" and any("pip" in c for c in calls)
          and any("audtheia.training.bioclip_export" in c for c in calls))
    check("the verifier is set up through the save path",
          saved and saved[-1]["path"] == f"{OUT_DIR}/bioclip2_visual.onnx" and saved[-1]["version"] == "BioCLIP 2")
    log = (w.repo / OUT_DIR / "prepare.log").read_text(encoding="utf-8")
    check("every step is written to the log", "Installing open_clip" in log and "PROGRESS loading the model" in log
          and "DONE:" in log)
    check("the status reports it prepared", job.status()["prepared"] is True)


def run() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        test_adapter(World(root / "a"))
        test_choice(World(root / "c"))
        test_engine(World(root / "e"))
        test_prepare(World(root / "p"))
    print(f"\nAll {_checks} BioCLIP 2 verification checks passed.")


if __name__ == "__main__":
    try:
        run()
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
