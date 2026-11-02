"""Verification for the vision engine's run configuration and its Gentle guarantee.

Path: tests/test_training_vision.py

Checks, without PyTorch, that a request from the Train view becomes a complete
run configuration taken from the settings, that Full speed and Gentle differ
only in how the work is divided and never in what is learned, that an ONNX file
is refused as a starting point because it cannot be trained, that unknown
settings and sizes are refused, that the exported model is checked for the
dataset's class count, and that the engine module imports nothing heavy
at module level, so the application can build configurations without the
training components installed.

Uses the standard library and the project itself.
"""

from __future__ import annotations

import ast
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from audtheia.config import DEFAULT_TRAINING  # noqa: E402
from audtheia.training import hardware as H  # noqa: E402
from audtheia.training import vision as V  # noqa: E402

CHECKS = {"passed": 0, "failed": 0}

LAPTOP = {"platform": "windows", "cpu": {"logical_cores": 12}, "gpus": []}


def check(label: str, condition: bool) -> None:
    CHECKS["passed" if condition else "failed"] += 1
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")


def raises(fn, exc=Exception) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


def test_config():
    print("run configuration")
    cfg = V.build_config(DEFAULT_TRAINING, base_key="nano", mode="gentle", hw=LAPTOP, device="cpu")
    d = DEFAULT_TRAINING["vision"]["defaults"]
    check("the size and resolution come from the settings", cfg["base_model"] == "nano" and cfg["resolution"] == 384
          and cfg["base_label"] == "Nano")
    check("every training value comes from the settings defaults",
          cfg["epochs"] == d["epochs"] and cfg["learning_rate"] == d["learning_rate"]
          and cfg["encoder_learning_rate"] == d["encoder_learning_rate"] and cfg["weight_decay"] == d["weight_decay"]
          and cfg["use_ema"] is d["use_ema"] and cfg["early_stopping"] is d["early_stopping"]
          and cfg["early_stopping_patience"] == d["early_stopping_patience"]
          and cfg["checkpoint_interval"] == d["checkpoint_interval"] and cfg["best_metric"] == d["best_metric"])
    check("the seed is the dataset split seed unless set", cfg["seed"] == DEFAULT_TRAINING["dataset"]["split"]["seed"])
    check("the processor trains in full precision", cfg["precision"] == "fp32" and V.precision_for("cuda") == "auto")
    check("the start is the pretrained weights by default", cfg["start_from"] == {"kind": "pretrained"})
    check("the safety limits travel with the run", cfg["hardware"]["limits"] == DEFAULT_TRAINING["hardware"]["limits"])
    check("Gentle feeds one image at a time, four per update, on half the threads",
          cfg["resources"]["micro_batch"] == 1 and cfg["resources"]["grad_accum_steps"] == 4
          and cfg["resources"]["cpu_threads"] == 6 and cfg["resources"]["gradient_checkpointing"] is True)
    over = V.build_config(DEFAULT_TRAINING, base_key="small", mode="full_speed", hw=LAPTOP, device="cpu",
                          overrides={"epochs": 3, "batch_size": 2, "seed": 11})
    check("the Advanced drawer can change a value", over["epochs"] == 3 and over["seed"] == 11
          and over["effective_batch"] == 2 and over["resolution"] == 512)
    check("an unknown setting is refused", raises(lambda: V.build_config(DEFAULT_TRAINING, base_key="nano",
                                                                       mode="gentle", hw=LAPTOP, device="cpu",
                                                                       overrides={"lr": 1}), V.VisionConfigError))
    check("zero epochs is refused", raises(lambda: V.build_config(DEFAULT_TRAINING, base_key="nano", mode="gentle",
                                                                 hw=LAPTOP, device="cpu", overrides={"epochs": 0}),
                                           V.VisionConfigError))
    check("a size not in the settings is refused",
          raises(lambda: V.build_config(DEFAULT_TRAINING, base_key="xlarge", mode="gentle", hw=LAPTOP, device="cpu"),
                 V.VisionConfigError))
    with tempfile.TemporaryDirectory() as td:
        onnx = Path(td) / "porifera_rfdetr.onnx"
        onnx.write_bytes(b"x")
        pth = Path(td) / "weights.pth"
        pth.write_bytes(b"x")
        check("an ONNX file is refused as a starting point with the reason",
              raises(lambda: V.build_config(DEFAULT_TRAINING, base_key="medium", mode="gentle", hw=LAPTOP, device="cpu",
                                            start_from={"kind": "checkpoint", "path": str(onnx)}), V.VisionConfigError))
        try:
            V.build_config(DEFAULT_TRAINING, base_key="medium", mode="gentle", hw=LAPTOP, device="cpu",
                           start_from={"kind": "checkpoint", "path": str(onnx)})
        except V.VisionConfigError as exc:
            check("the reason names the alternatives", "cannot be trained further" in str(exc) and "pretrained" in str(exc))
        ok = V.build_config(DEFAULT_TRAINING, base_key="medium", mode="gentle", hw=LAPTOP, device="cpu",
                            start_from={"kind": "checkpoint", "path": str(pth)})
        check("a PyTorch checkpoint is accepted", ok["start_from"]["path"] == str(pth))
        check("a missing checkpoint is refused",
              raises(lambda: V.build_config(DEFAULT_TRAINING, base_key="medium", mode="gentle", hw=LAPTOP, device="cpu",
                                            start_from={"kind": "checkpoint", "path": str(Path(td) / "no.pth")}),
                     V.VisionConfigError))
    check("an unknown start is refused",
          raises(lambda: V.build_config(DEFAULT_TRAINING, base_key="nano", mode="gentle", hw=LAPTOP, device="cpu",
                                        start_from={"kind": "magic"}), V.VisionConfigError))
    long_run = V.disk_parts_mb({"epochs": 100, "checkpoint_interval": 10}, 121.0)
    short_run = V.disk_parts_mb({"epochs": 3, "checkpoint_interval": 10}, 121.0)
    check("a long run keeps the last checkpoint and one per interval",
          long_run["checkpoints_kept"] == 11 and short_run["checkpoints_kept"] == 1)
    check("a checkpoint counts four times the weights, and the light files and export four more",
          short_run["checkpoint_mb"] == 484.0 and short_run["other_mb"] == 484.0)
    est = H.estimate_run(train_images=32, eval_images=8, epochs=3, seconds_per_train_image=None,
                         seconds_per_eval_image=None, checkpoint_mb=short_run["checkpoint_mb"],
                         checkpoints_kept=short_run["checkpoints_kept"], export_mb=short_run["other_mb"])
    check("the disk estimate for the measured Nano run is within a tenth of the 926 MB it used",
          abs(est["disk_gb"] * 1024.0 - 926.0) / 926.0 < 0.1)


def test_resume_checkpoint():
    print("which checkpoint a resume uses")
    import os
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        check("an empty folder has nothing to resume from", V.latest_full_checkpoint(d) is None)
        (d / "checkpoint_best_total.pth").write_bytes(b"x")
        check("the light best-model files are not resumable", V.latest_full_checkpoint(d) is None)
        for name, when in (("checkpoint_9.ckpt", 100), ("last.ckpt", 200)):
            (d / name).write_bytes(b"x")
            os.utime(d / name, (when, when))
        check("last.ckpt is used when it is the newest", V.latest_full_checkpoint(d).name == "last.ckpt")
        (d / "checkpoint_19.ckpt").write_bytes(b"x")
        os.utime(d / "checkpoint_19.ckpt", (300, 300))
        check("a newer numbered checkpoint wins", V.latest_full_checkpoint(d).name == "checkpoint_19.ckpt")
        os.utime(d / "last.ckpt", (300, 300))
        check("on a tie last.ckpt wins", V.latest_full_checkpoint(d).name == "last.ckpt")


class _Dim:
    def __init__(self, value):
        self.dim_value = value

    def HasField(self, name):
        return name == "dim_value" and self.dim_value is not None


class _Output:
    def __init__(self, name, dims):
        self.name = name
        self.type = type("T", (), {"tensor_type": type("S", (), {"shape": type("D", (), {"dim": dims})()})()})()


def _graph(outputs):
    return type("P", (), {"graph": type("G", (), {"output": outputs})()})()


def test_export_check():
    print("the exported model's class count is checked")
    proto = _graph([_Output("dets", [_Dim(1), _Dim(300), _Dim(4)]),
                    _Output("labels", [_Dim(1), _Dim(300), _Dim(4)])])
    check("the class score width is read from the labels output", V.label_columns(proto) == 4)
    check("a graph without a fixed width gives no answer",
          V.label_columns(_graph([_Output("labels", [_Dim(1), _Dim(300), _Dim(None)])])) is None
          and V.label_columns(_graph([_Output("dets", [_Dim(4)])])) is None)
    src = (REPO / "audtheia" / "training" / "vision.py").read_text(encoding="utf-8")
    check("the class count is always passed to the model, in training and in export",
          src.count('"num_classes": len(names)') == 1 and src.count("num_classes=len(names)") == 1)


def test_equivalence():
    print("Full speed and Gentle learn the same thing")
    for size in ("nano", "small", "medium", "large"):
        for device in ("cpu", "cuda"):
            pair = V.equivalent_modes(DEFAULT_TRAINING, base_key=size, hw=LAPTOP, device=device)
            check(f"{size} on {device}: no fixed quantity differs", pair["differences"] == [])
    pair = V.equivalent_modes(DEFAULT_TRAINING, base_key="nano", hw=LAPTOP, device="cpu", overrides={"batch_size": 8})
    f, g = pair["full_speed"], pair["gentle"]
    check("only the division of work differs",
          {k for k in f if f[k] != g[k]} == {"resources"}
          and f["resources"]["effective_batch"] == g["resources"]["effective_batch"] == 8)
    tampered = dict(g, resolution=320)
    check("a Gentle configuration with a lower resolution would be caught",
          H.check_equivalent(f, tampered) == ["resolution"])


def test_imports():
    print("imports")
    src = (REPO / "audtheia" / "training" / "vision.py").read_text(encoding="utf-8")
    check("vision.py is ASCII only", all(ord(c) < 128 for c in src))
    tree = ast.parse(src)
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            top.add((node.module or "").split(".")[0])
    check("no machine-learning library is imported at module level",
          not top & {"torch", "rfdetr", "pytorch_lightning", "onnx", "numpy"})
    check("the worker knows the engine by its key", "vision_detector" in __import__(
        "audtheia.training.worker", fromlist=["ENGINES"]).ENGINES)


def main() -> int:
    test_config()
    test_resume_checkpoint()
    test_export_check()
    test_equivalence()
    test_imports()
    print(f"\nResult: {CHECKS['passed']} passed, {CHECKS['failed']} failed")
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
