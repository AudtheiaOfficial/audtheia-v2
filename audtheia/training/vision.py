"""The vision training engine: fine-tunes an RF-DETR detector on a built dataset.

Path: audtheia/training/vision.py

Two halves.

The first half runs in the application and needs only the standard library. It
turns a request from the Train view, a built dataset and the hardware plan into
the complete run configuration, and checks that Full speed and Gentle would
learn exactly the same thing.

The second half runs inside the worker, in the separate training environment,
and is the only place PyTorch and the rfdetr library are imported. It trains
with the same open method Roboflow publishes, reports every batch and epoch to
the run folder so Pause, Cancel and the safety checks work, resumes from the
last saved epoch, and exports the best weights to ONNX with the class names
embedded, plus a labels file, so the existing verifier loads the result with no
change. The final accuracy is decided by the quantity and quality of the labels,
not by this code.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from audtheia.training import hardware

ENGINE_KEY = "vision_detector"

# The rfdetr releases this engine was tested against. Another release still runs,
# with a note in the log, and the hook this engine needs is checked at start.
TESTED_RFDETR = ("1.11.",)

# The rfdetr class for each base-model key in the settings.
SIZE_CLASSES = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "large": "RFDETRLarge"}

EXPORT_NAME = "model.onnx"
LABELS_NAME = "labels.json"


class VisionConfigError(ValueError):
    """A training request that cannot become a run, in plain words."""


# ---------------------------------------------------------------------------
# Application side: building the run configuration
# ---------------------------------------------------------------------------


def base_entry(training_cfg: Mapping, key: str) -> dict:
    for entry in (training_cfg.get("vision") or {}).get("base_models") or []:
        if entry.get("key") == key:
            if key not in SIZE_CLASSES:
                raise VisionConfigError(f"The base model {key!r} is not one this engine can train.")
            return dict(entry)
    raise VisionConfigError(f"There is no base model {key!r} in the settings.")


def precision_for(device: str) -> str:
    """Full precision on the processor, where rfdetr always trains in fp32, and
    mixed precision on a graphics card. Both modes of one run share it."""
    return "fp32" if device in ("cpu", "mps") else "auto"


def build_config(training_cfg: Mapping, *, base_key: str, mode: str, hw: Mapping, device: str,
                 overrides: Mapping | None = None, start_from: Mapping | None = None,
                 weights_dir: str | None = None) -> dict:
    """The complete configuration a vision run is started with.

    overrides may change any of the vision defaults from the settings (the
    Advanced drawer). start_from is {"kind": "pretrained"} by default, or
    {"kind": "checkpoint", "path": ...} for a person's own PyTorch checkpoint,
    or {"kind": "scratch"} for an untrained start.
    """
    entry = base_entry(training_cfg, base_key)
    defaults = dict(((training_cfg.get("vision") or {}).get("defaults")) or {})
    allowed = set(defaults) | {"epochs", "seed"}
    for key in (overrides or {}):
        if key not in allowed:
            raise VisionConfigError(f"{key!r} is not a vision training setting.")
    values = dict(defaults, **dict(overrides or {}))
    seed = int(values.get("seed", ((training_cfg.get("dataset") or {}).get("split") or {}).get("seed", 0)))
    epochs = int(values["epochs"])
    if epochs < 1:
        raise VisionConfigError("Training needs at least one epoch.")
    start = dict(start_from or {"kind": "pretrained"})
    if start.get("kind") not in ("pretrained", "checkpoint", "scratch"):
        raise VisionConfigError("Start from the pretrained weights, your own checkpoint, or scratch.")
    if start["kind"] == "checkpoint":
        path = str(start.get("path") or "")
        if path.lower().endswith(".onnx"):
            raise VisionConfigError("An ONNX file cannot be trained further. Choose a PyTorch checkpoint (.pth or "
                                    ".pt), or start from the pretrained weights of the chosen size.")
        if not path.lower().endswith((".pth", ".pt")) or not Path(path).is_file():
            raise VisionConfigError("The starting checkpoint must be an existing .pth or .pt file.")
    resources = hardware.resource_plan(mode, hw, training_cfg.get("hardware") or {},
                                       batch_size=int(values["batch_size"]),
                                       grad_accum_steps=int(values.get("grad_accum_steps", 1)))
    return {
        "engine": ENGINE_KEY,
        "base_model": base_key,
        "base_label": entry.get("label", base_key),
        "resolution": int(entry["resolution"]),
        "epochs": epochs,
        "seed": seed,
        "precision": precision_for(device),
        "device": device,
        "effective_batch": resources["effective_batch"],
        "learning_rate": float(values["learning_rate"]),
        "encoder_learning_rate": float(values["encoder_learning_rate"]),
        "weight_decay": float(values["weight_decay"]),
        "use_ema": bool(values["use_ema"]),
        "early_stopping": bool(values["early_stopping"]),
        "early_stopping_patience": int(values["early_stopping_patience"]),
        "early_stopping_min_delta": float(values["early_stopping_min_delta"]),
        "checkpoint_interval": int(values["checkpoint_interval"]),
        "best_metric": str(values["best_metric"]),
        "score_threshold": float(values.get("score_threshold", 0.3)),
        "start_from": start,
        "weights_dir": weights_dir,
        "resources": resources,
        "hardware": {"limits": dict(((training_cfg.get("hardware") or {}).get("limits")) or {})},
    }


def equivalent_modes(training_cfg: Mapping, *, base_key: str, hw: Mapping, device: str,
                     overrides: Mapping | None = None) -> dict:
    """Build both modes for one request and show that only the division of work differs."""
    full = build_config(training_cfg, base_key=base_key, mode="full_speed", hw=hw, device=device, overrides=overrides)
    gentle = build_config(training_cfg, base_key=base_key, mode="gentle", hw=hw, device=device, overrides=overrides)
    return {"full_speed": full, "gentle": gentle, "differences": hardware.check_equivalent(full, gentle)}


def disk_parts_mb(config: Mapping, parameters_mb: float) -> dict:
    """What a finished run leaves on disk, from the size of the model's weights.

    A resumable checkpoint holds the weights, their moving average and the
    optimiser's two running averages, so about four times the weights. The run
    keeps the last one and one every checkpoint_interval epochs. Beside them sit
    three light files of the weights alone (the best model, its moving average
    and the last moving average) and the exported model. Measured on a real Nano
    run of three epochs: 121 MB of weights, a 483 MB checkpoint, 926 MB in all.
    """
    weights = float(parameters_mb)
    interval = max(1, int(config.get("checkpoint_interval", 10)))
    return {"checkpoint_mb": 4.0 * weights, "checkpoints_kept": 1 + int(config["epochs"]) // interval,
            "other_mb": 4.0 * weights}


# ---------------------------------------------------------------------------
# Worker side: training
# ---------------------------------------------------------------------------


def _names_from_dataset(dataset_dir: Path) -> tuple:
    """Class names in label order with their GBIF keys, from the dataset the run trained on."""
    classes = json.loads((dataset_dir / "classes.json").read_text(encoding="utf-8"))
    ordered = sorted(classes, key=lambda c: int(c["category_id"]))
    names = [str(c["class_name"]) for c in ordered]
    keys = [c.get("gbif_usage_key") for c in ordered]
    return names, keys


def _figures(metrics: Mapping) -> dict:
    """The epoch figures worth keeping from Lightning's logged values."""
    wanted = {
        "train/loss": "train_loss", "train_loss": "train_loss", "val/loss": "val_loss",
        "val/mAP_50_95": "map50_95", "val/mAP_50": "map50", "val/ema_mAP_50_95": "ema_map50_95",
        "val/ema_mAP_50": "ema_map50", "val/mAR": "mar",
    }
    out = {}
    for key, name in wanted.items():
        if key in metrics:
            try:
                value = float(metrics[key])
            except (TypeError, ValueError):
                continue
            if value == value and abs(value) != float("inf"):
                out[name] = round(value, 6)
    return out


class VisionEngine:
    """Runs one RF-DETR training run inside the worker."""

    def __init__(self, config: Mapping):
        self.config = dict(config)

    def run(self, ctx) -> dict:
        cfg = self.config
        import torch  # noqa: PLC0415
        import rfdetr  # noqa: PLC0415
        import rfdetr.training as rf_training  # noqa: PLC0415
        import pytorch_lightning as pl  # noqa: PLC0415

        version = installed_version("rfdetr")
        if not version.startswith(TESTED_RFDETR):
            ctx.log(f"Note: rfdetr {version} is installed; this engine was tested with {TESTED_RFDETR[0]}x.")
        if not hasattr(rf_training, "build_trainer"):
            raise RuntimeError(f"rfdetr {version} does not offer the training hook Audtheia needs. "
                               "Remove and reinstall the training components.")
        threads = int((cfg.get("resources") or {}).get("cpu_threads") or 1)
        torch.set_num_threads(threads)
        run_dir = Path(ctx.run_dir)
        dataset_dir = Path(cfg["dataset_dir"])
        output_dir = run_dir / "train_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        names, gbif_keys = _names_from_dataset(dataset_dir)

        model_cls = getattr(rfdetr, SIZE_CLASSES[cfg["base_model"]])
        # The class count is always given explicitly. Left unset, the library takes
        # it from a starting checkpoint, so a model started from a two-class run
        # would keep two classes even when the dataset has three.
        model_kwargs: dict = {"gradient_checkpointing": bool(cfg["resources"].get("gradient_checkpointing")),
                              "num_classes": len(names)}
        start = cfg.get("start_from") or {"kind": "pretrained"}
        if start["kind"] == "checkpoint":
            model_kwargs["pretrain_weights"] = str(start["path"])
        elif start["kind"] == "scratch":
            model_kwargs["pretrain_weights"] = None
        else:
            default_name = Path(str(model_cls._model_config_class.model_fields["pretrain_weights"].default)).name
            weights_dir = Path(cfg.get("weights_dir") or (run_dir / "weights"))
            weights_dir.mkdir(parents=True, exist_ok=True)
            model_kwargs["pretrain_weights"] = str(weights_dir / default_name)
            if not (weights_dir / default_name).is_file():
                ctx.log(f"Downloading the pretrained {cfg['base_label']} weights once into {weights_dir}.")
        resume_from = None
        if ctx.resume:
            last = latest_full_checkpoint(output_dir)
            if last is None:
                ctx.nothing_to_resume()
            else:
                resume_from = str(last)
                ctx.log(f"Resuming from {last.name}.")

        ctx.log(f"Preparing the {cfg['base_label']} model at {cfg['resolution']} pixels on {cfg['device']}, "
                f"{cfg['resources']['micro_batch']} image(s) at a time, {cfg['resources']['grad_accum_steps']} "
                f"step(s) per update, {threads} processor thread(s).")
        model = model_cls(**model_kwargs)
        if start["kind"] == "checkpoint":
            earlier = [str(n) for n in (getattr(model.model, "class_names", None) or [])]
            if earlier and earlier != names:
                ctx.log(f"Note: the starting checkpoint knew {len(earlier)} class(es) ({', '.join(earlier)}); "
                        f"this dataset has {len(names)} ({', '.join(names)}). The part of the model that names "
                        "the class is resized and relearned, while the rest starts from the checkpoint.")
        epochs_total = int(cfg["epochs"])
        ctx.set_state("running", epochs_total=epochs_total)

        class AudtheiaProgress(pl.Callback):
            """Reports to the run folder and obeys Pause, Cancel and the safety checks."""

            def __init__(self):
                self.best = None

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                total = trainer.num_training_batches
                ctx.batch(int(trainer.current_epoch) + 1, int(batch_idx) + 1,
                          int(total) if isinstance(total, int) else None)
                ctx.step()

            def on_train_epoch_start(self, trainer, pl_module):
                ctx.step()

            def on_train_epoch_end(self, trainer, pl_module):
                figures = _figures({k: v for k, v in trainer.callback_metrics.items()})
                score = figures.get("ema_map50_95", figures.get("map50_95"))
                if score is not None and (self.best is None or score > self.best):
                    self.best = score
                epoch = int(trainer.current_epoch) + 1
                # Recorded without checking control: the epoch's checkpoint is written
                # after this hook, so a pause takes effect at the next batch instead.
                ctx.record_epoch(epoch, figures, best=("map", self.best) if self.best is not None else None)

        progress = AudtheiaProgress()
        original_build = rf_training.build_trainer

        def build_with_progress(*args: Any, **kwargs: Any):
            trainer = original_build(*args, **kwargs)
            trainer.callbacks.append(progress)
            return trainer

        rf_training.build_trainer = build_with_progress
        try:
            model.train(
                dataset_dir=str(dataset_dir), output_dir=str(output_dir), epochs=epochs_total,
                batch_size=int(cfg["resources"]["micro_batch"]),
                grad_accum_steps=int(cfg["resources"]["grad_accum_steps"]),
                lr=float(cfg["learning_rate"]), lr_encoder=float(cfg["encoder_learning_rate"]),
                weight_decay=float(cfg["weight_decay"]), use_ema=bool(cfg["use_ema"]),
                early_stopping=bool(cfg["early_stopping"]),
                early_stopping_patience=int(cfg["early_stopping_patience"]),
                early_stopping_min_delta=float(cfg["early_stopping_min_delta"]),
                checkpoint_interval=int(cfg["checkpoint_interval"]), best_model_metric=str(cfg["best_metric"]),
                num_workers=int(cfg["resources"]["dataloader_workers"]), seed=int(cfg["seed"]),
                device=str(cfg["device"]), resolution=int(cfg["resolution"]),
                amp_dtype=None if cfg["precision"] == "fp32" else "auto",
                tensorboard=False, progress_bar=None, run_test=False, resume=resume_from,
            )
        finally:
            rf_training.build_trainer = original_build

        ctx.log("Training finished. Exporting the best weights.")
        best_ckpt = output_dir / "checkpoint_best_total.pth"
        if not best_ckpt.is_file():
            raise RuntimeError("the best checkpoint was not written")
        export = export_onnx(model_cls, best_ckpt, run_dir / "export", names, gbif_keys,
                             notes={"run_id": cfg.get("run_id"), "dataset_signature": cfg.get("dataset_signature")})
        ctx.update(outputs={"onnx": export["onnx"], "labels": export["labels"], "checkpoint": str(best_ckpt)})
        return {"best_metric_name": "map", "best_metric_value": progress.best, "onnx": export["onnx"],
                "labels": export["labels"], "checkpoint": str(best_ckpt), "class_names": names}


def installed_version(distribution: str) -> str:
    """The installed version of a package, read from its metadata."""
    try:
        from importlib.metadata import version  # noqa: PLC0415
        return version(distribution)
    except Exception:  # noqa: BLE001
        return "unknown"


def latest_full_checkpoint(output_dir: Path) -> Path | None:
    """The newest resumable training checkpoint in a run's output folder.

    The library writes last.ckpt after every epoch when full checkpoints are kept
    less often than every epoch, and checkpoint_<epoch>.ckpt at each interval.
    With an interval of one only the numbered files exist, so both are considered
    and the one written last wins.
    """
    found = [p for p in Path(output_dir).glob("*.ckpt") if p.name == "last.ckpt" or p.name.startswith("checkpoint_")]
    if not found:
        return None
    return max(found, key=lambda p: (p.stat().st_mtime, p.name == "last.ckpt"))


def label_columns(proto: Any) -> int | None:
    """Width of the exported model's class scores: the classes plus one for
    background. None when the graph does not state it."""
    for out in proto.graph.output:
        if out.name == "labels":
            dims = out.type.tensor_type.shape.dim
            if dims and dims[-1].HasField("dim_value"):
                return int(dims[-1].dim_value)
    return None


def export_onnx(model_cls: Any, checkpoint: Path, export_dir: Path, names: list, gbif_keys: list,
                notes: Mapping | None = None) -> dict:
    """Export a checkpoint to ONNX at the model's own resolution, embed the class
    names the verifier reads, and write the labels file beside it."""
    import onnx  # noqa: PLC0415

    export_dir.mkdir(parents=True, exist_ok=True)
    work = export_dir / "_rfdetr_export"
    model = model_cls(pretrain_weights=str(checkpoint), num_classes=len(names))
    model.export(output_dir=str(work))
    produced = sorted(work.rglob("*.onnx"))
    if not produced:
        raise RuntimeError("the ONNX export wrote no file")
    target = export_dir / EXPORT_NAME
    proto = onnx.load(str(produced[0]))
    width = label_columns(proto)
    if width is not None and width != len(names) + 1:
        raise RuntimeError(f"the exported model scores {width - 1} class(es) but the dataset has {len(names)}; "
                           "nothing was saved")
    mapping = {i: name for i, name in enumerate(names)}
    meta = {p.key: p.value for p in proto.metadata_props}
    meta["names"] = repr(mapping)
    meta["audtheia_labels"] = json.dumps({str(i): {"name": n, "gbif_usage_key": k}
                                          for i, (n, k) in enumerate(zip(names, gbif_keys))}, sort_keys=True)
    if notes:
        meta["audtheia_notes"] = json.dumps(dict(notes), sort_keys=True)
    del proto.metadata_props[:]
    for key in sorted(meta):
        entry = proto.metadata_props.add()
        entry.key, entry.value = key, meta[key]
    onnx.save(proto, str(target))
    shutil.rmtree(work, ignore_errors=True)
    labels = export_dir / LABELS_NAME
    labels.write_text(json.dumps({str(i): {"name": n, "gbif_usage_key": k}
                                  for i, (n, k) in enumerate(zip(names, gbif_keys))}, indent=1, sort_keys=True),
                      encoding="utf-8")
    return {"onnx": str(target), "labels": str(labels)}


# ---------------------------------------------------------------------------
# Measuring speed for the estimate
# ---------------------------------------------------------------------------


def benchmark(base_key: str, *, device: str, threads: int, micro_batch: int, gradient_checkpointing: bool,
              resolution: int, repeats: int = 3) -> dict:
    """Seconds per training image and per evaluated image for one size on this computer.

    Runs a few forward and backward passes on random pixels with untrained
    weights, which take the same time as real ones and need no download.
    """
    import torch  # noqa: PLC0415
    import rfdetr  # noqa: PLC0415

    torch.set_num_threads(max(1, int(threads)))
    model = getattr(rfdetr, SIZE_CLASSES[base_key])(pretrain_weights=None,
                                                    gradient_checkpointing=bool(gradient_checkpointing))
    net = model.model.model.to(device)
    x = torch.rand(micro_batch, 3, resolution, resolution, device=device)
    net.train()
    out = net(x)
    sum(v.float().mean() for v in out.values() if hasattr(v, "float")).backward()
    t = time.perf_counter()
    for _ in range(repeats):
        net.zero_grad(set_to_none=True)
        out = net(x)
        sum(v.float().mean() for v in out.values() if hasattr(v, "float")).backward()
    train_s = (time.perf_counter() - t) / (repeats * micro_batch)
    net.eval()
    with torch.no_grad():
        net(x[:1])
        t = time.perf_counter()
        for _ in range(repeats):
            net(x[:1])
        eval_s = (time.perf_counter() - t) / repeats
    params_mb = sum(p.numel() * p.element_size() for p in net.parameters()) / (1024.0 * 1024.0)
    return {"base_model": base_key, "device": device, "threads": int(threads), "micro_batch": int(micro_batch),
            "parameters_mb": round(params_mb, 1),
            "gradient_checkpointing": bool(gradient_checkpointing), "resolution": int(resolution),
            "seconds_per_train_image": round(train_s, 4), "seconds_per_eval_image": round(eval_s, 4)}


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m audtheia.training.vision")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("benchmark")
    b.add_argument("--size", required=True, choices=sorted(SIZE_CLASSES))
    b.add_argument("--device", default="cpu")
    b.add_argument("--threads", type=int, default=1)
    b.add_argument("--micro-batch", type=int, default=1)
    b.add_argument("--checkpointing", action="store_true")
    b.add_argument("--resolution", type=int, required=True)
    args = parser.parse_args(argv)
    if args.cmd == "benchmark":
        result = benchmark(args.size, device=args.device, threads=args.threads, micro_batch=args.micro_batch,
                           gradient_checkpointing=args.checkpointing, resolution=args.resolution)
        print("AUDTHEIA_BENCHMARK " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
