"""The sound training engine: trains a species layer on top of a fixed base sound model.

Path: audtheia/training/acoustic.py

A base sound model, named in the settings for each acoustic engine, turns every
window of audio into a fixed list of numbers, its summary of the sound. This
engine learns only a small final layer that turns that summary into scores for
the species in the dataset. The base model itself is never changed.

Three parts.

The first part runs in the application and needs only the standard library. It
turns a request from the Train view and a built sound dataset into the complete
run configuration, and checks that Full speed and Gentle train with the same
settings. Gentle summarises one window at a time on half the processor threads;
the final layer is trained identically in both modes.

The second part is the final layer itself, written with NumPy only, so it can be
checked without TensorFlow: a linear layer trained with the Adam method on the
binary cross-entropy of every species, dropout on its input, and an exact,
resumable state saved after every epoch.

The third part runs inside the worker, in the separate sound training
environment, and is the only place TensorFlow is imported. It downloads the base
model once, checks it, summarises every window of the dataset, trains the layer
with progress, Pause and the safety checks, calibrates each species on the
validation windows, scores the test windows, and joins the base model and the
layer into one TFLite file that takes raw audio and returns one score per
species before the sigmoid. That file is loaded through the station's own
acoustic adapter and its scores are compared with the trained layer's before the
run is called finished.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Mapping, Sequence

from audtheia.training import hardware

ENGINE_KEY = "acoustic_head"

EXPORT_NAME = "model.tflite"
LABELS_TEXT_NAME = "labels.txt"
LABELS_NAME = "labels.json"
CALIBRATION_NAME = "calibration.json"
TEST_METRICS_NAME = "test_metrics.json"
HEAD_NAME = "head.npz"

# Largest difference allowed between the scores of the exported file, read
# through the station's adapter, and the trained layer's own scores. The base
# model in the file is a TFLite conversion, so tiny differences are expected.
ROUND_TRIP_TOLERANCE = 1e-3

# How the base model download is read, and how often its progress is logged.
DOWNLOAD_CHUNK = 1024 * 1024
DOWNLOAD_LOG_EVERY = 0.10

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-7


class AcousticConfigError(ValueError):
    """A sound training request that cannot become a run, in plain words."""


# ---------------------------------------------------------------------------
# Application side: building the run configuration
# ---------------------------------------------------------------------------


BASE_FIELDS = ("label", "download_url", "download_bytes", "input_key", "embeddings_signature",
               "embeddings_key", "embeddings_size", "licence")


def engine_entry(training_cfg: Mapping, key: str) -> dict:
    """The acoustic engine entry from the settings, refused when it has no base model yet."""
    for entry in (training_cfg.get("acoustic") or {}).get("engines") or []:
        if entry.get("key") == key:
            base = entry.get("base")
            if not base:
                raise AcousticConfigError(
                    f"The {entry.get('label', key)} sound engine has no base model in this version yet, so it "
                    "cannot train. Choose an engine that names a base model.")
            missing = [f for f in BASE_FIELDS if base.get(f) in (None, "")]
            if missing:
                raise AcousticConfigError(f"The base model of {key!r} is missing {', '.join(missing)} in the settings.")
            return dict(entry, base=dict(base))
    raise AcousticConfigError(f"There is no acoustic engine {key!r} in the settings.")


def window_samples(entry: Mapping) -> int:
    return int(round(float(entry["sample_rate"]) * float(entry["window_seconds"])))


def build_config(training_cfg: Mapping, *, engine_key: str, mode: str, hw: Mapping,
                 overrides: Mapping | None = None, bases_dir: str | None = None) -> dict:
    """The complete configuration a sound run is started with.

    overrides may change any of the acoustic defaults from the settings (the
    Advanced drawer) and the seed. bases_dir is where downloaded base models are
    kept, inside the sound training environment.
    """
    entry = engine_entry(training_cfg, engine_key)
    defaults = dict(((training_cfg.get("acoustic") or {}).get("defaults")) or {})
    allowed = set(defaults) | {"epochs", "seed"}
    for key in (overrides or {}):
        if key not in allowed:
            raise AcousticConfigError(f"{key!r} is not a sound training setting.")
    values = dict(defaults, **dict(overrides or {}))
    seed = int(values.get("seed", ((training_cfg.get("dataset") or {}).get("split") or {}).get("seed", 0)))
    epochs = int(values["epochs"])
    if epochs < 1:
        raise AcousticConfigError("Training needs at least one epoch.")
    batch = int(values["batch_size"])
    if batch < 1:
        raise AcousticConfigError("The batch must hold at least one window.")
    dropout = float(values["dropout"])
    if not 0.0 <= dropout < 1.0:
        raise AcousticConfigError("Dropout must be at least 0 and below 1.")
    rate = float(values["learning_rate"])
    if not rate > 0.0:
        raise AcousticConfigError("The learning rate must be above 0.")
    resources = hardware.resource_plan(mode, hw, training_cfg.get("hardware") or {}, batch_size=batch)
    base = entry["base"]
    return {
        "engine": ENGINE_KEY,
        "engine_key": engine_key,
        "engine_label": entry.get("label", engine_key),
        "base_model": engine_key,
        "base_label": base["label"],
        "base": base,
        "bases_dir": bases_dir,
        "sample_rate": int(entry["sample_rate"]),
        "window_seconds": float(entry["window_seconds"]),
        "window_samples": window_samples(entry),
        "epochs": epochs,
        "seed": seed,
        "precision": "fp32",
        "device": "cpu",
        "effective_batch": batch,
        "learning_rate": rate,
        "dropout": dropout,
        "early_stopping_patience": int(values.get("early_stopping_patience", 0)),
        "calibrate": bool(values.get("calibrate", True)),
        "report_confidence": float(((training_cfg.get("evaluation") or {}).get("report_confidence")) or 0.3),
        "calibration_bins": int(((training_cfg.get("evaluation") or {}).get("calibration_bins")) or 10),
        # Gentle summarises one window at a time; Full speed a batch at a time. The
        # final layer always trains on the full batch, so both learn the same.
        "embedding_batch": int(resources["micro_batch"]),
        "resources": resources,
        "hardware": {"limits": dict(((training_cfg.get("hardware") or {}).get("limits")) or {})},
    }


def equivalent_modes(training_cfg: Mapping, *, engine_key: str, hw: Mapping, overrides: Mapping | None = None) -> dict:
    """Build both modes for one request and show that only the division of work differs."""
    full = build_config(training_cfg, engine_key=engine_key, mode="full_speed", hw=hw, overrides=overrides)
    gentle = build_config(training_cfg, engine_key=engine_key, mode="gentle", hw=hw, overrides=overrides)
    differences = hardware.check_equivalent(full, gentle)
    for key in ("dropout", "learning_rate", "early_stopping_patience", "sample_rate", "window_samples"):
        if full.get(key) != gentle.get(key):
            differences.append(key)
    return {"full_speed": full, "gentle": gentle, "differences": differences}


def disk_mb(config: Mapping, windows: int) -> float:
    """What a run leaves on disk: the base download and its unpacked copy the first
    time, the window summaries, the layer and the exported file."""
    base = config["base"]
    download = float(base["download_bytes"]) / 1e6
    summaries = windows * int(base["embeddings_size"]) * 4 / 1e6
    export = download / 4.0
    return download * 2.0 + summaries * 1.1 + export


def base_dir(config: Mapping) -> Path:
    return Path(config["bases_dir"]) / str(config["engine_key"])


def base_ready(config: Mapping) -> bool:
    """Whether the base model of a run is already downloaded and checked."""
    marker = base_dir(config) / "base.json"
    try:
        info = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return info.get("download_url") == config["base"]["download_url"] and (Path(info.get("model_dir", "")) /
                                                                           "saved_model.pb").is_file()


# ---------------------------------------------------------------------------
# The final layer, NumPy only
# ---------------------------------------------------------------------------


def _np():
    import numpy  # noqa: PLC0415 - kept out of module level so the app imports this without NumPy work
    return numpy


def sigmoid(x):
    np = _np()
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def bce_with_logits(logits, targets) -> float:
    """Mean binary cross-entropy over every window and species, from logits."""
    np = _np()
    z = np.asarray(logits, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64)
    if z.size == 0:
        return 0.0
    return float(np.mean(np.logaddexp(0.0, z) - t * z))


class Head:
    """A linear layer from the base model's summary to one logit per species.

    Trained with Adam on the mean binary cross-entropy, with dropout on its
    input during training only. Every random draw comes from the run's seed and
    the epoch number, so a resumed run continues exactly as an uninterrupted one.
    """

    def __init__(self, n_inputs: int, n_classes: int, seed: int):
        np = _np()
        rng = np.random.RandomState(int(seed) % (2 ** 32))
        limit = math.sqrt(6.0 / (n_inputs + n_classes))
        self.W = rng.uniform(-limit, limit, size=(n_inputs, n_classes))
        self.b = np.zeros(n_classes)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)
        self.t = 0

    def logits(self, x):
        np = _np()
        return np.asarray(x, dtype=np.float64) @ self.W + self.b

    def gradients(self, x, y) -> tuple:
        """The batch loss and its gradient with respect to the weights and the bias."""
        np = _np()
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = x @ self.W + self.b
        loss = bce_with_logits(z, y)
        grad_z = (sigmoid(z) - y) / float(z.size)
        return loss, x.T @ grad_z, grad_z.sum(axis=0)

    def step(self, x, y, *, learning_rate: float, dropout: float, rng) -> float:
        """One Adam update on one batch. Returns the batch loss before the update."""
        np = _np()
        if dropout > 0.0:
            x = apply_dropout(x, dropout, rng)
        loss, gW, gb = self.gradients(x, y)
        self.t += 1
        for p, g, m, v in ((self.W, gW, self.mW, self.vW), (self.b, gb, self.mb, self.vb)):
            m *= ADAM_BETA1
            m += (1.0 - ADAM_BETA1) * g
            v *= ADAM_BETA2
            v += (1.0 - ADAM_BETA2) * g * g
            m_hat = m / (1.0 - ADAM_BETA1 ** self.t)
            v_hat = v / (1.0 - ADAM_BETA2 ** self.t)
            p -= learning_rate * m_hat / (np.sqrt(v_hat) + ADAM_EPSILON)
        return loss

    def state(self) -> dict:
        return {"W": self.W, "b": self.b, "mW": self.mW, "vW": self.vW, "mb": self.mb, "vb": self.vb,
                "t": _np().array(self.t)}

    def load_state(self, state: Mapping) -> None:
        np = _np()
        for key in ("W", "b", "mW", "vW", "mb", "vb"):
            setattr(self, key, np.array(state[key], dtype=np.float64))
        self.t = int(state["t"])


def apply_dropout(x, rate: float, rng):
    """Hide a share of the inputs at random and scale the rest up, so the expected
    input is unchanged and nothing needs rescaling when the layer is used."""
    np = _np()
    x = np.asarray(x, dtype=np.float64)
    keep = (rng.uniform(size=x.shape) >= rate).astype(np.float64)
    return x * keep / (1.0 - rate)


def epoch_batches(n: int, batch: int, seed: int, epoch: int) -> list:
    """The windows of each batch of one epoch, shuffled from the seed and the epoch."""
    np = _np()
    order = np.random.RandomState((int(seed) * 1000003 + int(epoch)) % (2 ** 32)).permutation(n)
    return [order[i:i + batch] for i in range(0, n, batch)]


def dropout_rng(seed: int, epoch: int):
    return _np().random.RandomState((int(seed) * 7919 + int(epoch) * 104729 + 1) % (2 ** 32))


def selection_key(cmap: float | None, val_loss: float | None) -> tuple:
    """How epochs are compared: the validation cmAP first, then the lower validation loss."""
    return (cmap if cmap is not None else -math.inf, -(val_loss if val_loss is not None else math.inf))


def train_head(head: Head, x_train, y_train, *, epochs: int, batch: int, learning_rate: float, dropout: float,
               seed: int, x_val=None, y_val=None, patience: int = 0, start_epoch: int = 1,
               progress: Mapping | None = None, on_batch: Callable | None = None,
               on_epoch: Callable | None = None) -> dict:
    """Train the layer, keeping the best epoch by validation cmAP.

    progress carries the state of a resumed run: best, best_key, bad_epochs. Each
    finished epoch calls on_epoch(epoch, figures, progress) after the layer and the
    progress have been updated, which is where the worker saves its checkpoint.
    """
    np = _np()
    from audtheia.training import metrics  # noqa: PLC0415

    state = dict(progress or {})
    best_key = tuple(state["best_key"]) if state.get("best_key") is not None else None
    bad = int(state.get("bad_epochs", 0))
    n = len(x_train)
    stopped_early = False
    for epoch in range(int(start_epoch), int(epochs) + 1):
        batches = epoch_batches(n, batch, seed, epoch)
        rng = dropout_rng(seed, epoch)
        losses = []
        weights = []
        for i, idx in enumerate(batches):
            losses.append(head.step(x_train[idx], y_train[idx], learning_rate=learning_rate, dropout=dropout, rng=rng))
            weights.append(len(idx))
            if on_batch is not None:
                on_batch(epoch, i + 1, len(batches))
        figures = {"train_loss": round(float(np.average(losses, weights=weights)), 6) if losses else None}
        cmap = None
        val_loss = None
        if x_val is not None and len(x_val):
            z = head.logits(x_val)
            val_loss = bce_with_logits(z, y_val)
            figures["val_loss"] = round(val_loss, 6)
            aps = [metrics.average_precision(y_val[:, c], sigmoid(z[:, c])) for c in range(z.shape[1])
                   if y_val[:, c].sum() > 0]
            aps = [a for a in aps if a is not None]
            if aps:
                cmap = float(np.mean(aps))
                figures["val_cmap"] = round(cmap, 6)
        key = selection_key(cmap, val_loss) if (x_val is not None and len(x_val)) else (float(epoch), 0.0)
        if best_key is None or key > best_key:
            best_key = key
            state["best"] = {"epoch": epoch, "W": head.W.copy(), "b": head.b.copy(), "cmap": cmap,
                             "val_loss": val_loss}
            bad = 0
        else:
            bad += 1
        state["best_key"] = list(best_key)
        state["bad_epochs"] = bad
        state["epoch"] = epoch
        if on_epoch is not None:
            on_epoch(epoch, figures, state)
        if patience > 0 and bad >= patience:
            stopped_early = True
            break
    state["stopped_early"] = stopped_early
    return state


def save_head_state(path: Path, head: Head, progress: Mapping) -> None:
    """Write the layer, its optimiser and the run's progress atomically."""
    np = _np()
    best = progress.get("best") or {}
    arrays = dict(head.state())
    if best:
        arrays["best_W"] = best["W"]
        arrays["best_b"] = best["b"]
    meta = {"epoch": progress.get("epoch"), "best_key": progress.get("best_key"),
            "bad_epochs": progress.get("bad_epochs", 0),
            "best": {k: best.get(k) for k in ("epoch", "cmap", "val_loss")} if best else None}
    arrays["meta"] = np.array(json.dumps(meta))
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def load_head_state(path: Path, head: Head) -> dict:
    np = _np()
    with np.load(path, allow_pickle=False) as data:
        head.load_state(data)
        meta = json.loads(str(data["meta"]))
        progress = {"epoch": meta["epoch"], "best_key": meta["best_key"], "bad_epochs": meta["bad_epochs"]}
        if meta.get("best") is not None:
            progress["best"] = dict(meta["best"], W=np.array(data["best_W"]), b=np.array(data["best_b"]))
    return progress


# ---------------------------------------------------------------------------
# Audio and labels
# ---------------------------------------------------------------------------


def resample(samples, source_rate: int, target_rate: int):
    """Change the sample rate by the Fourier method, which keeps the band below the
    lower of the two Nyquist frequencies and removes what is above it."""
    np = _np()
    x = np.asarray(samples, dtype=np.float64)
    if int(source_rate) == int(target_rate) or len(x) == 0:
        return x.astype(np.float32)
    n = len(x)
    m = int(round(n * float(target_rate) / float(source_rate)))
    spectrum = np.fft.rfft(x)
    keep = min(len(spectrum), m // 2 + 1)
    out_spectrum = np.zeros(m // 2 + 1, dtype=complex)
    out_spectrum[:keep] = spectrum[:keep]
    return (np.fft.irfft(out_spectrum, m) * (m / n)).astype(np.float32)


def fit_length(samples, length: int):
    np = _np()
    x = np.asarray(samples, dtype=np.float32)
    if len(x) >= length:
        return x[:length]
    return np.concatenate([x, np.zeros(length - len(x), dtype=np.float32)])


def dataset_windows(dataset_dir: Path) -> tuple:
    """The class names in label order with their GBIF keys and tested flags, and the
    windows of each split with their species, from a built sound dataset."""
    classes = json.loads((dataset_dir / "classes.json").read_text(encoding="utf-8"))
    ordered = sorted(classes, key=lambda c: int(c["class_index"]))
    names = [str(c["class_name"]) for c in ordered]
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("modality") == "vision":
        raise ValueError("this dataset holds images; the sound engine trains on a sound dataset")
    index = {name: i for i, name in enumerate(names)}
    splits: dict = {"train": [], "validation": [], "test": []}
    for item in manifest.get("items") or []:
        unknown = [label for label in item.get("labels") or [] if label not in index]
        if unknown:
            raise ValueError(f"the window {item['file']} is labelled {unknown[0]!r}, which is not a class of the dataset")
        splits.setdefault(item["split"], []).append({
            "file": item["file"], "labels": [index[label] for label in item.get("labels") or []],
            "sample_rate": int(item["sample_rate"]),
        })
    return names, [c.get("gbif_usage_key") for c in ordered], [bool(c.get("tested", True)) for c in ordered], splits


def label_matrix(items: Sequence[Mapping], n_classes: int):
    np = _np()
    y = np.zeros((len(items), n_classes), dtype=np.float64)
    for row, item in enumerate(items):
        for c in item["labels"]:
            y[row, c] = 1.0
    return y


# ---------------------------------------------------------------------------
# Worker side: the base model
# ---------------------------------------------------------------------------


def _safe_extract(archive: zipfile.ZipFile, target: Path) -> None:
    """Extract an archive, refusing any member that would land outside the target."""
    root = target.resolve()
    for member in archive.infolist():
        destination = (target / member.filename).resolve()
        if destination != root and root not in destination.parents:
            raise RuntimeError(f"the downloaded archive holds a file outside its own folder ({member.filename})")
    archive.extractall(target)


def check_base(model_dir: Path, base: Mapping, samples: int):
    """Load a base SavedModel and confirm it is the one the settings describe.
    Returns the summary function."""
    import tensorflow as tf  # noqa: PLC0415

    model = tf.saved_model.load(str(model_dir))
    signature = model.signatures.get(base["embeddings_signature"])
    if signature is None:
        raise RuntimeError(f"the base model has no {base['embeddings_signature']!r} entry point; it is not the "
                           f"{base['label']} model the settings describe")
    inputs = signature.structured_input_signature[1]
    spec = inputs.get(base["input_key"])
    if spec is None or list(spec.shape)[-1] != int(samples):
        raise RuntimeError(f"the base model does not take {samples} samples under {base['input_key']!r}")
    outputs = signature.structured_outputs
    out = outputs.get(base["embeddings_key"])
    if out is None or list(out.shape)[-1] != int(base["embeddings_size"]):
        raise RuntimeError(f"the base model's summary is not {base['embeddings_size']} numbers long")
    return model, signature


def ensure_base(config: Mapping, ctx, *, opener: Callable = urllib.request.urlopen) -> Path:
    """Download the base model once into the sound training environment and check it.
    Returns the folder of the SavedModel."""
    base = config["base"]
    target = base_dir(config)
    marker = target / "base.json"
    if base_ready(config):
        info = json.loads(marker.read_text(encoding="utf-8"))
        return Path(info["model_dir"])
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    part = target / "download.zip.part"
    expected = int(base["download_bytes"])
    ctx.log(f"Downloading the {base['label']} base model once, {expected / 1e6:.1f} MB, into {target}.")
    digest = hashlib.sha256()
    request = urllib.request.Request(base["download_url"], headers={"User-Agent": "audtheia-training"})
    got = 0
    next_note = DOWNLOAD_LOG_EVERY
    with opener(request, timeout=60) as response, open(part, "wb") as handle:
        while True:
            chunk = response.read(DOWNLOAD_CHUNK)
            if not chunk:
                break
            handle.write(chunk)
            digest.update(chunk)
            got += len(chunk)
            if expected and got / expected >= next_note:
                ctx.log(f"Downloaded {got / 1e6:.1f} of {expected / 1e6:.1f} MB.")
                next_note += DOWNLOAD_LOG_EVERY
            ctx.step()
    if got != expected:
        raise RuntimeError(f"the base model download is {got} bytes, not the {expected} the settings expect; it may "
                           "have been cut off, so nothing was kept")
    unpacked = target / "unpacked"
    with zipfile.ZipFile(part) as archive:
        _safe_extract(archive, unpacked)
    part.unlink()
    candidates = sorted(p.parent for p in unpacked.rglob("saved_model.pb"))
    chosen = None
    problems = []
    for candidate in candidates:
        try:
            check_base(candidate, base, int(config["window_samples"]))
        except Exception as exc:  # noqa: BLE001 - try the next candidate, report all if none fits
            problems.append(f"{candidate.name}: {exc}")
            continue
        chosen = candidate
        break
    if chosen is None:
        raise RuntimeError("the downloaded archive holds no base model that matches the settings"
                           + (f" ({'; '.join(problems)})" if problems else ""))
    marker.write_text(json.dumps({"download_url": base["download_url"], "download_bytes": expected,
                                  "sha256": digest.hexdigest(), "model_dir": str(chosen), "label": base["label"],
                                  "licence": base["licence"]}, indent=1), encoding="utf-8")
    ctx.log(f"The {base['label']} base model is ready and matches the settings.")
    return chosen


# ---------------------------------------------------------------------------
# Worker side: the engine
# ---------------------------------------------------------------------------


def summarise(signature, base: Mapping, windows, batch: int, on_batch: Callable | None = None):
    """The base model's summary of every window, batch by batch."""
    np = _np()
    import tensorflow as tf  # noqa: PLC0415

    out = np.zeros((len(windows), int(base["embeddings_size"])), dtype=np.float32)
    total = max(1, math.ceil(len(windows) / batch))
    for k, start in enumerate(range(0, len(windows), batch)):
        chunk = np.stack(windows[start:start + batch]).astype(np.float32)
        result = signature(**{base["input_key"]: tf.constant(chunk)})
        out[start:start + len(chunk)] = np.asarray(result[base["embeddings_key"]])
        if on_batch is not None:
            on_batch(k + 1, total)
    return out


def fuse_to_tflite(model, signature, base: Mapping, samples: int, W, b) -> bytes:
    """One TFLite file from raw audio to species logits: the base model's summary
    followed by the trained layer. No sigmoid, because the adapter applies it."""
    import tensorflow as tf  # noqa: PLC0415

    weights = tf.constant(_np().asarray(W, dtype="float32"))
    bias = tf.constant(_np().asarray(b, dtype="float32"))
    input_key, output_key = base["input_key"], base["embeddings_key"]

    class Fused(tf.Module):
        def __init__(self):
            super().__init__()
            self.base = model

        @tf.function(input_signature=[tf.TensorSpec([1, int(samples)], tf.float32)])
        def __call__(self, audio):
            summary = signature(**{input_key: audio})[output_key]
            return tf.matmul(summary, weights) + bias

    fused = Fused()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([fused.__call__.get_concrete_function()], fused)
    return converter.convert()


class AcousticEngine:
    """Runs one sound training run inside the worker."""

    def __init__(self, config: Mapping):
        self.config = dict(config)

    def run(self, ctx) -> dict:
        np = _np()
        cfg = self.config
        threads = int((cfg.get("resources") or {}).get("cpu_threads") or 1)
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf  # noqa: PLC0415

        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        from audtheia.training import metrics  # noqa: PLC0415
        from audtheia.training import windows as window_rules  # noqa: PLC0415

        run_dir = Path(ctx.run_dir)
        dataset_dir = Path(cfg["dataset_dir"])
        base = cfg["base"]
        samples = int(cfg["window_samples"])
        names, gbif_keys, tested, splits = dataset_windows(dataset_dir)
        if not splits["train"]:
            raise RuntimeError("the dataset has no training windows")
        ctx.log(f"Preparing the {cfg['engine_label']} engine on the {base['label']} base model: {len(names)} "
                f"species, {len(splits['train'])} training, {len(splits['validation'])} validation and "
                f"{len(splits['test'])} test windows, {threads} processor thread(s).")
        model_dir = ensure_base(cfg, ctx)
        model, signature = check_base(model_dir, base, samples)

        summaries_dir = run_dir / "summaries"
        summaries_dir.mkdir(parents=True, exist_ok=True)
        features: dict = {}
        for split in ("train", "validation", "test"):
            items = splits.get(split) or []
            path = summaries_dir / f"{split}.npy"
            if path.is_file():
                cached = np.load(path, allow_pickle=False)
                if cached.shape == (len(items), int(base["embeddings_size"])):
                    features[split] = cached
                    continue
            if not items:
                features[split] = np.zeros((0, int(base["embeddings_size"])), dtype=np.float32)
                continue
            ctx.update(stage="summarising")
            ctx.log(f"Summarising the {len(items)} {split} window(s) with the base model.")
            audio = []
            for item in items:
                raw, rate = window_rules.read_wav(dataset_dir / item["file"])
                audio.append(fit_length(resample(raw, rate, int(cfg["sample_rate"])), samples))
                ctx.step()

            def note(done, total):
                ctx.batch(0, done, total)
                ctx.step()

            features[split] = summarise(signature, base, audio, int(cfg["embedding_batch"]), on_batch=note)
            np.save(path.with_name(path.stem + ".tmp.npy"), features[split])
            os.replace(path.with_name(path.stem + ".tmp.npy"), path)

        n_classes = len(names)
        y = {split: label_matrix(splits.get(split) or [], n_classes) for split in ("train", "validation", "test")}
        head = Head(int(base["embeddings_size"]), n_classes, int(cfg["seed"]))
        state_path = run_dir / "checkpoints" / "head_state.npz"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        progress: dict = {}
        start_epoch = 1
        if ctx.resume:
            if state_path.is_file():
                progress = load_head_state(state_path, head)
                start_epoch = int(progress["epoch"]) + 1
                ctx.log(f"Resuming after epoch {progress['epoch']}.")
            else:
                ctx.nothing_to_resume()
        epochs_total = int(cfg["epochs"])
        ctx.update(stage="training")
        ctx.set_state("running", epochs_total=epochs_total)

        def on_batch(epoch, done, total):
            ctx.batch(epoch, done, total)
            ctx.step()

        def on_epoch(epoch, figures, state):
            save_head_state(state_path, head, state)
            best = state.get("best") or {}
            ctx.record_epoch(epoch, figures, best=("cmap", best.get("cmap")) if best.get("cmap") is not None else None)
            ctx.step()

        x_val = features["validation"] if len(features["validation"]) else None
        y_val = y["validation"] if x_val is not None else None
        if start_epoch <= epochs_total:
            progress = train_head(
                head, features["train"].astype(np.float64), y["train"], epochs=epochs_total,
                batch=int(cfg["effective_batch"]), learning_rate=float(cfg["learning_rate"]),
                dropout=float(cfg["dropout"]), seed=int(cfg["seed"]),
                x_val=None if x_val is None else x_val.astype(np.float64), y_val=y_val,
                patience=int(cfg["early_stopping_patience"]), start_epoch=start_epoch, progress=progress,
                on_batch=on_batch, on_epoch=on_epoch)
            if progress.get("stopped_early"):
                ctx.log(f"Stopped early after epoch {progress['epoch']}: the validation score had not improved for "
                        f"{cfg['early_stopping_patience']} epochs.")
        best = progress.get("best") or {}
        if not best:
            raise RuntimeError("no epoch finished, so there is no layer to export")
        W, b = best["W"], best["b"]
        ctx.log(f"Training finished. The best epoch was {best['epoch']}. Calibrating and scoring the test windows.")
        ctx.update(stage="exporting")

        def scores_of(x):
            return sigmoid(np.asarray(x, dtype=np.float64) @ W + b)

        calibrations = None
        if cfg.get("calibrate") and x_val is not None:
            calibrations = metrics.fit_calibrations(y["validation"], scores_of(x_val), names)
        test_metrics = None
        if len(features["test"]):
            test_metrics = metrics.evaluate_acoustic(
                y["test"], scores_of(features["test"]), names, threshold=float(cfg["report_confidence"]),
                calibrations=calibrations, calibration_bins=int(cfg["calibration_bins"]))
        export_dir = run_dir / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        ctx.log("Joining the base model and the trained layer into one TFLite file.")
        data = fuse_to_tflite(model, signature, base, samples, W, b)
        (export_dir / EXPORT_NAME).write_bytes(data)
        (export_dir / LABELS_TEXT_NAME).write_text("".join(n + "\n" for n in names), encoding="utf-8")
        (export_dir / LABELS_NAME).write_text(json.dumps(
            {str(i): {"name": n, "gbif_usage_key": k, "tested": t} for i, (n, k, t) in enumerate(zip(names, gbif_keys,
                                                                                                    tested))},
            indent=1, sort_keys=True), encoding="utf-8")
        np.savez(export_dir / HEAD_NAME, W=np.asarray(W, dtype=np.float32), b=np.asarray(b, dtype=np.float32))
        (export_dir / CALIBRATION_NAME).write_text(json.dumps({
            "note": ("Thresholds at which each species' calibrated probability reaches the target, fitted on the "
                     "validation windows. They are recorded with the model and are not applied by the station, "
                     "whose adapter has no threshold input."),
            "classes": calibrations}, indent=1, default=float), encoding="utf-8")
        if test_metrics is not None:
            (export_dir / TEST_METRICS_NAME).write_text(json.dumps(test_metrics, indent=1, default=float),
                                                        encoding="utf-8")
        check = round_trip(export_dir, cfg, names, features["test"] if len(features["test"]) else features["train"],
                           splits["test"] if len(features["test"]) else splits["train"], dataset_dir, W, b)
        ctx.log(f"The exported file was loaded through the station's acoustic adapter on {check['windows']} window(s). "
                f"Largest difference from the trained layer's scores: {check['max_difference']:.2e}.")
        headline = None
        if test_metrics is not None:
            headline = test_metrics.get("cmap")
        return {"best_metric_name": "cmap", "best_metric_value": best.get("cmap"), "best_epoch": best["epoch"],
                "test_cmap": headline, "tflite": str(export_dir / EXPORT_NAME),
                "labels": str(export_dir / LABELS_TEXT_NAME), "class_names": names, "round_trip": check}


def round_trip(export_dir: Path, cfg: Mapping, names: Sequence[str], features, items: Sequence[Mapping],
               dataset_dir: Path, W, b, limit: int = 8) -> dict:
    """Load the exported file through the station's own adapter and compare its
    scores with the trained layer's on real windows. Raises when they differ."""
    np = _np()
    from audtheia.pipeline.acoustic import TFLiteAcousticModel  # noqa: PLC0415
    from audtheia.training import windows as window_rules  # noqa: PLC0415

    adapter = TFLiteAcousticModel(model_path=export_dir / EXPORT_NAME, sample_rate=int(cfg["sample_rate"]),
                                  window_seconds=float(cfg["window_seconds"]),
                                  labels_path=export_dir / LABELS_TEXT_NAME)
    try:
        if [adapter.class_names.get(i) for i in range(len(names))] != list(names):
            raise RuntimeError("the adapter read the species names in a different order from the training")
        worst = 0.0
        count = min(limit, len(items))
        for k in range(count):
            raw, rate = window_rules.read_wav(dataset_dir / items[k]["file"])
            audio = fit_length(resample(raw, rate, int(cfg["sample_rate"])), int(cfg["window_samples"]))
            got = np.array([d.confidence for d in sorted(adapter.detect(audio, int(cfg["sample_rate"])),
                                                         key=lambda d: d.class_id)])
            want = sigmoid(np.asarray(features[k], dtype=np.float64) @ W + b)
            worst = max(worst, float(np.max(np.abs(got - want))))
    finally:
        adapter.close()
    if worst > ROUND_TRIP_TOLERANCE:
        raise RuntimeError(f"the exported file's scores differ from the trained layer's by {worst:.2e}, more than "
                           f"the {ROUND_TRIP_TOLERANCE:g} allowed; it was not kept as finished")
    return {"windows": count, "max_difference": worst, "tolerance": ROUND_TRIP_TOLERANCE}


# ---------------------------------------------------------------------------
# Measuring speed for the estimate
# ---------------------------------------------------------------------------


def benchmark(config: Mapping, *, repeats: int = 3) -> dict:
    """Seconds the base model takes to summarise one window on this computer, in the
    way the run would summarise them. Needs the base model downloaded."""
    np = _np()
    threads = int((config.get("resources") or {}).get("cpu_threads") or 1)
    import tensorflow as tf  # noqa: PLC0415

    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    marker = json.loads((base_dir(config) / "base.json").read_text(encoding="utf-8"))
    model, signature = check_base(Path(marker["model_dir"]), config["base"], int(config["window_samples"]))
    batch = int(config["embedding_batch"])
    audio = [np.random.RandomState(0).standard_normal(int(config["window_samples"])).astype(np.float32) * 0.05
             for _ in range(batch)]
    summarise(signature, config["base"], audio, batch)
    start = time.perf_counter()
    for _ in range(repeats):
        summarise(signature, config["base"], audio, batch)
    seconds = (time.perf_counter() - start) / (repeats * batch)
    del model
    return {"seconds_per_window": round(seconds, 4), "threads": threads, "embedding_batch": batch}


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m audtheia.training.acoustic")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("benchmark")
    b.add_argument("--config", required=True, help="a JSON file holding the run configuration")
    args = parser.parse_args(argv)
    if args.cmd == "benchmark":
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        print("AUDTHEIA_BENCHMARK " + json.dumps(benchmark(config), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
