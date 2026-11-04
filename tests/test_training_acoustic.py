"""Verification for the sound training engine, without TensorFlow.

Path: tests/test_training_acoustic.py

Checks that a request from the Train view becomes a complete sound run
configuration taken from the settings, that an engine with no base model yet is
refused in plain words, and that Full speed and Gentle differ only in how many
windows the base model summarises at a time. Checks the final layer itself: its
gradient against a finite-difference estimate, that it learns separable species,
that a run paused after any epoch and resumed ends with exactly the weights of an
uninterrupted run, and how the best epoch is chosen. Checks the change of sample
rate, the reading of a real built sound dataset, and the base model download:
the size is checked, an archive that would write outside its folder is refused,
a download is done once, and Pause works during it. The real base model, the
joining into one TFLite file and the round trip through the station's adapter
need TensorFlow and are proven separately with the real libraries.

Uses the standard library, NumPy and Pillow, plus the project itself.
"""

from __future__ import annotations

import ast
import io
import json
import math
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from training_fixture import World  # noqa: E402
from audtheia.config import DEFAULT_TRAINING  # noqa: E402
from audtheia.training import acoustic as A  # noqa: E402
from audtheia.training import dataset as ds  # noqa: E402
from audtheia.training import worker as W  # noqa: E402

CHECKS = {"passed": 0, "failed": 0}
LAPTOP = {"platform": "windows", "cpu": {"logical_cores": 12}, "gpus": []}
SP1 = ("Turdus migratorius", 9201)
SP2 = ("Cardinalis cardinalis", 9202)


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
    cfg = A.build_config(DEFAULT_TRAINING, engine_key="acoustic_head_a", mode="gentle", hw=LAPTOP, bases_dir="/b")
    d = DEFAULT_TRAINING["acoustic"]["defaults"]
    engine = DEFAULT_TRAINING["acoustic"]["engines"][0]
    check("the window and base model come from the settings", cfg["sample_rate"] == 48000
          and cfg["window_samples"] == 144000 and cfg["base"] == engine["base"] and cfg["base_label"] == "BirdNET 2.4")
    check("every training value comes from the settings defaults", cfg["epochs"] == d["epochs"]
          and cfg["effective_batch"] == d["batch_size"] and cfg["learning_rate"] == d["learning_rate"]
          and cfg["dropout"] == d["dropout"] and cfg["early_stopping_patience"] == d["early_stopping_patience"]
          and cfg["calibrate"] is d["calibrate"])
    check("the worker runs the sound engine", cfg["engine"] == "acoustic_head"
          and "acoustic_head" in W.ENGINES)
    check("Gentle summarises one window at a time on half the threads", cfg["embedding_batch"] == 1
          and cfg["resources"]["cpu_threads"] == 6)
    full = A.build_config(DEFAULT_TRAINING, engine_key="acoustic_head_a", mode="full_speed", hw=LAPTOP, bases_dir="/b")
    check("Full speed summarises a whole batch at a time", full["embedding_batch"] == d["batch_size"])
    pair = A.equivalent_modes(DEFAULT_TRAINING, engine_key="acoustic_head_a", hw=LAPTOP)
    f, g = pair["full_speed"], pair["gentle"]
    check("only the division of work differs between the modes", pair["differences"] == []
          and {k for k in f if f[k] != g[k]} == {"resources", "embedding_batch"})
    try:
        A.build_config(DEFAULT_TRAINING, engine_key="acoustic_head_b", mode="gentle", hw=LAPTOP)
        reason = ""
    except A.AcousticConfigError as exc:
        reason = str(exc)
    check("an engine with no base model yet is refused in plain words", "no base model in this version yet" in reason)
    for bad, label in (({"lr": 1}, "an unknown setting"), ({"epochs": 0}, "zero epochs"),
                       ({"dropout": 1.0}, "a dropout of 1"), ({"learning_rate": 0}, "a learning rate of 0"),
                       ({"batch_size": 0}, "an empty batch")):
        check(f"{label} is refused", raises(lambda b=bad: A.build_config(
            DEFAULT_TRAINING, engine_key="acoustic_head_a", mode="gentle", hw=LAPTOP, overrides=b),
            A.AcousticConfigError))
    check("an engine not in the settings is refused",
          raises(lambda: A.build_config(DEFAULT_TRAINING, engine_key="nope", mode="gentle", hw=LAPTOP),
                 A.AcousticConfigError))
    broken = json.loads(json.dumps(DEFAULT_TRAINING))
    del broken["acoustic"]["engines"][0]["base"]["embeddings_key"]
    check("a base model description missing a field is refused",
          raises(lambda: A.build_config(broken, engine_key="acoustic_head_a", mode="gentle", hw=LAPTOP),
                 A.AcousticConfigError))
    mb = A.disk_mb(cfg, 1000)
    check("the disk estimate counts the download, its unpacked copy, the summaries and the export",
          abs(mb - (124.522908 * 2 + 1000 * 1024 * 4 / 1e6 * 1.1 + 124.522908 / 4)) < 1e-6)


def test_head():
    print("the final layer")
    rng = np.random.RandomState(3)
    x = rng.randn(12, 7)
    y = (rng.rand(12, 3) > 0.5).astype(float)
    head = A.Head(7, 3, seed=5)
    loss, gW, gb = head.gradients(x, y)
    numeric = np.zeros_like(head.W)
    for i in range(7):
        for j in range(3):
            for sign in (1, -1):
                head.W[i, j] += sign * 1e-6
                numeric[i, j] += sign * A.bce_with_logits(head.logits(x), y) / 2e-6
                head.W[i, j] -= sign * 1e-6
    numeric_b = np.zeros_like(head.b)
    for j in range(3):
        for sign in (1, -1):
            head.b[j] += sign * 1e-6
            numeric_b[j] += sign * A.bce_with_logits(head.logits(x), y) / 2e-6
            head.b[j] -= sign * 1e-6
    check("the gradient matches a finite-difference estimate",
          float(np.max(np.abs(gW - numeric))) < 1e-7 and float(np.max(np.abs(gb - numeric_b))) < 1e-7
          and abs(loss - A.bce_with_logits(head.logits(x), y)) < 1e-12)
    before = head.W.copy()
    head.step(x, y, learning_rate=0.01, dropout=0.0, rng=None)
    check("the first Adam step moves each weight by the learning rate against its gradient",
          np.allclose(head.W - before, -0.01 * np.sign(gW), rtol=0, atol=1e-6) and head.t == 1)
    draws = np.mean([A.apply_dropout(np.ones((4, 5)), 0.25, np.random.RandomState(s)) for s in range(4000)], axis=0)
    hidden = A.apply_dropout(np.ones((200, 50)), 0.25, np.random.RandomState(1))
    check("dropout hides about the chosen share and keeps the expected input unchanged",
          float(np.max(np.abs(draws - 1.0))) < 0.05 and abs(float((hidden == 0).mean()) - 0.25) < 0.02)
    check("the loss from logits is stable for very large logits",
          math.isfinite(A.bce_with_logits(np.array([[1e4, -1e4]]), np.array([[0.0, 1.0]]))))
    check("the sigmoid is stable at both ends", float(A.sigmoid(np.array([-1e4]))[0]) == 0.0
          and float(A.sigmoid(np.array([1e4]))[0]) == 1.0)

    centres = rng.randn(3, 16) * 3
    def batch(n, seed):
        r = np.random.RandomState(seed)
        cls = r.randint(0, 3, n)
        feats = centres[cls] + r.randn(n, 16) * 0.5
        lab = np.zeros((n, 3)); lab[np.arange(n), cls] = 1
        return feats, lab
    xt, yt = batch(90, 1)
    xv, yv = batch(30, 2)
    learner = A.Head(16, 3, seed=1)
    state = A.train_head(learner, xt, yt, epochs=30, batch=16, learning_rate=0.05, dropout=0.1, seed=1,
                         x_val=xv, y_val=yv)
    check("it learns separable species", state["best"]["cmap"] is not None and state["best"]["cmap"] > 0.99)

    def run(split_at=None):
        h = A.Head(16, 3, seed=9)
        seen = []
        if split_at is None:
            st = A.train_head(h, xt, yt, epochs=10, batch=8, learning_rate=0.02, dropout=0.25, seed=9,
                              x_val=xv, y_val=yv, on_epoch=lambda e, f, s: seen.append(f))
            return h, st, seen
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "state.npz"
            st = A.train_head(h, xt, yt, epochs=split_at, batch=8, learning_rate=0.02, dropout=0.25, seed=9,
                              x_val=xv, y_val=yv, on_epoch=lambda e, f, s: (A.save_head_state(path, h, s),
                                                                            seen.append(f)))
            again = A.Head(16, 3, seed=123)
            progress = A.load_head_state(path, again)
            st = A.train_head(again, xt, yt, epochs=10, batch=8, learning_rate=0.02, dropout=0.25, seed=9,
                              x_val=xv, y_val=yv, start_epoch=progress["epoch"] + 1, progress=progress,
                              on_epoch=lambda e, f, s: seen.append(f))
            return again, st, seen
    a, sa, fa = run()
    b, sb, fb = run(split_at=4)
    check("a run paused after epoch 4 and resumed ends with exactly the weights of an uninterrupted run",
          np.array_equal(a.W, b.W) and np.array_equal(a.b, b.b) and a.t == b.t)
    check("and with the same best epoch and the same figures every epoch",
          sa["best"]["epoch"] == sb["best"]["epoch"] and np.array_equal(sa["best"]["W"], sb["best"]["W"]) and fa == fb)
    check("the epochs are compared by validation cmAP first, then by the lower validation loss",
          A.selection_key(0.9, 0.5) > A.selection_key(0.8, 0.1) and A.selection_key(0.9, 0.2) > A.selection_key(0.9, 0.3)
          and A.selection_key(None, 0.1) < A.selection_key(0.0, 9.0))
    flat = A.Head(16, 3, seed=2)
    stopped = A.train_head(flat, xt, yt, epochs=50, batch=16, learning_rate=0.0, dropout=0.0, seed=2,
                           x_val=xv, y_val=yv, patience=3)
    check("early stopping ends a run whose validation score stops rising",
          stopped["stopped_early"] is True and stopped["epoch"] == 4 and stopped["best"]["epoch"] == 1)
    nov = A.Head(16, 3, seed=2)
    last = A.train_head(nov, xt, yt, epochs=3, batch=16, learning_rate=0.01, dropout=0.0, seed=2)
    check("without validation windows the last epoch is kept", last["best"]["epoch"] == 3)
    check("the batches of an epoch cover every window once, in an order set by the seed and the epoch",
          sorted(np.concatenate(A.epoch_batches(10, 3, 7, 2)).tolist()) == list(range(10))
          and [b.tolist() for b in A.epoch_batches(10, 3, 7, 2)] == [b.tolist() for b in A.epoch_batches(10, 3, 7, 2)]
          and [b.tolist() for b in A.epoch_batches(10, 3, 7, 2)] != [b.tolist() for b in A.epoch_batches(10, 3, 7, 3)])


def test_audio():
    print("sample rate and window length")
    rate = 24000
    t = np.arange(rate) / rate
    tone = np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)
    up = A.resample(tone, rate, 48000)
    spectrum = np.abs(np.fft.rfft(up))
    peak = np.argmax(spectrum) * 48000 / len(up)
    check("doubling the rate keeps the length in time, the pitch and the strength",
          len(up) == 48000 and abs(peak - 1000.0) < 1.5 and abs(float(np.max(np.abs(up))) - 1.0) < 0.01)
    high = np.sin(2 * np.pi * 20000.0 * np.arange(48000) / 48000).astype(np.float32)
    down = A.resample(high, 48000, 32000)
    check("lowering the rate removes sound above the new limit instead of folding it down",
          len(down) == 32000 and float(np.max(np.abs(down))) < 1e-3)
    check("the same rate leaves the samples alone", np.array_equal(A.resample(tone, rate, rate), tone))
    check("a window is padded with silence or cut to its exact length",
          len(A.fit_length(np.ones(5), 8)) == 8 and float(A.fit_length(np.ones(5), 8)[-1]) == 0.0
          and len(A.fit_length(np.ones(9), 8)) == 8)


def test_dataset():
    print("reading a built sound dataset")
    with tempfile.TemporaryDirectory() as td:
        w = World(Path(td))
        for i in range(6):
            e = w.audio_event([SP1], station=f"A{i % 3}", first_seen=f"2026-03-{i + 1:02d}T06:00:00Z"); w.confirm(e)
            e = w.audio_event([SP2], station=f"A{i % 3}", first_seen=f"2026-04-{i + 1:02d}T06:00:00Z"); w.confirm(e)
        for i in range(3):
            e = w.audio_event([SP1], station="A0", first_seen=f"2026-05-{i + 1:02d}T06:00:00Z", burst=None); w.reject(e)
        built = ds.build_dataset(w.db, w.settings, ds.options_from_request(
            w.settings, {"modality": "audio", "engine": "acoustic_head_a", "background_max_fraction": 1.0}))
        folder = Path(built["path"])
        names, keys, tested, splits = A.dataset_windows(folder)
        check("the species come in label order with their GBIF keys",
              names == sorted(names) and set(names) == {SP1[0], SP2[0]} and len(keys) == 2 and len(tested) == 2)
        every = [i for s in splits.values() for i in s]
        y = A.label_matrix(every, len(names))
        check("each window carries its species, and a rejected clip none",
              y.shape == (len(every), 2) and int((y.sum(axis=1) == 0).sum()) >= 1 and int(y.sum()) == len(every)
              - int((y.sum(axis=1) == 0).sum()))
        check("the windows exist where the manifest says", all((folder / i["file"]).is_file() for i in every))
        listed = [d for d in ds.list_datasets(w.settings) if d["dataset_id"] == built["dataset_id"]]
        check("the list of built datasets names the sound engine", listed and listed[0]["engine"] == "acoustic_head_a")
        manifest = json.loads((folder / "manifest.json").read_text())
        manifest["items"][0]["labels"] = ["Nobody nowhere"]
        (folder / "manifest.json").write_text(json.dumps(manifest))
        check("a window labelled with a class the dataset does not have is refused",
              raises(lambda: A.dataset_windows(folder), ValueError))
        manifest["modality"] = "vision"
        (folder / "manifest.json").write_text(json.dumps(manifest))
        check("an image dataset is refused", raises(lambda: A.dataset_windows(folder), ValueError))


class FakeCtx:
    def __init__(self, stop_after=None):
        self.logs = []
        self.steps = 0
        self.stop_after = stop_after

    def log(self, text):
        self.logs.append(text)

    def step(self):
        self.steps += 1
        if self.stop_after is not None and self.steps >= self.stop_after:
            raise W.PauseRequested("paused", "Paused at your request.")


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def make_zip(members: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_base_download():
    print("the base model download")
    real_check = A.check_base
    checked = []
    A.check_base = lambda model_dir, base, samples: checked.append(Path(model_dir).name) or ("m", "s")
    try:
        with tempfile.TemporaryDirectory() as td:
            data = make_zip({"audio-model/saved_model.pb": b"pb", "audio-model/variables/variables.index": b"i",
                             "labels/en_us.txt": b"a\n"})
            calls = []

            def opener(request, timeout):
                calls.append(request.full_url)
                return FakeResponse(data)

            cfg = A.build_config(DEFAULT_TRAINING, engine_key="acoustic_head_a", mode="gentle", hw=LAPTOP,
                                 bases_dir=str(Path(td) / "bases"))
            cfg["base"] = dict(cfg["base"], download_bytes=len(data))
            ctx = FakeCtx()
            model_dir = A.ensure_base(cfg, ctx, opener=opener)
            check("the base model is downloaded once, unpacked and checked", model_dir.name == "audio-model"
                  and checked == ["audio-model"] and A.base_ready(cfg) and not (A.base_dir(cfg) / "download.zip.part").exists())
            marker = json.loads((A.base_dir(cfg) / "base.json").read_text())
            check("the record of the download keeps its address, size, checksum and licence",
                  marker["download_bytes"] == len(data) and len(marker["sha256"]) == 64
                  and marker["licence"] == "CC BY-NC-SA 4.0")
            A.ensure_base(cfg, ctx, opener=opener)
            check("a second run uses the downloaded copy", len(calls) == 1)
            check("Pause and the safety checks are obeyed while downloading", ctx.steps >= 1)

            short = dict(cfg, bases_dir=str(Path(td) / "short"))
            short["base"] = dict(cfg["base"], download_bytes=len(data) + 1)
            try:
                A.ensure_base(short, FakeCtx(), opener=opener)
                reason = ""
            except RuntimeError as exc:
                reason = str(exc)
            check("a download of the wrong size is refused and not kept", "not the" in reason
                  and not A.base_ready(short))

            evil = make_zip({"../../escape.txt": b"x"})
            bad = dict(cfg, bases_dir=str(Path(td) / "evil"))
            bad["base"] = dict(cfg["base"], download_bytes=len(evil))
            try:
                A.ensure_base(bad, FakeCtx(), opener=lambda r, timeout: FakeResponse(evil))
                reason = ""
            except RuntimeError as exc:
                reason = str(exc)
            check("an archive that would write outside its folder is refused",
                  "outside its own folder" in reason and not (Path(td) / "escape.txt").exists())

            empty = make_zip({"readme.txt": b"no model"})
            none = dict(cfg, bases_dir=str(Path(td) / "none"))
            none["base"] = dict(cfg["base"], download_bytes=len(empty))
            check("an archive without a matching model is refused",
                  raises(lambda: A.ensure_base(none, FakeCtx(), opener=lambda r, timeout: FakeResponse(empty)),
                         RuntimeError))

            paused = dict(cfg, bases_dir=str(Path(td) / "paused"))
            check("a pause during the download stops it",
                  raises(lambda: A.ensure_base(paused, FakeCtx(stop_after=1), opener=opener), W.PauseRequested)
                  and not A.base_ready(paused))
            A.ensure_base(paused, FakeCtx(), opener=opener)
            check("and the next start downloads it again from the beginning", A.base_ready(paused))
    finally:
        A.check_base = real_check


def test_imports():
    print("imports")
    src = (REPO / "audtheia" / "training" / "acoustic.py").read_text(encoding="utf-8")
    check("acoustic.py is ASCII only", all(ord(c) < 128 for c in src))
    top = set()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Import):
            top.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            top.add((node.module or "").split(".")[0])
    check("no machine-learning library or NumPy is imported at module level",
          not top & {"tensorflow", "numpy", "ai_edge_litert", "tflite_runtime"})


def main() -> int:
    test_config()
    test_head()
    test_audio()
    test_dataset()
    test_base_download()
    test_imports()
    print(f"\nResult: {CHECKS['passed']} passed, {CHECKS['failed']} failed")
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
