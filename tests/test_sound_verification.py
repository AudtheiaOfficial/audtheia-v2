"""Checks for the second sound model and its in-app download.

Path: tests/test_sound_verification.py

Runs in a temporary repository with its own settings, database, data folder and
models folder, so nothing here can reach a real installation's data, database,
reports or models. Uses a scripted model in place of Perch 2.0, so no model file
or download is needed. Proves:

  - Clips are read, resampled to 32 kHz and cut into five-second windows, and each
    window's top class is reported with its softmax probability.
  - A labels file that does not match the model's output is refused.
  - A pure sound event is checked by the second model when one is configured: it
    is verified when the field species is heard at or above the floor, recorded
    as differing when it is not heard, named by the model's version, counted in
    windows, and worded as a two-model check. With no clip it falls back to the
    confidence gate. The station's own record is never changed.
  - A download is checked against its recorded size and checksum before it is
    used, a file already in place with a matching checksum is used without
    downloading, and the model's slot is then filled in through the save path.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from training_fixture import World  # noqa: E402
from audtheia.analysis import taxa, verdicts  # noqa: E402
from audtheia.analysis.verify import VerifyEngine  # noqa: E402
from audtheia.app.model_downloads import DownloadJobs, downloadable  # noqa: E402
from audtheia.app.orchestrator import NullInterpreter, NullVerifier  # noqa: E402
from audtheia.inference import perch_onnx as perch  # noqa: E402

_checks = 0


def check(label: str, condition: bool) -> None:
    global _checks
    if not condition:
        raise AssertionError(label)
    _checks += 1
    print(f"  ok  {label}")


class Out:
    def __init__(self, name, shape):
        self.name, self.shape = name, shape


class ScriptedSession:
    """Returns one scripted logit row per window, in order."""

    def __init__(self, rows, width):
        self.rows, self.width, self.batches = rows, width, []

    def get_inputs(self):
        return [Out("inputs", ["batch", perch.WINDOW_SAMPLES])]

    def get_outputs(self):
        return [Out("embedding", ["batch", 1536]), Out("label", ["batch", self.width])]

    def run(self, names, feeds):
        batch = feeds["inputs"]
        self.batches.append(batch.shape)
        return [np.array([self.rows[i % len(self.rows)] for i in range(len(batch))], dtype=np.float32)]


class FakeSoundModel:
    configured = True

    def __init__(self, windows, version="Perch test"):
        self.windows, self.version = windows, version

    def verify_clip(self, path):
        return [{"scientific_name": n, "confidence": c} for n, c in self.windows] if Path(path).is_file() else []


def logits(width, top, value=8.0):
    row = np.zeros(width, dtype=np.float32)
    row[top] = value
    return row


def test_adapter(root: Path) -> None:
    print("\n[1] The adapter reads, resamples, windows and scores a clip")
    rate = 48000
    t = np.arange(int(rate * 12.0)) / rate
    tone = 0.5 * np.sin(2 * np.pi * 1000.0 * t)
    out = perch.resample(tone.astype(np.float32), rate, perch.SAMPLE_RATE)
    check("resampling keeps the duration", abs(len(out) - 12.0 * perch.SAMPLE_RATE) <= 1)
    spectrum = np.abs(np.fft.rfft(out[1000:33000]))
    peak = np.fft.rfftfreq(32000, 1.0 / perch.SAMPLE_RATE)[spectrum.argmax()]
    check("resampling keeps a 1 kHz tone at 1 kHz", abs(peak - 1000.0) <= 2.0)
    check("a clip is cut into five-second windows, the last padded",
          perch.windows(out).shape == (3, perch.WINDOW_SAMPLES))

    clip = root / "clip.wav"
    with wave.open(str(clip), "wb") as h:
        h.setnchannels(2)
        h.setsampwidth(2)
        h.setframerate(rate)
        pcm = (np.stack([tone, tone], axis=1) * 32767).astype("<i2")
        h.writeframes(pcm.tobytes())
    labels = ["Alpha one", "Poecile carolinensis", "Gamma three"]
    session = ScriptedSession([logits(3, 1), logits(3, 2, 2.0)], 3)
    verifier = perch.PerchVerifier(session, labels, version="Perch test")
    scored = verifier.verify_clip(clip)
    check("each window reports its top class", [w["scientific_name"] for w in scored]
          == ["Poecile carolinensis", "Gamma three", "Poecile carolinensis"])
    expected = float(np.exp(8.0) / (np.exp(8.0) + 2.0))
    check("a window's confidence is its softmax probability", abs(scored[0]["confidence"] - expected) < 1e-6)
    check("the model receives 32 kHz five-second windows", session.batches == [(3, perch.WINDOW_SAMPLES)])
    check("a missing clip scores nothing", verifier.verify_clip(root / "absent.wav") == [])

    labels_file = root / "labels.csv"
    labels_file.write_text("inat2024_fsd50k\r\nAlpha one\r\nPoecile carolinensis\r\nGamma three\r\n", encoding="utf-8")
    check("labels are read in order, the header skipped", perch.load_labels(labels_file) == labels)


def test_build(w: World) -> None:
    print("\n[2] The configured model is checked before it is used")
    check("with no model configured there is no sound verifier", perch.build_acoustic_verifier(w.settings) is None)
    models = w.repo / "models" / "acoustic" / "perch2.0"
    models.mkdir(parents=True, exist_ok=True)
    (models / "perch_v2.onnx").write_bytes(b"stand-in")
    (models / "labels.csv").write_text("header\nA a\nB b\n", encoding="utf-8")
    w.settings.raw["desktop_models"]["acoustic_perch"] = {
        "path": "models/acoustic/perch2.0/perch_v2.onnx", "labels_path": "models/acoustic/perch2.0/labels.csv",
        "version": "Perch test", "citation": None}
    check("a configured model on disk is reported as configured", perch.acoustic_verifier_configured(w.settings))
    try:
        perch.build_acoustic_verifier(w.settings, session=ScriptedSession([logits(3, 0)], 3))
        refused = False
    except perch.AcousticVerifierError as exc:
        refused = "do not belong together" in str(exc)
    check("a labels file whose length differs from the model's output is refused", refused)
    built = perch.build_acoustic_verifier(w.settings, session=ScriptedSession([logits(2, 0)], 2))
    check("a matching model and labels file build a verifier named by its version",
          built is not None and built.version == "Perch test")
    w.settings.raw["desktop_models"]["acoustic_perch"] = {"path": None, "labels_path": None, "version": None,
                                                          "citation": None}


def test_engine(w: World) -> None:
    print("\n[3] A sound event is checked by the second model")
    ident = taxa.identity_function(w.settings)
    heard = w.audio_event([("Poecile carolinensis_Carolina Chickadee", None)], first_seen="2026-03-01T10:00:00Z",
                          verified=None)
    before = dict(w.db.get_observation(heard["id"]))
    children_before = [dict(c) for c in w.db.list_child_detections(heard["id"])]
    model = FakeSoundModel([("Regulus satrapa", 0.05), ("Poecile carolinensis", 0.86), ("Poecile carolinensis", 0.92)])
    engine = VerifyEngine(settings=w.settings, db=w.db, verifier=NullVerifier(), interpreter=NullInterpreter(),
                          identify=ident, acoustic_verifier=model)
    engine.process(heard["id"])
    v = w.db.get_observation_verification(heard["id"])
    check("the field species heard above the floor is verified", v["verified"] == 1 and v["rfdetr_agrees_with_field"] == 1)
    check("the verdict names the second model and counts windows",
          v["rfdetr_version"] == "Perch test" and v["frames_scored"] == 3 and v["frames_in_agreement"] == 2)
    check("its confidence is the strongest window for the field species", abs(v["rfdetr_confidence"] - 0.92) < 1e-9)
    obs = w.db.get_observation(heard["id"])
    c = verdicts.classify(obs, v)
    check("it is classified as a two-model verdict", c == {"gate": verdicts.GATE_TWO_MODEL, "state": "verified",
                                                           "reason": None})
    text = verdicts.describe(obs, v, children=w.db.list_child_detections(heard["id"]))
    check("it is worded in windows, by the model's name",
          text.startswith("verified by the two-model check: Perch test recognized Poecile carolinensis in 2 of 3 "
                          "windows, strongest window 0.92"))
    check("the station's observation row is unchanged", dict(w.db.get_observation(heard["id"])) == before)
    check("the station's detections are unchanged",
          [dict(x) for x in w.db.list_child_detections(heard["id"])] == children_before)

    other = w.audio_event([("Poecile carolinensis_Carolina Chickadee", None)], first_seen="2026-03-02T10:00:00Z",
                          verified=None)
    VerifyEngine(settings=w.settings, db=w.db, verifier=NullVerifier(), interpreter=NullInterpreter(),
                 identify=ident, acoustic_verifier=FakeSoundModel([("Cardinalis cardinalis", 0.7)])).process(other["id"])
    v2 = w.db.get_observation_verification(other["id"])
    check("a field species the second model does not hear is not verified and differs",
          v2["verified"] == 0 and v2["rfdetr_agrees_with_field"] == 0 and v2["rfdetr_scientific_name"] == "Cardinalis cardinalis")

    weak = w.audio_event([("Poecile carolinensis_Carolina Chickadee", None)], first_seen="2026-03-03T10:00:00Z",
                         verified=None)
    VerifyEngine(settings=w.settings, db=w.db, verifier=NullVerifier(), interpreter=NullInterpreter(),
                 identify=ident, acoustic_verifier=FakeSoundModel([("Poecile carolinensis", 0.3)])).process(weak["id"])
    check("a field species heard below the floor is not verified",
          w.db.get_observation_verification(weak["id"])["verified"] == 0)

    clipless = w.audio_event([("Poecile carolinensis_Carolina Chickadee", None)], first_seen="2026-03-04T10:00:00Z",
                             verified=None)
    Path(clipless["path"]).unlink()
    VerifyEngine(settings=w.settings, db=w.db, verifier=NullVerifier(), interpreter=NullInterpreter(),
                 identify=ident, acoustic_verifier=model).process(clipless["id"])
    v3 = w.db.get_observation_verification(clipless["id"])
    check("with no clip to score, the event falls back to the confidence gate",
          v3["rfdetr_version"] is None
          and verdicts.classify(w.db.get_observation(clipless["id"]), v3)["gate"] == verdicts.GATE_ACOUSTIC)

    pairs = [(w.db.get_observation(e["id"]), w.db.get_observation_verification(e["id"]))
             for e in (heard, other, weak, clipless)]
    split = verdicts.split_counts(pairs)
    check("sound events are counted by the check that produced each verdict",
          split["audio"]["verified"] == 1 and split["audio"]["not_verified"] == 2 and split["audio"]["total"] == 4)
    check("the summary states the two checks separately",
          "1 of 3 audio events by the two-model check" in verdicts.summary_sentence(split)
          and "of 1 audio event by the acoustic confidence gate" in verdicts.summary_sentence(split))


def test_downloads(w: World) -> None:
    print("\n[4] A download is checked before it is used")
    source = w.root / "source.onnx"
    payload = b"a stand-in model file" * 1000
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    manifest = {"models": {"acoustic_perch": {
        "role": "desktop", "tier": "full", "kind": "file", "destination": "models/acoustic/perch2.0/perch_v2.onnx",
        "url": source.as_uri(), "sha256": digest, "size_bytes": len(payload), "license": "Apache-2.0",
        "citation": "test citation", "slot": "acoustic_perch", "labels_path": "audtheia/resources/perch/labels.csv",
        "version": "Perch 2.0", "in_app_download": True}}}
    (w.repo / "config" / "model_sources.json").write_text(json.dumps(manifest), encoding="utf-8")
    check("the manifest's in-app entries are offered", list(downloadable(w.repo)) == ["acoustic_perch"])

    saved = []
    jobs = DownloadJobs(w.settings, lambda s: saved.append(dict(s.raw["desktop_models"]["acoustic_perch"])))
    target = w.repo / "models" / "acoustic" / "perch2.0" / "perch_v2.onnx"
    if target.exists():
        target.unlink()
    job = jobs.start("acoustic_perch", wait=True)
    check("a download with the recorded checksum completes", job["state"] == "done" and target.read_bytes() == payload)
    check("the slot is filled in and saved",
          saved and saved[-1]["path"] == "models/acoustic/perch2.0/perch_v2.onnx"
          and saved[-1]["labels_path"] == "audtheia/resources/perch/labels.csv" and saved[-1]["version"] == "Perch 2.0")
    check("no partial file is left behind", not target.with_name(target.name + ".part").exists())

    calls = []

    def opener(request):
        calls.append(request)
        raise AssertionError("nothing should be downloaded")

    again = DownloadJobs(w.settings, lambda s: None, opener=opener).start("acoustic_perch", wait=True)
    check("a file already in place with a matching checksum is used without downloading",
          again["state"] == "done" and not calls and "nothing was downloaded" in again["message"])

    target.unlink()
    manifest["models"]["acoustic_perch"]["sha256"] = "0" * 64
    (w.repo / "config" / "model_sources.json").write_text(json.dumps(manifest), encoding="utf-8")
    bad = DownloadJobs(w.settings, lambda s: None).start("acoustic_perch", wait=True)
    check("a download whose checksum does not match is refused and not kept",
          bad["state"] == "error" and "checksum" in bad["message"] and not target.exists())

    manifest["models"]["acoustic_perch"]["destination"] = "../outside.onnx"
    (w.repo / "config" / "model_sources.json").write_text(json.dumps(manifest), encoding="utf-8")
    outside = DownloadJobs(w.settings, lambda s: None).start("acoustic_perch", wait=True)
    check("a destination outside the models folder is refused", outside["state"] == "error"
          and not (w.repo.parent / "outside.onnx").exists())


def test_status(w: World) -> None:
    print("\n[5] Re-verify offers only sound events a second model can listen to")
    from fastapi.testclient import TestClient
    from audtheia.app import server as srv

    models = w.repo / "models" / "acoustic" / "perch2.0"
    models.mkdir(parents=True, exist_ok=True)
    (models / "perch_v2.onnx").write_bytes(b"stand-in")
    w.settings.raw["desktop_models"]["acoustic_perch"] = {
        "path": "models/acoustic/perch2.0/perch_v2.onnx", "labels_path": "audtheia/resources/perch/labels.csv",
        "version": "Perch test", "citation": None}
    (w.repo / "audtheia" / "resources" / "perch").mkdir(parents=True, exist_ok=True)
    (w.repo / "audtheia" / "resources" / "perch" / "labels.csv").write_text("h\nA a\n", encoding="utf-8")
    w.audio_event([("Poecile carolinensis_Carolina Chickadee", None)], first_seen="2026-04-01T10:00:00Z")
    clipless = w.audio_event([("Poecile carolinensis_Carolina Chickadee", None)], first_seen="2026-04-02T10:00:00Z")
    with w.db.connect() as conn:
        conn.execute("UPDATE observations SET audio_clip_path = NULL WHERE id = ?", (clipless["id"],))
    st = TestClient(srv.create_app(w.settings, w.db)).get("/api/verify/status").json()
    check("the status says the sound model is set up and names it",
          st["acoustic_verifier_configured"] is True and st["acoustic_verifier_version"] == "Perch test")
    check("a gate-cleared sound event with a clip is offered, one with no clip is counted apart",
          st["audio"]["gate_only"] == 1 and st["audio"]["no_clip"] == 1)


def run() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        test_adapter(root)
        test_build(World(root / "b"))
        test_engine(World(root / "e"))
        test_downloads(World(root / "d"))
        test_status(World(root / "s"))
    print(f"\nAll {_checks} sound verification checks passed.")


if __name__ == "__main__":
    try:
        run()
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
