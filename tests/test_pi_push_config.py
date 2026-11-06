"""The Pi config-only push sends configuration, not code or models.

Path: tests/test_pi_push_config.py

A target-species edit (or any config change) reaches a Pi field station only
when it is pushed down. The full connect flow re-sends the code, the models, and
re-runs the Pi-side setup; the fast "push changes to the Pi" path should send
only the updated configuration. This proves that distinction against the
recording SSH runner, with no Pi and no network: in settings-only mode the
station's settings.json is sent and nothing else is, while the full mode still
sends the code archive.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from audtheia.config import load_settings  # noqa: E402
from scripts.bootstrap_setup_pi import provision, LoggingRunner  # noqa: E402


CHECKS = {"passed": 0, "failed": 0}


def check(label: str, condition: bool):
    if condition:
        CHECKS["passed"] += 1
        print(f"  PASS  {label}")
    else:
        CHECKS["failed"] += 1
        print(f"  FAIL  {label}")


def _settings(tmp: Path):
    base = json.loads((REPO / "config" / "settings.json").read_text(encoding="utf-8"))
    base["node"]["role"] = "desktop"
    path = tmp / "settings.push.json"
    path.write_text(json.dumps(base), encoding="utf-8")
    return load_settings(path)


def _puts(runner):
    return [c[2] for c in runner.calls if c[0] == "put"]


def _runs(runner):
    return [c[1] for c in runner.calls if c[0] == "run"]


def test_settings_only_sends_config_only():
    print("\n[1] Settings-only push sends the configuration and nothing else")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        settings = _settings(tmp)
        sid = settings.stations()[0]["station_id"]
        runner = LoggingRunner("pi.local", "pi", 22)
        provision(
            settings, sid, runner=runner, work_dir=tmp,
            make_key=False, generate_key=False,
            preauthorized_key=tmp / "fake_key", settings_only=True,
        )
        remotes = _puts(runner)
        check("the station settings.json is sent", any(r.endswith("settings.json") for r in remotes))
        check("the code archive is NOT sent", not any("audtheia-code.tar.gz" in r for r in remotes))
        check("no model file is sent", not any(r.endswith(".hef") or r.endswith(".onnx") for r in remotes))
        check("the Pi-side setup is NOT re-run", not any("setup-pi.sh" in cmd for cmd in _runs(runner)))


def test_full_push_sends_code():
    print("\n[2] A full connect still sends the code archive (contrast)")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        settings = _settings(tmp)
        sid = settings.stations()[0]["station_id"]
        runner = LoggingRunner("pi.local", "pi", 22)
        provision(
            settings, sid, runner=runner, work_dir=tmp,
            make_key=False, generate_key=False,
            preauthorized_key=tmp / "fake_key", settings_only=False,
        )
        remotes = _puts(runner)
        check("the code archive IS sent in a full connect", any("audtheia-code.tar.gz" in r for r in remotes))
        check("the Pi-side setup IS run in a full connect", any("setup-pi.sh" in cmd for cmd in _runs(runner)))


def test_settings_with_models():
    print("\n[3] A model switched on the desktop reaches the Pi with its labels")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "config").mkdir()
        base = json.loads((REPO / "config" / "settings.json").read_text(encoding="utf-8"))
        base["node"]["role"] = "desktop"
        folder = root / "models" / "trained" / "Audtheia-Reef-V1"
        folder.mkdir(parents=True)
        (folder / "model.tflite").write_bytes(b"tflite")
        (folder / "labels.txt").write_text("Poecile carolinensis\n", encoding="utf-8")
        station = base["stations"][0]
        station.setdefault("models", {}).setdefault("acoustic", {}).update(
            {"path": "models/trained/Audtheia-Reef-V1/model.tflite",
             "labels_path": "models/trained/Audtheia-Reef-V1/labels.txt", "version": "Audtheia-Reef-V1"})
        path = root / "config" / "settings.push.json"
        path.write_text(json.dumps(base), encoding="utf-8")
        settings = load_settings(path)
        runner = LoggingRunner("pi.local", "pi", 22)
        provision(settings, station["station_id"], runner=runner, work_dir=root,
                  make_key=False, generate_key=False, preauthorized_key=root / "fake_key",
                  settings_only=True, with_models=True)
        remotes = _puts(runner)
        check("the model file is sent to the path its configuration names",
              any(r.endswith("models/trained/Audtheia-Reef-V1/model.tflite") for r in remotes))
        check("its labels file travels with it",
              any(r.endswith("models/trained/Audtheia-Reef-V1/labels.txt") for r in remotes))
        check("the code archive is still not sent", not any("audtheia-code.tar.gz" in r for r in remotes))
        check("the Pi-side setup is still not re-run", not any("setup-pi.sh" in cmd for cmd in _runs(runner)))
        runner = LoggingRunner("pi.local", "pi", 22)
        provision(settings, station["station_id"], runner=runner, work_dir=root,
                  make_key=False, generate_key=False, preauthorized_key=root / "fake_key", settings_only=True)
        check("without the models choice no model file is sent",
              not any(r.endswith(".tflite") or r.endswith("labels.txt") for r in _puts(runner)))


def main() -> int:
    test_settings_only_sends_config_only()
    test_full_push_sends_code()
    test_settings_with_models()
    print(f"\n==== Pi push-config: {CHECKS['passed']} passed, {CHECKS['failed']} failed ====")
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
