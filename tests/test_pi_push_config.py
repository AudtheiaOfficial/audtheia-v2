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


def main() -> int:
    test_settings_only_sends_config_only()
    test_full_push_sends_code()
    print(f"\n==== Pi push-config: {CHECKS['passed']} passed, {CHECKS['failed']} failed ====")
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
