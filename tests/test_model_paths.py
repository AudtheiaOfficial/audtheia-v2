"""Model configuration: honest defaults, clearable paths, editable acoustic slots.

These checks exist because the configuration used to assert models that had never
existed. A path is a claim about a file, so the rules are: a new station claims
nothing, a claim can be withdrawn by clearing it to null, the key itself never
disappears (the configuration validator requires the key, only its value may be
null), and a desktop screening model that is the same file as the desktop
verification model is reported, because agreement between a model and itself is
not evidence.

Run from the repository root:  python tests/test_model_paths.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# The same shape checks the loader and the save path use, so a path this suite
# calls shareable is exactly one the rest of the system will also accept.
from audtheia.config import _contains_machine_path, _is_absolute_path_value  # noqa: E402

PASSED = 0
FAILED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ok    {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}" + (f": {detail}" if detail else ""))


def main() -> int:
    print("Model configuration: defaults, clearing, and acoustic editing")
    print("=" * 72)

    try:
        from fastapi.testclient import TestClient  # noqa: F401
    except Exception:
        print("  SKIP  the web framework is not installed")
        return 0

    from audtheia.app.server import _new_station_dict, _same_model_file
    from audtheia.config import ACOUSTIC_MODEL_SLOTS, load_settings

    # -- a new station claims no model at all ----------------------------
    station = _new_station_dict("11111111-1111-1111-1111-111111111111", "TestBay", "marine", None)
    models = station["models"]
    check("a new station has a visual_pi slot", "visual_pi" in models)
    check("a new station has a visual_desktop slot", "visual_desktop" in models)
    check("a new station claims no field screening model", models["visual_pi"]["path"] is None,
          repr(models["visual_pi"]["path"]))
    check("a new station claims no desktop screening model", models["visual_desktop"]["path"] is None,
          repr(models["visual_desktop"]["path"]))
    slots = models["acoustic"]["options"]
    check("a new station claims no acoustic model in any slot",
          all(slots[k].get("path") is None for k in ACOUSTIC_MODEL_SLOTS),
          repr({k: slots[k].get("path") for k in ACOUSTIC_MODEL_SLOTS}))
    check("every acoustic slot named in the vocabulary exists",
          all(k in slots for k in ACOUSTIC_MODEL_SLOTS))

    # A configured path must be a shareable destination: repository relative,
    # inside the models directory, and naming no one machine. Presence on disk
    # is deliberately not checked. Model weights are downloaded by setup and
    # excluded from version control, so on a fresh clone none of them exist yet
    # and a presence check would fail for every user before they had done
    # anything wrong. What must be guarded is the shape of the value: an
    # invented filename that was never produced by anything, or a path carrying
    # an account name into the published configuration.
    def configured_paths(node, found=None):
        found = [] if found is None else found
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("path", "labels_path") and isinstance(value, str):
                    found.append(value)
                else:
                    configured_paths(value, found)
        elif isinstance(node, list):
            for item in node:
                configured_paths(item, found)
        return found

    settings_raw = json.loads((REPO_ROOT / "config" / "settings.json").read_text(encoding="utf-8"))
    shipped = (configured_paths(settings_raw.get("stations", []))
               + configured_paths(settings_raw.get("desktop_models", {})))

    machine_specific = [p for p in shipped if _is_absolute_path_value(p) or _contains_machine_path(p)]
    check("no shipped model path names one person's machine",
          not machine_specific, f"machine specific: {machine_specific}")

    models_dir = settings_raw.get("paths", {}).get("models_dir", "models")
    stray = [p for p in shipped if not p.replace("\\", "/").startswith(models_dir.rstrip("/") + "/")]
    check("every shipped model path sits under the models directory",
          not stray, f"outside {models_dir}: {stray}")

    # The destination a downloader actually writes, rather than a filename that
    # reads plausibly and is produced by nothing, is what makes a path real.
    birdnet_default = "BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite"
    fetch_source = (REPO_ROOT / "scripts" / "fetch_birdnet.py").read_text(encoding="utf-8")
    named_birdnet = [p for p in shipped if p.endswith(".tflite")]
    check("a shipped BirdNET path names the file its downloader writes",
          all(Path(p).name == birdnet_default for p in named_birdnet)
          and birdnet_default in fetch_source,
          f"named: {named_birdnet}")

    # -- clearing a path to null, through the guarded write path ----------
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "config").mkdir()
        shutil.copy(REPO_ROOT / "config" / "settings.json", root / "config" / "settings.json")
        (root / "audtheia" / "storage").mkdir(parents=True)
        shutil.copy(REPO_ROOT / "audtheia" / "storage" / "schema.sql",
                    root / "audtheia" / "storage" / "schema.sql")

        settings = load_settings(str(root / "config" / "settings.json"))
        target = settings.stations()[0]
        sid = target["station_id"]

        from audtheia.app.server import _apply_setting_change, _editable_field_specs

        specs = _editable_field_specs()
        draft = json.loads(json.dumps(settings.raw))
        warnings: list = []

        _apply_setting_change(
            draft,
            {"scope": "station", "field": "visual_pi_path", "station_id": sid,
             "value": "models/visual/pi/my_screener.hef"},
            specs, warnings, root)
        applied = draft["stations"][0]["models"]["visual_pi"]["path"]
        check("a model path can be set", applied == "models/visual/pi/my_screener.hef", repr(applied))
        check("setting a path to an absent file warns rather than refusing",
              any("no file is present yet" in w for w in warnings), repr(warnings))

        warnings.clear()
        _apply_setting_change(
            draft,
            {"scope": "station", "field": "visual_pi_path", "station_id": sid, "value": ""},
            specs, warnings, root)
        entry = draft["stations"][0]["models"]["visual_pi"]
        check("an emptied path clears to null", entry["path"] is None, repr(entry["path"]))
        check("clearing a path keeps the key, which the validator requires", "path" in entry)
        check("clearing a path warns about nothing", not warnings, repr(warnings))

        # The acoustic model was unreachable from the interface before this.
        _apply_setting_change(
            draft,
            {"scope": "station", "field": "acoustic_birdnet_path", "station_id": sid,
             "value": "models/acoustic/birdnet/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite"},
            specs, warnings, root)
        heard = draft["stations"][0]["models"]["acoustic"]["options"]["birdnet"]["path"]
        check("an acoustic model path is editable",
              heard == "models/acoustic/birdnet/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite", repr(heard))

        _apply_setting_change(
            draft,
            {"scope": "station", "field": "acoustic_active", "station_id": sid, "value": "birdnet"},
            specs, warnings, root)
        check("the active acoustic slot is selectable",
              draft["stations"][0]["models"]["acoustic"]["active"] == "birdnet")

        # A Windows-style path saves in the one separator the field station reads.
        _apply_setting_change(
            draft,
            {"scope": "station", "field": "visual_desktop_path", "station_id": sid,
             "value": "models\\visual\\mine.onnx"},
            specs, warnings, root)
        stored = draft["stations"][0]["models"]["visual_desktop"]["path"]
        check("a pasted Windows path is stored with forward slashes",
              stored == "models/visual/mine.onnx", repr(stored))

        # Windows "Copy as path" wraps the path in double quotes and people
        # paste exactly that. Kept as written, the quotes become part of the
        # filename, nothing is ever found at it, and the interface reports a
        # model as missing while displaying a path that looks right on screen.
        _apply_setting_change(
            draft,
            {"scope": "station", "field": "visual_desktop_path", "station_id": sid,
             "value": '"C:\\Users\\somebody\\models\\mine.onnx"'},
            specs, warnings, root)
        quoted = draft["stations"][0]["models"]["visual_desktop"]["path"]
        check("a quoted Windows path loses its quotes",
              quoted == "C:/Users/somebody/models/mine.onnx", repr(quoted))

        _apply_setting_change(
            draft,
            {"scope": "station", "field": "visual_desktop_path", "station_id": sid,
             "value": "'models/visual/mine.onnx'"},
            specs, warnings, root)
        single = draft["stations"][0]["models"]["visual_desktop"]["path"]
        check("a single quoted path loses its quotes too",
              single == "models/visual/mine.onnx", repr(single))

        # Screening and verification by identical weights. The verifier is set
        # here rather than read from the shipped configuration, which names no
        # model at all: nothing ships as set, so a suite that needs a configured
        # model configures one.
        verifier = "models/visual/shared_weights.onnx"
        _apply_setting_change(
            draft,
            {"scope": "global", "field": "visual_rfdetr_path", "value": verifier},
            specs, warnings, root)
        warnings.clear()
        _apply_setting_change(
            draft,
            {"scope": "station", "field": "visual_desktop_path", "station_id": sid, "value": verifier},
            specs, warnings, root)
        check("screening with the verification model's own file is reported",
              any("not" in w and "independent" in w for w in warnings), repr(warnings))

        warnings.clear()
        _apply_setting_change(
            draft,
            {"scope": "station", "field": "visual_desktop_path", "station_id": sid,
             "value": "models/visual/a_different_model.onnx"},
            specs, warnings, root)
        check("a distinct screening model raises no independence warning",
              not any("independent" in w for w in warnings), repr(warnings))

        # A cleared configuration must still load.
        (root / "config" / "settings.json").write_text(
            json.dumps(draft, indent=2), encoding="utf-8")
        try:
            load_settings(str(root / "config" / "settings.json"))
            check("a configuration with cleared model paths still validates", True)
        except Exception as exc:  # noqa: BLE001
            check("a configuration with cleared model paths still validates", False, str(exc))

    check("the same file by two spellings is recognised as one model",
          _same_model_file("models/visual/porifera_rfdetr.onnx",
                           str(REPO_ROOT / "models" / "visual" / "porifera_rfdetr.onnx"),
                           REPO_ROOT))
    check("two different files are not treated as one model",
          not _same_model_file("models/visual/a.onnx", "models/visual/b.onnx", REPO_ROOT))
    check("a null path is never treated as matching anything",
          not _same_model_file(None, "models/visual/a.onnx", REPO_ROOT))

    print("=" * 72)
    print(f"RESULT: {PASSED} passed, {FAILED} failed")
    print("=" * 72)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
