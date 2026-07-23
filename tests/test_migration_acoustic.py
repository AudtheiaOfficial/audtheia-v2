"""One acoustic model per station: the migration keeps the model that was running.

A station sits in one place and listens with one model. The configuration used to
hold three named slots and a selector naming which was live, and only the selected
slot was ever loaded. Collapsing that to one model is a configuration contract
change, and a contract change has one dangerous failure mode: the paths that
actually run do not live in the committed file.

They live in config/settings.local.json, addressed by pointer, in the form
stations[station_id=...].models.acoustic.options.birdnet.path . After the
restructure those pointers resolve to nothing. The loader reports them stale and
carries on, so a working installation loses the model it was listening with and
is told nothing. That is not a hypothetical: it is exactly how a configured
capture source disappeared once already.

What is checked here:

  - The live slot's values survive; the slots that were never loaded do not.
  - A local pointer for the live slot is rewritten, and the rewritten pointer
    still resolves against the migrated configuration and delivers its value.
    This is the whole point of the migration, so it is proven by applying the
    real override merge rather than by comparing strings.
  - A local pointer for a slot that was not live is dropped rather than
    promoted, because promoting it would silently change which model runs.
  - Pointers that have nothing to do with the acoustic block are untouched.
  - The migration is idempotent, so running it twice is harmless.
  - Both files are backed up before either is written.
  - A station whose live slot held no path reads as having no model, rather than
    appearing configured.
  - Ambiguity is refused, not guessed.

Nothing here reads or writes the real configuration, the real local override
file, the real database, or any data directory. Every case runs inside a
temporary directory that is removed afterwards.

Run from the repository root:  python tests/test_migration_acoustic.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audtheia.config import apply_local_overrides, pointer_value  # noqa: E402

import migrate_acoustic_single_model as migration  # noqa: E402

PASSED = 0
FAILED = 0

REEF_ID = "00000000-0000-0000-0000-0000000000aa"
FOREST_ID = "00000000-0000-0000-0000-0000000000bb"

# Paths that look like the machine-specific values this file exists to protect.
# Neither names a real file; nothing here touches the filesystem outside a
# temporary directory.
LIVE_MODEL = r"C:\Users\someone\Models\acoustic\example_model.tflite"
LIVE_LABELS = r"C:\Users\someone\Models\acoustic\example_labels.txt"
UNUSED_MODEL = r"C:\Users\someone\Models\acoustic\never_loaded.tflite"
AUDIO_SOURCE = r"file:C:\Users\someone\Music\example.mp3"


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ok    {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}" + (f": {detail}" if detail else ""))


def _legacy_station(station_id: str, name: str, active: str, options: dict) -> dict:
    return {
        "station_id": station_id,
        "station_name": name,
        "models": {
            "visual_pi": {"path": None},
            "acoustic": {"active": active, "options": options},
        },
        "capture": {"source": {"video": None, "audio": None}},
    }


def _settings() -> dict:
    """A configuration in the retired shape, with two stations.

    The reef station's live slot is the one holding nothing, which is the state
    the open questions recorded on the real file: a station that looks configured
    and is not. The forest station's live slot holds a model, and a second slot
    holds a different one that was never loaded.
    """
    return {
        "settings_schema_version": "1",
        "stations": [
            _legacy_station(
                REEF_ID,
                "ExampleReef",
                "marine",
                {
                    "birdnet": {"path": None, "labels_path": None, "version": None, "citation": None},
                    "marine": {"path": None, "labels_path": None, "version": None, "citation": None},
                    "custom": {"path": None, "labels_path": None, "version": None, "citation": None},
                },
            ),
            _legacy_station(
                FOREST_ID,
                "ExampleForest",
                "birdnet",
                {
                    "birdnet": {
                        "path": "models/acoustic/committed.tflite",
                        "labels_path": None,
                        "version": "2.4",
                        "citation": "an example citation",
                    },
                    "marine": {"path": UNUSED_MODEL, "labels_path": None, "version": None, "citation": None},
                    "custom": {"path": None, "labels_path": None, "version": None, "citation": None},
                },
            ),
        ],
    }


def _overrides() -> dict:
    """A local file holding the paths that actually run on this machine."""
    return {
        f"stations[station_id={FOREST_ID}].models.acoustic.options.birdnet.path": LIVE_MODEL,
        f"stations[station_id={FOREST_ID}].models.acoustic.options.birdnet.labels_path": LIVE_LABELS,
        f"stations[station_id={FOREST_ID}].models.acoustic.options.marine.path": UNUSED_MODEL,
        f"stations[station_id={FOREST_ID}].capture.source.audio": AUDIO_SOURCE,
    }


def _write_pair(root: Path, settings: dict, overrides: dict) -> tuple[Path, Path]:
    (root / "config").mkdir(parents=True, exist_ok=True)
    settings_path = root / "config" / "settings.json"
    local_path = root / "config" / "settings.local.json"
    settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8", newline="\n")
    local_path.write_text(
        json.dumps({"overrides": overrides}, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return settings_path, local_path


def _run(settings_path: Path, local_path: Path, *, check_only: bool = False) -> int:
    argv = ["--settings", str(settings_path), "--local", str(local_path)]
    if check_only:
        argv.append("--check")
    return migration.main(argv)


def _station(raw: dict, station_id: str) -> dict:
    for station in raw["stations"]:
        if station["station_id"] == station_id:
            return station
    raise AssertionError(f"no station {station_id}")


def main() -> int:
    print("Acoustic migration: one model per station, and the running model survives")
    print("=" * 72)

    with tempfile.TemporaryDirectory() as tmp_name:
        root = Path(tmp_name)

        # -- the committed configuration ---------------------------------
        print("\nThe committed configuration")
        settings_path, local_path = _write_pair(root / "main", _settings(), _overrides())
        code = _run(settings_path, local_path)
        check("the migration reports success", code == 0, f"exit code {code}")

        migrated = json.loads(settings_path.read_text(encoding="utf-8"))
        forest = _station(migrated, FOREST_ID)["models"]["acoustic"]
        reef = _station(migrated, REEF_ID)["models"]["acoustic"]

        check("the selector is gone", "active" not in forest)
        check("the slot map is gone", "options" not in forest)
        check(
            "the live slot's path survives",
            forest.get("path") == "models/acoustic/committed.tflite",
            repr(forest.get("path")),
        )
        check("the live slot's version survives", forest.get("version") == "2.4")
        check("the live slot's citation survives", forest.get("citation") == "an example citation")
        check(
            "a model that was never loaded is not carried forward",
            UNUSED_MODEL not in json.dumps(migrated),
        )
        check(
            "the flat block carries every contracted key",
            all(key in forest for key in migration.FLAT_KEYS),
            f"missing {[k for k in migration.FLAT_KEYS if k not in forest]}",
        )
        check(
            "a station whose live slot held nothing reads as having no model",
            reef.get("path") is None and "active" not in reef,
        )

        # -- the local override file, and the trap it sets ---------------
        print("\nThe machine paths that actually run")
        local = json.loads(local_path.read_text(encoding="utf-8"))["overrides"]
        flat_path_pointer = f"stations[station_id={FOREST_ID}].models.acoustic.path"
        flat_labels_pointer = f"stations[station_id={FOREST_ID}].models.acoustic.labels_path"

        check("the live model pointer is rewritten flat", flat_path_pointer in local)
        check("the live labels pointer is rewritten flat", flat_labels_pointer in local)
        check(
            "the rewritten pointer still carries the same value",
            local.get(flat_path_pointer) == LIVE_MODEL,
            repr(local.get(flat_path_pointer)),
        )
        check(
            "a pointer for a slot that was not live is dropped",
            not any(".options." in p for p in local),
            repr([p for p in local if ".options." in p]),
        )
        check(
            "a pointer unrelated to the acoustic block is untouched",
            local.get(f"stations[station_id={FOREST_ID}].capture.source.audio") == AUDIO_SOURCE,
        )

        # The guarantee, proven rather than asserted: merge the rewritten local
        # file into the migrated configuration exactly as the loader does, and
        # confirm the machine path lands where the pipeline will read it. A
        # pointer that no longer resolves would report stale here, which is the
        # silent failure this whole migration exists to prevent.
        merged = json.loads(json.dumps(migrated))
        stale = apply_local_overrides(merged, local)
        check("no rewritten pointer is stale", stale == [], f"stale: {stale}")
        check(
            "the machine path reaches the flat acoustic block",
            pointer_value(merged, flat_path_pointer) == LIVE_MODEL,
            repr(pointer_value(merged, flat_path_pointer)),
        )
        check(
            "the merged station reads its model from the flat key",
            _station(merged, FOREST_ID)["models"]["acoustic"]["path"] == LIVE_MODEL,
        )

        # -- backups and line endings ------------------------------------
        print("\nBackups and file shape")
        backups = sorted(p.name for p in (root / "main" / "config").glob("*.backup-*"))
        check("the committed file is backed up", any("settings.json.backup-" in n for n in backups), str(backups))
        check("the local file is backed up", any("settings.local.json.backup-" in n for n in backups), str(backups))

        backup_name = next(n for n in backups if "settings.local.json.backup-" in n)
        backup_text = (root / "main" / "config" / backup_name).read_text(encoding="utf-8")
        check(
            "the dropped pointer is preserved in the backup",
            ".options.marine.path" in backup_text,
        )
        check(
            "the migrated files carry one line ending",
            b"\r\n" not in settings_path.read_bytes() and b"\r\n" not in local_path.read_bytes(),
        )

        # -- idempotence --------------------------------------------------
        print("\nRunning it twice")
        before_settings = settings_path.read_bytes()
        before_local = local_path.read_bytes()
        code = _run(settings_path, local_path)
        check("a second run reports success", code == 0, f"exit code {code}")
        check("a second run changes the configuration not at all", settings_path.read_bytes() == before_settings)
        check("a second run changes the machine paths not at all", local_path.read_bytes() == before_local)
        check(
            "a second run writes no further backup",
            len(list((root / "main" / "config").glob("*.backup-*"))) == 2,
        )

        # -- check mode writes nothing ------------------------------------
        print("\nCheck mode")
        settings_path2, local_path2 = _write_pair(root / "checkonly", _settings(), _overrides())
        before = settings_path2.read_bytes()
        code = _run(settings_path2, local_path2, check_only=True)
        check("check mode reports success", code == 0, f"exit code {code}")
        check("check mode writes nothing", settings_path2.read_bytes() == before)
        check(
            "check mode writes no backup",
            not list((root / "checkonly" / "config").glob("*.backup-*")),
        )

        # -- ambiguity is refused, not guessed -----------------------------
        print("\nAmbiguity")
        flat_settings, _ = migration.migrate_settings(_settings())
        ambiguous = {
            f"stations[station_id={FOREST_ID}].models.acoustic.options.birdnet.path": LIVE_MODEL,
            f"stations[station_id={FOREST_ID}].models.acoustic.options.marine.path": UNUSED_MODEL,
        }
        new_overrides, rewritten, dropped = migration.migrate_overrides(ambiguous, {})
        check(
            "with the selector gone and two slots named, nothing is rewritten",
            rewritten == [],
            str(rewritten),
        )
        check("both pointers are kept for a person to decide", len(new_overrides) == 2)
        check("the ambiguity is reported", len(dropped) == 2, str(dropped))

        # One named slot is unambiguous, so it is carried forward even without
        # the selector, which is the case of a person who edited settings.json
        # by hand before running this.
        single = {f"stations[station_id={FOREST_ID}].models.acoustic.options.birdnet.path": LIVE_MODEL}
        new_overrides, rewritten, dropped = migration.migrate_overrides(single, {})
        check(
            "one named slot is unambiguous and is rewritten",
            list(new_overrides) == [f"stations[station_id={FOREST_ID}].models.acoustic.path"],
            str(list(new_overrides)),
        )
        check("nothing is dropped in the unambiguous case", dropped == [], str(dropped))

        # -- an already-flat configuration ---------------------------------
        print("\nAn installation that is already flat")
        settings_path3, local_path3 = _write_pair(root / "flat", flat_settings, {})
        before = settings_path3.read_bytes()
        code = _run(settings_path3, local_path3)
        check("an already-flat configuration reports success", code == 0, f"exit code {code}")
        check("an already-flat configuration is not rewritten", settings_path3.read_bytes() == before)

        # -- no local file at all -------------------------------------------
        print("\nA fresh clone with no machine paths")
        (root / "fresh" / "config").mkdir(parents=True, exist_ok=True)
        settings_path4 = root / "fresh" / "config" / "settings.json"
        settings_path4.write_text(json.dumps(_settings(), indent=2) + "\n", encoding="utf-8", newline="\n")
        code = migration.main(
            ["--settings", str(settings_path4), "--local", str(root / "fresh" / "config" / "settings.local.json")]
        )
        check("a configuration with no local file migrates cleanly", code == 0, f"exit code {code}")
        fresh = json.loads(settings_path4.read_text(encoding="utf-8"))
        check(
            "the fresh configuration is flat",
            "options" not in _station(fresh, FOREST_ID)["models"]["acoustic"],
        )

    print("\n" + "=" * 72)
    print(f"{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
