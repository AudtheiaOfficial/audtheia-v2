"""Verification for the Train tab foundations: tables, migration, settings and storage.

Path: tests/test_training_contracts.py

Proves four things without any training library or hardware. The four new
tables exist on a fresh database with the constraints that keep model output and
expert action honest. The explicit migration adds them to an older database,
changes nothing else, writes a backup first, is safe to run twice, and ends with
definitions identical to a fresh install's. The training block in the settings
matches the documented defaults, loads from a file that lacks it, and refuses
nonsense with a plain message. The storage methods issue model versions without
collision, refuse a name that differs only by letter case, keep a finished run
immutable, and read the current state of a corrected box.

Uses only the standard library plus the project itself.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "audtheia" / "storage"))

import database as db  # noqa: E402
from database import Database, Observation, Station, TrainingRun, new_id, utc_now_iso  # noqa: E402
from audtheia.config import ConfigError, DEFAULT_TRAINING, load_settings  # noqa: E402

SCHEMA = REPO / "audtheia" / "storage" / "schema.sql"
SCRIPT = REPO / "scripts" / "migrate_add_training.py"
TABLES = ("frame_box_correction", "training_runs", "trained_models", "model_activation_log")
CHECKS = {"passed": 0, "failed": 0}


def check(label: str, condition: bool) -> None:
    CHECKS["passed" if condition else "failed"] += 1
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")


def raises(fn, exc=Exception) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


def fresh_db(directory: Path, name: str = "audtheia.db") -> Database:
    d = Database(str(directory / name))
    d.initialize_schema(SCHEMA)
    return d


def seed_observation(d: Database) -> str:
    sid = new_id()
    d.upsert_station(Station(id=sid, station_name="TrainTestReef", environment_type="marine",
                             created_at=utc_now_iso()))
    oid = new_id()
    now = utc_now_iso()
    d.insert_observation(Observation(
        id=oid, event_name=f"TrainTestReef_{oid[:8]}", station_id=sid, trigger_source="vision",
        first_seen=now, last_seen=now, duration=2.0, data_source="model", created_at=now))
    return oid


def normalised(conn: sqlite3.Connection, kind: str, names: tuple) -> dict:
    out = {}
    for name in names:
        row = conn.execute("SELECT sql FROM sqlite_master WHERE type = ? AND name = ?", (kind, name)).fetchone()
        out[name] = re.sub(r"\s+", " ", row[0]).strip() if row else None
    return out


def run_migration(db_path: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), "--db", str(db_path), *extra],
                          capture_output=True, text=True)


def test_tables(work: Path) -> None:
    print("\n[1] New tables on a fresh database")
    d = fresh_db(work)
    oid = seed_observation(d)
    with d.connect() as conn:
        present = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    check("all four tables exist", set(TABLES) <= present)

    good = d.add_frame_box_correction(oid, 0, action="adjust", corrector="tester",
                                      x1=1, y1=1, x2=5, y2=5, frame_width=10, frame_height=10)
    check("an adjusted box is stored with its provenance", good["data_source"] == "human_expert")
    check("a box with inverted corners is refused",
          raises(lambda: d.add_frame_box_correction(oid, 0, action="adjust", corrector="t",
                                                    x1=5, y1=1, x2=1, y2=5), sqlite3.IntegrityError))
    check("an added box without a taxon is refused",
          raises(lambda: d.add_frame_box_correction(oid, 0, action="add", corrector="t",
                                                    x1=1, y1=1, x2=2, y2=2), sqlite3.IntegrityError))
    check("an unknown action is refused",
          raises(lambda: d.add_frame_box_correction(oid, 0, action="invent", corrector="t"),
                 sqlite3.IntegrityError))

    check("an activation for the station slot needs a station",
          raises(lambda: d.add_model_activation(new_id(), slot="station_acoustic", action="activate",
                                                previous_json="{}", new_json="{}", activated_by="t"),
                 sqlite3.IntegrityError))


def test_migration(work: Path) -> None:
    print("\n[2] Explicit migration on an older database")
    d = fresh_db(work, "old.db")
    oid = seed_observation(d)
    old_path = work / "old.db"
    with d.connect() as conn:
        for name in ("model_activation_log", "trained_models", "training_runs", "frame_box_correction"):
            conn.execute(f"DROP TABLE {name}")
    with sqlite3.connect(str(old_path)) as c:
        before = c.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    probe = run_migration(old_path, "--check")
    check("check mode reports the migration is needed", probe.returncode == 0 and "needed" in probe.stdout)
    with sqlite3.connect(str(old_path)) as c:
        gone = c.execute("SELECT COUNT(*) FROM sqlite_master WHERE name = 'training_runs'").fetchone()[0]
    check("check mode changed nothing", gone == 0 and not list(work.glob("old.db.backup-*")))

    result = run_migration(old_path)
    check("migration succeeds", result.returncode == 0)
    check("a backup was written first", len(list(work.glob("old.db.backup-*"))) == 1)
    with sqlite3.connect(str(old_path)) as c:
        after = c.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        check("observations are untouched", after == before == 1)
        migrated_tables = normalised(c, "table", TABLES)
        migrated_indexes = normalised(c, "index", ("idx_frame_box_correction_observation",
                                                     "idx_training_runs_status", "idx_model_activation_slot"))
    fresh = fresh_db(work, "fresh.db")
    with fresh.connect() as c:
        check("migrated tables match a fresh install", migrated_tables == normalised(c, "table", TABLES))
        check("migrated indexes match a fresh install", migrated_indexes ==
              normalised(c, "index", tuple(migrated_indexes)))

    again = run_migration(old_path)
    check("a second run reports nothing to do", again.returncode == 0 and "Already migrated" in again.stdout)
    check("a second run writes no second backup", len(list(work.glob("old.db.backup-*"))) == 1)

    # Partial: only some tables missing.
    with sqlite3.connect(str(old_path)) as c:
        c.execute("DROP TABLE model_activation_log")
        c.execute("DROP TABLE trained_models")
    partial = run_migration(old_path)
    with sqlite3.connect(str(old_path)) as c:
        names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    check("a partly migrated database is completed", partial.returncode == 0 and set(TABLES) <= names)

    # The path every existing installation takes: the application adds missing
    # tables itself on launch, with no manual step.
    launch = fresh_db(work, "launch.db")
    with launch.connect() as c:
        for name in ("model_activation_log", "trained_models", "training_runs", "frame_box_correction"):
            c.execute(f"DROP TABLE {name}")
    launch.ensure_schema(SCHEMA)
    with sqlite3.connect(str(work / "launch.db")) as c:
        names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        problems = c.execute("PRAGMA foreign_key_check").fetchall()
    check("launching an older database adds the tables on its own", set(TABLES) <= names and problems == [])

    empty = work / "bare.db"
    sqlite3.connect(str(empty)).close()
    refused = run_migration(empty)
    check("a database without observations is refused", refused.returncode == 2)
    del oid


def test_settings(work: Path) -> None:
    print("\n[3] Training settings")
    real = load_settings(str(REPO / "config" / "settings.json"), apply_local=False)
    check("the shipped block equals the documented defaults", real.raw["training"] == DEFAULT_TRAINING)
    check("the accessor returns a fresh copy", real.training_config() is not real.training_config())
    cfg = real.training_config()
    cfg["dataset"]["max_frames_per_event"] = 999
    check("editing the copy cannot change the loaded settings",
          real.training_config()["dataset"]["max_frames_per_event"] == DEFAULT_TRAINING["dataset"]["max_frames_per_event"])
    check("the folders resolve to absolute paths", all(p.is_absolute() for p in real.training_paths().values()))
    check("the engines and base models are listed",
          len(real.training_acoustic_engines()) >= 2 and len(real.training_vision_bases()) >= 4)

    root = work / "sandbox"
    (root / "config").mkdir(parents=True)
    raw = json.loads((REPO / "config" / "settings.json").read_text(encoding="utf-8"))

    def write(mutated: dict) -> Path:
        path = root / "config" / "settings.json"
        path.write_text(json.dumps(mutated), encoding="utf-8")
        return path

    old = copy.deepcopy(raw)
    old.pop("training")
    loaded = load_settings(str(write(old)), apply_local=False)
    check("a file without the block still loads", "training" not in loaded.raw)
    check("and gets the defaults", loaded.training_config() == DEFAULT_TRAINING)

    def bad(change) -> bool:
        broken = copy.deepcopy(raw)
        change(broken)
        try:
            load_settings(str(write(broken)), apply_local=False)
        except ConfigError:
            return True
        return False

    check("a fraction above one is refused", bad(lambda r: r["training"]["dataset"].__setitem__("background_max_fraction", 1.5)))
    check("an unknown criterion is refused",
          bad(lambda r: r["training"]["dataset"].__setitem__("default_required_criteria", ["Nonsense"])))
    check("a non-numeric seed is refused", bad(lambda r: r["training"]["dataset"]["split"].__setitem__("seed", "x")))
    check("a true or false value is not accepted as a number", bad(lambda r: r["training"]["dataset"]["split"].__setitem__("seed", True)))
    check("a non-boolean enabled flag is refused", bad(lambda r: r["training"].__setitem__("enabled", "yes")))
    check("a path that is not text is refused", bad(lambda r: r["training"]["paths"].__setitem__("models_dir", 5)))


def test_storage(work: Path) -> None:
    print("\n[4] Storage methods")
    d = fresh_db(work, "storage.db")
    oid = seed_observation(d)
    check("a bad model name is refused",
          all(raises(lambda n=n: db.validate_model_user_name(n), ValueError)
              for n in ("", "../x", "a b", "-lead", "CON", "lpt1", "x" * 41)))
    check("a good model name is accepted unchanged", db.validate_model_user_name("  Reef_2-a ") == "Reef_2-a")

    def run(name: str) -> dict:
        r = TrainingRun(id=new_id(), run_name=name, planned_model_name=name, modality="vision",
                        engine="vision_detector", mode="gentle", status="queued",
                        dataset_signature="sig", dataset_dir="d", run_dir="r", seed=1,
                        base_label="base", config_json="{}", created_by="tester", created_at=utc_now_iso())
        return d.create_training_run(r)

    r1 = run("a")
    check("a run is stored as model output", r1["data_source"] == "model" and r1["epochs_done"] == 0)
    with d.connect() as conn:
        check("a run cannot be relabelled as a measurement",
              raises(lambda: conn.execute("UPDATE training_runs SET data_source = 'measured' WHERE id = ?",
                                          (r1["id"],)), sqlite3.IntegrityError))
    upd = d.update_training_run(r1["id"], status="running", epochs_done=3, started_at=utc_now_iso())
    check("progress fields update", upd["status"] == "running" and upd["epochs_done"] == 3)
    check("an immutable field cannot be updated", raises(lambda: d.update_training_run(r1["id"], seed=9), ValueError))
    d.update_training_run(r1["id"], status="completed")
    check("a completed run cannot change", raises(lambda: d.update_training_run(r1["id"], epochs_done=4), ValueError))
    check("runs list newest first and filter", [x["id"] for x in d.list_training_runs(status="completed")] == [r1["id"]])

    def reg(name: str, run_id: str) -> dict:
        return d.register_trained_model(user_name=name, modality="vision", engine="vision_detector",
                                        run_id=run_id, folder="f", model_file="m.onnx", labels_file="l.json",
                                        base_label="b", metrics_json="{}", dataset_signature="sig",
                                        class_count=2, created_by="tester")

    check("the first version is V1", d.next_model_version("Reef") == 1)
    m1 = reg("Reef", r1["id"])
    m2 = reg("Reef", r1["id"])
    check("names follow Audtheia-Name-Vn", (m1["name"], m2["name"]) == ("Audtheia-Reef-V1", "Audtheia-Reef-V2"))
    m3 = reg("reef", r1["id"])
    check("a name differing only by case shares the counter", m3["version_number"] == 3)
    check("another name starts at V1", reg("Coral", r1["id"])["version_number"] == 1)
    check("registry reads back", d.get_trained_model(m1["id"])["name"] == "Audtheia-Reef-V1"
          and len(d.list_trained_models()) == 4 and len(d.list_trained_models(modality="acoustic")) == 0)
    with d.connect() as conn:
        check("a duplicate lower-cased name is refused by the table",
              raises(lambda: conn.execute("UPDATE trained_models SET name_key = ? WHERE id = ?",
                                          (m1["name_key"], m2["id"])), sqlite3.IntegrityError))

    a1 = d.add_model_activation(m1["id"], slot="desktop_verifier", action="activate",
                                previous_json='{"p": 1}', new_json='{"p": 2}', activated_by="tester")
    d.add_model_activation(m2["id"], slot="desktop_verifier", action="activate",
                           previous_json='{"p": 2}', new_json='{"p": 3}', activated_by="tester",
                           acknowledged_license=True)
    hist = d.model_activations("desktop_verifier")
    check("activation history is newest first", hist[0]["model_id"] == m2["id"] and hist[0]["acknowledged_license"] == 1)
    check("the previous activation is the one a revert restores", d.previous_activation("desktop_verifier")["id"] == a1["id"])
    check("a slot changed once has nothing to revert to", d.previous_activation("desktop_screening") is None)

    b = d.add_frame_box_correction(oid, 2, action="adjust", corrector="t", x1=1, y1=1, x2=4, y2=4)
    d.add_frame_box_correction(oid, 2, action="adjust", corrector="t", x1=2, y1=2, x2=6, y2=6)
    added = d.add_frame_box_correction(oid, 2, action="add", corrector="t", x1=1, y1=1, x2=2, y2=2,
                                       corrected_gbif_usage_key="12345", corrected_scientific_name="Aplysina fistularis")
    state = d.frame_boxes_for_observation(oid)
    model_box = [s for s in state if s["box_key"] == "model"]
    check("the newest correction per box is the current state", len(model_box) == 1 and model_box[0]["x2"] == 6)
    check("an added box has its own lineage", added["box_key"] != "model" and len(state) == 2)
    d.add_frame_box_correction(oid, 2, action="cleared", corrector="t")
    check("a retraction supersedes the earlier correction",
          [s for s in d.frame_boxes_for_observation(oid) if s["box_key"] == "model"][0]["action"] == "cleared")
    check("a missing observation reads as empty", d.frame_boxes_for_observation(new_id()) == [])
    del b

    legacy = Database(str(work / "legacy.db"))
    with sqlite3.connect(str(work / "legacy.db")) as c:
        c.execute("CREATE TABLE observations (id TEXT PRIMARY KEY)")
    check("readers degrade to empty on a database without the tables",
          legacy.list_training_runs() == [] and legacy.list_trained_models() == []
          and legacy.get_training_run("x") is None and legacy.model_activations("desktop_verifier") == []
          and legacy.next_model_version("Reef") == 1 and legacy.frame_boxes_for_observation("x") == [])


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="audtheia-train-"))
    try:
        test_tables(work)
        test_migration(work)
        test_settings(work)
        test_storage(work)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"\nResult: {CHECKS['passed']} passed, {CHECKS['failed']} failed")
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
