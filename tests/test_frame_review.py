"""Verification for per-frame review and its effect on the longitudinal pass.

Path: tests/test_frame_review.py

An expert can mark an individual frame of a visual event accurate or inaccurate.
The properties the design depends on are checked here directly rather than
assumed:

  - a verdict stores with human provenance and reads back as the current one,
  - the history is append-only and 'cleared' returns a frame to unreviewed,
  - the summary counts distinct frames by their current verdict,
  - the measured observation is byte-identical before and after any review, which
    is the whole point: a review never edits what the model recorded,
  - the database refuses a verdict outside its vocabulary,
  - the one-time migration adds the table without touching existing rows and is
    idempotent,
  - the frames endpoint returns the per-frame species distribution and the
    curated summary, and the review endpoint records a verdict and refuses a bad
    one,
  - the dream pass excludes an event an expert has rejected or marked wholly
    inaccurate, and keeps a partly-reviewed one.

Built on a temporary database created from the real schema.sql, so the CHECK
constraints exercised here are the shipped ones. Standard library plus the
interface test client only, and it never opens the live database.

Run: python tests/test_frame_review.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from audtheia.storage.database import (  # noqa: E402
    ChildDetection,
    Database,
    Observation,
    Station,
    new_id,
    utc_now_iso,
)

SCHEMA = REPO / "audtheia" / "storage" / "schema.sql"
CHECKS = {"passed": 0, "failed": 0}


def check(label: str, condition: bool) -> None:
    if condition:
        CHECKS["passed"] += 1
        print(f"  PASS  {label}")
    else:
        CHECKS["failed"] += 1
        print(f"  FAIL  {label}")


def _db_with_event(db_path: Path, *, frame_count: int = 4):
    db = Database(db_path)
    db.initialize_schema(SCHEMA)
    now = utc_now_iso()
    station = Station(id=new_id(), station_name="S", environment_type="terrestrial", created_at=now)
    db.create_station(station)
    obs = Observation(
        id=new_id(),
        event_name="e-" + new_id()[:8],
        station_id=station.id,
        trigger_source="vision",
        first_seen=now,
        last_seen=now,
        duration=4.0,
        data_source="model",
        created_at=now,
        frame_count=frame_count,
        screening_confidence=0.9,
    )
    child = ChildDetection(
        id=new_id(), observation_id=obs.id, modality="vision", created_at=now,
        scientific_name="Aplysina fistularis", confidence=0.9,
    )
    db.insert_observation(obs, children=[child])
    return db, obs


def test_storage_and_provenance(db_path: Path) -> None:
    print("\nA verdict stores with human provenance and never edits the model")
    db, obs = _db_with_event(db_path)
    before = db.get_observation(obs.id)

    stored = db.add_frame_review(obs.id, 0, verdict="accurate", corrector="expert")
    check("a verdict stores its value", stored["verdict"] == "accurate")
    check("a verdict carries human provenance", stored["data_source"] == "human_expert")
    check("a verdict records its corrector", stored["corrector"] == "expert")
    check("a verdict is stamped", bool(stored["reviewed_at"]))

    db.add_frame_review(obs.id, 1, verdict="accurate", corrector="expert")
    db.add_frame_review(obs.id, 2, verdict="inaccurate", corrector="expert")
    s = db.frame_review_summary(obs.id)
    check("summary counts two accurate", s["accurate"] == 2)
    check("summary counts one inaccurate", s["inaccurate"] == 1)
    check("summary counts three reviewed", s["reviewed"] == 3)

    after = db.get_observation(obs.id)
    check("the measured observation is byte-identical after reviews", after == before)


def test_append_only_and_clear(db_path: Path) -> None:
    print("\nThe history is append-only and 'cleared' returns a frame to unreviewed")
    db, obs = _db_with_event(db_path)
    db.add_frame_review(obs.id, 2, verdict="inaccurate", corrector="expert")
    db.add_frame_review(obs.id, 2, verdict="accurate", corrector="expert")
    latest = {r["frame_index"]: r["verdict"] for r in db.frame_reviews_for_observation(obs.id)}
    check("the newest verdict wins for a frame", latest[2] == "accurate")
    check("the summary follows the newest verdict", db.frame_review_summary(obs.id)["accurate"] == 1)

    db.add_frame_review(obs.id, 2, verdict="cleared", corrector="expert")
    s = db.frame_review_summary(obs.id)
    check("a cleared frame is no longer accurate", s["accurate"] == 0)
    check("a cleared frame is not inaccurate", s["inaccurate"] == 0)
    check("a cleared frame is not counted as reviewed", s["reviewed"] == 0)
    per_frame = {r["frame_index"]: r["verdict"] for r in db.frame_reviews_for_observation(obs.id)}
    check("the cleared verdict is still visible in history", per_frame.get(2) == "cleared")


def test_constraint_refuses(db_path: Path) -> None:
    print("\nThe database refuses a verdict outside its vocabulary")
    db, obs = _db_with_event(db_path)
    refused = False
    try:
        db.add_frame_review(obs.id, 0, verdict="maybe", corrector="expert")
    except sqlite3.IntegrityError:
        refused = True
    check("a verdict of 'maybe' is refused by the CHECK constraint", refused)


def test_migration(db_path: Path) -> None:
    print("\nThe migration adds the table without touching existing rows, idempotently")
    import migrate_add_frame_review as mig  # noqa: E402

    db, obs = _db_with_event(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("DROP TABLE frame_review")
        conn.commit()
        before = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]

    rc = mig.main(["--db", str(db_path)])
    check("the migration reports success", rc == 0)
    with sqlite3.connect(str(db_path)) as conn:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='frame_review'"
        ).fetchone()
        after = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    check("the migration created the table", exists is not None)
    check("no observation was changed by the migration", before == after)

    rc2 = mig.main(["--db", str(db_path)])
    check("running the migration again is harmless", rc2 == 0)
    # The migrated table accepts a real verdict, so it matches the shipped schema.
    stored = db.add_frame_review(obs.id, 0, verdict="accurate", corrector="expert")
    check("the migrated table stores a verdict", stored["verdict"] == "accurate")


def test_ensure_schema_self_heal(db_path: Path) -> None:
    print("\nensure_schema re-adds a missing table on an existing database, data intact")
    db, obs = _db_with_event(db_path)
    # Simulate a database created by a version that predates frame_review.
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("DROP TABLE frame_review")
        conn.commit()

    db.ensure_schema(SCHEMA)
    check("the observation survived the self-heal", db.get_observation(obs.id) is not None)
    stored = db.add_frame_review(obs.id, 0, verdict="accurate", corrector="expert")
    check("a verdict can be stored after the self-heal", stored["verdict"] == "accurate")

    # Running it again changes nothing and does not raise.
    db.ensure_schema(SCHEMA)
    check("the self-heal is idempotent", db.frame_review_summary(obs.id)["accurate"] == 1)


def test_ensure_schema_fresh(db_path: Path) -> None:
    print("\nensure_schema also initializes a brand-new empty database")
    db = Database(db_path)
    db.ensure_schema(SCHEMA)
    from audtheia.storage.database import Station, new_id, utc_now_iso
    now = utc_now_iso()
    db.create_station(Station(id=new_id(), station_name="Fresh", environment_type="marine", created_at=now))
    check("a fresh database initialized by ensure_schema is usable", len(db.list_stations()) == 1)


def test_snapshot_matching_and_stamping(db_path: Path) -> None:
    print("\nReference snapshot dates match a taxon by name and stamp only unset records")
    from audtheia.storage.database import SpeciesReference
    db, obs = _db_with_event(db_path)  # its child carries scientific_name "Aplysina fistularis"
    db.upsert_species_reference(SpeciesReference(
        gbif_usage_key="9999", scientific_name="Aplysina fistularis", common_name="Yellow tube sponge",
        fetched_at=utc_now_iso(), gbif_snapshot_date="2026-01-01", iucn_fetch_date="2026-01-02"))

    check("a scientific name matches its reference",
          (db.find_species_reference_by_name("Aplysina fistularis") or {}).get("gbif_usage_key") == "9999")
    check("matching ignores case and spacing",
          (db.find_species_reference_by_name("aplysina   fistularis") or {}).get("gbif_usage_key") == "9999")
    check("a common name matches its reference",
          (db.find_species_reference_by_name("Yellow tube sponge") or {}).get("gbif_usage_key") == "9999")
    check("a misspelled or unknown name does not match",
          db.find_species_reference_by_name("Rofous Crowned Sparrow") is None)

    filled = db.stamp_observation_snapshot(obs.id, "2026-01-01", "2026-01-02")
    check("an unset record is stamped", filled is True)
    after = db.get_observation(obs.id)
    check("the snapshot date is stored", after["gbif_snapshot_date"] == "2026-01-01")
    check("the iucn date is stored", after["iucn_fetch_date"] == "2026-01-02")

    # A second stamp with different values must not overwrite the already-set one.
    again = db.stamp_observation_snapshot(obs.id, "1999-09-09", "1999-09-09")
    check("an already-stamped record is not overwritten", again is False)
    check("the original snapshot date survives", db.get_observation(obs.id)["gbif_snapshot_date"] == "2026-01-01")


def _make_settings(tmp: Path):
    from audtheia.config import load_settings
    base = json.loads((REPO / "config" / "settings.json").read_text(encoding="utf-8"))
    base["node"]["role"] = "desktop"
    base["node"]["active_station_id"] = None
    base["paths"]["data_dir"] = str((tmp / "data").resolve())
    base["paths"]["reports_dir"] = str((tmp / "reports_out").resolve())
    path = tmp / "settings.frame.json"
    path.write_text(json.dumps(base), encoding="utf-8")
    return load_settings(path)


def _seed_frames_on_disk(settings, obs, names):
    """Write a small event directory the frames endpoint can read.

    `names` is the per-frame class name list; its length is the frame count.
    """
    data_dir = Path(settings.path("data_dir")).resolve()
    event_dir = data_dir / "detections" / "visual" / obs.event_name
    event_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, name in enumerate(names):
        fname = f"frame_{i}.jpg"
        (event_dir / fname).write_bytes(b"not-a-real-image")
        lines.append(json.dumps({
            "index": i, "file": fname, "captured_at": obs.first_seen,
            "confidence": 0.8, "class_name": name, "bbox_xyxy": [1, 1, 2, 2],
        }))
    (event_dir / "annotations.jsonl").write_text("\n".join(lines), encoding="utf-8")
    return str((event_dir / "frame_0.jpg").resolve())


def test_endpoints(tmp: Path) -> None:
    print("\nThe frames endpoint distributes and curates; the review endpoint records")
    from fastapi.testclient import TestClient
    from audtheia.app import server as srv

    tmp.mkdir(parents=True, exist_ok=True)
    settings = _make_settings(tmp)
    db_path = tmp / "server.db"
    db, obs = _db_with_event(db_path, frame_count=4)
    # Give the event a representative frame and four frames on disk, three of one
    # species and one of another, so the distribution has something to report.
    rep = _seed_frames_on_disk(settings, obs, [
        "Aplysina fistularis", "Aplysina fistularis", "Aplysina fistularis", "Aplysina fulva",
    ])
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("UPDATE observations SET representative_frame = ? WHERE id = ?", (rep, obs.id))
        conn.commit()

    app = srv.create_app(settings, db)
    client = TestClient(app)

    r = client.get(f"/api/detections/{obs.id}/frames")
    check("the frames endpoint returns 200", r.status_code == 200)
    data = r.json()
    dist = {d["class_name"]: d["count"] for d in data.get("distribution", [])}
    check("the distribution counts the dominant species", dist.get("Aplysina fistularis") == 3)
    check("the distribution counts the second species", dist.get("Aplysina fulva") == 1)
    check("the summary reports the total frames", data["review_summary"]["total_frames"] == 4)
    check("a mixed track is flagged multiple_candidates", data["review_summary"]["multiple_candidates"] is True)

    bad = client.post(f"/api/detections/{obs.id}/frames/2/review", json={"verdict": "sideways"})
    check("an unknown verdict is a 400", bad.status_code == 400)
    missing = client.post(f"/api/detections/does-not-exist/frames/0/review", json={"verdict": "accurate"})
    check("an unknown observation is a 404", missing.status_code == 404)

    ok = client.post(f"/api/detections/{obs.id}/frames/3/review", json={"verdict": "inaccurate"})
    check("a valid verdict is a 201", ok.status_code == 201)
    summ = ok.json()["review_summary"]
    check("the returned summary counts the inaccurate frame", summ["inaccurate"] == 1)
    # The review response must carry the SAME full summary the frames read does,
    # so the kept count and trust update live on a toggle rather than blanking.
    check("the POST summary carries the total frame count", summ.get("total_frames") == 4)
    check("the POST summary carries the curated count", summ.get("curated_frame_count") == 3)
    check("the POST summary carries the trust weight",
          summ.get("trust") is not None and abs(summ["trust"] - 0.75) < 1e-9)
    check("the POST summary carries the multiple_candidates flag", summ.get("multiple_candidates") is True)

    again = client.get(f"/api/detections/{obs.id}/frames").json()
    curated = again["review_summary"]["curated_frame_count"]
    trust = again["review_summary"]["trust"]
    check("the curated count subtracts the inaccurate frame", curated == 3)
    check("the trust weight is three of four", abs(trust - 0.75) < 1e-9)
    marked = [f for f in again["frames"] if f["index"] == 3]
    check("the reviewed frame reports its verdict", marked and marked[0]["review"] == "inaccurate")


def test_dream_gate(tmp: Path) -> None:
    print("\nThe dream pass excludes a rejected or wholly-inaccurate event, keeps a partial one")
    from audtheia.analysis.dream import DreamEngine

    tmp.mkdir(parents=True, exist_ok=True)
    settings = _make_settings(tmp)
    db, obs = _db_with_event(tmp / "gate.db", frame_count=4)
    engine = DreamEngine(settings=settings, db=db)

    check("a normal event is not excluded", engine._event_excluded(obs.id, 4) is False)

    # Mark two of four frames inaccurate: a partly-reviewed event still counts.
    db.add_frame_review(obs.id, 0, verdict="inaccurate", corrector="expert")
    db.add_frame_review(obs.id, 1, verdict="inaccurate", corrector="expert")
    check("a partly-inaccurate event is kept", engine._event_excluded(obs.id, 4) is False)

    # Mark the remaining two inaccurate: every frame is now discredited.
    db.add_frame_review(obs.id, 2, verdict="inaccurate", corrector="expert")
    db.add_frame_review(obs.id, 3, verdict="inaccurate", corrector="expert")
    check("a wholly-inaccurate event is excluded", engine._event_excluded(obs.id, 4) is True)

    # A fresh event that is rejected at the event level is excluded regardless.
    db2, obs2 = _db_with_event(tmp / "gate2.db", frame_count=4)
    engine2 = DreamEngine(settings=settings, db=db2)
    check("an unreviewed event is not excluded", engine2._event_excluded(obs2.id, 4) is False)
    db2.add_correction(obs2.id, verdict="reject", corrector="expert")
    check("a rejected event is excluded", engine2._event_excluded(obs2.id, 4) is True)


def main() -> int:
    print("=" * 72)
    print("Per-frame review: storage, provenance, migration, endpoints, dream gate")
    print("=" * 72)
    if not SCHEMA.exists():
        print("  FAIL  schema.sql not found")
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        test_storage_and_provenance(root / "store.db")
        test_append_only_and_clear(root / "clear.db")
        test_constraint_refuses(root / "constraint.db")
        test_migration(root / "migrate.db")
        test_ensure_schema_self_heal(root / "heal.db")
        test_ensure_schema_fresh(root / "fresh.db")
        test_snapshot_matching_and_stamping(root / "snapshot.db")
        test_endpoints(root / "endpoints")
        test_dream_gate(root / "gate")
    print("\n" + "=" * 72)
    print(f"RESULT: {CHECKS['passed']} passed, {CHECKS['failed']} failed")
    print("=" * 72)
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
