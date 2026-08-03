"""Archiving copies captured frames, and reclaim frees only what it copied.

Path: tests/test_archive.py

The reclaim path deletes real captured images, so this proves the guards hold:
frames are copied to the destination with a metadata sidecar; the originals are
removed only after the copy is confirmed and only from inside the detections
directory; a file outside that directory is never touched; the destination may
not sit inside the captured-data folder; and the observation record survives a
reclaim untouched.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from audtheia.config import load_settings  # noqa: E402
from audtheia.storage.database import (  # noqa: E402
    Database, Station, Observation, ChildDetection, new_id, utc_now_iso,
)
from audtheia.storage.archive import archive_events, ArchiveError  # noqa: E402

CHECKS = {"passed": 0, "failed": 0}


def check(label, cond):
    CHECKS["passed" if cond else "failed"] += 1
    print(("  PASS  " if cond else "  FAIL  ") + label)


def _settings(tmp: Path):
    base = json.loads((REPO / "config" / "settings.json").read_text(encoding="utf-8"))
    base["node"]["role"] = "desktop"
    base["paths"]["db_path"] = str((tmp / "audtheia.db"))
    base["paths"]["data_dir"] = str((tmp / "data"))
    base["paths"]["detections_visual_dir"] = str((tmp / "data" / "detections" / "visual"))
    p = tmp / "s.json"
    p.write_text(json.dumps(base), encoding="utf-8")
    return load_settings(p)


def _event(db, settings, station_id, name):
    now = utc_now_iso()
    oid = new_id()
    db.insert_observation(
        Observation(id=oid, event_name=name, station_id=station_id, trigger_source="vision",
                    first_seen=now, last_seen=now, duration=1.0, data_source="model",
                    created_at=now, frame_count=2, screening_confidence=0.9),
        children=[ChildDetection(id=new_id(), observation_id=oid, modality="vision",
                                 created_at=now, confidence=0.9, scientific_name="Taxon")],
    )
    folder = Path(settings.path("detections_visual_dir")) / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "frame_000000.jpg").write_bytes(b"IMG0")
    (folder / "frame_000001.jpg").write_bytes(b"IMG1")
    (folder / "annotations.jsonl").write_text('{"index":0}\n', encoding="utf-8")
    return oid, folder


def test_archive_and_reclaim():
    print("\n[1] Archive copies frames, reclaim frees originals, record survives")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        settings = _settings(tmp)
        db = Database(settings.db_path()); db.initialize_schema(REPO / "audtheia" / "storage" / "schema.sql")
        st = Station(id=new_id(), station_name="S", environment_type="terrestrial", created_at=utc_now_iso())
        db.create_station(st)
        oid, folder = _event(db, settings, st.id, "S_2026-01-01_abcdef12")

        # A file outside the detections directory must never be touched.
        sentinel = tmp / "keep_me.txt"; sentinel.write_text("do not delete", encoding="utf-8")

        target = tmp / "archive"
        res = archive_events(db, settings, target_dir=str(target), reclaim=True)

        check("one event was archived", res["archived"] == 1)
        check("one event was reclaimed", res["reclaimed"] == 1)
        check("the frames were copied to the destination",
              (target / "S_2026-01-01_abcdef12" / "frame_000000.jpg").read_bytes() == b"IMG0")
        check("a metadata sidecar was written",
              (target / "S_2026-01-01_abcdef12" / "metadata.json").is_file())
        check("the original frames were freed", not folder.exists())
        check("the observation record survives the reclaim", db.get_observation(oid) is not None)
        check("a file outside the detections directory is untouched", sentinel.read_text() == "do not delete")


def test_reclaim_off_keeps_originals():
    print("\n[2] Without reclaim, the originals are kept")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        settings = _settings(tmp)
        db = Database(settings.db_path()); db.initialize_schema(REPO / "audtheia" / "storage" / "schema.sql")
        st = Station(id=new_id(), station_name="S", environment_type="terrestrial", created_at=utc_now_iso())
        db.create_station(st)
        _oid, folder = _event(db, settings, st.id, "S_2026-01-02_beefbeef")
        res = archive_events(db, settings, target_dir=str(tmp / "arch"), reclaim=False)
        check("archived without reclaiming", res["archived"] == 1 and res["reclaimed"] == 0)
        check("the originals remain on disk", (folder / "frame_000000.jpg").is_file())


def test_destination_inside_data_is_refused():
    print("\n[3] A destination inside the captured-data folder is refused")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        settings = _settings(tmp)
        db = Database(settings.db_path()); db.initialize_schema(REPO / "audtheia" / "storage" / "schema.sql")
        inside = Path(settings.path("detections_visual_dir")) / "sub"
        refused = False
        try:
            archive_events(db, settings, target_dir=str(inside), reclaim=True)
        except ArchiveError:
            refused = True
        check("archiving into the captured-data folder is refused", refused)


def main() -> int:
    test_archive_and_reclaim()
    test_reclaim_off_keeps_originals()
    test_destination_inside_data_is_refused()
    print(f"\n==== archive: {CHECKS['passed']} passed, {CHECKS['failed']} failed ====")
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
