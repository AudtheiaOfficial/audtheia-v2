"""A miniature, realistic record for the training checks.

Path: tests/training_fixture.py

Builds a temporary repository root with its own settings, database, event folders
and clips, so the training code is exercised end to end against the real storage
layer and the real on-disk event format without touching the project's data,
database, reports or models. Not a test itself, so the runner does not collect it.
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from audtheia.config import load_settings  # noqa: E402
from audtheia.storage.database import (  # noqa: E402
    ChildDetection, Database, Observation, ObservationVerification, Station, new_id, utc_now_iso,
)

SCHEMA = REPO / "audtheia" / "storage" / "schema.sql"
IMG_W, IMG_H = 64, 48
BOX = [10.0, 8.0, 40.0, 30.0]


class World:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.repo = self.root / "repo"
        (self.repo / "config").mkdir(parents=True, exist_ok=True)
        base = json.loads((REPO / "config" / "settings.json").read_text(encoding="utf-8"))
        base["node"]["role"] = "desktop"
        base["node"]["active_station_id"] = None
        base["paths"]["db_path"] = str((self.repo / "database" / "audtheia.db").resolve())
        base["paths"]["schema_path"] = str(SCHEMA)
        base["paths"]["gbif_backbone_path"] = str((self.repo / "backbone").resolve())
        self.settings_path = self.repo / "config" / "settings.json"
        self.settings_path.write_text(json.dumps(base), encoding="utf-8")
        self.settings = load_settings(self.settings_path)
        (self.repo / "database").mkdir(parents=True, exist_ok=True)
        self.db = Database(self.settings.db_path())
        self.db.initialize_schema(SCHEMA)
        self.data = Path(self.settings.path("data_dir"))
        self.stations: dict = {}
        self.counter = 0

    # -- records -----------------------------------------------------------

    def station(self, name: str, environment: str = "marine") -> str:
        if name not in self.stations:
            sid = new_id()
            self.db.upsert_station(Station(id=sid, station_name=name, environment_type=environment,
                                           created_at=utc_now_iso()))
            self.stations[name] = sid
        return self.stations[name]

    def _next(self) -> int:
        self.counter += 1
        return self.counter

    def vision_event(self, species, *, station="ReefA", first_seen="2026-03-01T10:00:00Z", frames=6,
                     qc="qc_passed", verified=True, agrees=1, identical_frames=False, class_names=None,
                     no_frames_on_disk=False, name=None) -> dict:
        """One tracked event with saved frames. species is a list of (scientific_name, gbif_key)."""
        n = self._next()
        sid = self.station(station)
        oid = new_id()
        event = name or f"{station}_ev{n:04d}"
        event_dir = self.data / "detections" / "visual" / event
        event_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for i in range(frames):
            seed = 7 if identical_frames else (n * 1000 + i)
            rng = np.random.RandomState(seed)
            arr = rng.randint(0, 255, size=(IMG_H, IMG_W, 3), dtype=np.uint8)
            fname = f"frame_{i:06d}.jpg"
            if not no_frames_on_disk:
                Image.fromarray(arr).save(event_dir / fname, quality=95)
            cls = (class_names[i % len(class_names)] if class_names else species[i % len(species)][0])
            rows.append({"index": i, "file": fname, "class_name": cls, "bbox_xyxy": BOX, "confidence": 0.8})
        (event_dir / "annotations.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        now = utc_now_iso()
        rel = f"data/detections/visual/{event}/frame_000000.jpg"
        children = [ChildDetection(id=new_id(), observation_id=oid, modality="vision", created_at=now,
                                   gbif_usage_key=str(k) if k is not None else None, scientific_name=s,
                                   common_name=s, confidence=0.8, bbox_x=10, bbox_y=8, bbox_w=30, bbox_h=22)
                    for s, k in species]
        self.db.insert_observation(Observation(
            id=oid, event_name=event, station_id=sid, trigger_source="vision", first_seen=first_seen,
            last_seen=first_seen, duration=2.0, data_source="model", created_at=now, qc_state=qc,
            representative_frame=rel, frame_count=frames), children=children)
        if verified is not None:
            self.db.upsert_observation_verification(ObservationVerification(
                observation_id=oid, created_at=now, verified=1 if verified else 0,
                rfdetr_agrees_with_field=agrees))
        return {"id": oid, "event": event, "dir": event_dir, "children": [c.id for c in children],
                "station_id": sid, "frames": frames, "species": species}

    def audio_event(self, species, *, station="ReefA", first_seen="2026-03-01T10:00:00Z", seconds=8.0,
                    rate=24000, burst=(4.0, 5.5), qc="qc_passed", verified=True, name=None) -> dict:
        n = self._next()
        sid = self.station(station)
        oid = new_id()
        event = name or f"{station}_au{n:04d}"
        folder = self.data / "detections" / "audio"
        folder.mkdir(parents=True, exist_ok=True)
        rng = np.random.RandomState(n)
        t = np.arange(int(seconds * rate)) / float(rate)
        sig = 0.01 * rng.randn(len(t))
        if burst:
            m = (t >= burst[0]) & (t < burst[1])
            sig[m] += 0.5 * np.sin(2 * np.pi * 2000.0 * t[m])
        pcm = (np.clip(sig, -1, 1) * 32767).astype("<i2")
        path = folder / f"{event}.wav"
        with wave.open(str(path), "wb") as h:
            h.setnchannels(1)
            h.setsampwidth(2)
            h.setframerate(rate)
            h.writeframes(pcm.tobytes())
        now = utc_now_iso()
        children = [ChildDetection(id=new_id(), observation_id=oid, modality="audio", created_at=now,
                                   gbif_usage_key=str(k) if k is not None else None, scientific_name=s,
                                   common_name=s, confidence=0.7) for s, k in species]
        self.db.insert_observation(Observation(
            id=oid, event_name=event, station_id=sid, trigger_source="audio", first_seen=first_seen,
            last_seen=first_seen, duration=seconds, data_source="model", created_at=now, qc_state=qc,
            audio_clip_path=f"data/detections/audio/{event}.wav", audio_true_duration_seconds=seconds),
            children=children)
        if verified is not None:
            self.db.upsert_observation_verification(ObservationVerification(
                observation_id=oid, created_at=now, verified=1 if verified else 0))
        return {"id": oid, "event": event, "path": path, "children": [c.id for c in children],
                "station_id": sid, "species": species}

    # -- expert actions ------------------------------------------------------

    def confirm(self, ev, detection=None, modality=None):
        return self.db.add_correction(ev["id"], verdict="confirm", corrector="tester",
                                      detection_id=detection, modality=modality)

    def reject(self, ev, detection=None, modality=None):
        return self.db.add_correction(ev["id"], verdict="reject", corrector="tester",
                                      detection_id=detection, modality=modality)

    def relabel(self, ev, name, gbif=None, detection=None, modality=None):
        return self.db.add_correction(
            ev["id"], verdict="relabel", corrector="tester", detection_id=detection, modality=modality,
            corrected_scientific_name=name, corrected_gbif_usage_key=str(gbif) if gbif is not None else None)

    def frame(self, ev, index, verdict):
        return self.db.add_frame_review(ev["id"], index, verdict=verdict, corrector="tester")


def fake_rollup(table):
    """A roll-up lookup for tests. table maps a class key to {"genus": (key, name), "family": (key, name)}."""
    def lookup(class_keys, level):
        parents, info = {}, {}
        for key in class_keys:
            entry = table.get(key)
            if entry is None:
                info[key] = {"reason": "not in the test table"}
                parents[key] = None
                continue
            info[key] = {"rank": "SPECIES"}
            pair = entry.get(level)
            parents[key] = {"key": pair[0], "name": pair[1]} if pair else None
        return parents, info
    return lookup


def make_backbone(directory: Path) -> Path:
    """A tiny backbone in the shipped layout: index.db plus a tab separated simple.txt."""
    import sqlite3

    directory.mkdir(parents=True, exist_ok=True)
    taxa = [
        # key, accepted, name, rank, family key, genus key
        ("100", "100", "Aplysina fistularis", "SPECIES", "10", "20"),
        ("101", "100", "Verongia fistularis", "SPECIES", "10", "20"),
        ("102", "102", "Aplysina lacunosa", "SPECIES", "10", "20"),
        ("20", "20", "Aplysina", "GENUS", "10", "20"),
        ("10", "10", "Aplysinidae", "FAMILY", "10", None),
        ("200", "200", "Poecile carolinensis", "SPECIES", "11", "21"),
        ("21", "21", "Poecile", "GENUS", "11", "21"),
        ("11", "11", "Paridae", "FAMILY", "11", None),
        ("300", "300", "Dup name", "SPECIES", "12", "22"),
        ("301", "301", "Dup name", "SPECIES", "12", "23"),
        ("5", "5", "Animalia", "KINGDOM", None, None),
    ]
    conn = sqlite3.connect(directory / "index.db")
    conn.execute(
        "CREATE TABLE taxon_index (usage_key TEXT PRIMARY KEY, canonical_name TEXT NOT NULL, scientific_name TEXT,"
        " status TEXT NOT NULL, accepted_key TEXT NOT NULL, accepted_name TEXT, name_lower TEXT NOT NULL) STRICT")
    for key, acc, name, rank, _f, _g in taxa:
        # Like the shipped index, only species and their synonyms are indexed. A genus,
        # a family or anything higher is named from the full backbone file.
        if rank == "SPECIES":
            conn.execute("INSERT INTO taxon_index VALUES (?,?,?,?,?,?,?)",
                         (key, name, name, "ACCEPTED" if key == acc else "SYNONYM", acc, None, name.lower()))
    conn.commit()
    conn.close()
    lines = []
    for key, acc, name, rank, fam, gen in taxa:
        cols = [""] * 20
        cols[0] = key
        cols[1] = ""
        cols[4] = "ACCEPTED" if key == acc else "SYNONYM"
        cols[5] = rank
        cols[14] = fam or ""
        cols[15] = gen or ""
        cols[19] = name
        lines.append("\t".join(cols))
    (directory / "simple.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory
