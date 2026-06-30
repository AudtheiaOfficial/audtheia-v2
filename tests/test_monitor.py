"""Mocked-hardware verification for the field-station detection loop.

Path: tests/test_monitor.py

Runs the loop end to end with no camera, accelerator, or sensors present. A
scripted detector and a scripted frame source stand in for the hardware, and a
capture sink that gathers nothing stands in for the not-yet-built audio,
location, and environmental captures. Every tuning value is read from the real
configuration through the real loader, and every row is read back and checked
against the real schema.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from audtheia.config import load_settings  # noqa: E402
from audtheia.storage.database import Database, Station, utc_now_iso  # noqa: E402
from audtheia.pipeline.monitor import (  # noqa: E402
    Frame,
    Monitor,
    NullTriggerSink,
    RawDetection,
    build_tracker_from_capture,
)

ISO = "%Y-%m-%dT%H:%M:%S.%fZ"


# ---------------------------------------------------------------------------
# Mock hardware
# ---------------------------------------------------------------------------


class ScriptedFrameSource:
    """Replays a fixed number of frames at a fixed rate. Each frame carries a
    small synthetic image so the loop's frame-writing path runs for real."""

    def __init__(self, n_frames: int, fps: float, base_time: datetime, *, time_provisional: int = 0):
        self._n = n_frames
        self._fps = fps
        self._base = base_time
        self._i = 0
        self._time_provisional = time_provisional

    def read(self):
        if self._i >= self._n:
            return None
        t = self._base + timedelta(seconds=self._i / self._fps)
        frame = Frame(
            index=self._i,
            image=np.zeros((24, 32, 3), dtype=np.uint8),
            captured_at=t.strftime(ISO),
            time_provisional=self._time_provisional,
        )
        self._i += 1
        return frame

    def close(self):
        pass


class ScriptedDetector:
    """Returns the scripted detections for each frame index."""

    def __init__(self, class_names: dict[int, str], script: dict[int, list[RawDetection]]):
        self._class_names = class_names
        self._script = script

    @property
    def class_names(self):
        return self._class_names

    def detect(self, frame):
        return list(self._script.get(frame.index, []))

    def close(self):
        pass


def moving_box(i: int, conf: float, class_id: int, class_name: str) -> RawDetection:
    """A detection whose box drifts a little each frame so the tracker links it
    across frames the way a real animal moving slowly would."""
    x = 10.0 + i * 0.5
    y = 10.0 + i * 0.3
    return RawDetection(x1=x, y1=y, x2=x + 40.0, y2=y + 30.0, confidence=conf, class_id=class_id, class_name=class_name)


def script_to_json(script: dict[int, list[RawDetection]]) -> str:
    """Serialize a detection script to the on-disk fixture format (the loader
    used by the in-code list plus an optional fixture loader)."""
    return json.dumps(
        {
            str(idx): [
                {
                    "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
                    "confidence": d.confidence, "class_id": d.class_id, "class_name": d.class_name,
                }
                for d in dets
            ]
            for idx, dets in script.items()
        }
    )


def script_from_json(text: str) -> dict[int, list[RawDetection]]:
    raw = json.loads(text)
    return {
        int(idx): [RawDetection(**d) for d in dets]
        for idx, dets in raw.items()
    }


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def make_pi_settings():
    """Write a field-station copy of the real configuration (role pi, with an
    active station) and load it through the real loader.

    The copy is written inside config/ so the loader anchors every relative
    path to the real repository root exactly as it would in the field. The
    caller removes it afterward.
    """
    base = json.loads((REPO / "config" / "settings.json").read_text(encoding="utf-8"))
    base["node"]["role"] = "pi"
    base["node"]["active_station_id"] = base["stations"][0]["station_id"]
    path = REPO / "config" / "settings.pi.test.json"
    path.write_text(json.dumps(base), encoding="utf-8")
    return load_settings(path), path


def fresh_db(settings, tmp: Path) -> Database:
    db_path = tmp / "audtheia.db"
    db = Database(str(db_path), **settings.database_kwargs())
    db.initialize_schema(settings.schema_path())
    station = settings.active_station()
    db.create_station(
        Station(
            id=station["station_id"],
            station_name=station["station_name"],
            environment_type=station["environment_type"],
            created_at=utc_now_iso(),
        )
    )
    return db


def clean_visual_dir(settings):
    vdir = Path(settings.path("detections_visual_dir"))
    if vdir.exists():
        shutil.rmtree(vdir)
    vdir.mkdir(parents=True, exist_ok=True)
    return vdir


def run_monitor(settings, db, script, *, n_frames, station_override=None):
    station = station_override or settings.active_station()
    fps = station["capture"]["fps"]
    src = ScriptedFrameSource(n_frames, fps, datetime(2026, 6, 30, 12, 0, 0, tzinfo=timezone.utc))
    detector = ScriptedDetector({0: "Aplysina_fistularis", 1: "Xestospongia_muta"}, script)
    tracker = build_tracker_from_capture(station["capture"])
    mon = Monitor(
        settings=settings,
        station=station,
        db=db,
        frame_source=src,
        detector=detector,
        tracker=tracker,
        trigger_sink=NullTriggerSink(),
    )
    mon.run()
    return mon


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

CHECKS = {"passed": 0, "failed": 0}


def check(label: str, condition: bool):
    if condition:
        CHECKS["passed"] += 1
        print(f"  PASS  {label}")
    else:
        CHECKS["failed"] += 1
        print(f"  FAIL  {label}")


# Columns every observation row must carry, from the schema.
REQUIRED_COLS = [
    "id", "event_name", "station_id", "trigger_source", "first_seen", "last_seen",
    "duration", "time_provisional", "data_source", "qc_state", "created_at",
]


def validate_observation_row(row: dict):
    ok = all(row.get(c) is not None for c in REQUIRED_COLS)
    check("observation row carries every required column", ok)
    check("trigger_source is vision", row["trigger_source"] == "vision")
    check("data_source is model", row["data_source"] == "model")
    check("qc_state starts pending", row["qc_state"] == "qc_pending")
    check(
        "provisional salience is in range and equals screening confidence",
        row["salience_provisional"] is not None
        and 0.0 <= row["salience_provisional"] <= 1.0
        and abs(row["salience_provisional"] - row["screening_confidence"]) < 1e-9,
    )
    check("authoritative salience is untouched at capture", "salience_authoritative" not in row)
    check("event_name follows station_date_short form", row["event_name"].startswith("ExampleReef_2026-06-30_"))
    check("screening_model_version stamped from config (null here, by config)", "screening_model_version" in row)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def scenario_single_track(settings):
    print("\n[1] One animal across many frames collapses to one observation")
    vdir = clean_visual_dir(settings)
    with tempfile.TemporaryDirectory() as d:
        db = fresh_db(settings, Path(d))
        # Present frames 3..40 with a clear confidence peak at frame 20, then a
        # gap longer than the close threshold (30) to end the encounter.
        script = {}
        for i in range(3, 41):
            conf = 0.95 if i == 20 else 0.6 + 0.2 * np.sin(i)
            conf = float(min(max(conf, 0.3), 0.99))
            script[i] = [moving_box(i, conf, 0, "Aplysina_fistularis")]
        mon = run_monitor(settings, db, script, n_frames=80)

        obs = db.list_observations()
        check("exactly one observation written", len(obs) == 1)
        if not obs:
            return
        row = obs[0]
        validate_observation_row(row)

        children = db.list_child_detections(row["id"])
        check("one child detection (one resolved taxon)", len(children) == 1)
        if children:
            c = children[0]
            check("child modality is vision", c["modality"] == "vision")
            check("child carries the model label as common name", c["common_name"] == "Aplysina_fistularis")
            check("child carries a bounding box", None not in (c["bbox_x"], c["bbox_y"], c["bbox_w"], c["bbox_h"]))

        event_dir = vdir / row["event_name"]
        jpgs = sorted(event_dir.glob("frame_*.jpg"))
        check("every detected frame stored as a raw image", len(jpgs) == row["frame_count"])
        check("frame_count is positive and matches stored frames", row["frame_count"] == len(jpgs) > 0)

        jsonl = (event_dir / "annotations.jsonl").read_text(encoding="utf-8").strip().splitlines()
        check("one annotation line per stored frame", len(jsonl) == len(jpgs))

        manifest = json.loads((event_dir / "annotations.json").read_text(encoding="utf-8"))
        check("manifest records the representative frame", manifest["representative_frame"] is not None)
        check("representative frame path points at a real file", (vdir / ".." / ".." / ".." / row["representative_frame"]).resolve().exists()
              or (Path(settings.repo_root) / row["representative_frame"]).exists())
        # The representative frame is the highest-confidence frame.
        best_line = max((json.loads(x) for x in jsonl), key=lambda r: r["confidence"])
        check("representative frame is the highest-confidence frame", manifest["representative_frame"] == best_line["file"])

        check("duration is the true window length", abs(row["duration"] - (len(jpgs) and (json.loads(jsonl[-1])["index"] - json.loads(jsonl[0])["index"]) / settings.active_station()["capture"]["fps"])) < 1e-6)
        check("writer reported no failures and no skips", mon.events_failed == 0 and mon.observations_skipped_no_track == 0)


def scenario_no_detection(settings):
    print("\n[2] A stream with no detections writes nothing")
    vdir = clean_visual_dir(settings)
    with tempfile.TemporaryDirectory() as d:
        db = fresh_db(settings, Path(d))
        mon = run_monitor(settings, db, {}, n_frames=60)
        obs = db.list_observations()
        check("no observations written for an empty stream", len(obs) == 0)
        check("no event folders created", len(list(vdir.glob("*"))) == 0)
        check("writer wrote nothing and skipped nothing", mon.events_written == 0 and mon.observations_skipped_no_track == 0)


def scenario_long_track_segments(settings):
    print("\n[3] A long encounter stays one record while its media segments roll")
    vdir = clean_visual_dir(settings)
    # Override only the media-segment bound for this test so a roll happens
    # quickly; identity must still never split.
    station = json.loads(json.dumps(settings.active_station()))
    station["capture"]["max_event_duration_seconds"] = 2.0  # 2 seconds per media segment
    with tempfile.TemporaryDirectory() as d:
        db = fresh_db(settings, Path(d))
        # 120 continuous frames at 15 fps is 8 seconds, so several 2-second
        # media segments, but one animal, so one record.
        script = {i: [moving_box(i, 0.8, 0, "Aplysina_fistularis")] for i in range(0, 120)}
        run_monitor(settings, db, script, n_frames=160, station_override=station)

        obs = db.list_observations()
        check("a long continuous encounter is exactly one observation", len(obs) == 1)
        if not obs:
            return
        row = obs[0]
        event_dir = vdir / row["event_name"]
        manifest = json.loads((event_dir / "annotations.json").read_text(encoding="utf-8"))
        check("the encounter rolled into more than one media segment", manifest["segment_count"] > 1)
        # Every annotation shares the one observation identity.
        segs = {json.loads(x)["segment"] for x in (event_dir / "annotations.jsonl").read_text().strip().splitlines()}
        check("media segments are numbered across the single event", len(segs) == manifest["segment_count"])
        check("one identity across the whole encounter", manifest["observation_id"] == row["id"])
        check("duration spans the whole encounter, not one segment", row["duration"] > 2.0)


def scenario_brief_occlusion(settings):
    print("\n[4] A brief gap does not split one animal into two records")
    clean_visual_dir(settings)
    with tempfile.TemporaryDirectory() as d:
        db = fresh_db(settings, Path(d))
        close_frames = settings.active_station()["capture"]["bytetrack"]["track_close_frames"]  # 30
        script = {}
        for i in range(3, 18):  # present
            script[i] = [moving_box(i, 0.8, 0, "Aplysina_fistularis")]
        # gap of 5 frames (well under the 30-frame close threshold)
        for i in range(23, 45):  # reappears
            script[i] = [moving_box(i, 0.8, 0, "Aplysina_fistularis")]
        run_monitor(settings, db, script, n_frames=90)
        obs = db.list_observations()
        check(f"one record across a {5}-frame gap (close threshold {close_frames})", len(obs) == 1)


def scenario_two_animals(settings):
    print("\n[5] Two animals in view at once become two records")
    clean_visual_dir(settings)
    with tempfile.TemporaryDirectory() as d:
        db = fresh_db(settings, Path(d))
        script = {}
        for i in range(3, 40):
            a = RawDetection(20 + i * 0.3, 20, 60 + i * 0.3, 60, 0.85, 0, "Aplysina_fistularis")
            b = RawDetection(400 - i * 0.3, 300, 440 - i * 0.3, 340, 0.85, 1, "Xestospongia_muta")
            script[i] = [a, b]
        run_monitor(settings, db, script, n_frames=80)
        obs = db.list_observations()
        check("two simultaneous tracks write two observations", len(obs) == 2)
        names = {db.list_child_detections(o["id"])[0]["common_name"] for o in obs if db.list_child_detections(o["id"])}
        check("the two records carry the two distinct taxa", names == {"Aplysina_fistularis", "Xestospongia_muta"})


def scenario_json_fixture(settings):
    print("\n[6] The same loop runs from a JSON detection fixture")
    clean_visual_dir(settings)
    with tempfile.TemporaryDirectory() as d:
        db = fresh_db(settings, Path(d))
        script = {i: [moving_box(i, 0.8, 0, "Aplysina_fistularis")] for i in range(3, 30)}
        fixture = Path(d) / "feed.json"
        fixture.write_text(script_to_json(script), encoding="utf-8")
        loaded = script_from_json(fixture.read_text(encoding="utf-8"))
        check("fixture round-trips to the same detection count", sum(len(v) for v in loaded.values()) == len(script))
        run_monitor(settings, db, loaded, n_frames=70)
        obs = db.list_observations()
        check("the fixture-driven run writes one observation", len(obs) == 1)


def scenario_tuning_from_config(settings):
    print("\n[7] Every tuning value is taken from configuration, nothing hardcoded")
    station = settings.active_station()
    cap = station["capture"]
    # Prove the monitor reads these from the station rather than holding its own.
    with tempfile.TemporaryDirectory() as d:
        db = fresh_db(settings, Path(d))
        clean_visual_dir(settings)
        mon = run_monitor(settings, db, {}, n_frames=1)
        check("close threshold came from config", mon._track_close_frames == cap["bytetrack"]["track_close_frames"])
        check("media-segment bound came from config", mon._max_event_duration_seconds == float(cap["max_event_duration_seconds"]))
        check("representative-frame rule came from config", mon._representative_frame_rule == cap["representative_frame_rule"])
        check("embedding cap came from config", mon._max_embedding_bytes == settings.max_embedding_bytes())
        check("visual directory resolved through the loader", str(mon._visual_dir) == settings.path("detections_visual_dir"))


def main():
    settings, cfg_path = make_pi_settings()
    try:
        print("loader role:", settings.node_role, "| active station:", settings.active_station()["station_name"])
        scenario_single_track(settings)
        scenario_no_detection(settings)
        scenario_long_track_segments(settings)
        scenario_brief_occlusion(settings)
        scenario_two_animals(settings)
        scenario_json_fixture(settings)
        scenario_tuning_from_config(settings)
        clean_visual_dir(settings)  # leave the tree clean
    finally:
        if cfg_path.exists():
            cfg_path.unlink()

    print(f"\n==== {CHECKS['passed']} passed, {CHECKS['failed']} failed ====")
    return 0 if CHECKS["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
