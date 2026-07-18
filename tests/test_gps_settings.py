"""Interface checks for station coordinates and the saved color theme.

Path: tests/test_gps_settings.py

Confirms the two settings paths added for the location view and the remembered
theme, through the real backend with a temporary database and a temporary copy
of the configuration:

  - A station's fixed coordinates can be set, cleared, and read back, and an
    out-of-range value is refused with the whole batch left unsaved.
  - The color theme is saved on the hub through the same guarded path the rest of
    the settings use, and comes back in the configuration a fresh page reads.
  - A saved location status of "station_configured" flows through the location
    endpoint unchanged, so an entered position is never reported as a live fix.

The web framework is imported inside the run so a machine without it reports a
skip rather than a failure, matching how the rest of the suite treats optional
libraries.
"""

from __future__ import annotations

import json
import sys
import tempfile
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from audtheia.config import load_settings  # noqa: E402
from audtheia.storage.database import Database  # noqa: E402
from audtheia.app import server as srv  # noqa: E402

CHECKS = {"passed": 0, "failed": 0}


def check(label: str, condition: bool) -> None:
    if condition:
        CHECKS["passed"] += 1
        print(f"  PASS  {label}")
    else:
        CHECKS["failed"] += 1
        print(f"  FAIL  {label}")


def _make_app(work: Path):
    raw = json.loads((REPO / "config" / "settings.json").read_text(encoding="utf-8"))
    raw["paths"]["db_path"] = str(work / "audtheia.db")
    raw["paths"]["schema_path"] = str((REPO / "audtheia" / "storage" / "schema.sql").resolve())
    settings_path = work / "settings.json"
    settings_path.write_text(json.dumps(raw), encoding="utf-8")
    settings = load_settings(settings_path)
    db = Database(settings.db_path())
    db.initialize_schema(settings.schema_path())
    return settings, settings_path, db, raw


def run() -> int:
    print("=" * 72)
    print("Station coordinates and saved theme: interface checks")
    print("=" * 72)
    try:
        from fastapi.testclient import TestClient
    except Exception:  # noqa: BLE001 - a missing web framework is a skip, not a failure
        print("  SKIP  the web framework is not installed")
        return 0

    work = Path(tempfile.mkdtemp(prefix="audtheia-gps-settings-"))
    settings, settings_path, db, raw = _make_app(work)
    app = srv.create_app(settings, db)
    client = TestClient(app)

    station_id = raw["stations"][1]["station_id"]  # the station whose coordinates start unset

    print("\n[1] A station's coordinates and the theme save in one batch")
    resp = client.post("/api/settings/update", json={"changes": [
        {"scope": "station", "station_id": station_id, "field": "station_latitude", "value": 18.4655},
        {"scope": "station", "station_id": station_id, "field": "station_longitude", "value": -66.1057},
        {"scope": "station", "station_id": station_id, "field": "station_elevation", "value": 25},
        {"scope": "global", "field": "ui_theme", "value": "cyberpunk"},
        {"scope": "global", "field": "ui_last_dark", "value": "cyberpunk"},
        {"scope": "global", "field": "ui_last_light", "value": "forest"},
    ]})
    check("the batch is accepted", resp.status_code == 200)
    cfg = resp.json().get("config", {}) if resp.status_code == 200 else {}
    station = next((s for s in cfg.get("stations", []) if s["station_id"] == station_id), {})
    check("coordinates are stored", station.get("location") == {"latitude": 18.4655, "longitude": -66.1057, "elevation": 25})
    check("the theme is stored", cfg.get("ui", {}).get("theme") == "cyberpunk")

    on_disk = json.loads(settings_path.read_text(encoding="utf-8"))
    check("the changes are written to the configuration file", on_disk.get("ui", {}).get("theme") == "cyberpunk")

    print("\n[2] An out-of-range coordinate is refused and nothing is saved")
    bad = client.post("/api/settings/update", json={"changes": [
        {"scope": "station", "station_id": station_id, "field": "station_latitude", "value": 120},
    ]})
    check("a latitude past 90 is rejected", bad.status_code == 422)
    reread = json.loads(settings_path.read_text(encoding="utf-8"))
    still = next((s for s in reread["stations"] if s["station_id"] == station_id), {})
    check("the earlier valid latitude is untouched", still.get("location", {}).get("latitude") == 18.4655)

    print("\n[3] A coordinate can be cleared")
    cleared = client.post("/api/settings/update", json={"changes": [
        {"scope": "station", "station_id": station_id, "field": "station_elevation", "value": None},
    ]})
    check("clearing elevation is accepted", cleared.status_code == 200)
    station2 = next((s for s in cleared.json()["config"]["stations"] if s["station_id"] == station_id), {})
    check("elevation is now empty while the position remains", station2["location"]["elevation"] is None
          and station2["location"]["latitude"] == 18.4655)

    print("\n[4] The new fields are exposed and the location endpoint passes the status through")
    got = client.get("/api/settings").json()
    check("the configuration includes the saved theme", "ui" in got.get("config", {}))
    editable = got.get("editable_fields", {})
    check("the theme and coordinate fields are editable",
          "ui_theme" in editable.get("global", []) and "station_latitude" in editable.get("station", []))

    # An entered position stored on an observation is reported as such, not as a fix.
    real_station_id = raw["stations"][0]["station_id"]
    from audtheia.storage.database import Station, utc_now_iso  # noqa: E402
    db.create_station(Station(id=real_station_id, station_name=raw["stations"][0]["station_name"],
                              environment_type=raw["stations"][0]["environment_type"], created_at=utc_now_iso()))
    with db.connect() as conn:
        oid = str(uuid.uuid4())
        conn.execute(
            """INSERT INTO observations
               (id, event_name, station_id, trigger_source, first_seen, last_seen, duration,
                time_provisional, data_source, gps_latitude, gps_longitude, gps_status, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (oid, f"GpsTest_{oid[:8]}", real_station_id, "vision", "2026-07-17T00:00:00Z",
             "2026-07-17T00:00:01Z", 1.0, 0, "model", 18.21, -67.15, "station_configured", "2026-07-17T00:00:02Z"),
        )
    rows = client.get("/api/gps").json()
    match = [r for r in rows if r.get("observation_id") == oid]
    check("the located observation is returned", len(match) == 1)
    check("its entered-position status is passed through unchanged",
          bool(match) and match[0].get("gps_status") == "station_configured")

    print("\n" + "=" * 72)
    print(f"RESULT: {CHECKS['passed']} passed, {CHECKS['failed']} failed")
    print("=" * 72)
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    sys.exit(run())
