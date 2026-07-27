"""Verification for the guided species-data setup (index build and reference fetch).

Path: tests/test_species_setup.py

Two setup steps that used to require a command line are now interface actions,
each running as a background job the interface polls. This checks the new server
code, entirely offline:

  - the index status endpoint reports whether the index and backbone are present,
  - the build endpoint builds the taxonomic index from a tiny synthetic backbone
    in the background, its status reaches done, and a species search then works,
  - a second build without force is refused, and a build is desktop-only,
  - the reference status endpoint reports what a fetch would cover and which
    credential is set,
  - the reference fetch endpoint runs the fetch script through the injectable
    client seam and reaches done without any network call.

The real 2.4 GB backbone and the live GBIF and IUCN calls are never touched: the
index is built from a handful of synthetic rows in the real column layout, and
the fetch runs through a stand-in client. Standard library plus the interface
test client only.

Run: python tests/test_species_setup.py
"""

from __future__ import annotations

import json
import sys
import time
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from audtheia.storage.database import Database  # noqa: E402

SCHEMA = REPO / "audtheia" / "storage" / "schema.sql"
CHECKS = {"passed": 0, "failed": 0}


def check(label: str, condition: bool) -> None:
    if condition:
        CHECKS["passed"] += 1
        print(f"  PASS  {label}")
    else:
        CHECKS["failed"] += 1
        print(f"  FAIL  {label}")


def _backbone_row(usage_key, parent_key, status, rank, scientific, canonical):
    """One tab-delimited backbone row in the real simple.txt column layout."""
    cols = [""] * 20
    cols[0] = str(usage_key)
    cols[1] = str(parent_key)
    cols[4] = status
    cols[5] = rank
    cols[18] = scientific
    cols[19] = canonical
    return "\t".join(cols)


def _write_backbone(backbone_dir: Path) -> None:
    backbone_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _backbone_row(1, 0, "ACCEPTED", "SPECIES", "Junco hyemalis (Linnaeus, 1758)", "Junco hyemalis"),
        _backbone_row(2, 0, "ACCEPTED", "SPECIES", "Cyanocitta cristata (Linnaeus, 1758)", "Cyanocitta cristata"),
        # A genus row, which the builder must drop (species rank only).
        _backbone_row(3, 0, "ACCEPTED", "GENUS", "Junco Wagler, 1831", "Junco"),
        # A synonym resolving to the accepted Junco hyemalis.
        _backbone_row(4, 1, "SYNONYM", "SPECIES", "Junco oreganus", "Junco oreganus"),
    ]
    (backbone_dir / "simple.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _make_settings(tmp: Path, *, role: str = "desktop"):
    from audtheia.config import load_settings
    base = json.loads((REPO / "config" / "settings.json").read_text(encoding="utf-8"))
    base["node"]["role"] = role
    base["node"]["active_station_id"] = None
    base["paths"]["gbif_backbone_path"] = str((tmp / "backbone").resolve())
    base["paths"]["db_path"] = str((tmp / "setup.db").resolve())
    base["paths"]["data_dir"] = str((tmp / "data").resolve())
    base["paths"]["reports_dir"] = str((tmp / "reports_out").resolve())
    path = tmp / "settings.setup.json"
    path.write_text(json.dumps(base), encoding="utf-8")
    return load_settings(path)


def _client(settings, db):
    from fastapi.testclient import TestClient
    from audtheia.app import server as srv
    return TestClient(srv.create_app(settings, db)), srv


def _poll_until_done(client, path, tries=40, delay=0.25):
    for _ in range(tries):
        job = client.get(path).json()["job"]
        if job.get("status") in ("done", "error"):
            return job
        time.sleep(delay)
    return client.get(path).json()["job"]


def test_index_build(tmp: Path) -> None:
    print("\nThe index build endpoint builds in the background and search then works")
    tmp.mkdir(parents=True, exist_ok=True)
    _write_backbone(tmp / "backbone")
    settings = _make_settings(tmp)
    db = Database(settings.db_path())
    db.initialize_schema(SCHEMA)
    client, srv = _client(settings, db)

    before = client.get("/api/species/index/status").json()
    check("status sees the backbone present", before["backbone_present"] is True)
    check("status sees no index yet", before["index_present"] is False)

    started = client.post("/api/species/index/build")
    check("the build starts with 202", started.status_code == 202)

    job = _poll_until_done(client, "/api/species/index/status")
    check("the build reaches done", job["status"] == "done")
    after = client.get("/api/species/index/status").json()
    check("the index is now present", after["index_present"] is True)
    # Two species rows kept, the genus dropped, the synonym resolved: three names.
    check("the index kept the species and the resolved synonym", after["index_names"] == 3)

    found = client.get("/api/species/search", params={"q": "junco"}).json()
    check("search returns from the freshly built index", found["index_available"] is True and len(found["results"]) >= 1)

    again = client.post("/api/species/index/build")
    check("a second build without force is refused", again.status_code == 409)
    forced = client.post("/api/species/index/build", params={"force": True})
    check("a forced rebuild is accepted", forced.status_code == 202)
    _poll_until_done(client, "/api/species/index/status")


def test_reference_fetch(tmp: Path) -> None:
    print("\nThe reference fetch endpoint runs the fetch through the client seam, offline")
    tmp.mkdir(parents=True, exist_ok=True)
    _write_backbone(tmp / "backbone")
    settings = _make_settings(tmp)
    db = Database(settings.db_path())
    db.initialize_schema(SCHEMA)

    # Seed a reference and a matching, unstamped observation, so the fetch's
    # backfill has an existing record to complete.
    from audtheia.storage.database import (
        Station, Observation, ChildDetection, SpeciesReference, new_id, utc_now_iso,
    )
    now = utc_now_iso()
    st = Station(id=new_id(), station_name="Ref", environment_type="marine", created_at=now)
    db.create_station(st)
    obs = Observation(
        id=new_id(), event_name="ref-" + new_id()[:8], station_id=st.id, trigger_source="vision",
        first_seen=now, last_seen=now, duration=1.0, data_source="model", created_at=now,
        frame_count=1, screening_confidence=0.9)
    db.insert_observation(obs, children=[ChildDetection(
        id=new_id(), observation_id=obs.id, modality="vision", created_at=now,
        scientific_name="Aplysina fistularis", confidence=0.9)])
    db.upsert_species_reference(SpeciesReference(
        gbif_usage_key="9999", scientific_name="Aplysina fistularis", fetched_at=now,
        gbif_snapshot_date="2026-01-01", iucn_fetch_date="2026-01-02"))

    client, srv = _client(settings, db)

    status = client.get("/api/species/reference/status").json()
    check("reference status reports the store size", "references_stored" in status)
    check("reference status reports the target species list", isinstance(status["target_species"], list))
    check("reference status reports whether an IUCN token is set", "iucn_token_present" in status)

    # A stand-in client so the fetch never reaches the network. With no configured
    # target species the fetch stores nothing, but the endpoint, the background
    # job, the script load, and the client seam are all exercised end to end.
    class _StubClient:
        def get_json(self, url, params=None, headers=None):
            return None

    srv._species_fetch_client_factory = lambda: _StubClient()
    try:
        started = client.post("/api/species/reference/fetch")
        check("the fetch starts with 202", started.status_code == 202)
        job = _poll_until_done(client, "/api/species/reference/status")
        check("the fetch reaches done without a network call", job["status"] == "done")
        result = job.get("result") or {}
        check("the fetch reports the four outcome counts",
              all(k in result for k in ("fetched", "cached", "unmatched", "failed")))
        # The backfill runs after the fetch and completes the seeded record.
        check("the fetch reports how many existing records it stamped",
              result.get("stamped_existing") == 1)
        check("the seeded observation is now stamped with the snapshot date",
              db.get_observation(obs.id)["gbif_snapshot_date"] == "2026-01-01")
    finally:
        srv._species_fetch_client_factory = None


def main() -> int:
    print("=" * 72)
    print("Species-data setup: guided index build and reference fetch")
    print("=" * 72)
    if not SCHEMA.exists():
        print("  FAIL  schema.sql not found")
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        test_index_build(root / "build")
        test_reference_fetch(root / "ref")
    print("\n" + "=" * 72)
    print(f"RESULT: {CHECKS['passed']} passed, {CHECKS['failed']} failed")
    print("=" * 72)
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
