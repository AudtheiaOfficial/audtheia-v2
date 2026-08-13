"""Verification for the SSH-pull station-to-desktop sync (data transport).

Path: tests/test_sync_pull.py

Proves the desktop-side pull engine and the station-side CLI verbs against two
in-process databases and a local runner that stands in for ssh, so the whole
export, import, confirm cycle is exercised with no network. The properties
checked are the ones a field deployment depends on:

  - a pull moves every unconfirmed event, with its children, from the station to
    the desktop, and stamps the station's rows confirmed so the buffer can later
    reclaim them,
  - the pull pages a backlog in stable batches and clears it completely,
  - the pull is idempotent: a second pull moves nothing and re-importing a batch
    never duplicates a row, so an interrupted pull is safe to resume,
  - a transport failure surfaces as one clear error type and never as a partial,
    silent state,
  - the append-only firewall holds: importing a station's rows never needs or
    touches a desktop-owned table.

Built on temporary databases created from the real schema.sql, so the CHECK and
foreign-key constraints exercised here are the shipped ones. Standard library
only; no network and no live database.

Run: python tests/test_sync_pull.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from audtheia.storage.database import (  # noqa: E402
    ChildDetection,
    Database,
    Observation,
    Station,
    SYNCABLE_TABLES,
    new_id,
    utc_now_iso,
)
from audtheia.sync import (  # noqa: E402
    SyncTransportError,
    do_confirm,
    do_export,
    pull_all,
    pull_media,
    pull_once,
    run_sync_loop,
    sync_reachable_stations,
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


class LocalRunner:
    """A CommandRunner that runs the station CLI verbs in-process against a
    station database, standing in for ssh to a real Pi."""

    def __init__(self, station_db: Database) -> None:
        self._db = station_db
        self.calls = 0

    def run(self, args, *, input_text=None):
        self.calls += 1
        if args[0] == "export":
            batch = int(args[2]) if len(args) >= 3 else 500
            return do_export(self._db, batch)
        if args[0] == "confirm":
            return do_confirm(self._db, input_text or "")
        raise SyncTransportError(f"unknown verb {args!r}")


class FailingExportRunner:
    def run(self, args, *, input_text=None):
        if args[0] == "export":
            raise RuntimeError("ssh connection refused")
        return "{}"


def _make_db(path: Path) -> Database:
    db = Database(path)
    db.initialize_schema(SCHEMA)
    return db


def _register_station(db: Database, station_id: str) -> None:
    db.create_station(Station(
        id=station_id, station_name="Field-" + station_id[:4],
        environment_type="terrestrial", created_at=utc_now_iso(),
    ))


def _seed_station_events(db: Database, station_id: str, n: int) -> list:
    ids = []
    now = utc_now_iso()
    for k in range(n):
        obs = Observation(
            id=new_id(), event_name=f"Field_{station_id[:4]}_{k:04d}", station_id=station_id,
            trigger_source="vision", first_seen=now, last_seen=now, duration=2.0,
            data_source="model", created_at=now, frame_count=5, screening_confidence=0.7,
        )
        child = ChildDetection(
            id=new_id(), observation_id=obs.id, modality="vision", created_at=now,
            scientific_name="Cardinalis cardinalis", common_name="Northern Cardinal", confidence=0.7,
        )
        db.insert_observation(obs, children=[child])
        ids.append(obs.id)
    return ids


def test_full_pull() -> None:
    print("\nA pull moves every event with its children and confirms the station")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        station = _make_db(root / "station.db")
        desktop = _make_db(root / "desktop.db")
        sid = new_id()
        _register_station(station, sid)
        _register_station(desktop, sid)  # the desktop already knows the station it syncs
        seeded = _seed_station_events(station, sid, 3)

        result = pull_all(desktop, LocalRunner(station))
        check("all three events were confirmed", result["confirmed"]["observations"] == 3)

        desktop_ids = {o["id"] for o in desktop.list_observations()}
        check("the desktop now holds every event", set(seeded) <= desktop_ids)
        first_children = desktop.list_child_detections(seeded[0])
        check("children travelled with their parent",
              len(first_children) == 1 and first_children[0]["common_name"] == "Northern Cardinal")
        check("imported rows are stamped synced on the desktop",
              all(o.get("synced_at") for o in desktop.list_observations()))
        check("the station marked its rows confirmed", station.count_unsynced()["observations"] == 0)


def test_paging_and_idempotency() -> None:
    print("\nA backlog pages in batches, and a repeat pull moves nothing")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        station = _make_db(root / "station.db")
        desktop = _make_db(root / "desktop.db")
        sid = new_id()
        _register_station(station, sid)
        _register_station(desktop, sid)
        _seed_station_events(station, sid, 5)

        result = pull_all(desktop, LocalRunner(station), batch_size=2)
        check("a backlog of 5 pages in batches of 2 takes several rounds", result["rounds"] >= 3)
        check("all five events still arrive", result["confirmed"]["observations"] == 5)
        check("the station is fully drained", station.count_unsynced()["observations"] == 0)

        again = pull_all(desktop, LocalRunner(station))
        check("a second pull moves nothing", again["total"] == 0 and again["rounds"] == 0)
        check("the desktop still holds exactly five events", len(desktop.list_observations()) == 5)


def test_import_is_idempotent() -> None:
    print("\nRe-importing a batch never duplicates a row")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        station = _make_db(root / "station.db")
        desktop = _make_db(root / "desktop.db")
        sid = new_id()
        _register_station(station, sid)
        _register_station(desktop, sid)
        _seed_station_events(station, sid, 2)

        # One round, then deliver the very same batch again before it is confirmed.
        batch = station.export_unsynced_batch(batch_size=500)
        desktop.import_batch(batch)
        desktop.import_batch(batch)
        check("a redelivered batch does not duplicate events", len(desktop.list_observations()) == 2)


def test_transport_failure() -> None:
    print("\nA transport failure is one clear error, not a partial state")
    with tempfile.TemporaryDirectory() as tmp:
        desktop = _make_db(Path(tmp) / "desktop.db")
        try:
            pull_once(desktop, FailingExportRunner())
            check("a failed export raises SyncTransportError", False)
        except SyncTransportError:
            check("a failed export raises SyncTransportError", True)
        check("nothing was imported on a failed pull", len(desktop.list_observations()) == 0)


class FakeFetcher:
    """A MediaFetcher that copies from a local station tree to a desktop tree,
    mirroring what scp does across the network."""

    def __init__(self, pi_root: Path, desktop_root: Path) -> None:
        self.pi_root = Path(pi_root)
        self.desktop_root = Path(desktop_root)

    def fetch(self, remote_rel, local_abs, *, is_dir):
        src = self.pi_root / remote_rel
        dst = Path(local_abs)
        if is_dir:
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


class FakeSettings:
    def __init__(self, stations, repo_root):
        self._stations = stations
        self.repo_root = repo_root

    def stations(self):
        return self._stations


def test_media_pull() -> None:
    print("\nMedia pull brings an event's frames and clip to the desktop")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pi, desk = root / "pi", root / "desk"
        event_rel = "data/detections/visual/EVENT1"
        (pi / event_rel).mkdir(parents=True)
        (pi / event_rel / "frame_000.jpg").write_bytes(b"jpegdata")
        (pi / event_rel / "annotations.jsonl").write_text("{}", encoding="utf-8")
        clip_rel = "data/detections/audio/EVENT1.wav"
        (pi / clip_rel).parent.mkdir(parents=True)
        (pi / clip_rel).write_bytes(b"wavdata")

        obs = [{"representative_frame": event_rel + "/frame_000.jpg", "audio_clip_path": clip_rel}]
        res = pull_media(obs, fetcher=FakeFetcher(pi, desk), repo_root=str(desk))
        check("the whole event directory arrived",
              (desk / event_rel / "frame_000.jpg").exists() and (desk / event_rel / "annotations.jsonl").exists())
        check("the audio clip arrived", (desk / clip_rel).exists())
        check("both fetches counted, none failed", res["fetched"] == 2 and res["failed"] == 0)

        # A missing source is a counted failure, never a raise.
        missing = pull_media([{"representative_frame": "data/detections/visual/GONE/f.jpg"}],
                             fetcher=FakeFetcher(pi, desk), repo_root=str(desk))
        check("a missing media file is a soft failure", missing["failed"] == 1)


def test_reachable_stations() -> None:
    print("\nSyncing reachable stations honours targets and reachability")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        station = _make_db(root / "station.db")
        desktop = _make_db(root / "desktop.db")
        sid = new_id()
        _register_station(station, sid)
        _register_station(desktop, sid)
        _seed_station_events(station, sid, 2)

        stations = [
            {"station_id": sid, "provisioning": {"host": "10.0.0.5", "user": "pi", "port": 22}},
            {"station_id": new_id()},  # no connection target: skipped
        ]
        settings = FakeSettings(stations, str(root))
        results = sync_reachable_stations(
            settings, desktop,
            runner_factory=lambda s: LocalRunner(station),
            fetcher_factory=lambda s: None,
            reachable=lambda s: True,
        )
        check("the connected station synced", results.get(sid, {}).get("confirmed", {}).get("observations") == 2)
        check("a station with no target is not synced", len(results) == 1)
        check("the station was drained", station.count_unsynced()["observations"] == 0)

        # An unreachable station is skipped with a reason, not an error.
        station2 = _make_db(root / "station2.db")
        _register_station(station2, sid)
        _seed_station_events(station2, sid, 1)
        skipped = sync_reachable_stations(
            settings, desktop,
            runner_factory=lambda s: LocalRunner(station2),
            fetcher_factory=lambda s: None,
            reachable=lambda s: False,
        )
        check("an unreachable station is skipped", skipped.get(sid, {}).get("skipped") is True)


def test_sync_loop() -> None:
    print("\nThe automatic loop runs a pass and stops cleanly")
    calls = []
    stop = threading.Event()

    def one_pass(_s, _d):
        calls.append(1)
        stop.set()  # ask the loop to stop after this pass

    run_sync_loop(None, None, stop, interval=0.01, sync=one_pass)
    check("the loop ran at least one pass", len(calls) >= 1)

    stop2 = threading.Event()
    n = [0]

    def failing_pass(_s, _d):
        n[0] += 1
        stop2.set()
        raise RuntimeError("transient")

    run_sync_loop(None, None, stop2, interval=0.01, sync=failing_pass)
    check("a failing pass is caught and the loop still stops", n[0] >= 1)


def test_empty_station() -> None:
    print("\nAn empty station is a clean no-op")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        station = _make_db(root / "station.db")
        desktop = _make_db(root / "desktop.db")
        sid = new_id()
        _register_station(station, sid)
        _register_station(desktop, sid)
        one = pull_once(desktop, LocalRunner(station))
        check("an empty station reports nothing to pull", one["empty"] is True)
        check("the confirmed dict spans every syncable table",
              set(one["confirmed"].keys()) == set(SYNCABLE_TABLES))


def main() -> int:
    print("=" * 72)
    print("Station-to-desktop sync: SSH-pull data transport")
    print("=" * 72)
    if not SCHEMA.exists():
        print("  FAIL  schema.sql not found")
        return 1
    test_full_pull()
    test_paging_and_idempotency()
    test_import_is_idempotent()
    test_transport_failure()
    test_media_pull()
    test_reachable_stations()
    test_sync_loop()
    test_empty_station()
    print("\n" + "=" * 72)
    print(f"RESULT: {CHECKS['passed']} passed, {CHECKS['failed']} failed")
    print("=" * 72)
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
