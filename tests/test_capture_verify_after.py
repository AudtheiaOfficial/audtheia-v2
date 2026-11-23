"""Desktop capture detects alone, and its events are verified once it ends.

Path: tests/test_capture_verify_after.py

The desktop verifier and the language model are large and run on the same
processor as detection. Loading them at the start of every capture delayed the
start, and verifying while frames were still arriving slowed detection. These
checks prove that:

  - A capture station loads no desktop verifier, no sound verifier and no
    language model.
  - While a capture runs, its events are quality-controlled but not verified.
  - When the source runs out, the capture reports as finished, quality control
    runs once more, and the step after capture is called exactly once.
  - Stopping a capture also calls the step after capture once.
  - Verification after capture repeats until no event is left waiting, so
    events written while it ran are not missed.
  - The capture status reports stations that are starting and those whose last
    capture is being verified.

The monitor and the models are stand-ins and the database is temporary; nothing
in the repository's data, database or reports folders is touched.
"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from audtheia.app import orchestrator as orch  # noqa: E402
from audtheia.app.orchestrator import DesktopStation, NullInterpreter, NullVerifier  # noqa: E402
from audtheia.config import load_settings  # noqa: E402
from audtheia.storage.database import Database, Station  # noqa: E402

_checks = 0


def check(label: str, condition: bool) -> None:
    global _checks
    if not condition:
        raise AssertionError(label)
    _checks += 1
    print(f"  ok  {label}")


class FakeMonitor:
    """A capture that runs until its source ends after `seconds`, or until stopped."""

    def __init__(self, seconds: float):
        self._seconds = seconds
        self._stop = threading.Event()
        self._producer_thread = None

    def _run(self):
        self._stop.wait(self._seconds)

    def start(self):
        self._producer_thread = threading.Thread(target=self._run, daemon=True)
        self._producer_thread.start()

    def stop(self):
        self._stop.set()


def station(tmp: Path):
    settings = load_settings()
    settings.raw["paths"]["db_path"] = str(tmp / "audtheia.db")
    st = settings.stations()[0]
    db = Database(settings.db_path(), **settings.database_kwargs())
    db.initialize_schema(settings.schema_path())
    db.create_station(Station(id=st["station_id"], station_name=st["station_name"],
                              environment_type=st["environment_type"], created_at="2026-01-01T00:00:00Z"))
    return settings, st, db


def counted(desktop: DesktopStation) -> dict:
    """Count the station's quality-control and verification calls."""
    calls = {"qc": 0, "verify": 0}

    def qc():
        calls["qc"] += 1
        return 0

    def verify():
        calls["verify"] += 1
        return 0

    desktop.qc_pending = qc
    desktop.verify_pending = verify
    desktop.dream_once = lambda: None
    desktop.report_once = lambda: None
    desktop._ensure_station_registered = lambda: None
    return calls


def run() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        settings, st, db = station(tmp)

        print("\n[1] A capture station loads only what detects")
        loaded = []
        original = (DesktopStation._build_verifier, DesktopStation._build_acoustic_verifier,
                    DesktopStation._build_llm_backends)
        DesktopStation._build_verifier = staticmethod(lambda *a, **k: loaded.append("verifier"))
        DesktopStation._build_acoustic_verifier = staticmethod(lambda *a, **k: loaded.append("sound"))
        DesktopStation._build_llm_backends = staticmethod(lambda *a, **k: loaded.append("llm"))
        try:
            cap = DesktopStation.build_capture(settings, station_id=st["station_id"])
        finally:
            (DesktopStation._build_verifier, DesktopStation._build_acoustic_verifier,
             DesktopStation._build_llm_backends) = (staticmethod(original[0]), staticmethod(original[1]),
                                                    staticmethod(original[2]))
        check("no verifier, sound verifier or language model is loaded", loaded == [])
        check("the capture station has no configured verifier", cap.verifier_configured is False)

        print("\n[2] A capture whose source runs out")
        desktop = DesktopStation(settings=settings, station=st, db=db, verifier=NullVerifier(),
                                 interpreter=NullInterpreter())
        calls = counted(desktop)
        finished = []
        desktop._build_monitor = lambda: FakeMonitor(0.6)
        live = desktop.start_background(verify_interval_seconds=0.1, verify=False,
                                        on_finish=lambda: finished.append(time.time()))
        time.sleep(0.3)
        check("while it runs, it reports as running", live.running())
        check("while it runs, events are quality-controlled", calls["qc"] >= 1)
        check("while it runs, nothing is verified", calls["verify"] == 0)
        for _ in range(50):
            if finished:
                break
            time.sleep(0.1)
        check("when the source ends, the step after capture is called once", len(finished) == 1)
        check("when the source ends, the capture reports as finished", not live.running())
        check("verification never ran during the capture", calls["verify"] == 0)
        live.stop()
        time.sleep(0.2)
        check("stopping a finished capture does not call the step after capture again", len(finished) == 1)

        print("\n[3] A capture that is stopped")
        desktop2 = DesktopStation(settings=settings, station=st, db=db, verifier=NullVerifier(),
                                  interpreter=NullInterpreter())
        calls2 = counted(desktop2)
        finished2 = []
        desktop2._build_monitor = lambda: FakeMonitor(60.0)
        live2 = desktop2.start_background(verify_interval_seconds=0.1, verify=False,
                                          on_finish=lambda: finished2.append(1))
        time.sleep(0.25)
        qc_before = calls2["qc"]
        live2.stop()
        for _ in range(50):
            if finished2:
                break
            time.sleep(0.1)
        check("stopping calls the step after capture once", finished2 == [1])
        check("quality control runs once more after the capture stops", calls2["qc"] > qc_before)
        check("a stopped capture reports as not running", not live2.running())

        print("\n[4] Verification after capture")
        desktop3 = DesktopStation(settings=settings, station=st, db=db, verifier=NullVerifier(),
                                  interpreter=NullInterpreter())
        rounds = iter([3, 2, 0])
        seen = {"qc": 0}
        desktop3.qc_pending = lambda: seen.__setitem__("qc", seen["qc"] + 1) or 0
        desktop3.verify_pending = lambda: next(rounds)
        check("it repeats until nothing is left waiting and counts every record",
              desktop3.verify_after_capture() == 5)
        check("each round quality-controls first", seen["qc"] == 3)

        print("\n[5] The capture status")
        from fastapi.testclient import TestClient
        from audtheia.app import server as srv

        client = TestClient(srv.create_app(settings, db))
        with srv._capture_lock:
            srv._CAPTURE_STARTING.add(st["station_id"])
        try:
            status = client.get("/api/capture/status").json()
        finally:
            with srv._capture_lock:
                srv._CAPTURE_STARTING.discard(st["station_id"])
        check("a station whose source is opening is reported as starting",
              status.get("starting") == [st["station_id"]])
        check("the status reports verifying and audio states",
              "verifying" in status and "starting_audio" in status and "audio_start_errors" in status)

    print(f"\nAll {_checks} capture checks passed.")


if __name__ == "__main__":
    try:
        run()
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
