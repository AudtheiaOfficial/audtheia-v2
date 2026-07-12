"""Desktop orchestrator: run a whole station on one computer, no field hardware.

Path: audtheia/app/orchestrator.py

A field station and the desktop hub normally divide the work: the station
captures and quality-controls, the desktop verifies, dreams, and reports. This
module lets one desktop computer play both parts, so a person can clone the
project and run the entire pipeline against an ordinary webcam, stream, or video
file with no Raspberry Pi at all.

It composes the parts that already exist, each through its own interface, and
adds nothing to them. Capture runs the detection loop over the desktop video
source; quality control finalizes each observation; verification re-scores the
event through the desktop model and opens the gate; the dream pass discovers
longitudinal patterns; and reports are generated on their schedule. The language
model is optional: the dream pass narrates and clusters without one, and
verification runs with a plain interpreter that adds no points, so a station runs
end to end with no language model present and gains richer interpretation the
moment one is added.

Each backend (the verifier, the interpreter, and the optional dream narrator and
clusterer) is an argument, so the whole chain can be exercised against scripted
stand-ins with no model files. The build classmethod wires the real desktop
backends, degrading to a verifier that finds nothing when the verification model
is not present yet, so the pipeline always flows.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

from audtheia.storage.database import Database
from audtheia.analysis.observation import QCEngine, QC_PENDING
from audtheia.analysis.verify import VerifyEngine
from audtheia.analysis.dream import DreamEngine
from audtheia.reports.generate import generate_report
from audtheia.pipeline.monitor import Monitor, NullTriggerSink, build_tracker_from_capture

logger = logging.getLogger("audtheia.app.orchestrator")

__all__ = ["DesktopStation", "NullVerifier", "NullInterpreter", "DEFAULT_VERIFY_INTERVAL_SECONDS"]

# How many recent records a single quality-control sweep examines. Records
# awaiting control are the newest, which this ordering reaches first, so the
# bound keeps a sweep cheap while never missing a fresh record.
DEFAULT_SWEEP_SCAN_LIMIT = 1000

# How often the live loop verifies newly captured observations. Verification is
# cheap and event-shaped, so a short interval keeps the interface current without
# busy-waiting.
DEFAULT_VERIFY_INTERVAL_SECONDS = 10.0

# Schedule words mapped to a plain interval in seconds, for the live loop's dream
# and report cadence. A schedule the desktop does not recognize falls back to
# daily, so an unusual value still runs rather than never running.
_SCHEDULE_SECONDS = {
    "hourly": 3600,
    "daily": 86400,
    "weekly": 604800,
    "biweekly": 1209600,
    "monthly": 2592000,
}


class NullVerifier:
    """A verifier that finds nothing, used when the verification model is not
    present yet. Verification still runs and records a no-detection verdict, so
    the pipeline flows and the gate simply stays closed until a real model is in
    place."""

    version = None

    def verify_frames(self, frame_paths) -> list:
        return [{"gbif_usage_key": None, "scientific_name": None, "confidence": None} for _ in frame_paths]


class NullInterpreter:
    """An interpreter that adds no interpretive points, so verification runs with
    no language model. The desktop language-model interpreter is a drop-in
    replacement that fills these in downstream."""

    version = None

    def interpret(self, context) -> list:
        return []


class DesktopStation:
    """One station captured and analyzed on a single desktop computer."""

    def __init__(
        self,
        *,
        settings,
        station: dict,
        db: Database,
        verifier,
        interpreter,
        narrator=None,
        clusterer=None,
    ) -> None:
        self._settings = settings
        self._station = station
        self._db = db
        self._verifier = verifier
        self._interpreter = interpreter
        self._narrator = narrator
        self._clusterer = clusterer
        self._station_id = station["station_id"]

    # -- construction ----------------------------------------------------

    @classmethod
    def build(cls, settings, *, station_id: Optional[str] = None) -> "DesktopStation":
        """Wire a desktop station from configuration and the real backends.

        The verification model is loaded when present; when it is not, a verifier
        that finds nothing is used so the pipeline still runs. The interpreter and
        the dream narrator and clusterer default to the no-language-model path.
        """
        station = cls._resolve_station(settings, station_id)
        db = Database(settings.db_path(), **settings.database_kwargs())
        verifier = cls._build_verifier(settings)
        interpreter, narrator = cls._build_llm_backends(settings)
        return cls(
            settings=settings,
            station=station,
            db=db,
            verifier=verifier,
            interpreter=interpreter,
            narrator=narrator,
        )

    @staticmethod
    def _resolve_station(settings, station_id: Optional[str]) -> dict:
        if station_id is not None:
            return settings.station(station_id)
        for station in settings.stations():
            if settings.capture_source(station).get("video"):
                return station
        raise ValueError(
            "no desktop capture station is configured; set capture.source.video on a "
            "station (for example 'webcam:0', 'url:rtsp://...', 'stream:https://...', "
            "or 'file:/path/to/clip.mp4')."
        )

    @staticmethod
    def _build_verifier(settings):
        from audtheia.inference.rfdetr_onnx import build_verifier

        try:
            return build_verifier(settings)
        except Exception as exc:  # noqa: BLE001 - a missing model must not stop the desktop
            logger.warning(
                "the RF-DETR verification model is not available (%s); using a verifier "
                "that finds nothing, so the pipeline runs and the gate stays closed until "
                "a model is placed.",
                exc,
            )
            return NullVerifier()

    @staticmethod
    def _build_llm_backends(settings):
        """Wire the desktop language model into an interpreter and a narrator.

        One GGUF model is loaded once and shared, so verification interpretation
        and dream narration do not each load their own copy. When the model is
        absent, or its runtime is not installed, this degrades to an interpreter
        that adds no points and no narrator, so the pipeline still runs and gains
        interpretation and narration the moment a model is placed. The clustering
        backend is left unset, so the dream pass simply contributes no novel
        groupings until one is provided.
        """
        try:
            from audtheia.inference.gguf_llm import (
                load_completer,
                build_interpreter,
                build_narrator,
            )

            completer = load_completer(settings)
        except Exception as exc:  # noqa: BLE001 - a missing model must not stop the desktop
            logger.warning(
                "the desktop language model is not available (%s); running without "
                "interpretation and narration until a GGUF model is placed.",
                exc,
            )
            return NullInterpreter(), None

        interpreter = build_interpreter(settings, completer=completer)
        narrator = build_narrator(settings, completer=completer)
        return interpreter, narrator

    # -- composable stages -----------------------------------------------

    def _build_monitor(self, *, frame_source=None, detector=None, tracker=None, trigger_sink=None) -> Monitor:
        from audtheia.pipeline import drivers

        frame_source = frame_source or drivers.build_frame_source(self._settings, self._station)
        detector = detector or drivers.build_detector(self._settings, self._station)
        tracker = tracker or build_tracker_from_capture(self._station["capture"])
        trigger_sink = trigger_sink or NullTriggerSink()
        return Monitor(
            settings=self._settings,
            station=self._station,
            db=self._db,
            frame_source=frame_source,
            detector=detector,
            tracker=tracker,
            trigger_sink=trigger_sink,
        )

    def capture(self, **monitor_overrides) -> int:
        """Run capture over the desktop source to its end, returning events written.

        For a video file this processes the clip and returns; for a live camera or
        stream it runs until the source stops. The stored frames are the same raw
        frames the verifier re-scores, so nothing extra is captured for the
        desktop path.
        """
        monitor = self._build_monitor(**monitor_overrides)
        monitor.run()
        return monitor.events_written

    def qc_pending(self) -> int:
        """Finalize every observation still awaiting quality control."""
        engine = QCEngine(settings=self._settings, db=self._db)
        recent = self._db.list_observations(station_id=self._station_id, limit=DEFAULT_SWEEP_SCAN_LIMIT)
        pending = [row["id"] for row in recent if row.get("qc_state") == QC_PENDING]
        for observation_id in pending:
            engine.process(observation_id)
        return len(pending)

    def verify_pending(self) -> int:
        """Re-score and gate every eligible observation not yet verified."""
        engine = VerifyEngine(
            settings=self._settings,
            db=self._db,
            verifier=self._verifier,
            interpreter=self._interpreter,
        )
        return engine.sweep(station_id=self._station_id)

    def dream_once(self):
        """Run one longitudinal dream pass over the verified record."""
        engine = DreamEngine(
            settings=self._settings,
            db=self._db,
            narrator=self._narrator,
            clusterer=self._clusterer,
        )
        return engine.run_pass()

    def report_once(self, *, formats=None):
        """Generate a report for this station."""
        return generate_report(self._settings, self._db, station_id=self._station_id, formats=formats)

    def run_once(self, **monitor_overrides) -> dict:
        """Drive the entire chain once: capture, control, verify, dream, report.

        Ideal for a video file or a single scripted run. Returns a small summary
        of each stage so a caller, or a test, can confirm the whole pipeline ran.
        """
        captured = self.capture(**monitor_overrides)
        controlled = self.qc_pending()
        verified = self.verify_pending()
        dream = self.dream_once()
        report = self.report_once()
        logger.info(
            "desktop run for %s: %d captured, %d quality-controlled, %d verified, "
            "dream pass %s, report %s",
            self._station.get("station_name"),
            captured, controlled, verified,
            getattr(dream, "status", "?"),
            getattr(report, "station_id", "?"),
        )
        return {"captured": captured, "controlled": controlled, "verified": verified, "dream": dream, "report": report}

    # -- the live desktop process ----------------------------------------

    def serve(self, *, host: Optional[str] = None, port: Optional[int] = None,
              verify_interval_seconds: float = DEFAULT_VERIFY_INTERVAL_SECONDS) -> None:
        """Run the live desktop station: capture, a background scheduler, and the UI.

        Capture runs in one thread against the live source; a scheduler thread
        verifies new observations often and runs the dream pass and reports on
        their configured cadence; the web interface serves in the main thread so
        stopping it (Ctrl-C) brings the whole station down cleanly. This is the
        one-command desktop experience.
        """
        stop = threading.Event()
        monitor = self._build_monitor()
        monitor.start()

        scheduler = threading.Thread(
            target=self._scheduler_loop,
            args=(stop, verify_interval_seconds),
            name="audtheia-desktop-scheduler",
            daemon=True,
        )
        scheduler.start()

        from audtheia.app.server import create_app
        import uvicorn

        app = create_app(self._settings, self._db)
        server_cfg = self._settings.raw.get("server", {})
        host = host or server_cfg.get("host", "127.0.0.1")
        port = int(port or server_cfg.get("port", 8000))

        logger.info("desktop station serving at http://%s:%d", host, port)
        try:
            uvicorn.run(app, host=host, port=port, log_level="info")
        finally:
            stop.set()
            monitor.stop()
            scheduler.join(timeout=5.0)

    def _scheduler_loop(self, stop: threading.Event, verify_interval_seconds: float) -> None:
        dream_period = self._schedule_seconds(self._settings.raw.get("schedules", {}).get("dream_pass", {}).get("schedule"))
        report_period = self._schedule_seconds(self._settings.raw.get("schedules", {}).get("reports", {}).get("schedule"))
        elapsed = 0.0
        last_dream = 0.0
        last_report = 0.0
        while not stop.is_set():
            try:
                self.verify_pending()
                if elapsed - last_dream >= dream_period:
                    self.dream_once()
                    last_dream = elapsed
                if elapsed - last_report >= report_period:
                    self.report_once()
                    last_report = elapsed
            except Exception:  # noqa: BLE001 - one bad cycle must not stop the station
                logger.exception("a desktop scheduler cycle failed; continuing")
            stop.wait(verify_interval_seconds)
            elapsed += verify_interval_seconds

    @staticmethod
    def _schedule_seconds(schedule: Optional[str]) -> float:
        return float(_SCHEDULE_SECONDS.get((schedule or "daily").lower(), _SCHEDULE_SECONDS["daily"]))


def main(argv: Optional[list] = None) -> int:
    """Run a desktop station from the command line.

    With no arguments it runs the first station that has a desktop capture source
    and serves the interface until stopped. Pass --once to run the whole chain a
    single time over the source and exit, which suits a video file.
    """
    import argparse
    import logging as _logging
    import sys as _sys

    from audtheia.config import load_settings, ConfigError

    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        prog="python -m audtheia.app.orchestrator",
        description=(
            "Run one station on this desktop with no field hardware: capture from a "
            "webcam, stream, or file, then quality control, verification, the dream "
            "pass, reports, and the web interface."
        ),
    )
    parser.add_argument("--station-id", default=None,
                        help="Run a specific station by id; by default the first station with a capture source.")
    parser.add_argument("--once", action="store_true",
                        help="Run the whole chain once over the source and exit, without serving the interface.")
    parser.add_argument("--host", default=None, help="Interface host; by default the configured server host.")
    parser.add_argument("--port", default=None, type=int, help="Interface port; by default the configured server port.")
    args = parser.parse_args(_sys.argv[1:] if argv is None else argv)

    try:
        settings = load_settings()
        station = DesktopStation.build(settings, station_id=args.station_id)
    except (ConfigError, ValueError) as exc:
        print(f"Desktop station could not start: {exc}", file=_sys.stderr)
        return 1

    if args.once:
        station.run_once()
        return 0
    try:
        station.serve(host=args.host, port=args.port)
        return 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
