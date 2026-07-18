"""Audtheia field-station runner.

Path: audtheia/pipeline/__main__.py

This is the one process a field station runs. On the Pi it is launched as
`python -m audtheia.pipeline` and kept alive by the system service the setup
installs, so a station that loses power comes back up on its own.

It does not reimplement any capture or analysis. It composes the parts that
already exist, each through its own interface:

  - the vision detection loop, which watches the camera and closes one event per
    tracked animal,
  - the independent acoustic detector and the audio capture that rides along
    with a vision event,
  - the location-and-environment capture that records where the station is and
    what its sensors read, and
  - the deterministic quality-control engine that finalizes each stored record.

Every sense is chosen from configuration. A sense whose sensor is switched on and
whose hardware driver is installed runs against the real device; a sense with no
driver yet, or one a station has turned off, falls back to a null source that
captures nothing, so the station runs and stays truthful with that sense simply
absent. The hardware drivers (the camera frame source, the accelerator detector,
the satellite receiver, the sensor bank, and the audio device) are supplied by an
optional drivers module the station installs for its own hardware; until one is
present, its sense is reported inactive and the runner keeps serving the senses
that are.

Whatever is active, the runner always runs the quality-control engine: it sweeps
the local store for records that capture has written but not yet finalized and
processes each one, so the deterministic field tier keeps every observation
complete and downstream-ready. The engine honors the configured analysis
location, so a deployment that defers quality control to the desktop leaves the
field records pending on purpose.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from typing import Optional

from audtheia.config import ConfigError, load_settings
from audtheia.storage.database import Database
from audtheia.analysis.observation import QCEngine, QC_PENDING


logger = logging.getLogger("audtheia.pipeline")

# How often the runner sweeps the store for records to finalize. Capture writes a
# record and moves on; this interval bounds how long a freshly written record
# waits before the deterministic engine finalizes it. It is a named constant
# here, a sensible starting value a deployment can tune, pending a configuration
# home for field-runner timing.
DEFAULT_SWEEP_SECONDS = 5.0

# How many of the most recent records a single sweep examines. Records awaiting
# quality control are the newest ones, which this ordering reaches first, so the
# bound keeps a sweep cheap on a station holding a long rolling buffer. A record
# that ever falls behind this window still gets its authoritative check on the
# desktop, which re-verifies every observation.
DEFAULT_SWEEP_SCAN_LIMIT = 1000


# ---------------------------------------------------------------------------
# Choosing the hardware drivers.
# ---------------------------------------------------------------------------


def _resolve_drivers(explicit):
    """The hardware drivers module, or nothing when none is installed.

    A station supplies its own drivers as audtheia.pipeline.drivers, which builds
    the real camera, accelerator, receiver, sensor bank, and audio device for its
    hardware. When that module is absent, every sense falls back to a null source,
    which is the correct behavior for a station whose hardware is not wired yet. A
    test injects its own object here to drive the composition without any device.
    """
    if explicit is not None:
        return explicit
    try:
        from audtheia.pipeline import drivers  # optional, installed with the hardware
        return drivers
    except Exception:  # noqa: BLE001 - absence is expected and means null sources
        return None


def _sensor_enabled(station: dict, sensor: str) -> bool:
    return bool(station.get("sensors", {}).get(sensor, {}).get("enabled", False))


# ---------------------------------------------------------------------------
# The composed station.
# ---------------------------------------------------------------------------


class FieldStation:
    """The active senses of one station, plus the quality-control engine.

    Build it with build, then start the capture loops, serve the
    quality-control sweep until asked to stop, and stop the loops cleanly. What is
    active depends entirely on configuration and on which hardware drivers are
    installed; the quality-control engine always runs.
    """

    def __init__(
        self,
        *,
        settings,
        station: dict,
        db: Database,
        qc_engine: QCEngine,
        monitor=None,
        acoustic=None,
        active_senses: Optional[dict] = None,
    ) -> None:
        self._settings = settings
        self._station = station
        self._db = db
        self._qc_engine = qc_engine
        self._monitor = monitor
        self._acoustic = acoustic
        self.active_senses = active_senses or {}
        self._station_id = station["station_id"]

    # -- construction ----------------------------------------------------

    @classmethod
    def build(cls, settings, station: dict, *, drivers=None) -> "FieldStation":
        drivers = _resolve_drivers(drivers)
        db = Database(settings.db_path(), **settings.database_kwargs())

        active: dict[str, str] = {}

        environment_capture = cls._build_environment(settings, station, drivers, active)
        acoustic, acoustic_sink = cls._build_acoustic(
            settings, station, db, drivers, environment_capture, active
        )
        monitor = cls._build_monitor(
            settings, station, db, drivers, environment_capture, acoustic_sink, active
        )

        qc_engine = QCEngine(settings=settings, db=db)

        return cls(
            settings=settings,
            station=station,
            db=db,
            qc_engine=qc_engine,
            monitor=monitor,
            acoustic=acoustic,
            active_senses=active,
        )

    @staticmethod
    def _build_environment(settings, station, drivers, active):
        from audtheia.pipeline.environment import (
            EnvironmentCapture,
            NullEnvironmentCapture,
            NullGpsSource,
            NullSensorBank,
        )

        has_gps = _sensor_enabled(station, "gps")
        has_channels = any(c.get("enabled", False) for c in station.get("channels", []))
        # A station with an entered fixed position but no receiver and no sensors
        # still has a location worth recording on every event, so the capture is
        # built for it rather than skipped.
        location = station.get("location") or {}
        has_location = location.get("latitude") is not None and location.get("longitude") is not None
        if not has_gps and not has_channels and not has_location:
            active["environment"] = "inactive: no receiver, sensors, or coordinates configured"
            return NullEnvironmentCapture()

        gps_source = None
        sensor_bank = None
        if drivers is not None and hasattr(drivers, "build_gps_source"):
            gps_source = drivers.build_gps_source(settings, station)
        if drivers is not None and hasattr(drivers, "build_sensor_bank"):
            sensor_bank = drivers.build_sensor_bank(settings, station)

        note = []
        if has_gps and gps_source is None:
            note.append("receiver driver not installed")
        if has_channels and sensor_bank is None:
            note.append("sensor driver not installed")
        active["environment"] = "active" if not note else "partial (" + "; ".join(note) + ")"

        return EnvironmentCapture(
            settings=settings,
            station=station,
            gps_source=gps_source or NullGpsSource(),
            sensor_bank=sensor_bank or NullSensorBank(),
        )

    @staticmethod
    def _build_acoustic(settings, station, db, drivers, environment_capture, active):
        if not _sensor_enabled(station, "audio"):
            active["audio"] = "inactive: audio switched off"
            return None, None
        if drivers is None or not hasattr(drivers, "build_audio_source"):
            active["audio"] = "inactive: audio driver not installed"
            return None, None
        try:
            from audtheia.pipeline.acoustic import (
                AcousticMonitor,
                AcousticTriggerSink,
                build_acoustic_model,
            )

            model = build_acoustic_model(station, settings)
            audio_source = drivers.build_audio_source(settings, station)
            monitor = AcousticMonitor(
                settings=settings,
                station=station,
                db=db,
                audio_source=audio_source,
                model=model,
                environment_capture=environment_capture,
            )
            # The audio that rides along with a vision event reads the same shared
            # ring buffer this monitor fills, and carries the same model version.
            trigger_sink = AcousticTriggerSink(
                settings=settings,
                station=station,
                ring_buffer=monitor.ring_buffer,
                acoustic_model_version=model.version,
            )
            active["audio"] = "active"
            return monitor, trigger_sink
        except Exception as exc:  # noqa: BLE001 - a missing model or driver disables the sense, never the station
            active["audio"] = f"inactive: {exc}"
            logger.warning("acoustic capture could not start: %s", exc)
            return None, None

    @staticmethod
    def _build_monitor(settings, station, db, drivers, environment_capture, acoustic_sink, active):
        if not _sensor_enabled(station, "camera"):
            active["vision"] = "inactive: camera switched off"
            return None
        if drivers is None or not (
            hasattr(drivers, "build_frame_source") and hasattr(drivers, "build_detector")
        ):
            active["vision"] = "inactive: camera or detector driver not installed"
            return None
        try:
            from audtheia.pipeline.monitor import Monitor, build_tracker_from_capture
            from audtheia.pipeline.composer import ComposedTriggerSink

            frame_source = drivers.build_frame_source(settings, station)
            detector = drivers.build_detector(settings, station)
            tracker = build_tracker_from_capture(station["capture"])
            trigger_sink = ComposedTriggerSink(
                acoustic_sink=acoustic_sink,
                environment_capture=environment_capture,
            )
            monitor = Monitor(
                settings=settings,
                station=station,
                db=db,
                frame_source=frame_source,
                detector=detector,
                tracker=tracker,
                trigger_sink=trigger_sink,
            )
            active["vision"] = "active"
            return monitor
        except Exception as exc:  # noqa: BLE001 - a missing driver disables the sense, never the station
            active["vision"] = f"inactive: {exc}"
            logger.warning("vision detection could not start: %s", exc)
            return None

    # -- lifecycle -------------------------------------------------------

    def start(self) -> None:
        """Start whichever capture loops are active. The sweep runs separately."""
        if self._acoustic is not None:
            self._acoustic.start()
        if self._monitor is not None:
            self._monitor.start()

    def sweep_once(self) -> int:
        """Finalize any records capture has written but not yet completed.

        Returns how many records were handed to the engine. The engine is
        idempotent and honors the configured analysis location, so a record that
        is already finalized, or one this node defers to the desktop, is left as
        it is.
        """
        recent = self._db.list_observations(
            station_id=self._station_id, limit=DEFAULT_SWEEP_SCAN_LIMIT
        )
        pending = [row["id"] for row in recent if row.get("qc_state") == QC_PENDING]
        for observation_id in pending:
            self._qc_engine.process(observation_id)
        return len(pending)

    def serve(self, stop_event: threading.Event, *, interval: float = DEFAULT_SWEEP_SECONDS) -> None:
        """Run the quality-control sweep until asked to stop."""
        while not stop_event.is_set():
            try:
                self.sweep_once()
            except Exception:  # noqa: BLE001 - one bad sweep must not stop the station
                logger.exception("a quality-control sweep failed; continuing")
            stop_event.wait(interval)

    def stop(self) -> None:
        """Ask the capture loops to finish and shut down."""
        if self._monitor is not None:
            self._monitor.stop()
        if self._acoustic is not None:
            self._acoustic.stop()


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def _select_station(settings, station_id: Optional[str]) -> dict:
    if station_id is not None:
        return settings.station(station_id)
    if settings.node_role == "pi":
        station = settings.active_station()
        if station is None:
            raise ConfigError(
                "node.role is 'pi' but no active station is set; set node.active_station_id"
            )
        return station
    raise ConfigError(
        "the field runner runs on a field station (node.role 'pi'); this node's "
        "role is not 'pi'. Pass --station-id to run a specific station for testing."
    )


def run_field(
    settings=None,
    station: Optional[dict] = None,
    *,
    drivers=None,
    stop_event: Optional[threading.Event] = None,
    run_once: bool = False,
    sweep_seconds: float = DEFAULT_SWEEP_SECONDS,
    station_id: Optional[str] = None,
) -> None:
    """Compose and run one field station.

    With no arguments it reads the configuration, runs as the station this node
    is set to, starts every active sense, and serves the quality-control sweep
    until stopped. A test can pass an explicit station, a drivers object, a stop
    event, or run_once to drive one sweep and return.
    """
    if settings is None:
        settings = load_settings()
    if station is None:
        station = _select_station(settings, station_id)
    if stop_event is None:
        stop_event = threading.Event()

    field = FieldStation.build(settings, station, drivers=drivers)

    logger.info("field station %s", station.get("station_name"))
    for sense, state in field.active_senses.items():
        logger.info("  %s: %s", sense, state)

    field.start()
    try:
        if run_once:
            field.sweep_once()
        else:
            field.serve(stop_event, interval=sweep_seconds)
    finally:
        field.stop()


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        prog="python -m audtheia.pipeline",
        description="Run one Audtheia field station: every active sense plus the quality-control engine.",
    )
    parser.add_argument(
        "--station-id",
        default=None,
        help="Run a specific station by id instead of this node's active station.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single quality-control sweep and exit, without starting the capture loops for long.",
    )
    parser.add_argument(
        "--sweep-seconds",
        type=float,
        default=DEFAULT_SWEEP_SECONDS,
        help="Seconds between quality-control sweeps.",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    stop_event = threading.Event()

    def _handle_signal(signum, _frame):
        logger.info("received signal %s; shutting down", signum)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_signal)
        except (ValueError, OSError):
            # Signals can only be installed on the main thread; a caller running
            # this off the main thread supplies its own stop path.
            pass

    try:
        run_field(
            stop_event=stop_event,
            run_once=args.once,
            sweep_seconds=args.sweep_seconds,
            station_id=args.station_id,
        )
        return 0
    except ConfigError as exc:
        print(f"Field runner stopped: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
