"""Audtheia field-station environmental and location capture.

Path: audtheia/pipeline/environment.py

When a field station opens an observation, whether a camera saw an animal or a
hydrophone heard one, it records the physical conditions of that moment: where
the station is, and what its water, air, or soil sensors read. This module is
that capture. It reads the satellite receiver and every configured sensor
channel once per event and returns them ready to store beside the detection.

Three ideas shape everything here.

  A reading reports its own outcome, never a bare number. Every sensor read
  comes back as a small result that says whether the channel was tried, whether
  it succeeded, and what it returned, so the difference between "measured a
  value", "the channel exists but gave nothing this time", and "the sensor
  failed" is a fact the hardware reported, not a guess made after the fact. That
  distinction is the whole point of an absence: "not detected", "not measured",
  and "sensor error" are three different scientific statements.

  Marine channels carry a standard quality flag. A water sensor's value is
  checked against the operating ranges the configuration gives it and flagged on
  the recognised oceanographic scale, so a reviewer reads the same flags they
  would from any other instrument rather than a private scheme.

  The satellite receiver is also the clock. In the field there is no internet to
  set the time from, so the station takes coordinated universal time from the
  satellites. A small clock helper remembers the last good satellite time and
  reports whether the station's clock has ever been set, so a reading taken
  before the first fix is marked as having a provisional time rather than being
  trusted as if it were disciplined. The battery-backed hardware clock carries
  the time forward between fixes once it has been set.

The satellite receiver and the sensor bank are reached through small
interfaces, so this module runs end to end against scripted readings with no
receiver or sensor present, and the real drivers drop in later without touching
this code. The sensor set itself, its units, and its quality-control ranges are
entirely a matter of configuration, so a marine station and a forest station
run the same code with different sensors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Protocol, runtime_checkable

from audtheia.pipeline.monitor import CaptureResult, ChannelReading, ISO_FORMAT

__all__ = [
    "GpsRead",
    "SensorRead",
    "GpsSource",
    "SensorBank",
    "FieldClock",
    "EnvironmentCapture",
    "NullGpsSource",
    "NullSensorBank",
    "NullEnvironmentCapture",
    "STATUS_MEASURED",
    "STATUS_NOT_MEASURED",
    "STATUS_BELOW_DETECTION_LIMIT",
    "STATUS_SENSOR_ERROR",
    "STATUS_NOT_APPLICABLE",
    "QARTOD_PASS",
    "QARTOD_NOT_EVALUATED",
    "QARTOD_SUSPECT",
    "QARTOD_FAIL",
    "QARTOD_MISSING",
]

logger = logging.getLogger("audtheia.pipeline.environment")

# The missing-data vocabulary, one term per outcome. A value that was captured
# normally is measured; a channel that exists but returned nothing this event is
# not_measured; a value under the sensor's stated limit of detection is recorded
# as such rather than as zero or absent; a channel that was expected but failed
# is a sensor_error; a channel that does not apply to this deployment is
# not_applicable. Keeping these as named constants means a status is never a
# loose string a typo could slip past.
STATUS_MEASURED = "measured"
STATUS_NOT_MEASURED = "not_measured"
STATUS_BELOW_DETECTION_LIMIT = "below_detection_limit"
STATUS_SENSOR_ERROR = "sensor_error"
STATUS_NOT_APPLICABLE = "not_applicable"

# The oceanographic quality-control flag scale, applied to marine channels only.
# One (pass) means the value sits inside its expected operating range; two (not
# evaluated) means no range was configured to judge it against; three (suspect)
# means the value is physically plausible for the instrument but outside the
# expected range, so it is of high interest; four (fail) means the value is
# outside what the instrument can physically report and cannot be trusted; nine
# (missing) means there was no value to evaluate.
QARTOD_PASS = 1
QARTOD_NOT_EVALUATED = 2
QARTOD_SUSPECT = 3
QARTOD_FAIL = 4
QARTOD_MISSING = 9


# ===========================================================================
# Value types crossing the interfaces
# ===========================================================================


@dataclass
class GpsRead:
    """The outcome of one read of the satellite receiver.

    attempted is true whenever a read was tried, which is false only when the
    station has no receiver configured. ok is true when a usable fix was
    obtained. latitude, longitude, and elevation are the fix when ok is true and
    empty otherwise. utc_time is the satellite's coordinated universal time for
    the fix, the drift-free clock source the station has no internet to get
    otherwise. error carries the reason a read failed, for the station log.
    """

    attempted: bool
    ok: bool
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    elevation: Optional[float] = None
    utc_time: Optional[str] = None
    error: Optional[str] = None


@dataclass
class SensorRead:
    """The outcome of one read of one sensor channel.

    channel is the channel identifier from the configuration. attempted is true
    whenever a read was tried. ok is true when the sensor returned a usable
    value. value is that reading when ok is true and empty otherwise. error
    carries the reason a read failed, for the station log. The channel's unit
    and its quality-control ranges are not carried here: they belong to the
    configuration, and the capture applies them, so a driver only has to report
    what happened at the pin.
    """

    channel: str
    attempted: bool
    ok: bool
    value: Optional[float] = None
    error: Optional[str] = None


# ===========================================================================
# Interfaces (the seams real hardware drops into)
# ===========================================================================


@runtime_checkable
class GpsSource(Protocol):
    """A source of satellite fixes. The real one wraps the receiver; a test one
    returns scripted fixes; the null one reports no receiver. read returns the
    outcome of a single read, always as a structured result, never a bare value
    or a bare None, so the capture can assign a status from the outcome alone."""

    def read(self) -> GpsRead: ...

    def close(self) -> None: ...


@runtime_checkable
class SensorBank(Protocol):
    """The station's set of environmental sensors. The real one wraps the wired
    channels; a test one returns scripted readings; the null one has none.
    read_all returns one structured result per channel it was asked to read, in
    any order, so the capture can pair each with its configuration and assign a
    status from the outcome alone."""

    def read_all(self, channel_ids: list[str]) -> list[SensorRead]: ...

    def close(self) -> None: ...


# ===========================================================================
# A satellite source and a sensor bank that report nothing present
# ===========================================================================


class NullGpsSource:
    """A satellite source for a station with no receiver, or for running the
    capture before the receiver is wired. Every read reports that nothing was
    attempted, so the location is recorded as not measured rather than invented."""

    def read(self) -> GpsRead:
        return GpsRead(attempted=False, ok=False)

    def close(self) -> None:
        pass


class NullSensorBank:
    """A sensor bank for a station with no environmental sensors, or for running
    the capture before they are wired. It returns no readings, so the event is
    recorded with its location and detection and simply no sensor channels."""

    def read_all(self, channel_ids: list[str]) -> list[SensorRead]:
        return []

    def close(self) -> None:
        pass


# ===========================================================================
# The clock the satellites discipline
# ===========================================================================


def _parse_iso(value: str) -> datetime:
    return datetime.strptime(value, ISO_FORMAT).replace(tzinfo=timezone.utc)


class FieldClock:
    """Tracks whether the station's clock has been set by the satellites yet.

    A field station has no internet to set its time from, so coordinated
    universal time comes from the satellite receiver. Until the first fix, any
    time the station stamps is only as good as an unset clock, so it is marked
    provisional. Once a fix has disciplined the clock, the battery-backed
    hardware clock carries that time forward between fixes, so later readings are
    trusted even while the receiver has no live fix.

    This helper holds only that state. It does not stamp rows itself: the parts
    of the pipeline that create a record write the provisional marker, and they
    read it from here, so there is one answer to "has the clock been set" for the
    whole station.
    """

    def __init__(self) -> None:
        self._disciplined = False
        self._last_utc: Optional[str] = None

    def observe(self, gps: GpsRead) -> None:
        """Update the clock state from a satellite read.

        A usable fix that carries a time disciplines the clock and is remembered
        as the last good time; a read without a fix leaves the state unchanged,
        which is exactly the hardware clock holding time between fixes.
        """
        if gps.ok and gps.utc_time:
            self._disciplined = True
            self._last_utc = gps.utc_time

    @property
    def disciplined(self) -> bool:
        """True once a satellite fix has ever set the clock."""
        return self._disciplined

    @property
    def last_utc(self) -> Optional[str]:
        """The most recent satellite time seen, or nothing before the first fix."""
        return self._last_utc

    def time_provisional(self) -> int:
        """1 until the clock has been disciplined, 0 afterward.

        This is the value a frame source or an audio source stamps on the record
        it creates, so a capture taken before the first fix is never trusted as
        if its time were authoritative.
        """
        return 0 if self._disciplined else 1


# ===========================================================================
# Status and quality-flag assignment
# ===========================================================================


def _gps_status(gps: GpsRead) -> str:
    """Turn a satellite read outcome into a missing-data status.

    A read that was never attempted means the station has no receiver, so its
    location does not apply here; an attempted read that failed is a sensor
    error; an attempted read with no fix is a channel that exists but gave
    nothing this event; a usable fix is measured.
    """
    if not gps.attempted:
        return STATUS_NOT_APPLICABLE
    if gps.error is not None:
        return STATUS_SENSOR_ERROR
    if gps.ok:
        return STATUS_MEASURED
    return STATUS_NOT_MEASURED


def _channel_status(read: SensorRead, detection_limit: Optional[float]) -> str:
    """Turn a sensor read outcome into a missing-data status.

    A read that reported an error is a sensor error; a read that came back with
    no value is a channel that gave nothing this event; a value under the
    channel's stated limit of detection is recorded as below that limit, distinct
    from zero or absent; any other value is measured.
    """
    if read.error is not None:
        return STATUS_SENSOR_ERROR
    if not read.ok or read.value is None:
        return STATUS_NOT_MEASURED
    if detection_limit is not None and read.value < detection_limit:
        return STATUS_BELOW_DETECTION_LIMIT
    return STATUS_MEASURED


def _qartod_flag(
    status: str,
    value: Optional[float],
    qc: Optional[dict],
) -> int:
    """Assign an oceanographic quality flag to a marine channel value.

    A channel with no value to judge is flagged missing. With no configured
    ranges the value is not evaluated. Otherwise the value is checked first
    against the sensor's physical range, then against the expected operating
    range: outside what the instrument can report is a fail, physically possible
    but outside the expected range is suspect, and inside the expected range is a
    pass. A below-detection-limit value still has a real number and is range
    checked like any other.
    """
    if value is None or status in (STATUS_NOT_MEASURED, STATUS_SENSOR_ERROR, STATUS_NOT_APPLICABLE):
        return QARTOD_MISSING
    if not qc:
        return QARTOD_NOT_EVALUATED

    sensor_range = qc.get("sensor_range")
    gross_range = qc.get("gross_range")

    if sensor_range is not None:
        smin = sensor_range.get("min")
        smax = sensor_range.get("max")
        if (smin is not None and value < smin) or (smax is not None and value > smax):
            return QARTOD_FAIL

    if gross_range is not None:
        gmin = gross_range.get("min")
        gmax = gross_range.get("max")
        if (gmin is not None and value < gmin) or (gmax is not None and value > gmax):
            return QARTOD_SUSPECT
        return QARTOD_PASS

    # A sensor range was given and passed, but no expected operating range was
    # configured to judge the value more tightly.
    return QARTOD_NOT_EVALUATED


# ===========================================================================
# The environmental capture
# ===========================================================================


class EnvironmentCapture:
    """Reads location and every configured sensor channel for one event.

    This is the location-and-environment leg of a station's simultaneous
    capture. It is called once per event, on either trigger, and fills only the
    location and environmental-channel fields of the shared capture result; the
    audio and visual legs fill their own fields and are merged alongside it. The
    channel set, each channel's unit, and its quality-control ranges come
    entirely from the station configuration, so no sensor is ever named in code.

    The satellite read also feeds the shared clock, so the clock is disciplined
    as a side effect of the ordinary capture the station already performs.
    """

    def __init__(
        self,
        *,
        settings,
        station: dict,
        gps_source: GpsSource,
        sensor_bank: SensorBank,
        clock: Optional[FieldClock] = None,
    ) -> None:
        self._settings = settings
        self._station = station
        self._gps = gps_source
        self._sensors = sensor_bank
        self._clock = clock or FieldClock()

        sensors_cfg = station.get("sensors", {})
        gps_cfg = sensors_cfg.get("gps", {})
        self._gps_enabled = bool(gps_cfg.get("enabled", False))

        # Only channels that are present and switched on are read. A channel the
        # deployment has turned off is not part of this station's record, exactly
        # as an unconfigured sensor is not, so it produces no row rather than a
        # manufactured absence.
        self._channels = [
            c for c in station.get("channels", []) if c.get("enabled", False)
        ]
        self._channel_by_id = {c["id"]: c for c in self._channels}

    @property
    def clock(self) -> FieldClock:
        """The shared clock this capture disciplines, so a frame or audio source
        can read the same clock state when it stamps a record."""
        return self._clock

    def capture(self, first_seen: str, last_seen: str) -> CaptureResult:
        """Read location and all enabled sensors for one event window.

        Returns a capture result carrying only the location fields and the
        per-channel environmental readings. The window is accepted for interface
        symmetry with the other capture legs; the reads themselves are the
        station's current conditions at the moment of the trigger.
        """
        result = CaptureResult()

        self._capture_gps(result)
        self._capture_channels(result)

        return result

    # -- location --------------------------------------------------------

    def _capture_gps(self, result: CaptureResult) -> None:
        if not self._gps_enabled:
            # No receiver on this station: the location does not apply, and the
            # clock is left as it is.
            result.gps_status = STATUS_NOT_APPLICABLE
            return

        try:
            gps = self._gps.read()
        except Exception as exc:  # noqa: BLE001 - a driver fault is recorded, never fatal
            logger.warning("gps read failed: %s", exc)
            result.gps_status = STATUS_SENSOR_ERROR
            return

        # Feed the clock first, so a fix in this very read disciplines the clock
        # the rest of the pipeline reads from.
        self._clock.observe(gps)

        result.gps_latitude = gps.latitude
        result.gps_longitude = gps.longitude
        result.gps_elevation = gps.elevation
        result.gps_status = _gps_status(gps)

    # -- environmental channels -----------------------------------------

    def _capture_channels(self, result: CaptureResult) -> None:
        if not self._channels:
            return

        channel_ids = [c["id"] for c in self._channels]
        try:
            reads = self._sensors.read_all(channel_ids)
        except Exception as exc:  # noqa: BLE001 - a bank fault marks every channel, never fatal
            logger.warning("sensor bank read failed: %s", exc)
            reads = [
                SensorRead(channel=cid, attempted=True, ok=False, error=str(exc))
                for cid in channel_ids
            ]

        by_channel = {r.channel: r for r in reads}

        for channel in self._channels:
            channel_id = channel["id"]
            read = by_channel.get(channel_id)
            if read is None:
                # The bank was asked for this channel but returned nothing for
                # it; that is a channel with no value this event, recorded as
                # such rather than dropped.
                read = SensorRead(channel=channel_id, attempted=True, ok=False)
            result.environmental_readings.append(
                self._reading_for(channel, read)
            )

    def _reading_for(self, channel: dict, read: SensorRead) -> ChannelReading:
        qc = channel.get("qc") or {}
        detection_limit = qc.get("detection_limit")
        status = _channel_status(read, detection_limit)
        value = read.value if read.ok else None

        qartod: Optional[int] = None
        if channel.get("marine", False):
            qartod = _qartod_flag(status, value, qc)

        return ChannelReading(
            channel=channel["id"],
            status=status,
            value=value,
            unit=channel.get("unit"),
            qartod_flag=qartod,
        )

    def close(self) -> None:
        """Release the underlying receiver and sensor bank."""
        try:
            self._gps.close()
        finally:
            self._sensors.close()


# ===========================================================================
# An environment capture that gathers nothing
# ===========================================================================


class NullEnvironmentCapture:
    """A location-and-environment capture that gathers nothing, for a station
    with neither a receiver nor sensors, or for wiring the loop before they
    exist. It returns an empty result, so the event is complete and valid with
    those fields simply absent, and the real capture attaches later without
    changing its caller."""

    def capture(self, first_seen: str, last_seen: str) -> CaptureResult:
        return CaptureResult()

    def close(self) -> None:
        pass
