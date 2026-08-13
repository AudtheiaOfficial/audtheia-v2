"""Field-station hardware drivers for Audtheia.

Path: audtheia/pipeline/field_drivers.py

PENDING ON-DEVICE VALIDATION. The drivers here are written against the documented
library and sensor protocols (sounddevice for a live microphone or hydrophone,
the Atlas Scientific EZO I2C command set for environmental sensors, and standard
NMEA 0183 for a satellite receiver), but they have not yet been run against
physical hardware. They are structured so that on-device validation is a matter
of confirming a few device specifics (a sound device index, an I2C bus number and
addresses, a serial port and baud rate), not of writing new logic. See
docs/field-drivers.md.

The field runner (audtheia/pipeline/__main__.py) builds a station's senses from
its optional drivers module through small factory functions. This module supplies
the field-station side of that seam: a live audio source, an I2C environmental
sensor bank, and an NMEA satellite receiver, each conforming exactly to the
Protocols the pipeline already defines (AudioSource in acoustic.py, SensorBank and
GpsSource in environment.py). They are re-exported from audtheia/pipeline/drivers.py
so a station's drivers module exposes them where the field runner looks.

Two design rules make this testable and safe:

  Every hardware library (sounddevice, smbus2, pyserial) is imported lazily inside
  the factory function that needs it, so importing this module never requires a
  Raspberry Pi, and a desktop that reaches this seam simply reports the sense
  inactive rather than failing to load.

  Every source wraps a backend that is injected, so the reading and parsing logic
  is exercised end to end against a scripted fake with no device present; the
  factory functions supply the real backend. This is the same pattern the desktop
  drivers use for the camera and the detector.

Nothing here changes the provenance firewall. A sensor read that fails returns a
structured result marked not-ok with a reason, never a fabricated value; a GPS
read with no fix returns not-ok, so the location is recorded as not measured
rather than invented; audio carries the same timestamp discipline as the rest of
the pipeline.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np

from audtheia.pipeline.acoustic import AudioBlock
from audtheia.pipeline.environment import GpsRead, SensorRead
from audtheia.pipeline.monitor import ISO_FORMAT

logger = logging.getLogger("audtheia.pipeline.field_drivers")

__all__ = [
    "FieldHardwareError",
    "LiveAudioSource",
    "I2CSensorBank",
    "NmeaGpsSource",
    "GpsdGpsSource",
    "build_audio_source",
    "build_sensor_bank",
    "build_gps_source",
    "DEFAULT_AUDIO_RATE",
    "DEFAULT_I2C_BUS",
    "DEFAULT_GPS_PORT",
    "DEFAULT_GPS_BAUD",
]

# A default audio rate for a live source when the acoustic model's rate is not
# stated in configuration. 48 kHz is the BirdNET analysis rate and a rate every
# common capture device supports; the real rate is read from the model entry when
# it is set.
DEFAULT_AUDIO_RATE = 48000

# The Raspberry Pi's user I2C bus is bus 1. The GPS defaults match the Pi's
# primary UART and the near-universal NMEA baud. All are overridable per station.
DEFAULT_I2C_BUS = 1
DEFAULT_GPS_PORT = "/dev/serial0"
DEFAULT_GPS_BAUD = 9600


class FieldHardwareError(RuntimeError):
    """A field driver could not be built because a library or device was absent.

    Raised only by the audio builder, which the field runner calls inside a guard
    that turns it into an inactive sense. The sensor and GPS builders never raise;
    they return nothing and the runner falls back to the null source, so a missing
    library never stops a station.
    """


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime(ISO_FORMAT)


# ===========================================================================
# Live audio: a microphone or a hydrophone-through-a-converter
# ===========================================================================


class LiveAudioSource:
    """An `AudioSource` that reads blocks from a live capture stream.

    The stream is any object with ``read(frames) -> array`` returning that many
    frames of float audio in [-1, 1] (a bare array, or the ``(array, overflowed)``
    pair the sounddevice input stream returns) and a ``close()``. Multi-channel
    input is mixed to mono, which is what the acoustic models expect. Each block
    is stamped with the capture time; a station whose clock is not yet disciplined
    by a satellite fix marks the block provisional.
    """

    def __init__(
        self,
        stream,
        *,
        sample_rate: int,
        block_frames: int,
        clock_now: Callable[[], str] = _utc_now_iso,
        time_provisional: int = 0,
    ) -> None:
        self._stream = stream
        self._rate = int(sample_rate)
        self._block = max(1, int(block_frames))
        self._now = clock_now
        self._tp = int(time_provisional)

    def read(self) -> Optional[AudioBlock]:
        data = self._stream.read(self._block)
        # The sounddevice input stream returns (frames, overflowed); a fake or a
        # simpler backend may return the array alone.
        if isinstance(data, tuple):
            data = data[0]
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if arr.size == 0:
            return None
        return AudioBlock(
            samples=arr,
            sample_rate=self._rate,
            captured_at=self._now(),
            time_provisional=self._tp,
        )

    def close(self) -> None:
        try:
            self._stream.close()
        except Exception:  # noqa: BLE001 - closing a device must never raise upward
            pass


class _SoundDeviceStream:
    """Wrap a sounddevice InputStream to the plain ``read(frames)`` seam above."""

    def __init__(self, stream) -> None:
        self._stream = stream

    def read(self, frames: int):
        data, _overflowed = self._stream.read(frames)
        return data

    def close(self) -> None:
        try:
            self._stream.stop()
        finally:
            self._stream.close()


def _import_sounddevice():
    try:
        import sounddevice  # noqa: PLC0415 - optional, only for a live audio device
    except ImportError as exc:
        raise FieldHardwareError(
            "live audio capture needs the 'sounddevice' package (and PortAudio). "
            "Install it on the station, or configure a file or URL audio source "
            "instead."
        ) from exc
    return sounddevice


def _is_live_audio_spec(spec) -> bool:
    """Whether a configured audio spec names a live device rather than a file/URL.

    A live spec is empty (no desktop source, so use the wired device), or one of
    the explicit device words, or a ``device:...`` selector. Anything else is a
    file path or a URL and is served by the desktop audio source, so a field
    station can also replay a recording for a bench test.
    """
    if not spec:
        return True
    s = str(spec).strip().lower()
    return s in ("live", "device", "mic", "microphone", "hydrophone") or s.startswith("device:")


def build_audio_source(settings, station: dict):
    """Build the station's acoustic capture source: a live device, or a file/URL.

    The target rate is the acoustic model's configured sample rate, so the blocks
    this source hands out already match what the model reads; when the model rate
    is not stated, a safe default is used and the device is opened at it. A file or
    URL audio spec is delegated to the desktop audio source, so the same station
    can be driven from a recording on the bench and from a live hydrophone in the
    field with only a configuration change.

    Raises FieldHardwareError when a live device is requested but the audio library
    is not installed; the field runner turns that into an inactive audio sense.
    """
    acoustic = station.get("models", {}).get("acoustic", {}) or {}
    rate = int(acoustic.get("sample_rate") or DEFAULT_AUDIO_RATE)

    spec = settings.capture_source(station).get("audio")
    if not _is_live_audio_spec(spec):
        from audtheia.pipeline.audio_sources import build_desktop_audio_source  # noqa: PLC0415

        return build_desktop_audio_source(spec, target_rate=rate)

    audio_cfg = station.get("sensors", {}).get("audio", {}) or {}
    driver = audio_cfg.get("driver", {}) if isinstance(audio_cfg, dict) else {}
    device = driver.get("device")  # None selects the system default input
    channels = int(driver.get("channels", 1))
    block_seconds = float(driver.get("block_seconds", 1.0))

    sd = _import_sounddevice()
    try:
        stream = sd.InputStream(
            samplerate=rate, channels=channels, dtype="float32", device=device
        )
        stream.start()
    except Exception as exc:  # noqa: BLE001 - a device that will not open is a clear, reportable failure
        raise FieldHardwareError(
            f"the live audio device could not be opened at {rate} Hz "
            f"({type(exc).__name__}: {exc}). Check the device index and that it "
            f"supports this rate, or configure a file audio source."
        ) from exc

    return LiveAudioSource(
        _SoundDeviceStream(stream),
        sample_rate=rate,
        block_frames=int(round(rate * block_seconds)),
    )


# ===========================================================================
# Environmental sensors over I2C (Atlas Scientific EZO family)
# ===========================================================================


def _parse_i2c_address(value) -> int:
    """Accept an I2C address as an int or a hex string like ``0x66``."""
    if isinstance(value, int):
        return value
    text = str(value).strip()
    return int(text, 16) if text.lower().startswith("0x") else int(text)


def _read_atlas_ezo(bus, address: int, *, settle_seconds: float, sleep: Callable[[float], None]):
    """Read one Atlas Scientific EZO circuit over I2C.

    Returns ``(ok, value, error)``. The EZO command set is uniform across the
    circuit types (temperature, pH, dissolved oxygen, conductivity, and the rest):
    write the single-byte read command, wait for the circuit to take the reading,
    then read the response, whose first byte is a status code (1 success, 2 command
    error, 254 still processing, 255 no data) and whose remaining bytes are the
    reading as a null-terminated ASCII number. Because every circuit speaks this
    same protocol, one reader serves them all and the specific circuit is only a
    label in configuration, never a branch in code.
    """
    bus.write(address, b"R")
    sleep(max(0.0, float(settle_seconds)))
    raw = bytes(bus.read(address, 32))
    if not raw:
        return False, None, "no response from the sensor"
    code = raw[0]
    if code == 254:
        return False, None, "sensor still processing the reading"
    if code == 255:
        return False, None, "sensor had no data to send"
    if code == 2:
        return False, None, "sensor reported a command error"
    if code != 1:
        return False, None, f"unexpected sensor status code {code}"
    payload = raw[1:].split(b"\x00", 1)[0]
    text = payload.decode("ascii", "ignore").strip()
    try:
        return True, float(text), None
    except ValueError:
        return False, None, f"unreadable sensor value {text!r}"


# The sensor-type prefixes this bank can read today. Every Atlas EZO circuit
# shares one I2C protocol, so the family is matched by prefix rather than listing
# each circuit; a new interface or a non-EZO sensor is reported as unsupported
# rather than guessed at.
_SUPPORTED_SENSOR_PREFIXES = ("atlas_ezo",)


class I2CSensorBank:
    """A `SensorBank` over an I2C bus, driven entirely by station configuration.

    The bus is any object with ``write(address, data)`` and
    ``read(address, length) -> bytes`` and a ``close()``; the real one wraps
    smbus2, a test one scripts the bytes. Each channel names its I2C address and
    circuit type in configuration, so which sensors exist and where is a
    deployment concern and never hardcoded here. A read that fails, for any
    reason, becomes a not-ok result with a reason, so one unresponsive sensor
    never stops the others and never fabricates a value.
    """

    def __init__(self, bus, channels: list, *, sleep: Callable[[float], None] = time.sleep) -> None:
        self._bus = bus
        self._by_id = {c["id"]: c for c in channels}
        self._sleep = sleep

    def read_all(self, channel_ids: list) -> list:
        out: list = []
        for channel_id in channel_ids:
            out.append(self._read_one(channel_id))
        return out

    def _read_one(self, channel_id: str) -> SensorRead:
        cfg = self._by_id.get(channel_id)
        if cfg is None:
            return SensorRead(channel=channel_id, attempted=False, ok=False,
                              error="channel is not configured on this station")
        driver = cfg.get("driver", {}) or {}
        interface = driver.get("interface")
        if interface != "i2c":
            return SensorRead(channel=channel_id, attempted=False, ok=False,
                              error=f"unsupported sensor interface {interface!r}")
        sensor_type = str(driver.get("type", ""))
        if not sensor_type.startswith(_SUPPORTED_SENSOR_PREFIXES):
            return SensorRead(channel=channel_id, attempted=False, ok=False,
                              error=f"unsupported sensor type {sensor_type!r}")
        try:
            address = _parse_i2c_address(driver.get("address"))
        except (TypeError, ValueError):
            return SensorRead(channel=channel_id, attempted=False, ok=False,
                              error=f"invalid I2C address {driver.get('address')!r}")
        settle = float(driver.get("read_settle_seconds", 0.9))
        try:
            ok, value, error = _read_atlas_ezo(self._bus, address, settle_seconds=settle, sleep=self._sleep)
        except Exception as exc:  # noqa: BLE001 - a bus error on one channel is reported, not raised
            ok, value, error = False, None, f"{type(exc).__name__}: {exc}"
        return SensorRead(channel=channel_id, attempted=True, ok=ok, value=value, error=error)

    def close(self) -> None:
        try:
            self._bus.close()
        except Exception:  # noqa: BLE001
            pass


class _Smbus2Bus:
    """Wrap smbus2 to the plain ``write``/``read`` seam the sensor bank uses."""

    def __init__(self, bus_number: int) -> None:
        from smbus2 import SMBus  # noqa: PLC0415 - optional, only on a station with sensors

        self._SMBus = SMBus
        self._bus = SMBus(bus_number)

    def write(self, address: int, data: bytes) -> None:
        from smbus2 import i2c_msg  # noqa: PLC0415

        self._bus.i2c_rdwr(i2c_msg.write(address, data))

    def read(self, address: int, length: int) -> bytes:
        from smbus2 import i2c_msg  # noqa: PLC0415

        msg = i2c_msg.read(address, length)
        self._bus.i2c_rdwr(msg)
        return bytes(msg)

    def close(self) -> None:
        self._bus.close()


def build_sensor_bank(settings, station: dict):
    """Build the station's I2C sensor bank, or nothing when it cannot be built.

    Only channels present and switched on are wired. When the station has no
    enabled channels, or the I2C library is not installed, or the bus will not
    open, this returns nothing and the field runner falls back to the null sensor
    bank, so a station always runs and simply records no sensor channels. The
    reason is logged for the station operator. This builder never raises, because
    the field runner reads sensors outside a guard.
    """
    channels = [c for c in station.get("channels", []) if c.get("enabled", False)]
    if not channels:
        return None
    sensors_cfg = station.get("sensors", {}) or {}
    bus_number = int((sensors_cfg.get("i2c", {}) or {}).get("bus", DEFAULT_I2C_BUS))
    try:
        bus = _Smbus2Bus(bus_number)
    except ImportError:
        logger.warning(
            "environmental sensors are configured but the 'smbus2' package is not "
            "installed, so no sensor channels will be read on this station."
        )
        return None
    except Exception as exc:  # noqa: BLE001 - a bus that will not open leaves the station running without sensors
        logger.warning("I2C bus %s could not be opened (%s: %s); no sensor channels will be read.",
                       bus_number, type(exc).__name__, exc)
        return None
    return I2CSensorBank(bus, channels)


# ===========================================================================
# The satellite receiver (NMEA 0183 over a serial port)
# ===========================================================================


def _nmea_to_decimal(value: str, hemisphere: str) -> Optional[float]:
    """Convert an NMEA ddmm.mmmm coordinate and hemisphere to signed decimal degrees."""
    if not value:
        return None
    try:
        raw = float(value)
    except ValueError:
        return None
    degrees = int(raw // 100)
    minutes = raw - degrees * 100
    decimal = degrees + minutes / 60.0
    if hemisphere in ("S", "W"):
        decimal = -decimal
    return decimal


def _nmea_checksum_ok(sentence: str) -> bool:
    """Validate an NMEA sentence's ``*HH`` checksum when present; accept if absent."""
    if "*" not in sentence:
        return True
    body, _, checksum = sentence.partition("*")
    body = body.lstrip("$")
    try:
        expected = int(checksum[:2], 16)
    except ValueError:
        return False
    got = 0
    for ch in body:
        got ^= ord(ch)
    return got == expected


def _iso_from_nmea(utc_time: str, utc_date: str) -> Optional[str]:
    """Build a UTC ISO timestamp from an NMEA time (hhmmss.ss) and date (ddmmyy)."""
    if not utc_time or not utc_date or len(utc_time) < 6 or len(utc_date) < 6:
        return None
    try:
        hh, mm, ss = int(utc_time[0:2]), int(utc_time[2:4]), int(float(utc_time[4:]))
        dd, mo, yy = int(utc_date[0:2]), int(utc_date[2:4]), int(utc_date[4:6])
    except ValueError:
        return None
    dt = datetime(2000 + yy, mo, dd, hh, mm, min(ss, 59), tzinfo=timezone.utc)
    return dt.strftime(ISO_FORMAT)


def parse_nmea_fix(lines: list) -> GpsRead:
    """Parse a batch of NMEA sentences into a single fix result.

    Reads the recommended-minimum (RMC) sentence for position, validity, and the
    full UTC date and time, and the fix (GGA) sentence for altitude. The most
    recent valid RMC in the batch wins, so a stream that has just acquired a fix is
    reported as fixed. A batch with no valid position returns a not-ok result with
    a reason, so the capture records the location as not measured rather than
    inventing one. Malformed or wrong-checksum sentences are ignored rather than
    trusted.
    """
    lat = lon = elev = None
    utc_iso = None
    have_fix = False
    for line in lines:
        s = str(line).strip()
        if not s or not s.startswith("$") or not _nmea_checksum_ok(s):
            continue
        body = s.split("*", 1)[0]
        parts = body.split(",")
        talker = parts[0][3:] if len(parts[0]) >= 6 else ""
        if talker == "RMC" and len(parts) >= 10 and parts[2] == "A":
            rlat = _nmea_to_decimal(parts[3], parts[4])
            rlon = _nmea_to_decimal(parts[5], parts[6])
            if rlat is not None and rlon is not None:
                lat, lon = rlat, rlon
                utc_iso = _iso_from_nmea(parts[1], parts[9])
                have_fix = True
        elif talker == "GGA" and len(parts) >= 10 and parts[6] not in ("", "0"):
            try:
                elev = float(parts[9]) if parts[9] != "" else elev
            except ValueError:
                pass
    if have_fix:
        return GpsRead(attempted=True, ok=True, latitude=lat, longitude=lon,
                       elevation=elev, utc_time=utc_iso)
    return GpsRead(attempted=True, ok=False, error="no valid satellite fix in the NMEA stream")


class NmeaGpsSource:
    """A `GpsSource` that reads NMEA sentences from a serial receiver.

    The port is any object with ``read_lines() -> list[str]`` and a ``close()``;
    the real one wraps pyserial, a test one scripts the sentences. Each read drains
    the sentences waiting on the port and returns the most recent valid fix, or a
    structured not-ok result when there is none.
    """

    def __init__(self, port) -> None:
        self._port = port

    def read(self) -> GpsRead:
        try:
            lines = self._port.read_lines()
        except Exception as exc:  # noqa: BLE001 - a serial read error is reported, not raised
            return GpsRead(attempted=True, ok=False, error=f"{type(exc).__name__}: {exc}")
        return parse_nmea_fix(lines)

    def close(self) -> None:
        try:
            self._port.close()
        except Exception:  # noqa: BLE001
            pass


class _SerialLines:
    """Wrap a pyserial port to the plain ``read_lines`` seam the GPS source uses."""

    def __init__(self, serial_port) -> None:
        self._serial = serial_port
        self._buffer = ""

    def read_lines(self) -> list:
        waiting = getattr(self._serial, "in_waiting", 0) or 0
        chunk = self._serial.read(waiting) if waiting else b""
        if chunk:
            self._buffer += chunk.decode("ascii", "ignore")
        lines = self._buffer.split("\n")
        self._buffer = lines.pop()  # keep any partial trailing line for next read
        return [ln.strip("\r") for ln in lines]

    def close(self) -> None:
        self._serial.close()


class GpsdGpsSource:
    """A `GpsSource` that reads a fix from a running gpsd daemon.

    gpsd already parses the receiver and serves a JSON TPV (time-position-velocity)
    report. The backend is any object with ``read_tpv() -> dict`` and ``close()``;
    the real one is a small socket client, a test one scripts the report. A gpsd
    fix mode of 2 (2D) or 3 (3D) with a position is a fix; anything less is
    reported as not-ok, so the location is recorded as not measured rather than
    invented. This is the alternative to the serial NMEA source; the choice is a
    configuration one.
    """

    def __init__(self, backend) -> None:
        self._backend = backend

    def read(self) -> GpsRead:
        try:
            report = self._backend.read_tpv()
        except Exception as exc:  # noqa: BLE001 - a gpsd read error is reported, not raised
            return GpsRead(attempted=True, ok=False, error=f"{type(exc).__name__}: {exc}")
        if not report:
            return GpsRead(attempted=True, ok=False, error="gpsd returned no position report")
        lat, lon = report.get("lat"), report.get("lon")
        mode = int(report.get("mode", 0) or 0)
        ok = mode >= 2 and lat is not None and lon is not None
        return GpsRead(
            attempted=True, ok=ok,
            latitude=lat if ok else None,
            longitude=lon if ok else None,
            elevation=report.get("alt") if ok else None,
            utc_time=report.get("time") if ok else None,
            error=None if ok else "no gpsd fix yet",
        )

    def close(self) -> None:
        close = getattr(self._backend, "close", None)
        if close:
            try:
                close()
            except Exception:  # noqa: BLE001
                pass


class _GpsdBackend:
    """A tiny gpsd client over a socket: connect, enable the watch, read one TPV.

    Uses only the standard library, so gpsd support needs no extra Python package,
    only the gpsd daemon running on the station.
    """

    def __init__(self, host: str, port: int) -> None:
        import socket  # noqa: PLC0415 - stdlib

        self._sock = socket.create_connection((host, int(port)), timeout=5.0)
        self._sock.sendall(b'?WATCH={"enable":true,"json":true}\n')
        self._buffer = b""

    def read_tpv(self):
        import json  # noqa: PLC0415 - stdlib

        deadline = time.time() + 5.0
        while time.time() < deadline:
            self._buffer += self._sock.recv(4096)
            while b"\n" in self._buffer:
                line, self._buffer = self._buffer.split(b"\n", 1)
                try:
                    obj = json.loads(line.decode("ascii", "ignore"))
                except ValueError:
                    continue
                if obj.get("class") == "TPV":
                    return obj
        return None

    def close(self) -> None:
        try:
            self._sock.close()
        except Exception:  # noqa: BLE001
            pass


def build_gps_source(settings, station: dict):
    """Build the station's satellite receiver source, or nothing when it cannot.

    Built only when the station has GPS switched on. The driver's ``interface``
    selects the transport: ``serial`` (the default) reads NMEA from a serial
    receiver at ``port``/``baud``, and ``gpsd`` reads from a gpsd daemon at
    ``host``/``port`` (defaulting to localhost:2947). When the needed library is
    absent or the receiver will not open, this returns nothing and the field runner
    falls back to the null source, so the station runs and records its location as
    not measured (a station with fixed, surveyed coordinates still carries them
    from configuration). The reason is logged. This builder never raises, because
    the field runner reads GPS outside a guard.
    """
    gps_cfg = (station.get("sensors", {}) or {}).get("gps", {}) or {}
    if not gps_cfg.get("enabled", False):
        return None
    driver = gps_cfg.get("driver", {}) or {}
    interface = str(driver.get("interface", "serial")).lower()

    if interface == "gpsd":
        host = str(driver.get("host", "127.0.0.1"))
        port = int(driver.get("port", 2947))
        try:
            return GpsdGpsSource(_GpsdBackend(host, port))
        except Exception as exc:  # noqa: BLE001 - an unreachable daemon leaves the station running without GPS
            logger.warning("gpsd at %s:%s could not be reached (%s: %s); the receiver will not be read.",
                           host, port, type(exc).__name__, exc)
            return None

    port_name = str(driver.get("port", DEFAULT_GPS_PORT))
    baud = int(driver.get("baud", DEFAULT_GPS_BAUD))
    try:
        import serial  # noqa: PLC0415 - optional, only on a station with a serial receiver
    except ImportError:
        logger.warning(
            "GPS is enabled with a serial receiver but the 'pyserial' package is "
            "not installed, so the receiver will not be read on this station."
        )
        return None
    try:
        port = serial.Serial(port_name, baud, timeout=0)
    except Exception as exc:  # noqa: BLE001 - a port that will not open leaves the station running without GPS
        logger.warning("GPS port %s could not be opened at %s baud (%s: %s); the receiver will not be read.",
                       port_name, baud, type(exc).__name__, exc)
        return None
    return NmeaGpsSource(_SerialLines(port))
