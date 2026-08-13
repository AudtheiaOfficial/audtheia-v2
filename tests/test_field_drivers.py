"""Verification for the field-station hardware drivers.

Path: tests/test_field_drivers.py

The field drivers wrap real hardware, which is not present here, so each driver is
built to take its backend as an argument and this suite exercises the reading and
parsing logic against scripted fakes: a fake I2C bus, a fake serial receiver, and a
fake audio stream. What is proven is exactly the logic that does not need a device:

  - a live audio block is mixed to mono, carries the model rate and a timestamp,
    and an exhausted stream ends cleanly,
  - the Atlas EZO I2C read decodes a success, and reports every failure status
    (still processing, no data, command error, unreadable value) as a not-ok
    result rather than a fabricated value,
  - the sensor bank honours configuration: an unsupported interface or type, an
    unconfigured channel, and a bad address are each reported, never guessed,
  - NMEA parsing recovers latitude, longitude, altitude, and UTC from a valid
    stream, rejects a void fix and a bad checksum, and reports no-fix as not-ok,
    so a location is never invented.

Standard library plus numpy only; no hardware, no database, no network. This does
not exercise the lazy hardware imports (sounddevice, smbus2, pyserial), which are
the part that needs on-device validation.

Run: python tests/test_field_drivers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from audtheia.pipeline.field_drivers import (  # noqa: E402
    GpsdGpsSource,
    I2CSensorBank,
    LiveAudioSource,
    NmeaGpsSource,
    _is_live_audio_spec,
    _nmea_checksum_ok,
    _nmea_to_decimal,
    _parse_i2c_address,
    _read_atlas_ezo,
    parse_nmea_fix,
)

from audtheia.pipeline.drivers import (  # noqa: E402
    HailoYoloDetector,
    _hailo_nms_to_detections,
    _looks_like_raw_yolo,
    _screening_entry,
    _single_output,
    _yolo_raw_to_detections,
)

CHECKS = {"passed": 0, "failed": 0}


def check(label: str, condition: bool) -> None:
    if condition:
        CHECKS["passed"] += 1
        print(f"  PASS  {label}")
    else:
        CHECKS["failed"] += 1
        print(f"  FAIL  {label}")


def _close(a, b, tol=1e-4) -> bool:
    return a is not None and b is not None and abs(a - b) <= tol


# --- fakes -----------------------------------------------------------------


class FakeBus:
    def __init__(self, responses):
        self.responses = dict(responses)
        self.writes = []

    def write(self, address, data):
        self.writes.append((address, bytes(data)))

    def read(self, address, length):
        return self.responses.get(address, b"")

    def close(self):
        pass


class FakeAudioStream:
    def __init__(self, blocks):
        self._blocks = list(blocks)
        self._i = 0

    def read(self, frames):
        if self._i >= len(self._blocks):
            return np.zeros((0,), dtype=np.float32)
        block = self._blocks[self._i]
        self._i += 1
        return block

    def close(self):
        pass


class FakePort:
    def __init__(self, lines):
        self._lines = list(lines)

    def read_lines(self):
        return self._lines

    def close(self):
        pass


def _nmea(body: str) -> str:
    """Build an NMEA sentence with a correct checksum from its body (no leading $)."""
    checksum = 0
    for ch in body:
        checksum ^= ord(ch)
    return f"${body}*{checksum:02X}"


# --- audio -----------------------------------------------------------------


def test_live_audio() -> None:
    print("\nLive audio mixes to mono, carries the rate, and ends cleanly")
    stereo = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)  # 2 frames, 2 channels
    src = LiveAudioSource(FakeAudioStream([stereo]), sample_rate=48000, block_frames=2,
                          clock_now=lambda: "2026-08-12T12:00:00Z")
    block = src.read()
    check("a block is returned", block is not None)
    check("the block carries the model rate", block.sample_rate == 48000)
    check("stereo is mixed to mono", block.samples.ndim == 1 and block.samples.shape[0] == 2)
    check("the mono mix is the channel average", _close(float(block.samples[0]), 0.3) and _close(float(block.samples[1]), 0.7))
    check("the block is stamped", block.captured_at == "2026-08-12T12:00:00Z")
    check("an exhausted stream returns None", src.read() is None)

    # A (frames, overflowed) tuple, as the real sounddevice stream returns.
    mono = np.array([0.1, 0.1], dtype=np.float32)
    src2 = LiveAudioSource(FakeAudioStream([(mono, False)]), sample_rate=16000, block_frames=2)
    b2 = src2.read()
    check("a (frames, overflowed) tuple is accepted", b2 is not None and _close(float(b2.samples[0]), 0.1))


# --- sensors ---------------------------------------------------------------


def test_atlas_ezo_read() -> None:
    print("\nAtlas EZO reads decode a value and report every failure honestly")
    nosleep = lambda _s: None  # noqa: E731 - a stand-in for time.sleep in the test

    ok, value, err = _read_atlas_ezo(FakeBus({0x66: b"\x01" + b"25.125\x00\x00"}), 0x66, settle_seconds=0.0, sleep=nosleep)
    check("a success decodes the value", ok and _close(value, 25.125) and err is None)

    ok, value, err = _read_atlas_ezo(FakeBus({0x66: b"\xfe"}), 0x66, settle_seconds=0.0, sleep=nosleep)
    check("still-processing is not-ok", not ok and value is None and "processing" in err)

    ok, value, err = _read_atlas_ezo(FakeBus({0x66: b"\xff"}), 0x66, settle_seconds=0.0, sleep=nosleep)
    check("no-data is not-ok", not ok and "no data" in err)

    ok, value, err = _read_atlas_ezo(FakeBus({0x66: b"\x02"}), 0x66, settle_seconds=0.0, sleep=nosleep)
    check("a command error is not-ok", not ok and "command error" in err)

    ok, value, err = _read_atlas_ezo(FakeBus({0x66: b"\x01" + b"abc\x00"}), 0x66, settle_seconds=0.0, sleep=nosleep)
    check("an unreadable value is not-ok", not ok and "unreadable" in err)

    ok, value, err = _read_atlas_ezo(FakeBus({}), 0x66, settle_seconds=0.0, sleep=nosleep)
    check("no response is not-ok", not ok and "no response" in err)


def test_sensor_bank() -> None:
    print("\nThe sensor bank honours configuration and never guesses")
    channels = [
        {"id": "water_temp_c", "enabled": True, "driver": {"interface": "i2c", "address": "0x66", "type": "atlas_ezo_rtd"}},
        {"id": "ph", "enabled": True, "driver": {"interface": "i2c", "address": "0x63", "type": "atlas_ezo_ph"}},
        {"id": "soil", "enabled": True, "driver": {"interface": "onewire", "type": "ds18b20"}},
        {"id": "bad_addr", "enabled": True, "driver": {"interface": "i2c", "address": "not-hex", "type": "atlas_ezo_do"}},
    ]
    bus = FakeBus({0x66: b"\x01" + b"25.10\x00", 0x63: b"\x01" + b"8.02\x00"})
    bank = I2CSensorBank(bus, channels, sleep=lambda _s: None)
    reads = {r.channel: r for r in bank.read_all(["water_temp_c", "ph", "soil", "bad_addr", "missing"])}

    check("temperature reads its value", reads["water_temp_c"].ok and _close(reads["water_temp_c"].value, 25.10))
    check("pH reads its value", reads["ph"].ok and _close(reads["ph"].value, 8.02))
    check("a non-I2C interface is unsupported, not attempted",
          reads["soil"].attempted is False and reads["soil"].ok is False and "interface" in reads["soil"].error)
    check("a bad address is reported, not attempted",
          reads["bad_addr"].attempted is False and "address" in reads["bad_addr"].error)
    check("an unconfigured channel is reported, not attempted",
          reads["missing"].attempted is False and "not configured" in reads["missing"].error)
    check("the correct read command was written to each sensor",
          all(w[1] == b"R" for w in bus.writes) and len(bus.writes) == 2)


# --- gps -------------------------------------------------------------------


def test_nmea_helpers() -> None:
    print("\nNMEA coordinate and checksum helpers")
    check("north/east is positive decimal", _close(_nmea_to_decimal("4807.038", "N"), 48.1173, tol=1e-3))
    check("south/west is negative decimal", _close(_nmea_to_decimal("01131.000", "W"), -11.51667, tol=1e-3))
    good = _nmea("GPRMC,120000,A,4807.038,N,01131.000,E,0.0,0.0,120826,,")
    check("a correct checksum validates", _nmea_checksum_ok(good))
    check("a wrong checksum is rejected", not _nmea_checksum_ok(good[:-1] + "0"))
    check("no checksum is accepted", _nmea_checksum_ok("$GPRMC,120000,A,4807.038,N,01131.000,E"))


def test_nmea_fix() -> None:
    print("\nNMEA parsing recovers a fix and refuses to invent one")
    lines = [
        _nmea("GPRMC,120000,A,4807.038,N,01131.000,E,0.0,0.0,120826,,"),
        _nmea("GPGGA,120000,4807.038,N,01131.000,E,1,08,0.9,12.5,M,0.0,M,,"),
    ]
    fix = parse_nmea_fix(lines)
    check("a valid stream is a fix", fix.attempted and fix.ok)
    check("latitude is recovered", _close(fix.latitude, 48.1173, tol=1e-3))
    check("longitude is recovered", _close(fix.longitude, 11.51667, tol=1e-3))
    check("altitude is recovered from GGA", _close(fix.elevation, 12.5))
    check("UTC is recovered from RMC", fix.utc_time is not None and fix.utc_time.startswith("2026-08-12T12:00:00"))

    void = parse_nmea_fix([_nmea("GPRMC,120000,V,,,,,0.0,0.0,120826,,")])
    check("a void fix is not-ok", void.attempted and not void.ok and "no valid" in void.error)

    garbage = parse_nmea_fix(["not a sentence", "$GPRMC,broken", _nmea("GPRMC,120000,A,4807.038,N,01131.000,E,0.0,0.0,120826,,")[:-1] + "0"])
    check("malformed and bad-checksum lines yield no fix", not garbage.ok)

    src = NmeaGpsSource(FakePort(lines))
    check("the source returns the parsed fix", src.read().ok)
    check("an empty stream reports no fix", not NmeaGpsSource(FakePort([])).read().ok)


class FakeGpsd:
    def __init__(self, report):
        self._report = report

    def read_tpv(self):
        return self._report

    def close(self):
        pass


def test_gpsd() -> None:
    print("\ngpsd source maps a TPV report and refuses a poor fix")
    good = GpsdGpsSource(FakeGpsd({"class": "TPV", "mode": 3, "lat": 48.1173, "lon": 11.5167,
                                   "alt": 12.5, "time": "2026-08-12T12:00:00.000Z"}))
    r = good.read()
    check("a 3D fix is ok", r.attempted and r.ok)
    check("gpsd position is mapped", _close(r.latitude, 48.1173) and _close(r.longitude, 11.5167) and _close(r.elevation, 12.5))
    check("gpsd time carries through", r.utc_time == "2026-08-12T12:00:00.000Z")
    check("a no-fix mode is not-ok", not GpsdGpsSource(FakeGpsd({"mode": 1})).read().ok)
    check("an empty report is not-ok", not GpsdGpsSource(FakeGpsd(None)).read().ok)


# --- config helpers --------------------------------------------------------


def test_config_helpers() -> None:
    print("\nConfiguration helpers classify specs and addresses")
    check("an empty spec is a live device", _is_live_audio_spec(None) and _is_live_audio_spec(""))
    check("device words are live", _is_live_audio_spec("hydrophone") and _is_live_audio_spec("device:2"))
    check("a file path is not live", not _is_live_audio_spec("C:/clip.wav") and not _is_live_audio_spec("https://x/y"))
    check("a hex address parses", _parse_i2c_address("0x66") == 0x66 and _parse_i2c_address(99) == 99)


class FakeSettings:
    def __init__(self, role):
        self._role = role

    @property
    def node_role(self):
        return self._role

    def desktop_visual_model(self, station):
        return dict(station.get("models", {}).get("visual_desktop", {}))


def _raw_yolo_tensor():
    """A raw YOLO output (1, 4+classes, anchors) with one strong box, class 0.

    The box is at input-pixel centre 320,320 with size 64, which for a 640x480
    frame letterboxed into 640x640 (scale 1, pad y 80) maps to 288,208,352,272.
    """
    arr = np.zeros((1, 6, 10), dtype=np.float32)  # 4 box + 2 classes, 10 anchors
    arr[0, :, 0] = [320, 320, 64, 64, 0.9, 0.0]
    return arr


def test_hailo_raw_path() -> None:
    print("\nHailo raw-tensor decode matches the desktop decode")
    arr = _raw_yolo_tensor()
    single = _single_output(arr)
    check("a single raw tensor is recognized", _looks_like_raw_yolo(single, 2))
    dets = _yolo_raw_to_detections(arr, 1.0, (0, 80), 640, 480, {0: "a", 1: "b"}, 0.25, 0.45)
    check("one box survives the confidence floor", len(dets) == 1)
    d = dets[0]
    check("the box is mapped out of the letterbox to frame pixels",
          _close(d.x1, 288) and _close(d.y1, 208) and _close(d.x2, 352) and _close(d.y2, 272))
    check("the class and score carry through", d.class_id == 0 and d.class_name == "a" and _close(d.confidence, 0.9))


def test_hailo_nms_path() -> None:
    print("\nHailo on-chip-NMS decode maps normalized per-class boxes")
    # class 0 has one box; the row is [y_min, x_min, y_max, x_max, score] normalized.
    outputs = [np.array([[0.45, 0.45, 0.55, 0.55, 0.8]], dtype=np.float32), np.zeros((0, 5), dtype=np.float32)]
    check("a per-class list is not mistaken for a raw tensor", _single_output(outputs) is None)
    dets = _hailo_nms_to_detections(outputs, 1.0, (0, 80), 640, 480, 640, 640, {0: "a", 1: "b"}, 0.25)
    check("one decoded box is produced", len(dets) == 1)
    d = dets[0]
    check("the normalized box maps to frame pixels",
          _close(d.x1, 288) and _close(d.y1, 208) and _close(d.x2, 352) and _close(d.y2, 272))
    check("the class index becomes the class", d.class_id == 0 and d.class_name == "a")


def test_hailo_detector_routes() -> None:
    print("\nThe Hailo detector auto-routes raw and NMS outputs")
    det = HailoYoloDetector(object(), class_names={0: "a", 1: "b"}, input_size=(640, 640))
    raw = det._postprocess(_raw_yolo_tensor(), 1.0, (0, 80), 640, 480)
    nms = det._postprocess([np.array([[0.45, 0.45, 0.55, 0.55, 0.8]], dtype=np.float32),
                            np.zeros((0, 5), dtype=np.float32)], 1.0, (0, 80), 640, 480)
    check("a raw tensor routes to the raw decode", len(raw) == 1 and _close(raw[0].x1, 288))
    check("a per-class list routes to the NMS decode", len(nms) == 1 and _close(nms[0].x1, 288))


def test_screening_entry() -> None:
    print("\nThe screening model is chosen by node role")
    station = {"models": {"visual_pi": {"path": "m.hef"}, "visual_desktop": {"path": "m.onnx"}}}
    check("a field station screens on its .hef",
          _screening_entry(FakeSettings("pi"), station).get("path") == "m.hef")
    check("the desktop screens on its ONNX model",
          _screening_entry(FakeSettings("desktop"), station).get("path") == "m.onnx")


def main() -> int:
    print("=" * 72)
    print("Field-station hardware drivers (logic, against scripted fakes)")
    print("=" * 72)
    test_live_audio()
    test_atlas_ezo_read()
    test_sensor_bank()
    test_nmea_helpers()
    test_nmea_fix()
    test_gpsd()
    test_config_helpers()
    test_hailo_raw_path()
    test_hailo_nms_path()
    test_hailo_detector_routes()
    test_screening_entry()
    print("\n" + "=" * 72)
    print(f"RESULT: {CHECKS['passed']} passed, {CHECKS['failed']} failed")
    print("=" * 72)
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
