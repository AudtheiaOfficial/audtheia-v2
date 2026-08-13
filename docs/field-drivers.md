# Field-station hardware drivers

This document describes the drivers that give a Raspberry Pi 5 field station its
senses, and the on-device validation checklist to confirm each one against real
hardware. The drivers are in
[`audtheia/pipeline/field_drivers.py`](../audtheia/pipeline/field_drivers.py) and
the accelerator detector is in
[`audtheia/pipeline/drivers.py`](../audtheia/pipeline/drivers.py); both are
re-exported so a station's drivers module exposes them where the field runner
looks.

## Status

The field drivers are implemented against the documented library and device
protocols and unit-tested against scripted hardware backends (`tests/test_field_drivers.py`),
but they have not yet been run on physical hardware. They are marked pending
on-device validation. The design keeps every hardware call behind an injected
backend and a lazy import, so validation is a matter of confirming a few device
specifics, not of writing new logic. A station always runs: a missing library or an
unopened device leaves that sense inactive rather than stopping capture.

## What each driver does

**Live audio (`build_audio_source`).** Captures from a microphone or a
hydrophone-through-a-converter using `sounddevice`, mixes to mono, and hands the
acoustic monitor blocks at the acoustic model's sample rate. A file or URL audio
spec is delegated to the desktop audio source instead, so the same station can be
bench-tested from a recording and run live in the field with only a configuration
change.

**Environmental sensors (`build_sensor_bank`).** Reads the station's enabled I2C
channels over `smbus2`, driven entirely by configuration. The bundled reader
speaks the Atlas Scientific EZO command set, which is uniform across the circuit
types (temperature, pH, dissolved oxygen, conductivity, and the rest), so one
reader serves the family and the specific circuit is only a label in
configuration. A failed read returns a structured not-ok result with a reason,
never a fabricated value.

**Satellite receiver (`build_gps_source`).** Reads NMEA 0183 sentences from a
serial receiver using `pyserial`, recovers latitude, longitude, altitude, and the
UTC that disciplines the station clock, and validates each sentence's checksum. No
fix is reported as not measured rather than invented; a station with fixed,
surveyed coordinates still carries them from configuration.

**Accelerator detector (`build_detector`, Hailo path).** Runs the station's
compiled `.hef` model on the Hailo NPU through HailoRT. The model file extension
selects the runtime, so a `.hef` screens on the accelerator and an ONNX model
screens on the desktop CPU; the node role selects which model entry to load
(`models.visual_pi` on a field station, `models.visual_desktop` on the desktop).
The decode auto-detects whether the `.hef` was compiled with on-chip non-maximum
suppression (already-decoded boxes) or outputs raw YOLO tensors (decoded on the Pi
exactly as the desktop detector does), so either compile choice works with no code
change.

## Configuration

All hardware specifics live in a station's configuration, never in code. The keys
below extend the station block in `config/settings.json`; sensible defaults apply
when a key is absent.

```jsonc
{
  "sensors": {
    "audio": { "enabled": true, "driver": { "device": null, "channels": 1, "block_seconds": 1.0 } },
    "gps":   { "enabled": true, "driver": { "interface": "serial", "port": "/dev/serial0", "baud": 9600 } },
    "i2c":   { "bus": 1 }
  },
  "channels": [
    {
      "id": "water_temp_c", "unit": "degC", "marine": true, "enabled": true,
      "driver": { "interface": "i2c", "address": "0x66", "type": "atlas_ezo_rtd", "read_settle_seconds": 0.9 }
    }
  ],
  "models": {
    "visual_pi": { "path": "models/visual/pi/detector.hef", "version": "1", "input_size": [640, 640] }
  }
}
```

- Audio `device` is a sound-device index or name; `null` selects the system default
  input. `channels` and `block_seconds` shape each block.
- GPS `interface` is `serial` (the default) or `gpsd`. A serial receiver names its
  `port` and `baud`; a `gpsd` receiver names the daemon `host` and `port`
  (defaulting to `127.0.0.1:2947`) and needs no Python serial package, only a
  running gpsd.
- The I2C `bus` is the Pi's user bus (1 by default). Each channel names its
  `address` and EZO `type`; `read_settle_seconds` is how long the circuit needs to
  take a reading before it is read back.
- `models.visual_pi.path` is the compiled `.hef`. `input_size` is optional; the
  detector reads the input dimensions from the `.hef` when it can.

## Required system packages

The hardware libraries are optional and install on the station, not on the desktop:

```
pip install sounddevice smbus2 pyserial
# and the Raspberry Pi AI HAT+ 2 / HailoRT stack for the accelerator:
#   sudo apt install hailo-all
# sounddevice needs PortAudio:
#   sudo apt install libportaudio2
```

## On-device validation checklist

Run through this once with the physical hardware attached. Each step confirms a
single device specific; none requires changing the driver logic.

1. **Audio.** Set `sensors.audio.driver.device` to the capture device (list them
   with `python -c "import sounddevice; print(sounddevice.query_devices())"`).
   Confirm live acoustic detections appear on the Audio tab and that the block rate
   matches the acoustic model's sample rate.
2. **I2C sensors.** Confirm each circuit's address with `i2cdetect -y 1` and that
   it matches the channel's `driver.address`. Confirm each channel reports a
   plausible value, and that unplugging a sensor produces a not-ok read rather than
   a stall.
3. **GPS.** Confirm the receiver's serial port and baud, that raw NMEA appears on
   the port, and that a fix populates latitude, longitude, altitude, and a UTC that
   advances the station clock out of the provisional state.
4. **Accelerator.** Place the compiled `.hef` at `models.visual_pi.path` with a
   labels file beside it. Confirm the model loads on the NPU, that detections carry
   the right class names and plausible boxes, and, if boxes look shifted or scaled,
   check whether the `.hef` was compiled with on-chip NMS and confirm the decode
   path the detector chose. Confirm the `infer` call matches the installed HailoRT
   version.

When all four pass on real hardware, update the README project-status note from
"validating on hardware" to field-proven, and record the validated device details
in your deployment notes.
