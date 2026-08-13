---
name: Field hardware and on-device validation
about: Report or help validate a field-station driver on real hardware
title: "[field] "
labels: field-hardware
---

Audtheia's field drivers are implemented and are being validated on physical
hardware. This is the place to report results or offer help. See
[`docs/field-drivers.md`](../../docs/field-drivers.md) for the drivers and the
validation checklist.

**Which sense**
- [ ] Live audio (microphone or hydrophone)
- [ ] Environmental sensors (I2C)
- [ ] GPS (NMEA receiver)
- [ ] Accelerator detector (Hailo `.hef`)

**Hardware**
- Raspberry Pi model and OS:
- AI HAT+ 2 / HailoRT version (if relevant):
- Sensor, hydrophone, or receiver models (if relevant):

**What you observed**
What worked, what did not, and any error or log output (with secrets removed).

**Configuration**
The relevant `sensors` / `channels` / `models.visual_pi` settings (redact any
credential).
