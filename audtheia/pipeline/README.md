# audtheia/pipeline

Field-station runtime. These modules run on the Raspberry Pi and turn the live
camera and sensor streams into observation records. Nothing in this folder runs a
language model.

| File | Runs on | Role |
|------|---------|------|
| monitor.py | Field station (Pi) | Reads the camera, runs the vision model on the Hailo NPU, joins each tracked animal across frames into one event with ByteTrack on the CPU, and fires the capture of the other sensor streams when a track closes. |
| acoustic.py | Field station (Pi) | Runs the selected acoustic model (BirdNET for terrestrial and avian sites, a marine passive-acoustic model underwater). It captures an audio clip for a vision event and opens its own event on an acoustic detection. |
| environment.py | Field station (Pi) | Reads the GPS fix, disciplines the clock to coordinated universal time, and reads the configured water, air, or soil sensors when an event fires. |

The camera and the accelerator stay on separate paths so detection never blocks on
the slower captures.
