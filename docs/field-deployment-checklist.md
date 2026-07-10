# Field Deployment Checklist

This is the practical checklist for standing up a real Audtheia field station and confirming it works, from the bench before you leave, through mounting it at the site, to verifying that clean, trustworthy data is reaching your desktop. It assumes you have already built a station following the [hardware guide](hardware.md) and prepared its detection model following the [custom models guide](custom-models.md). Work through it in order; the goal is that a researcher can deploy without further help and know, before leaving the site, that the station is recording good data.

Copy this list for each deployment and check items off as you go. The notes under each section explain why the item matters.

## 1. On the bench, before you travel

Do all of this at home or in the lab, where a problem is cheap to fix.

- [ ] The station is assembled and the Raspberry Pi boots with the accelerator, camera, hydrophone or microphone, GPS, and sensors connected.
- [ ] The desktop has provisioned the station over the network, and running the station once completes a clean quality-control sweep.
- [ ] The field detection model (the compiled `.hef`) is in place and named in the station's settings, and the acoustic model is selected for your environment.
- [ ] The sensors you actually have are enabled in the settings, and the ones you do not have are disabled, so the record never expects a channel that cannot report.
- [ ] A test in front of the camera produces a detection, writes one observation, and that observation appears on the desktop after a sync.
- [ ] The privacy setting that discards human detections is on, and, if your model has a human class, its label is named so the discard actually applies.

## 2. Power and autonomy

A station that browns out and misses a season of data is the most expensive failure there is, so size power for the worst case, not the average.

- [ ] You have measured the station's real average power draw with an inline meter, using your actual camera settings and a realistic event rate.
- [ ] The solar panel is sized for your site's worst-month peak sun hours, with margin, using the method in the [hardware guide](hardware.md).
- [ ] The battery is sized for your target days of autonomy (three to five days is prudent for an unattended marine station in a storm-prone region), and it is fully charged before you leave.
- [ ] The charge controller, regulator, and connectors are rated for the panel and battery, and every outdoor connection is sealed.

## 3. Biofouling protection (marine sites)

In warm, productive water the camera's optical port fouls within days to weeks, and a fouled lens sees nothing.

- [ ] The optical port has its anti-fouling measures applied (a copper shroud or faceplate, a transparent coating on the glass, and a scheduled ultraviolet treatment if power allows), following the [hardware guide](hardware.md).
- [ ] The port is recessed or flush where possible, so it fouls more slowly and is easier to clean.
- [ ] You have a cleaning cadence planned (for example every one to two weeks during the fouling season) and a way to reach the station to do it.
- [ ] You understand that the acoustic trigger keeps events flowing while the camera is degraded, so a fouling lens means reduced vision, not a dead station.

## 4. Network, hotspot, and access

- [ ] The station broadcasts its own Wi-Fi network, and you can connect a phone or laptop to it and open the station's local address to see live status.
- [ ] Key-based access from the desktop to the station is set up, so later connections do not need a password.
- [ ] You know how you will bring the desktop and the station together for syncs: by carrying the laptop within range, by returning the station's drive, or by a shared network at a base.

## 5. Time and location

- [ ] The GPS has a satellite fix and the station's clock is disciplined to coordinated universal time. Timestamps taken before a fix are marked as provisional, which is expected, but a fix should be present at deployment.
- [ ] The real-time-clock backup battery is installed, so the clock survives a reboot between fixes.
- [ ] The station's recorded position matches where you actually placed it.

## 6. Mounting and physical deployment

- [ ] The station is mounted securely against surge, current, and weather, so it does not shift; a station that moves is one whose field of view and GPS track no longer match its records.
- [ ] The camera is aimed at the habitat or substrate you intend to study.
- [ ] The solar panel has a clear sky angle and a surface you can wipe, and you expect its output to fall as it collects grime and salt.
- [ ] The enclosure is closed, watertight, and its cable glands are sealed, with the accelerator's cooling given a thermal path.

## 7. First power-on at the site

- [ ] The station boots, its service starts, and the senses you configured report as active.
- [ ] A first real detection appears in the live feed within a reasonable time for your site.
- [ ] The storage status shows the buffer filling as expected, with plenty of headroom.

## 8. The first sync and the cold-start dream pass

- [ ] The desktop connects to the station and pulls its records; the pull is append-only and resumable, so an interrupted sync leaves no gaps and no duplicates.
- [ ] The pulled observations appear in the desktop database, organized under the station's identifier.
- [ ] You run a first longitudinal pass on the desktop. Expect it to consolidate the early record and to propose few or no candidate patterns at first: a pattern needs a data span behind it, so the value of the pass grows as the record lengthens. This cold start is normal, not a fault.

## 9. Verifying the data is trustworthy

Before you rely on anything, confirm the record is clean.

- [ ] Every stored value carries its provenance and its quality-control status, and measured readings are clearly separated from any downstream interpretation.
- [ ] Marine sensor channels carry their quality flags, so a suspect reading is marked rather than trusted silently.
- [ ] The desktop verification has run over the observations, and the verified detections are marked as such.
- [ ] A first report generates cleanly, with every value labeled by its source and the taxonomy and status snapshot dates disclosed.
- [ ] You have logged the deployment date, the site, and your anti-fouling and cleaning plan, so a later drop in visual detections can be read against the fouling and cleaning record rather than mistaken for a real change in the animals.

## 10. Ongoing, once it is running

- [ ] You clean the optical port on the cadence you planned, and you log each cleaning.
- [ ] You sync on a cadence that keeps the station's buffer from filling, remembering that the station never deletes data it has not yet copied to the desktop; at a hard limit it pauses new capture and alerts rather than dropping records.
- [ ] You check the battery and panel on each visit, and clear grime from the panel.
- [ ] You keep track of which model versions are deployed, so a change in detections can be attributed to a model update rather than to the environment.

## Where to go next

- To build or revise the station, see the [hardware guide](hardware.md).
- To prepare or retrain its models, see the [custom models guide](custom-models.md).
- To understand what the desktop does with the record over time, see the [dream pass guide](dream-pass.md).
- To set up the desktop and connect a station, see the [README](../README.md).
