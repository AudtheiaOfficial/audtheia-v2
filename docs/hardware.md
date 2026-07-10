# Hardware Guide

This guide describes how to build a physical Audtheia field station: the parts, how they fit together, how to size the power system so a station can run unattended for months, and how to keep a marine camera clear of biofouling. It is written for a researcher, not an engineer, and no soldering or programming is required for the reference build.

You do not need any of this hardware to try Audtheia. The desktop application runs the full pipeline on an ordinary computer with no field devices at all, capturing from a webcam, a network stream, or a video file. See the main [README](../README.md) for that path. Build a field station when you want continuous, low power, solar capable monitoring at a fixed site over weeks and months, which is what makes the science compound.

## The two ways to run Audtheia, side by side

| | Desktop hardware-free mode | Field station |
|---|---|---|
| Where it runs | Any personal computer | A Raspberry Pi 5 in the field, plus your computer as the hub |
| What it watches | A webcam, a network or web page stream, or a video file | A dedicated camera and, for marine sites, a hydrophone, at a fixed location |
| Detection model | An RF-DETR model run through ONNX Runtime on the computer | A custom detector compiled to the accelerator, run on the Hailo NPU |
| Power | Wall power | Solar and battery, or wall power near shore |
| Best for | Trying the system, teaching, comparing methods, analyzing recorded footage | Real longitudinal field studies, Marine Protected Area work |

Both run the same pipeline: detection triggers a complete multimodal observation, quality control finalizes it, the desktop verifies and dreams over the record, and reports are generated with every value labeled by its provenance. The field station simply adds real sensors and real deployment.

## What a field station is made of

A station has five parts: the compute and accelerator, the camera, the microphone or hydrophone, the location and environmental sensors, and the power and housing that keep it all running outdoors.

### Compute and accelerator

The brain of a station is a Raspberry Pi 5 with the Raspberry Pi AI HAT+ 2 attached. The [AI HAT+ 2](https://www.raspberrypi.com/products/ai-hat-plus-2/) carries a Hailo-10H neural accelerator rated at 40 TOPS with 8GB of its own onboard memory, released in January 2026 and actively supported. That accelerator is dedicated entirely to continuous vision, which is what lets a station check every frame it can without ever pausing to do anything else. Everything else, tracking, the acoustic model, sensor reads, and the deterministic quality control, runs on the Pi's own processor, so the two never compete.

- Raspberry Pi 5, 8GB (choose the 16GB board only if you plan to run a larger language model on the same Pi, which the field tier does not require)
- Raspberry Pi AI HAT+ 2 (Hailo-10H, 40 TOPS, 8GB)
- The official Active Cooler and the 27W USB-C power supply, both required for sustained inference
- A small RTC backup battery (CR2032), which holds the clock between satellite fixes so timestamps stay trustworthy across reboots

### Camera

The [Raspberry Pi Camera Module 3 Wide](https://www.raspberrypi.com/products/camera-module-3/) (a 12 megapixel autofocus sensor with a wide field of view) is the reference camera for continuous detection. For an underwater deployment it sits behind the housing's optical port, which is the surface the biofouling section below is about.

### Microphone or hydrophone

Audio is a first class trigger in Audtheia, not an afterthought: a sound can open an observation on its own, and it keeps events flowing when the camera view is degraded.

For a terrestrial or coastal-avian site, a standard microphone or an AudioMoth class recorder feeding the Pi is enough, paired with the BirdNET acoustic model.

For an underwater site the recorder must be a hydrophone, not a waterproofed microphone. A condenser microphone in a housing cannot hear underwater; a hydrophone is built for it. The reference marine build uses an [Aquarian Audio H2d](https://www.aquarianaudio.com/) hydrophone element with its matching preamplifier into a USB audio interface the Pi reads in real time. A note on the alternatives: research grade SoundTrap class recorders store internally and are recovered later, so they do not stream live and do not fit a detection triggered system; a digital "smart hydrophone" over Ethernet is the higher end live option; and a HydroMoth is the budget option with limited high frequency performance.

### Location and environment

- A [u-blox M10 USB GPS](https://www.u-blox.com/) dongle gives the station its position and, just as importantly, its authoritative clock. All timestamps are stored in coordinated universal time, disciplined by the satellite fix.
- For a marine site, the reference environmental suite is a set of [Atlas Scientific EZO](https://atlas-scientific.com/) probes (pH, dissolved oxygen, conductivity for salinity, and temperature) on a carrier board. Every reading is stored with a quality flag on the standard oceanographic scale, so a suspect value is never mistaken for a clean one.
- For a terrestrial site, swap the water probes for air temperature and humidity, soil moisture, and light sensors on the same interface. This is a configuration change with no code change: you enable the channels you have in the settings file.

### Storage, power, and housing

- A 1TB USB solid state drive holds the station's rolling buffer and local database. The station never deletes data that has not yet been copied to your desktop; it fills, prompts a sync, and only ever clears records the desktop already holds.
- A solar panel, a LiFePO4 battery, and a 5V regulator power a months long deployment. Sizing them is the subject of the power section below.
- An IP67 rated waterproof enclosure with cable glands protects the electronics. The optical port for the camera is the one surface that must stay both watertight and clear.

## Reference marine station: the parts at a glance

Prices move, especially for Pi adjacent parts, so the figures below are indicative and meant for planning. Always open the live retailer page for the current price and stock. The reference marine station lands around 1,300 to 1,500 US dollars with the full sensor suite; a terrestrial station is closer to 450 to 550, mostly because the hydrophone chain and the water probes are the expensive parts.

| Part | Purpose | Indicative cost (USD) |
|---|---|---|
| Raspberry Pi 5, 8GB | Orchestration, tracking, acoustic model, quality control | ~95 |
| Raspberry Pi AI HAT+ 2 (Hailo-10H) | Continuous vision on the accelerator | ~130 |
| Active Cooler + 27W USB-C supply | Sustained inference without throttling | ~17 |
| RTC battery (CR2032) | Holds the clock between satellite fixes | ~5 |
| Camera Module 3 Wide | Continuous detection | ~35 |
| Aquarian H2d element + preamp + USB audio interface | Live underwater acoustic capture | ~200 + ~190 + ~40 |
| u-blox M10 USB GPS | Location and authoritative time | ~25 |
| Atlas Scientific EZO pH, DO, conductivity, temperature + carrier | Marine environmental data | ~510 |
| 1TB USB SSD | Rolling buffer and local database | ~90 |
| Solar panel + LiFePO4 battery + regulator | Months long deployment | ~90 and up, depending on site |
| IP67 enclosure + cable glands | Protection from water and weather | ~40 |

A terrestrial station replaces the hydrophone chain with a standard microphone or an AudioMoth class recorder, and the water probes with air, soil, and light sensors on the same interface, with no change to the software.

## Powering an unattended station

The most common question for a remote deployment is whether the sun and a battery can keep the station alive. The honest answer is that it depends on three things you can estimate, and this section gives you the method rather than a single number, because the right answer is different for every site.

### Why the answer is favorable to begin with

Audtheia is deliberately energy proportional. The station does almost no heavy work until something happens in front of the sensor: a frame with no detection is discarded immediately, and the accelerator and processor only spend real energy when there is a real event to record. That keeps a station in the single digit watts range in ordinary conditions, which is what makes solar operation realistic at all. The exact figure depends on your camera frame rate and resolution and on how often events occur, so the first step is always to measure.

### Step 1: measure the real draw

Before sizing anything, put an inline power meter on the station and run it for a representative day, with your real camera settings and at a site with a realistic event rate. Record the average power in watts. A bench estimate is fine to start, but a measured average is what makes the rest of the calculation trustworthy, and it is the headline number worth reporting in a paper about the system.

### Step 2: daily energy

Multiply the average power by 24 hours to get the energy the station uses in a day, in watt hours:

```
daily energy (Wh) = average power (W) x 24
```

For example, a station that averages 4 watts uses about 96 watt hours per day.

### Step 3: size the solar panel

Solar output is measured in peak sun hours, the number of hours per day the site effectively receives full strength sunlight. Size the panel for your site's worst month, not its annual average, because the system has to survive the cloudy season, not the sunny one:

```
panel rating (W) = daily energy (Wh) / (worst-month peak sun hours x 0.7)
```

The 0.7 is a practical derating for charge controller losses, panel soiling, heat, and imperfect sun angle. For a Caribbean coastal site, the worst month is typically in the rainy season and lands around four and a half to five peak sun hours; look up your exact location in a free solar database such as the Global Solar Atlas or NREL's PVWatts rather than guessing. Continuing the 96 watt hour example at 4.5 peak sun hours, the panel works out to about 96 / (4.5 x 0.7), roughly 30 watts, so a 50 to 100 watt panel gives comfortable margin for dirty days and a fouled panel surface.

### Step 4: size the battery for autonomy

Autonomy is how many days the station can run with no useful sun, through a storm or a stretch of heavy overcast. A LiFePO4 battery can safely use about 90 percent of its rated capacity, so:

```
battery capacity (Ah) = (daily energy (Wh) x autonomy days) / (battery voltage x 0.9)
```

Two to three days of autonomy suits a sunny climate; for an unattended marine station in a region with tropical storms, three to five days is the prudent choice. For the 96 watt hour example at 3 days of autonomy on a 12 volt battery, that is (96 x 3) / (12 x 0.9), about 27 amp hours, so a 30 amp hour LiFePO4 battery is a sound pick with a little headroom.

### A note for your own deployment

Measure first, size for the worst month, and give both the panel and the battery margin, because a field station that browns out and misses a season of data is far more expensive than the extra panel watts. Record your measured average draw and your energy per confirmed observation; those two numbers are the clearest evidence that an event driven design saves power, and they belong in any writeup of the system.

## Keeping the camera clear: biofouling

In warm, productive water the single hardest part of a long marine deployment is not the electronics, it is keeping the camera's optical port clean. A lens in Caribbean coastal water can be visibly fouled by a film, and then by bryozoans, barnacles, and tubeworms, within days to a few weeks, and a fouled lens sees nothing. This is a real, well studied problem for every marine optical instrument, and there is no single perfect fix, so the plan is to combine a few approaches and to lean on the one safeguard the system gives you for free.

### The approaches, and their honest trade-offs

- **Copper around the port.** Copper is the workhorse passive antifoulant: a copper shroud, faceplate, or ring around the optical port strongly slows fouling because copper ions are toxic to the organisms that settle. Its main limitation is that its effectiveness fades over time as the surface oxidizes and leaches, so it is a "buys you weeks to months" measure, not a permanent one.
- **Ultraviolet-C light.** Periodically illuminating the port with UV-C damages the genetic material of settling organisms and is one of the more effective methods for sensitive optical surfaces, where a mechanical wiper would smear or scratch. Field trials have kept UV treated optical housings clean for months while untreated surfaces fouled heavily. It costs a little power, which folds into the budget above.
- **Transparent antifouling coatings.** Because a camera port must stay clear, any coating on the glass itself has to be transparent and nontoxic; several such coatings exist and reduce how readily organisms adhere, extending the interval between cleanings.
- **Mechanical wipers.** A wiper that sweeps the port is a common tool, but it is the least suited to a camera lens: wipers foul and jam themselves, and they struggle on curved or recessed optical surfaces. Treat a wiper as a supplement, not the primary defense for the camera.
- **A recessed or flush port and a maintenance schedule.** A recessed port fouls a little more slowly and is easier to clean, and for a site you can revisit, a simple cleaning cadence, for example every one to two weeks in productive water, is often the most reliable measure of all.

### The safeguard already built in

Audtheia's dual trigger design is itself a biofouling mitigation. Because an acoustic detection opens a full observation on its own, a station whose camera has begun to foul keeps recording events through its hydrophone, so a degrading lens means reduced vision, not a dead station. This is one of the reasons the marine reference build treats the hydrophone as essential rather than optional.

### A practical Caribbean plan

For a warm, productive coastal site, a sound starting combination is a copper faceplate or shroud around a recessed optical port, a transparent antifouling coating on the glass, a scheduled UV-C treatment if power allows, and a cleaning visit every week or two during the fouling season, with the acoustic trigger carrying events between cleanings. Log when you clean, so a drop in visual detections can be read against the fouling and cleaning record rather than mistaken for a real change in the animals.

## Assembly and deployment notes

- Mount the Pi, the accelerator, and the storage inside the IP67 enclosure, and bring the camera, hydrophone, GPS, and sensor cables out through sealed cable glands. Keep the enclosure's cooling in mind: the Active Cooler needs air, so a sealed box benefits from a thermal path to the housing wall.
- Mount the station so the camera looks at the substrate or habitat you care about, and secure it against surge and current; a station that shifts is a station whose GPS track and field of view no longer match its records.
- Give the solar panel a clean sky angle and a surface you can wipe, and expect its output to drop as it too collects grime and salt.

## Where to go next

- To turn a trained detector into the file a station runs, and to train your own models for your species and site, see the [custom models guide](custom-models.md).
- To set up the desktop and push a configuration to a station, see the [README](../README.md).
- To understand what the desktop does with a station's data over time, see the [dream pass guide](dream-pass.md).

## References

- Raspberry Pi AI HAT+ 2 product page: https://www.raspberrypi.com/products/ai-hat-plus-2/
- On antifouling for marine optical sensors, including copper, UV-C, coatings, and wipers: Antibiofouling Coatings for Marine Sensors, ACS Sensors, 2025, https://pubs.acs.org/doi/10.1021/acssensors.4c02670
- On UV-C for biofouling control on optical instruments: Marine Technology News, https://www.marinetechnologynews.com/news/biofouling-foiled-light-harnessed-500695
- Off-grid solar and LiFePO4 sizing method (peak sun hours, autonomy days): https://www.howtogosolar.org/off-grid-solar-sizing/
- Free site-specific solar data: Global Solar Atlas (https://globalsolaratlas.info/) and NREL PVWatts (https://pvwatts.nrel.gov/)
