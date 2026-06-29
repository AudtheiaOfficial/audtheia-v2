# Audtheia configuration

This folder holds the single source of every value Audtheia can be configured
with. Nothing elsewhere in the system is hardcoded: model paths, sensor
channels, capture tuning, schedules, and credentials all live here, and every
component reads them through one loader so the same configuration behaves
identically on the desktop and on a field station.

There are two files you work with:

- `settings.json`: the full configuration. It is safe to commit and contains
  no secrets.
- `secrets.json`: your private credentials. It is never committed. Copy
  `secrets.example.json` to `secrets.json` and fill in your own values, or
  supply them through environment variables instead.

A third file, `audtheia/config.py`, is the loader. It reads `settings.json`,
merges your secrets, resolves every path for your operating system, and checks
the whole file for mistakes before anything runs. If a value is missing or out
of range, it stops with a message that names the exact key so you can fix it
quickly.

## How a station is set up

You do not edit station entries by hand in normal use. You add and name a
station from the desktop application, in Settings, where the application
assigns its unique identifier, records it in the database, and writes the
matching entry here. When you connect a Raspberry Pi field station, the desktop
sends that Pi a copy of its own station entry, so the field station always
knows exactly which site it is and how it should behave. The examples in
`settings.json` are there to show the shape; replace them with your own sites.

## Operating systems

Every path in `settings.json` is written with forward slashes and is relative
to the repository folder. The loader turns each one into the correct absolute
path for your computer, so one configuration file works without changes on
Windows, macOS, Linux, and Raspberry Pi OS. If you keep your long-term data on
an external drive, you can write a full absolute path instead and it will be
used exactly as given.

## The settings, block by block

### node

Tells a copy of Audtheia what it is.

- `role`: `desktop` for the hub that manages everything, or `pi` for a field
  station.
- `active_station_id`: on a field station, the one station it runs as. The
  desktop fills this in for you when it configures a Pi. It is left empty on
  the desktop.

### paths

Where things live, relative to the repository folder.

- `db_path`: the SQLite database file. On the desktop this is your full,
  long-term record. On a Pi it is the rolling field buffer.
- `schema_path`: the database definition used to create a fresh database.
- `data_dir`, `detections_visual_dir`, `detections_audio_dir`, `gps_dir`:
  where captured frames, audio clips, and location tracks are stored.
- `reports_dir`: where generated reports are written.
- `models_dir`: the root of the model folders.
- `gbif_backbone_path`: the bundled taxonomic reference shipped with Audtheia.

### database

- `wal`: keeps the live feed and the web interface readable while data is being
  written. Leave this on.
- `busy_timeout_ms`: how long, in milliseconds, to wait for a brief lock to
  clear before reporting an error.

### sync

- `batch_size`: how many records a field station sends to the desktop in one
  group when they connect.

### embeddings

A feature embedding is an optional numeric fingerprint of a detection frame
that lets the desktop find look-alike events over time. It is turned off by
default because it adds to what a field station has to store and send.

- `forward_embeddings`: keep an embedding with each event and include it when
  syncing. Off by default.
- `max_embedding_bytes`: a safety limit. A correct embedding is always the same
  size, so one over this limit signals a model or setup mismatch and is refused
  rather than stored, which keeps malformed data out of the record without ever
  discarding a real observation. Set it to your model's exact vector size, which
  is its number of dimensions multiplied by the bytes per value (for example a
  2048-dimension vector of 4-byte values is 8192 bytes).

### buffer

How a field station manages limited storage. It never deletes data the desktop
has not yet received.

- `high_water_pct`: the fill level that prompts a sync and shows a storage
  notice.
- `hard_ceiling_pct`: the fill level at which the station raises a critical
  alert.
- `auto_sync_when_reachable`: sync automatically whenever the desktop is
  available.
- `pause_capture_at_ceiling`: at the ceiling, pause new capture rather than
  drop anything still waiting to sync.

### telemetry

Station health and effort reporting.

- `heartbeat_seconds`: how often, in seconds, a station records a health and
  effort snapshot.
- `energy_meter`: power-measurement hardware. When `enabled` is off, the energy
  fields stay empty rather than being guessed. Turn it on and set the interface
  and address only if you have a meter fitted.

### analysis

- `per_observation_analysis_location`: where each event's quality-control and
  consolidation step runs. The default, `pi`, does it on the field station,
  which is fast and uses very little power. Set it to `desktop` only for a
  power-critical deployment where you want the field station to do nothing
  beyond capture.

### schedules

The only two activities that run on a timer rather than on a detection.

- `reports.schedule`: `daily`, `weekly`, `biweekly`, or `on_demand`.
- `reports.formats`: any of `pdf` and `csv`.
- `dream_pass.schedule`: `daily`, `weekly`, `biweekly`, or `manual`.
- `dream_pass.budget`: how much work the longitudinal analysis does in one run
  and how large its working memory may grow. `epoch_batch_size` is how many
  records it consolidates at a time; `max_cycles_per_pass` caps the work per run
  (zero means clear the full backlog); `substrate_exemplar_cap` and
  `substrate_candidate_pattern_cap` bound its working memory so a long-running
  station analyzes no slower than a new one. The defaults are sensible starting
  points and can be tuned later.

### server

- `host` and `port`: where the local web interface is served. The default
  serves only to the local machine on port 8000.

### localization

- `local_timezone`: the time zone used only for display. Stored data is always
  in coordinated universal time. The default, `auto`, follows your computer's
  own time zone with no extra setup. You may set a named zone such as
  `America/Puerto_Rico` instead; on a system without the time-zone database
  installed, a named zone needs the `tzdata` package, while `auto` always works.

### privacy

- `discard_human_detections`: when on, detections identified as people are not
  kept. Leave this on unless your study and permits specifically require
  otherwise.

### network

- `hotspot_ssid_pattern`: the name a field station broadcasts in the field. The
  station name is filled into `{station_name}`.
- `ssh`: how the desktop reaches a field station to configure it. The password
  or key for this lives in `secrets.json`, never here.

### desktop_models

Models that run on the desktop hub across every station.

- `visual_rfdetr`: the high-accuracy verification model.
- `llm`: the model used for the longitudinal analysis.

Each entry has a `path`, a `version` that is recorded with every result it
produces so your data stays reproducible, and a `citation` so the model's
authors are credited in your reports and publications.

### stations

One entry per site. Each entry carries:

- `station_id`: a unique identifier assigned by the application.
- `station_name`: your readable name for the site.
- `environment_type`: one of `marine`, `terrestrial`, `estuarine`,
  `freshwater`, or `mixed`.
- `habitat`: an optional, more specific description of the site. See the list
  below.
- `target_species`: the species this site focuses on, used when fetching
  reference data.
- `sensors`: which capture devices are active (`camera`, `audio`, `gps`).
- `channels`: the environmental sensors at this site (see below).
- `models`: the field-station models for this site (see below).
- `capture`: detection and recording tuning for this site (see below).

#### channels

Each channel is one environmental sensor reading. The set is entirely yours to
define, which is what lets a marine, terrestrial, freshwater, or estuarine site
use the same software with no code change.

- `id`: the channel name, recorded with every reading. Use a clear, stable name
  such as `water_temp_c` or `soil_moisture_pct`.
- `unit`: the unit the value is reported in.
- `marine`: whether this is a marine channel, which is what marks it for the
  ocean-data quality scale used in reports.
- `enabled`: whether the channel is read.
- `driver` (optional): how the sensor is read, such as its interface and
  address.
- `qc` (optional): quality-control bounds. `gross_range` is the range a value
  must fall within to look plausible; `sensor_range` is what the instrument can
  physically report; `detection_limit` is the smallest value the sensor can
  distinguish from none. These are used to flag readings automatically.

#### models

- `visual_pi`: the continuous detection model for this site's field station.
- `acoustic`: the sound model. `active` selects which option is used, and the
  `options` block holds the available models. Each has a `path`, a `version`,
  and a `citation`. The `birdnet` option is the default for sites with birds and
  other in-air callers. The `marine` option is for underwater sound and is left
  empty until you supply a model. The `custom` option is for a model you train
  yourself.

#### capture

- `fps` and `resolution`: how often and at what size frames are checked.
- `bytetrack`: how repeated frames of the same animal are joined into a single
  event. `track_activation_threshold` is the confidence needed to start
  tracking, `minimum_matching_threshold` is how closely frames must match to be
  the same animal, `track_close_frames` is how many missing frames end an event,
  and `frame_rate` matches your capture rate.
- `representative_frame_rule`: which frame represents the event. The highest
  confidence frame is used.
- `max_event_duration_seconds`: the longest a single event may run before it is
  split.
- `audio`: how much sound is kept around an event. `pre_roll_seconds` and
  `post_roll_seconds` are the lead-in and lead-out; `max_clip_seconds` caps the
  stored clip while the true event length is always recorded.
- `soundscape`: an optional continuous sound-index recording for marine sites,
  off by default.

## Habitat values

`habitat` is optional and adds a more specific description of a site without
changing its environment type. Choose one of:

Marine: `coral_reef`, `rocky_reef`, `kelp_forest`, `seagrass_meadow`,
`open_ocean`, `deep_sea`, `intertidal_zone`, `sandy_seabed`, `marine_mangrove`.

Estuarine: `estuary`, `salt_marsh`, `brackish_lagoon`, `tidal_flat`.

Freshwater: `lake`, `river`, `stream`, `pond`, `freshwater_wetland`.

Terrestrial: `forest_boreal`, `forest_temperate`, `forest_tropical`,
`grassland`, `savanna`, `shrubland`, `desert`, `tundra`,
`terrestrial_wetland`, `agricultural_land`, `urban`.

Mixed: `mixed_habitat`.

## Secrets

Keep credentials out of `settings.json` and out of version control.

1. Copy `secrets.example.json` to `secrets.json` in this folder.
2. Fill in the values you have: GBIF and IUCN credentials for fetching species
   reference data, and the hotspot and SSH passwords for field stations.
3. `secrets.json` is ignored by version control and is read automatically by
   the loader.

You can also supply any secret through an environment variable instead of the
file, which overrides the file value. The variable name is `AUDTHEIA_SECRET_`
followed by the key in capitals, for example `AUDTHEIA_SECRET_GBIF_PASSWORD`.

## Using the loader in code

```python
from audtheia.config import load_settings
from audtheia.storage.database import Database

settings = load_settings()                 # reads config/settings.json
db = Database(settings.db_path(), **settings.database_kwargs())
db.initialize_schema(settings.schema_path())
```

On a field station, `settings.active_station()` returns the single site that
station runs as, with its channels, models, and capture tuning ready to use.
