-- ============================================================================
-- Audtheia V2 — Database Schema
-- Path: audtheia/storage/schema.sql
--
-- Single source of truth for every table in the system. Every later file
-- (database.py, monitor.py, acoustic.py, environment.py, observation.py,
-- verify.py, dream.py, generate.py, server.py) conforms to this contract.
--
-- Implements Technical Decisions Log #9–#52. See audtheia-v2-decisions-log.md
-- for the reasoning behind every constraint below. Do not alter a constraint
-- here without a corresponding dated entry in that log.
--
-- Engine requirements:
--   - SQLite 3.37.0+ (STRICT tables). Verified against SQLite 3.45.1.
--   - The application MUST issue `PRAGMA foreign_keys = ON;` on every
--     connection it opens — this pragma is per-connection in SQLite and is
--     NOT persisted by being declared in this file.
--
-- Ownership separation (decision #47):
--   Pi-owned (written at field capture):     stations, observations,
--       child_detections, environmental_readings, soundscape_index_readings,
--       station_telemetry, station_telemetry_errors
--   Desktop-owned (written only on desktop): observation_verification,
--       interpretations, dream_passes, patterns, pattern_observations
--   Synced desktop -> Pi (not part of the Pi -> desktop pull): skills
--   Reference cache (written by setup/fetch scripts): species_reference
--
-- The append-only Pi -> desktop sync (Session 2) only ever INSERTs into the
-- Pi-owned tables on the desktop side and never touches a desktop-owned
-- table or column — this is what makes ownership separation structurally
-- enforceable rather than a convention.
-- ============================================================================

PRAGMA foreign_keys = ON;

-- ============================================================================
-- STATIONS
-- Not part of settings.json (which holds local runtime config for the
-- station the Pi/desktop is currently running as). This table is the
-- database-level station registry so every other table can carry a real
-- foreign key to station_id, including from the desktop's unified
-- multi-station view (decisions #10, #11).
-- ============================================================================
CREATE TABLE stations (
    id                  TEXT PRIMARY KEY,          -- UUID
    station_name        TEXT NOT NULL UNIQUE,      -- human-assigned, used in event_name (#39)
    environment_type    TEXT NOT NULL CHECK (environment_type IN
                            ('marine', 'terrestrial', 'estuarine', 'freshwater', 'mixed')),
    created_at          TEXT NOT NULL,              -- UTC ISO8601
    notes                TEXT
) STRICT;

CREATE INDEX idx_stations_environment_type ON stations(environment_type);

-- ============================================================================
-- OBSERVATIONS  (Pi-owned)
-- The multimodal "event memory" (decision #37). One row = one ByteTrack
-- event or one acoustic-onset event, never one row per frame (decision #36,
-- rejected R14). Carries every Pi-owned field required by decisions
-- #9, #10, #14, #25 (top-level), #26, #36, #37, #39, #40, #42, #46, #49.
-- ============================================================================
CREATE TABLE observations (
    id                              TEXT PRIMARY KEY,    -- UUID, generated on the Pi at capture (#9)
    event_name                      TEXT NOT NULL UNIQUE, -- [StationName]_[ISO8601-date]_[shortUUID] (#39)
    station_id                      TEXT NOT NULL REFERENCES stations(id),  -- #10

    trigger_source                  TEXT NOT NULL CHECK (trigger_source IN ('vision', 'audio', 'sensor')),  -- #37

    -- ByteTrack / acoustic event fields (#36, #37)
    representative_frame             TEXT,        -- file path under data/detections/visual/; nullable (pure audio event with no usable frame)
    frame_count                      INTEGER,
    first_seen                       TEXT NOT NULL,  -- UTC ISO8601 (#40)
    last_seen                        TEXT NOT NULL,  -- UTC ISO8601 (#40)
    duration                         REAL NOT NULL,  -- seconds; the TRUE event duration even if a capped clip is shorter (#38, white-matter lossless rule)

    -- Time base (#40): a capture before the first GPS fix / disciplined-clock
    -- state is flagged here rather than silently treated as authoritative.
    time_provisional                 INTEGER NOT NULL DEFAULT 0 CHECK (time_provisional IN (0, 1)),

    -- Provenance + lifecycle (#14, #49)
    -- data_source here describes the row-level detection event itself
    -- (almost always 'model' — the screening model's positive call;
    -- 'sensor' is reserved for a future sensor-threshold trigger, #37).
    data_source                      TEXT NOT NULL CHECK (data_source IN
                                        ('sensor', 'database', 'model', 'llm_inferred', 'dream')),
    qc_state                         TEXT NOT NULL DEFAULT 'qc_pending' CHECK (qc_state IN
                                        ('qc_pending', 'qc_passed', 'qc_deferred', 'verified')),
    -- Controlled code set for WHY a record was deferred (#49). This exact
    -- vocabulary is an implementation choice made in this session (not yet
    -- ratified in the decisions log) — observation.py (Session 7) is the
    -- authority that actually assigns these and may need to refine the set;
    -- flagged in this session's handoff note.
    qc_reason                        TEXT CHECK (qc_reason IS NULL OR qc_reason IN
                                        ('schema_novel_shape', 'incomplete_record', 'sensor_conflict',
                                         'low_confidence_unclassified', 'manual_review_requested')),

    -- Forwarded plasticity metadata + version provenance (#26, #42)
    screening_confidence              REAL CHECK (screening_confidence IS NULL OR
                                        (screening_confidence >= 0 AND screening_confidence <= 1)),
    screening_model_version           TEXT,         -- YOLO11 .hef version that produced this call
    acoustic_model_version            TEXT,         -- nullable; populated only when audio was involved
    gbif_snapshot_date                TEXT,         -- GBIF backbone snapshot date behind any taxonomic match
    iucn_fetch_date                   TEXT,         -- IUCN fetch date behind any conservation-status match

    -- Salience: provisional slot only (#27, #31). Ingredients retained per
    -- #31: confidence is screening_confidence above; the provisional
    -- "immediate novelty" ingredient is anomaly_magnitude_provisional below.
    -- Authoritative slot + rarity + full baseline deviation live on the
    -- desktop side in observation_verification (#47 ownership separation).
    salience_provisional              REAL CHECK (salience_provisional IS NULL OR
                                        (salience_provisional >= 0 AND salience_provisional <= 1)),
    anomaly_magnitude_provisional      REAL,

    -- Audio window (#38)
    audio_clip_path                    TEXT,         -- file path under data/detections/audio/
    audio_true_duration_seconds         REAL,         -- always recorded even when the stored clip is capped
    audio_capped                       INTEGER CHECK (audio_capped IS NULL OR audio_capped IN (0, 1)),

    -- GPS (#6, #40). A GPS read is one structured read => one status, per
    -- the structured-read contract (#51) — not a per-coordinate status.
    gps_latitude                       REAL,
    gps_longitude                      REAL,
    gps_elevation                      REAL,
    gps_status                         TEXT CHECK (gps_status IS NULL OR gps_status IN
                                        ('measured', 'not_measured', 'below_detection_limit',
                                         'sensor_error', 'not_applicable')),

    -- Optional feature embedding forwarding (#26, O6) — settings-gated,
    -- off by default. NULL whenever the toggle is off.
    feature_embedding                  BLOB,

    created_at                         TEXT NOT NULL,   -- UTC ISO8601, Pi-side write time
    synced_at                          TEXT             -- set by the desktop on successful pull (#12, #44); NULL until synced
) STRICT;

-- Composite (station_id, first_seen): serves the dominant desktop read —
-- "this station's events over a time range" (analytics, reports, verify
-- backlog) — and, by leftmost-prefix, also every station_id-only lookup, so
-- it fully replaces a standalone station_id index with no loss.
CREATE INDEX idx_observations_station_id_first_seen ON observations(station_id, first_seen);
-- Cross-station time scans (whole-deployment timelines) still need first_seen alone.
CREATE INDEX idx_observations_first_seen ON observations(first_seen);
-- PARTIAL index on the append-only pull's single hottest query
-- ("what has not synced yet?" = WHERE synced_at IS NULL). It indexes only
-- un-synced rows, so it stays tiny and rows leave it the moment the desktop
-- stamps synced_at (#12, #44). This is the never-evict-unsynced buffer's
-- read path, so it must be fast and bounded.
CREATE INDEX idx_observations_synced_pending ON observations(synced_at) WHERE synced_at IS NULL;
CREATE INDEX idx_observations_qc_state ON observations(qc_state);
CREATE INDEX idx_observations_data_source ON observations(data_source);

-- ============================================================================
-- CHILD_DETECTIONS  (Pi-owned)
-- Multi-taxon sub-detections within one event (decision #39: "each detected
-- taxon is a clearly sub-identified detection within the event"). Generalized
-- across both modalities (vision multi-species AND audio multi-species, e.g.
-- BirdNET returning more than one species per clip) via `modality` rather
-- than building two near-identical tables.
-- ============================================================================
CREATE TABLE child_detections (
    id                      TEXT PRIMARY KEY,
    observation_id            TEXT NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
    modality                  TEXT NOT NULL CHECK (modality IN ('vision', 'audio')),

    gbif_usage_key             TEXT,         -- nullable: GBIF backbone taxon key if matched
    scientific_name            TEXT,
    common_name                TEXT,

    confidence                 REAL CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),

    -- Bounding box: meaningful for modality='vision' only; NULL for audio
    bbox_x                     REAL,
    bbox_y                     REAL,
    bbox_w                     REAL,
    bbox_h                     REAL,

    data_source                 TEXT NOT NULL CHECK (data_source = 'model'),
    status                      TEXT NOT NULL DEFAULT 'measured' CHECK (status IN
                                    ('measured', 'not_measured', 'below_detection_limit',
                                     'sensor_error', 'not_applicable')),
    created_at                   TEXT NOT NULL
) STRICT;

CREATE INDEX idx_child_detections_observation_id ON child_detections(observation_id);
CREATE INDEX idx_child_detections_modality ON child_detections(modality);

-- ============================================================================
-- ENVIRONMENTAL_READINGS  (Pi-owned)
-- Generic channel-as-row design (confirmed with Andy this session) so the
-- water/air/soil sensor SET is entirely a settings.json concern with zero
-- schema or code change between deployment types (Master Concept §8,
-- "Terrestrial use ... same I2C interface, no code change").
-- ============================================================================
CREATE TABLE environmental_readings (
    id                  TEXT PRIMARY KEY,
    observation_id         TEXT NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
    channel               TEXT NOT NULL,    -- e.g. 'water_temp_c' / 'ph' / 'soil_moisture_pct'; channel set lives in settings.json, never hardcoded in code
    value                  REAL,
    unit                   TEXT,
    data_source            TEXT NOT NULL CHECK (data_source = 'sensor'),
    status                 TEXT NOT NULL CHECK (status IN
                                ('measured', 'not_measured', 'below_detection_limit',
                                 'sensor_error', 'not_applicable')),
    qartod_flag             INTEGER CHECK (qartod_flag IS NULL OR qartod_flag IN (1, 2, 3, 4, 9)),  -- marine channels only (#25)
    created_at               TEXT NOT NULL
) STRICT;

CREATE INDEX idx_environmental_readings_observation_id ON environmental_readings(observation_id);
CREATE INDEX idx_environmental_readings_channel ON environmental_readings(channel);

-- ============================================================================
-- SOUNDSCAPE_INDEX_READINGS  (Pi-owned)
-- Optional, default-off continuous marine add-on (decision #38). Independent
-- of any single observation — it's a continuous time series, not event-gated.
-- ============================================================================
CREATE TABLE soundscape_index_readings (
    id              TEXT PRIMARY KEY,
    station_id        TEXT NOT NULL REFERENCES stations(id) ON DELETE CASCADE,
    recorded_at        TEXT NOT NULL,   -- UTC ISO8601
    metric             TEXT NOT NULL,   -- e.g. band-limited SPL label or named acoustic index
    value               REAL NOT NULL,
    data_source         TEXT NOT NULL CHECK (data_source = 'sensor'),
    created_at           TEXT NOT NULL,
    synced_at            TEXT
) STRICT;

CREATE INDEX idx_soundscape_station_id ON soundscape_index_readings(station_id);
CREATE INDEX idx_soundscape_recorded_at ON soundscape_index_readings(recorded_at);
-- Partial: same append-only-pull pattern as observations (see above).
CREATE INDEX idx_soundscape_synced_pending ON soundscape_index_readings(synced_at) WHERE synced_at IS NULL;

-- ============================================================================
-- SPECIES_REFERENCE  (reference cache; written by setup/fetch-species-data.sh, Session 17)
-- Per-species GBIF occurrence + IUCN cache, fetched once under the user's own
-- credentials (decision #15) with a documented re-fetch path and snapshot
-- stamping (decision #43). Not bulk-shipped data (rejected R10) — populated
-- locally per target species.
-- ============================================================================
CREATE TABLE species_reference (
    gbif_usage_key          TEXT PRIMARY KEY,
    scientific_name           TEXT NOT NULL,
    common_name                TEXT,
    taxonomic_rank             TEXT,
    iucn_status                TEXT,
    iucn_fetch_date             TEXT,
    gbif_occurrence_count       INTEGER,
    gbif_snapshot_date          TEXT,
    data_source                 TEXT NOT NULL CHECK (data_source = 'database'),
    fetched_at                   TEXT NOT NULL
) STRICT;

-- ============================================================================
-- SKILLS  (desktop-owned; synced desktop -> Pi on connect, per the #12 merge
-- rule — NOT part of the Pi -> desktop observation pull)
-- The `tier` attribute is the structural enforcement point for decision #45:
-- the engine decides where a skill runs by reading this column, not by
-- inspecting the skill's free-text content.
-- ============================================================================
CREATE TABLE skills (
    id                  TEXT PRIMARY KEY,
    title                 TEXT NOT NULL,
    trigger_condition       TEXT NOT NULL,    -- "when to use"
    instruction              TEXT NOT NULL,    -- "how to apply"
    tier                     TEXT NOT NULL CHECK (tier IN ('deterministic_flag', 'interpretive')),  -- #45
    created_at                TEXT NOT NULL,
    updated_at                TEXT NOT NULL
) STRICT;

-- ============================================================================
-- OBSERVATION_VERIFICATION  (desktop-owned)
-- One-to-one extension of an observation, written only by verify.py
-- (decisions #30, #47). The `verified` flag here is exactly the occipital
-- gate that dream.py's REM phase reads (decision #34). Authoritative
-- salience and its remaining ingredients (rarity, baseline deviation,
-- refined anomaly magnitude) live here per #27/#31, never on the Pi-owned
-- observations table — this is what makes the append-only pull structurally
-- incapable of clobbering a desktop-written value.
-- ============================================================================
CREATE TABLE observation_verification (
    observation_id                       TEXT PRIMARY KEY REFERENCES observations(id) ON DELETE CASCADE,
    verified                              INTEGER NOT NULL DEFAULT 0 CHECK (verified IN (0, 1)),  -- the occipital gate (#34)
    rfdetr_version                        TEXT,
    salience_authoritative                 REAL CHECK (salience_authoritative IS NULL OR
                                            (salience_authoritative >= 0 AND salience_authoritative <= 1)),
    rarity_score                           REAL,
    baseline_deviation                      REAL,
    anomaly_magnitude_authoritative          REAL,
    verified_at                             TEXT,
    created_at                              TEXT NOT NULL
) STRICT;

-- PARTIAL: the occipital gate (#34). dream.py's REM phase only ever asks
-- "which observations are verified?" (verified = 1), so indexing only those
-- rows by observation_id makes the gate scan minimal and self-documenting.
CREATE INDEX idx_observation_verification_gate ON observation_verification(observation_id) WHERE verified = 1;

-- ============================================================================
-- INTERPRETATIONS  (desktop-owned)
-- The ~15-20 interpretive points (Master Concept §6) owned by verify.py
-- (decision #30) plus interpretive-skill outputs (decision #45). Always
-- data_source = 'llm_inferred'; never written by the field tier (#24, #51).
-- A many-rows-per-observation design because an observation can carry
-- several distinct interpretive points simultaneously.
-- ============================================================================
CREATE TABLE interpretations (
    id                  TEXT PRIMARY KEY,
    observation_id         TEXT NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
    point_type             TEXT NOT NULL CHECK (point_type IN
                                ('ecological_role', 'rarity_score', 'anomaly_flag', 'cross_modal_attribution',
                                 'behavioral_context', 'seasonal_assessment', 'habitat_quality_flag',
                                 'interaction_pattern', 'skill_note')),
    value                   TEXT NOT NULL,
    confidence               REAL CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    data_source              TEXT NOT NULL CHECK (data_source = 'llm_inferred'),
    produced_by               TEXT NOT NULL CHECK (produced_by IN ('verify', 'dream', 'skill')),
    model_version              TEXT,
    skill_id                   TEXT REFERENCES skills(id),  -- nullable; set only when produced_by = 'skill'
    created_at                  TEXT NOT NULL
) STRICT;

CREATE INDEX idx_interpretations_observation_id ON interpretations(observation_id);
CREATE INDEX idx_interpretations_point_type ON interpretations(point_type);

-- ============================================================================
-- STATION_TELEMETRY  (Pi-owned)
-- One row per station per heartbeat interval (decision #48). Three field
-- groups: effort/coverage, health, energy (nullable — populated only when
-- an energy meter is configured). Effort denominators and energy-per-window
-- are DERIVED at analysis time by aggregating/differencing these rows —
-- never stored here as derived values.
-- ============================================================================
CREATE TABLE station_telemetry (
    id                          TEXT PRIMARY KEY,
    station_id                    TEXT NOT NULL REFERENCES stations(id) ON DELETE CASCADE,
    recorded_at                   TEXT NOT NULL,   -- UTC ISO8601

    -- Effort / coverage
    camera_uptime_seconds            REAL,
    frames_processed                  INTEGER,
    frames_dropped                    INTEGER,
    valid_audio_seconds                REAL,
    npu_active_seconds                  REAL,
    effective_detection_fps              REAL,

    -- Health
    station_temperature_c                REAL,
    buffer_fill_pct                       REAL,
    sync_lag_seconds                       REAL,

    -- Energy (nullable; only when an energy meter is configured, #48)
    avg_power_w                            REAL,
    cumulative_joules                       REAL,

    data_source                             TEXT NOT NULL CHECK (data_source = 'sensor'),
    created_at                               TEXT NOT NULL,
    synced_at                                TEXT
) STRICT;

CREATE INDEX idx_station_telemetry_station_id ON station_telemetry(station_id);
CREATE INDEX idx_station_telemetry_recorded_at ON station_telemetry(recorded_at);
-- Partial: same append-only-pull pattern as observations (see above).
CREATE INDEX idx_station_telemetry_synced_pending ON station_telemetry(synced_at) WHERE synced_at IS NULL;

-- ============================================================================
-- STATION_TELEMETRY_ERRORS  (Pi-owned)
-- Per-channel health error counts (decision #48: "per-channel error counts"),
-- channel-keyed rather than fixed columns for the same no-hardcoding reason
-- as environmental_readings.
-- ============================================================================
CREATE TABLE station_telemetry_errors (
    id                  TEXT PRIMARY KEY,
    telemetry_id           TEXT NOT NULL REFERENCES station_telemetry(id) ON DELETE CASCADE,
    channel                TEXT NOT NULL,
    error_count             INTEGER NOT NULL DEFAULT 0
) STRICT;

CREATE INDEX idx_telemetry_errors_telemetry_id ON station_telemetry_errors(telemetry_id);

-- ============================================================================
-- DREAM_PASSES  (desktop-owned)
-- One row per pass (decision #50). The checkpoint/status home decision #50
-- resolved O8 with. phase_reached + status + checkpoint_watermark together
-- drive the Brain-panel progress display and resumability (decision #35).
-- ============================================================================
CREATE TABLE dream_passes (
    id                          TEXT PRIMARY KEY,
    station_scope                  TEXT,        -- NULL = all stations, or a specific station_id; not FK-enforced since scope may span multiple stations
    phase_reached                   TEXT NOT NULL CHECK (phase_reached IN ('nrem_a', 'nrem_b', 'rem', 'complete')),
    status                          TEXT NOT NULL CHECK (status IN ('running', 'paused', 'complete', 'error')),
    started_at                       TEXT NOT NULL,
    ended_at                          TEXT,
    cycles_completed                  INTEGER NOT NULL DEFAULT 0,
    work_budget_consumed               INTEGER NOT NULL DEFAULT 0,
    checkpoint_watermark                TEXT,    -- last committed consolidation watermark (#35)
    data_source                         TEXT NOT NULL CHECK (data_source = 'dream'),
    created_at                           TEXT NOT NULL
) STRICT;

CREATE INDEX idx_dream_passes_status ON dream_passes(status);

-- ============================================================================
-- PATTERNS  (desktop-owned)
-- One row per candidate hypothesis (decision #50). Always tagged
-- data_source = 'dream'; never presented as an established finding
-- (decision #28). effect_size_type travels with effect_size because a bare
-- effect size is uninterpretable.
-- ============================================================================
CREATE TABLE patterns (
    id                      TEXT PRIMARY KEY,
    dream_pass_id              TEXT NOT NULL REFERENCES dream_passes(id) ON DELETE CASCADE,
    data_source                 TEXT NOT NULL CHECK (data_source = 'dream'),
    status                       TEXT NOT NULL DEFAULT 'candidate' CHECK (status IN ('candidate', 'verified', 'rejected')),
    dream_phase                  TEXT NOT NULL CHECK (dream_phase IN ('nrem', 'rem')),
    confidence                    REAL CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    effect_size                    REAL,
    effect_size_type                TEXT CHECK (effect_size_type IS NULL OR effect_size_type IN ('r', 'cohens_d', 'log_odds')),
    data_span_start                  TEXT NOT NULL,   -- UTC ISO8601
    data_span_end                     TEXT NOT NULL,   -- UTC ISO8601
    n                                INTEGER NOT NULL,
    p_value                           REAL,            -- nullable: Session 9 statistical-validity layer (O22)
    q_value                            REAL,            -- nullable: Benjamini-Hochberg FDR-adjusted (O22)
    autocorr_adjusted                   INTEGER CHECK (autocorr_adjusted IS NULL OR autocorr_adjusted IN (0, 1)),
    model_version                        TEXT,
    description                           TEXT NOT NULL,
    created_at                             TEXT NOT NULL
) STRICT;

CREATE INDEX idx_patterns_dream_pass_id ON patterns(dream_pass_id);
CREATE INDEX idx_patterns_status ON patterns(status);

-- ============================================================================
-- PATTERN_OBSERVATIONS  (desktop-owned)
-- Junction linking each candidate pattern to its supporting observation UUIDs
-- — the "memories" (decision #50). Makes every dream hypothesis reproducible
-- from its exact source observations; the measured/inferred firewall and the
-- occipital gate carried to their conclusion.
-- ============================================================================
CREATE TABLE pattern_observations (
    pattern_id          TEXT NOT NULL REFERENCES patterns(id) ON DELETE CASCADE,
    observation_id          TEXT NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
    PRIMARY KEY (pattern_id, observation_id)
) STRICT, WITHOUT ROWID;
-- WITHOUT ROWID: a pure junction (composite PK, no other columns, never
-- referenced by another table) stores rows directly in the PK b-tree, so the
-- redundant rowid index is eliminated.

-- Reverse lookup: the composite PK is keyed (pattern_id, observation_id), so
-- "which patterns cite THIS observation?" and the ON DELETE CASCADE fired when
-- an observation is removed both need observation_id indexed independently.
CREATE INDEX idx_pattern_observations_observation_id ON pattern_observations(observation_id);

-- ============================================================================
-- End of schema.sql
-- ============================================================================
