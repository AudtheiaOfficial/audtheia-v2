-- ============================================================================
-- Audtheia V2 Database Schema
-- Path: audtheia/storage/schema.sql
--
-- Single source of truth for every table in the system. Every component that
-- touches storage conforms to this contract. Do not alter a constraint here
-- without updating the code that depends on it.
--
-- Engine requirements:
--   - SQLite 3.37.0+ (STRICT tables). Verified against SQLite 3.45.1.
--   - The application MUST issue `PRAGMA foreign_keys = ON;` on every
--     connection it opens. This pragma is per-connection in SQLite and is
--     NOT persisted by being declared in this file.
--
-- Ownership separation:
--   Pi-owned (written at field capture):     stations, observations,
--       child_detections, environmental_readings, soundscape_index_readings,
--       station_telemetry, station_telemetry_errors
--   Desktop-owned (written only on desktop): observation_verification,
--       interpretations, dream_passes, patterns, pattern_observations,
--       site_baselines
--   Synced desktop -> Pi (not part of the Pi -> desktop pull): skills
--   Reference cache (written by setup/fetch scripts): species_reference
--
-- The append-only Pi -> desktop sync only ever INSERTs into the Pi-owned
-- tables on the desktop side and never touches a desktop-owned table or
-- column, which is what makes ownership separation structurally enforceable
-- rather than a convention.
-- ============================================================================

PRAGMA foreign_keys = ON;

-- ============================================================================
-- STATIONS
-- Not part of settings.json (which holds local runtime config for the
-- station the Pi/desktop is currently running as). This table is the
-- database-level station registry so every other table can carry a real
-- foreign key to station_id, including from the desktop's unified
-- multi-station view.
-- ============================================================================
CREATE TABLE stations (
    id                  TEXT PRIMARY KEY,          -- UUID
    station_name        TEXT NOT NULL UNIQUE,      -- human-assigned, used in event_name
    environment_type    TEXT NOT NULL CHECK (environment_type IN
                            ('marine', 'terrestrial', 'estuarine', 'freshwater', 'mixed')),
    created_at          TEXT NOT NULL,              -- UTC ISO8601
    notes                TEXT
) STRICT;

CREATE INDEX idx_stations_environment_type ON stations(environment_type);

-- ============================================================================
-- OBSERVATIONS  (Pi-owned)
-- The multimodal "event memory". One row = one ByteTrack event or one
-- acoustic-onset event, never one row per frame.
-- ============================================================================
CREATE TABLE observations (
    id                              TEXT PRIMARY KEY,    -- UUID, generated on the Pi at capture
    event_name                      TEXT NOT NULL UNIQUE, -- [StationName]_[ISO8601-date]_[shortUUID]
    station_id                      TEXT NOT NULL REFERENCES stations(id),

    trigger_source                  TEXT NOT NULL CHECK (trigger_source IN ('vision', 'audio', 'sensor')),

    -- ByteTrack / acoustic event fields
    representative_frame             TEXT,        -- file path under data/detections/visual/; nullable (pure audio event with no usable frame)
    frame_count                      INTEGER,
    first_seen                       TEXT NOT NULL,  -- UTC ISO8601
    last_seen                        TEXT NOT NULL,  -- UTC ISO8601
    duration                         REAL NOT NULL,  -- seconds; the TRUE event duration even if a capped clip is shorter

    -- Time base: a capture before the first GPS fix / disciplined-clock state
    -- is flagged here rather than silently treated as authoritative.
    time_provisional                 INTEGER NOT NULL DEFAULT 0 CHECK (time_provisional IN (0, 1)),

    -- Provenance and lifecycle.
    -- data_source here describes the row-level detection event itself
    -- (almost always 'model', the screening model's positive call;
    -- 'sensor' is reserved for a future sensor-threshold trigger).
    data_source                      TEXT NOT NULL CHECK (data_source IN
                                        ('sensor', 'database', 'model', 'llm_inferred', 'dream')),
    qc_state                         TEXT NOT NULL DEFAULT 'qc_pending' CHECK (qc_state IN
                                        ('qc_pending', 'qc_passed', 'qc_deferred', 'verified')),
    -- Controlled code set for WHY a record was deferred. The field quality
    -- engine assigns exactly one of these when it cannot pass a record; a
    -- passed or pending record leaves it NULL.
    qc_reason                        TEXT CHECK (qc_reason IS NULL OR qc_reason IN
                                        ('schema_novel_shape', 'incomplete_record', 'sensor_conflict',
                                         'low_confidence_unclassified', 'manual_review_requested',
                                         'firewall_violation')),

    -- Forwarded plasticity metadata and version provenance.
    screening_confidence              REAL CHECK (screening_confidence IS NULL OR
                                        (screening_confidence >= 0 AND screening_confidence <= 1)),
    screening_model_version           TEXT,         -- YOLO11 .hef version that produced this call
    acoustic_model_version            TEXT,         -- nullable; populated only when audio was involved
    gbif_snapshot_date                TEXT,         -- GBIF backbone snapshot date behind any taxonomic match
    iucn_fetch_date                   TEXT,         -- IUCN fetch date behind any conservation-status match

    -- Salience: provisional slot only. The ingredients are retained so the
    -- combination formula can be added later with no schema change: confidence
    -- is screening_confidence above, and the provisional immediate-novelty
    -- ingredient is anomaly_magnitude_provisional below. The authoritative
    -- slot, rarity, and full baseline deviation live on the desktop side in
    -- observation_verification, so a station-to-desktop pull can never touch
    -- them.
    salience_provisional              REAL CHECK (salience_provisional IS NULL OR
                                        (salience_provisional >= 0 AND salience_provisional <= 1)),
    anomaly_magnitude_provisional      REAL,

    -- Audio window.
    audio_clip_path                    TEXT,         -- file path under data/detections/audio/
    audio_true_duration_seconds         REAL,         -- always recorded even when the stored clip is capped
    audio_capped                       INTEGER CHECK (audio_capped IS NULL OR audio_capped IN (0, 1)),

    -- GPS. A GPS read is one structured read, so it carries one status for the
    -- whole fix rather than a per-coordinate status.
    gps_latitude                       REAL,
    gps_longitude                      REAL,
    gps_elevation                      REAL,
    gps_status                         TEXT CHECK (gps_status IS NULL OR gps_status IN
                                        ('measured', 'not_measured', 'below_detection_limit',
                                         'sensor_error', 'not_applicable')),

    -- Optional feature embedding forwarding, settings-gated and off by
    -- default. NULL whenever the toggle is off.
    feature_embedding                  BLOB,

    created_at                         TEXT NOT NULL,   -- UTC ISO8601, Pi-side write time
    synced_at                          TEXT             -- set by the desktop on successful pull; NULL until synced
) STRICT;

-- Composite (station_id, first_seen): serves the dominant desktop read,
-- "this station's events over a time range" (analytics, reports, verify
-- backlog), and by leftmost-prefix also every station_id-only lookup, so it
-- fully replaces a standalone station_id index with no loss.
CREATE INDEX idx_observations_station_id_first_seen ON observations(station_id, first_seen);
-- Cross-station time scans (whole-deployment timelines) still need first_seen alone.
CREATE INDEX idx_observations_first_seen ON observations(first_seen);
-- PARTIAL index on the append-only pull's single hottest query
-- ("what has not synced yet?" = WHERE synced_at IS NULL). It indexes only
-- un-synced rows, so it stays tiny and rows leave it the moment the desktop
-- stamps synced_at. This is the never-evict-unsynced buffer's read path, so it
-- must be fast and bounded.
CREATE INDEX idx_observations_synced_pending ON observations(synced_at) WHERE synced_at IS NULL;
CREATE INDEX idx_observations_qc_state ON observations(qc_state);
CREATE INDEX idx_observations_data_source ON observations(data_source);

-- ============================================================================
-- CHILD_DETECTIONS  (Pi-owned)
-- Multi-taxon sub-detections within one event: each detected taxon is a
-- clearly sub-identified detection within the event. Generalized across both
-- modalities (vision multi-species AND audio multi-species, for example
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
-- Generic channel-as-row design so the water/air/soil sensor SET is entirely
-- a settings.json concern with zero schema or code change between deployment
-- types (a terrestrial station uses the same I2C interface with no code
-- change).
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
    qartod_flag             INTEGER CHECK (qartod_flag IS NULL OR qartod_flag IN (1, 2, 3, 4, 9)),  -- marine channels only
    created_at               TEXT NOT NULL
) STRICT;

CREATE INDEX idx_environmental_readings_observation_id ON environmental_readings(observation_id);
CREATE INDEX idx_environmental_readings_channel ON environmental_readings(channel);

-- ============================================================================
-- SOUNDSCAPE_INDEX_READINGS  (Pi-owned)
-- Optional, default-off continuous marine add-on. Independent of any single
-- observation: it is a continuous time series, not event-gated.
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
-- SPECIES_REFERENCE  (reference cache; written by the species-fetch script)
-- Per-species GBIF occurrence + IUCN cache, fetched once under the user's own
-- credentials with a documented re-fetch path and snapshot stamping. Not
-- bulk-shipped data; populated locally per target species.
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
-- SKILLS  (desktop-owned; synced desktop -> Pi on connect, NOT part of the
-- Pi -> desktop observation pull)
-- The `tier` attribute is the structural enforcement point for skill
-- placement: the engine decides where a skill runs by reading this column,
-- not by inspecting the skill's free-text content.
-- ============================================================================
CREATE TABLE skills (
    id                  TEXT PRIMARY KEY,
    title                 TEXT NOT NULL,
    trigger_condition       TEXT NOT NULL,    -- "when to use"
    instruction              TEXT NOT NULL,    -- "how to apply"
    tier                     TEXT NOT NULL CHECK (tier IN ('deterministic_flag', 'interpretive')),
    created_at                TEXT NOT NULL,
    updated_at                TEXT NOT NULL
) STRICT;

-- ============================================================================
-- OBSERVATION_VERIFICATION  (desktop-owned)
-- One-to-one extension of an observation, written only by the desktop
-- verification step. The `verified` flag here is exactly the gate the dream
-- pass's generative phase reads. Authoritative salience and its remaining
-- ingredients (rarity, baseline deviation, refined anomaly magnitude) live
-- here, never on the Pi-owned observations table, which is what makes the
-- append-only pull structurally incapable of clobbering a desktop-written
-- value.
-- ============================================================================
CREATE TABLE observation_verification (
    observation_id                       TEXT PRIMARY KEY REFERENCES observations(id) ON DELETE CASCADE,
    verified                              INTEGER NOT NULL DEFAULT 0 CHECK (verified IN (0, 1)),  -- the gate the dream pass reads
    rfdetr_version                        TEXT,

    -- The desktop verification verdict, kept as measured model facts so an
    -- RF-DETR result that disagrees with the field screening call is recorded
    -- here rather than by altering the station-owned observation row. These are
    -- the aggregate over every frame the verifier scored for the event, which is
    -- what lets a per-frame misclassification in a long track be caught instead
    -- of trusting a single representative frame. All are nullable because a pure
    -- audio event has no frame to re-score.
    rfdetr_gbif_usage_key                 TEXT,        -- taxon the verifier resolved for the event
    rfdetr_scientific_name                TEXT,
    rfdetr_confidence                     REAL CHECK (rfdetr_confidence IS NULL OR
                                            (rfdetr_confidence >= 0 AND rfdetr_confidence <= 1)),
    rfdetr_agrees_with_field              INTEGER CHECK (rfdetr_agrees_with_field IS NULL OR
                                            rfdetr_agrees_with_field IN (0, 1)),  -- NULL when there is no field label to compare
    frames_scored                         INTEGER,     -- how many frames the verifier scored for this event
    frames_in_agreement                   INTEGER,     -- of those, how many matched the resolved taxon

    salience_authoritative                 REAL CHECK (salience_authoritative IS NULL OR
                                            (salience_authoritative >= 0 AND salience_authoritative <= 1)),
    rarity_score                           REAL,
    baseline_deviation                      REAL,
    anomaly_magnitude_authoritative          REAL,
    verified_at                             TEXT,
    created_at                              TEXT NOT NULL
) STRICT;

-- PARTIAL: the verification gate. The dream pass's generative phase only ever
-- asks "which observations are verified?" (verified = 1), so indexing only
-- those rows by observation_id makes the gate scan minimal and self-documenting.
CREATE INDEX idx_observation_verification_gate ON observation_verification(observation_id) WHERE verified = 1;

-- ============================================================================
-- INTERPRETATIONS  (desktop-owned)
-- The interpretive points owned by the desktop verification step, plus
-- interpretive-skill outputs. Always data_source = 'llm_inferred'; never
-- written by the field tier. A many-rows-per-observation design because an
-- observation can carry several distinct interpretive points simultaneously.
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
-- One row per station per heartbeat interval. Three field groups:
-- effort/coverage, health, energy (nullable, populated only when an energy
-- meter is configured). Effort denominators and energy-per-window are DERIVED
-- at analysis time by aggregating/differencing these rows, never stored here
-- as derived values.
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

    -- Energy (nullable; only when an energy meter is configured)
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
-- Per-channel health error counts, channel-keyed rather than fixed columns
-- for the same no-hardcoding reason as environmental_readings.
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
-- One row per pass. phase_reached + status + checkpoint_watermark together
-- drive the Brain-panel progress display and resumability.
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
    checkpoint_watermark                TEXT,    -- last committed consolidation watermark
    data_source                         TEXT NOT NULL CHECK (data_source = 'dream'),
    created_at                           TEXT NOT NULL
) STRICT;

CREATE INDEX idx_dream_passes_status ON dream_passes(status);

-- ============================================================================
-- PATTERNS  (desktop-owned)
-- One row per candidate hypothesis. Always tagged data_source = 'dream';
-- never presented as an established finding. effect_size_type travels with
-- effect_size because a bare effect size is uninterpretable.
-- ============================================================================
CREATE TABLE patterns (
    id                      TEXT PRIMARY KEY,
    dream_pass_id              TEXT NOT NULL REFERENCES dream_passes(id) ON DELETE CASCADE,
    data_source                 TEXT NOT NULL CHECK (data_source = 'dream'),
    status                       TEXT NOT NULL DEFAULT 'candidate' CHECK (status IN ('candidate', 'verified', 'rejected')),
    dream_phase                  TEXT NOT NULL CHECK (dream_phase IN ('nrem', 'rem')),
    -- The class of hypothesis, so patterns are filterable by kind rather than
    -- only by free-text description. Nullable so a future class can be added
    -- without a migration; the values below are the classes produced today.
    pattern_type                  TEXT CHECK (pattern_type IS NULL OR pattern_type IN
                                    ('temporal_shift', 'co_occurrence', 'envelope_correlation', 'novel_cluster')),
    confidence                    REAL CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    effect_size                    REAL,
    effect_size_type                TEXT CHECK (effect_size_type IS NULL OR effect_size_type IN ('r', 'cohens_d', 'log_odds')),
    -- The named test or method behind effect_size (for example spearman_rho,
    -- mann_kendall), so the statistic that produced a candidate is auditable.
    statistic                        TEXT,
    data_span_start                  TEXT NOT NULL,   -- UTC ISO8601
    data_span_end                     TEXT NOT NULL,   -- UTC ISO8601
    n                                INTEGER NOT NULL,
    p_value                           REAL,            -- nullable: filled by the statistical-validity layer
    q_value                            REAL,            -- nullable: Benjamini-Hochberg FDR-adjusted
    autocorr_adjusted                   INTEGER CHECK (autocorr_adjusted IS NULL OR autocorr_adjusted IN (0, 1)),
    model_version                        TEXT,
    description                           TEXT NOT NULL,
    created_at                             TEXT NOT NULL
) STRICT;

CREATE INDEX idx_patterns_dream_pass_id ON patterns(dream_pass_id);
CREATE INDEX idx_patterns_status ON patterns(status);

-- ============================================================================
-- PATTERN_OBSERVATIONS  (desktop-owned)
-- Junction linking each candidate pattern to its supporting observation UUIDs,
-- the "memories". Makes every dream hypothesis reproducible from its exact
-- source observations.
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
-- SITE_BASELINES  (desktop-owned)
-- The permanent, compact baseline gist the consolidation phase builds and the
-- authoritative salience calculation reads. One row is a running statistical
-- summary for a single cell: a station's readings of one signal, for one
-- taxon group, within one recurring period (for example one calendar month).
--
-- Robust location and scale are kept alongside the ordinary mean and standard
-- deviation. The consolidation phase recomputes the exact median and a scaled
-- median absolute deviation over a cell's full membership each pass, because
-- these resist the very extremes the salience anomaly term is meant to detect,
-- where a mean and standard deviation would be dragged toward an outlier and
-- mask it. The plain mean and standard deviation are retained as descriptive
-- context, not as the anomaly scale.
--
-- This gist is permanent: it is never the target of the downscaling phase,
-- which prunes only derived working memory. It is desktop-owned, so a
-- station-to-desktop pull can never touch it.
-- ============================================================================
CREATE TABLE site_baselines (
    id                          TEXT PRIMARY KEY,
    station_id                    TEXT NOT NULL REFERENCES stations(id) ON DELETE CASCADE,

    -- The cell key. period_granularity names the binning rule (for example
    -- 'month'); period_key is the specific recurring bin within it (for example
    -- '06' for June). group_type/group_key name the taxon grouping, with
    -- 'all' / 'ALL' meaning the cell pools every taxon. signal is the
    -- environmental channel id, drawn from the station's configured channels.
    period_granularity             TEXT NOT NULL,
    period_key                      TEXT NOT NULL,
    group_type                      TEXT NOT NULL CHECK (group_type IN ('species', 'all')),
    group_key                        TEXT NOT NULL,
    signal                           TEXT NOT NULL,

    n                                INTEGER NOT NULL DEFAULT 0,   -- readings summarized in this cell
    median                           REAL,     -- robust center used for the anomaly z-score
    mad_scaled                       REAL,     -- 1.4826 * median absolute deviation: robust scale
    mean                             REAL,     -- descriptive only
    sd                               REAL,     -- descriptive only
    min_value                        REAL,
    max_value                        REAL,

    data_span_start                   TEXT NOT NULL,   -- UTC ISO8601 of the earliest reading in the cell
    data_span_end                      TEXT NOT NULL,  -- UTC ISO8601 of the latest reading in the cell
    data_source                        TEXT NOT NULL CHECK (data_source = 'dream'),
    updated_at                          TEXT NOT NULL,
    created_at                           TEXT NOT NULL,

    -- One cell per (station, period rule, period, taxon group, signal), so a
    -- consolidation pass upserts the recomputed summary onto a stable key.
    UNIQUE (station_id, period_granularity, period_key, group_type, group_key, signal)
) STRICT;

-- The authoritative salience read looks a cell up by its full natural key; the
-- UNIQUE constraint above already provides that index. This second index
-- serves the "every cell for this station" scan a pass uses when refreshing a
-- station's gist.
CREATE INDEX idx_site_baselines_station_id ON site_baselines(station_id);

-- ============================================================================
-- End of schema.sql
-- ============================================================================
