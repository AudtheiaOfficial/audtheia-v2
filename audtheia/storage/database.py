"""Audtheia V2 data-access layer.

Path: audtheia/storage/database.py

This module is the single gateway between Python and the SQLite database
defined in schema.sql. Every other component (the field pipeline, the
desktop analysis tier, the report generator, and the web backend) reads and
writes through the Database class here, so the storage contract lives in
exactly one place.

Design rules this layer enforces:

  - The database path is always supplied by the caller. Nothing here knows
    or guesses a file location; the runtime configuration provides it.
  - Foreign keys are enabled on every connection, because SQLite turns them
    off by default and the setting does not persist across connections.
  - Identity and time are owned by the caller. Observation UUIDs are created
    at capture and flow unchanged through every later stage, and timestamps
    are recorded in UTC by the component that took the reading. The only
    timestamp this layer generates itself is the bookkeeping moment a row is
    confirmed synced.
  - Provenance is mandatory. Every write that stores a data point requires
    its data_source (and, for sensor channels, its measurement status) to be
    passed explicitly, so measured values can never be silently mislabelled
    as inferred ones or the reverse.
  - The field station is the source of truth for the observations it
    captures; the desktop is the source of truth for verification, dream
    output, and reports. The append-only pull from station to desktop only
    ever inserts station-owned rows and never overwrites a value, so a
    desktop-written result can never be clobbered by a later pull.

Only the Python standard library is used (sqlite3, uuid, dataclasses,
base64, datetime, contextlib).
"""

from __future__ import annotations

import base64
import dataclasses
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

__all__ = [
    "Database",
    "Station",
    "Observation",
    "ChildDetection",
    "EnvironmentalReading",
    "SoundscapeReading",
    "SpeciesReference",
    "Skill",
    "ObservationVerification",
    "Interpretation",
    "StationTelemetry",
    "TelemetryError",
    "DreamPass",
    "Pattern",
    "SiteBaseline",
    "utc_now_iso",
    "new_id",
    "run_sync_round",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MAX_EMBEDDING_BYTES",
    "DEFAULT_BUSY_TIMEOUT_MS",
]


# ---------------------------------------------------------------------------
# Defaults. These are starting values only. The runtime configuration is free
# to override every one of them; none of them is a hardcoded policy.
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE = 500
DEFAULT_MAX_EMBEDDING_BYTES = 8192
DEFAULT_BUSY_TIMEOUT_MS = 5000

# The three station-owned tables that move from a field station up to the
# desktop. Each one carries its own synced_at column; the queue of work is
# simply the rows where that column is still empty. Child tables
# (child_detections, environmental_readings, station_telemetry_errors) have no
# synced_at of their own: they travel as part of their parent and are deleted
# with it, so they never need an independent sync cursor.
SYNCABLE_TABLES = ("observations", "soundscape_index_readings", "station_telemetry")


def new_id() -> str:
    """Return a fresh random UUID as text.

    Provided as a convenience for callers that need an identifier (for example
    a child-detection row created during capture). Observation identifiers are
    created by the capture stage itself so that the same value follows the
    record through quality control, sync, verification, and reporting; this
    layer never assigns an observation's primary key on the caller's behalf.
    """
    return str(uuid.uuid4())


def utc_now_iso() -> str:
    """Return the current moment as a UTC ISO 8601 string.

    Used only for the desktop-side bookkeeping timestamp that marks when a row
    was confirmed synced. All scientific timestamps are recorded upstream in
    UTC by the component that made the measurement and are passed in by the
    caller, never invented here.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ===========================================================================
# Row types
#
# One dataclass per table. Field names match column names exactly, which is
# what lets the generic insert helper map a dataclass straight onto a row.
# Required fields (no default) are the columns the database marks NOT NULL
# without a default of its own, so leaving one out is caught at the moment the
# row object is built rather than deep inside a database error. Provenance
# columns (data_source, and the per-channel measurement status) are required
# wherever a value can be measured or inferred, so the distinction is always
# stated, never assumed.
# ===========================================================================


@dataclass
class Station:
    id: str
    station_name: str
    environment_type: str  # marine / terrestrial / estuarine / freshwater / mixed
    created_at: str
    notes: Optional[str] = None


@dataclass
class Observation:
    # Required identity, event window, provenance, and write time.
    id: str
    event_name: str
    station_id: str
    trigger_source: str  # vision / audio / sensor
    first_seen: str
    last_seen: str
    duration: float  # true event length in seconds, even if a stored clip is shorter
    data_source: str  # provenance of the detection event itself
    created_at: str

    # Time-base flag: a capture taken before the clock was disciplined by a
    # satellite fix is marked here rather than being trusted as authoritative.
    time_provisional: int = 0

    # Record lifecycle state. Starts pending; the field quality-control engine
    # advances it, and unclassifiable records are deferred to the desktop.
    qc_state: str = "qc_pending"
    qc_reason: Optional[str] = None

    # Event detail and the metadata the desktop uses to re-weigh a detection.
    representative_frame: Optional[str] = None
    frame_count: Optional[int] = None
    screening_confidence: Optional[float] = None
    screening_model_version: Optional[str] = None
    acoustic_model_version: Optional[str] = None
    gbif_snapshot_date: Optional[str] = None
    iucn_fetch_date: Optional[str] = None

    # Provisional salience and its on-station ingredient. The authoritative
    # salience and the remaining ingredients are computed later on the desktop
    # and stored in a desktop-owned table, so a pull can never overwrite them.
    salience_provisional: Optional[float] = None
    anomaly_magnitude_provisional: Optional[float] = None

    # Audio window. The true duration is always kept even when the stored clip
    # is capped, so a shortened clip never hides how long the event really was.
    audio_clip_path: Optional[str] = None
    audio_true_duration_seconds: Optional[float] = None
    audio_capped: Optional[int] = None

    # One satellite read is one outcome, so a single status covers the fix.
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    gps_elevation: Optional[float] = None
    gps_status: Optional[str] = None

    # Optional detection-frame embedding, off by default. When present it is
    # raw bytes; it is base64-encoded only while travelling in a sync payload.
    feature_embedding: Optional[bytes] = None

    # Empty until the desktop confirms it has safely received this row.
    synced_at: Optional[str] = None


@dataclass
class ChildDetection:
    id: str
    observation_id: str
    modality: str  # vision / audio
    created_at: str
    data_source: str = "model"
    status: str = "measured"
    gbif_usage_key: Optional[str] = None
    scientific_name: Optional[str] = None
    common_name: Optional[str] = None
    confidence: Optional[float] = None
    bbox_x: Optional[float] = None
    bbox_y: Optional[float] = None
    bbox_w: Optional[float] = None
    bbox_h: Optional[float] = None


@dataclass
class EnvironmentalReading:
    id: str
    observation_id: str
    channel: str  # for example water_temp_c, ph, soil_moisture_pct; named in config
    status: str  # measured / not_measured / below_detection_limit / sensor_error / not_applicable
    created_at: str
    data_source: str = "sensor"
    value: Optional[float] = None
    unit: Optional[str] = None
    qartod_flag: Optional[int] = None  # marine channels only: 1 pass, 2 not evaluated, 3 suspect, 4 fail, 9 missing


@dataclass
class SoundscapeReading:
    id: str
    station_id: str
    recorded_at: str
    metric: str
    value: float
    created_at: str
    data_source: str = "sensor"
    synced_at: Optional[str] = None


@dataclass
class SpeciesReference:
    gbif_usage_key: str
    scientific_name: str
    fetched_at: str
    data_source: str = "database"
    common_name: Optional[str] = None
    taxonomic_rank: Optional[str] = None
    iucn_status: Optional[str] = None
    iucn_fetch_date: Optional[str] = None
    gbif_occurrence_count: Optional[int] = None
    gbif_snapshot_date: Optional[str] = None


@dataclass
class Skill:
    id: str
    title: str
    trigger_condition: str
    instruction: str
    tier: str  # deterministic_flag (runs on the station) / interpretive (runs on the desktop)
    created_at: str
    updated_at: str
    # JSON condition the field engine compiles into a pure function, or None.
    condition: Optional[str] = None


@dataclass
class ObservationVerification:
    observation_id: str
    created_at: str
    verified: int = 0  # the gate the dream pass reads before generative work
    rfdetr_version: Optional[str] = None

    # The desktop verification verdict, aggregated over every frame the verifier
    # scored, so an RF-DETR result that disagrees with the field screening call
    # is recorded here rather than by touching the station-owned observation row.
    # All are optional because a pure audio event carries no frame to re-score.
    rfdetr_gbif_usage_key: Optional[str] = None
    rfdetr_scientific_name: Optional[str] = None
    rfdetr_confidence: Optional[float] = None
    rfdetr_agrees_with_field: Optional[int] = None  # None when there is no field label to compare
    frames_scored: Optional[int] = None
    frames_in_agreement: Optional[int] = None

    salience_authoritative: Optional[float] = None
    rarity_score: Optional[float] = None
    baseline_deviation: Optional[float] = None
    anomaly_magnitude_authoritative: Optional[float] = None
    verified_at: Optional[str] = None


@dataclass
class ObservationCorrection:
    """One expert judgement about one claim, appended and never overwritten.

    A correction is a separate assertion standing alongside the model's, not a
    replacement for it. The field call and the desktop verdict stay exactly as
    their producers recorded them, which is what keeps a measured observation
    distinguishable from a human interpretation of it. A change of mind is a
    new row; the newest row per target wins on read.

    No confidence is ever carried here. A percentage is a property of a model's
    output and means nothing attached to a person, so an expert identification
    is displayed as such rather than as a number.
    """

    id: str
    observation_id: str
    verdict: str  # confirm / relabel / reject
    corrector: str
    corrected_at: str
    created_at: str
    data_source: str = "human_expert"
    status: str = "measured"
    detection_id: Optional[str] = None
    modality: Optional[str] = None
    corrected_scientific_name: Optional[str] = None
    corrected_common_name: Optional[str] = None
    corrected_gbif_usage_key: Optional[str] = None
    gbif_snapshot_date: Optional[str] = None
    reason: Optional[str] = None
    salience_corrected: Optional[float] = None


@dataclass
class FrameReview:
    """One expert verdict on one saved frame of a visual event, appended.

    A frame verdict states only whether that saved frame is an accurate
    detection ('accurate'), is not ('inaccurate'), or retracts an earlier
    verdict ('cleared'). It carries no taxon and no confidence: which species a
    frame shows is the model's own per-frame annotation, and the corrected
    species for the whole event is an observation_corrections relabel.

    The measured event is never touched. Its frame_count, duration,
    screening_confidence and salience stay exactly as captured; an 'inaccurate'
    verdict is subtracted only from the expert-curated view and from the frame
    count the analysis trusts. A change of mind is a new row; the newest row per
    (observation_id, frame_index) wins on read.
    """

    id: str
    observation_id: str
    frame_index: int
    verdict: str  # accurate / inaccurate / cleared
    corrector: str
    reviewed_at: str
    created_at: str
    data_source: str = "human_expert"
    reason: Optional[str] = None


@dataclass
class Interpretation:
    id: str
    observation_id: str
    point_type: str
    value: str
    produced_by: str  # verify / dream / skill
    created_at: str
    data_source: str = "llm_inferred"
    confidence: Optional[float] = None
    model_version: Optional[str] = None
    skill_id: Optional[str] = None


@dataclass
class SkillFlagRow:
    """One fired deterministic_flag skill on one event: a derived reading.

    A field flag is a measured or derived reading of the record, never an
    inference, so it lives in its own table with data_source fixed to
    'rule_derived' and never enters the inferred-only interpretations table. It
    stands beside the measurement and never alters it.
    """

    id: str
    observation_id: str
    skill_id: str
    skill_title: str
    flag_name: str
    created_at: str
    data_source: str = "rule_derived"


@dataclass
class StationTelemetry:
    id: str
    station_id: str
    recorded_at: str
    created_at: str
    data_source: str = "sensor"
    # Effort and coverage.
    camera_uptime_seconds: Optional[float] = None
    frames_processed: Optional[int] = None
    frames_dropped: Optional[int] = None
    valid_audio_seconds: Optional[float] = None
    npu_active_seconds: Optional[float] = None
    effective_detection_fps: Optional[float] = None
    # Health.
    station_temperature_c: Optional[float] = None
    buffer_fill_pct: Optional[float] = None
    sync_lag_seconds: Optional[float] = None
    # Energy, present only when a power meter is configured.
    avg_power_w: Optional[float] = None
    cumulative_joules: Optional[float] = None
    synced_at: Optional[str] = None


@dataclass
class TelemetryError:
    id: str
    telemetry_id: str
    channel: str
    error_count: int = 0


@dataclass
class DreamPass:
    id: str
    phase_reached: str  # nrem_a / nrem_b / rem / complete
    status: str  # running / paused / complete / error
    started_at: str
    created_at: str
    data_source: str = "dream"
    station_scope: Optional[str] = None  # empty means all stations
    ended_at: Optional[str] = None
    cycles_completed: int = 0
    work_budget_consumed: int = 0
    checkpoint_watermark: Optional[str] = None


@dataclass
class Pattern:
    id: str
    dream_pass_id: str
    dream_phase: str  # nrem / rem
    data_span_start: str
    data_span_end: str
    n: int
    description: str
    created_at: str
    data_source: str = "dream"
    status: str = "candidate"  # candidate / verified / rejected
    # The class of hypothesis, so patterns are filterable by kind.
    pattern_type: Optional[str] = None  # temporal_shift / co_occurrence / envelope_correlation / novel_cluster
    confidence: Optional[float] = None
    effect_size: Optional[float] = None
    effect_size_type: Optional[str] = None  # r / cohens_d / log_odds
    # The named test or method behind effect_size, for auditability.
    statistic: Optional[str] = None  # for example spearman_rho, mann_kendall
    p_value: Optional[float] = None
    q_value: Optional[float] = None
    autocorr_adjusted: Optional[int] = None
    model_version: Optional[str] = None


@dataclass
class SiteBaseline:
    id: str
    station_id: str
    period_granularity: str
    period_key: str
    group_type: str  # species / all
    group_key: str  # a gbif usage key, or ALL when the cell pools every taxon
    signal: str  # environmental channel id
    data_span_start: str
    data_span_end: str
    updated_at: str
    created_at: str
    data_source: str = "dream"
    n: int = 0
    # Robust location and scale drive the anomaly z-score; the plain mean and
    # standard deviation are descriptive only.
    median: Optional[float] = None
    mad_scaled: Optional[float] = None
    mean: Optional[float] = None
    sd: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


# Maps each row type to its table, so the generic helpers know where a
# dataclass belongs without the caller repeating the table name.
_TABLE_FOR_TYPE = {
    Station: "stations",
    Observation: "observations",
    ChildDetection: "child_detections",
    EnvironmentalReading: "environmental_readings",
    SoundscapeReading: "soundscape_index_readings",
    SpeciesReference: "species_reference",
    Skill: "skills",
    ObservationVerification: "observation_verification",
    ObservationCorrection: "observation_corrections",
    FrameReview: "frame_review",
    Interpretation: "interpretations",
    SkillFlagRow: "skill_flags",
    StationTelemetry: "station_telemetry",
    TelemetryError: "station_telemetry_errors",
    DreamPass: "dream_passes",
    Pattern: "patterns",
    SiteBaseline: "site_baselines",
}


# ===========================================================================
# Database
# ===========================================================================


class Database:
    """A handle on one Audtheia SQLite database file.

    A single instance is shared across a process. Each operation opens a short
    connection, does its work inside one transaction, and closes, which keeps
    the handle safe to use from the field pipeline's worker threads and from
    the desktop web backend's request handlers alike.

    Write-ahead logging is enabled so that a reader (the live feed, the web
    interface) is never blocked by a concurrent writer (capture, sync), and a
    busy timeout lets a brief lock clear instead of failing immediately. This
    suits a local disk or a directly attached SSD, which are the supported
    stores; a networked share is not supported for the live database.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        wal: bool = True,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    ) -> None:
        self.db_path = str(db_path)
        self.wal = wal
        self.busy_timeout_ms = busy_timeout_ms

    # -- connection -------------------------------------------------------

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Open one connection, run one transaction, then close.

        Commits if the body returns normally and rolls back on any error, so a
        multi-step write (an event and its child rows, for example) either
        lands whole or not at all.
        """
        conn = sqlite3.connect(self.db_path, timeout=self.busy_timeout_ms / 1000.0)
        try:
            conn.row_factory = sqlite3.Row
            # Enforce foreign keys for this connection (off by default in SQLite).
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(f"PRAGMA busy_timeout = {int(self.busy_timeout_ms)}")
            if self.wal:
                conn.execute("PRAGMA journal_mode = WAL")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize_schema(self, schema_path: str | Path) -> None:
        """Create every table and index by running schema.sql.

        Safe to call against a fresh file. Uses its own connection because a
        schema script manages its own transactions.
        """
        sql = Path(schema_path).read_text(encoding="utf-8")
        conn = sqlite3.connect(self.db_path, timeout=self.busy_timeout_ms / 1000.0)
        try:
            conn.executescript(sql)
            conn.commit()
        finally:
            conn.close()

    def ensure_schema(self, schema_path: str | Path) -> None:
        """Apply the schema additively: create any table or index a database made
        by an earlier version is missing, and leave every existing table and all
        its data untouched.

        Unlike initialize_schema, which expects a fresh file and fails on a
        database that already has these tables, this is safe to run on every
        launch. It reads the shipped schema and runs an idempotent copy of it,
        with CREATE TABLE and CREATE INDEX rewritten to CREATE ... IF NOT EXISTS
        in memory only, so re-applying it is a no-op for what already exists
        while it adds whatever is new. This is what lets a pulled update work
        with no manual migration step.

        The schema file on disk is never modified, so it keeps its bare CREATE
        statements for a fresh install and for the migration scripts that read
        it verbatim; only this in-memory copy is made idempotent. Only missing
        tables are added, never altered or dropped, so no existing column,
        constraint, or row is touched; a change to an existing table still needs
        its own migration.
        """
        sql = Path(schema_path).read_text(encoding="utf-8")
        sql = re.sub(r"(?i)\bCREATE\s+TABLE\s+(?!IF\s+NOT\s+EXISTS)", "CREATE TABLE IF NOT EXISTS ", sql)
        sql = re.sub(r"(?i)\bCREATE\s+INDEX\s+(?!IF\s+NOT\s+EXISTS)", "CREATE INDEX IF NOT EXISTS ", sql)
        conn = sqlite3.connect(self.db_path, timeout=self.busy_timeout_ms / 1000.0)
        try:
            conn.executescript(sql)
            conn.commit()
        finally:
            conn.close()

    # -- generic row helpers ---------------------------------------------

    @staticmethod
    def _insert_row(conn: sqlite3.Connection, row, *, ignore: bool = False) -> int:
        """Insert one dataclass row into its table.

        Column names come from the dataclass definition, never from caller
        input, so the statement is built from a fixed, trusted set of names.
        When ignore is set, a primary-key collision is skipped instead of
        raising, which is what makes a re-delivered sync batch harmless.
        """
        table = _TABLE_FOR_TYPE[type(row)]
        cols = [f.name for f in dataclasses.fields(row)]
        vals = [getattr(row, c) for c in cols]
        verb = "INSERT OR IGNORE INTO" if ignore else "INSERT INTO"
        placeholders = ", ".join("?" for _ in cols)
        sql = f"{verb} {table} ({', '.join(cols)}) VALUES ({placeholders})"
        cur = conn.execute(sql, vals)
        return cur.rowcount

    @staticmethod
    def _upsert_row(conn: sqlite3.Connection, row, pk: tuple[str, ...]) -> None:
        """Insert a row, or update its non-key columns if the key already exists.

        Used for the reference data the desktop owns and pushes to a station
        (the station registry and the skills), where the desktop's copy is
        authoritative and a later edit such as a rename should replace the
        earlier value. This is deliberately different from the station-to-
        desktop pull, which never updates an existing row.
        """
        table = _TABLE_FOR_TYPE[type(row)]
        cols = [f.name for f in dataclasses.fields(row)]
        vals = [getattr(row, c) for c in cols]
        placeholders = ", ".join("?" for _ in cols)
        updates = ", ".join(f"{c} = excluded.{c}" for c in cols if c not in pk)
        conflict = ", ".join(pk)
        sql = (
            f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT({conflict}) DO UPDATE SET {updates}"
        )
        conn.execute(sql, vals)

    @staticmethod
    def _one(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> Optional[dict]:
        cur = conn.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row is not None else None

    @staticmethod
    def _all(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
        cur = conn.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]

    # -- stations (desktop-authored reference) ---------------------------

    def create_station(self, station: Station) -> None:
        with self.connect() as conn:
            self._insert_row(conn, station)

    def upsert_station(self, station: Station) -> None:
        """Insert or replace a station by its identifier.

        This is how a station defined on the desktop reaches a field station:
        the registry is pushed down so that every observation can carry a real
        reference to its station. An edit on the desktop replaces the earlier
        copy on the next connect.
        """
        with self.connect() as conn:
            self._upsert_row(conn, station, pk=("id",))

    def get_station(self, station_id: str) -> Optional[dict]:
        with self.connect() as conn:
            return self._one(conn, "SELECT * FROM stations WHERE id = ?", (station_id,))

    def get_station_by_name(self, station_name: str) -> Optional[dict]:
        with self.connect() as conn:
            return self._one(
                conn, "SELECT * FROM stations WHERE station_name = ?", (station_name,)
            )

    def list_stations(self) -> list[dict]:
        with self.connect() as conn:
            return self._all(conn, "SELECT * FROM stations ORDER BY station_name")

    # -- observations and their captured children ------------------------

    def insert_observation(
        self,
        observation: Observation,
        *,
        children: Optional[Iterable[ChildDetection]] = None,
        environmental_readings: Optional[Iterable[EnvironmentalReading]] = None,
        max_embedding_bytes: Optional[int] = DEFAULT_MAX_EMBEDDING_BYTES,
    ) -> None:
        """Write one multimodal event and everything captured with it, atomically.

        The event row, every per-taxon detection, and every sensor channel are
        written inside a single transaction, so a half-stored event can never
        appear. If a feature embedding is present and a size limit is given, an
        oversized embedding is rejected here rather than stored or shortened:
        a feature vector is fixed-size, so an over-limit one signals a model or
        configuration mismatch, and refusing it loudly keeps malformed data out
        of the record without ever discarding a valid one.
        """
        self._check_embedding(observation.feature_embedding, max_embedding_bytes)
        with self.connect() as conn:
            self._insert_row(conn, observation)
            for child in children or ():
                self._insert_row(conn, child)
            for reading in environmental_readings or ():
                self._insert_row(conn, reading)

    @staticmethod
    def _check_embedding(embedding: Optional[bytes], max_bytes: Optional[int]) -> None:
        if embedding is not None and max_bytes is not None and len(embedding) > max_bytes:
            raise ValueError(
                f"feature_embedding is {len(embedding)} bytes, over the "
                f"{max_bytes}-byte limit; refusing to store a malformed embedding "
                f"rather than truncate or drop data"
            )

    def get_observation(self, observation_id: str) -> Optional[dict]:
        with self.connect() as conn:
            return self._one(
                conn, "SELECT * FROM observations WHERE id = ?", (observation_id,)
            )

    @staticmethod
    def _observation_filters(station_id, since, until, species, trigger) -> tuple:
        """Build the shared WHERE clauses for listing and counting observations.

        `species` matches events that have a child detection of that name in any
        modality, via a subquery, so the same filter drives the paged list and
        its total count identically.
        """
        clauses: list = []
        params: list = []
        if station_id is not None:
            clauses.append("station_id = ?")
            params.append(station_id)
        if since is not None:
            clauses.append("first_seen >= ?")
            params.append(since)
        if until is not None:
            clauses.append("first_seen <= ?")
            params.append(until)
        if species:
            clauses.append(
                "id IN (SELECT observation_id FROM child_detections WHERE common_name = ?)"
            )
            params.append(species)
        if trigger is not None:
            clauses.append("trigger_source = ?")
            params.append(trigger)
        return clauses, params

    def list_observations(
        self,
        *,
        station_id: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        species: Optional[str] = None,
        trigger: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> list[dict]:
        """List events, newest window first, optionally bounded and paginated.

        `species` narrows to events with a matching detection; `trigger` narrows
        by trigger source (for example 'vision' or 'audio'); `limit`/`offset` page
        the result. Offset is honored only alongside a limit (SQLite pages with
        `LIMIT ... OFFSET`).
        """
        clauses, params = self._observation_filters(station_id, since, until, species, trigger)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        tail = " ORDER BY first_seen DESC, id"
        if limit is not None:
            tail += " LIMIT ?"
            params.append(int(limit))
            if offset:
                tail += " OFFSET ?"
                params.append(int(offset))
        with self.connect() as conn:
            return self._all(conn, f"SELECT * FROM observations{where}{tail}", tuple(params))

    def count_observations(
        self,
        *,
        station_id: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        species: Optional[str] = None,
        trigger: Optional[str] = None,
    ) -> int:
        """Total events matching the same filters as `list_observations`, for paging."""
        clauses, params = self._observation_filters(station_id, since, until, species, trigger)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self.connect() as conn:
            row = self._one(conn, f"SELECT COUNT(*) AS n FROM observations{where}", tuple(params))
            return int((row or {}).get("n") or 0)

    def list_species(self, *, station_id: Optional[str] = None, modality: Optional[str] = None) -> list[str]:
        """Distinct resolved species names, for a species filter.

        Names come from `child_detections.common_name`, optionally bounded to one
        station and/or one modality ('vision' or 'audio'), ordered alphabetically.
        """
        clauses = ["c.common_name IS NOT NULL"]
        params: list = []
        if station_id is not None:
            clauses.append("o.station_id = ?")
            params.append(station_id)
        if modality is not None:
            clauses.append("c.modality = ?")
            params.append(modality)
        where = " WHERE " + " AND ".join(clauses)
        with self.connect() as conn:
            rows = self._all(
                conn,
                "SELECT DISTINCT c.common_name AS name FROM child_detections c "
                "JOIN observations o ON o.id = c.observation_id" + where +
                " ORDER BY c.common_name",
                tuple(params),
            )
            return [r["name"] for r in rows]

    def delete_observations(self, observation_ids: list[str]) -> dict:
        """Delete observations and everything that hangs off them.

        Child rows in child_detections, environmental_readings,
        observation_verification, interpretations, and pattern_observations are
        removed automatically by ON DELETE CASCADE, which the connection enables
        with `PRAGMA foreign_keys = ON`. The visual frame and audio clip paths
        are read before the rows go, and returned, so the caller can remove those
        files from disk. Ids that do not exist are ignored; the returned
        `deleted` list reflects only rows that were actually present.
        """
        ids = [i for i in dict.fromkeys(observation_ids) if i]
        if not ids:
            return {"deleted": [], "media": []}
        marks = ",".join("?" for _ in ids)
        with self.connect() as conn:
            rows = self._all(
                conn,
                f"SELECT id, representative_frame, audio_clip_path "
                f"FROM observations WHERE id IN ({marks})",
                tuple(ids),
            )
            present = [r["id"] for r in rows]
            media: list[str] = []
            for r in rows:
                if r.get("representative_frame"):
                    media.append(r["representative_frame"])
                if r.get("audio_clip_path"):
                    media.append(r["audio_clip_path"])
            if present:
                conn.execute(
                    f"DELETE FROM observations WHERE id IN ({','.join('?' for _ in present)})",
                    tuple(present),
                )
        return {"deleted": present, "media": media}

    def salience_counts(self, station_id: str, species_key: Optional[str]) -> dict:
        """Return the novelty and rarity counts for a species, over all history.

        Species counts are taken over `child_detections.common_name` regardless
        of modality, so a species contributes whether it was seen or heard:

        - ``n_s``         observations of this species at this station,
        - ``t_station``   total observations at this station,
        - ``count_s``     observations of this species across the whole record,
        - ``count_total`` total observations in the record.

        These feed `pipeline.salience`, where local novelty is the smoothed
        surprisal of ``n_s`` within ``t_station`` and global rarity the smoothed
        surprisal of ``count_s`` within ``count_total``. Callers query this
        *before* inserting the new observation, so an event never counts toward
        its own baseline. A missing `species_key` yields zero species counts
        (treated as first-seen), with the station and record totals still
        populated.
        """
        with self.connect() as conn:
            t_station_row = self._one(
                conn,
                "SELECT COUNT(*) AS n FROM observations WHERE station_id = ?",
                (station_id,),
            )
            t_station = int((t_station_row or {}).get("n") or 0)
            total_row = self._one(conn, "SELECT COUNT(*) AS n FROM observations")
            count_total = int((total_row or {}).get("n") or 0)

            n_s = 0
            count_s = 0
            if species_key:
                n_s_row = self._one(
                    conn,
                    "SELECT COUNT(DISTINCT o.id) AS n"
                    " FROM observations o JOIN child_detections c ON c.observation_id = o.id"
                    " WHERE o.station_id = ? AND c.common_name = ?",
                    (station_id, species_key),
                )
                n_s = int((n_s_row or {}).get("n") or 0)
                count_s_row = self._one(
                    conn,
                    "SELECT COUNT(DISTINCT o.id) AS n"
                    " FROM observations o JOIN child_detections c ON c.observation_id = o.id"
                    " WHERE c.common_name = ?",
                    (species_key,),
                )
                count_s = int((count_s_row or {}).get("n") or 0)
        return {"n_s": n_s, "t_station": t_station, "count_s": count_s, "count_total": count_total}

    def list_synced_since(
        self,
        after_synced: Optional[str],
        after_id: Optional[str] = None,
        *,
        limit: Optional[int] = None,
        station_id: Optional[str] = None,
    ) -> list[dict]:
        """List events in the order the desktop received them, after a cursor.

        The longitudinal pass consumes newly-arrived events in a stable order so
        it can stop and resume without missing or repeating one. Arrival order,
        not event time, is the right cursor: a station can sync a batch of
        older-timestamped events late, and ordering by arrival guarantees those
        late arrivals are still picked up on the next pass rather than falling
        behind an event-time watermark. Only rows the desktop has confirmed
        received (synced_at is set) are eligible.

        The cursor is the composite of arrival time and identifier, because a
        single sync stamps one identical arrival time across every row it
        imports, so arrival time alone is not unique. Advancing on the pair means
        a batch boundary that falls inside one sync's rows resumes at the exact
        next row rather than skipping the remaining rows that share that arrival
        time. Passing nothing for the cursor starts from the beginning; passing
        an arrival time without an identifier resumes strictly after that arrival
        time.
        """
        clauses = ["synced_at IS NOT NULL"]
        params: list = []
        if after_synced is not None and after_id is not None:
            # Keyset predicate over (synced_at, id): strictly after the pair.
            clauses.append("(synced_at > ? OR (synced_at = ? AND id > ?))")
            params.extend([after_synced, after_synced, after_id])
        elif after_synced is not None:
            clauses.append("synced_at > ?")
            params.append(after_synced)
        if station_id is not None:
            clauses.append("station_id = ?")
            params.append(station_id)
        where = " WHERE " + " AND ".join(clauses)
        tail = " ORDER BY synced_at, id"
        if limit is not None:
            tail += " LIMIT ?"
            params.append(limit)
        with self.connect() as conn:
            return self._all(conn, f"SELECT * FROM observations{where}{tail}", tuple(params))

    def set_observation_qc(
        self, observation_id: str, qc_state: str, qc_reason: Optional[str] = None
    ) -> None:
        """Record the station-side quality-control outcome for an event.

        This writes only station-owned lifecycle columns; the verified state
        and everything interpretive are written separately on the desktop.
        """
        with self.connect() as conn:
            conn.execute(
                "UPDATE observations SET qc_state = ?, qc_reason = ? WHERE id = ?",
                (qc_state, qc_reason, observation_id),
            )

    def set_observation_provisional_salience(
        self,
        observation_id: str,
        salience_provisional: Optional[float],
        anomaly_magnitude_provisional: Optional[float] = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE observations SET salience_provisional = ?, "
                "anomaly_magnitude_provisional = ? WHERE id = ?",
                (salience_provisional, anomaly_magnitude_provisional, observation_id),
            )

    def list_child_detections(self, observation_id: str) -> list[dict]:
        with self.connect() as conn:
            return self._all(
                conn,
                "SELECT * FROM child_detections WHERE observation_id = ? ORDER BY id",
                (observation_id,),
            )

    def taxon_event_counts(self, station_id: str) -> dict:
        """Count, per taxon, how many of a station's events include that taxon.

        The longitudinal pass reads this to score a taxon's local rarity: a taxon
        present in few of a station's events is rare at that site and draws more
        salience, mirroring the novelty basis of a saliency model. Counting is by
        distinct event, so a taxon detected across many frames of one encounter
        counts once, consistent with the one-event-one-record rule. The taxon is
        keyed by its backbone usage key when one was resolved, falling back to the
        screening model's common name, so a taxon is counted consistently whether
        or not the taxonomic backbone lookup has filled its key yet. A detection
        that carries neither key nor name contributes to no taxon.

        Returns a fresh dictionary with total_events (the station's event count,
        the rarity denominator) and taxon_events (each taxon key mapped to the
        number of the station's events it appears in). Both are read in one
        connection so the pair is internally consistent.
        """
        with self.connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) AS c FROM observations WHERE station_id = ?",
                (station_id,),
            ).fetchone()["c"]
            rows = conn.execute(
                "SELECT COALESCE(c.gbif_usage_key, c.common_name) AS taxon, "
                "COUNT(DISTINCT c.observation_id) AS n "
                "FROM child_detections c "
                "JOIN observations o ON o.id = c.observation_id "
                "WHERE o.station_id = ? "
                "AND COALESCE(c.gbif_usage_key, c.common_name) IS NOT NULL "
                "GROUP BY COALESCE(c.gbif_usage_key, c.common_name)",
                (station_id,),
            ).fetchall()
        return {
            "total_events": int(total),
            "taxon_events": {r["taxon"]: int(r["n"]) for r in rows},
        }

    def insert_environmental_reading(self, reading: EnvironmentalReading) -> None:
        """Store one environmental-channel reading for an existing observation.

        Capture normally writes every channel's reading together with its
        observation. This adds a single reading afterward, which the field
        quality-control step uses to record a channel the station's
        configuration expected but that produced no reading during capture: the
        gap is filled with an explicit missing-data status rather than left
        silent. The row is inserted exactly as the caller built it, so the
        caller is responsible for the measurement status and never for
        inventing a value.
        """
        with self.connect() as conn:
            self._insert_row(conn, reading)

    def list_environmental_readings(self, observation_id: str) -> list[dict]:
        with self.connect() as conn:
            return self._all(
                conn,
                "SELECT * FROM environmental_readings WHERE observation_id = ? ORDER BY channel",
                (observation_id,),
            )

    def list_environmental_readings_for_baseline(
        self,
        station_id: str,
        channel: str,
        *,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> list[dict]:
        """List one station's readings of one channel across many events.

        The longitudinal pass builds a baseline cell by summarizing every
        reading of a signal in a period, which spans many observations, so this
        joins readings back to their parent event for the station filter and the
        event-time window. Each row carries the reading's value, its measurement
        status, and its QARTOD flag together with the parent event's first_seen,
        so the caller can bin by event time and apply the pass-flag precondition
        without a second query. Ordering by event time gives a stable, seekable
        stream.
        """
        clauses = ["o.station_id = ?", "r.channel = ?"]
        params: list = [station_id, channel]
        if since is not None:
            clauses.append("o.first_seen >= ?")
            params.append(since)
        if until is not None:
            clauses.append("o.first_seen <= ?")
            params.append(until)
        where = " AND ".join(clauses)
        sql = (
            "SELECT r.observation_id AS observation_id, r.channel AS channel, "
            "r.value AS value, r.unit AS unit, r.status AS status, "
            "r.qartod_flag AS qartod_flag, o.first_seen AS first_seen "
            "FROM environmental_readings r "
            "JOIN observations o ON o.id = r.observation_id "
            f"WHERE {where} ORDER BY o.first_seen, r.observation_id"
        )
        with self.connect() as conn:
            return self._all(conn, sql, tuple(params))

    # -- soundscape index (optional continuous stream) -------------------

    def insert_soundscape_reading(self, reading: SoundscapeReading) -> None:
        with self.connect() as conn:
            self._insert_row(conn, reading)

    def list_soundscape_readings(
        self, station_id: str, *, since: Optional[str] = None, limit: Optional[int] = None
    ) -> list[dict]:
        params: list = [station_id]
        sql = "SELECT * FROM soundscape_index_readings WHERE station_id = ?"
        if since is not None:
            sql += " AND recorded_at >= ?"
            params.append(since)
        sql += " ORDER BY recorded_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self.connect() as conn:
            return self._all(conn, sql, tuple(params))

    # -- station telemetry and its per-channel error rows ----------------

    def insert_station_telemetry(
        self,
        telemetry: StationTelemetry,
        *,
        errors: Optional[Iterable[TelemetryError]] = None,
    ) -> None:
        """Write one heartbeat and any per-channel error counts atomically."""
        with self.connect() as conn:
            self._insert_row(conn, telemetry)
            for err in errors or ():
                self._insert_row(conn, err)

    def list_station_telemetry(
        self, station_id: str, *, since: Optional[str] = None, limit: Optional[int] = None
    ) -> list[dict]:
        params: list = [station_id]
        sql = "SELECT * FROM station_telemetry WHERE station_id = ?"
        if since is not None:
            sql += " AND recorded_at >= ?"
            params.append(since)
        sql += " ORDER BY recorded_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self.connect() as conn:
            return self._all(conn, sql, tuple(params))

    def list_telemetry_errors(self, telemetry_id: str) -> list[dict]:
        with self.connect() as conn:
            return self._all(
                conn,
                "SELECT * FROM station_telemetry_errors WHERE telemetry_id = ? ORDER BY channel",
                (telemetry_id,),
            )

    # -- species reference cache -----------------------------------------

    def upsert_species_reference(self, species: SpeciesReference) -> None:
        """Store or refresh one species' reference data, keyed by its taxon key.

        A later fetch under a documented refresh path replaces the cached copy
        and updates its snapshot dates, so every dependent record can disclose
        how current its taxonomy and conservation status are.
        """
        with self.connect() as conn:
            self._upsert_row(conn, species, pk=("gbif_usage_key",))

    def get_species_reference(self, gbif_usage_key: str) -> Optional[dict]:
        with self.connect() as conn:
            return self._one(
                conn,
                "SELECT * FROM species_reference WHERE gbif_usage_key = ?",
                (gbif_usage_key,),
            )

    def list_species_reference(self) -> list[dict]:
        with self.connect() as conn:
            return self._all(
                conn, "SELECT * FROM species_reference ORDER BY scientific_name"
            )

    @staticmethod
    def _normalize_taxon_name(name: Optional[str]) -> str:
        """Lowercase a name and reduce every run of non-alphanumeric characters to
        a single space, so a scientific name and a common name match regardless of
        hyphens, extra spaces, or case. It does not repair a misspelling: a name
        the fetch never resolved will simply not match, which is correct."""
        if not name:
            return ""
        return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()

    def find_species_reference_by_name(self, name: Optional[str]) -> Optional[dict]:
        """The cached reference for a detection's label, matched by name.

        A capture labels a detection with the model's own class name, which may be
        a scientific name or a common name, so this matches a reference whose
        scientific_name OR common_name equals it after normalization. Returns None
        when nothing matches (an unfetched or differently-named taxon), so the
        caller leaves the snapshot unset rather than guessing. The reference cache
        is small (one row per target species), so a linear scan is ample and keeps
        the match logic in one readable place.
        """
        target = self._normalize_taxon_name(name)
        if not target:
            return None
        for ref in self.list_species_reference():
            if target in (self._normalize_taxon_name(ref.get("scientific_name")),
                          self._normalize_taxon_name(ref.get("common_name"))):
                return ref
        return None

    def stamp_observation_snapshot(
        self, observation_id: str, gbif_snapshot_date: Optional[str], iucn_fetch_date: Optional[str]
    ) -> bool:
        """Fill an observation's reference snapshot dates, only when they are unset.

        The observation's taxonomic snapshot is meant to be written once, at
        capture. This completes it for a record captured before its species
        reference existed, and it is guarded to touch only a row whose date is
        still NULL, so a value already stamped is never overwritten and no
        measured field is affected. Returns True when a row was filled.
        """
        with self.connect() as conn:
            cur = conn.execute(
                "UPDATE observations SET gbif_snapshot_date = ?, iucn_fetch_date = ? "
                "WHERE id = ? AND gbif_snapshot_date IS NULL AND iucn_fetch_date IS NULL",
                (gbif_snapshot_date, iucn_fetch_date, observation_id),
            )
            return cur.rowcount > 0

    # -- skills (desktop-authored; pushed to a station) ------------------

    def upsert_skill(self, skill: Skill) -> None:
        with self.connect() as conn:
            self._upsert_row(conn, skill, pk=("id",))

    def get_skill(self, skill_id: str) -> Optional[dict]:
        with self.connect() as conn:
            return self._one(conn, "SELECT * FROM skills WHERE id = ?", (skill_id,))

    def list_skills(self, *, tier: Optional[str] = None) -> list[dict]:
        if tier is not None:
            with self.connect() as conn:
                return self._all(
                    conn, "SELECT * FROM skills WHERE tier = ? ORDER BY title", (tier,)
                )
        with self.connect() as conn:
            return self._all(conn, "SELECT * FROM skills ORDER BY title")

    def delete_skill(self, skill_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM skills WHERE id = ?", (skill_id,))

    # -- skill flags (field-derived; measured/derived, never inferred) ---

    def insert_skill_flag(self, flag: SkillFlagRow) -> None:
        """Record one fired deterministic_flag skill, idempotently.

        Re-scanning the same event and skill leaves the single existing row in
        place instead of adding a duplicate, because the flag is a derived
        reading of a measurement that did not change. The unique key on
        (observation_id, skill_id) is what makes a re-run harmless.
        """
        with self.connect() as conn:
            self._insert_row(conn, flag, ignore=True)

    def clear_skill_flag(self, observation_id: str, skill_id: str) -> None:
        """Remove a skill's flag from one event.

        Used by a re-scan that finds a skill's condition no longer holds, for
        example after the skill was edited, so a stale flag never lingers.
        """
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM skill_flags WHERE observation_id = ? AND skill_id = ?",
                (observation_id, skill_id),
            )

    def list_skill_flags(self, observation_id: str) -> list[dict]:
        """The current derived flags on one event, ordered for a stable read."""
        with self.connect() as conn:
            return self._all(
                conn,
                "SELECT * FROM skill_flags WHERE observation_id = ? ORDER BY skill_title, id",
                (observation_id,),
            )

    def count_skill_flags_by_skill(self) -> dict:
        """How many events each skill has flagged, keyed by skill id."""
        with self.connect() as conn:
            rows = self._all(
                conn,
                "SELECT skill_id AS sid, COUNT(*) AS n FROM skill_flags GROUP BY skill_id",
            )
        return {r["sid"]: int(r["n"]) for r in rows}

    # -- observation verification (desktop-owned) ------------------------

    def upsert_observation_verification(self, verification: ObservationVerification) -> None:
        """Write the desktop's verification result for one event.

        This row carries the verified flag the dream pass reads before any
        generative work, plus the authoritative salience and its ingredients.
        It lives in its own table so that a station-to-desktop pull, which only
        ever touches station-owned tables, can never overwrite it.
        """
        with self.connect() as conn:
            self._upsert_row(conn, verification, pk=("observation_id",))

    def get_observation_verification(self, observation_id: str) -> Optional[dict]:
        with self.connect() as conn:
            return self._one(
                conn,
                "SELECT * FROM observation_verification WHERE observation_id = ?",
                (observation_id,),
            )

    def list_verified_observation_ids(
        self, *, station_id: Optional[str] = None
    ) -> list[str]:
        """Return the identifiers of events the desktop has cleared.

        This is the gate the dream pass applies before generative pattern work:
        statistics are built from everything, but new claims are only allowed to
        rest on confirmed detections.
        """
        sql = (
            "SELECT v.observation_id AS oid FROM observation_verification v "
            "WHERE v.verified = 1"
        )
        params: tuple = ()
        if station_id is not None:
            sql += (
                " AND v.observation_id IN "
                "(SELECT id FROM observations WHERE station_id = ?)"
            )
            params = (station_id,)
        sql += " ORDER BY v.observation_id"
        with self.connect() as conn:
            return [r["oid"] for r in conn.execute(sql, params).fetchall()]

    def set_authoritative_salience(
        self,
        observation_id: str,
        *,
        salience_authoritative: Optional[float],
        baseline_deviation: Optional[float] = None,
        anomaly_magnitude_authoritative: Optional[float] = None,
    ) -> None:
        """Write the desktop's final authoritative salience and its ingredients.

        This updates only the three salience columns on an existing
        verification row, leaving the verification verdict, the gate, and the
        interpreter's rarity ingredient exactly as the verification step wrote
        them. The longitudinal pass computes salience against the full baseline,
        which the verification step could not see, so this is where the interim
        salience is replaced by the value derived from the site gist. The row
        must already exist, which it does for any observation the desktop has
        seen, so this never creates a verification record on its own.
        """
        with self.connect() as conn:
            conn.execute(
                "UPDATE observation_verification SET "
                "salience_authoritative = ?, baseline_deviation = ?, "
                "anomaly_magnitude_authoritative = ? WHERE observation_id = ?",
                (
                    salience_authoritative,
                    baseline_deviation,
                    anomaly_magnitude_authoritative,
                    observation_id,
                ),
            )

    # -- expert corrections (human-owned) --------------------------------

    def add_correction(
        self,
        observation_id: str,
        *,
        verdict: str,
        corrector: str,
        detection_id: Optional[str] = None,
        modality: Optional[str] = None,
        corrected_scientific_name: Optional[str] = None,
        corrected_common_name: Optional[str] = None,
        corrected_gbif_usage_key: Optional[str] = None,
        gbif_snapshot_date: Optional[str] = None,
        reason: Optional[str] = None,
        salience_corrected: Optional[float] = None,
    ) -> dict:
        """Append one expert judgement and return the row as stored.

        Nothing already in the database is modified. The model's screening
        confidence, the field species call, and the desktop verification verdict
        are all left untouched, because a correction is an additional claim
        rather than an edit to an existing one. The table's own CHECK
        constraints are the last line of defence on the verdict vocabulary and
        on whether a name is required, so this deliberately does not duplicate
        that validation in Python where the two could drift apart.
        """
        row = ObservationCorrection(
            id=new_id(),
            observation_id=observation_id,
            verdict=verdict,
            corrector=corrector,
            corrected_at=utc_now_iso(),
            created_at=utc_now_iso(),
            detection_id=detection_id,
            modality=modality,
            corrected_scientific_name=corrected_scientific_name,
            corrected_common_name=corrected_common_name,
            corrected_gbif_usage_key=corrected_gbif_usage_key,
            gbif_snapshot_date=gbif_snapshot_date,
            reason=reason,
            salience_corrected=salience_corrected,
        )
        with self.connect() as conn:
            self._insert_row(conn, row)
            return self._one(
                conn, "SELECT * FROM observation_corrections WHERE id = ?", (row.id,)
            )

    def corrections_for_observation(self, observation_id: str) -> list[dict]:
        """Every correction on one event, newest first.

        Newest first because the history is read to answer "what does the expert
        currently say", and the answer is the first row rather than the last.
        The tiebreak is rowid rather than the primary key, because the key is a
        random UUID and would order two same-instant corrections arbitrarily,
        whereas rowid follows insertion order and so keeps a fast
        confirm-then-change-mind in the sequence it actually happened.
        """
        with self.connect() as conn:
            return self._all(
                conn,
                "SELECT * FROM observation_corrections WHERE observation_id = ? "
                "ORDER BY corrected_at DESC, rowid DESC",
                (observation_id,),
            )

    def latest_correction(
        self, observation_id: str, detection_id: Optional[str] = None
    ) -> Optional[dict]:
        """The current expert position on one target.

        A detection_id names one box inside a multi-taxon event; omitting it
        asks about the event as a whole. The two are kept strictly separate:
        an event-level rejection does not answer a question about a specific
        box, and a correction to one box says nothing about its neighbours.
        """
        sql = "SELECT * FROM observation_corrections WHERE observation_id = ?"
        params: tuple = (observation_id,)
        if detection_id is None:
            sql += " AND detection_id IS NULL"
        else:
            sql += " AND detection_id = ?"
            params += (detection_id,)
        sql += " ORDER BY corrected_at DESC, rowid DESC LIMIT 1"
        with self.connect() as conn:
            try:
                return self._one(conn, sql, params)
            except sqlite3.OperationalError:
                # A database that predates the corrections migration has no such
                # table. That is a database waiting to be migrated, not a fault,
                # and it must not cost the reader their detection list: an
                # unreviewed event and an unmigratable one look the same here,
                # and both are correctly described as "no correction yet".
                return None

    def correction_counts(self) -> dict:
        """How much of the record a person has actually reviewed.

        Counts distinct targets rather than rows so that a corrector who
        changed their mind three times about one detection is not mistaken for
        three reviews. This is what the Brain audit view reports against.
        """
        with self.connect() as conn:
            rows = self._all(
                conn,
                "SELECT verdict, COUNT(*) AS n FROM ("
                "  SELECT observation_id, detection_id, verdict FROM ("
                "    SELECT observation_id, detection_id, verdict,"
                "      ROW_NUMBER() OVER ("
                "        PARTITION BY observation_id, IFNULL(detection_id, '')"
                "        ORDER BY corrected_at DESC, rowid DESC) AS rn"
                "    FROM observation_corrections"
                "  ) WHERE rn = 1"
                ") GROUP BY verdict",
            )
        counts = {"confirm": 0, "relabel": 0, "reject": 0}
        for row in rows:
            counts[row["verdict"]] = row["n"]
        counts["total"] = sum(counts[v] for v in ("confirm", "relabel", "reject"))
        return counts

    # -- per-frame review (desktop-owned) --------------------------------

    def add_frame_review(
        self,
        observation_id: str,
        frame_index: int,
        *,
        verdict: str,
        corrector: str,
        reason: Optional[str] = None,
    ) -> dict:
        """Append one per-frame verdict and return the row as stored.

        Nothing measured is modified. The event's frame_count, duration,
        confidence and salience are left exactly as captured; this appends a
        separate human claim about one frame. The table's CHECK constraint is
        the authority on the verdict vocabulary, so this does not duplicate that
        validation in Python where the two could drift apart.
        """
        row = FrameReview(
            id=new_id(),
            observation_id=observation_id,
            frame_index=frame_index,
            verdict=verdict,
            corrector=corrector,
            reviewed_at=utc_now_iso(),
            created_at=utc_now_iso(),
            reason=reason,
        )
        with self.connect() as conn:
            self._insert_row(conn, row)
            return self._one(conn, "SELECT * FROM frame_review WHERE id = ?", (row.id,))

    def frame_reviews_for_observation(self, observation_id: str) -> list[dict]:
        """The current verdict per reviewed frame of one event.

        One row per frame_index, the most recent verdict for that frame, in
        frame order. A frame whose latest verdict is 'cleared' is returned as
        such so a reader can tell a retracted frame from an unreviewed one; the
        summary below folds 'cleared' back into unreviewed. Degrades to an empty
        list on a database that predates this migration, so an unmigrated record
        reads as "no review yet" rather than costing the reader their frames.
        """
        with self.connect() as conn:
            try:
                return self._all(
                    conn,
                    "SELECT observation_id, frame_index, verdict, corrector, reviewed_at, reason FROM ("
                    "  SELECT *, ROW_NUMBER() OVER ("
                    "    PARTITION BY frame_index ORDER BY reviewed_at DESC, rowid DESC) AS rn"
                    "  FROM frame_review WHERE observation_id = ?"
                    ") WHERE rn = 1 ORDER BY frame_index",
                    (observation_id,),
                )
            except sqlite3.OperationalError:
                return []

    def frame_review_summary(self, observation_id: str) -> dict:
        """How many of an event's frames an expert has marked accurate or not.

        Counts distinct frames by their current verdict, so a reviewer who
        changed their mind about one frame is counted once. 'cleared' folds back
        into unreviewed. Returns zeros on an unmigrated database rather than
        raising, matching the graceful degradation the corrections reader uses.
        """
        counts = {"accurate": 0, "inaccurate": 0}
        with self.connect() as conn:
            try:
                rows = self._all(
                    conn,
                    "SELECT verdict, COUNT(*) AS n FROM ("
                    "  SELECT frame_index, verdict, ROW_NUMBER() OVER ("
                    "    PARTITION BY frame_index ORDER BY reviewed_at DESC, rowid DESC) AS rn"
                    "  FROM frame_review WHERE observation_id = ?"
                    ") WHERE rn = 1 AND verdict IN ('accurate', 'inaccurate') GROUP BY verdict",
                    (observation_id,),
                )
            except sqlite3.OperationalError:
                rows = []
        for row in rows:
            counts[row["verdict"]] = row["n"]
        counts["reviewed"] = counts["accurate"] + counts["inaccurate"]
        return counts

    # -- interpretations (desktop-owned) ---------------------------------

    def insert_interpretation(self, interpretation: Interpretation) -> None:
        with self.connect() as conn:
            self._insert_row(conn, interpretation)

    def list_interpretations(self, observation_id: str) -> list[dict]:
        with self.connect() as conn:
            return self._all(
                conn,
                "SELECT * FROM interpretations WHERE observation_id = ? ORDER BY created_at, id",
                (observation_id,),
            )

    # -- dream passes, patterns, and their supporting memories -----------

    def create_dream_pass(self, dream_pass: DreamPass) -> None:
        with self.connect() as conn:
            self._insert_row(conn, dream_pass)

    def update_dream_pass(
        self,
        dream_pass_id: str,
        *,
        phase_reached: Optional[str] = None,
        status: Optional[str] = None,
        ended_at: Optional[str] = None,
        cycles_completed: Optional[int] = None,
        work_budget_consumed: Optional[int] = None,
        checkpoint_watermark: Optional[str] = None,
    ) -> None:
        """Advance a dream pass's progress, committing per cycle.

        Only the fields supplied are changed, so a per-cycle checkpoint can move
        the watermark forward without disturbing anything else, which is what
        lets a paused pass resume from exactly where it stopped.
        """
        sets = []
        params: list = []
        for name, value in (
            ("phase_reached", phase_reached),
            ("status", status),
            ("ended_at", ended_at),
            ("cycles_completed", cycles_completed),
            ("work_budget_consumed", work_budget_consumed),
            ("checkpoint_watermark", checkpoint_watermark),
        ):
            if value is not None:
                sets.append(f"{name} = ?")
                params.append(value)
        if not sets:
            return
        params.append(dream_pass_id)
        with self.connect() as conn:
            conn.execute(
                f"UPDATE dream_passes SET {', '.join(sets)} WHERE id = ?", tuple(params)
            )

    def get_dream_pass(self, dream_pass_id: str) -> Optional[dict]:
        with self.connect() as conn:
            return self._one(
                conn, "SELECT * FROM dream_passes WHERE id = ?", (dream_pass_id,)
            )

    def list_dream_passes(self, *, status: Optional[str] = None) -> list[dict]:
        if status is not None:
            with self.connect() as conn:
                return self._all(
                    conn,
                    "SELECT * FROM dream_passes WHERE status = ? ORDER BY started_at DESC",
                    (status,),
                )
        with self.connect() as conn:
            return self._all(
                conn, "SELECT * FROM dream_passes ORDER BY started_at DESC"
            )

    def insert_pattern(
        self, pattern: Pattern, *, observation_ids: Optional[Iterable[str]] = None
    ) -> None:
        """Write one candidate pattern and link it to the events behind it.

        The links make every hypothesis reproducible from its exact source
        observations. The pattern and its links are written together so a
        pattern never exists without its supporting memories.
        """
        with self.connect() as conn:
            self._insert_row(conn, pattern)
            for oid in observation_ids or ():
                conn.execute(
                    "INSERT OR IGNORE INTO pattern_observations (pattern_id, observation_id) "
                    "VALUES (?, ?)",
                    (pattern.id, oid),
                )

    def link_pattern_observations(
        self, pattern_id: str, observation_ids: Iterable[str]
    ) -> None:
        with self.connect() as conn:
            for oid in observation_ids:
                conn.execute(
                    "INSERT OR IGNORE INTO pattern_observations (pattern_id, observation_id) "
                    "VALUES (?, ?)",
                    (pattern_id, oid),
                )

    def set_pattern_status(self, pattern_id: str, status: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE patterns SET status = ? WHERE id = ?", (status, pattern_id)
            )

    def get_pattern(self, pattern_id: str) -> Optional[dict]:
        with self.connect() as conn:
            return self._one(conn, "SELECT * FROM patterns WHERE id = ?", (pattern_id,))

    def list_patterns(
        self, *, dream_pass_id: Optional[str] = None, status: Optional[str] = None
    ) -> list[dict]:
        clauses = []
        params: list = []
        if dream_pass_id is not None:
            clauses.append("dream_pass_id = ?")
            params.append(dream_pass_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self.connect() as conn:
            return self._all(
                conn, f"SELECT * FROM patterns{where} ORDER BY created_at DESC", tuple(params)
            )

    def list_pattern_observations(self, pattern_id: str) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT observation_id FROM pattern_observations WHERE pattern_id = ? "
                "ORDER BY observation_id",
                (pattern_id,),
            ).fetchall()
            return [r["observation_id"] for r in rows]

    # -- site baselines (desktop-owned, the permanent gist) --------------

    def upsert_site_baseline(self, baseline: SiteBaseline) -> None:
        """Insert or replace one baseline cell by its natural key.

        A consolidation pass recomputes a cell's summary from its full
        membership and writes the result onto the stable cell key, so re-running
        a pass refreshes the gist in place rather than accumulating duplicates.
        The insert carries the created_at the caller supplies for a new cell;
        the update leaves created_at untouched and moves only the recomputed
        summary and updated_at forward.
        """
        with self.connect() as conn:
            cols = [f.name for f in dataclasses.fields(baseline)]
            vals = [getattr(baseline, c) for c in cols]
            placeholders = ", ".join("?" for _ in cols)
            conflict = (
                "station_id, period_granularity, period_key, "
                "group_type, group_key, signal"
            )
            # created_at is preserved on update; everything else is refreshed.
            updates = ", ".join(
                f"{c} = excluded.{c}" for c in cols if c != "created_at"
            )
            conn.execute(
                f"INSERT INTO site_baselines ({', '.join(cols)}) "
                f"VALUES ({placeholders}) "
                f"ON CONFLICT({conflict}) DO UPDATE SET {updates}",
                vals,
            )

    def get_site_baseline(
        self,
        station_id: str,
        period_granularity: str,
        period_key: str,
        group_type: str,
        group_key: str,
        signal: str,
    ) -> Optional[dict]:
        """Fetch one baseline cell by its full natural key, or nothing."""
        with self.connect() as conn:
            return self._one(
                conn,
                "SELECT * FROM site_baselines WHERE station_id = ? "
                "AND period_granularity = ? AND period_key = ? "
                "AND group_type = ? AND group_key = ? AND signal = ?",
                (station_id, period_granularity, period_key, group_type, group_key, signal),
            )

    def list_site_baselines(
        self, *, station_id: Optional[str] = None
    ) -> list[dict]:
        """List baseline cells, optionally for one station."""
        if station_id is not None:
            with self.connect() as conn:
                return self._all(
                    conn,
                    "SELECT * FROM site_baselines WHERE station_id = ? "
                    "ORDER BY period_granularity, period_key, group_type, group_key, signal",
                    (station_id,),
                )
        with self.connect() as conn:
            return self._all(
                conn,
                "SELECT * FROM site_baselines "
                "ORDER BY station_id, period_granularity, period_key, group_type, group_key, signal",
            )

    # ====================================================================
    # Append-only sync: field station to desktop
    #
    # This layer owns the meaning of a sync, not the network that carries it.
    # A station exports the rows it has not yet had confirmed, the desktop
    # imports them, and the station marks them confirmed once the desktop says
    # it has them. The transport that moves a batch between two machines is
    # provided separately by the web backend.
    #
    # The work queue is simply the rows whose synced_at is still empty.
    # Because every primary key is a globally unique identifier and an import
    # ignores a key it already holds, a batch can be delivered more than once
    # with no duplicates, and an interruption is recovered by re-exporting:
    # any row the desktop did not confirm is still in the queue and comes
    # again, while any row it did confirm is skipped on the next import.
    # ====================================================================

    @staticmethod
    def _encode_embedding(value) -> Optional[str]:
        if value is None:
            return None
        return base64.b64encode(value).decode("ascii")

    @staticmethod
    def _decode_embedding(value) -> Optional[bytes]:
        if value is None:
            return None
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        return base64.b64decode(value)

    def export_unsynced_batch(
        self,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        forward_embeddings: bool = False,
        max_embedding_bytes: Optional[int] = DEFAULT_MAX_EMBEDDING_BYTES,
    ) -> dict:
        """Collect the next batch of unconfirmed station-owned rows.

        Each event is returned with its captured children attached, and each
        telemetry heartbeat with its error rows, so a parent always travels
        with the rows that belong to it. Rows are taken in write order so a
        large backlog is cleared in stable, repeatable pages.

        Embeddings are off by default. When carried, an embedding is encoded so
        the batch is plain serializable data; an embedding over the size limit
        is defensively left out of the payload rather than carried, though the
        limit is normally enforced earlier, when the row is first written.
        """
        with self.connect() as conn:
            observations = self._export_observations(
                conn, batch_size, forward_embeddings, max_embedding_bytes
            )
            soundscape = self._export_simple(
                conn, "soundscape_index_readings", batch_size, order="recorded_at"
            )
            telemetry = self._export_telemetry(conn, batch_size)
        return {
            "observations": observations,
            "soundscape_index_readings": soundscape,
            "station_telemetry": telemetry,
        }

    def _export_observations(
        self,
        conn: sqlite3.Connection,
        batch_size: int,
        forward_embeddings: bool,
        max_embedding_bytes: Optional[int],
    ) -> list[dict]:
        rows = conn.execute(
            "SELECT * FROM observations WHERE synced_at IS NULL "
            "ORDER BY created_at, id LIMIT ?",
            (batch_size,),
        ).fetchall()
        out: list[dict] = []
        for row in rows:
            d = dict(row)
            raw = d.get("feature_embedding")
            if not forward_embeddings:
                d["feature_embedding"] = None
            elif raw is not None and max_embedding_bytes is not None and len(raw) > max_embedding_bytes:
                # Should not occur, since oversized embeddings are refused at
                # write time; left out here only as a last-resort guard so a
                # malformed row can never bloat a payload.
                d["feature_embedding"] = None
            else:
                d["feature_embedding"] = self._encode_embedding(raw)
            oid = d["id"]
            d["child_detections"] = self._all(
                conn,
                "SELECT * FROM child_detections WHERE observation_id = ? ORDER BY id",
                (oid,),
            )
            d["environmental_readings"] = self._all(
                conn,
                "SELECT * FROM environmental_readings WHERE observation_id = ? ORDER BY id",
                (oid,),
            )
            out.append(d)
        return out

    def _export_telemetry(self, conn: sqlite3.Connection, batch_size: int) -> list[dict]:
        rows = conn.execute(
            "SELECT * FROM station_telemetry WHERE synced_at IS NULL "
            "ORDER BY created_at, id LIMIT ?",
            (batch_size,),
        ).fetchall()
        out: list[dict] = []
        for row in rows:
            d = dict(row)
            d["station_telemetry_errors"] = self._all(
                conn,
                "SELECT * FROM station_telemetry_errors WHERE telemetry_id = ? ORDER BY id",
                (d["id"],),
            )
            out.append(d)
        return out

    def _export_simple(
        self, conn: sqlite3.Connection, table: str, batch_size: int, *, order: str
    ) -> list[dict]:
        return self._all(
            conn,
            f"SELECT * FROM {table} WHERE synced_at IS NULL ORDER BY {order}, id LIMIT ?",
            (batch_size,),
        )

    def import_batch(self, batch: dict, *, synced_at: Optional[str] = None) -> dict:
        """Ingest a batch on the desktop, skipping anything already held.

        Every row is inserted only if its identifier is new, so re-delivering a
        batch changes nothing and an existing desktop row is never altered. The
        moment of receipt is stamped on each top-level row's own copy here.
        Returns, per table, the identifiers now present on the desktop, which is
        what the station marks confirmed; that set is the same whether a row was
        freshly inserted or already held from an earlier, interrupted delivery.
        """
        stamp = synced_at or utc_now_iso()
        confirmed = {t: [] for t in SYNCABLE_TABLES}
        with self.connect() as conn:
            for obs in batch.get("observations", []):
                d = dict(obs)
                children = d.pop("child_detections", []) or []
                readings = d.pop("environmental_readings", []) or []
                d["feature_embedding"] = self._decode_embedding(d.get("feature_embedding"))
                d["synced_at"] = stamp
                self._insert_dict(conn, "observations", d, ignore=True)
                for child in children:
                    self._insert_dict(conn, "child_detections", dict(child), ignore=True)
                for reading in readings:
                    self._insert_dict(conn, "environmental_readings", dict(reading), ignore=True)
                confirmed["observations"].append(d["id"])

            for snd in batch.get("soundscape_index_readings", []):
                d = dict(snd)
                d["synced_at"] = stamp
                self._insert_dict(conn, "soundscape_index_readings", d, ignore=True)
                confirmed["soundscape_index_readings"].append(d["id"])

            for tel in batch.get("station_telemetry", []):
                d = dict(tel)
                errors = d.pop("station_telemetry_errors", []) or []
                d["synced_at"] = stamp
                self._insert_dict(conn, "station_telemetry", d, ignore=True)
                for err in errors:
                    self._insert_dict(conn, "station_telemetry_errors", dict(err), ignore=True)
                confirmed["station_telemetry"].append(d["id"])
        return confirmed

    @staticmethod
    def _insert_dict(
        conn: sqlite3.Connection, table: str, d: dict, *, ignore: bool = False
    ) -> int:
        """Insert a plain row dictionary into a named station-owned table.

        The table name is one of a fixed internal set and the column names come
        from the dictionary's keys, which originate from this module's own
        export, so the statement is built only from trusted names.
        """
        cols = list(d.keys())
        vals = [d[c] for c in cols]
        verb = "INSERT OR IGNORE INTO" if ignore else "INSERT INTO"
        placeholders = ", ".join("?" for _ in cols)
        sql = f"{verb} {table} ({', '.join(cols)}) VALUES ({placeholders})"
        return conn.execute(sql, vals).rowcount

    def mark_synced(
        self, table: str, ids: Iterable[str], *, synced_at: Optional[str] = None
    ) -> int:
        """Stamp confirmed rows on the station side once the desktop has them.

        Called only after the desktop confirms receipt. This flag is what the
        rolling-buffer cleaner reads, so a row is only ever cleanable after the
        desktop is known to hold it. An already-stamped row is left untouched,
        so a repeated confirmation is harmless.
        """
        if table not in SYNCABLE_TABLES:
            raise ValueError(f"{table} is not a syncable table")
        stamp = synced_at or utc_now_iso()
        ids = list(ids)
        if not ids:
            return 0
        with self.connect() as conn:
            cur = conn.executemany(
                f"UPDATE {table} SET synced_at = ? WHERE id = ? AND synced_at IS NULL",
                [(stamp, i) for i in ids],
            )
            return cur.rowcount

    def count_unsynced(self) -> dict:
        """Report how many rows in each syncable table still await confirmation.

        Feeds the desktop's storage view and the station's own buffer status, so
        a growing backlog is visible to the user well before any limit is near.
        """
        with self.connect() as conn:
            return {
                t: conn.execute(
                    f"SELECT COUNT(*) AS c FROM {t} WHERE synced_at IS NULL"
                ).fetchone()["c"]
                for t in SYNCABLE_TABLES
            }

    def clean_synced(self) -> dict:
        """Reclaim space on a station by deleting only confirmed rows.

        A row is removed only when its synced_at is set, meaning the desktop
        holds the authoritative copy. Rows still awaiting confirmation are never
        touched, so the station cannot lose data that has not yet reached the
        desktop. Captured children are removed with their parent automatically.
        """
        removed = {}
        with self.connect() as conn:
            for table in SYNCABLE_TABLES:
                cur = conn.execute(f"DELETE FROM {table} WHERE synced_at IS NOT NULL")
                removed[table] = cur.rowcount
        return removed


def run_sync_round(
    station: Database,
    desktop: Database,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    forward_embeddings: bool = False,
    max_embedding_bytes: Optional[int] = DEFAULT_MAX_EMBEDDING_BYTES,
) -> dict:
    """Run one export, import, and confirm cycle between two open databases.

    Convenience for tests and for the case where both databases are reachable
    in one process. In a field deployment the same three steps run across the
    network: a station exports, the desktop imports, and the station marks the
    confirmed rows. Returns the count confirmed per table this round.
    """
    batch = station.export_unsynced_batch(
        batch_size=batch_size,
        forward_embeddings=forward_embeddings,
        max_embedding_bytes=max_embedding_bytes,
    )
    confirmed = desktop.import_batch(batch)
    counts = {}
    for table in SYNCABLE_TABLES:
        counts[table] = station.mark_synced(table, confirmed[table])
    return counts
