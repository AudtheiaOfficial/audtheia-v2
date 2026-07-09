"""Audtheia configuration loader.

Path: audtheia/config.py

This module is the single place that reads, validates, and resolves the
Audtheia configuration. Every other component (the field pipeline, the desktop
analysis tier, the report generator, and the web backend) reads its settings
through the Settings object returned here, so no value is ever hardcoded and no
module parses the configuration file on its own.

What the loader does:

  - Reads config/settings.json (the committed, secret-free configuration).
  - Reads an optional, gitignored config/secrets.json and lets environment
    variables override any secret, so credentials stay out of the repository.
  - Resolves every path against the repository root using pathlib, so one
    forward-slash configuration works the same on Windows, macOS, Linux, and
    Raspberry Pi OS. Absolute paths are honored so a desktop store can live on
    an external drive.
  - Validates the file against the database's controlled vocabularies and the
    expected shape before any component runs, so a malformed configuration is
    caught immediately and explained, rather than failing deep inside a later
    stage.

Only the Python standard library is used (json, os, pathlib, datetime, and the
optional zoneinfo for named time zones).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Controlled vocabularies.
#
# These mirror the CHECK constraints and the operating contract that the
# database schema defines, so the loader can reject an out-of-range value
# before it ever reaches a write. The channel set itself is deliberately NOT
# fixed here: channels are a per-deployment setting, so the loader validates
# their shape, never their membership.
# ---------------------------------------------------------------------------

ENVIRONMENT_TYPES = ("marine", "terrestrial", "estuarine", "freshwater", "mixed")

NODE_ROLES = ("desktop", "pi")

ANALYSIS_LOCATIONS = ("pi", "desktop")

REPORT_SCHEDULES = ("daily", "weekly", "biweekly", "on_demand")
DREAM_SCHEDULES = ("daily", "weekly", "biweekly", "manual")
REPORT_FORMATS = ("pdf", "csv")

REPRESENTATIVE_FRAME_RULES = ("highest_confidence",)

# The recurring-period binning rules the longitudinal baseline can use. A cell
# groups a signal's readings by one of these, so a January reading is compared
# against other Januaries rather than against a year-round average that the
# seasonal cycle would dominate.
BASELINE_PERIOD_GRANULARITIES = ("month", "iso_week", "doy")

# The multichannel anomaly aggregators the salience calculation can use. The
# independent chi-square aggregator combines each qualifying channel's squared
# robust z-score into one calibrated surprise value whose scale does not grow
# with the number of channels a station happens to carry.
SALIENCE_AGGREGATORS = ("chi2_independent",)

# Documented defaults for the analysis blocks, applied by the accessors when a
# configuration omits the block. These are starting values, not hardcoded
# policy: a configuration is free to override every one. The salience weights
# are declared prioritization priors, not empirically fitted quantities, so the
# score they produce ranks attention and is never treated as an inferential
# statistic.
DEFAULT_BASELINE_PERIOD_GRANULARITY = "month"
DEFAULT_SALIENCE_WEIGHTS = {"confidence": 0.40, "anomaly": 0.60, "rarity": 0.0}
DEFAULT_SALIENCE_MIN_EFFECTIVE_N = 8
DEFAULT_SALIENCE_AGGREGATOR = "chi2_independent"

# Documented defaults for the analysis thresholds, applied by the accessor when
# a configuration omits a value. Every value here equals the constant its module
# used before these moved into configuration, so an omitted block reproduces the
# earlier behavior exactly. A configuration is free to override any of them.
DEFAULT_THRESHOLDS = {
    "field_qc": {"pass_confidence": 0.10},
    "verification": {"clear_confidence": 0.50, "max_frames_scored": 32},
    "dream": {
        "min_periods_for_trend": 4,
        "min_events_for_correlation": 8,
        "min_events_for_co_occurrence": 8,
        "min_abs_effect": 0.2,
        "max_p_value": 0.05,
    },
}

# Documented defaults for the privacy block. Discarding human detections is on by
# default; the human class set is empty by default, so with no configured human
# class the discard matches nothing and is inert until a deployment names the
# class or classes its own detection model uses for people.
DEFAULT_PRIVACY = {"discard_human_detections": True, "human_class_names": []}

# Optional, additive site descriptor. Independent of environment_type, which is
# the schema's fixed five-value field. A station may set one of these for a
# richer, reviewer-legible description of where it is deployed; it is never
# required and never changes how a value is stored.
ALLOWED_HABITATS = frozenset(
    {
        # Marine
        "coral_reef",
        "rocky_reef",
        "kelp_forest",
        "seagrass_meadow",
        "open_ocean",
        "deep_sea",
        "intertidal_zone",
        "sandy_seabed",
        "marine_mangrove",
        # Estuarine
        "estuary",
        "salt_marsh",
        "brackish_lagoon",
        "tidal_flat",
        # Freshwater
        "lake",
        "river",
        "stream",
        "pond",
        "freshwater_wetland",
        # Terrestrial
        "forest_boreal",
        "forest_temperate",
        "forest_tropical",
        "grassland",
        "savanna",
        "shrubland",
        "desert",
        "tundra",
        "terrestrial_wetland",
        "agricultural_land",
        "urban",
        # Catch-all for genuinely mixed sites
        "mixed_habitat",
    }
)

# The prefix used for environment-variable secret overrides, for example
# AUDTHEIA_SECRET_GBIF_PASSWORD overrides the "gbif_password" secret.
SECRET_ENV_PREFIX = "AUDTHEIA_SECRET_"


class ConfigError(ValueError):
    """Raised when the configuration is missing, malformed, or inconsistent.

    The message always names the offending key and what was expected, so the
    person editing the configuration can fix it without reading this code.
    """


# ---------------------------------------------------------------------------
# Small validation helpers. Each raises ConfigError with a clear path-like key
# so a problem points straight at the place in the file that caused it.
# ---------------------------------------------------------------------------


def _require(mapping: Any, key: str, where: str) -> Any:
    if not isinstance(mapping, dict):
        raise ConfigError(f"{where} must be an object")
    if key not in mapping:
        raise ConfigError(f"{where}.{key} is required")
    return mapping[key]


def _require_type(value: Any, types: tuple, where: str) -> Any:
    if not isinstance(value, types):
        names = " or ".join(t.__name__ for t in types)
        raise ConfigError(f"{where} must be {names}")
    return value


def _require_choice(value: Any, choices: tuple, where: str) -> Any:
    if value not in choices:
        allowed = ", ".join(str(c) for c in choices)
        raise ConfigError(f"{where} must be one of: {allowed} (got {value!r})")
    return value


def _require_positive_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} must be a number")
    if value <= 0:
        raise ConfigError(f"{where} must be greater than zero")
    return value


def _require_nonnegative_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where} must be an integer")
    if value < 0:
        raise ConfigError(f"{where} must be zero or greater")
    return value


# ===========================================================================
# Settings
# ===========================================================================


class Settings:
    """A validated, path-resolved view of one Audtheia configuration.

    Construct it through load_settings, which reads and checks the file. Once
    built, the accessors below give downstream code exactly what it needs (the
    resolved database path, connection options, sync options, the active
    station, the display time zone) without any component touching the raw
    file or guessing a default.
    """

    def __init__(self, raw: dict, *, settings_path: Path, repo_root: Path, secrets: dict) -> None:
        self.raw = raw
        self.settings_path = settings_path
        self.repo_root = repo_root
        self.secrets = secrets

    # -- paths -----------------------------------------------------------

    def resolve_path(self, value: str) -> Path:
        """Resolve a configured path to an absolute filesystem path.

        A relative path is taken from the repository root so the same file
        works on every operating system; an absolute path is used as given so
        a store can live on an external drive.
        """
        p = Path(value)
        if p.is_absolute():
            return p
        return (self.repo_root / p).resolve()

    def db_path(self) -> str:
        return str(self.resolve_path(self.raw["paths"]["db_path"]))

    def schema_path(self) -> str:
        return str(self.resolve_path(self.raw["paths"]["schema_path"]))

    def path(self, key: str) -> str:
        """Resolve any named entry in the paths block."""
        paths = self.raw["paths"]
        if key not in paths:
            raise ConfigError(f"paths.{key} is not defined")
        return str(self.resolve_path(paths[key]))

    # -- database and sync options --------------------------------------

    def database_kwargs(self) -> dict:
        """Keyword arguments for constructing the storage layer's Database."""
        db = self.raw["database"]
        return {"wal": db["wal"], "busy_timeout_ms": db["busy_timeout_ms"]}

    def sync_kwargs(self) -> dict:
        """Keyword arguments for an export and sync round."""
        emb = self.raw["embeddings"]
        return {
            "batch_size": self.raw["sync"]["batch_size"],
            "forward_embeddings": emb["forward_embeddings"],
            "max_embedding_bytes": emb["max_embedding_bytes"],
        }

    def max_embedding_bytes(self) -> int:
        return self.raw["embeddings"]["max_embedding_bytes"]

    # -- media encoding --------------------------------------------------

    def image_encoding(self) -> dict:
        """The stored-frame image format and quality.

        Returned as a fresh dictionary so a caller cannot mutate the loaded
        configuration by editing what it gets back.
        """
        return dict(self.raw["media"]["image"])

    def audio_encoding(self) -> dict:
        """The stored-clip audio format and sample width, as a fresh dictionary."""
        return dict(self.raw["media"]["audio"])

    def acoustic_tuning(self, station: dict) -> dict:
        """The station's acoustic onset threshold and silence-close gap.

        Returned as a fresh dictionary. A station that has not set these gets an
        empty dictionary, and the acoustic capture applies its own documented
        defaults, so a station file without the block still runs.
        """
        return dict(station.get("capture", {}).get("acoustic", {}))

    def capture_source(self, station: dict) -> dict:
        """The station's desktop capture source (video and optional audio).

        Returned as a fresh dictionary. A station with no desktop source gets an
        empty dictionary, which the desktop drivers read as no source configured,
        so a field-only station is simply not run as a desktop capture.
        """
        return dict(station.get("capture", {}).get("source", {}))

    def desktop_visual_model(self, station: dict) -> dict:
        """The station's desktop screening-model entry (path, version, citation).

        Returned as a fresh dictionary. A station with no desktop screening model
        gets an empty dictionary.
        """
        return dict(station.get("models", {}).get("visual_desktop", {}))

    # -- node and stations ----------------------------------------------

    @property
    def node_role(self) -> str:
        return self.raw["node"]["role"]

    def stations(self) -> list[dict]:
        return self.raw["stations"]

    def station(self, station_id: str) -> dict:
        for s in self.raw["stations"]:
            if s["station_id"] == station_id:
                return s
        raise ConfigError(f"no station with station_id {station_id!r}")

    def active_station(self) -> Optional[dict]:
        """The station this node runs as.

        On a field station this is the one station the node operates; on the
        desktop, which manages every station, there is no single active
        station and this returns nothing.
        """
        if self.node_role != "pi":
            return None
        return self.station(self.raw["node"]["active_station_id"])

    def channels(self, station_id: str) -> list[dict]:
        return self.station(station_id).get("channels", [])

    def analysis_location(self) -> str:
        return self.raw["analysis"]["per_observation_analysis_location"]

    def baseline_config(self) -> dict:
        """The longitudinal baseline settings, with documented defaults applied.

        Returned as a fresh dictionary so a caller cannot mutate the loaded
        configuration. A configuration that omits the block still gets a
        complete, usable set of values.
        """
        block = dict(self.raw.get("analysis", {}).get("baseline", {}))
        block.setdefault("period_granularity", DEFAULT_BASELINE_PERIOD_GRANULARITY)
        return block

    def salience_config(self) -> dict:
        """The authoritative-salience settings, with documented defaults applied.

        Returned as a fresh dictionary. The weights are prioritization priors,
        not fitted quantities; the calculation renormalizes over whichever
        ingredients are present, so zeroing a weight cleanly drops that
        ingredient. A configuration that omits any part gets the defaults for
        just that part.
        """
        raw_block = self.raw.get("analysis", {}).get("salience", {})
        weights = dict(DEFAULT_SALIENCE_WEIGHTS)
        weights.update(raw_block.get("weights", {}) or {})
        anomaly = {
            "min_effective_n": DEFAULT_SALIENCE_MIN_EFFECTIVE_N,
            "aggregator": DEFAULT_SALIENCE_AGGREGATOR,
        }
        anomaly.update(raw_block.get("anomaly", {}) or {})
        return {"weights": weights, "anomaly": anomaly}

    def dream_budget(self) -> dict:
        """The work budget for one longitudinal pass, as a fresh dictionary."""
        return dict(self.raw["schedules"]["dream_pass"]["budget"])

    def thresholds_config(self) -> dict:
        """The analysis thresholds, with documented defaults applied.

        Returned as a fresh dictionary. Each group (field_qc, verification,
        dream) is merged over its defaults, so a configuration that omits the
        block, or any value in it, gets the documented default for exactly that
        value and nothing behaves differently from before these thresholds were
        configurable.
        """
        raw_block = self.raw.get("analysis", {}).get("thresholds", {}) or {}
        out: dict = {}
        for group, defaults in DEFAULT_THRESHOLDS.items():
            merged = dict(defaults)
            merged.update(raw_block.get(group, {}) or {})
            out[group] = merged
        return out

    def privacy_config(self) -> dict:
        """The privacy settings, with documented defaults applied.

        Returned as a fresh dictionary. discard_human_detections defaults to
        true; human_class_names defaults to an empty list, so the discard is
        inert until a deployment names the class or classes its own detection
        model uses for people.
        """
        raw_block = self.raw.get("privacy", {}) or {}
        return {
            "discard_human_detections": bool(
                raw_block.get("discard_human_detections", DEFAULT_PRIVACY["discard_human_detections"])
            ),
            "human_class_names": list(raw_block.get("human_class_names", []) or []),
        }

    # -- time base -------------------------------------------------------

    def resolve_timezone(self) -> tzinfo:
        """The display time zone. Stored data is always in UTC.

        The default, "auto", follows the host computer's own time zone, which
        needs no extra data and works everywhere. A named zone (for example
        "America/Puerto_Rico") is honored when the host can resolve it; on a
        system without the time-zone database installed, that raises a clear
        error pointing at how to fix it, while "auto" keeps working.
        """
        name = self.raw["localization"]["local_timezone"]
        if name == "auto":
            local = datetime.now(timezone.utc).astimezone().tzinfo
            if local is None:
                return timezone.utc
            return local
        try:
            from zoneinfo import ZoneInfo

            return ZoneInfo(name)
        except Exception as exc:  # noqa: BLE001 - re-raised with guidance
            raise ConfigError(
                f"localization.local_timezone {name!r} could not be resolved "
                f"({exc}). Use \"auto\" to follow the host time zone, or install "
                f"the time-zone database for a named zone."
            ) from None


# ===========================================================================
# Loading and validation
# ===========================================================================


def _repo_root_for(settings_path: Path) -> Path:
    """The repository root, inferred from the settings file location.

    The configuration lives at config/settings.json, so the repository root is
    the parent of the config directory. This keeps every relative path in the
    file anchored to a single, predictable place.
    """
    return settings_path.resolve().parent.parent


def _load_secrets(secrets_path: Path) -> dict:
    """Load the optional secrets file, then apply environment overrides.

    The secrets file is never committed and may be absent, in which case an
    empty set of secrets is returned. Any value can be overridden by an
    environment variable, which is how a deployment supplies credentials
    without writing them to disk at all.
    """
    secrets: dict = {}
    if secrets_path.exists():
        with secrets_path.open(encoding="utf-8") as fh:
            loaded = json.load(fh)
        if not isinstance(loaded, dict):
            raise ConfigError("secrets file must contain a JSON object")
        secrets.update(loaded)
    for env_key, env_value in os.environ.items():
        if env_key.startswith(SECRET_ENV_PREFIX):
            secret_key = env_key[len(SECRET_ENV_PREFIX):].lower()
            secrets[secret_key] = env_value
    return secrets


def load_settings(
    settings_path: Optional[str | Path] = None,
    *,
    secrets_path: Optional[str | Path] = None,
) -> Settings:
    """Read, validate, and resolve a configuration file.

    With no argument, the configuration is read from config/settings.json
    relative to this package's repository. Pass an explicit path to load a
    different file (for example a field station's own copy). The returned
    Settings object is fully validated; if anything is wrong, a ConfigError
    explains exactly what and where.
    """
    if settings_path is None:
        # audtheia/config.py -> repository root -> config/settings.json
        settings_path = Path(__file__).resolve().parent.parent / "config" / "settings.json"
    settings_path = Path(settings_path)

    if not settings_path.exists():
        raise ConfigError(f"settings file not found at {settings_path}")

    with settings_path.open(encoding="utf-8") as fh:
        try:
            raw = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"settings file is not valid JSON: {exc}") from None

    if not isinstance(raw, dict):
        raise ConfigError("settings file must contain a JSON object")

    repo_root = _repo_root_for(settings_path)

    _validate(raw)

    if secrets_path is None:
        secrets_ref = raw.get("secrets", {}).get("path", "config/secrets.json")
        secrets_path = (
            Path(secrets_ref)
            if Path(secrets_ref).is_absolute()
            else repo_root / secrets_ref
        )
    secrets = _load_secrets(Path(secrets_path))

    return Settings(raw, settings_path=settings_path, repo_root=repo_root, secrets=secrets)


def _validate(raw: dict) -> None:
    """Check the whole configuration, raising ConfigError on the first problem."""

    _require(raw, "settings_schema_version", "settings")
    _require_type(raw["settings_schema_version"], (str,), "settings_schema_version")

    # -- node ------------------------------------------------------------
    node = _require(raw, "node", "settings")
    role = _require(node, "role", "node")
    _require_choice(role, NODE_ROLES, "node.role")

    # -- paths -----------------------------------------------------------
    paths = _require(raw, "paths", "settings")
    for key in (
        "db_path",
        "schema_path",
        "data_dir",
        "detections_visual_dir",
        "detections_audio_dir",
        "gps_dir",
        "reports_dir",
        "models_dir",
        "gbif_backbone_path",
    ):
        _require_type(_require(paths, key, "paths"), (str,), f"paths.{key}")

    # -- database --------------------------------------------------------
    db = _require(raw, "database", "settings")
    _require_type(_require(db, "wal", "database"), (bool,), "database.wal")
    _require_positive_number(_require(db, "busy_timeout_ms", "database"), "database.busy_timeout_ms")

    # -- sync ------------------------------------------------------------
    sync = _require(raw, "sync", "settings")
    _require_positive_number(_require(sync, "batch_size", "sync"), "sync.batch_size")

    # -- embeddings ------------------------------------------------------
    emb = _require(raw, "embeddings", "settings")
    _require_type(_require(emb, "forward_embeddings", "embeddings"), (bool,), "embeddings.forward_embeddings")
    _require_positive_number(_require(emb, "max_embedding_bytes", "embeddings"), "embeddings.max_embedding_bytes")

    # -- media -----------------------------------------------------------
    # Stored-frame and stored-clip encoding. Kept here so a deployment can tune
    # how detections are saved without editing code, while a validated shape
    # keeps a malformed value from reaching the capture stage.
    media = _require(raw, "media", "settings")
    image = _require(media, "image", "media")
    _require_type(_require(image, "format", "media.image"), (str,), "media.image.format")
    quality = _require(image, "quality", "media.image")
    if isinstance(quality, bool) or not isinstance(quality, int) or not (1 <= quality <= 100):
        raise ConfigError("media.image.quality must be an integer between 1 and 100")
    audio = _require(media, "audio", "media")
    _require_type(_require(audio, "format", "media.audio"), (str,), "media.audio.format")
    sample_width = _require(audio, "sample_width_bytes", "media.audio")
    if isinstance(sample_width, bool) or not isinstance(sample_width, int) or sample_width < 1:
        raise ConfigError("media.audio.sample_width_bytes must be an integer of one or more")

    # -- buffer ----------------------------------------------------------
    buf = _require(raw, "buffer", "settings")
    high = _require(buf, "high_water_pct", "buffer")
    ceiling = _require(buf, "hard_ceiling_pct", "buffer")
    _require_positive_number(high, "buffer.high_water_pct")
    _require_positive_number(ceiling, "buffer.hard_ceiling_pct")
    if not (0 < high < ceiling <= 100):
        raise ConfigError("buffer thresholds must satisfy 0 < high_water_pct < hard_ceiling_pct <= 100")
    _require_type(_require(buf, "auto_sync_when_reachable", "buffer"), (bool,), "buffer.auto_sync_when_reachable")
    _require_type(_require(buf, "pause_capture_at_ceiling", "buffer"), (bool,), "buffer.pause_capture_at_ceiling")

    # -- telemetry -------------------------------------------------------
    tel = _require(raw, "telemetry", "settings")
    _require_positive_number(_require(tel, "heartbeat_seconds", "telemetry"), "telemetry.heartbeat_seconds")
    meter = _require(tel, "energy_meter", "telemetry")
    _require_type(_require(meter, "enabled", "telemetry.energy_meter"), (bool,), "telemetry.energy_meter.enabled")

    # -- analysis (resolves the default analysis location) ---------------
    analysis = _require(raw, "analysis", "settings")
    _require_choice(
        _require(analysis, "per_observation_analysis_location", "analysis"),
        ANALYSIS_LOCATIONS,
        "analysis.per_observation_analysis_location",
    )

    # The baseline and salience blocks are optional: a configuration that omits
    # them runs on the documented defaults through the accessors. When present,
    # their shape is checked strictly so a typo is caught here, not mid-pass.
    baseline = analysis.get("baseline")
    if baseline is not None:
        _require_type(baseline, (dict,), "analysis.baseline")
        gran = baseline.get("period_granularity")
        if gran is not None:
            _require_choice(gran, BASELINE_PERIOD_GRANULARITIES, "analysis.baseline.period_granularity")

    salience = analysis.get("salience")
    if salience is not None:
        _require_type(salience, (dict,), "analysis.salience")
        weights = salience.get("weights")
        if weights is not None:
            _require_type(weights, (dict,), "analysis.salience.weights")
            total = 0.0
            for key in ("confidence", "anomaly", "rarity"):
                if key in weights:
                    w = weights[key]
                    if isinstance(w, bool) or not isinstance(w, (int, float)) or w < 0:
                        raise ConfigError(f"analysis.salience.weights.{key} must be a number of zero or more")
                    total += float(w)
            if total <= 0:
                raise ConfigError("analysis.salience.weights must not all be zero")
        anomaly = salience.get("anomaly")
        if anomaly is not None:
            _require_type(anomaly, (dict,), "analysis.salience.anomaly")
            if "min_effective_n" in anomaly:
                _require_nonnegative_int(anomaly["min_effective_n"], "analysis.salience.anomaly.min_effective_n")
            if "aggregator" in anomaly:
                _require_choice(anomaly["aggregator"], SALIENCE_AGGREGATORS, "analysis.salience.anomaly.aggregator")

    # The thresholds block is optional: a configuration that omits it runs on
    # the documented defaults through the accessor. When present, its shape is
    # checked strictly so a typo is caught here rather than mid-pass.
    thresholds = analysis.get("thresholds")
    if thresholds is not None:
        _require_type(thresholds, (dict,), "analysis.thresholds")
        fq = thresholds.get("field_qc")
        if fq is not None:
            _require_type(fq, (dict,), "analysis.thresholds.field_qc")
            if "pass_confidence" in fq:
                pc = fq["pass_confidence"]
                if isinstance(pc, bool) or not isinstance(pc, (int, float)) or not (0 <= pc <= 1):
                    raise ConfigError("analysis.thresholds.field_qc.pass_confidence must be a number between 0 and 1")
        ver = thresholds.get("verification")
        if ver is not None:
            _require_type(ver, (dict,), "analysis.thresholds.verification")
            if "clear_confidence" in ver:
                cc = ver["clear_confidence"]
                if isinstance(cc, bool) or not isinstance(cc, (int, float)) or not (0 <= cc <= 1):
                    raise ConfigError("analysis.thresholds.verification.clear_confidence must be a number between 0 and 1")
            if "max_frames_scored" in ver:
                _require_positive_number(ver["max_frames_scored"], "analysis.thresholds.verification.max_frames_scored")
        dre = thresholds.get("dream")
        if dre is not None:
            _require_type(dre, (dict,), "analysis.thresholds.dream")
            for k in ("min_periods_for_trend", "min_events_for_correlation", "min_events_for_co_occurrence"):
                if k in dre:
                    _require_positive_number(dre[k], f"analysis.thresholds.dream.{k}")
            if "min_abs_effect" in dre:
                mae = dre["min_abs_effect"]
                if isinstance(mae, bool) or not isinstance(mae, (int, float)) or mae < 0:
                    raise ConfigError("analysis.thresholds.dream.min_abs_effect must be a number of zero or more")
            if "max_p_value" in dre:
                mpv = dre["max_p_value"]
                if isinstance(mpv, bool) or not isinstance(mpv, (int, float)) or not (0 <= mpv <= 1):
                    raise ConfigError("analysis.thresholds.dream.max_p_value must be a number between 0 and 1")

    # -- schedules -------------------------------------------------------
    schedules = _require(raw, "schedules", "settings")
    reports = _require(schedules, "reports", "schedules")
    _require_choice(_require(reports, "schedule", "schedules.reports"), REPORT_SCHEDULES, "schedules.reports.schedule")
    formats = _require(reports, "formats", "schedules.reports")
    _require_type(formats, (list,), "schedules.reports.formats")
    for fmt in formats:
        _require_choice(fmt, REPORT_FORMATS, "schedules.reports.formats[]")

    dream = _require(schedules, "dream_pass", "schedules")
    _require_choice(_require(dream, "schedule", "schedules.dream_pass"), DREAM_SCHEDULES, "schedules.dream_pass.schedule")
    budget = _require(dream, "budget", "schedules.dream_pass")
    for key in (
        "epoch_batch_size",
        "max_cycles_per_pass",
        "substrate_exemplar_cap",
        "substrate_candidate_pattern_cap",
    ):
        _require_nonnegative_int(_require(budget, key, "schedules.dream_pass.budget"), f"schedules.dream_pass.budget.{key}")

    # -- server ----------------------------------------------------------
    server = _require(raw, "server", "settings")
    _require_type(_require(server, "host", "server"), (str,), "server.host")
    port = _require(server, "port", "server")
    if isinstance(port, bool) or not isinstance(port, int) or not (1 <= port <= 65535):
        raise ConfigError("server.port must be an integer between 1 and 65535")

    # -- localization ----------------------------------------------------
    loc = _require(raw, "localization", "settings")
    _require_type(_require(loc, "local_timezone", "localization"), (str,), "localization.local_timezone")

    # -- privacy (optional) ----------------------------------------------
    privacy = raw.get("privacy")
    if privacy is not None:
        _require_type(privacy, (dict,), "privacy")
        if "discard_human_detections" in privacy:
            _require_type(privacy["discard_human_detections"], (bool,), "privacy.discard_human_detections")
        human_names = privacy.get("human_class_names")
        if human_names is not None:
            _require_type(human_names, (list,), "privacy.human_class_names")
            for i, name in enumerate(human_names):
                _require_type(name, (str,), f"privacy.human_class_names[{i}]")

    # -- desktop models --------------------------------------------------
    desktop_models = _require(raw, "desktop_models", "settings")
    for key in ("visual_rfdetr", "llm"):
        entry = _require(desktop_models, key, "desktop_models")
        _require_type(entry, (dict,), f"desktop_models.{key}")
        _require(entry, "path", f"desktop_models.{key}")

    # -- stations --------------------------------------------------------
    stations = _require(raw, "stations", "settings")
    _require_type(stations, (list,), "stations")
    if not stations:
        raise ConfigError("at least one station must be defined in stations")

    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for index, station in enumerate(stations):
        _validate_station(station, index, seen_ids, seen_names)

    # A field station must point at exactly one station it operates, and that
    # station must exist; this is what setup pushes down to each Pi.
    if role == "pi":
        active = node.get("active_station_id")
        if not active:
            raise ConfigError("node.active_station_id is required when node.role is 'pi'")
        if active not in seen_ids:
            raise ConfigError(f"node.active_station_id {active!r} does not match any station")


def _validate_station(station: Any, index: int, seen_ids: set, seen_names: set) -> None:
    where = f"stations[{index}]"
    _require_type(station, (dict,), where)

    station_id = _require(station, "station_id", where)
    _require_type(station_id, (str,), f"{where}.station_id")
    if station_id in seen_ids:
        raise ConfigError(f"duplicate station_id {station_id!r}")
    seen_ids.add(station_id)

    station_name = _require(station, "station_name", where)
    _require_type(station_name, (str,), f"{where}.station_name")
    if station_name in seen_names:
        raise ConfigError(f"duplicate station_name {station_name!r}")
    seen_names.add(station_name)

    _require_choice(
        _require(station, "environment_type", where),
        ENVIRONMENT_TYPES,
        f"{where}.environment_type",
    )

    habitat = station.get("habitat")
    if habitat is not None and habitat not in ALLOWED_HABITATS:
        raise ConfigError(
            f"{where}.habitat {habitat!r} is not in the allowed habitat list; "
            f"omit it or choose a listed value"
        )

    target_species = station.get("target_species", [])
    _require_type(target_species, (list,), f"{where}.target_species")

    sensors = _require(station, "sensors", where)
    _require_type(sensors, (dict,), f"{where}.sensors")

    channels = _require(station, "channels", where)
    _require_type(channels, (list,), f"{where}.channels")
    seen_channel_ids: set[str] = set()
    for c_index, channel in enumerate(channels):
        _validate_channel(channel, f"{where}.channels[{c_index}]", seen_channel_ids)

    models = _require(station, "models", where)
    _require_type(models, (dict,), f"{where}.models")
    visual_pi = _require(models, "visual_pi", f"{where}.models")
    _require(visual_pi, "path", f"{where}.models.visual_pi")
    acoustic = _require(models, "acoustic", f"{where}.models")
    _require(acoustic, "active", f"{where}.models.acoustic")

    # Optional desktop screening model. A desktop node that runs capture without
    # field hardware detects with this model through ONNX Runtime. When absent, a
    # station has no desktop screening model, so a field-only station still loads.
    visual_desktop = models.get("visual_desktop")
    if visual_desktop is not None:
        _require(visual_desktop, "path", f"{where}.models.visual_desktop")

    capture = _require(station, "capture", where)
    _validate_capture(capture, f"{where}.capture")


def _validate_channel(channel: Any, where: str, seen_channel_ids: set) -> None:
    _require_type(channel, (dict,), where)
    channel_id = _require(channel, "id", where)
    _require_type(channel_id, (str,), f"{where}.id")
    if channel_id in seen_channel_ids:
        raise ConfigError(f"duplicate channel id {channel_id!r} in {where}")
    seen_channel_ids.add(channel_id)

    _require_type(_require(channel, "unit", where), (str,), f"{where}.unit")
    _require_type(_require(channel, "marine", where), (bool,), f"{where}.marine")
    _require_type(_require(channel, "enabled", where), (bool,), f"{where}.enabled")

    # Optional quality-control bounds. Shape is checked when present so the
    # environment tier can rely on it; values are not required here.
    qc = channel.get("qc")
    if qc is not None:
        _require_type(qc, (dict,), f"{where}.qc")
        for range_key in ("gross_range", "sensor_range"):
            rng = qc.get(range_key)
            if rng is not None:
                _require_type(rng, (dict,), f"{where}.qc.{range_key}")
                for bound in ("min", "max"):
                    if bound in rng and not isinstance(rng[bound], (int, float)):
                        raise ConfigError(f"{where}.qc.{range_key}.{bound} must be a number")


def _validate_capture(capture: Any, where: str) -> None:
    _require_type(capture, (dict,), where)
    _require_positive_number(_require(capture, "fps", where), f"{where}.fps")

    resolution = _require(capture, "resolution", where)
    _require_positive_number(_require(resolution, "width", f"{where}.resolution"), f"{where}.resolution.width")
    _require_positive_number(_require(resolution, "height", f"{where}.resolution"), f"{where}.resolution.height")

    bytetrack = _require(capture, "bytetrack", where)
    for key in ("track_activation_threshold", "minimum_matching_threshold", "track_close_frames", "frame_rate"):
        _require_positive_number(_require(bytetrack, key, f"{where}.bytetrack"), f"{where}.bytetrack.{key}")

    _require_choice(
        _require(capture, "representative_frame_rule", where),
        REPRESENTATIVE_FRAME_RULES,
        f"{where}.representative_frame_rule",
    )
    _require_positive_number(
        _require(capture, "max_event_duration_seconds", where), f"{where}.max_event_duration_seconds"
    )

    audio = _require(capture, "audio", where)
    for key in ("pre_roll_seconds", "post_roll_seconds", "max_clip_seconds"):
        value = _require(audio, key, f"{where}.audio")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ConfigError(f"{where}.audio.{key} must be a number of zero or more seconds")

    soundscape = _require(capture, "soundscape", where)
    _require_type(_require(soundscape, "enabled", f"{where}.soundscape"), (bool,), f"{where}.soundscape.enabled")

    # Optional per-station acoustic sensitivity. When present, its shape is
    # checked so the acoustic capture can rely on it; when absent, the acoustic
    # capture applies its own documented defaults, so an older station file
    # without this block still loads.
    acoustic = capture.get("acoustic")
    if acoustic is not None:
        _require_type(acoustic, (dict,), f"{where}.acoustic")
        onset = _require(acoustic, "onset_threshold", f"{where}.acoustic")
        if isinstance(onset, bool) or not isinstance(onset, (int, float)) or not (0 < onset <= 1):
            raise ConfigError(
                f"{where}.acoustic.onset_threshold must be a number greater than 0 and at most 1"
            )
        _require_positive_number(
            _require(acoustic, "silence_close_seconds", f"{where}.acoustic"),
            f"{where}.acoustic.silence_close_seconds",
        )

    # Optional desktop capture source. When present, a desktop node can run this
    # station's capture against an ordinary webcam, network stream, or video file
    # instead of field hardware; its shape is checked so the desktop drivers can
    # rely on it. When absent, a station has no desktop source and is simply not
    # run on the desktop, so a field-only station file still loads unchanged.
    source = capture.get("source")
    if source is not None:
        _require_type(source, (dict,), f"{where}.source")
        _require_type(_require(source, "video", f"{where}.source"), (str,), f"{where}.source.video")
        audio = source.get("audio")
        if audio is not None:
            _require_type(audio, (str,), f"{where}.source.audio")
