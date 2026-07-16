"""Audtheia V2 desktop web backend.

Path: audtheia/app/server.py

The desktop hub serves a small local web application over the loopback
interface. This module builds that backend: a FastAPI application whose
endpoints read the authoritative record and expose the desktop controls the
interface needs. It talks to the database only through the data-access layer,
so the storage contract stays in one place, and it reaches the report generator
only through that module's public entry point.

Design commitments:

  - Local only. The server binds to the host and port in the configuration,
    which default to the loopback address. There is no cloud dependency, no
    content delivery network, and no outbound call at request time.

  - Provenance survives the wire. Every value the database stores with a source
    and a status is returned with that source and status intact, so the
    interface can show measured and inferred data as distinctly as the record
    keeps them. Candidate patterns from the longitudinal pass are returned under
    an explicit hypothesis framing, never as findings.

  - Read first. The endpoints in this backend serve the record and expose the
    two desktop controls that are not reads: pausing or resuming a longitudinal
    pass, and asking for a report to be produced. Editing configuration is a
    read-only view here; a guarded write path is a separate, dedicated addition.

  - Nothing is scheduled here. A report is produced only when asked for, as a
    background task so the request returns at once. The scheduler that runs
    reports and the longitudinal pass on a cadence lives elsewhere.

The web framework is imported inside the application factory, not at module
import, so this file imports cleanly with the framework absent; only building or
running the application requires it. That keeps the module importable for tests
and tooling that do not need a live server.
"""

import copy
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# The single URL prefix every API route sits under, kept in one place so it is
# never spelled out ad hoc.
API_PREFIX = "/api"

# Keys whose values are blanked before configuration is returned, in case a
# deployment ever placed a secret inline rather than in the separate secrets
# file. The committed configuration holds no secrets, so this is defense in
# depth, not the primary boundary.
_REDACT_KEYS = frozenset({"password", "secret", "token", "api_key", "apikey", "credential"})

# The framing every candidate pattern is returned under, matching how the record
# stores it: a hypothesis, never an established finding.
_HYPOTHESIS_FRAMING = "candidate_hypothesis"

# A skill's type sets which tier evaluates it, and it is the one field the
# storage layer constrains, so the backend validates it against the same two
# values the schema accepts before a write is attempted.
SKILL_TIERS = ("deterministic_flag", "interpretive")

# Upper bounds on the length of each authored skill field. A skill is a short,
# human-written rule, so these are generous limits that stop an accidental paste
# of a whole document from being stored, not a policy on wording.
_SKILL_TITLE_MAX = 200
_SKILL_TEXT_MAX = 4000


class BackendError(RuntimeError):
    """A backend operation failed for a reason the operator should see."""


class BackendDependencyError(BackendError):
    """The web framework needed to build or run the app is not installed."""


class SettingsUpdateError(BackendError):
    """A requested configuration change is not allowed or not valid.

    Raised while a settings edit is being checked, before anything is written,
    so a rejected change leaves the saved configuration untouched. The route
    turns this into a clear client error rather than a server fault.
    """


# ===========================================================================
# Helpers that shape database rows into responses without losing provenance
# ===========================================================================


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _redact(value: Any) -> Any:
    """Return a copy of a configuration value with any secret-like field blanked."""
    if isinstance(value, dict):
        out = {}
        for key, inner in value.items():
            if isinstance(key, str) and key.lower() in _REDACT_KEYS and inner not in (None, ""):
                out[key] = "***redacted***"
            else:
                out[key] = _redact(inner)
        return out
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _frame_pattern(pattern: dict, supporting_ids: Optional[list] = None) -> dict:
    """Return a pattern row with the explicit hypothesis framing attached.

    The stored row already carries data_source 'dream' and the full statistic
    line; this adds a plain framing field and, when asked, the identifiers of
    the events the candidate rests on, so a consumer cannot present it as more
    than a hypothesis.
    """
    out = dict(pattern)
    out["framing"] = _HYPOTHESIS_FRAMING
    if supporting_ids is not None:
        out["supporting_observation_ids"] = supporting_ids
    return out


def _has_audio(observation: dict) -> bool:
    return bool(observation.get("audio_clip_path")) or observation.get("trigger_source") == "audio"


def _has_gps(observation: dict) -> bool:
    return observation.get("gps_latitude") is not None or observation.get("gps_longitude") is not None


def _taxon_key(detection: dict) -> Optional[str]:
    return detection.get("gbif_usage_key") or detection.get("common_name") or detection.get("scientific_name")


def _compute_analytics(db, *, station_id, since, until) -> dict:
    """Derive biodiversity summaries from the records in a window.

    These are computations over stored detections, not new measurements, and are
    labeled as derived. Detection counts are raw; an effort-normalized rate is
    not asserted here, matching how the record leaves rigorous rarity to a
    downstream measured statistic.
    """
    observations = db.list_observations(station_id=station_id, since=since, until=until)
    total = len(observations)
    by_trigger: dict = {}
    by_qc: dict = {}
    by_modality = {"vision": 0, "audio": 0}
    taxon_events: dict = {}
    verified = 0

    for obs in observations:
        by_trigger[obs.get("trigger_source") or "unknown"] = by_trigger.get(obs.get("trigger_source") or "unknown", 0) + 1
        by_qc[obs.get("qc_state") or "unknown"] = by_qc.get(obs.get("qc_state") or "unknown", 0) + 1
        v = db.get_observation_verification(obs["id"])
        if v and v.get("verified"):
            verified += 1
        seen: set = set()
        for det in db.list_child_detections(obs["id"]):
            by_modality[det.get("modality", "vision")] = by_modality.get(det.get("modality", "vision"), 0) + 1
            key = _taxon_key(det)
            if key and key not in seen:
                seen.add(key)
                taxon_events[key] = taxon_events.get(key, 0) + 1

    return {
        "provenance": "derived",
        "note": "computed from the records in this window; not a new measurement",
        "window": {"station_id": station_id, "since": since, "until": until},
        "total_events": total,
        "species_richness": len(taxon_events),
        "verified_count": verified,
        "verified_fraction": (verified / total) if total else 0.0,
        "events_by_trigger_source": by_trigger,
        "events_by_qc_state": by_qc,
        "detections_by_modality": by_modality,
        "taxon_event_counts": taxon_events,
    }


def _llm_directory(settings) -> Path:
    """The folder that holds the desktop language models.

    The configured desktop model path may name the folder itself (the common
    case) or a single model file; either way the folder is where installed
    models are listed and where a person drops a new one.
    """
    configured = settings.raw.get("desktop_models", {}).get("llm", {}).get("path")
    if not configured:
        return Path(settings.repo_root) / "models" / "llm"
    p = Path(configured)
    if not p.is_absolute():
        p = Path(settings.repo_root) / p
    return p if p.is_dir() else p.parent


def _list_gguf_models(settings) -> list:
    """Every GGUF model file installed in the language-model folder, by name."""
    directory = _llm_directory(settings)
    if not directory.is_dir():
        return []
    out = []
    for f in sorted(directory.glob("*.gguf")):
        try:
            size = f.stat().st_size
        except OSError:
            size = None
        out.append({"name": f.name, "size_bytes": size})
    return out


def _active_llm_name(settings) -> Optional[str]:
    """The model the desktop would load now.

    A configured path that names a file selects that file. A path that names the
    folder falls back to the first model by name, which is the same rule the
    model loader applies, so what the interface shows as active is what actually
    runs. No model present means nothing is active.
    """
    configured = settings.raw.get("desktop_models", {}).get("llm", {}).get("path")
    if configured:
        p = Path(configured)
        if not p.is_absolute():
            p = Path(settings.repo_root) / p
        if p.is_file():
            return p.name
    models = _list_gguf_models(settings)
    return models[0]["name"] if models else None


def _llm_runtime_available() -> bool:
    """Whether the language-model runtime is importable, without importing it.

    Checked through the import system's spec lookup so the heavy runtime is not
    loaded just to report that it is present.
    """
    import importlib.util

    return importlib.util.find_spec("llama_cpp") is not None


def _persist_settings(settings) -> None:
    """Write the in-memory configuration back to its file, validated and atomically.

    The configuration is validated before it is written, so an invalid change is
    refused rather than saved, and the file is replaced in a single step, so a
    reader never sees a half-written file. Secrets live in their own file and are
    never part of what is written here.
    """
    from audtheia.config import _validate, ConfigError

    try:
        _validate(settings.raw)
    except ConfigError as exc:
        raise BackendError(f"refusing to save an invalid configuration: {exc}") from exc

    target = Path(settings.settings_path)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".settings-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(settings.raw, fh, indent=2)
            fh.write("\n")
        os.replace(tmp_name, str(target))
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# The credential keys a person may set from the interface. These are the
# species-data credentials only; the hotspot and SSH passwords belong to station
# provisioning and are never editable through this path.
SPECIES_SECRET_KEYS = ("iucn_api_key", "gbif_username", "gbif_password")


def _secrets_file_path(settings) -> Path:
    """The resolved path to the local, uncommitted secrets file."""
    ref = settings.raw.get("secrets", {}).get("path", "config/secrets.json")
    path = Path(ref)
    if not path.is_absolute():
        path = Path(settings.repo_root) / path
    return path


def _read_secrets_file(path: Path) -> dict:
    """Read the secrets file if present, returning an empty map otherwise."""
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a malformed file is treated as empty here
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write_secrets_file(path: Path, secrets: dict) -> None:
    """Write the secrets file atomically, creating its directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".secrets-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(secrets, fh, indent=2)
            fh.write("\n")
        os.replace(tmp_name, str(path))
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _species_secret_status(settings) -> dict:
    """Which species-data credentials are set, never their values."""
    secrets = getattr(settings, "secrets", {}) or {}
    return {key: bool(str(secrets.get(key) or "").strip()) for key in SPECIES_SECRET_KEYS}


# In-progress field-station provisioning runs, keyed by station id. Each entry
# holds the running process and the log file its output streams to, so the
# interface can poll a connection without blocking the request that started it.
_PROVISION_JOBS: dict = {}


# Running desktop capture sessions, keyed by station id. Each value is a
# LiveCapture handle whose stop brings down the monitor and scheduler threads it
# started, so the interface can start and stop capture without a terminal.
_CAPTURE_JOBS: dict = {}


def _new_station_dict(station_id: str, name: str, environment: str, habitat: Optional[str]) -> dict:
    """A complete, valid station configuration with sensible defaults.

    The identifier is generated, the device sensors default to on, the channel
    list starts empty (environmental sensors are added from the Sensors settings),
    and the capture and model blocks carry the same defaults as the reference
    stations, so a new station validates and runs without further editing.
    """
    station = {
        "station_id": station_id,
        "station_name": name,
        "environment_type": environment,
        "target_species": [],
        "sensors": {"camera": {"enabled": True}, "audio": {"enabled": True}, "gps": {"enabled": True}},
        "channels": [],
        "models": {
            "visual_pi": {"path": "models/visual/pi/yolo11.hef", "version": None, "citation": None},
            "acoustic": {
                "active": "birdnet",
                "options": {
                    "birdnet": {"path": "models/acoustic/birdnet/BirdNET_GLOBAL_6K.tflite", "version": None, "citation": "Kahl et al., BirdNET, Ecological Informatics, 2021"},
                    "marine": {"path": None, "version": None, "citation": None},
                    "custom": {"path": None, "version": None, "citation": None},
                },
            },
        },
        "capture": {
            "fps": 10,
            "resolution": {"width": 1280, "height": 720},
            "bytetrack": {"track_activation_threshold": 0.25, "minimum_matching_threshold": 0.8, "track_close_frames": 20, "frame_rate": 10},
            "representative_frame_rule": "highest_confidence",
            "max_event_duration_seconds": 300,
            "audio": {"pre_roll_seconds": 3.0, "post_roll_seconds": 3.0, "max_clip_seconds": 30.0},
            "acoustic": {"onset_threshold": 0.5, "silence_close_seconds": 3.0},
            "soundscape": {"enabled": False, "metrics": [], "cadence_seconds": 60},
        },
    }
    if habitat:
        station["habitat"] = habitat
    return station


# ===========================================================================
# The guarded settings write path
#
# Editing configuration from the interface is deliberately narrow. Only the
# fields listed in the allowlist below can be changed, so a system-owned value
# such as a station's identifier or a derived path is never reachable from a
# request; a field that is not listed is simply refused. Each listed field
# carries a small validator, and after every change in a batch is applied to a
# working copy the whole configuration is validated once more and written
# atomically through the same path the rest of the backend uses. A rejected
# change leaves the saved file exactly as it was.
# ===========================================================================


def _v_bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise SettingsUpdateError(f"{where} must be true or false")
    return value


def _v_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SettingsUpdateError(f"{where} must be a number")
    return value


def _v_positive_number(value: Any, where: str) -> float:
    number = _v_number(value, where)
    if number <= 0:
        raise SettingsUpdateError(f"{where} must be greater than zero")
    return number


def _v_nonnegative_number(value: Any, where: str) -> float:
    number = _v_number(value, where)
    if number < 0:
        raise SettingsUpdateError(f"{where} must be zero or greater")
    return number


def _v_unit_interval(value: Any, where: str) -> float:
    number = _v_number(value, where)
    if not (0 <= number <= 1):
        raise SettingsUpdateError(f"{where} must be between 0 and 1")
    return number


def _v_positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SettingsUpdateError(f"{where} must be a whole number of one or more")
    return value


def _v_nonnegative_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SettingsUpdateError(f"{where} must be a whole number of zero or more")
    return value


def _v_int_range(low: int, high: int):
    def _inner(value: Any, where: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not (low <= value <= high):
            raise SettingsUpdateError(f"{where} must be a whole number between {low} and {high}")
        return value
    return _inner


def _v_nonempty_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SettingsUpdateError(f"{where} must be a non-empty text value")
    return value.strip()


def _v_str_or_null(value: Any, where: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SettingsUpdateError(f"{where} must be text, or empty to clear it")
    trimmed = value.strip()
    return trimmed or None


def _v_choice(choices):
    def _inner(value: Any, where: str):
        if value not in choices:
            raise SettingsUpdateError(f"{where} must be one of: {', '.join(choices)}")
        return value
    return _inner


def _v_report_formats(value: Any, where: str) -> list:
    from audtheia.config import REPORT_FORMATS

    if not isinstance(value, list) or not value:
        raise SettingsUpdateError(f"{where} must list at least one format")
    out: list = []
    for fmt in value:
        if fmt not in REPORT_FORMATS:
            raise SettingsUpdateError(f"{where} entries must be one of: {', '.join(REPORT_FORMATS)}")
        if fmt not in out:
            out.append(fmt)
    return out


def _v_timezone(value: Any, where: str) -> str:
    """Accept only a time zone the host can actually resolve, or 'auto'.

    This mirrors what the runtime does when it localizes a timestamp, so a value
    that saves here is a value the interface can display with, rather than one
    that validates as text but fails the first time it is used.
    """
    if not isinstance(value, str) or not value.strip():
        raise SettingsUpdateError(f"{where} must be a time zone name or 'auto'")
    name = value.strip()
    if name == "auto":
        return name
    try:
        from zoneinfo import ZoneInfo

        ZoneInfo(name)
    except Exception as exc:  # noqa: BLE001 - reported as a clear client error
        raise SettingsUpdateError(
            f"{where} {name!r} is not a resolvable time zone; use 'auto' or a name "
            f"like 'America/Puerto_Rico'"
        ) from None
    return name


def _editable_field_specs() -> dict:
    """The allowlist of user-editable fields, grouped by scope.

    Each entry maps a stable field key the interface sends to the location of the
    value and the validator that guards it. Anything absent from this map cannot
    be changed through the interface.
    """
    from audtheia.config import (
        REPORT_SCHEDULES,
        DREAM_SCHEDULES,
        ANALYSIS_LOCATIONS,
        BASELINE_PERIOD_GRANULARITIES,
    )

    return {
        "global": {
            "reports_schedule": {"path": ["schedules", "reports", "schedule"], "validate": _v_choice(REPORT_SCHEDULES)},
            "reports_formats": {"path": ["schedules", "reports", "formats"], "validate": _v_report_formats},
            "dream_schedule": {"path": ["schedules", "dream_pass", "schedule"], "validate": _v_choice(DREAM_SCHEDULES)},
            "local_timezone": {"path": ["localization", "local_timezone"], "validate": _v_timezone},
            "visual_rfdetr_path": {"path": ["desktop_models", "visual_rfdetr", "path"], "validate": _v_nonempty_str, "is_path": True},
            "visual_rfdetr_version": {"path": ["desktop_models", "visual_rfdetr", "version"], "validate": _v_str_or_null},
            "visual_rfdetr_citation": {"path": ["desktop_models", "visual_rfdetr", "citation"], "validate": _v_str_or_null},
            "analysis_location": {"path": ["analysis", "per_observation_analysis_location"], "validate": _v_choice(ANALYSIS_LOCATIONS)},
            "baseline_period_granularity": {"path": ["analysis", "baseline", "period_granularity"], "validate": _v_choice(BASELINE_PERIOD_GRANULARITIES)},
            "salience_weight_confidence": {"path": ["analysis", "salience", "weights", "confidence"], "validate": _v_nonnegative_number},
            "salience_weight_anomaly": {"path": ["analysis", "salience", "weights", "anomaly"], "validate": _v_nonnegative_number},
            "salience_weight_rarity": {"path": ["analysis", "salience", "weights", "rarity"], "validate": _v_nonnegative_number},
            "salience_min_effective_n": {"path": ["analysis", "salience", "anomaly", "min_effective_n"], "validate": _v_nonnegative_int},
            "field_qc_pass_confidence": {"path": ["analysis", "thresholds", "field_qc", "pass_confidence"], "validate": _v_unit_interval},
            "verification_clear_confidence": {"path": ["analysis", "thresholds", "verification", "clear_confidence"], "validate": _v_unit_interval},
            "verification_max_frames_scored": {"path": ["analysis", "thresholds", "verification", "max_frames_scored"], "validate": _v_positive_number},
            "dream_min_periods_for_trend": {"path": ["analysis", "thresholds", "dream", "min_periods_for_trend"], "validate": _v_positive_number},
            "dream_min_events_for_correlation": {"path": ["analysis", "thresholds", "dream", "min_events_for_correlation"], "validate": _v_positive_number},
            "dream_min_events_for_co_occurrence": {"path": ["analysis", "thresholds", "dream", "min_events_for_co_occurrence"], "validate": _v_positive_number},
            "dream_min_abs_effect": {"path": ["analysis", "thresholds", "dream", "min_abs_effect"], "validate": _v_nonnegative_number},
            "dream_max_p_value": {"path": ["analysis", "thresholds", "dream", "max_p_value"], "validate": _v_unit_interval},
            "media_image_format": {"path": ["media", "image", "format"], "validate": _v_nonempty_str},
            "media_image_quality": {"path": ["media", "image", "quality"], "validate": _v_int_range(1, 100)},
            "media_audio_format": {"path": ["media", "audio", "format"], "validate": _v_nonempty_str},
            "media_audio_sample_width_bytes": {"path": ["media", "audio", "sample_width_bytes"], "validate": _v_positive_int},
            "buffer_high_water_pct": {"path": ["buffer", "high_water_pct"], "validate": _v_positive_number},
            "buffer_hard_ceiling_pct": {"path": ["buffer", "hard_ceiling_pct"], "validate": _v_positive_number},
            "buffer_auto_sync_when_reachable": {"path": ["buffer", "auto_sync_when_reachable"], "validate": _v_bool},
            "buffer_pause_capture_at_ceiling": {"path": ["buffer", "pause_capture_at_ceiling"], "validate": _v_bool},
        },
        "station": {
            "station_name": {"path": ["station_name"], "validate": _v_nonempty_str},
            "capture_source_video": {"path": ["capture", "source", "video"], "validate": _v_str_or_null},
            "capture_source_audio": {"path": ["capture", "source", "audio"], "validate": _v_str_or_null},
            "sensor_camera_enabled": {"path": ["sensors", "camera", "enabled"], "validate": _v_bool},
            "sensor_audio_enabled": {"path": ["sensors", "audio", "enabled"], "validate": _v_bool},
            "sensor_gps_enabled": {"path": ["sensors", "gps", "enabled"], "validate": _v_bool},
            "visual_pi_path": {"path": ["models", "visual_pi", "path"], "validate": _v_nonempty_str, "is_path": True},
            "visual_desktop_path": {"path": ["models", "visual_desktop", "path"], "validate": _v_nonempty_str, "is_path": True},
            "capture_fps": {"path": ["capture", "fps"], "validate": _v_positive_number},
            "resolution_width": {"path": ["capture", "resolution", "width"], "validate": _v_positive_number},
            "resolution_height": {"path": ["capture", "resolution", "height"], "validate": _v_positive_number},
            "bytetrack_track_activation_threshold": {"path": ["capture", "bytetrack", "track_activation_threshold"], "validate": _v_positive_number},
            "bytetrack_minimum_matching_threshold": {"path": ["capture", "bytetrack", "minimum_matching_threshold"], "validate": _v_positive_number},
            "bytetrack_track_close_frames": {"path": ["capture", "bytetrack", "track_close_frames"], "validate": _v_positive_number},
            "bytetrack_frame_rate": {"path": ["capture", "bytetrack", "frame_rate"], "validate": _v_positive_number},
            "max_event_duration_seconds": {"path": ["capture", "max_event_duration_seconds"], "validate": _v_positive_number},
        },
        "channel": {
            "enabled": {"path": ["enabled"], "validate": _v_bool},
            "unit": {"path": ["unit"], "validate": _v_nonempty_str},
            "marine": {"path": ["marine"], "validate": _v_bool},
            "driver_interface": {"path": ["driver", "interface"], "validate": _v_nonempty_str},
            "driver_address": {"path": ["driver", "address"], "validate": _v_str_or_null},
            "driver_type": {"path": ["driver", "type"], "validate": _v_nonempty_str},
            "qc_gross_min": {"path": ["qc", "gross_range", "min"], "validate": _v_number},
            "qc_gross_max": {"path": ["qc", "gross_range", "max"], "validate": _v_number},
            "qc_sensor_min": {"path": ["qc", "sensor_range", "min"], "validate": _v_number},
            "qc_sensor_max": {"path": ["qc", "sensor_range", "max"], "validate": _v_number},
        },
    }


def _set_nested(target: dict, segments: list, value: Any) -> None:
    """Set value at a nested key path, creating intermediate objects as needed."""
    cursor = target
    for segment in segments[:-1]:
        nxt = cursor.get(segment)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[segment] = nxt
        cursor = nxt
    cursor[segments[-1]] = value


def _find_station_in(draft: dict, station_id: Any) -> dict:
    if not station_id or not isinstance(station_id, str):
        raise SettingsUpdateError("a station_id is required for this change")
    for station in draft.get("stations", []):
        if station.get("station_id") == station_id:
            return station
    raise SettingsUpdateError(f"no station with id {station_id}")


def _find_channel_in(station: dict, channel_id: Any) -> dict:
    if not channel_id or not isinstance(channel_id, str):
        raise SettingsUpdateError("a channel_id is required for this change")
    for channel in station.get("channels", []):
        if channel.get("id") == channel_id:
            return channel
    raise SettingsUpdateError(f"no channel with id {channel_id}")


def _apply_setting_change(draft: dict, change: Any, specs: dict, warnings: list, repo_root) -> None:
    """Validate one change and apply it to the working configuration copy."""
    if not isinstance(change, dict):
        raise SettingsUpdateError("each change must be an object")
    scope = change.get("scope")
    field = change.get("field")
    scope_specs = specs.get(scope)
    if scope_specs is None:
        raise SettingsUpdateError(f"unknown change scope {scope!r}")
    spec = scope_specs.get(field)
    if spec is None:
        raise SettingsUpdateError(f"{field!r} is not an editable field")

    where = f"{scope}.{field}"
    value = spec["validate"](change.get("value"), where)

    if scope == "global":
        target = draft
    elif scope == "station":
        target = _find_station_in(draft, change.get("station_id"))
    else:
        station = _find_station_in(draft, change.get("station_id"))
        target = _find_channel_in(station, change.get("channel_id"))

    _set_nested(target, spec["path"], value)

    # A model path that names a file not yet present is allowed and noted, not
    # refused, so a person can point at a model they are about to add.
    if spec.get("is_path") and isinstance(value, str):
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = Path(repo_root) / candidate
        if not candidate.exists():
            warnings.append(f"{where}: no file is present yet at {value}")


# How many recent observations to scan when finding each channel's most recent
# reading for the sensors overview. Readings are captured at detection events, so
# a few hundred recent observations covers the latest value per channel without a
# new storage query.
_READING_SCAN_LIMIT = 300


def _commit_draft(settings, draft: dict) -> None:
    """Swap a validated working copy in as the live configuration, or roll back.

    The draft becomes the live configuration only if it passes the full validator
    and writes; any failure restores the previous configuration untouched.
    """
    original = settings.raw
    settings.raw = draft
    try:
        _persist_settings(settings)
    except BackendError:
        settings.raw = original
        raise


def _clean_channel_request(request) -> dict:
    """Validate and shape a new environmental channel from a request.

    A channel is an environmental sensor a station records: an identifier, a unit,
    whether it is a marine channel (which carries an oceanographic quality flag),
    whether it is enabled, its hardware driver, and optional quality-control bounds.
    The shape mirrors the channels already in the configuration so a new one reads
    and validates exactly like the reference ones.
    """
    channel_id = request.id
    if not isinstance(channel_id, str) or not channel_id.strip():
        raise SettingsUpdateError("a channel id is required")

    channel: dict = {
        "id": channel_id.strip(),
        "unit": _v_nonempty_str(request.unit, "channel.unit"),
        "marine": request.marine if isinstance(request.marine, bool) else False,
        "enabled": request.enabled if isinstance(request.enabled, bool) else True,
    }

    driver = request.driver
    if driver is not None:
        if not isinstance(driver, dict):
            raise SettingsUpdateError("channel.driver must be an object")
        cleaned_driver: dict = {}
        for key in ("interface", "address", "type"):
            if driver.get(key) not in (None, ""):
                cleaned_driver[key] = str(driver[key])
        if cleaned_driver:
            channel["driver"] = cleaned_driver

    qc = request.qc
    if qc is not None:
        if not isinstance(qc, dict):
            raise SettingsUpdateError("channel.qc must be an object")
        cleaned_qc: dict = {}
        for range_key in ("gross_range", "sensor_range"):
            rng = qc.get(range_key)
            if isinstance(rng, dict):
                bounds: dict = {}
                for bound in ("min", "max"):
                    if rng.get(bound) is not None:
                        bounds[bound] = _v_number(rng[bound], f"channel.qc.{range_key}.{bound}")
                if bounds:
                    cleaned_qc[range_key] = bounds
        if qc.get("detection_limit") is not None:
            cleaned_qc["detection_limit"] = _v_number(qc["detection_limit"], "channel.qc.detection_limit")
        if cleaned_qc:
            channel["qc"] = cleaned_qc

    return channel


def _latest_channel_readings(db, station_id) -> dict:
    """The most recent stored reading for each channel of one station.

    Built from the existing readers rather than a new query: it scans recent
    observations and keeps the newest reading seen per channel. A station with no
    captured readings yet returns an empty map, which the interface shows as a
    channel that is configured but has not reported.
    """
    latest: dict = {}
    try:
        observations = db.list_observations(station_id=station_id, limit=_READING_SCAN_LIMIT)
    except Exception:  # noqa: BLE001 - an unreadable record must not break the overview
        return latest
    for obs in observations:
        for reading in db.list_environmental_readings(obs["id"]):
            channel = reading.get("channel")
            if channel is None:
                continue
            stamp = reading.get("created_at") or ""
            current = latest.get(channel)
            if current is None or stamp > (current.get("created_at") or ""):
                latest[channel] = {
                    "value": reading.get("value"),
                    "unit": reading.get("unit"),
                    "status": reading.get("status"),
                    "qartod_flag": reading.get("qartod_flag"),
                    "created_at": reading.get("created_at"),
                }
    return latest


def _dir_size(path: Path) -> Optional[int]:
    """Total size in bytes of every file under a directory, or None if unreadable.

    A missing directory reports zero, which is the honest figure before any data
    has been captured.
    """
    if not path.exists():
        return 0
    total = 0
    try:
        for entry in path.rglob("*"):
            try:
                if entry.is_file():
                    total += entry.stat().st_size
            except OSError:
                continue
    except Exception:  # noqa: BLE001 - an unreadable tree reports unknown, not an error
        return None
    return total


def _list_report_bundles(reports_dir: Path) -> list:
    """List report bundles already on disk, newest first."""
    if not reports_dir.exists():
        return []
    bundles = []
    for entry in sorted(reports_dir.iterdir(), reverse=True):
        if not entry.is_dir():
            continue
        files = [str(p.relative_to(reports_dir)).replace("\\", "/")
                 for p in sorted(entry.rglob("*")) if p.is_file()]
        bundles.append({
            "name": entry.name,
            "modified_utc": datetime.fromtimestamp(entry.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "files": files,
        })
    return bundles


# ===========================================================================
# Application factory
# ===========================================================================


def create_app(settings, database):
    """Build the FastAPI application bound to one settings object and database.

    The web framework is imported here rather than at module load, so importing
    this module never requires it; only building the app does. A clear error is
    raised if the framework is absent.
    """
    try:
        from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
        from pydantic import BaseModel
    except ImportError as exc:  # pragma: no cover - exercised only without the framework
        raise BackendDependencyError(
            "The web backend needs the fastapi and uvicorn packages, which are "
            "not installed. Install them with: pip install fastapi uvicorn"
        ) from exc

    app = FastAPI(
        title="Audtheia",
        description="Local desktop backend for the Audtheia environmental record.",
        version="2",
        docs_url=f"{API_PREFIX}/docs",
        openapi_url=f"{API_PREFIX}/openapi.json",
    )

    db = database

    class ReportRequest(BaseModel):
        station_id: Optional[str] = None
        start: Optional[str] = None
        end: Optional[str] = None
        formats: Optional[list] = None

    class SkillRequest(BaseModel):
        title: Optional[str] = None
        trigger_condition: Optional[str] = None
        instruction: Optional[str] = None
        tier: Optional[str] = None

    class LlmSelectRequest(BaseModel):
        name: Optional[str] = None

    class SettingsUpdateRequest(BaseModel):
        changes: Optional[list] = None

    class ChannelRequest(BaseModel):
        id: Optional[str] = None
        unit: Optional[str] = None
        marine: Optional[bool] = None
        enabled: Optional[bool] = None
        driver: Optional[dict] = None
        qc: Optional[dict] = None

    class SecretsRequest(BaseModel):
        values: Optional[dict] = None

    class StationCreateRequest(BaseModel):
        station_name: Optional[str] = None
        environment_type: Optional[str] = None
        habitat: Optional[str] = None

    class ProvisionRequest(BaseModel):
        host: Optional[str] = None
        user: Optional[str] = None
        port: Optional[int] = None

    class ObservationDeleteRequest(BaseModel):
        ids: Optional[list] = None

    # -- meta ------------------------------------------------------------

    @app.get(f"{API_PREFIX}/health")
    def health():
        return {"status": "ok", "time_utc": _utc_now_iso(), "timezone_display": str(settings.resolve_timezone())}

    # -- stations --------------------------------------------------------

    @app.get(f"{API_PREFIX}/stations")
    def stations():
        return db.list_stations()

    @app.get(f"{API_PREFIX}/stations/{{station_id}}")
    def station(station_id):
        row = db.get_station(station_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"no station with id {station_id}")
        return row

    # -- detections (visual events plus the desktop verification verdict) --

    @app.get(f"{API_PREFIX}/detections")
    def detections(station_id: str | None = Query(default=None), since: str | None = Query(default=None),
                   until: str | None = Query(default=None), species: str | None = Query(default=None),
                   limit: int = Query(default=100, ge=1, le=100000), offset: int = Query(default=0, ge=0)):
        sp = species.strip() if species and species.strip() else None
        total = db.count_observations(station_id=station_id, since=since, until=until, species=sp)
        rows = db.list_observations(station_id=station_id, since=since, until=until, species=sp,
                                    limit=limit, offset=offset)
        out = []
        for obs in rows:
            item = dict(obs)
            item["vision_detections"] = [c for c in db.list_child_detections(obs["id"]) if c.get("modality") == "vision"]
            item["verification"] = db.get_observation_verification(obs["id"])
            out.append(item)
        return {"items": out, "total": total, "limit": limit, "offset": offset}

    # Registered before '/detections/{observation_id}' so the literal path is not
    # captured as an id. Returns the full species list for the filter dropdown.
    @app.get(f"{API_PREFIX}/detections/species")
    def detection_species(station_id: str | None = Query(default=None)):
        return db.list_species(station_id=station_id)

    @app.post(f"{API_PREFIX}/detections/delete")
    def delete_detections(request: ObservationDeleteRequest):
        """Delete one or more observations and their stored media.

        Removing an observation cascades in the database to its child detections,
        environmental readings, verification, interpretations, and pattern links.
        The stored frame and audio clip are then removed from disk, guarded by the
        same in-data-directory check the media route uses, so nothing outside the
        data directory is ever touched. Unknown ids are ignored.
        """
        ids = [str(i) for i in (request.ids or []) if i]
        if not ids:
            raise HTTPException(status_code=400, detail="no observation ids were given to delete")
        result = db.delete_observations(ids)
        data_dir = Path(settings.path("data_dir")).resolve()
        media_removed = 0
        for rel in result.get("media", []):
            try:
                raw = Path(rel)
                target = (raw if raw.is_absolute() else Path(settings.repo_root) / raw).resolve()
            except (OSError, ValueError):
                continue
            if (target == data_dir or data_dir in target.parents) and target.is_file():
                try:
                    target.unlink()
                    media_removed += 1
                except OSError:
                    pass
        return {
            "status": "deleted",
            "ids": result.get("deleted", []),
            "count": len(result.get("deleted", [])),
            "media_removed": media_removed,
        }

    @app.get(f"{API_PREFIX}/detections/{{observation_id}}")
    def detection_detail(observation_id):
        obs = db.get_observation(observation_id)
        if obs is None:
            raise HTTPException(status_code=404, detail=f"no observation with id {observation_id}")
        children = db.list_child_detections(observation_id)
        return {
            "observation": obs,
            "vision_detections": [c for c in children if c.get("modality") == "vision"],
            "audio_detections": [c for c in children if c.get("modality") == "audio"],
            "environment": db.list_environmental_readings(observation_id),
            "verification": db.get_observation_verification(observation_id),
            "interpretations": db.list_interpretations(observation_id),
        }

    @app.get(f"{API_PREFIX}/detections/{{observation_id}}/frames")
    def detection_frames(observation_id):
        """Return every stored frame of an event with its per-frame annotation.

        The capture pipeline writes each detected frame to the event directory and
        appends one line per frame to `annotations.jsonl` (index, timestamp,
        confidence, box), alongside an `annotations.json` manifest. This read-only
        endpoint surfaces both so the interface can audit an observation's stats —
        the frame count, the true duration, and the per-frame confidences — rather
        than asking the scientist to trust them. Boxes are converted to the same
        x/y/w/h form the card overlay uses. Nothing is written or deleted.
        """
        obs = db.get_observation(observation_id)
        if obs is None:
            raise HTTPException(status_code=404, detail=f"no observation with id {observation_id}")
        rep = obs.get("representative_frame")
        if not rep:
            return {"observation": obs, "manifest": None, "frames": []}

        data_dir = Path(settings.path("data_dir")).resolve()
        rep_path = Path(rep)
        rep_abs = (rep_path if rep_path.is_absolute() else Path(settings.repo_root) / rep_path).resolve()
        if data_dir not in rep_abs.parents and rep_abs != data_dir:
            raise HTTPException(status_code=400, detail="frame path is outside the data directory")
        event_dir = rep_abs.parent

        manifest = None
        manifest_file = event_dir / "annotations.json"
        if manifest_file.is_file():
            try:
                manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = None

        frames = []
        index_file = event_dir / "annotations.jsonl"
        if index_file.is_file():
            try:
                lines = index_file.read_text(encoding="utf-8").splitlines()
            except OSError:
                lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fname = rec.get("file")
                if not fname:
                    continue
                frame_abs = (event_dir / fname).resolve()
                try:
                    rel = frame_abs.relative_to(Path(settings.repo_root)).as_posix()
                except ValueError:
                    rel = str(frame_abs)
                frame = {
                    "index": rec.get("index"),
                    "path": rel,
                    "captured_at": rec.get("captured_at"),
                    "confidence": rec.get("confidence"),
                    "class_name": rec.get("class_name"),
                }
                bbox = rec.get("bbox_xyxy") or []
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    frame["bbox_x"] = x1
                    frame["bbox_y"] = y1
                    frame["bbox_w"] = x2 - x1
                    frame["bbox_h"] = y2 - y1
                frames.append(frame)

        return {"observation": obs, "manifest": manifest, "frames": frames}

    # -- audio -----------------------------------------------------------

    @app.get(f"{API_PREFIX}/audio")
    def audio(station_id: str | None = Query(default=None), since: str | None = Query(default=None),
              until: str | None = Query(default=None), limit: int = Query(default=100, ge=1, le=1000)):
        rows = db.list_observations(station_id=station_id, since=since, until=until, limit=limit)
        out = []
        for obs in rows:
            if not _has_audio(obs):
                continue
            out.append({
                "observation_id": obs["id"],
                "event_name": obs.get("event_name"),
                "station_id": obs.get("station_id"),
                "first_seen": obs.get("first_seen"),
                "audio_clip_path": obs.get("audio_clip_path"),
                "audio_true_duration_seconds": obs.get("audio_true_duration_seconds"),
                "audio_capped": obs.get("audio_capped"),
                "acoustic_model_version": obs.get("acoustic_model_version"),
                "audio_detections": [c for c in db.list_child_detections(obs["id"]) if c.get("modality") == "audio"],
            })
        return out

    # -- gps -------------------------------------------------------------

    @app.get(f"{API_PREFIX}/gps")
    def gps(station_id: str | None = Query(default=None), since: str | None = Query(default=None),
            until: str | None = Query(default=None), limit: int = Query(default=500, ge=1, le=5000)):
        rows = db.list_observations(station_id=station_id, since=since, until=until, limit=limit)
        out = []
        for obs in rows:
            if not _has_gps(obs):
                continue
            out.append({
                "observation_id": obs["id"],
                "event_name": obs.get("event_name"),
                "station_id": obs.get("station_id"),
                "first_seen": obs.get("first_seen"),
                "gps_latitude": obs.get("gps_latitude"),
                "gps_longitude": obs.get("gps_longitude"),
                "gps_elevation": obs.get("gps_elevation"),
                "gps_status": obs.get("gps_status"),
                "time_provisional": obs.get("time_provisional"),
            })
        return out

    # -- analytics -------------------------------------------------------

    @app.get(f"{API_PREFIX}/analytics")
    def analytics(station_id=Query(default=None), since=Query(default=None), until=Query(default=None)):
        return _compute_analytics(db, station_id=station_id, since=since, until=until)

    # -- brain: models and memory, learning, skills ----------------------

    @app.get(f"{API_PREFIX}/brain/models")
    def brain_models():
        stations_models = []
        for station_conf in settings.stations():
            stations_models.append({
                "station_id": station_conf.get("station_id"),
                "station_name": station_conf.get("station_name"),
                "models": station_conf.get("models", {}),
            })
        return {"desktop_models": settings.raw.get("desktop_models", {}), "stations": stations_models}

    @app.get(f"{API_PREFIX}/brain/llm")
    def brain_llm():
        """Show the installed language models, which one is active, and the folder.

        The desktop dream pass and the verification interpreter both run this
        model. Selecting a different one takes effect the next time the station
        starts, since a model is loaded once when the desktop process begins.
        """
        return {
            "configured": _redact(dict(settings.raw.get("desktop_models", {}).get("llm", {}))),
            "directory": str(_llm_directory(settings)),
            "available": _list_gguf_models(settings),
            "active": _active_llm_name(settings),
            "runtime_available": _llm_runtime_available(),
            "note": "the desktop dream pass and interpretation use this model; a change applies the next time the station starts.",
        }

    @app.post(f"{API_PREFIX}/brain/llm/select")
    def select_llm(request: LlmSelectRequest):
        """Choose which installed GGUF model the desktop uses.

        The selected file is confirmed to be a GGUF model inside the language
        model folder before the choice is saved, so a crafted name cannot point
        the configuration at a file outside it. The path is stored the same way
        the other model paths are, relative to the project when it sits inside it.
        """
        if settings.node_role != "desktop":
            raise HTTPException(
                status_code=403,
                detail="the desktop language model is managed on the desktop; this node is not the desktop.",
            )
        name = (request.name or "").strip()
        if not name:
            raise HTTPException(status_code=422, detail="a model name is required")

        directory = _llm_directory(settings).resolve()
        target = (directory / name).resolve()
        if directory not in target.parents:
            raise HTTPException(status_code=400, detail="the model is outside the language model folder")
        if target.suffix != ".gguf" or not target.is_file():
            raise HTTPException(status_code=404, detail=f"no GGUF model named {name}")

        try:
            stored = str(target.relative_to(Path(settings.repo_root).resolve())).replace("\\", "/")
        except ValueError:
            stored = str(target)
        settings.raw.setdefault("desktop_models", {}).setdefault("llm", {})["path"] = stored
        _persist_settings(settings)
        return brain_llm()

    @app.get(f"{API_PREFIX}/brain/memory")
    def brain_memory(station_id=Query(default=None)):
        baselines = db.list_site_baselines(station_id=station_id)
        return {
            "site_baselines": baselines,
            "baseline_count": len(baselines),
            "note": "the permanent site gist the longitudinal pass builds and authoritative salience reads",
        }

    @app.get(f"{API_PREFIX}/brain/learning")
    def brain_learning(dream_pass_id=Query(default=None), status=Query(default=None)):
        patterns = db.list_patterns(dream_pass_id=dream_pass_id, status=status)
        framed = [_frame_pattern(p, db.list_pattern_observations(p["id"])) for p in patterns]
        return {
            "dream_passes": db.list_dream_passes(),
            "patterns": framed,
            "note": "patterns are candidate hypotheses, each traceable to its supporting events",
        }

    @app.get(f"{API_PREFIX}/brain/skills")
    def brain_skills(tier=Query(default=None)):
        return db.list_skills(tier=tier)

    def _require_desktop_author():
        """Refuse a skill write anywhere but the desktop.

        Skills are authored on the desktop and pushed down to a station on
        connect; a station never originates one. Guarding the write path to the
        desktop keeps that ownership rule true no matter which node happens to be
        serving the interface.
        """
        if settings.node_role != "desktop":
            raise HTTPException(
                status_code=403,
                detail="skills are authored on the desktop and synced to stations; "
                       "this node is not the desktop.",
            )

    def _clean_skill_request(request) -> dict:
        """Validate and normalize an authored skill, or reject it with a reason.

        A skill is four short pieces of text a person writes: a title, a trigger
        that says when it applies, an instruction that says what to do, and a type
        that decides which tier runs it. Each text field must be present and
        non-empty once trimmed, and within its length bound; the type must be one
        of the two the storage layer accepts. The type is what enforces the
        measured-versus-inferred firewall, so it is checked here rather than left
        to fail deeper in.
        """
        def _text(field_value, name, max_len):
            if not isinstance(field_value, str) or not field_value.strip():
                raise HTTPException(status_code=422, detail=f"skill {name} is required")
            trimmed = field_value.strip()
            if len(trimmed) > max_len:
                raise HTTPException(
                    status_code=422,
                    detail=f"skill {name} is longer than the {max_len}-character limit",
                )
            return trimmed

        tier = request.tier
        if tier not in SKILL_TIERS:
            raise HTTPException(
                status_code=422,
                detail=f"skill type must be one of: {', '.join(SKILL_TIERS)}",
            )
        return {
            "title": _text(request.title, "title", _SKILL_TITLE_MAX),
            "trigger_condition": _text(request.trigger_condition, "trigger", _SKILL_TEXT_MAX),
            "instruction": _text(request.instruction, "instruction", _SKILL_TEXT_MAX),
            "tier": tier,
        }

    @app.post(f"{API_PREFIX}/brain/skills", status_code=201)
    def create_skill(request: SkillRequest):
        """Author a new skill.

        The skill is written to the desktop's authoritative store; it reaches a
        station on the next connect through the existing settings-and-skills push,
        which this endpoint does not need to trigger.
        """
        from audtheia.storage.database import Skill, new_id, utc_now_iso

        _require_desktop_author()
        fields = _clean_skill_request(request)
        now = utc_now_iso()
        skill = Skill(
            id=new_id(),
            title=fields["title"],
            trigger_condition=fields["trigger_condition"],
            instruction=fields["instruction"],
            tier=fields["tier"],
            created_at=now,
            updated_at=now,
        )
        db.upsert_skill(skill)
        return db.get_skill(skill.id)

    @app.put(f"{API_PREFIX}/brain/skills/{{skill_id}}")
    def update_skill(skill_id, request: SkillRequest):
        """Edit an existing skill, keeping its identity and creation time."""
        from audtheia.storage.database import Skill

        _require_desktop_author()
        existing = db.get_skill(skill_id)
        if existing is None:
            raise HTTPException(status_code=404, detail=f"no skill with id {skill_id}")
        fields = _clean_skill_request(request)
        skill = Skill(
            id=skill_id,
            title=fields["title"],
            trigger_condition=fields["trigger_condition"],
            instruction=fields["instruction"],
            tier=fields["tier"],
            created_at=existing["created_at"],
            updated_at=_utc_now_iso(),
        )
        db.upsert_skill(skill)
        return db.get_skill(skill_id)

    @app.delete(f"{API_PREFIX}/brain/skills/{{skill_id}}")
    def delete_skill(skill_id):
        """Remove a skill from the desktop store."""
        _require_desktop_author()
        existing = db.get_skill(skill_id)
        if existing is None:
            raise HTTPException(status_code=404, detail=f"no skill with id {skill_id}")
        db.delete_skill(skill_id)
        return {"status": "deleted", "id": skill_id}

    # -- dream pass status and controls ----------------------------------

    @app.get(f"{API_PREFIX}/dream/status")
    def dream_status():
        passes = db.list_dream_passes()
        active = next((p for p in passes if p.get("status") == "running"), None)
        return {"passes": passes, "active": active}

    @app.post(f"{API_PREFIX}/dream/{{dream_pass_id}}/pause")
    def dream_pause(dream_pass_id):
        return _set_dream_status(dream_pass_id, "paused")

    @app.post(f"{API_PREFIX}/dream/{{dream_pass_id}}/resume")
    def dream_resume(dream_pass_id):
        return _set_dream_status(dream_pass_id, "running")

    def _set_dream_status(dream_pass_id, new_status):
        current = db.get_dream_pass(dream_pass_id)
        if current is None:
            raise HTTPException(status_code=404, detail=f"no dream pass with id {dream_pass_id}")
        # Pausing and resuming set the cooperative signal the pass itself reads
        # between cycles; a finished or errored pass is not a valid target.
        if current.get("status") in ("complete", "error"):
            raise HTTPException(
                status_code=409,
                detail=f"dream pass is {current.get('status')} and cannot be {new_status}",
            )
        db.update_dream_pass(dream_pass_id, status=new_status)
        return db.get_dream_pass(dream_pass_id)

    # -- reports: list existing, and produce a new one in the background --

    @app.get(f"{API_PREFIX}/reports")
    def reports():
        reports_dir = Path(settings.path("reports_dir"))
        formats = settings.raw["schedules"]["reports"]["formats"]
        return {"reports_dir": str(reports_dir), "configured_formats": formats, "bundles": _list_report_bundles(reports_dir)}

    @app.post(f"{API_PREFIX}/reports", status_code=202)
    def create_report(request: ReportRequest, background: BackgroundTasks):
        from audtheia.reports.generate import generate_report

        stamp = _utc_now_iso()

        def _job():
            generate_report(
                settings, db,
                station_id=request.station_id, start=request.start, end=request.end,
                formats=request.formats, generated_at=stamp,
            )

        background.add_task(_job)
        return {
            "status": "scheduled",
            "generated_at": stamp,
            "scope": {"station_id": request.station_id, "start": request.start, "end": request.end},
            "formats": request.formats or settings.raw["schedules"]["reports"]["formats"],
            "note": "generation runs in the background; poll GET /api/reports for the new bundle",
        }

    @app.get(f"{API_PREFIX}/reports/file")
    def report_file(path: str = Query(...)):
        """Return one file from inside the reports directory.

        The requested path is resolved and confirmed to sit within the reports
        directory before anything is read, so a crafted path cannot escape it.
        """
        reports_dir = Path(settings.path("reports_dir")).resolve()
        target = (reports_dir / path).resolve()
        if reports_dir not in target.parents and target != reports_dir:
            raise HTTPException(status_code=400, detail="path is outside the reports directory")
        if not target.is_file():
            raise HTTPException(status_code=404, detail="no such report file")
        return FileResponse(str(target))

    @app.get(f"{API_PREFIX}/media")
    def media_file(path: str = Query(...)):
        """Serve a stored detection frame or audio clip from inside the data directory.

        The stored path is resolved and confirmed to sit within the data directory
        before anything is read, so a crafted path cannot escape it.
        """
        data_dir = Path(settings.path("data_dir")).resolve()
        raw = Path(path)
        target = (raw if raw.is_absolute() else Path(settings.repo_root) / raw).resolve()
        if data_dir not in target.parents and target != data_dir:
            raise HTTPException(status_code=400, detail="path is outside the data directory")
        if not target.is_file():
            raise HTTPException(status_code=404, detail="no such media file")
        return FileResponse(str(target))

    # -- settings (a read view, and a guarded edit for allowlisted fields) --

    @app.get(f"{API_PREFIX}/settings")
    def get_settings():
        from audtheia.config import ALLOWED_HABITATS, ENVIRONMENT_TYPES

        return {
            "config": _redact(settings.raw),
            "secrets_configured": bool(getattr(settings, "secrets", None)),
            "secrets_status": _species_secret_status(settings),
            "node_role": settings.node_role,
            "editable_fields": {scope: sorted(fields.keys()) for scope, fields in _editable_field_specs().items()},
            "allowed_habitats": sorted(ALLOWED_HABITATS),
            "environment_types": list(ENVIRONMENT_TYPES),
            "note": "secrets are never returned; a listed field can be changed through the settings update path",
        }

    @app.post(f"{API_PREFIX}/settings/secrets")
    def update_secrets(request: SecretsRequest):
        """Set or clear the species-data credentials in the local secrets file.

        Only the species credentials can be set here, and only their presence is
        ever reported back, never their values. An empty string clears a
        credential. The file is written atomically and stays out of the committed
        configuration.
        """
        if settings.node_role != "desktop":
            raise HTTPException(status_code=403, detail="credentials are managed on the desktop; this node is not the desktop.")
        values = request.values
        if not isinstance(values, dict) or not values:
            raise HTTPException(status_code=422, detail="no credential values were provided")

        path = _secrets_file_path(settings)
        current = _read_secrets_file(path)
        for key, value in values.items():
            if key not in SPECIES_SECRET_KEYS:
                raise HTTPException(status_code=422, detail=f"{key!r} is not an editable credential")
            if not isinstance(value, str):
                raise HTTPException(status_code=422, detail=f"{key} must be text")
            current[key] = value
            settings.secrets[key] = value

        _write_secrets_file(path, current)
        return {
            "secrets_status": _species_secret_status(settings),
            "note": "credentials saved to the local secrets file; they are never committed.",
        }

    @app.post(f"{API_PREFIX}/settings/update")
    def update_settings(request: SettingsUpdateRequest):
        """Apply one or more allowlisted configuration changes, or reject them all.

        Only fields named in the allowlist can be changed, so a system-owned value
        is never reachable here. The batch is applied to a working copy, the whole
        configuration is validated, and only then is it written atomically through
        the shared persist path. Any invalid change leaves the saved file untouched.
        """
        if settings.node_role != "desktop":
            raise HTTPException(
                status_code=403,
                detail="configuration is edited on the desktop; this node is not the desktop.",
            )

        changes = request.changes
        if not isinstance(changes, list) or not changes:
            raise HTTPException(status_code=422, detail="no changes were provided")

        specs = _editable_field_specs()
        draft = copy.deepcopy(settings.raw)
        warnings: list = []
        try:
            for change in changes:
                _apply_setting_change(draft, change, specs, warnings, settings.repo_root)
        except SettingsUpdateError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        original = settings.raw
        settings.raw = draft
        try:
            _persist_settings(settings)
        except BackendError as exc:
            settings.raw = original
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return {
            "config": _redact(settings.raw),
            "warnings": warnings,
            "note": "saved; capture, model, and schedule changes take effect the next time the station starts.",
        }

    def _draft_station(station_id):
        """A deep copy of the configuration and the target station within it."""
        draft = copy.deepcopy(settings.raw)
        for station in draft.get("stations", []):
            if station.get("station_id") == station_id:
                return draft, station
        raise HTTPException(status_code=404, detail=f"no station with id {station_id}")

    @app.post(f"{API_PREFIX}/settings/stations/{{station_id}}/channels", status_code=201)
    def add_channel(station_id, request: ChannelRequest):
        """Add an environmental channel (a sensor) to a station.

        The channel is shaped and checked, appended to a working copy, and the
        whole configuration is validated before it is written, so a duplicate
        identifier or a malformed channel is refused with the file left untouched.
        """
        if settings.node_role != "desktop":
            raise HTTPException(status_code=403, detail="stations are configured on the desktop; this node is not the desktop.")
        try:
            channel = _clean_channel_request(request)
        except SettingsUpdateError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        draft, station = _draft_station(station_id)
        station.setdefault("channels", []).append(channel)
        try:
            _commit_draft(settings, draft)
        except BackendError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"config": _redact(settings.raw), "note": "channel added; it takes effect the next time the station starts."}

    @app.delete(f"{API_PREFIX}/settings/stations/{{station_id}}/channels/{{channel_id}}")
    def remove_channel(station_id, channel_id):
        """Remove an environmental channel from a station."""
        if settings.node_role != "desktop":
            raise HTTPException(status_code=403, detail="stations are configured on the desktop; this node is not the desktop.")
        draft, station = _draft_station(station_id)
        channels = station.get("channels", [])
        kept = [c for c in channels if c.get("id") != channel_id]
        if len(kept) == len(channels):
            raise HTTPException(status_code=404, detail=f"no channel with id {channel_id}")
        station["channels"] = kept
        try:
            _commit_draft(settings, draft)
        except BackendError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"config": _redact(settings.raw), "note": "channel removed."}

    @app.get(f"{API_PREFIX}/sensors")
    def sensors_overview(station_id: str | None = Query(default=None)):
        """A per-station view of environmental channels and each one's latest reading.

        The configured channels come from settings; the latest value, unit,
        quality status, and marine quality flag come from the stored record. A
        channel with no readings yet is returned with a null reading, which the
        interface shows as configured but not yet reporting.
        """
        stations_out = []
        for station in settings.stations():
            sid = station.get("station_id")
            if station_id and sid != station_id:
                continue
            latest = _latest_channel_readings(db, sid)
            channels = []
            for channel in station.get("channels", []):
                channels.append({
                    "id": channel.get("id"),
                    "unit": channel.get("unit"),
                    "marine": channel.get("marine"),
                    "enabled": channel.get("enabled"),
                    "driver": channel.get("driver"),
                    "latest_reading": latest.get(channel.get("id")),
                })
            stations_out.append({
                "station_id": sid,
                "station_name": station.get("station_name"),
                "environment_type": station.get("environment_type"),
                "sensors": station.get("sensors", {}),
                "channels": channels,
            })
        return {
            "stations": stations_out,
            "note": "environmental readings are captured at each detection; a channel appears here once its station is capturing.",
        }

    @app.get(f"{API_PREFIX}/storage")
    def storage_status():
        """Live storage figures for the store this node keeps.

        Disk capacity comes from the filesystem that holds the data; the database
        and folder sizes are measured on disk; the sync backlog is the count of
        records not yet confirmed elsewhere. Reading these never changes anything.
        """
        import shutil

        db_path = Path(settings.db_path())
        data_dir = Path(settings.path("data_dir"))
        reports_dir = Path(settings.path("reports_dir"))

        anchor = next((p for p in (data_dir, db_path.parent, Path(settings.repo_root)) if p.exists()), Path("."))
        disk = {"total": None, "used": None, "free": None}
        try:
            usage = shutil.disk_usage(str(anchor))
            disk = {"total": usage.total, "used": usage.used, "free": usage.free}
        except Exception:  # noqa: BLE001 - a platform without the call reports unknown
            pass

        try:
            db_size = db_path.stat().st_size if db_path.exists() else None
        except OSError:
            db_size = None

        try:
            unsynced = db.count_unsynced()
        except Exception:  # noqa: BLE001 - a store without the syncable tables reports none
            unsynced = {}

        return {
            "role": settings.node_role,
            "disk": disk,
            "database": {"path": str(db_path), "size": db_size},
            "data": {"path": str(data_dir), "size": _dir_size(data_dir)},
            "reports": {"path": str(reports_dir), "size": _dir_size(reports_dir)},
            "unsynced": unsynced,
            "total_unsynced": sum(unsynced.values()) if unsynced else 0,
            "note": "syncing and cleaning a field station's buffer run when a Pi is connected.",
        }

    # -- stations: add and remove (desktop-authored configuration) ---------

    @app.post(f"{API_PREFIX}/settings/stations", status_code=201)
    def add_station(request: StationCreateRequest):
        """Create a new station with a generated identifier and safe defaults."""
        if settings.node_role != "desktop":
            raise HTTPException(status_code=403, detail="stations are configured on the desktop; this node is not the desktop.")
        from audtheia.config import ENVIRONMENT_TYPES, ALLOWED_HABITATS
        from audtheia.storage.database import new_id

        name = (request.station_name or "").strip()
        if not name:
            raise HTTPException(status_code=422, detail="a station name is required")
        environment = request.environment_type
        if environment not in ENVIRONMENT_TYPES:
            raise HTTPException(status_code=422, detail=f"environment_type must be one of: {', '.join(ENVIRONMENT_TYPES)}")
        habitat = (request.habitat or "").strip() or None
        if habitat and habitat not in ALLOWED_HABITATS:
            raise HTTPException(status_code=422, detail="habitat is not in the allowed list; leave it blank or choose a listed value")

        station = _new_station_dict(new_id(), name, environment, habitat)
        draft = copy.deepcopy(settings.raw)
        draft.setdefault("stations", []).append(station)
        try:
            _commit_draft(settings, draft)
        except BackendError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"config": _redact(settings.raw), "station_id": station["station_id"], "note": "station added; add its sensors and connect its Pi from here."}

    @app.delete(f"{API_PREFIX}/settings/stations/{{station_id}}")
    def remove_station(station_id):
        """Remove a station, keeping at least one defined."""
        if settings.node_role != "desktop":
            raise HTTPException(status_code=403, detail="stations are configured on the desktop; this node is not the desktop.")
        draft = copy.deepcopy(settings.raw)
        stations = draft.get("stations", [])
        kept = [s for s in stations if s.get("station_id") != station_id]
        if len(kept) == len(stations):
            raise HTTPException(status_code=404, detail=f"no station with id {station_id}")
        if not kept:
            raise HTTPException(status_code=409, detail="at least one station must remain")
        draft["stations"] = kept
        try:
            _commit_draft(settings, draft)
        except BackendError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"config": _redact(settings.raw), "note": "station removed."}

    # -- guided Pi provisioning (key-first, no password) -------------------

    def _require_station(station_id):
        for station in settings.stations():
            if station.get("station_id") == station_id:
                return station
        raise HTTPException(status_code=404, detail=f"no station with id {station_id}")

    def _pi_script() -> Path:
        return Path(settings.repo_root) / "scripts" / "bootstrap_setup_pi.py"

    @app.get(f"{API_PREFIX}/stations/{{station_id}}/provision/key")
    def provision_key(station_id):
        """The desktop's public key for a station, to authorize on the Pi at flash time."""
        if settings.node_role != "desktop":
            raise HTTPException(status_code=403, detail="provisioning runs from the desktop; this node is not the desktop.")
        _require_station(station_id)
        try:
            proc = subprocess.run(
                [sys.executable, str(_pi_script()), "--station-id", station_id, "--show-key"],
                capture_output=True, text=True, cwd=str(settings.repo_root), timeout=30,
            )
        except Exception as exc:  # noqa: BLE001 - reported as a clear server error
            raise HTTPException(status_code=500, detail=f"could not produce the key: {exc}") from exc
        if proc.returncode != 0 or not proc.stdout.strip():
            raise HTTPException(status_code=500, detail=(proc.stderr or "could not produce the key").strip()[:300])
        return {
            "public_key": proc.stdout.strip(),
            "note": "authorize this key on the Pi: paste it into Raspberry Pi Imager's public-key field when flashing, or append it to ~/.ssh/authorized_keys on the Pi.",
        }

    @app.post(f"{API_PREFIX}/stations/{{station_id}}/provision")
    def start_provision(station_id, request: ProvisionRequest):
        """Begin connecting a station's Pi in the background, using key authentication."""
        if settings.node_role != "desktop":
            raise HTTPException(status_code=403, detail="provisioning runs from the desktop; this node is not the desktop.")
        _require_station(station_id)
        host = (request.host or "").strip()
        user = (request.user or "").strip()
        port = int(request.port or 22)
        if not host:
            raise HTTPException(status_code=422, detail="the Pi's address is required (an IP, or a name ending in .local)")
        if not user:
            raise HTTPException(status_code=422, detail="the Pi's login user is required")

        existing = _PROVISION_JOBS.get(station_id)
        if existing and existing["proc"].poll() is None:
            raise HTTPException(status_code=409, detail="a connection is already in progress for this station")

        # Remember the connection target on the station so it need not be entered
        # again. This is additive and never required for the configuration to load.
        draft = copy.deepcopy(settings.raw)
        for station in draft.get("stations", []):
            if station.get("station_id") == station_id:
                station["provisioning"] = {"host": host, "user": user, "port": port}
        try:
            _commit_draft(settings, draft)
        except BackendError:
            pass

        log_path = Path(tempfile.gettempdir()) / f"audtheia-provision-{station_id}.log"
        log = open(log_path, "w", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                [sys.executable, str(_pi_script()), "--station-id", station_id,
                 "--host", host, "--user", user, "--port", str(port), "--key-auth"],
                stdout=log, stderr=subprocess.STDOUT, cwd=str(settings.repo_root),
            )
        finally:
            log.close()
        _PROVISION_JOBS[station_id] = {"proc": proc, "log": str(log_path), "started": _utc_now_iso()}
        return {"state": "running", "note": "connecting to the Pi; poll the status endpoint for progress."}

    @app.get(f"{API_PREFIX}/stations/{{station_id}}/provision/status")
    def provision_status(station_id):
        """Report progress of an in-flight or finished provisioning run."""
        job = _PROVISION_JOBS.get(station_id)
        if not job:
            return {"state": "idle", "log": ""}
        returncode = job["proc"].poll()
        try:
            log_text = Path(job["log"]).read_text(encoding="utf-8")
        except OSError:
            log_text = ""
        if returncode is None:
            state = "running"
        elif returncode == 0:
            state = "succeeded"
        else:
            state = "failed"
        return {"state": state, "returncode": returncode, "started": job.get("started"), "log": log_text}

    # -- desktop capture control (start and stop the live loop) -----------

    @app.post(f"{API_PREFIX}/capture/{{station_id}}/start")
    def start_capture(station_id):
        """Start desktop capture and processing for a station in background threads."""
        if settings.node_role != "desktop":
            raise HTTPException(status_code=403, detail="capture runs on the desktop; this node is not the desktop.")
        station = _require_station(station_id)
        source = ((station.get("capture", {}) or {}).get("source", {}) or {}).get("video")
        if not source:
            raise HTTPException(status_code=422, detail="set a desktop capture source for this station first (Detections, Set capture source)")

        # Desktop capture detects with the station's own screening model; without
        # it there is nothing to detect, so it is required and checked up front
        # with a clear message rather than failing cryptically deeper in.
        model = ((station.get("models", {}) or {}).get("visual_desktop", {}) or {}).get("path")
        model_path = None
        if model:
            model_path = Path(model)
            if not model_path.is_absolute():
                model_path = Path(settings.repo_root) / model_path
        if not (model_path and model_path.exists()):
            raise HTTPException(
                status_code=422,
                detail="this station has no desktop detector model in place. Edit the station and set its Desktop screening model (an ONNX file under models/); capture needs it to detect.",
            )

        job = _CAPTURE_JOBS.get(station_id)
        if job and job.running():
            raise HTTPException(status_code=409, detail="capture is already running for this station")
        try:
            from audtheia.app.orchestrator import DesktopStation

            desktop = DesktopStation.build(settings, station_id=station_id)
            live = desktop.start_background()
        except Exception as exc:  # noqa: BLE001 - reported as a clear client error
            raise HTTPException(status_code=422, detail=f"could not start capture: {exc}") from exc
        _CAPTURE_JOBS[station_id] = live
        return {"state": "running", "note": "capturing from the source; detections appear as the model finds them."}

    @app.post(f"{API_PREFIX}/capture/{{station_id}}/stop")
    def stop_capture(station_id):
        """Stop a station's running desktop capture."""
        if settings.node_role != "desktop":
            raise HTTPException(status_code=403, detail="capture runs on the desktop; this node is not the desktop.")
        job = _CAPTURE_JOBS.get(station_id)
        if not job:
            return {"state": "idle"}
        try:
            job.stop()
        finally:
            _CAPTURE_JOBS.pop(station_id, None)
        return {"state": "stopped"}

    @app.get(f"{API_PREFIX}/capture/status")
    def capture_status():
        """Which stations are currently capturing on the desktop."""
        running = [sid for sid, job in _CAPTURE_JOBS.items() if job.running()]
        return {"running": running}

    # -- static frontend (served locally, present from a later step) ------

    _mount_static(app, settings, StaticFiles)

    return app


def _mount_static(app, settings, StaticFiles) -> None:
    """Mount the single-page frontend if its files are present.

    The frontend is added in a later step. Until its directory exists this does
    nothing, so the backend runs on its own with the API fully available.
    """
    try:
        static_dir = Path(settings.path("static_dir")) if "static_dir" in settings.raw.get("paths", {}) else None
    except Exception:  # noqa: BLE001 - a missing path key simply means no frontend yet
        static_dir = None
    if static_dir is None:
        # Fall back to the conventional location next to this module.
        candidate = Path(__file__).resolve().parent / "static"
        static_dir = candidate if candidate.is_dir() and any(candidate.iterdir()) else None
    if static_dir and static_dir.is_dir() and any(static_dir.iterdir()):
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


# ===========================================================================
# Running the server
# ===========================================================================


def run(settings=None, database=None) -> None:
    """Start the backend on the configured host and port.

    Loads configuration and opens the database if they are not supplied, then
    serves until interrupted. Binds to the loopback address by default, so the
    interface is reachable only from the desktop it runs on.
    """
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise BackendDependencyError(
            "Running the backend needs the uvicorn server, which is not "
            "installed. Install it with: pip install uvicorn"
        ) from exc

    if settings is None:
        from audtheia.config import load_settings
        settings = load_settings()
    if database is None:
        from audtheia.storage.database import Database
        database = Database(settings.db_path(), **settings.database_kwargs())

    app = create_app(settings, database)
    server = settings.raw.get("server", {})
    host = server.get("host", "127.0.0.1")
    port = int(server.get("port", 8000))
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
