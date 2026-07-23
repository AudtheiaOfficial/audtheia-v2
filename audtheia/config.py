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
  - Reads an optional, gitignored config/settings.local.json holding the
    absolute filesystem paths that belong to one machine and to no other, so a
    person's account name, home directory, or external drive letter never
    becomes part of the committed configuration. See the local-overrides
    section below for why this is a hard rule rather than a convention.
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
import re
from datetime import datetime, timezone, tzinfo
from pathlib import Path, PurePosixPath, PureWindowsPath
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

REPORT_SCHEDULES = ("hourly", "daily", "weekly", "biweekly", "monthly")
DREAM_SCHEDULES = ("hourly", "daily", "weekly", "biweekly", "monthly")
REPORT_FORMATS = ("pdf", "csv")

REPRESENTATIVE_FRAME_RULES = ("highest_confidence",)

# The acoustic model is a single flat block per station, not a set of named
# slots. A station sits in one place and listens with one model, so the block
# carries that model's path, labels path, and the audio shape read from or
# proposed for the file (sample_rate, window_seconds, output_key), plus version
# and citation. The keys a valid block may carry; each is optional, and the
# shape is checked in _validate_station. No model family is ever named here: the
# adapter is chosen at load time from the file's own form, not from any name.
ACOUSTIC_BLOCK_KEYS = (
    "path",
    "labels_path",
    "sample_rate",
    "window_seconds",
    "output_key",
    "version",
    "citation",
)

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

    def __init__(
        self,
        raw: dict,
        *,
        settings_path: Path,
        repo_root: Path,
        secrets: dict,
        local_overrides_path: Optional[Path] = None,
        stale_local_overrides: Optional[list] = None,
    ) -> None:
        self.raw = raw
        self.settings_path = settings_path
        self.repo_root = repo_root
        self.secrets = secrets
        # Where this machine's absolute paths are kept. Defaulted rather than
        # required so a Settings built directly by a test still works.
        self.local_overrides_path = (
            local_overrides_path
            if local_overrides_path is not None
            else repo_root / LOCAL_OVERRIDES_DEFAULT_PATH
        )
        # Pointers in the local file that matched nothing in the committed
        # configuration, usually a station that has since been removed. Kept so
        # the interface can report a stale local file instead of ignoring it.
        self.stale_local_overrides = list(stale_local_overrides or [])

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


# ===========================================================================
# Local overrides
#
# An absolute filesystem path describes one machine. It carries the account
# name of whoever is logged in, their home directory, and often the drive they
# happened to plug in. config/settings.json is committed, so an absolute path
# written into it is published to everyone who reads the repository, and stays
# in the history after it is removed from the file.
#
# Absolute paths are still legitimate: a desktop store may live on an external
# drive and a model folder may sit outside the repository. So they are not
# refused, they are relocated. Every absolute path lives in
# config/settings.local.json, which .gitignore excludes, and is merged back
# over the committed configuration at load time. The committed file therefore
# describes the deployment and the local file describes the machine.
#
# The split is decided by shape rather than by a list of known fields, so a
# path field added later is protected without anyone remembering to protect
# it. A value qualifies when its key names a path and the value is absolute.
# ===========================================================================


LOCAL_OVERRIDES_DEFAULT_PATH = "config/settings.local.json"

# The one configuration file that machine overrides apply to. Anything else is a
# scenario configuration that must describe itself completely; see load_settings
# for why inheriting a machine path into one of those is dangerous.
CANONICAL_SETTINGS_FILENAME = "settings.json"

# Keys whose value is a filesystem path. Matched by shape so a field added
# later is covered without editing this module.
_PATH_KEY_SUFFIXES = ("_path", "_dir")
_PATH_KEY_EXACT = ("path",)

# Keys that identify an element of a list, so an override written against a
# station survives that station being reordered in the committed file. Checked
# in order; the first one an element carries is the one used.
_IDENTITY_KEYS = ("station_id", "channel_id", "id")

# Containers whose every scalar member may name something on this machine, and
# so must be addressable by an override whatever it currently holds.
#
# A capture source is the case this exists for. It is stored as a source
# expression rather than as a path, and it is legitimately empty on a fresh
# clone, so neither the key nor the value identifies it as machine specific
# while it is blank. Recognising it by the container it sits in gives it a
# stable pointer from the start. Without one, a machine path entered in the
# interface is routed correctly into the local file and then cannot be read
# back, because the pointer it was filed under does not exist in a
# configuration whose committed value is still empty. The setting appears to
# save and is gone on the next load.
_MACHINE_VALUE_CONTAINERS = ("capture.source",)


def _is_path_key(key: str) -> bool:
    """Whether a configuration key holds a filesystem path."""
    return key in _PATH_KEY_EXACT or key.endswith(_PATH_KEY_SUFFIXES)


def _is_machine_value_container(prefix: str) -> bool:
    """Whether a container holds values that may name this machine.

    Matched on the pointer rather than on a station identifier so it holds for
    every station, and for a desktop configuration that has no station list.
    """
    return any(prefix == name or prefix.endswith("." + name) for name in _MACHINE_VALUE_CONTAINERS)


def _is_absolute_path_value(value: Any) -> bool:
    """Whether a value is an absolute path under either operating system's rules.

    Both conventions are checked because one configuration travels between a
    Windows desktop and a Raspberry Pi. Python's own Path only understands the
    host's convention, so a Windows drive letter read on the Pi, or a POSIX
    root read on Windows, would otherwise look relative and slip through.
    """
    if not isinstance(value, str) or not value:
        return False
    return PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()


# A machine path buried inside a longer string. The capture source is the case
# this exists for: it is stored as a source expression rather than as a path,
# something of the form  file:"C:\Users\somebody\Downloads\clip.mp3" , so the
# key is not a path key and the value is not a bare path, and both of the checks
# above miss it entirely. A drive letter or a UNC prefix appearing anywhere in a
# string is machine-specific wherever it sits, so it is matched on sight.
#
# The lookbehind is what keeps a URL out of this. A drive letter is a single
# letter before the colon, so without it the "s://" inside "https://" matches
# and every citation and video source in the configuration is mistaken for a
# machine path.
_EMBEDDED_MACHINE_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]|\\\\[^\\/\s]+[\\/]")


def _contains_machine_path(value: Any) -> bool:
    """Whether a string carries a machine-specific path anywhere inside it."""
    if not isinstance(value, str) or not value:
        return False
    return _EMBEDDED_MACHINE_PATH.search(value) is not None


def unquote_path_value(value: Any) -> Any:
    """A configured path with one matched pair of surrounding quotes removed.

    Windows "Copy as path" wraps a path in double quotes and people paste that
    verbatim. Left as written the quotes become part of the filename, nothing is
    ever found there, and the interface truthfully reports a missing file while
    displaying a path that looks correct. Removing them on load rather than only
    on save means a path stored before this existed starts working by itself,
    instead of having to be found and retyped.

    Separators are deliberately left alone. The save path evens them out, and a
    value written into the local override file by hand is that machine's own
    business; rewriting it here would change a stored value that already works.

    Only values under a path key are passed here. A capture source is a source
    expression rather than a path, and its own reader already unquotes it.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1].strip()
    return text


def normalise_path_values(raw: dict) -> None:
    """Unquote every configured path in place, so every reader sees the same thing."""
    for _pointer, key, container, value in _walk_path_values(raw):
        if _is_path_key(key) and isinstance(value, str):
            container[key] = unquote_path_value(value)


def _element_pointer(element: Any, index: int) -> str:
    """The pointer segment naming one element of a list.

    An element that identifies itself is referred to by that identity, so the
    override still applies after the list is reordered. Anything else falls
    back to its position, which is all there is to go on.
    """
    if isinstance(element, dict):
        for key in _IDENTITY_KEYS:
            value = element.get(key)
            if isinstance(value, str) and value:
                return f"[{key}={value}]"
    return f"[{index}]"


def _walk_path_values(node: Any, prefix: str = ""):
    """Yield every (pointer, key, container, value) whose key names a path.

    Walks dictionaries and lists together because model paths live inside the
    stations list, not only at the top level.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            pointer = f"{prefix}.{key}" if prefix else key
            scalar = not isinstance(value, (dict, list))
            named_path = _is_path_key(key) and scalar
            # A capture source is addressable whether or not it is filled in,
            # so an override survives the value being empty in the committed
            # file, which is how a fresh clone and a privacy-cleaned station
            # both start out.
            machine_slot = _is_machine_value_container(prefix) and scalar
            # A value carrying a drive letter is machine-specific whatever its
            # key is called, so it is treated as a path field on the strength of
            # its content alone.
            if named_path or machine_slot or _contains_machine_path(value):
                yield pointer, key, node, value
            else:
                yield from _walk_path_values(value, pointer)
    elif isinstance(node, list):
        for index, element in enumerate(node):
            yield from _walk_path_values(element, f"{prefix}{_element_pointer(element, index)}")


def collect_absolute_paths(raw: dict) -> dict:
    """Every absolute path in a configuration, as a pointer to value map.

    This is what must not reach the committed file. Exposed rather than kept
    private because both the loader and the backend's save path need it, and
    because a test proves the guarantee by calling it directly.
    """
    found = {}
    for pointer, _key, _container, value in _walk_path_values(raw):
        if _is_absolute_path_value(value) or _contains_machine_path(value):
            found[pointer] = value
    return found


def _set_pointer(raw: dict, pointer: str, value: Any) -> bool:
    """Write a value at a pointer, returning whether the pointer resolved.

    A pointer that names a station no longer present resolves to nothing and is
    ignored, so a stale local file is inert rather than an error.
    """
    for target_pointer, key, container, _value in _walk_path_values(raw):
        if target_pointer == pointer:
            container[key] = value
            return True
    return False


def pointer_value(raw: dict, pointer: str, default: Any = None) -> Any:
    """The value at a pointer, or a default when the pointer resolves to nothing."""
    for target_pointer, _key, _container, value in _walk_path_values(raw):
        if target_pointer == pointer:
            return value
    return default


def apply_local_overrides(raw: dict, overrides: dict) -> list:
    """Merge a local override map into a configuration in place.

    Returns the pointers that did not resolve, so a caller can report a stale
    local file rather than silently ignoring it.
    """
    stale = []
    for pointer, value in overrides.items():
        if not _set_pointer(raw, pointer, value):
            stale.append(pointer)
    return stale


def _load_local_overrides(local_path: Path) -> dict:
    """Read the optional local override file.

    The file may be absent, which is the normal case on a fresh clone and means
    the committed configuration is used exactly as written.
    """
    if not local_path.exists():
        return {}
    try:
        loaded = json.loads(local_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"local settings file is not valid JSON: {exc}") from None
    if not isinstance(loaded, dict):
        raise ConfigError("local settings file must contain a JSON object")
    overrides = loaded.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ConfigError("local settings file: 'overrides' must be a JSON object")
    return overrides


def local_overrides_path_for(raw: dict, repo_root: Path) -> Path:
    """Where the local override file lives for a given configuration.

    Configurable in the same way the secrets file is, so a deployment that
    keeps machine state elsewhere can say so, and defaulted otherwise.
    """
    ref = (raw.get("local_overrides") or {}).get("path") or LOCAL_OVERRIDES_DEFAULT_PATH
    path = Path(ref)
    return path if path.is_absolute() else repo_root / path


def load_settings(
    settings_path: Optional[str | Path] = None,
    *,
    secrets_path: Optional[str | Path] = None,
    apply_local: Optional[bool] = None,
) -> Settings:
    """Read, validate, and resolve a configuration file.

    With no argument, the configuration is read from config/settings.json
    relative to this package's repository. Pass an explicit path to load a
    different file (for example a field station's own copy). The returned
    Settings object is fully validated; if anything is wrong, a ConfigError
    explains exactly what and where.

    Machine-specific absolute paths are merged in from config/settings.local.json
    when the canonical settings file is being loaded. Pass apply_local to force
    that on or off; the default decides by filename, which keeps a throwaway
    scenario configuration from inheriting this machine's paths.
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

    # The machine's own absolute paths are merged in before validation, so what
    # is checked is the configuration that will actually run, and so a required
    # path supplied only by the local file is not reported as missing.
    #
    # Overrides apply only to the canonical settings file. The test suites write
    # throwaway configurations beside the real one, each deliberately redirected
    # at a temporary sandbox, and several of them clear their working directories
    # with shutil.rmtree. If a machine override could reach one of those, a
    # sandboxed data directory could be silently replaced by the real one and a
    # suite would delete captured field data. A scenario configuration therefore
    # describes itself completely and inherits nothing from this machine.
    local_path = local_overrides_path_for(raw, repo_root)
    if apply_local is None:
        apply_local = settings_path.name == CANONICAL_SETTINGS_FILENAME
    local_overrides = _load_local_overrides(local_path) if apply_local else {}
    stale_pointers = apply_local_overrides(raw, local_overrides)

    # Paths are cleaned before validation so what is checked, displayed, and
    # opened at runtime are the same string, whether it came from the committed
    # file, the local override file, or a quoted paste made before this existed.
    normalise_path_values(raw)

    _validate(raw)

    if secrets_path is None:
        secrets_ref = raw.get("secrets", {}).get("path", "config/secrets.json")
        secrets_path = (
            Path(secrets_ref)
            if Path(secrets_ref).is_absolute()
            else repo_root / secrets_ref
        )
    secrets = _load_secrets(Path(secrets_path))

    return Settings(
        raw,
        settings_path=settings_path,
        repo_root=repo_root,
        secrets=secrets,
        local_overrides_path=local_path,
        stale_local_overrides=stale_pointers,
    )


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

    # -- ui (optional) ---------------------------------------------------
    # Remembered interface preferences, so a choice such as the color theme
    # survives a restart and follows the hub rather than living only in one
    # browser. The block is optional and every value in it is optional, so a
    # configuration without it simply falls back to the interface defaults.
    ui = raw.get("ui")
    if ui is not None:
        _require_type(ui, (dict,), "ui")
        for key in ("theme", "last_dark", "last_light"):
            if key in ui and ui[key] is not None:
                _require_type(ui[key], (str,), f"ui.{key}")

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

    # Optional fixed position. A station deployed at a known, surveyed point may
    # record its coordinates here so events carry a location even with no live
    # receiver. Each part is optional and may be null; when present it is bounded
    # to a valid decimal-degree range so a typo is caught here rather than at
    # capture. Coordinates are WGS84 decimal degrees, elevation is meters.
    location = station.get("location")
    if location is not None:
        _require_type(location, (dict,), f"{where}.location")
        latitude = location.get("latitude")
        if latitude is not None:
            if isinstance(latitude, bool) or not isinstance(latitude, (int, float)) or not (-90 <= latitude <= 90):
                raise ConfigError(f"{where}.location.latitude must be a number between -90 and 90, or null")
        longitude = location.get("longitude")
        if longitude is not None:
            if isinstance(longitude, bool) or not isinstance(longitude, (int, float)) or not (-180 <= longitude <= 180):
                raise ConfigError(f"{where}.location.longitude must be a number between -180 and 180, or null")
        elevation = location.get("elevation")
        if elevation is not None and (isinstance(elevation, bool) or not isinstance(elevation, (int, float))):
            raise ConfigError(f"{where}.location.elevation must be a number, or null")

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

    # The acoustic model is one flat block. Every key is optional, so a station
    # with no acoustic model set still loads and honestly reads as having none;
    # what is present is shape-checked so the acoustic capture can rely on it.
    acoustic = _require(models, "acoustic", f"{where}.models")
    _require_type(acoustic, (dict,), f"{where}.models.acoustic")
    for str_key in ("path", "labels_path", "output_key"):
        value = acoustic.get(str_key)
        if value is not None and not isinstance(value, str):
            raise ConfigError(f"{where}.models.acoustic.{str_key} must be a string or null")
    sample_rate = acoustic.get("sample_rate")
    if sample_rate is not None and (
        isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0
    ):
        raise ConfigError(f"{where}.models.acoustic.sample_rate must be a positive integer or null")
    window_seconds = acoustic.get("window_seconds")
    if window_seconds is not None and (
        isinstance(window_seconds, bool) or not isinstance(window_seconds, (int, float)) or window_seconds <= 0
    ):
        raise ConfigError(f"{where}.models.acoustic.window_seconds must be a positive number or null")

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
        # Video and audio are both optional desktop sources: a station may run
        # visual capture, acoustic capture, both, or neither (a Pi-only station).
        # Each is checked only when present, so an audio-only station is valid.
        video = source.get("video")
        if video is not None:
            _require_type(video, (str,), f"{where}.source.video")
        audio = source.get("audio")
        if audio is not None:
            _require_type(audio, (str,), f"{where}.source.audio")
