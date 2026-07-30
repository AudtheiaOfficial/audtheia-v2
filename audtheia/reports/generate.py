"""Audtheia V2 desktop report generation.

Path: audtheia/reports/generate.py

This module turns the authoritative desktop store into two shareable outputs:
a human-readable PDF and a set of machine-readable CSV tables. It runs on the
desktop hub only, because that is where the complete record lives (field
detections after they sync, the desktop verification verdict, the interpretive
points, and the longitudinal pass output).

Reporting is one of only two activities in the system that run on a schedule
the operator sets; it consumes the analysis output and writes nothing back to
the record, so a report can be regenerated any number of times without changing
a single stored value.

Three commitments shape everything below:

  - Provenance travels with every value. Each data point is shown together with
    its source (a sensor reading, a detection model call, a downstream language
    model inference, or the longitudinal pass) and its quality-control or
    missing-data status. Measured values and inferred values are kept visually
    and structurally distinct, never blended.

  - A discovered pattern is a candidate hypothesis, never a finding. Every
    pattern from the longitudinal pass is labeled as a candidate and always
    carries its effect size, the kind of effect size it is, the test behind it,
    and the exact span of data it rests on.

  - Data age is disclosed. The taxonomic backbone snapshot date and the
    conservation-status fetch date behind any taxonomic or conservation field
    are shown, as are the versions of the models and data snapshots that stand
    behind each result.

The CSV output uses only the Python standard library. The PDF output uses the
fpdf2 library, imported lazily inside the renderer so that importing this module,
gathering the report model, and writing the CSV all work with the PDF library
absent; only an actual PDF render requires it, and it fails loudly with a clear
message if the library is missing.

All configuration (the output directory, the requested formats, and the display
time zone) is read through the validated settings loader. Stored timestamps are
always in UTC; the PDF localizes them for display while the CSV preserves the
exact stored UTC strings.
"""

from __future__ import annotations

import csv
import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from typing import Any, Iterable, Optional


# ===========================================================================
# Controlled-vocabulary display labels
#
# These mirror the values the schema stores. They are presentation labels only;
# the stored codes are never altered. Keeping them here means a report never
# hardcodes a species, a channel, or a model, only the fixed status vocabularies
# the database itself defines.
# ===========================================================================

# The provenance of a value: where it came from.
DATA_SOURCE_LABELS = {
    "sensor": "measured (sensor)",
    "database": "reference cache",
    "model": "measured (detection model)",
    "llm_inferred": "inferred (language model)",
    "dream": "candidate (longitudinal pass)",
}

# Which provenance values are measured facts and which are downstream inference.
# This split is what the report renders as a visible firewall.
MEASURED_SOURCES = frozenset({"sensor", "model", "database"})
INFERRED_SOURCES = frozenset({"llm_inferred", "dream"})

# Measurement or missing-data status for a single value.
STATUS_LABELS = {
    "measured": "measured",
    "not_measured": "not measured",
    "below_detection_limit": "below detection limit",
    "sensor_error": "sensor error",
    "not_applicable": "not applicable",
}

# Record-level lifecycle state.
QC_STATE_LABELS = {
    "qc_pending": "quality control pending",
    "qc_passed": "quality control passed",
    "qc_deferred": "deferred to desktop review",
    "verified": "verified",
}

# The QARTOD flag scale, applied to marine sensor channels only.
QARTOD_LABELS = {
    1: "pass",
    2: "not evaluated",
    3: "suspect",
    4: "fail",
    9: "missing",
}

# The kind of effect size a candidate pattern carries, so a bare number is never
# shown without saying what it means.
EFFECT_SIZE_TYPE_LABELS = {
    "r": "correlation coefficient r",
    "cohens_d": "standardized mean difference (Cohen's d)",
    "log_odds": "log odds ratio",
}

# The class of a candidate pattern.
PATTERN_TYPE_LABELS = {
    "temporal_shift": "temporal shift",
    "co_occurrence": "co-occurrence",
    "envelope_correlation": "environmental envelope correlation",
    "novel_cluster": "novel cluster",
}

DREAM_PHASE_LABELS = {"nrem": "consolidation phase", "rem": "generative phase"}

# The single framing every candidate pattern is presented under.
HYPOTHESIS_BANNER = "CANDIDATE HYPOTHESIS (not an established finding)"

# The compact UTC stamp used to name a report bundle folder.
_BUNDLE_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"

# The size the report bundle names an all-stations scope.
_ALL_STATIONS_LABEL = "all_stations"


class ReportError(RuntimeError):
    """A report could not be produced for a reason the operator should see."""


class ReportDependencyError(ReportError):
    """A requested output needs a library that is not installed.

    Raised only when a PDF is actually requested and the PDF library is absent,
    so that importing this module and producing CSV never depend on it.
    """


# ===========================================================================
# The report model
#
# Gathering reads the database once into these plain containers, and both the
# CSV writer and the PDF writer render from them. Neither renderer touches the
# database, so the two outputs are always built from an identical snapshot.
# ===========================================================================


@dataclass
class ObservationRecord:
    """One event and everything captured or derived for it."""

    observation: dict
    vision_detections: list[dict] = field(default_factory=list)
    audio_detections: list[dict] = field(default_factory=list)
    environment: list[dict] = field(default_factory=list)
    verification: Optional[dict] = None
    interpretations: list[dict] = field(default_factory=list)


@dataclass
class PatternRecord:
    """One candidate pattern with the events it rests on and its pass context."""

    pattern: dict
    supporting_observation_ids: list[str] = field(default_factory=list)
    dream_pass: Optional[dict] = None


@dataclass
class TelemetryRecord:
    """One station heartbeat and its per-channel error counts."""

    telemetry: dict
    errors: list[dict] = field(default_factory=list)


@dataclass
class ReportModel:
    """A complete, render-ready snapshot for one report."""

    generated_at_utc: str
    display_timezone: str
    scope_label: str
    station_id: Optional[str]
    window_start_utc: Optional[str]
    window_end_utc: Optional[str]
    stations: list[dict] = field(default_factory=list)
    records: list[ObservationRecord] = field(default_factory=list)
    patterns: list[PatternRecord] = field(default_factory=list)
    telemetry: list[TelemetryRecord] = field(default_factory=list)
    species_reference: list[dict] = field(default_factory=list)
    analytics: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)


@dataclass
class ReportResult:
    """What a generate call produced, so a caller can locate the outputs."""

    bundle_dir: Path
    formats: list[str] = field(default_factory=list)
    pdf_path: Optional[Path] = None
    csv_paths: list[Path] = field(default_factory=list)


# ===========================================================================
# Time helpers
#
# Storage is always UTC. The CSV keeps the stored strings exactly; the PDF
# localizes them for a reader using the configured display time zone.
# ===========================================================================


def _parse_utc(value: Optional[str]) -> Optional[datetime]:
    """Parse a stored ISO 8601 timestamp as an aware UTC datetime.

    A trailing Z is accepted, and a value with no zone is taken to be UTC,
    which is the storage contract. A value that cannot be parsed returns
    nothing rather than raising, so one malformed cell never aborts a report.
    """
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z") or text.endswith("z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _format_local(value: Optional[str], tz: tzinfo) -> str:
    """Render a stored UTC timestamp in the display time zone for a reader."""
    dt = _parse_utc(value)
    if dt is None:
        return "not recorded" if not value else str(value)
    local = dt.astimezone(tz)
    return local.strftime("%Y-%m-%d %H:%M:%S %Z").strip()


def _tz_name(tz: tzinfo) -> str:
    """A short name for the display time zone, for the report header."""
    try:
        name = datetime.now(tz).strftime("%Z")
    except Exception:  # noqa: BLE001 - fall back to the object's own label
        name = ""
    return name or str(tz)


# ===========================================================================
# Value formatting and provenance tagging
# ===========================================================================


def _num(value: Any, digits: int = 3) -> str:
    """Format a number for display, or say plainly when there is none."""
    if value is None:
        return "not recorded"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value:  # not-a-number guard
            return "not recorded"
        text = f"{value:.{digits}f}"
        # Trim trailing zeros while keeping a leading digit.
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"
    return str(value)


def _source_label(data_source: Optional[str]) -> str:
    if not data_source:
        return "unlabeled"
    return DATA_SOURCE_LABELS.get(data_source, str(data_source))


def _status_label(status: Optional[str]) -> str:
    if not status:
        return ""
    return STATUS_LABELS.get(status, str(status))


def _qartod_label(flag: Optional[int]) -> str:
    if flag is None:
        return ""
    return QARTOD_LABELS.get(int(flag), str(flag))


def _provenance_tag(data_source: Optional[str], status: Optional[str] = None,
                    qartod_flag: Optional[int] = None) -> str:
    """A compact bracketed provenance tag shown next to a value.

    Every rendered value carries one, so a reader never has to guess whether a
    number was measured, inferred, or derived, or whether it passed quality
    control.
    """
    parts = [_source_label(data_source)]
    status_text = _status_label(status)
    if status_text:
        parts.append(status_text)
    qartod_text = _qartod_label(qartod_flag)
    if qartod_text:
        parts.append("QARTOD " + qartod_text)
    return "[" + " | ".join(parts) + "]"


def _is_inferred(data_source: Optional[str]) -> bool:
    return data_source in INFERRED_SOURCES


def _taxon_name(detection: dict) -> str:
    """The best available display name for a detected taxon."""
    return (
        detection.get("scientific_name")
        or detection.get("common_name")
        or detection.get("gbif_usage_key")
        or "unidentified"
    )


def _taxon_key(detection: dict) -> Optional[str]:
    """A stable identity for a taxon, matching how local rarity is counted."""
    return detection.get("gbif_usage_key") or detection.get("common_name") or detection.get("scientific_name")


# ===========================================================================
# Gathering: read the database once into a ReportModel
# ===========================================================================


class ReportGenerator:
    """Builds reports from one settings object and one database handle.

    Everything is read through the database access layer; this class issues no
    SQL of its own, so the storage contract stays in exactly one place.
    """

    def __init__(self, settings, database) -> None:
        self._settings = settings
        self._db = database

    # -- configuration ---------------------------------------------------

    def _resolve_formats(self, formats: Optional[Iterable[str]]) -> list[str]:
        """The output formats to produce, from the argument or the settings.

        The settings value has already been validated by the loader against the
        allowed set, so it is trusted here. An explicit argument overrides it,
        for an on-demand report that wants only one format.
        """
        if formats is not None:
            requested = [str(f).lower() for f in formats]
        else:
            reports = self._settings.raw["schedules"]["reports"]
            requested = [str(f).lower() for f in reports["formats"]]
        allowed = {"pdf", "csv"}
        chosen = [f for f in requested if f in allowed]
        if not chosen:
            raise ReportError(
                "no supported output format was requested; expected any of "
                "pdf, csv"
            )
        # Preserve a stable order regardless of how they were listed.
        return [f for f in ("pdf", "csv") if f in chosen]

    def _reports_dir(self, output_dir: Optional[str | Path]) -> Path:
        if output_dir is not None:
            return Path(output_dir)
        return Path(self._settings.path("reports_dir"))

    # -- public entry ----------------------------------------------------

    def generate(
        self,
        *,
        station_id: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        formats: Optional[Iterable[str]] = None,
        output_dir: Optional[str | Path] = None,
        generated_at: Optional[str] = None,
    ) -> ReportResult:
        """Produce the requested reports and return where they were written.

        station_id limits the report to one station; leaving it empty reports
        every station. start and end are UTC ISO 8601 bounds on event time;
        either may be omitted for an open-ended window. formats defaults to the
        configured schedule's formats. output_dir defaults to the configured
        reports directory. generated_at lets a caller pin the report's own
        timestamp; it defaults to now in UTC.
        """
        chosen_formats = self._resolve_formats(formats)
        tz = self._settings.resolve_timezone()
        stamp = generated_at or _utc_now_iso()

        model = self._gather(
            station_id=station_id, start=start, end=end,
            generated_at=stamp, tz=tz,
        )

        bundle_dir = self._reports_dir(output_dir) / _bundle_name(model, stamp)
        result = ReportResult(bundle_dir=bundle_dir, formats=[])

        if "csv" in chosen_formats:
            csv_paths = write_csv_bundle(model, bundle_dir / "csv")
            result.csv_paths = csv_paths
            result.formats.append("csv")

        if "pdf" in chosen_formats:
            # Render the executive-summary figures alongside the PDF, into an
            # assets folder in the same bundle. When matplotlib is absent this
            # returns nothing and the PDF simply omits its figures.
            from audtheia.reports.charts import render_charts
            charts = render_charts(model, bundle_dir / "assets")
            pdf_path = write_pdf(model, bundle_dir / "report.pdf", charts=charts)
            result.pdf_path = pdf_path
            result.formats.append("pdf")

        return result

    # -- gathering -------------------------------------------------------

    def _gather(self, *, station_id, start, end, generated_at, tz) -> ReportModel:
        db = self._db

        stations = db.list_stations()
        if station_id is not None:
            stations = [s for s in stations if s["id"] == station_id]
            if not stations:
                raise ReportError(f"no station with id {station_id!r} is registered")
            scope_label = stations[0]["station_name"]
        else:
            scope_label = "all stations"

        observations = db.list_observations(station_id=station_id, since=start, until=end)
        # Present events oldest first, so a report reads as a timeline.
        observations = sorted(observations, key=lambda o: (o.get("first_seen") or "", o.get("id") or ""))

        records: list[ObservationRecord] = []
        in_scope_ids: set[str] = set()
        for obs in observations:
            oid = obs["id"]
            in_scope_ids.add(oid)
            children = db.list_child_detections(oid)
            vision = [c for c in children if c.get("modality") == "vision"]
            audio = [c for c in children if c.get("modality") == "audio"]
            records.append(
                ObservationRecord(
                    observation=obs,
                    vision_detections=vision,
                    audio_detections=audio,
                    environment=db.list_environmental_readings(oid),
                    verification=db.get_observation_verification(oid),
                    interpretations=db.list_interpretations(oid),
                )
            )

        patterns = self._gather_patterns(
            station_id=station_id, start=start, end=end, in_scope_ids=in_scope_ids
        )
        telemetry = self._gather_telemetry(stations, start=start)
        species_reference = self._gather_species(records)

        analytics = _compute_analytics(records, telemetry)
        provenance = _collect_provenance(records, patterns, species_reference)

        return ReportModel(
            generated_at_utc=generated_at,
            display_timezone=_tz_name(tz),
            scope_label=scope_label,
            station_id=station_id,
            window_start_utc=start,
            window_end_utc=end,
            stations=stations,
            records=records,
            patterns=patterns,
            telemetry=telemetry,
            species_reference=species_reference,
            analytics=analytics,
            provenance=provenance,
        )

    def _gather_patterns(self, *, station_id, start, end, in_scope_ids) -> list[PatternRecord]:
        db = self._db
        scoped = station_id is not None or start is not None or end is not None
        out: list[PatternRecord] = []
        pass_cache: dict[str, Optional[dict]] = {}
        for pat in db.list_patterns():
            supporting = db.list_pattern_observations(pat["id"])
            if scoped and not (set(supporting) & in_scope_ids):
                # A candidate that rests on no event in this report's scope is
                # left out, so a station or window report only shows patterns it
                # can actually trace to its own events.
                continue
            pass_id = pat.get("dream_pass_id")
            if pass_id not in pass_cache:
                pass_cache[pass_id] = db.get_dream_pass(pass_id) if pass_id else None
            out.append(
                PatternRecord(
                    pattern=pat,
                    supporting_observation_ids=supporting,
                    dream_pass=pass_cache.get(pass_id),
                )
            )
        return out

    def _gather_telemetry(self, stations, *, start) -> list[TelemetryRecord]:
        db = self._db
        out: list[TelemetryRecord] = []
        for station in stations:
            rows = db.list_station_telemetry(station["id"], since=start)
            for row in rows:
                out.append(
                    TelemetryRecord(
                        telemetry=row,
                        errors=db.list_telemetry_errors(row["id"]),
                    )
                )
        return out

    def _gather_species(self, records) -> list[dict]:
        db = self._db
        keys: set[str] = set()
        for rec in records:
            for det in list(rec.vision_detections) + list(rec.audio_detections):
                key = det.get("gbif_usage_key")
                if key:
                    keys.add(key)
        all_species = db.list_species_reference()
        if not keys:
            return all_species
        matched = [s for s in all_species if s.get("gbif_usage_key") in keys]
        # Fall back to the full cache if none of the detected taxa resolved to a
        # cached key, so the data-age disclosure is never silently empty.
        return matched or all_species


# ===========================================================================
# Derived analytics and provenance inventory
#
# These are computations over the records already in the report, never new
# measurements. They are labeled as derived wherever they are shown, so a
# summary count is never mistaken for a sensor reading or a model call.
# ===========================================================================


def _compute_analytics(records: list[ObservationRecord], telemetry: list[TelemetryRecord]) -> dict:
    total_events = len(records)
    by_trigger: dict[str, int] = {}
    by_qc: dict[str, int] = {}
    verified_count = 0
    detections_by_modality = {"vision": 0, "audio": 0}
    taxon_events: dict[str, int] = {}
    taxon_display: dict[str, str] = {}
    channel_status: dict[str, dict[str, int]] = {}
    first_times = []
    last_times = []

    for rec in records:
        obs = rec.observation
        trig = obs.get("trigger_source") or "unknown"
        by_trigger[trig] = by_trigger.get(trig, 0) + 1
        qc = obs.get("qc_state") or "unknown"
        by_qc[qc] = by_qc.get(qc, 0) + 1
        if rec.verification and rec.verification.get("verified"):
            verified_count += 1
        ft = _parse_utc(obs.get("first_seen"))
        lt = _parse_utc(obs.get("last_seen"))
        if ft:
            first_times.append(ft)
        if lt:
            last_times.append(lt)

        detections_by_modality["vision"] += len(rec.vision_detections)
        detections_by_modality["audio"] += len(rec.audio_detections)

        seen_here: set[str] = set()
        for det in list(rec.vision_detections) + list(rec.audio_detections):
            key = _taxon_key(det)
            if not key:
                continue
            taxon_display.setdefault(key, _taxon_name(det))
            if key not in seen_here:
                seen_here.add(key)
                taxon_events[key] = taxon_events.get(key, 0) + 1

        for reading in rec.environment:
            ch = reading.get("channel") or "unknown"
            bucket = channel_status.setdefault(ch, {})
            status = reading.get("status") or "unknown"
            bucket[status] = bucket.get(status, 0) + 1

    effort = _summarize_effort(telemetry)

    return {
        "total_events": total_events,
        "events_by_trigger_source": by_trigger,
        "events_by_qc_state": by_qc,
        "verified_count": verified_count,
        "verified_fraction": (verified_count / total_events) if total_events else 0.0,
        "species_richness": len(taxon_events),
        "detections_by_modality": detections_by_modality,
        "taxon_event_counts": taxon_events,
        "taxon_display_names": taxon_display,
        "channel_status_counts": channel_status,
        "earliest_event_utc": (min(first_times).strftime("%Y-%m-%dT%H:%M:%SZ") if first_times else None),
        "latest_event_utc": (max(last_times).strftime("%Y-%m-%dT%H:%M:%SZ") if last_times else None),
        "effort": effort,
    }


def _summarize_effort(telemetry: list[TelemetryRecord]) -> dict:
    """Aggregate raw effort context from telemetry heartbeats.

    These are summed measured quantities that give a denominator for reading the
    counts above; they are not turned into a normalized rate here, because a
    defensible effort-normalized rarity figure is a separate piece of work and
    is deliberately not asserted in a report.
    """
    per_station: dict[str, dict[str, float]] = {}
    for rec in telemetry:
        t = rec.telemetry
        sid = t.get("station_id") or "unknown"
        bucket = per_station.setdefault(
            sid,
            {"heartbeats": 0, "camera_uptime_seconds": 0.0, "frames_processed": 0.0,
             "npu_active_seconds": 0.0, "valid_audio_seconds": 0.0},
        )
        bucket["heartbeats"] += 1
        for key in ("camera_uptime_seconds", "frames_processed", "npu_active_seconds", "valid_audio_seconds"):
            value = t.get(key)
            if isinstance(value, (int, float)):
                bucket[key] += value
    return per_station


def _collect_provenance(records, patterns, species_reference) -> dict:
    screening_versions: set[str] = set()
    acoustic_versions: set[str] = set()
    rfdetr_versions: set[str] = set()
    interpretation_versions: set[str] = set()
    pattern_versions: set[str] = set()
    gbif_dates: set[str] = set()
    iucn_dates: set[str] = set()

    for rec in records:
        obs = rec.observation
        _add(screening_versions, obs.get("screening_model_version"))
        _add(acoustic_versions, obs.get("acoustic_model_version"))
        _add(gbif_dates, obs.get("gbif_snapshot_date"))
        _add(iucn_dates, obs.get("iucn_fetch_date"))
        if rec.verification:
            _add(rfdetr_versions, rec.verification.get("rfdetr_version"))
        for interp in rec.interpretations:
            _add(interpretation_versions, interp.get("model_version"))

    for prec in patterns:
        _add(pattern_versions, prec.pattern.get("model_version"))

    for sp in species_reference:
        _add(gbif_dates, sp.get("gbif_snapshot_date"))
        _add(iucn_dates, sp.get("iucn_fetch_date"))

    return {
        "screening_model_versions": sorted(screening_versions),
        "acoustic_model_versions": sorted(acoustic_versions),
        "rfdetr_versions": sorted(rfdetr_versions),
        "interpretation_model_versions": sorted(interpretation_versions),
        "pattern_model_versions": sorted(pattern_versions),
        "gbif_snapshot_dates": sorted(gbif_dates),
        "iucn_fetch_dates": sorted(iucn_dates),
    }


def _add(target: set, value) -> None:
    if value:
        target.add(str(value))


# ===========================================================================
# Bundle naming and time-of-generation helpers
# ===========================================================================


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _bundle_name(model: ReportModel, generated_at: str) -> str:
    dt = _parse_utc(generated_at) or datetime.now(timezone.utc)
    stamp = dt.strftime(_BUNDLE_STAMP_FORMAT)
    if model.station_id is not None and model.stations:
        raw = model.stations[0].get("station_name") or _ALL_STATIONS_LABEL
    else:
        raw = _ALL_STATIONS_LABEL
    return f"report_{_slug(raw)}_{stamp}"


def _slug(text: str) -> str:
    """A filesystem-safe token for a station name, for a folder name."""
    safe = [c if (c.isalnum() or c in "-_") else "_" for c in str(text)]
    token = "".join(safe).strip("_")
    return token or _ALL_STATIONS_LABEL


# ===========================================================================
# CSV output (standard library only)
#
# One table per section. Every value column that has a provenance in the schema
# is written next to its data_source and status, so the firewall survives an
# export into a spreadsheet.
# ===========================================================================


def write_csv_bundle(model: ReportModel, csv_dir: str | Path) -> list[Path]:
    """Write every section as its own CSV file and return the paths written."""
    out_dir = Path(csv_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    written.append(_write_csv(out_dir / "observations.csv", _OBSERVATION_HEADER, _rows_observations(model)))
    written.append(_write_csv(out_dir / "detections.csv", _DETECTION_HEADER, _rows_vision_detections(model)))
    written.append(_write_csv(out_dir / "verification.csv", _VERIFICATION_HEADER, _rows_verification(model)))
    written.append(_write_csv(out_dir / "audio.csv", _AUDIO_HEADER, _rows_audio(model)))
    written.append(_write_csv(out_dir / "environment.csv", _ENVIRONMENT_HEADER, _rows_environment(model)))
    written.append(_write_csv(out_dir / "interpretations.csv", _INTERPRETATION_HEADER, _rows_interpretations(model)))
    written.append(_write_csv(out_dir / "patterns.csv", _PATTERN_HEADER, _rows_patterns(model)))
    written.append(_write_csv(out_dir / "pattern_observations.csv", _PATTERN_OBS_HEADER, _rows_pattern_obs(model)))
    written.append(_write_csv(out_dir / "analytics.csv", _ANALYTICS_HEADER, _rows_analytics(model)))
    written.append(_write_csv(out_dir / "telemetry.csv", _TELEMETRY_HEADER, _rows_telemetry(model)))
    written.append(_write_csv(out_dir / "species_reference.csv", _SPECIES_HEADER, _rows_species(model)))
    written.append(_write_csv(out_dir / "provenance.csv", _PROVENANCE_HEADER, _rows_provenance(model)))
    return written


def _write_csv(path: Path, header: list[str], rows: Iterable[list]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if cell is None else cell for cell in row])
    return path


_OBSERVATION_HEADER = [
    "observation_id", "event_name", "station_id", "trigger_source",
    "first_seen_utc", "last_seen_utc", "duration_seconds", "time_provisional",
    "qc_state", "qc_reason", "event_data_source",
    "screening_confidence", "screening_model_version", "acoustic_model_version",
    "gbif_snapshot_date", "iucn_fetch_date",
    "salience_provisional", "anomaly_magnitude_provisional",
    "gps_latitude", "gps_longitude", "gps_elevation", "gps_status",
    "audio_clip_path", "audio_true_duration_seconds", "audio_capped",
    "representative_frame", "frame_count",
]


def _rows_observations(model: ReportModel):
    for rec in model.records:
        o = rec.observation
        yield [
            o.get("id"), o.get("event_name"), o.get("station_id"), o.get("trigger_source"),
            o.get("first_seen"), o.get("last_seen"), o.get("duration"), o.get("time_provisional"),
            o.get("qc_state"), o.get("qc_reason"), o.get("data_source"),
            o.get("screening_confidence"), o.get("screening_model_version"), o.get("acoustic_model_version"),
            o.get("gbif_snapshot_date"), o.get("iucn_fetch_date"),
            o.get("salience_provisional"), o.get("anomaly_magnitude_provisional"),
            o.get("gps_latitude"), o.get("gps_longitude"), o.get("gps_elevation"), o.get("gps_status"),
            o.get("audio_clip_path"), o.get("audio_true_duration_seconds"), o.get("audio_capped"),
            o.get("representative_frame"), o.get("frame_count"),
        ]


_DETECTION_HEADER = [
    "observation_id", "detection_id", "modality",
    "gbif_usage_key", "scientific_name", "common_name", "confidence",
    "data_source", "status", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
]


def _rows_vision_detections(model: ReportModel):
    for rec in model.records:
        for det in rec.vision_detections:
            yield [
                det.get("observation_id"), det.get("id"), det.get("modality"),
                det.get("gbif_usage_key"), det.get("scientific_name"), det.get("common_name"), det.get("confidence"),
                det.get("data_source"), det.get("status"),
                det.get("bbox_x"), det.get("bbox_y"), det.get("bbox_w"), det.get("bbox_h"),
            ]


_VERIFICATION_HEADER = [
    "observation_id", "verified", "verdict_data_source", "rfdetr_version",
    "rfdetr_gbif_usage_key", "rfdetr_scientific_name", "rfdetr_confidence",
    "rfdetr_agrees_with_field", "frames_scored", "frames_in_agreement",
    "salience_authoritative_derived", "rarity_score_derived",
    "baseline_deviation_derived", "anomaly_magnitude_authoritative_derived",
    "verified_at_utc",
]


def _rows_verification(model: ReportModel):
    for rec in model.records:
        v = rec.verification
        if not v:
            continue
        yield [
            v.get("observation_id"), v.get("verified"), "model", v.get("rfdetr_version"),
            v.get("rfdetr_gbif_usage_key"), v.get("rfdetr_scientific_name"), v.get("rfdetr_confidence"),
            v.get("rfdetr_agrees_with_field"), v.get("frames_scored"), v.get("frames_in_agreement"),
            v.get("salience_authoritative"), v.get("rarity_score"),
            v.get("baseline_deviation"), v.get("anomaly_magnitude_authoritative"),
            v.get("verified_at"),
        ]


_AUDIO_HEADER = [
    "observation_id", "event_name", "audio_clip_path", "audio_true_duration_seconds",
    "audio_capped", "acoustic_model_version",
    "detection_id", "gbif_usage_key", "scientific_name", "common_name",
    "confidence", "data_source", "status",
]


def _rows_audio(model: ReportModel):
    for rec in model.records:
        o = rec.observation
        has_clip = bool(o.get("audio_clip_path")) or o.get("trigger_source") == "audio"
        if not rec.audio_detections and not has_clip:
            continue
        base = [
            o.get("id"), o.get("event_name"), o.get("audio_clip_path"),
            o.get("audio_true_duration_seconds"), o.get("audio_capped"), o.get("acoustic_model_version"),
        ]
        if rec.audio_detections:
            for det in rec.audio_detections:
                yield base + [
                    det.get("id"), det.get("gbif_usage_key"), det.get("scientific_name"), det.get("common_name"),
                    det.get("confidence"), det.get("data_source"), det.get("status"),
                ]
        else:
            # An audio event with a clip but no classified taxon still gets a row,
            # so a captured but unresolved sound is never dropped from the record.
            yield base + [None, None, None, None, None, None, None]


_ENVIRONMENT_HEADER = [
    "observation_id", "first_seen_utc", "channel", "value", "unit",
    "data_source", "status", "qartod_flag", "qartod_label",
]


def _rows_environment(model: ReportModel):
    for rec in model.records:
        first_seen = rec.observation.get("first_seen")
        for reading in rec.environment:
            yield [
                reading.get("observation_id"), first_seen, reading.get("channel"),
                reading.get("value"), reading.get("unit"),
                reading.get("data_source"), reading.get("status"),
                reading.get("qartod_flag"), _qartod_label(reading.get("qartod_flag")),
            ]


_INTERPRETATION_HEADER = [
    "observation_id", "point_type", "value", "confidence",
    "data_source", "produced_by", "model_version", "skill_id", "created_at_utc",
]


def _rows_interpretations(model: ReportModel):
    for rec in model.records:
        for interp in rec.interpretations:
            yield [
                interp.get("observation_id"), interp.get("point_type"), interp.get("value"), interp.get("confidence"),
                interp.get("data_source"), interp.get("produced_by"), interp.get("model_version"),
                interp.get("skill_id"), interp.get("created_at"),
            ]


_PATTERN_HEADER = [
    "pattern_id", "framing", "dream_pass_id", "data_source", "status",
    "dream_phase", "pattern_type", "confidence",
    "effect_size", "effect_size_type", "statistic",
    "data_span_start_utc", "data_span_end_utc", "n",
    "p_value", "q_value", "autocorr_adjusted", "model_version", "description",
]


def _rows_patterns(model: ReportModel):
    for prec in model.patterns:
        p = prec.pattern
        yield [
            p.get("id"), "candidate_hypothesis", p.get("dream_pass_id"), p.get("data_source"), p.get("status"),
            p.get("dream_phase"), p.get("pattern_type"), p.get("confidence"),
            p.get("effect_size"), p.get("effect_size_type"), p.get("statistic"),
            p.get("data_span_start"), p.get("data_span_end"), p.get("n"),
            p.get("p_value"), p.get("q_value"), p.get("autocorr_adjusted"), p.get("model_version"), p.get("description"),
        ]


_PATTERN_OBS_HEADER = ["pattern_id", "observation_id"]


def _rows_pattern_obs(model: ReportModel):
    for prec in model.patterns:
        for oid in prec.supporting_observation_ids:
            yield [prec.pattern.get("id"), oid]


_ANALYTICS_HEADER = ["metric", "key", "value", "provenance"]


def _rows_analytics(model: ReportModel):
    a = model.analytics
    note = "derived from the records in this report"
    yield ["total_events", "", a.get("total_events"), note]
    yield ["species_richness", "", a.get("species_richness"), note]
    yield ["verified_count", "", a.get("verified_count"), note]
    yield ["verified_fraction", "", _num(a.get("verified_fraction")), note]
    yield ["earliest_event_utc", "", a.get("earliest_event_utc"), note]
    yield ["latest_event_utc", "", a.get("latest_event_utc"), note]
    for key, value in sorted(a.get("events_by_trigger_source", {}).items()):
        yield ["events_by_trigger_source", key, value, note]
    for key, value in sorted(a.get("events_by_qc_state", {}).items()):
        yield ["events_by_qc_state", key, value, note]
    for key, value in sorted(a.get("detections_by_modality", {}).items()):
        yield ["detections_by_modality", key, value, note]
    names = a.get("taxon_display_names", {})
    for key, value in sorted(a.get("taxon_event_counts", {}).items(), key=lambda kv: (-kv[1], kv[0])):
        label = names.get(key, key)
        yield ["taxon_event_counts", label, value, note]
    for channel, statuses in sorted(a.get("channel_status_counts", {}).items()):
        for status, count in sorted(statuses.items()):
            yield ["channel_status_counts", f"{channel}:{status}", count, note]
    for sid, bucket in sorted(a.get("effort", {}).items()):
        for key, value in sorted(bucket.items()):
            yield ["effort", f"{sid}:{key}", _num(value), "derived from measured telemetry"]


_TELEMETRY_HEADER = [
    "station_id", "recorded_at_utc", "data_source",
    "camera_uptime_seconds", "frames_processed", "frames_dropped",
    "valid_audio_seconds", "npu_active_seconds", "effective_detection_fps",
    "station_temperature_c", "buffer_fill_pct", "sync_lag_seconds",
    "avg_power_w", "cumulative_joules", "error_channels",
]


def _rows_telemetry(model: ReportModel):
    for rec in model.telemetry:
        t = rec.telemetry
        errors = "; ".join(f"{e.get('channel')}={e.get('error_count')}" for e in rec.errors)
        yield [
            t.get("station_id"), t.get("recorded_at"), t.get("data_source"),
            t.get("camera_uptime_seconds"), t.get("frames_processed"), t.get("frames_dropped"),
            t.get("valid_audio_seconds"), t.get("npu_active_seconds"), t.get("effective_detection_fps"),
            t.get("station_temperature_c"), t.get("buffer_fill_pct"), t.get("sync_lag_seconds"),
            t.get("avg_power_w"), t.get("cumulative_joules"), errors,
        ]


_SPECIES_HEADER = [
    "gbif_usage_key", "scientific_name", "common_name", "taxonomic_rank",
    "iucn_status", "iucn_fetch_date", "gbif_occurrence_count", "gbif_snapshot_date",
    "data_source", "fetched_at",
]


def _rows_species(model: ReportModel):
    for sp in model.species_reference:
        yield [
            sp.get("gbif_usage_key"), sp.get("scientific_name"), sp.get("common_name"), sp.get("taxonomic_rank"),
            sp.get("iucn_status"), sp.get("iucn_fetch_date"), sp.get("gbif_occurrence_count"), sp.get("gbif_snapshot_date"),
            sp.get("data_source"), sp.get("fetched_at"),
        ]


_PROVENANCE_HEADER = ["category", "value"]


def _rows_provenance(model: ReportModel):
    p = model.provenance
    for category in _PROVENANCE_ORDER:
        values = p.get(category, [])
        if not values:
            yield [category, ""]
        for value in values:
            yield [category, value]


_PROVENANCE_ORDER = [
    "screening_model_versions", "acoustic_model_versions", "rfdetr_versions",
    "interpretation_model_versions", "pattern_model_versions",
    "gbif_snapshot_dates", "iucn_fetch_dates",
]


# ===========================================================================
# PDF output (fpdf2, imported lazily behind a seam)
# ===========================================================================


def _load_pdf_backend():
    """Import the PDF library on demand.

    Kept out of module import so this file loads, gathers a model, and writes CSV
    with the library absent. A missing library is reported plainly here, at the
    one place it is actually needed.
    """
    try:
        from fpdf import FPDF
        from fpdf.enums import XPos, YPos
    except ImportError as exc:  # pragma: no cover - exercised only without the library
        raise ReportDependencyError(
            "PDF output needs the fpdf2 library, which is not installed. "
            "Install it with: pip install fpdf2  (or choose the CSV format, "
            "which needs no extra library)."
        ) from exc
    return FPDF, XPos, YPos


def _pdf_safe(text: Any) -> str:
    """Make text safe for the built-in PDF fonts.

    The built-in fonts cover the Latin-1 range. Any character outside it is
    replaced so an unusual glyph in a name never aborts a render. The CSV output
    keeps the exact original text, so nothing is lost overall.
    """
    if text is None:
        return ""
    return str(text).encode("latin-1", "replace").decode("latin-1")


# Text colors that make the measured-versus-inferred split visible at a glance.
_COLOR_TEXT = (17, 24, 39)          # near-black: measured facts and headings
_COLOR_INFERRED = (109, 40, 217)    # violet: inferred and candidate values
_COLOR_DERIVED = (75, 85, 99)       # gray: derived summaries
_COLOR_MUTED = (107, 114, 128)      # gray: provenance tags and captions
_COLOR_RULE = (209, 213, 219)       # light gray: separators
_COLOR_BANNER = (180, 83, 9)        # amber: the candidate-hypothesis banner


def write_pdf(model: ReportModel, pdf_path: str | Path, *, charts: Optional[dict] = None) -> Path:
    """Render the report model to a PDF at the given path.

    charts maps a figure name to a PNG path (from audtheia.reports.charts). It is
    optional: with no figures the report still renders, omitting the summary
    figures rather than failing.
    """
    FPDF, XPos, YPos = _load_pdf_backend()
    out_path = Path(pdf_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = _ReportPdf(model, FPDF, XPos, YPos, charts=charts or {})
    doc.build()
    doc.output(str(out_path))
    return out_path


class _ReportPdf:
    """A thin builder around one fpdf2 document.

    fpdf2 types are passed in rather than imported at module load, so this class
    is only ever constructed after the seam has confirmed the library is present.
    """

    def __init__(self, model: ReportModel, FPDF, XPos, YPos, *, charts: Optional[dict] = None) -> None:
        self.model = model
        self.XPos = XPos
        self.YPos = YPos
        self.charts = charts or {}

        class _Doc(FPDF):
            def header(inner) -> None:  # noqa: N805 - fpdf2 calls this
                # The cover page carries its own banner, so the running header is
                # suppressed there and drawn on every content page after it.
                if getattr(inner, "_no_header", False):
                    return
                inner.set_font("Helvetica", "I", 8)
                inner.set_text_color(*_COLOR_MUTED)
                inner.cell(0, 6, _pdf_safe("Audtheia environmental report"),
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="R")
                inner.set_draw_color(*_COLOR_RULE)
                y = inner.get_y()
                inner.line(inner.l_margin, y, inner.w - inner.r_margin, y)
                inner.ln(2)

            def footer(inner) -> None:  # noqa: N805 - fpdf2 calls this
                inner.set_y(-12)
                inner.set_font("Helvetica", "I", 8)
                inner.set_text_color(*_COLOR_MUTED)
                inner.cell(0, 8, _pdf_safe(f"Page {inner.page_no()}"),
                           new_x=XPos.RIGHT, new_y=YPos.TOP, align="C")

        self.pdf = _Doc(orientation="P", unit="mm", format="A4")
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.pdf.set_margins(14, 12, 14)
        self.epw = self.pdf.w - self.pdf.l_margin - self.pdf.r_margin

    # -- low-level text helpers -----------------------------------------

    def _set(self, style: str = "", size: float = 10, color=_COLOR_TEXT) -> None:
        self.pdf.set_font("Helvetica", style, size)
        self.pdf.set_text_color(*color)

    def _title(self, text: str) -> None:
        self._set("B", 20)
        self.pdf.multi_cell(self.epw, 9, _pdf_safe(text),
                            new_x=self.XPos.LMARGIN, new_y=self.YPos.NEXT)
        self.pdf.ln(1)

    def _section(self, text: str) -> None:
        self.pdf.ln(2)
        self._set("B", 13)
        self.pdf.multi_cell(self.epw, 7, _pdf_safe(text),
                            new_x=self.XPos.LMARGIN, new_y=self.YPos.NEXT)
        self.pdf.set_draw_color(*_COLOR_RULE)
        y = self.pdf.get_y()
        self.pdf.line(self.pdf.l_margin, y, self.pdf.w - self.pdf.r_margin, y)
        self.pdf.ln(2)

    def _subhead(self, text: str) -> None:
        self._set("B", 10.5)
        self.pdf.multi_cell(self.epw, 5.5, _pdf_safe(text),
                            new_x=self.XPos.LMARGIN, new_y=self.YPos.NEXT)

    def _para(self, text: str, color=_COLOR_TEXT, style: str = "", size: float = 9.5) -> None:
        self._set(style, size, color)
        self.pdf.multi_cell(self.epw, 5, _pdf_safe(text),
                            new_x=self.XPos.LMARGIN, new_y=self.YPos.NEXT)

    def _kv(self, label: str, value: str, tag: str = "", inferred: bool = False,
            derived: bool = False) -> None:
        """One labeled value line with its provenance tag.

        Inferred and derived values are set apart in style and color, so a
        reader can see at a glance which numbers are measured and which are not.
        """
        label_w = 55
        self._set("B", 9.5, _COLOR_TEXT)
        self.pdf.cell(label_w, 5, _pdf_safe(label),
                      new_x=self.XPos.RIGHT, new_y=self.YPos.TOP)
        if inferred:
            color, style = _COLOR_INFERRED, "I"
        elif derived:
            color, style = _COLOR_DERIVED, "I"
        else:
            color, style = _COLOR_TEXT, ""
        self._set(style, 9.5, color)
        value_text = _pdf_safe(value)
        if tag:
            value_text = value_text + "  " + tag
        self.pdf.multi_cell(self.epw - label_w, 5, value_text,
                            new_x=self.XPos.LMARGIN, new_y=self.YPos.NEXT)

    def _bullet(self, text: str, color=_COLOR_TEXT, style: str = "") -> None:
        self._set(style, 9.5, color)
        self.pdf.cell(4, 5, _pdf_safe("-"), new_x=self.XPos.RIGHT, new_y=self.YPos.TOP)
        self.pdf.multi_cell(self.epw - 4, 5, _pdf_safe(text),
                            new_x=self.XPos.LMARGIN, new_y=self.YPos.NEXT)

    def _gap(self, height: float = 2) -> None:
        self.pdf.ln(height)

    # -- document assembly ----------------------------------------------

    def build(self) -> None:
        self._cover_page()
        self.pdf.add_page()
        self._executive_summary()
        self._reading_guide()
        self._detections_section()
        self._audio_section()
        self._environment_section()
        self._analytics_section()
        self._patterns_section()
        self._provenance_section()

    def output(self, path: str) -> None:
        self.pdf.output(path)

    # -- figures ---------------------------------------------------------

    def _figure(self, name: str, caption: str = "") -> None:
        """Embed one rendered figure at column width, with an optional caption.

        A figure that was not produced (no data, or matplotlib absent) is simply
        skipped, so the summary shows only figures that carry meaning.
        """
        path = self.charts.get(name)
        if not path:
            return
        try:
            self.pdf.image(str(path), w=self.epw)
        except Exception:  # noqa: BLE001 - a bad image never breaks the report
            return
        if caption:
            self._set("I", 8, _COLOR_MUTED)
            self.pdf.multi_cell(self.epw, 4.5, _pdf_safe(caption),
                                new_x=self.XPos.LMARGIN, new_y=self.YPos.NEXT)
        self._gap(2)

    # -- cover page ------------------------------------------------------

    def _cover_page(self) -> None:
        """A dedicated title page: wordmark, title, scope, window, timestamp."""
        m = self.model
        self.pdf._no_header = True
        self.pdf.add_page()
        self.pdf._no_header = False

        # A full-width banner band at the top, in the product's amber.
        band_h = 46
        self.pdf.set_fill_color(*_COLOR_BANNER)
        self.pdf.rect(0, 0, self.pdf.w, band_h, style="F")
        self.pdf.set_xy(self.pdf.l_margin, 16)
        self._set("B", 12, (255, 255, 255))
        self.pdf.cell(0, 7, _pdf_safe("AUDTHEIA"),
                      new_x=self.XPos.LMARGIN, new_y=self.YPos.NEXT)
        self.pdf.set_x(self.pdf.l_margin)
        self._set("", 10, (255, 255, 255))
        self.pdf.cell(0, 6, _pdf_safe("Offline environmental intelligence"),
                      new_x=self.XPos.LMARGIN, new_y=self.YPos.NEXT)

        # The title and scope, below the band.
        self.pdf.set_xy(self.pdf.l_margin, band_h + 22)
        self._set("B", 26, _COLOR_TEXT)
        self.pdf.multi_cell(self.epw, 11, _pdf_safe("Environmental monitoring report"),
                            new_x=self.XPos.LMARGIN, new_y=self.YPos.NEXT)
        self._gap(4)
        self._set("B", 13, _COLOR_TEXT)
        self.pdf.multi_cell(self.epw, 7, _pdf_safe(m.scope_label),
                            new_x=self.XPos.LMARGIN, new_y=self.YPos.NEXT)
        self._gap(2)
        self._set("", 10.5, _COLOR_MUTED)
        self.pdf.multi_cell(self.epw, 6, _pdf_safe(_describe_window(m)),
                            new_x=self.XPos.LMARGIN, new_y=self.YPos.NEXT)

        # A footer block near the bottom of the cover with the generation stamp.
        self.pdf.set_xy(self.pdf.l_margin, self.pdf.h - 40)
        self.pdf.set_draw_color(*_COLOR_RULE)
        self.pdf.line(self.pdf.l_margin, self.pdf.get_y(), self.pdf.w - self.pdf.r_margin, self.pdf.get_y())
        self._gap(3)
        gen_local = _format_local(m.generated_at_utc, _display_tz(m))
        self._set("", 9, _COLOR_MUTED)
        self.pdf.multi_cell(self.epw, 5, _pdf_safe(f"Generated {gen_local}"),
                            new_x=self.XPos.LMARGIN, new_y=self.YPos.NEXT)
        self.pdf.multi_cell(self.epw, 5,
                            _pdf_safe(f"Times shown in {m.display_timezone}; stored data is UTC. "
                                      "Generated offline on the desktop from the local record."),
                            new_x=self.XPos.LMARGIN, new_y=self.YPos.NEXT)

    # -- executive summary ----------------------------------------------

    def _executive_summary(self) -> None:
        """A plain-language overview and the key figures, for any reader."""
        self._section("Executive summary")
        self._para(_summary_narrative(self.model))
        self._gap(2)
        self._figure("detection_timeline", "Events recorded across the reporting window.")
        self._figure("species_composition", "The taxa recorded, ranked by how many events each appeared in.")
        self._figure("confidence_distribution",
                     "How confident the detection models were, across every detection in scope.")
        self._figure("verification_summary",
                     "How many events cleared the desktop verification gate. A field call and the "
                     "desktop verdict are kept as separate records; verification never rewrites a measurement.")

    def _reading_guide(self) -> None:
        m = self.model
        self._section("How to read the labels")
        self._para(
            "Every value in the detailed sections carries a bracketed tag naming "
            "where it came from and its quality-control or missing-data status. "
            "Measured values are set in plain text; inferred values from a language "
            "model and candidate results from the longitudinal pass are set in "
            "violet italics; derived summaries are set in gray italics.")
        self._gap(1)
        self._bullet("measured (sensor): a direct sensor reading.")
        self._bullet("measured (detection model): a call from a detection model on captured media.")
        self._bullet("inferred (language model): a downstream interpretation, not a measurement.",
                     color=_COLOR_INFERRED, style="I")
        self._bullet("candidate (longitudinal pass): a hypothesis, never an established finding.",
                     color=_COLOR_INFERRED, style="I")
        self._bullet("derived: a summary computed from the records in this report.",
                     color=_COLOR_DERIVED, style="I")
        self._gap(2)

        a = m.analytics
        self._subhead("At a glance")
        self._kv("Stations in scope", str(len(m.stations)), derived=True)
        self._kv("Events", str(a.get("total_events", 0)), derived=True)
        self._kv("Distinct taxa", str(a.get("species_richness", 0)), derived=True)
        self._kv("Verified events",
                 f"{a.get('verified_count', 0)} of {a.get('total_events', 0)}"
                 f" ({_num(a.get('verified_fraction'))})", derived=True)
        self._kv("Candidate hypotheses", str(len(m.patterns)), inferred=True)

    def _detections_section(self) -> None:
        self._section("Detections and verification")
        vision_records = [r for r in self.model.records if r.vision_detections or r.verification]
        if not vision_records:
            self._para("No visual detections fall in this report's scope.", color=_COLOR_MUTED)
            return
        tz = _display_tz(self.model)
        for rec in vision_records:
            self._observation_header(rec, tz)
            for det in rec.vision_detections:
                name = _taxon_name(det)
                tag = _provenance_tag(det.get("data_source"), det.get("status"))
                self._kv(
                    "Taxon",
                    f"{name} (confidence {_num(det.get('confidence'))})",
                    tag=tag,
                )
            self._verification_block(rec)
            self._interpretation_block(rec)
            self._rule()

    def _audio_section(self) -> None:
        self._section("Audio")
        audio_records = [
            r for r in self.model.records
            if r.audio_detections or r.observation.get("audio_clip_path")
            or r.observation.get("trigger_source") == "audio"
        ]
        if not audio_records:
            self._para("No acoustic captures fall in this report's scope.", color=_COLOR_MUTED)
            return
        tz = _display_tz(self.model)
        for rec in audio_records:
            o = rec.observation
            self._observation_header(rec, tz)
            self._kv("Clip", o.get("audio_clip_path") or "no clip stored",
                     tag=_provenance_tag("sensor", "measured" if o.get("audio_clip_path") else "not_measured"))
            self._kv("True duration (s)", _num(o.get("audio_true_duration_seconds")),
                     tag=_provenance_tag("sensor"))
            capped = o.get("audio_capped")
            if capped is not None:
                self._kv("Clip capped", "yes" if capped else "no", tag=_provenance_tag("sensor"))
            self._kv("Acoustic model", o.get("acoustic_model_version") or "not recorded",
                     tag="[model version]")
            for det in rec.audio_detections:
                tag = _provenance_tag(det.get("data_source"), det.get("status"))
                self._kv("Sound identified as",
                         f"{_taxon_name(det)} (confidence {_num(det.get('confidence'))})", tag=tag)
            if not rec.audio_detections:
                self._para("Captured sound was not resolved to a taxon.", color=_COLOR_MUTED, style="I")
            self._rule()

    def _environment_section(self) -> None:
        self._section("Environment")
        env_records = [r for r in self.model.records if r.environment]
        if not env_records:
            self._para("No environmental readings fall in this report's scope.", color=_COLOR_MUTED)
            return
        tz = _display_tz(self.model)
        for rec in env_records:
            self._observation_header(rec, tz)
            for reading in rec.environment:
                unit = reading.get("unit") or ""
                value = _num(reading.get("value"))
                display = f"{value} {unit}".strip()
                tag = _provenance_tag(
                    reading.get("data_source"), reading.get("status"), reading.get("qartod_flag")
                )
                self._kv(reading.get("channel") or "channel", display, tag=tag)
            self._rule()

    def _analytics_section(self) -> None:
        self._section("Analytics")
        a = self.model.analytics
        self._para(
            "These are summaries computed from the records in this report, not "
            "new measurements. Detection counts are raw; an effort-normalized "
            "rate is not asserted here.", color=_COLOR_MUTED)
        self._gap(1)
        self._kv("Total events", str(a.get("total_events", 0)), derived=True)
        self._kv("Distinct taxa (richness)", str(a.get("species_richness", 0)), derived=True)
        span = f"{_format_local(a.get('earliest_event_utc'), _display_tz(self.model))}" \
               f"  to  {_format_local(a.get('latest_event_utc'), _display_tz(self.model))}"
        self._kv("Event time span", span, derived=True)

        self._gap(1)
        self._subhead("Events by trigger source")
        for key, value in sorted(a.get("events_by_trigger_source", {}).items()):
            self._bullet(f"{key}: {value}", color=_COLOR_DERIVED, style="I")

        self._subhead("Detections by modality")
        for key, value in sorted(a.get("detections_by_modality", {}).items()):
            self._bullet(f"{key}: {value}", color=_COLOR_DERIVED, style="I")

        names = a.get("taxon_display_names", {})
        counts = a.get("taxon_event_counts", {})
        if counts:
            self._subhead("Events per taxon")
            for key, value in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
                self._bullet(f"{names.get(key, key)}: {value}", color=_COLOR_DERIVED, style="I")

        channels = a.get("channel_status_counts", {})
        if channels:
            self._subhead("Environmental channel coverage (readings by status)")
            for channel, statuses in sorted(channels.items()):
                summary = ", ".join(f"{_status_label(s)}: {c}" for s, c in sorted(statuses.items()))
                self._bullet(f"{channel}: {summary}", color=_COLOR_DERIVED, style="I")

        effort = a.get("effort", {})
        if effort:
            self._subhead("Effort context (summed from telemetry)")
            for sid, bucket in sorted(effort.items()):
                station_name = self._station_name(sid)
                self._bullet(
                    f"{station_name}: {int(bucket.get('heartbeats', 0))} heartbeats, "
                    f"camera uptime {_num(bucket.get('camera_uptime_seconds'))} s, "
                    f"frames processed {_num(bucket.get('frames_processed'))}, "
                    f"NPU active {_num(bucket.get('npu_active_seconds'))} s",
                    color=_COLOR_DERIVED, style="I")

    def _patterns_section(self) -> None:
        self._section("Candidate hypotheses from the longitudinal pass")
        if not self.model.patterns:
            self._para("The longitudinal pass produced no candidate patterns in scope.",
                       color=_COLOR_MUTED)
            return
        self._set("B", 9.5, _COLOR_BANNER)
        self.pdf.multi_cell(self.epw, 5, _pdf_safe(
            "Every item in this section is a candidate hypothesis. None is an "
            "established finding. Each rests only on the events listed with it."),
            new_x=self.XPos.LMARGIN, new_y=self.YPos.NEXT)
        self._gap(2)
        tz = _display_tz(self.model)
        for prec in self.model.patterns:
            self._pattern_card(prec, tz)
            self._rule()

    def _pattern_card(self, prec: PatternRecord, tz: tzinfo) -> None:
        p = prec.pattern
        self._set("B", 9, _COLOR_BANNER)
        self.pdf.multi_cell(self.epw, 5, _pdf_safe(HYPOTHESIS_BANNER),
                            new_x=self.XPos.LMARGIN, new_y=self.YPos.NEXT)
        ptype = PATTERN_TYPE_LABELS.get(p.get("pattern_type"), p.get("pattern_type") or "pattern")
        self._subhead(_pdf_safe(ptype))
        self._para(p.get("description") or "", color=_COLOR_INFERRED, style="I")

        effect_type = EFFECT_SIZE_TYPE_LABELS.get(p.get("effect_size_type"), p.get("effect_size_type") or "unspecified")
        self._kv("Effect size", f"{_num(p.get('effect_size'))} ({effect_type})",
                 tag="[candidate (longitudinal pass)]", inferred=True)
        self._kv("Test / statistic", p.get("statistic") or "not recorded", inferred=True)
        span = f"{_format_local(p.get('data_span_start'), tz)}  to  {_format_local(p.get('data_span_end'), tz)}"
        self._kv("Data span", span, inferred=True)
        self._kv("Supporting events (n)", str(p.get("n")), inferred=True)
        self._kv("p-value", _num(p.get("p_value")), inferred=True)
        self._kv("q-value (FDR-adjusted)", _num(p.get("q_value")), inferred=True)
        self._kv("Autocorrelation adjusted", _yes_no(p.get("autocorr_adjusted")), inferred=True)
        self._kv("Confidence", _num(p.get("confidence")), inferred=True)
        self._kv("Model version", p.get("model_version") or "not recorded", inferred=True)
        if prec.dream_pass:
            phase = DREAM_PHASE_LABELS.get(prec.pattern.get("dream_phase"), prec.pattern.get("dream_phase") or "")
            self._kv("Pass phase", phase, inferred=True)
        supporting = prec.supporting_observation_ids
        if supporting:
            preview = ", ".join(supporting[:8])
            if len(supporting) > 8:
                preview += f", and {len(supporting) - 8} more"
            self._kv("Traces to events", preview, inferred=True)

    def _provenance_section(self) -> None:
        self._section("Provenance and data age")
        p = self.model.provenance
        self._para(
            "The versions and snapshot dates behind the results above. Taxonomic "
            "and conservation fields are only as current as the dates shown.",
            color=_COLOR_MUTED)
        self._gap(1)
        self._provenance_line("Screening model versions", p.get("screening_model_versions"))
        self._provenance_line("Acoustic model versions", p.get("acoustic_model_versions"))
        self._provenance_line("Verification (RF-DETR) versions", p.get("rfdetr_versions"))
        self._provenance_line("Interpretation model versions", p.get("interpretation_model_versions"))
        self._provenance_line("Longitudinal pass model versions", p.get("pattern_model_versions"))
        self._provenance_line("Taxonomic backbone snapshot dates", p.get("gbif_snapshot_dates"))
        self._provenance_line("Conservation-status fetch dates", p.get("iucn_fetch_dates"))

        if self.model.species_reference:
            self._gap(2)
            self._subhead("Species reference (cached taxonomy and conservation status)")
            for sp in self.model.species_reference:
                name = sp.get("scientific_name") or sp.get("gbif_usage_key") or "unknown"
                status = sp.get("iucn_status") or "no status"
                self._kv(
                    name,
                    f"IUCN {status}; taxonomy snapshot {sp.get('gbif_snapshot_date') or 'not recorded'}; "
                    f"status fetched {sp.get('iucn_fetch_date') or 'not recorded'}",
                    tag=_provenance_tag("database"))

    def _provenance_line(self, label: str, values) -> None:
        text = ", ".join(values) if values else "not recorded"
        self._kv(label, text, tag="[reference]", derived=True)

    # -- shared record rendering ----------------------------------------

    def _observation_header(self, rec: ObservationRecord, tz: tzinfo) -> None:
        o = rec.observation
        self._subhead(o.get("event_name") or o.get("id") or "event")
        station = self._station_name(o.get("station_id"))
        when = _format_local(o.get("first_seen"), tz)
        self._para(
            f"Station {station}  |  {when}  |  trigger: {o.get('trigger_source')}  |  "
            f"duration {_num(o.get('duration'))} s",
            color=_COLOR_MUTED, size=8.5)
        qc = QC_STATE_LABELS.get(o.get("qc_state"), o.get("qc_state") or "")
        reason = o.get("qc_reason")
        qc_text = qc + (f" ({reason})" if reason else "")
        provisional = "  time provisional (pre-fix clock)" if o.get("time_provisional") else ""
        self._para(f"Record status: {qc_text}{provisional}", color=_COLOR_MUTED, size=8.5)
        gps_status = o.get("gps_status")
        if o.get("gps_latitude") is not None or gps_status:
            loc = f"{_num(o.get('gps_latitude'), 5)}, {_num(o.get('gps_longitude'), 5)}"
            self._kv("Location", loc, tag=_provenance_tag("sensor", gps_status))
        screening = o.get("screening_model_version")
        if screening:
            self._para(f"Screening model: {screening}", color=_COLOR_MUTED, size=8.5)

    def _verification_block(self, rec: ObservationRecord) -> None:
        v = rec.verification
        if not v:
            self._para("Not yet verified on the desktop.", color=_COLOR_MUTED, style="I", size=9)
            return
        verified = "verified" if v.get("verified") else "not verified"
        agrees = v.get("rfdetr_agrees_with_field")
        agree_text = "" if agrees is None else (", agrees with field" if agrees else ", disagrees with field")
        resolved = v.get("rfdetr_scientific_name") or v.get("rfdetr_gbif_usage_key") or "no taxon resolved"
        self._kv(
            "Verification",
            f"{verified}{agree_text}: {resolved} (confidence {_num(v.get('rfdetr_confidence'))}, "
            f"{_num(v.get('frames_in_agreement'))} of {_num(v.get('frames_scored'))} frames)",
            tag=_provenance_tag("model"))
        if v.get("rfdetr_version"):
            self._para(f"Verifier model: {v.get('rfdetr_version')}", color=_COLOR_MUTED, size=8.5)
        if v.get("salience_authoritative") is not None:
            self._kv("Salience (authoritative)", _num(v.get("salience_authoritative")), derived=True)

    def _interpretation_block(self, rec: ObservationRecord) -> None:
        for interp in rec.interpretations:
            label = interp.get("point_type") or "interpretation"
            tag = _provenance_tag(interp.get("data_source"))
            by = interp.get("produced_by")
            suffix = f" (by {by})" if by else ""
            self._kv(
                label.replace("_", " "),
                f"{interp.get('value')}{suffix}",
                tag=tag, inferred=True)

    def _rule(self) -> None:
        self._gap(1)
        self.pdf.set_draw_color(*_COLOR_RULE)
        y = self.pdf.get_y()
        self.pdf.line(self.pdf.l_margin, y, self.pdf.w - self.pdf.r_margin, y)
        self._gap(2)

    def _station_name(self, station_id: Optional[str]) -> str:
        for s in self.model.stations:
            if s.get("id") == station_id:
                return s.get("station_name") or station_id or "unknown"
        return station_id or "unknown"


# ===========================================================================
# Small module-level helpers used by both renderers
# ===========================================================================


def _display_tz(model: ReportModel) -> tzinfo:
    """Reconstruct a usable tzinfo for display from the model's zone name.

    The gather step recorded the zone's short name for the header. For value
    formatting the renderers need a real tzinfo, so this resolves one, falling
    back to UTC if the name cannot be resolved on this host.
    """
    name = model.display_timezone
    if not name:
        return timezone.utc
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:  # noqa: BLE001 - an abbreviation is not always resolvable
        # The stamp strings already carry their own offset when parsed, so UTC
        # here only affects the trailing label, never the instant shown.
        return timezone.utc


def _describe_window(model: ReportModel) -> str:
    start = model.window_start_utc
    end = model.window_end_utc
    if not start and not end:
        return "Window: all available records"
    tz = _display_tz(model)
    left = _format_local(start, tz) if start else "the earliest record"
    right = _format_local(end, tz) if end else "the latest record"
    return f"Window: {left}  to  {right}"


def _yes_no(value) -> str:
    if value is None:
        return "not recorded"
    return "yes" if value else "no"


def _summary_narrative(model: ReportModel) -> str:
    """A plain-language overview of the record, built only from counted facts.

    Every figure comes straight from the analytics already computed over the
    stored rows, so the paragraph never states anything the tables do not. An
    empty record is described plainly rather than dressed up.
    """
    a = model.analytics or {}
    total = int(a.get("total_events") or 0)
    if total == 0:
        return ("No events fall within this report's scope and window, so there is "
                "nothing to summarize yet. Once detections are captured and "
                "quality-controlled, this summary fills in automatically.")

    modality = a.get("detections_by_modality") or {}
    vision = int(modality.get("vision") or 0)
    audio = int(modality.get("audio") or 0)
    richness = int(a.get("species_richness") or 0)
    verified = int(a.get("verified_count") or 0)
    pct = int(round(100 * (a.get("verified_fraction") or 0)))

    names = a.get("taxon_display_names") or {}
    counts = a.get("taxon_event_counts") or {}
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
    top_names = [str(names.get(k, k)) for k, _ in top]

    event_word = "event" if total == 1 else "events"
    taxa_word = "taxon" if richness == 1 else "taxa"
    parts = [
        f"Across this window, Audtheia recorded {total} {event_word} at {model.scope_label}, "
        f"comprising {vision} visual and {audio} acoustic detections spanning {richness} {taxa_word}."
    ]
    if verified:
        parts.append(f"{verified} of {total} {event_word} ({pct}%) cleared desktop verification.")
    else:
        parts.append("No event has cleared desktop verification yet, so the record so far rests on "
                     "the field station's own calls.")
    if top_names:
        joined = ", ".join(top_names[:-1]) + (", and " + top_names[-1] if len(top_names) > 1 else top_names[0])
        parts.append(f"The most frequently recorded were {joined}.")

    n_patterns = len(model.patterns or [])
    if n_patterns:
        p_word = "pattern" if n_patterns == 1 else "patterns"
        parts.append(f"The longitudinal pass proposed {n_patterns} candidate {p_word}, each a hypothesis "
                     "for further study rather than an established finding.")
    else:
        parts.append("The longitudinal pass proposed no candidate patterns in this window; with more "
                     "accumulated events it surfaces trends, correlations, and co-occurrences.")
    return " ".join(parts)


# ===========================================================================
# Convenience entry point
# ===========================================================================


def generate_report(
    settings,
    database,
    *,
    station_id: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    formats: Optional[Iterable[str]] = None,
    output_dir: Optional[str | Path] = None,
    generated_at: Optional[str] = None,
) -> ReportResult:
    """Build a report in one call.

    A thin wrapper around ReportGenerator for callers (a scheduler or the web
    backend) that hold a settings object and a database handle and want the
    outputs written in one step.
    """
    return ReportGenerator(settings, database).generate(
        station_id=station_id, start=start, end=end,
        formats=formats, output_dir=output_dir, generated_at=generated_at,
    )
