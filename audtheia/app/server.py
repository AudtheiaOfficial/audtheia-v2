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


class BackendError(RuntimeError):
    """A backend operation failed for a reason the operator should see."""


class BackendDependencyError(BackendError):
    """The web framework needed to build or run the app is not installed."""


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
                   until: str | None = Query(default=None), limit: int = Query(default=100, ge=1, le=1000)):
        rows = db.list_observations(station_id=station_id, since=since, until=until, limit=limit)
        out = []
        for obs in rows:
            item = dict(obs)
            item["vision_detections"] = [c for c in db.list_child_detections(obs["id"]) if c.get("modality") == "vision"]
            item["verification"] = db.get_observation_verification(obs["id"])
            out.append(item)
        return out

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

    # -- settings (read-only view; secrets never leave the desktop) -------

    @app.get(f"{API_PREFIX}/settings")
    def get_settings():
        return {
            "config": _redact(settings.raw),
            "secrets_configured": bool(getattr(settings, "secrets", None)),
            "note": "read-only view; editing configuration is a separate, guarded operation",
        }

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
