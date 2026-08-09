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
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
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


# The three-way correction vocabulary, matching the CHECK constraint on
# observation_corrections. Kept here so a request is refused with a clear
# message before it reaches the database rather than as a constraint failure.
_CORRECTION_VERDICTS = ("confirm", "relabel", "reject")

# The per-frame review vocabulary, mirrored here so a request is refused with a
# clear message before it reaches the frame_review CHECK constraint. 'cleared'
# retracts an earlier verdict, returning a frame to unreviewed.
_FRAME_REVIEW_VERDICTS = ("accurate", "inaccurate", "cleared")

# Until the application has user accounts, every correction is attributed to a
# single generic expert. The column is NOT NULL because an anonymous
# identification is not reviewable, and this is the honest placeholder rather
# than a fabricated identity.
_DEFAULT_CORRECTOR = "expert"

# The prebuilt taxonomic index, written beside the shipped backbone by
# scripts/build_gbif_index.py.
_GBIF_INDEX_FILENAME = "index.db"

# A correction search box is read at a glance, so more rows than this would be
# scrolled past rather than considered.
_SPECIES_SEARCH_LIMIT = 20

# The species-data setup steps (building the taxonomic index, fetching per-species
# reference data) run for minutes, so a request cannot wait for them. Each runs in
# a background worker whose progress is held in this in-process registry and read
# back by a status endpoint the interface polls. The registry is process-local and
# the app is a single local process, so a plain lock is all the coordination
# needed. Nothing here is persisted: a job's state lives only for the run.
_jobs: dict = {}
_jobs_lock = threading.Lock()

# A test seam for the reference fetch, so the online GBIF and IUCN calls can be
# replaced by a stand-in client in a check. None in normal use, where the fetch
# script makes real calls under the user's own credentials.
_species_fetch_client_factory = None


def _job_snapshot(name: str) -> dict:
    """The current state of one background job, or an idle marker."""
    with _jobs_lock:
        job = _jobs.get(name)
        return dict(job) if job else {"status": "idle"}


def _start_background_job(name: str, target) -> bool:
    """Start one named background job unless it is already running.

    `target` is called in a worker thread with a single `update(**fields)`
    callback it uses to report progress into the job's state. A clean return
    marks the job done and stores its result; a raised exception marks it failed
    and stores the message, so the interface can show either without the request
    that started it having waited.
    """
    with _jobs_lock:
        existing = _jobs.get(name)
        if existing and existing.get("status") == "running":
            return False
        _jobs[name] = {
            "status": "running",
            "started_at": _utc_now_iso(),
            "finished_at": None,
            "message": "starting",
            "progress": None,
            "result": None,
            "error": None,
        }

    def _update(**fields) -> None:
        with _jobs_lock:
            if name in _jobs:
                _jobs[name].update(fields)

    def _run() -> None:
        try:
            result = target(_update)
            _update(status="done", finished_at=_utc_now_iso(), progress=1.0,
                    message="finished", result=result)
        except Exception as exc:  # noqa: BLE001 - reported to the interface as a failed job
            _update(status="error", finished_at=_utc_now_iso(), error=str(exc))

    threading.Thread(target=_run, name="audtheia-job-" + name, daemon=True).start()
    return True


def _load_script_module(settings, module_name: str):
    """Import one script from the repository's scripts directory by file path.

    The setup scripts are plain modules rather than an installed package, so this
    loads one by its path. The scripts directory ships alongside the audtheia
    package, so it is found relative to this module first (audtheia/app -> the
    repository root -> scripts), which holds whether the app runs from the source
    tree or a copy and does not depend on the scripts directory being on
    sys.path. The configured repository root is a fallback.
    """
    candidates = [
        Path(__file__).resolve().parents[2] / "scripts" / (module_name + ".py"),
        Path(settings.repo_root) / "scripts" / (module_name + ".py"),
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"the setup script {module_name}.py was not found (looked in {candidates[0].parent})")
    spec = importlib.util.spec_from_file_location("audtheia_script_" + module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _backbone_file(settings) -> Path:
    """The shipped GBIF backbone export the taxonomic index is built from."""
    return Path(settings.path("gbif_backbone_path")) / "simple.txt"


def _index_name_count(index_path: Path) -> Optional[int]:
    """How many names a built index holds, or None if it cannot be read."""
    if not index_path.is_file():
        return None
    try:
        conn = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
        try:
            return conn.execute("SELECT COUNT(*) FROM taxon_index").fetchone()[0]
        finally:
            conn.close()
    except sqlite3.Error:
        return None


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


def _gbif_index_path(settings) -> Path:
    """Where the prebuilt taxonomic index lives, following the configured backbone."""
    return Path(settings.path("gbif_backbone_path")) / _GBIF_INDEX_FILENAME


def _resolve_usage_key(settings, usage_key: str) -> Optional[dict]:
    """Resolve a usage key to the accepted taxon it names, or None if unknown.

    A key that the interface offered can still be a synonym, so this follows
    accepted_key to the taxon the record should actually carry. That means an
    expert can search under the name they know and the database still ends up
    with one name per organism rather than a scatter of historical ones.

    Returns None rather than raising for an unknown key, so the caller decides
    whether that is a client error or a missing index.
    """
    index_path = _gbif_index_path(settings)
    if not index_path.is_file():
        return None
    try:
        conn = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT usage_key, canonical_name, scientific_name, status, accepted_key "
            "FROM taxon_index WHERE usage_key = ?",
            (str(usage_key),),
        ).fetchone()
        if row is None:
            return None
        if row["status"] != "ACCEPTED" and row["accepted_key"]:
            accepted = conn.execute(
                "SELECT usage_key, canonical_name, scientific_name FROM taxon_index "
                "WHERE usage_key = ?",
                (row["accepted_key"],),
            ).fetchone()
            if accepted is not None:
                row = accepted
    except sqlite3.Error:
        return None
    finally:
        conn.close()

    # The snapshot the name resolved against, so a later backbone revision can
    # be told apart from the taxonomy in force when the call was made.
    try:
        snapshot = datetime.fromtimestamp(
            index_path.stat().st_mtime, tz=timezone.utc
        ).strftime("%Y-%m-%d")
    except OSError:
        snapshot = None

    return {
        "usage_key": row["usage_key"],
        "scientific_name": row["canonical_name"],
        "snapshot_date": snapshot,
    }


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


_CONF_BANDS = ("0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0")


def _compute_audit(db, *, station_id, since, until) -> dict:
    """Derive an audit view of the record.

    Four questions a reviewer will ask, answered from stored rows only: how often
    the desktop verifier agreed with the field station's call, what quality
    control did and why it deferred anything, how confident the models were, and
    which model and data-snapshot versions produced the record.

    Every figure here is a count or an average over rows that were already
    written. Nothing is a new measurement and nothing is inferred, so these
    numbers can be quoted directly as evidence of how the system behaved.
    """
    observations = db.list_observations(station_id=station_id, since=since, until=until)
    total = len(observations)

    verification = {
        "with_verdict": 0, "verified": 0, "agree": 0, "disagree": 0,
        "not_comparable": 0, "frames_scored": 0, "frames_in_agreement": 0,
    }
    verifier_confidence: list = []
    by_state: dict = {}
    by_reason: dict = {}
    confidence: dict = {"vision": [], "audio": []}
    versions: dict = {
        "screening_model_version": {}, "acoustic_model_version": {},
        "rfdetr_version": {}, "gbif_snapshot_date": {}, "iucn_fetch_date": {},
    }
    periods: dict = {}

    # Expert corrections are a separate, human-sourced producer and are counted on
    # their own line, never folded into the automated verifier's numbers, because a
    # human confirmation and a machine re-score are different claims with different
    # provenance. "targets" counts distinct corrected targets (a whole event, or one
    # box inside a multi-taxon event); "observations_with_correction" counts events
    # carrying any expert verdict. "available" is False only when the database
    # predates the corrections table, which is a database waiting to be migrated
    # rather than a fault, and reads as "no corrections yet".
    expert: dict = {
        "observations_with_correction": 0, "confirm": 0, "relabel": 0,
        "reject": 0, "targets": 0, "available": True,
    }

    def _bump(bucket: dict, value) -> None:
        bucket_key = value if value not in (None, "") else "not stated"
        bucket[bucket_key] = bucket.get(bucket_key, 0) + 1

    for obs in observations:
        by_state[obs.get("qc_state") or "unknown"] = by_state.get(obs.get("qc_state") or "unknown", 0) + 1
        if obs.get("qc_reason"):
            by_reason[obs["qc_reason"]] = by_reason.get(obs["qc_reason"], 0) + 1
        for key in ("screening_model_version", "acoustic_model_version", "gbif_snapshot_date", "iucn_fetch_date"):
            _bump(versions[key], obs.get(key))

        # Grouped by calendar month, which is coarse enough to be readable and
        # fine enough to show a drift in confidence over a deployment.
        period = (obs.get("first_seen") or "")[:7] or "unknown"
        cell = periods.setdefault(period, {"period": period, "events": 0, "verified": 0, "sum": 0.0, "n": 0})
        cell["events"] += 1

        verdict = db.get_observation_verification(obs["id"])
        if verdict:
            verification["with_verdict"] += 1
            if verdict.get("verified"):
                verification["verified"] += 1
                cell["verified"] += 1
            agrees = verdict.get("rfdetr_agrees_with_field")
            if agrees == 1:
                verification["agree"] += 1
            elif agrees == 0:
                verification["disagree"] += 1
            else:
                # No field label to compare against, for example a pure audio event.
                verification["not_comparable"] += 1
            verification["frames_scored"] += int(verdict.get("frames_scored") or 0)
            verification["frames_in_agreement"] += int(verdict.get("frames_in_agreement") or 0)
            if verdict.get("rfdetr_confidence") is not None:
                verifier_confidence.append(float(verdict["rfdetr_confidence"]))
            _bump(versions["rfdetr_version"], verdict.get("rfdetr_version"))

        for det in db.list_child_detections(obs["id"]):
            value = det.get("confidence")
            if value is None:
                continue
            modality = det.get("modality") or "vision"
            confidence.setdefault(modality, []).append(float(value))
            cell["sum"] += float(value)
            cell["n"] += 1

        # The current expert position on each corrected target in this event. Rows
        # come back newest first, so the first verdict seen for a target is the one
        # that currently stands; a corrector who changed their mind is counted once.
        if expert["available"]:
            try:
                rows = db.corrections_for_observation(obs["id"])
            except sqlite3.OperationalError:
                expert["available"] = False
                rows = []
            latest_by_target: dict = {}
            for row in rows:
                target = row.get("detection_id")
                if target not in latest_by_target:
                    latest_by_target[target] = row.get("verdict")
            if latest_by_target:
                expert["observations_with_correction"] += 1
                for verdict in latest_by_target.values():
                    if verdict in ("confirm", "relabel", "reject"):
                        expert[verdict] += 1
                        expert["targets"] += 1

    def _stats(values: list) -> dict:
        if not values:
            return {"n": 0, "mean": None, "min": None, "max": None}
        return {"n": len(values), "mean": sum(values) / len(values), "min": min(values), "max": max(values)}

    # A coarse histogram, so the shape of the distribution is visible and not
    # only its average: a mean of 0.7 built from many 0.4s and 0.95s is a very
    # different picture from one built from consistent 0.7s.
    buckets = {band: 0 for band in _CONF_BANDS}
    for values in confidence.values():
        for value in values:
            buckets[_CONF_BANDS[min(int(value * 5), 4)]] += 1

    verification["frame_agreement_fraction"] = (
        verification["frames_in_agreement"] / verification["frames_scored"]
        if verification["frames_scored"] else None
    )
    verification["mean_verifier_confidence"] = (
        sum(verifier_confidence) / len(verifier_confidence) if verifier_confidence else None
    )

    trend = []
    for key in sorted(periods):
        cell = periods[key]
        trend.append({
            "period": cell["period"], "events": cell["events"], "verified": cell["verified"],
            "mean_confidence": (cell["sum"] / cell["n"]) if cell["n"] else None,
        })

    return {
        "provenance": "derived",
        "note": "counts and averages over rows already stored; not a new measurement and not an inference",
        "window": {"station_id": station_id, "since": since, "until": until},
        "events": total,
        "verification": verification,
        "expert": expert,
        "qc": {"by_state": by_state, "by_reason": by_reason},
        "confidence": {"by_modality": {k: _stats(v) for k, v in confidence.items()}, "buckets": buckets},
        "versions": versions,
        "trend": trend,
        # The inferred model-trust layer: per-species accuracy and per-model
        # rollups. Computed over the whole expert-reviewed record and clearly
        # labelled inferred and cumulative, so it stands apart from the windowed,
        # measured counts above and is never mistaken for one of them.
        "model_trust": db.model_trust_snapshot(),
    }


def _event_trust(obs: dict, children: list, index: dict) -> dict:
    """Event Trust for one observation, an inferred value, or an honest absence.

    Event Trust is ``D * Acc(s, M)``: how strongly the event was detected, times
    how reliably the model that produced the headline call gets that species
    right. ``D`` is salience's exact detection evidence, ``1 - (1 - C)(1 - A)``,
    reused here so this never re-derives or alters it. ``C`` is the strongest
    visual confidence in the event and ``A`` the strongest acoustic one, each 0
    when that channel did not fire, so a single-modality event degrades to the
    live channel and a corroborated one is scored higher.

    The species and model are taken from the event's highest-confidence
    identified detection, which is the call a card headlines and a reviewer
    judges; its modality selects the screening or acoustic model version. When
    the species has no expert reviews under that model, ``Acc`` is not
    computable, so Event Trust is returned as not computable with its reason
    rather than as a fabricated number. Every value carries the model tag and the
    inference provenance.
    """
    from audtheia.analysis.model_trust import event_trust as _event_trust_value
    from audtheia.pipeline.salience import detection_evidence as _detection_evidence

    def _strongest(dets) -> float:
        vals = [float(c["confidence"]) for c in dets if c.get("confidence") is not None]
        return max(vals) if vals else 0.0

    vision = [c for c in children if (c.get("modality") or "vision") == "vision"]
    audio = [c for c in children if c.get("modality") == "audio"]
    c_eff = _strongest(vision)
    a_eff = _strongest(audio)
    evidence = _detection_evidence(c_eff, a_eff)

    identified = [c for c in children if (c.get("gbif_usage_key") or c.get("scientific_name"))]
    if not identified:
        return {
            "computable": False,
            "provenance": "inferred",
            "reason": "this event has no identified species, so it cannot be scored",
            "detection_evidence": evidence,
            "c_eff": c_eff,
            "a_eff": a_eff,
        }

    rep = max(identified, key=lambda c: (c.get("confidence") or 0.0))
    modality = rep.get("modality") or "vision"
    model_version = (
        obs.get("acoustic_model_version") if modality == "audio"
        else obs.get("screening_model_version")
    )
    species_key = rep.get("gbif_usage_key") or rep.get("scientific_name")
    species_label = rep.get("scientific_name") or rep.get("common_name") or species_key
    accuracy = index.get((model_version, species_key))

    result = {
        "provenance": "inferred",
        "model_version": model_version,
        "species_key": species_key,
        "species_label": species_label,
        "detection_evidence": evidence,
        "c_eff": c_eff,
        "a_eff": a_eff,
    }
    if accuracy is None:
        result["computable"] = False
        result["reason"] = "no expert reviews yet for this species under this model"
        return result
    result["computable"] = True
    result["accuracy"] = accuracy
    result["value"] = _event_trust_value(c_eff, a_eff, accuracy)
    return result


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


# A short, plain-language pointer for the one failure a person cannot fix by
# themselves: the model runtime was installed for a different processor type, so
# it cannot start. The exact commands live in the guide, not in the interface, so
# the panel stays calm and readable.
_LLM_CPU_REMEDY = (
    "This is a one-time setup step, not a problem with your data. The model "
    "runtime was installed for a different processor type, so it cannot start on "
    "this computer. Installing a matching build fixes it; the exact steps are in "
    "docs/language-model.md."
)


def _llm_status(settings) -> dict:
    """A calm, plain-language readiness status for the desktop language model.

    The language model is optional enrichment: the whole platform runs without it.
    So this never alarms. It reports one of a few friendly states without loading
    the model (loading is slow): the runtime is missing, no model is present yet,
    a model is present and will load on the next run, or, in the one case a person
    truly cannot self-diagnose, the runtime does not match this computer's
    processor and the guide has the fix. Raw error codes and build flags stay in
    the log and the guide, never in the panel.
    """
    if not _llm_runtime_available():
        return {
            "status": "runtime_missing",
            "message": ("The optional language model is not set up yet, so interpretation and narration "
                        "are off. Everything else works without it."),
            "remedy": "To turn it on, install the runtime: pip install llama-cpp-python. Then drop a .gguf model into the folder below and reload.",
        }
    if not _active_llm_name(settings):
        return {
            "status": "no_model",
            "message": "No language model is set up yet, so interpretation and narration are off.",
            "remedy": "Drop a .gguf model file into the language-model folder shown below, then reload to turn it on.",
        }

    # A model is present. The only failure worth surfacing here is the one a
    # person cannot decode alone: the runtime does not match this CPU. Any other
    # recorded error (for example a stale "no model configured" from before a
    # model was placed) is not shown, because the model is present now and will be
    # tried again on the next run; the detail stays in the log.
    try:
        from audtheia.app.orchestrator import last_llm_error
        recorded = (last_llm_error() or "").lower()
    except Exception:  # noqa: BLE001 - status must never fail because of an import
        recorded = ""
    if any(k in recorded for k in ("0xc000001d", "illegal instruction", "1073741795")):
        return {
            "status": "cpu_incompatible",
            "message": "The model is installed but could not start on this computer's processor.",
            "remedy": _LLM_CPU_REMEDY,
        }

    return {
        "status": "model_present",
        "message": "The language model is set. It loads the next time the desktop runs verification or the longitudinal pass.",
        "remedy": "",
    }


# The header written into the local override file, so a person who opens it
# knows what it is and why their paths are in it rather than in the main file.
_LOCAL_OVERRIDES_COMMENT = (
    "Machine-specific absolute paths for this computer only. Written "
    "automatically. This file is excluded from version control so that an "
    "account name, home directory, or drive letter is never published. Edit "
    "config/settings.json for anything that describes the deployment rather "
    "than this machine."
)


def _write_json_atomically(target: Path, payload: dict, prefix: str) -> None:
    """Replace a JSON file in one step so a reader never sees it half written."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=prefix, suffix=".tmp")
    try:
        # newline="\n" pins the line endings. Without it, Python's text mode
        # writes Windows line endings on Windows, so the same configuration file
        # would differ byte for byte between a desktop and a field station and
        # would show up as an entirely rewritten file in version control.
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        os.replace(tmp_name, str(target))
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _persist_settings(settings) -> list:
    """Write the configuration back to disk, machine-specific paths separated out.

    The configuration is validated before anything is written, so an invalid
    change is refused rather than saved. What is then written goes to two files.
    Every absolute path is a description of this one computer and carries the
    account name of whoever is running it, so those values go to the gitignored
    local file; the committed file keeps the value it already held. The split is
    decided by shape, not by a list of known fields, so a path field added later
    is covered without anyone remembering to cover it.

    The check before the write is deliberate rather than defensive. A leak of
    this kind is silent, reaches a public remote, and survives in the history
    after the file is cleaned, so a save that would reintroduce one is refused
    outright instead of trusted to the splitting logic being correct.

    A configuration that already holds a machine path is migrated rather than
    rejected. Refusing the save in that case blocked every unrelated edit, down
    to changing the colour theme, because the offending value was one the person
    had not touched. The live value is filed in the local override file and the
    committed file is cleaned wherever a clean value can stand in its place.
    Where the field is required and no published value exists to fall back to,
    the save still proceeds and reports what remains, since blocking the edit
    neither removes the path nor tells anyone it is there.

    Returns the warnings raised while splitting, so a caller can pass them on.
    """
    from audtheia.config import (
        _validate,
        ConfigError,
        collect_absolute_paths,
        pointer_value,
        _is_absolute_path_value,
        _contains_machine_path,
    )

    try:
        _validate(settings.raw)
    except ConfigError as exc:
        raise BackendError(f"refusing to save an invalid configuration: {exc}") from exc

    target = Path(settings.settings_path)

    # What the committed file holds today. Any path being moved out keeps the
    # value already published rather than being blanked, so a save made on a
    # machine with an external store does not strip the defaults every other
    # machine relies on.
    previous: dict = {}
    if target.exists():
        try:
            loaded = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                previous = loaded
        except (OSError, json.JSONDecodeError):
            previous = {}

    overrides = collect_absolute_paths(settings.raw)

    tracked = copy.deepcopy(settings.raw)
    from audtheia.config import apply_local_overrides as _apply

    warnings: list = []

    # What the committed file should hold in place of each machine path. The
    # value already published is the right stand-in, unless it is itself a
    # machine path, in which case there is nothing clean to fall back to and the
    # field is cleared so the path stops being published.
    stand_in = {}
    polluted = []
    for pointer in overrides:
        prior = pointer_value(previous, pointer)
        if _is_absolute_path_value(prior) or _contains_machine_path(prior):
            stand_in[pointer] = None
            polluted.append(pointer)
        else:
            stand_in[pointer] = prior
    _apply(tracked, stand_in)

    # Clearing a required field would make the committed file unloadable, which
    # helps nobody. Where that happens the published value is put back and the
    # person is told which field still names their machine, so they can point it
    # somewhere shared or accept it knowingly.
    if polluted:
        try:
            _validate(tracked)
        except ConfigError:
            _apply(tracked, {p: pointer_value(previous, p) for p in polluted})
            listed = ", ".join(sorted(polluted))
            warnings.append(
                f"the saved configuration still names this machine at {listed}, "
                "because the field is required and has no shared value to fall "
                "back to. Point it at a location inside the repository, or move "
                "it to the local override file, before publishing this file."
            )

    introduced = sorted(set(collect_absolute_paths(tracked)) - set(polluted))
    if introduced:
        listed = ", ".join(introduced)
        raise BackendError(
            "refusing to save: absolute paths would be written to the committed "
            f"configuration ({listed}). An absolute path identifies one machine "
            "and must go to the local override file instead."
        )

    _write_json_atomically(target, tracked, ".settings-")

    local_target = Path(settings.local_overrides_path)
    if overrides or local_target.exists():
        _write_json_atomically(
            local_target,
            {"_comment": _LOCAL_OVERRIDES_COMMENT, "overrides": overrides},
            ".settings-local-",
        )

    return warnings


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
        # newline="\n" pins the line endings. Without it, Python's text mode
        # writes Windows line endings on Windows, so the same configuration file
        # would differ byte for byte between a desktop and a field station and
        # would show up as an entirely rewritten file in version control.
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
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

# Running desktop ACOUSTIC capture sessions, keyed by station id, kept separate
# from the vision jobs so a station can run one, the other, or both at once.
_AUDIO_CAPTURE_JOBS: dict = {}


def _new_station_dict(station_id: str, name: str, environment: str, habitat: Optional[str]) -> dict:
    """A complete, valid station configuration with sensible defaults.

    The identifier is generated, the device sensors default to on, and the
    channel list starts empty (environmental sensors are added from the Sensors
    settings). Every model slot is present but unset. A station's models are the
    one thing this cannot guess: the field screener and the desktop verifier are
    trained on the species the person deployed the station to watch, so naming a
    file here would assert a model that does not exist and, worse, would name an
    architecture the model may not be. The station validates with the slots null
    and reports itself honestly as having no model set until it is pointed at
    real files under Settings, Model paths.
    """
    station = {
        "station_id": station_id,
        "station_name": name,
        "environment_type": environment,
        "target_species": [],
        "location": {"latitude": None, "longitude": None, "elevation": None},
        "sensors": {"camera": {"enabled": True}, "audio": {"enabled": True}, "gps": {"enabled": True}},
        "channels": [],
        "models": {
            "visual_pi": {"path": None, "version": None, "citation": None},
            "visual_desktop": {"path": None, "version": None, "citation": None},
            "acoustic": {
                "path": None,
                "labels_path": None,
                "sample_rate": None,
                "window_seconds": None,
                "output_key": None,
                "version": None,
                "citation": None,
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


def _v_path_or_null(value: Any, where: str) -> Optional[str]:
    """A model path, or null to state plainly that no model is set.

    Clearing a model path has to be possible, because the alternative is a
    configuration that always names a file whether or not one exists. The key
    itself must stay: the configuration validator requires the key to be
    present, and only its value may be null, so this returns None rather than
    signalling a deletion. Backslashes are normalised so a path pasted from a
    Windows file dialog resolves the same way on the field station.

    A matched pair of surrounding quotes is removed first. Windows "Copy as
    path" wraps the path in double quotes and people paste exactly that, which
    is the normal way to get a long path into the field. Kept as written, the
    quotes become part of the name, no file is ever found at it, and the
    interface reports a model as missing while pointing at a path that looks
    correct on screen. Capture sources have always accepted the quoted form, so
    a model path accepting it too is consistency rather than a new allowance.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise SettingsUpdateError(f"{where} must be a file path, or empty to clear it")
    trimmed = value.strip()
    if len(trimmed) >= 2 and trimmed[0] == trimmed[-1] and trimmed[0] in ("'", '"'):
        trimmed = trimmed[1:-1].strip()
    trimmed = trimmed.replace("\\", "/")
    return trimmed or None


def _v_choice(choices):
    def _inner(value: Any, where: str):
        if value not in choices:
            raise SettingsUpdateError(f"{where} must be one of: {', '.join(choices)}")
        return value
    return _inner


def _v_number_in_range(low: float, high: float, unit: str):
    """A number within an inclusive range, or null to clear the value.

    Used for a station's fixed coordinates: a person can enter a position or
    clear it, and a value outside its valid degree range is refused here with a
    clear reason rather than reaching the record.
    """
    def _inner(value: Any, where: str) -> Optional[float]:
        if value is None:
            return None
        number = _v_number(value, where)
        if not (low <= number <= high):
            raise SettingsUpdateError(f"{where} must be between {low} and {high} {unit}, or empty to clear it")
        return number
    return _inner


def _v_number_or_null(value: Any, where: str) -> Optional[float]:
    """Any number, or null to clear the value."""
    if value is None:
        return None
    return _v_number(value, where)


def _v_positive_number_or_null(value: Any, where: str) -> Optional[float]:
    """A number greater than zero, or null to clear the value.

    Used for an acoustic model's window length, which a person may enter or
    leave unset for the model file's own shape to supply.
    """
    if value is None:
        return None
    return _v_positive_number(value, where)


def _v_positive_int_or_null(value: Any, where: str) -> Optional[int]:
    """A whole number greater than zero, or null to clear the value.

    Used for an acoustic model's sample rate, which is not stored in the model
    file and so is entered or confirmed once rather than read.
    """
    if value is None:
        return None
    return _v_positive_int(value, where)


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
            "ui_theme": {"path": ["ui", "theme"], "validate": _v_str_or_null},
            "ui_last_dark": {"path": ["ui", "last_dark"], "validate": _v_str_or_null},
            "ui_last_light": {"path": ["ui", "last_light"], "validate": _v_str_or_null},
            "visual_rfdetr_path": {"path": ["desktop_models", "visual_rfdetr", "path"], "validate": _v_path_or_null, "is_path": True},
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
            "station_latitude": {"path": ["location", "latitude"], "validate": _v_number_in_range(-90, 90, "degrees")},
            "station_longitude": {"path": ["location", "longitude"], "validate": _v_number_in_range(-180, 180, "degrees")},
            "station_elevation": {"path": ["location", "elevation"], "validate": _v_number_or_null},
            "visual_pi_path": {"path": ["models", "visual_pi", "path"], "validate": _v_path_or_null, "is_path": True},
            "visual_pi_version": {"path": ["models", "visual_pi", "version"], "validate": _v_str_or_null},
            "visual_pi_citation": {"path": ["models", "visual_pi", "citation"], "validate": _v_str_or_null},
            "visual_desktop_path": {"path": ["models", "visual_desktop", "path"], "validate": _v_path_or_null, "is_path": True},
            "visual_desktop_version": {"path": ["models", "visual_desktop", "version"], "validate": _v_str_or_null},
            "visual_desktop_citation": {"path": ["models", "visual_desktop", "citation"], "validate": _v_str_or_null},
            # The acoustic model is one flat block: a station listens with one
            # model in one place. The path and labels file are set here, along
            # with the audio shape the file expects (sample rate, window length,
            # output key) and the version and citation. No model family is named.
            "acoustic_path": {"path": ["models", "acoustic", "path"], "validate": _v_path_or_null, "is_path": True},
            "acoustic_labels_path": {"path": ["models", "acoustic", "labels_path"], "validate": _v_path_or_null, "is_path": True},
            "acoustic_sample_rate": {"path": ["models", "acoustic", "sample_rate"], "validate": _v_positive_int_or_null},
            "acoustic_window_seconds": {"path": ["models", "acoustic", "window_seconds"], "validate": _v_positive_number_or_null},
            "acoustic_output_key": {"path": ["models", "acoustic", "output_key"], "validate": _v_str_or_null},
            "acoustic_version": {"path": ["models", "acoustic", "version"], "validate": _v_str_or_null},
            "acoustic_citation": {"path": ["models", "acoustic", "citation"], "validate": _v_str_or_null},
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


def _same_model_file(left: Any, right: Any, repo_root) -> bool:
    """Whether two configured paths name the same file on this machine.

    Compared after resolution rather than as text, so a relative path and an
    absolute one that reach the same file are recognised as the same model. A
    path that cannot be resolved falls back to a normalised text comparison.
    """
    if not isinstance(left, str) or not isinstance(right, str):
        return False

    def _resolve(value: str):
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = Path(repo_root) / candidate
        try:
            return candidate.resolve()
        except OSError:
            return None

    a, b = _resolve(left), _resolve(right)
    if a is not None and b is not None:
        return a == b
    return left.strip().replace("\\", "/") == right.strip().replace("\\", "/")


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
    # refused, so a person can point at a model they are about to add. Clearing a
    # path to null is a deliberate statement that no model is set and needs no
    # warning at all.
    if spec.get("is_path") and isinstance(value, str):
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = Path(repo_root) / candidate
        if not candidate.exists():
            warnings.append(f"{where}: no file is present yet at {value}")

        # A slot takes one kind of file, and the kinds are not interchangeable:
        # an acoustic model in a visual screening slot reports itself present
        # and then fails at capture, while the acoustic slot it belonged in
        # still reads as empty. The shape is knowable from the filename, so it
        # is said here rather than left to be discovered during a run.
        expected = _EXPECTED_MODEL_SUFFIXES.get(field)
        suffix = candidate.suffix.lower()
        if expected and suffix and suffix not in expected:
            warnings.append(
                f"{where}: this slot expects {' or '.join(sorted(expected))} and "
                f"the file given ends in {suffix}. {_SLOT_PURPOSE.get(field, '')} "
                f"Check the file is in the slot you meant."
            )

        # A classifier reports a class number, and the labels file is what turns
        # that number into a name. Without one the model runs perfectly well and
        # every detection is recorded as a bare index, which reads like a
        # species identifier and is not one. Said at the moment the model is
        # set, because that is the moment the labels file is to hand. The
        # acoustic slot never enforces a file type: the model may be TFLite, a
        # SavedModel, ONNX, or another form, so only the missing-labels case and
        # a label-count mismatch are reported, never a suffix.
        if field == "acoustic_path":
            station = _find_station_in(draft, change.get("station_id")) or {}
            acoustic = ((station.get("models") or {}).get("acoustic")) or {}
            labels_value = acoustic.get("labels_path")
            if not labels_value:
                warnings.append(
                    f"{where}: no labels file is set for this model, so detections "
                    f"will be recorded by class number instead of a name. Set the "
                    f"acoustic labels file as well."
                )
            else:
                warnings.extend(_acoustic_label_count_warnings(where, candidate, labels_value, repo_root))

    # Verification by the same weights is not verification. A desktop screening
    # model that is the same file as the hub verifier means an event is scored
    # and then re-scored by one model, so the agreement recorded against it
    # carries no independent evidence. Allowed, because a person may deliberately
    # run one model while trialling, but never allowed to happen silently.
    if spec.get("is_path") and field == "visual_desktop_path" and isinstance(value, str):
        verifier = ((draft.get("desktop_models") or {}).get("visual_rfdetr") or {}).get("path")
        if verifier and _same_model_file(value, verifier, repo_root):
            warnings.append(
                f"{where}: this is the same file as the desktop verification model. "
                f"Screening and verification would run identical weights, so the "
                f"agreement figures recorded for this station would not be "
                f"independent evidence."
            )


# What kind of file each visual slot takes. These two are fixed by the runtime
# that loads them, not by what is being studied: a Hailo accelerator reads a
# compiled .hef and nothing else, and the desktop reads ONNX through ONNX
# Runtime. So a mismatch here is always an error worth reporting.
#
# The acoustic slots are deliberately absent. An acoustic model may be TFLite,
# a TensorFlow SavedModel, ONNX, or another form entirely, depending on what
# the person trained and on what they are listening to. Whoever is recording
# cetaceans, bats, or reef fish will not be carrying the same file type as
# whoever is recording birds, and the application must not assume otherwise.
_EXPECTED_MODEL_SUFFIXES = {
    "visual_pi_path": {".hef"},
    "visual_desktop_path": {".onnx"},
    "visual_rfdetr_path": {".onnx"},
}

# Said alongside a mismatch so the message names what the slot is for, rather
# than only reporting that the file chosen is wrong.
_SLOT_PURPOSE = {
    "visual_pi_path": "This slot is the station's own screening model, compiled for its accelerator.",
    "visual_desktop_path": "This slot screens video frames during desktop capture. A model that listens to audio belongs in the acoustic slot.",
    "visual_rfdetr_path": "This slot re-scores saved video frames on this computer.",
}

def _station_deployment(station: dict) -> dict:
    """Where a station runs, read from its configuration rather than assumed.

    A station is a field deployment when it carries a screening model compiled
    for its accelerator (models.visual_pi.path, a .hef that cannot exist without
    the field hardware). It runs on the desktop when it carries a desktop
    screening model (models.visual_desktop.path) or a desktop capture source
    (capture.source.video or capture.source.audio). A station may be both, and
    one that is neither is simply unconfigured and is reported as such rather
    than being filed under either. This is what a station is classified by, so a
    station created on this computer is not assumed to be out in a field.
    """
    models = station.get("models") or {}
    source = (station.get("capture") or {}).get("source") or {}
    field = bool((models.get("visual_pi") or {}).get("path"))
    desktop = (
        bool((models.get("visual_desktop") or {}).get("path"))
        or bool(source.get("video"))
        or bool(source.get("audio"))
    )
    return {"field": field, "desktop": desktop, "configured": field or desktop}


def _normalise_entered_path(text: str) -> str:
    """Strip one matched quote pair and normalise separators, as the save path does.

    Windows "Copy as path" wraps a path in double quotes and uses backslashes;
    pasted verbatim, the quotes would become part of the name and no file would
    be found. This applies the same healing the guarded save applies, so a probe
    at entry resolves the same file the save eventually will.
    """
    trimmed = str(text).strip()
    if len(trimmed) >= 2 and trimmed[0] == trimmed[-1] and trimmed[0] in ("'", '"'):
        trimmed = trimmed[1:-1].strip()
    return trimmed.replace("\\", "/")


def _load_acoustic_shape_proposals(settings) -> dict:
    """The sample-rate proposal table, keyed by measured model fingerprint.

    Lives in config/model_sources.json as data, not code, so no model family is
    named in any configuration key or validation rule. Read only to offer a
    proposed sample rate for a recognised fingerprint; a missing or unreadable
    file simply yields no proposal, and the rate is then entered by hand.
    """
    try:
        path = Path(settings.repo_root) / "config" / "model_sources.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        block = data.get("acoustic_shape_proposals", {})
        by_fingerprint = block.get("by_fingerprint", {})
        return by_fingerprint if isinstance(by_fingerprint, dict) else {}
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}


def _count_label_lines(labels_path: Path) -> Optional[int]:
    """How many class names a labels file holds, or None if it cannot be read.

    A `.json` file may hold a list of names or an index-to-name map; any other
    file is one name per line, blank lines and comment lines ignored. A file that
    cannot be read yields None so the caller simply skips the count check.
    """
    try:
        text = labels_path.read_text(encoding="utf-8")
    except OSError:
        return None
    if labels_path.suffix.lower() == ".json":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, (list, tuple)):
            return len(parsed)
        if isinstance(parsed, dict):
            return len(parsed)
        return None
    return len([ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")])


def _acoustic_label_count_warnings(where: str, model_candidate: Path, labels_value: str, repo_root) -> list:
    """Warn when a labels file's line count disagrees with the model's class head.

    A labels file with one fewer or one more line than the model has classes is
    a silent off-by-one that a presence check cannot catch: every detection is
    then named by the wrong neighbour. The class-head width is read from the
    model file when it is a probeable `.tflite`; when it is not, or either file
    is absent, no count is possible and nothing is warned, since a missing file
    is already reported elsewhere. No file type is required of the model here.
    """
    warnings: list = []
    labels_candidate = Path(labels_value)
    if not labels_candidate.is_absolute():
        labels_candidate = Path(repo_root) / labels_candidate
    if not labels_candidate.exists() or not model_candidate.exists():
        return warnings
    if model_candidate.suffix.lower() != ".tflite":
        return warnings
    try:
        from audtheia.pipeline.acoustic import probe_acoustic_model
        probed = probe_acoustic_model(model_candidate)
        class_count = probed.get("read", {}).get("class_count")
    except Exception:  # noqa: BLE001 - a model that cannot be probed just skips the check
        class_count = None
    label_count = _count_label_lines(labels_candidate)
    if class_count and label_count and class_count != label_count:
        warnings.append(
            f"{where}: this labels file has {label_count} names but the model has "
            f"{class_count} classes. A mismatch means detections are named by the "
            f"wrong class. Check the labels file matches the model."
        )
    return warnings


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
        # Paths are listed relative to the bundle, not the reports directory, so
        # a link is built as bundle + "/" + file without doubling the bundle name.
        # The chart images under assets/ are embedded in the PDF, so they are not
        # offered as separate downloads; the PDF and the CSV data are the
        # deliverables a person opens.
        files = []
        for p in sorted(entry.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(entry)
            if rel.parts and rel.parts[0] == "assets":
                continue
            files.append(str(rel).replace("\\", "/"))
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

    class DataDirectoryRequest(BaseModel):
        path: Optional[str] = None

    class ArchiveRequest(BaseModel):
        start: Optional[str] = None
        end: Optional[str] = None
        station_id: Optional[str] = None
        target_dir: Optional[str] = None
        reclaim: Optional[bool] = False

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
        # The checkable condition a field skill runs on, as {source, field, op, value}.
        condition: Optional[dict] = None

    class LlmSelectRequest(BaseModel):
        name: Optional[str] = None

    class RetrainingExportRequest(BaseModel):
        kind: Optional[str] = None
        station_id: Optional[str] = None
        confidence_below: Optional[float] = None
        include_disagreements: Optional[bool] = None
        include_deferred: Optional[bool] = None
        force: Optional[bool] = False

    class SettingsUpdateRequest(BaseModel):
        changes: Optional[list] = None

    class ChannelRequest(BaseModel):
        id: Optional[str] = None
        unit: Optional[str] = None
        marine: Optional[bool] = None
        enabled: Optional[bool] = None
        driver: Optional[dict] = None
        qc: Optional[dict] = None

    class TargetSpeciesRequest(BaseModel):
        name: Optional[str] = None

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

    class CorrectionRequest(BaseModel):
        verdict: Optional[str] = None
        detection_id: Optional[str] = None
        modality: Optional[str] = None
        gbif_usage_key: Optional[str] = None
        reason: Optional[str] = None
        corrector: Optional[str] = None

    class ModelProbeRequest(BaseModel):
        path: Optional[str] = None
        kind: Optional[str] = None
        labels_path: Optional[str] = None
        field: Optional[str] = None

    class FrameReviewRequest(BaseModel):
        verdict: Optional[str] = None  # accurate / inaccurate / cleared
        corrector: Optional[str] = None
        reason: Optional[str] = None

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
        # The Detections view is visual events only; acoustic events live on the
        # Audio tab, so they are never mixed into this list.
        total = db.count_observations(station_id=station_id, since=since, until=until, species=sp, trigger="vision")
        rows = db.list_observations(station_id=station_id, since=since, until=until, species=sp,
                                    trigger="vision", limit=limit, offset=offset)
        # Built once per request, not per card: the accuracy lookup Event Trust
        # needs is derived from the whole reviewed record and is the same for
        # every event in the page.
        trust_index = db.event_trust_index()
        out = []
        for obs in rows:
            item = dict(obs)
            children = db.list_child_detections(obs["id"])
            item["vision_detections"] = [c for c in children if c.get("modality") == "vision"]
            item["verification"] = db.get_observation_verification(obs["id"])
            # The expert's current position on this event, so a card can show a
            # reviewed identification instead of a model percentage without a
            # second request per card.
            item["correction"] = db.latest_correction(obs["id"])
            # Event Trust for the card chip: shown only when computable, otherwise
            # a clear "not yet rated" carrying its reason. Inference, model-tagged.
            item["event_trust"] = _event_trust(obs, children, trust_index)
            out.append(item)
        return {"items": out, "total": total, "limit": limit, "offset": offset}

    # Registered before '/detections/{observation_id}' so the literal path is not
    # captured as an id. Returns the full species list for the filter dropdown.
    @app.get(f"{API_PREFIX}/detections/species")
    def detection_species(station_id: str | None = Query(default=None)):
        return db.list_species(station_id=station_id, modality="vision")

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
            "skill_flags": db.list_skill_flags(observation_id),
            # Event Trust for the modal's "how these numbers were derived" block,
            # shown beside salience and labelled inference. The detection evidence
            # here is salience's own D; the accuracy is the per-species figure for
            # the model that produced the headline call.
            "event_trust": _event_trust(obs, children, db.event_trust_index()),
        }

    def _event_frames_on_disk(obs):
        """Read an event's saved frames and per-frame species distribution.

        Returns (manifest, frames, distribution). Empty frames and distribution
        when the event has no representative frame or its files are absent.
        Raises 400 if the stored frame path escapes the data directory. This is
        shared by the frames read and the review write so both agree on the exact
        frame set, and therefore on the curated count and trust computed from it.
        """
        rep = obs.get("representative_frame")
        if not rep:
            return None, [], []

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

        # The per-frame species distribution, from the model's own per-frame class
        # names, turns a strip that flickers between similar species into a summary
        # an expert reads at a glance and is what surfaces a genuinely mixed track.
        dist_counts: dict = {}
        for f in frames:
            name = f.get("class_name")
            if name:
                dist_counts[name] = dist_counts.get(name, 0) + 1
        distribution = [
            {"class_name": name, "count": n}
            for name, n in sorted(dist_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        return manifest, frames, distribution

    def _frame_review_summary(observation_id, frames, distribution):
        """The curated review summary the interface shows above the frame strip.

        The measured numbers on the observation are never changed by a review; the
        curated count subtracts only the frames explicitly marked inaccurate,
        leaving accurate and not-yet-reviewed frames in the trusted set. Built the
        same way for both the frames read and the review write, so the summary a
        toggle returns is identical to the one a reload would show.
        """
        summary = db.frame_review_summary(observation_id)
        total = len(frames)
        inaccurate = summary.get("inaccurate", 0)
        curated = max(0, total - inaccurate)
        return {
            "total_frames": total,
            "accurate": summary.get("accurate", 0),
            "inaccurate": inaccurate,
            "reviewed": summary.get("reviewed", 0),
            "curated_frame_count": curated,
            # Share of frames not marked inaccurate. 1.0 when nothing is flagged,
            # 0.0 when every frame is. The pass and analytics read this as the
            # event's trust weight; it never overwrites a measured value.
            "trust": (curated / total) if total else None,
            # A derived display hint only: the track carried more than one species
            # across its frames. Not a stored verdict and not a taxonomic claim.
            "multiple_candidates": len(distribution) > 1,
        }

    @app.get(f"{API_PREFIX}/detections/{{observation_id}}/frames")
    def detection_frames(observation_id):
        """Return every stored frame of an event with its per-frame annotation.

        The capture pipeline writes each detected frame to the event directory and
        appends one line per frame to `annotations.jsonl` (index, timestamp,
        confidence, box), alongside an `annotations.json` manifest. This read-only
        endpoint surfaces both so the interface can audit an observation's stats
        (the frame count, the true duration, and the per-frame confidences) rather
        than asking the scientist to trust them. Boxes are converted to the same
        x/y/w/h form the card overlay uses. Nothing is written or deleted.
        """
        obs = db.get_observation(observation_id)
        if obs is None:
            raise HTTPException(status_code=404, detail=f"no observation with id {observation_id}")
        manifest, frames, distribution = _event_frames_on_disk(obs)
        skill_flags = db.list_skill_flags(observation_id)
        if not frames:
            return {"observation": obs, "manifest": manifest, "frames": [],
                    "distribution": [], "review_summary": _frame_review_summary(observation_id, [], []),
                    "skill_flags": skill_flags}

        # Attach the current expert verdict to each frame for the strip display.
        reviews = {
            r["frame_index"]: r["verdict"]
            for r in db.frame_reviews_for_observation(observation_id)
        }
        for f in frames:
            verdict = reviews.get(f.get("index"))
            f["review"] = verdict if verdict in ("accurate", "inaccurate") else None

        return {
            "observation": obs,
            "manifest": manifest,
            "frames": frames,
            "distribution": distribution,
            "review_summary": _frame_review_summary(observation_id, frames, distribution),
            "skill_flags": skill_flags,
        }

    @app.post(f"{API_PREFIX}/detections/{{observation_id}}/frames/{{frame_index}}/review", status_code=201)
    def review_frame(observation_id, frame_index: int, request: FrameReviewRequest):
        """Record one expert verdict on one saved frame of an event.

        Nothing measured is modified. The event's frame_count, duration,
        confidence and salience stay exactly as captured; this appends a separate
        human claim about one frame, with its own provenance. A change of mind is
        another append, so the review history stays legible. 'cleared' retracts
        an earlier verdict, returning the frame to unreviewed. The response
        carries the same full curated summary the frames read returns, so the
        interface updates the kept count and the trust weight live on each toggle.
        """
        obs = db.get_observation(observation_id)
        if obs is None:
            raise HTTPException(status_code=404, detail=f"no observation with id {observation_id}")
        if frame_index < 0:
            raise HTTPException(status_code=400, detail="frame_index must be zero or greater")
        verdict = (request.verdict or "").strip()
        if verdict not in _FRAME_REVIEW_VERDICTS:
            raise HTTPException(
                status_code=400,
                detail=f"verdict must be one of {', '.join(_FRAME_REVIEW_VERDICTS)}",
            )
        stored = db.add_frame_review(
            observation_id,
            frame_index,
            verdict=verdict,
            corrector=(request.corrector or "").strip() or _DEFAULT_CORRECTOR,
            reason=(request.reason or "").strip() or None,
        )
        _, frames, distribution = _event_frames_on_disk(obs)
        return {"review": stored, "review_summary": _frame_review_summary(observation_id, frames, distribution)}

    @app.post(f"{API_PREFIX}/detections/{{observation_id}}/frames/review-all", status_code=201)
    def review_all_frames(observation_id, request: FrameReviewRequest):
        """Record one verdict on every saved frame of an event at once.

        This backs the "mark all accurate" action, so a reviewer accepts the
        whole event in one step and then only has to mark the few wrong frames,
        whose later per-frame verdict wins on read. Nothing measured is modified;
        each frame gets its own appended human claim, exactly like a single
        review. The response carries the same curated summary the frames read
        returns, so the interface updates live.
        """
        obs = db.get_observation(observation_id)
        if obs is None:
            raise HTTPException(status_code=404, detail=f"no observation with id {observation_id}")
        verdict = (request.verdict or "").strip()
        if verdict not in _FRAME_REVIEW_VERDICTS:
            raise HTTPException(
                status_code=400,
                detail=f"verdict must be one of {', '.join(_FRAME_REVIEW_VERDICTS)}",
            )
        _, frames, distribution = _event_frames_on_disk(obs)
        indices = [f.get("index") for f in frames if f.get("index") is not None]
        written = db.add_frame_reviews_bulk(
            observation_id, indices,
            verdict=verdict,
            corrector=(request.corrector or "").strip() or _DEFAULT_CORRECTOR,
        )
        return {"written": written, "review_summary": _frame_review_summary(observation_id, frames, distribution)}

    # -- expert corrections ----------------------------------------------

    @app.get(f"{API_PREFIX}/species/search")
    def species_search(q: str | None = Query(default=None)):
        """Prefix search over the shipped taxonomic backbone.

        This is the only way a corrected name can enter the record. The caller
        gets back usage keys, and the correction endpoint accepts nothing else,
        so a typo cannot become a taxon. Synonyms are returned with the accepted
        name they resolve to, because an expert who types a name they learned
        thirty years ago should still find the taxon it became.

        An absent index is not an error. The confirm and reject verdicts need no
        index at all, so a missing one degrades relabelling rather than breaking
        review, and the interface can say so plainly instead of failing.
        """
        term = (q or "").strip().lower()
        if len(term) < 2:
            return {"results": [], "index_available": True}

        index_path = _gbif_index_path(settings)
        if not index_path.is_file():
            return {"results": [], "index_available": False}

        # Read-only, and the LIKE pattern is escaped so a name containing a
        # wildcard character is searched for literally rather than matching
        # everything.
        pattern = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        try:
            conn = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
        except sqlite3.Error:
            return {"results": [], "index_available": False}
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT usage_key, canonical_name, scientific_name, status, accepted_name "
                "FROM taxon_index WHERE name_lower LIKE ? ESCAPE '\\' "
                "ORDER BY LENGTH(canonical_name), canonical_name LIMIT ?",
                (pattern, _SPECIES_SEARCH_LIMIT),
            ).fetchall()
        except sqlite3.Error:
            return {"results": [], "index_available": False}
        finally:
            conn.close()

        return {
            "results": [
                {
                    "usage_key": r["usage_key"],
                    "canonical_name": r["canonical_name"],
                    "scientific_name": r["scientific_name"],
                    "status": r["status"],
                    "accepted_name": r["accepted_name"],
                }
                for r in rows
            ],
            "index_available": True,
        }

    # -- species data setup (the taxonomic index and the reference fetch) -
    # These two guided actions replace the command-line setup scripts, so a
    # non-programmer can prepare species naming and conservation data from the
    # interface. Each runs in the background because it takes minutes, and the
    # interface polls its status endpoint. The reference fetch is the single
    # user-initiated online step in the desktop app; the detection and report
    # paths remain fully offline.

    @app.get(f"{API_PREFIX}/species/index/status")
    def species_index_status():
        """Whether the taxonomic index exists, and any build in progress."""
        index_path = _gbif_index_path(settings)
        backbone = _backbone_file(settings)
        return {
            "job": _job_snapshot("species_index"),
            "index_present": index_path.is_file(),
            "index_names": _index_name_count(index_path),
            "backbone_present": backbone.is_file(),
            "backbone_path": str(backbone),
        }

    @app.post(f"{API_PREFIX}/species/index/build", status_code=202)
    def species_index_build(force: bool = Query(default=False)):
        """Build the taxonomic index from the shipped backbone, in the background.

        Relabelling a detection to a corrected species searches this index, so
        this is what turns relabelling on for a fresh install. Confirm, reject,
        and per-frame review need no index and work without it.
        """
        if settings.node_role != "desktop":
            raise HTTPException(status_code=403, detail="the species index is built on the desktop; this node is not the desktop.")
        backbone = _backbone_file(settings)
        if not backbone.is_file():
            raise HTTPException(
                status_code=422,
                detail=(f"the GBIF backbone file was not found at {backbone}. "
                        "Fetch it with the setup step first, then build the index."),
            )
        index_path = _gbif_index_path(settings)
        if index_path.is_file() and not force:
            raise HTTPException(
                status_code=409,
                detail="a taxonomic index already exists; pass force to rebuild it from the backbone.",
            )

        def _target(update):
            module = _load_script_module(settings, "build_gbif_index")

            def _progress(pct, scanned, kept):
                update(progress=round(pct / 100.0, 4),
                       message=f"scanned {scanned:,} rows, kept {kept:,}")

            names = module.build(backbone, index_path, on_progress=_progress)
            return {"names": names}

        if not _start_background_job("species_index", _target):
            raise HTTPException(status_code=409, detail="an index build is already running.")
        return {"started": True, "job": _job_snapshot("species_index")}

    @app.get(f"{API_PREFIX}/species/reference/status")
    def species_reference_status():
        """How much reference data is on file, and what a fetch would cover."""
        target = sorted({
            name.strip()
            for station in settings.stations()
            for name in (station.get("target_species") or [])
            if isinstance(name, str) and name.strip()
        })
        # The fetch also covers the taxa already detected in the record, so it
        # works for a deployment that captured before declaring target species.
        detected = db.list_detected_taxa()
        return {
            "job": _job_snapshot("species_reference"),
            "references_stored": len(db.list_species_reference()),
            "target_species": target,
            "detected_species": sorted(detected),
            "iucn_token_present": bool(settings.secrets.get("iucn_api_key")),
        }

    def _stamp_existing_observations() -> int:
        """Fill the reference snapshot dates on already-captured records.

        New captures stamp their taxonomy snapshot at capture, but records taken
        before their species reference was fetched carry none. This completes them
        from the reference cache after a fetch: for each record still missing the
        dates, it matches its dominant taxon's label to a fetched reference and,
        only when the date is still unset, stamps it. It never overwrites a value
        and never touches a measured field, and a taxon with no matching reference
        is left unchanged. Returns how many records it completed.
        """
        stamped = 0
        for obs in db.list_observations():
            # Skip only a record that already has both dates. A record with one
            # date but not the other (for example a GBIF date stamped before the
            # conservation status could be fetched) is still processed, so the
            # missing date is filled independently.
            if obs.get("gbif_snapshot_date") and obs.get("iucn_fetch_date"):
                continue
            children = db.list_child_detections(obs["id"])
            if not children:
                continue
            dominant = max(children, key=lambda c: c.get("confidence") or 0.0)
            name = dominant.get("scientific_name") or dominant.get("common_name")
            ref = db.find_species_reference_by_name(name)
            if not ref or not (ref.get("gbif_snapshot_date") or ref.get("iucn_fetch_date")):
                continue
            if db.stamp_observation_snapshot(obs["id"], ref.get("gbif_snapshot_date"), ref.get("iucn_fetch_date")):
                stamped += 1
        return stamped

    @app.post(f"{API_PREFIX}/species/reference/fetch", status_code=202)
    def species_reference_fetch(refresh: bool = Query(default=False)):
        """Fetch GBIF and IUCN reference data for the stations' target species.

        GBIF naming and the global occurrence count need no account. The IUCN
        token, when present, only adds the Red List conservation status; without
        it that one field is left blank and reported, and everything else still
        fetches. After the fetch, this also stamps the taxonomy snapshot onto
        already-captured records whose taxon now has a reference, so the versions
        panel fills for both new and existing detections whose label matches a
        fetched name. This is a user-initiated online setup step, distinct from
        the offline detection and report paths.
        """
        if settings.node_role != "desktop":
            raise HTTPException(status_code=403, detail="reference data is fetched on the desktop; this node is not the desktop.")

        def _target(update):
            import types
            module = _load_script_module(settings, "bootstrap_fetch_species")
            # Cover the taxa actually detected in the record in addition to the
            # configured target species, so the fetch fills reference data for a
            # deployment that captured before declaring any targets. The script
            # unions these with each station's target_species and deduplicates.
            detected = db.list_detected_taxa()
            args = types.SimpleNamespace(station_id=None, species=detected or None, from_file=None, refresh=refresh)
            client = _species_fetch_client_factory() if _species_fetch_client_factory else None
            update(message="contacting GBIF"
                   + (" and IUCN" if settings.secrets.get("iucn_api_key") else " (no IUCN token, conservation status will be blank)"))
            outcome = dict(module.run(settings, client=client, args=args) or {})
            update(message="stamping snapshot dates on existing records")
            outcome["stamped_existing"] = _stamp_existing_observations()
            return outcome

        if not _start_background_job("species_reference", _target):
            raise HTTPException(status_code=409, detail="a reference fetch is already running.")
        return {"started": True, "job": _job_snapshot("species_reference")}

    @app.get(f"{API_PREFIX}/observations/{{observation_id}}/corrections")
    def observation_corrections(observation_id):
        obs = db.get_observation(observation_id)
        if obs is None:
            raise HTTPException(status_code=404, detail=f"no observation with id {observation_id}")
        return {"corrections": db.corrections_for_observation(observation_id)}

    @app.post(f"{API_PREFIX}/observations/{{observation_id}}/correct", status_code=201)
    def correct_observation(observation_id, request: CorrectionRequest):
        """Record one expert judgement about one claim.

        Nothing the model wrote is touched. The screening confidence, the field
        species call, and the desktop verification verdict all stay exactly as
        their producers recorded them; this appends a separate assertion with
        its own provenance. A change of mind is another append, so the review
        history stays legible rather than being flattened into a final state.

        A relabel must arrive as a usage key the backbone returned. Free text is
        refused outright, because a correction that cannot be resolved to a real
        taxon is worth less than no correction at all.
        """
        obs = db.get_observation(observation_id)
        if obs is None:
            raise HTTPException(status_code=404, detail=f"no observation with id {observation_id}")

        verdict = (request.verdict or "").strip()
        if verdict not in _CORRECTION_VERDICTS:
            raise HTTPException(
                status_code=400,
                detail=f"verdict must be one of {', '.join(_CORRECTION_VERDICTS)}",
            )

        modality = (request.modality or "").strip() or None
        if modality is not None and modality not in ("vision", "audio"):
            raise HTTPException(status_code=400, detail="modality must be vision or audio")

        detection_id = (request.detection_id or "").strip() or None
        if detection_id is not None:
            known = {c["id"] for c in db.list_child_detections(observation_id)}
            if detection_id not in known:
                raise HTTPException(
                    status_code=400,
                    detail=f"detection {detection_id} does not belong to observation {observation_id}",
                )

        usage_key = (request.gbif_usage_key or "").strip() or None
        if verdict == "reject" and usage_key is not None:
            raise HTTPException(
                status_code=400,
                detail="a rejection asserts that nothing is present, so it cannot carry a species",
            )
        if verdict == "relabel" and usage_key is None:
            raise HTTPException(
                status_code=400,
                detail="a relabel must name a species chosen from the taxonomic search",
            )

        scientific_name = None
        common_name = None
        snapshot_date = None
        if usage_key is not None:
            resolved = _resolve_usage_key(settings, usage_key)
            if resolved is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"usage key {usage_key} is not in the shipped taxonomic backbone",
                )
            # A synonym is stored as the accepted taxon it resolves to, so the
            # record never accumulates two names for one organism.
            usage_key = resolved["usage_key"]
            scientific_name = resolved["scientific_name"]
            snapshot_date = resolved["snapshot_date"]
            reference = db.get_species_reference(usage_key)
            if reference is not None:
                common_name = reference.get("common_name")

        stored = db.add_correction(
            observation_id,
            verdict=verdict,
            corrector=(request.corrector or "").strip() or _DEFAULT_CORRECTOR,
            detection_id=detection_id,
            modality=modality,
            corrected_scientific_name=scientific_name,
            corrected_common_name=common_name,
            corrected_gbif_usage_key=usage_key,
            gbif_snapshot_date=snapshot_date,
            reason=(request.reason or "").strip() or None,
            # Left unset deliberately. Salience is normalised against k, the size
            # of the detecting model's label universe, and the desktop backend
            # does not have the field detector loaded, so any k it picked would
            # be a different denominator from the one the provisional value used.
            # A number computed that way would not be comparable to the value
            # sitting beside it, and an incomparable number is worse than none.
            # The column is nullable precisely so this can be filled in later by
            # the pass that does know k.
            salience_corrected=None,
        )
        return {"correction": stored}

    # -- audio -----------------------------------------------------------

    @app.get(f"{API_PREFIX}/audio")
    def audio(station_id: str | None = Query(default=None), since: str | None = Query(default=None),
              until: str | None = Query(default=None), species: str | None = Query(default=None),
              limit: int = Query(default=100, ge=1, le=100000), offset: int = Query(default=0, ge=0)):
        sp = species.strip() if species and species.strip() else None
        # Acoustic events only (trigger 'audio'); visual events live on Detections.
        total = db.count_observations(station_id=station_id, since=since, until=until, species=sp, trigger="audio")
        rows = db.list_observations(station_id=station_id, since=since, until=until, species=sp,
                                    trigger="audio", limit=limit, offset=offset)
        trust_index = db.event_trust_index()
        out = []
        for obs in rows:
            item = dict(obs)
            children = db.list_child_detections(obs["id"])
            item["audio_detections"] = [c for c in children if c.get("modality") == "audio"]
            item["verification"] = db.get_observation_verification(obs["id"])
            item["correction"] = db.latest_correction(obs["id"])
            item["event_trust"] = _event_trust(obs, children, trust_index)
            out.append(item)
        return {"items": out, "total": total, "limit": limit, "offset": offset}

    @app.get(f"{API_PREFIX}/audio/species")
    def audio_species(station_id: str | None = Query(default=None)):
        return db.list_species(station_id=station_id, modality="audio")

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
        """Every configured model, and whether its file is actually on disk.

        A configured path is a statement of intent, not proof that the file
        arrived. Several models are downloaded or exported by hand after setup, so
        reporting presence separately keeps a path that points at nothing from
        looking like a model that is ready to run.
        """
        stations_models = []
        for station_conf in settings.stations():
            stations_models.append({
                "station_id": station_conf.get("station_id"),
                "station_name": station_conf.get("station_name"),
                "models": station_conf.get("models", {}),
                "deployment": _station_deployment(station_conf),
            })
        desktop_models = settings.raw.get("desktop_models", {})

        # Collected without touching the loaded configuration, so nothing here can
        # write a derived value back into settings.json.
        files: dict = {}

        def _note(path) -> None:
            if not path or not isinstance(path, str) or path in files:
                return
            resolved = Path(path)
            if not resolved.is_absolute():
                resolved = Path(settings.repo_root) / resolved
            present = resolved.is_file()
            files[path] = {
                "present": present,
                "size_bytes": resolved.stat().st_size if present else None,
            }

        def _walk(entry) -> None:
            if isinstance(entry, dict):
                for key, value in entry.items():
                    if key == "path":
                        _note(value)
                    else:
                        _walk(value)

        _walk(desktop_models)
        for item in stations_models:
            _walk(item.get("models", {}))

        return {"desktop_models": desktop_models, "stations": stations_models, "files": files}

    @app.post(f"{API_PREFIX}/models/probe")
    def probe_model(request: ModelProbeRequest):
        """Confirm, at the moment a path is entered, whether its file is present.

        A path typed or pasted into a model field is checked here without waiting
        for a save or a trip elsewhere: the same quote and separator handling the
        save path uses is applied, so a path copied from a Windows dialog resolves
        to the same file, and the file's presence and size are reported so a
        zero-byte or partial download is visible. For an acoustic model the file's
        own audio shape is read and reported separately from any proposed sample
        rate, and when a labels file is given its line count is compared with the
        model's class-head width. No file type is ever required of an acoustic
        model; only visual slots have a fixed runtime format, reported on save.
        """
        raw = request.path
        if not isinstance(raw, str) or not raw.strip():
            return {"path": raw, "present": False, "size_bytes": None, "note": "no path given"}
        resolved_text = _normalise_entered_path(raw)
        resolved = Path(resolved_text)
        if not resolved.is_absolute():
            resolved = Path(settings.repo_root) / resolved
        present = resolved.exists()
        is_file = resolved.is_file()
        size_bytes = resolved.stat().st_size if is_file else None
        result: dict = {
            "path": resolved_text,
            "present": present,
            "is_file": is_file,
            "size_bytes": size_bytes,
        }

        # A visual slot is fixed by the runtime that loads it (a Hailo accelerator
        # reads .hef, the desktop reads .onnx), so the wrong file type is worth
        # flagging at the moment it is entered. Acoustic slots never enforce a
        # type, per the taxon-agnostic rule, so no suffix is checked for them.
        expected = _EXPECTED_MODEL_SUFFIXES.get(request.field or "")
        if expected:
            suffix = resolved.suffix.lower()
            result["expected_suffixes"] = sorted(expected)
            result["suffix_ok"] = (not suffix) or (suffix in expected)

        if (request.kind or "").lower() == "acoustic" and present:
            try:
                from audtheia.pipeline.acoustic import probe_acoustic_model
                proposals = _load_acoustic_shape_proposals(settings)
                probed = probe_acoustic_model(resolved, proposals=proposals)
                result["acoustic"] = probed
                class_count = probed.get("read", {}).get("class_count")
            except Exception as exc:  # noqa: BLE001 - an unprobeable model still reports presence
                result["acoustic"] = {"error": str(exc)}
                class_count = None

            labels_raw = request.labels_path
            if isinstance(labels_raw, str) and labels_raw.strip():
                labels_resolved = Path(_normalise_entered_path(labels_raw))
                if not labels_resolved.is_absolute():
                    labels_resolved = Path(settings.repo_root) / labels_resolved
                labels_present = labels_resolved.is_file()
                label_count = _count_label_lines(labels_resolved) if labels_present else None
                result["labels"] = {
                    "present": labels_present,
                    "count": label_count,
                    "matches_class_count": (
                        None if not (class_count and label_count) else class_count == label_count
                    ),
                }
        return result

    @app.get(f"{API_PREFIX}/brain/llm")
    def brain_llm():
        """Show the installed language models, which one is active, and the folder.

        The desktop dream pass and the verification interpreter both run this
        model. Selecting a different one takes effect the next time the station
        starts, since a model is loaded once when the desktop process begins.
        """
        status = _llm_status(settings)
        return {
            "configured": _redact(dict(settings.raw.get("desktop_models", {}).get("llm", {}))),
            "directory": str(_llm_directory(settings)),
            "available": _list_gguf_models(settings),
            "active": _active_llm_name(settings),
            "runtime_available": _llm_runtime_available(),
            "status": status["status"],
            "status_message": status["message"],
            "remedy": status["remedy"],
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

    @app.get(f"{API_PREFIX}/brain/retraining/candidates")
    def retraining_candidates(station_id=Query(default=None), confidence_below: float = Query(default=0.45)):
        """How many detections would be exported for retraining, and why."""
        from audtheia.analysis.retraining import candidate_summary

        return candidate_summary(db, station_id=station_id, confidence_below=confidence_below)

    @app.post(f"{API_PREFIX}/brain/retraining/export")
    def retraining_export(request: RetrainingExportRequest):
        """Write a retraining package for one modality and report what it holds.

        Exports are written on the desktop, which is where the record and the
        media live. Nothing leaves the machine; the result is a folder a person
        can open, correct, and feed into training.
        """
        if settings.node_role != "desktop":
            raise HTTPException(
                status_code=403,
                detail="retraining exports are produced on the desktop; this node is not the desktop.",
            )
        from audtheia.analysis.retraining import (
            RetrainingExportError, export_acoustic, export_vision,
        )

        kind = (request.kind or "").strip()
        if kind not in ("vision", "acoustic"):
            raise HTTPException(status_code=422, detail="kind must be 'vision' or 'acoustic'")
        builder = export_vision if kind == "vision" else export_acoustic
        try:
            return builder(
                db, settings,
                station_id=request.station_id,
                confidence_below=request.confidence_below if request.confidence_below is not None else 0.45,
                include_disagreements=True if request.include_disagreements is None else request.include_disagreements,
                include_deferred=True if request.include_deferred is None else request.include_deferred,
                force=bool(request.force),
            )
        except RetrainingExportError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"could not write the export: {exc}") from exc

    @app.get(f"{API_PREFIX}/brain/audit")
    def brain_audit(station_id=Query(default=None), since=Query(default=None), until=Query(default=None)):
        """Evidence of how the system behaved, derived from the stored record."""
        return _compute_audit(db, station_id=station_id, since=since, until=until)

    @app.get(f"{API_PREFIX}/brain/model-trust")
    def brain_model_trust():
        """Per-species model accuracy and per-model rollups, an inferred layer.

        A read-only sibling to the audit endpoint for callers that want only the
        model-trust snapshot: the per-species accuracy table sorted with the
        fine-tuning targets first, the micro and macro rollups per model, and the
        confusion counts, each keyed to a model version and tagged inference. With
        no expert reviews it returns an explicit empty-with-reason shape, not a
        table of zeros. Nothing here is written or alters a measurement.
        """
        return db.model_trust_snapshot()

    @app.get(f"{API_PREFIX}/brain/skills")
    def brain_skills(tier=Query(default=None)):
        # Attach how many events each field skill has flagged, counted from the
        # stored flags, so the panel can show a saved skill's real effect rather
        # than only its definition.
        skills = db.list_skills(tier=tier)
        counts = db.count_skill_flags_by_skill()
        for skill in skills:
            skill["flagged_events"] = int(counts.get(skill.get("id"), 0))
        return skills

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
            "condition": _clean_skill_condition(request.condition, tier),
        }

    def _clean_skill_condition(raw, tier) -> Optional[str]:
        """Validate a skill's checkable condition, or refuse it with a reason.

        The condition is what makes a field skill actually run, so it is checked
        against the same narrow vocabulary the field engine compiles, here at the
        door rather than silently failing later. An interpretive skill cannot
        carry one: it does not run at the field tier, so a condition on it would
        be a promise nothing keeps.
        """
        from audtheia.analysis.observation import parse_condition

        if raw in (None, "", {}):
            return None
        if tier != "deterministic_flag":
            raise HTTPException(
                status_code=422,
                detail="only a field skill can carry a checkable condition; an interpretive skill runs on the desktop",
            )
        if not isinstance(raw, dict):
            raise HTTPException(status_code=422, detail="a skill condition must be an object")
        spec = parse_condition(raw)
        if spec is None:
            raise HTTPException(
                status_code=422,
                detail=("this condition cannot be checked against a measured value. It needs a "
                        "source, a field, a comparison, and a number the record actually holds"),
            )
        return json.dumps(spec)

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
            condition=fields["condition"],
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
            condition=fields["condition"],
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

    @app.post(f"{API_PREFIX}/brain/skills/apply")
    def apply_skills(station_id: Optional[str] = Query(default=None)):
        """Apply the current field skills to existing records, now.

        New captures run field skills during quality control, but a record
        finalized before a skill existed is never revisited by it. This walks
        the record and re-evaluates every deterministic-flag skill over each
        event, recording any flag that now fires and clearing any that no longer
        does. It never alters a measured record or its quality-control decision.
        """
        if settings.node_role != "desktop":
            raise HTTPException(status_code=403, detail="processing runs on the desktop; this node is not the desktop.")
        from audtheia.app.orchestrator import DesktopStation

        scanned = 0
        flags = 0
        stations = _station_ids_for(station_id)
        try:
            for sid in stations:
                result = DesktopStation.build(settings, station_id=sid).apply_skills()
                scanned += result["scanned"]
                flags += result["flags"]
        except Exception as exc:  # noqa: BLE001 - reported as a clear client error
            raise HTTPException(status_code=422, detail=f"could not apply skills: {exc}") from exc
        note = (f"scanned {scanned} record(s); {flags} skill flag(s) now stand on the record"
                if scanned else "no records to scan yet")
        return {"ran": True, "scanned": scanned, "flags": flags, "stations": len(stations), "note": note}

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

    @app.delete(f"{API_PREFIX}/reports/{{bundle}}")
    def delete_report(bundle: str):
        """Delete one generated report bundle and everything inside it.

        Only a single bundle directly under the reports directory can be removed:
        the name is resolved and confirmed to be an immediate child of the reports
        directory, so a crafted name cannot escape it or reach any other part of
        the disk. Nothing outside the reports directory is ever touched, and the
        record, the database, and the captured media are never involved.
        """
        if settings.node_role != "desktop":
            raise HTTPException(status_code=403, detail="reports are managed on the desktop; this node is not the desktop.")
        reports_dir = Path(settings.path("reports_dir")).resolve()
        target = (reports_dir / bundle).resolve()
        # The target must be an immediate child of the reports directory: its
        # parent is exactly the reports directory, and it is not the directory
        # itself. This refuses '..', nested paths, and the reports root.
        if target.parent != reports_dir or target == reports_dir:
            raise HTTPException(status_code=400, detail="a report name must name one bundle in the reports folder")
        if not target.is_dir():
            raise HTTPException(status_code=404, detail=f"no report bundle named {bundle}")
        import shutil
        try:
            shutil.rmtree(target)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"could not delete the report: {exc}") from exc
        return {"status": "deleted", "bundle": bundle}

    # -- on-demand processing: run the longitudinal pass, quality control,
    # verification, and reports now, rather than waiting for the capture-time
    # scheduler. The scheduler only advances while capture is running and counts
    # elapsed capture-thread time, so a desktop that captures in bursts never
    # reaches a weekly cadence; these controls let a person exercise each stage
    # directly and see what it did.

    def _station_ids_for(station_id):
        """The station ids a run-now control should act on.

        A named station acts on that station alone; no station named acts on every
        configured station, so a control offered while viewing all stations does
        what the reader sees. An unknown id is refused rather than silently doing
        nothing.
        """
        if station_id:
            _require_station(station_id)
            return [station_id]
        return [s.get("station_id") for s in settings.stations() if s.get("station_id")]

    @app.post(f"{API_PREFIX}/qc/run")
    def qc_run(station_id: Optional[str] = Query(default=None)):
        """Finalize every observation still awaiting quality control, now."""
        if settings.node_role != "desktop":
            raise HTTPException(status_code=403, detail="processing runs on the desktop; this node is not the desktop.")
        from audtheia.app.orchestrator import DesktopStation

        finalized = 0
        stations = _station_ids_for(station_id)
        try:
            for sid in stations:
                finalized += DesktopStation.build(settings, station_id=sid).qc_pending()
        except Exception as exc:  # noqa: BLE001 - reported as a clear client error
            raise HTTPException(status_code=422, detail=f"could not run quality control: {exc}") from exc
        return {
            "ran": True, "finalized": finalized, "stations": len(stations),
            "note": ("finalized every record still awaiting quality control"
                     if finalized else "no record was awaiting quality control"),
        }

    @app.post(f"{API_PREFIX}/verify/run")
    def verify_run(station_id: Optional[str] = Query(default=None)):
        """Re-score and gate every eligible observation not yet verified, now."""
        if settings.node_role != "desktop":
            raise HTTPException(status_code=403, detail="processing runs on the desktop; this node is not the desktop.")
        from audtheia.app.orchestrator import DesktopStation

        verified = 0
        stations = _station_ids_for(station_id)
        try:
            for sid in stations:
                verified += DesktopStation.build(settings, station_id=sid).verify_pending()
        except Exception as exc:  # noqa: BLE001 - reported as a clear client error
            raise HTTPException(status_code=422, detail=f"could not run verification: {exc}") from exc
        # The desktop verifier is a single installation-wide model, so it gives a
        # meaningful verdict only for stations studying its target group; running
        # it over records it was not trained for is why a per-station verifier is
        # tracked but not built yet. That is stated so a zero here is read as scope,
        # not failure.
        note = (f"re-scored and gated {verified} eligible observation(s)" if verified
                else "no eligible observation was awaiting verification, or none fell in the desktop verifier's target group")
        return {"ran": True, "verified": verified, "stations": len(stations), "note": note}

    @app.post(f"{API_PREFIX}/dream/run")
    def dream_run(station_id: Optional[str] = Query(default=None)):
        """Run one longitudinal (dream) pass over the verified record, now."""
        if settings.node_role != "desktop":
            raise HTTPException(status_code=403, detail="the longitudinal pass runs on the desktop; this node is not the desktop.")
        active = next((p for p in db.list_dream_passes() if p.get("status") == "running"), None)
        if active:
            raise HTTPException(status_code=409, detail="a longitudinal pass is already running; let it finish or pause it first.")
        from audtheia.app.orchestrator import DesktopStation

        # The pass reads the whole verified record, so it is built against one
        # station only to wire the engine; its scope is the record, not that
        # station. A configured station id is used when present so the build is
        # deterministic even before any capture source is set.
        stations = _station_ids_for(station_id)
        if not stations:
            raise HTTPException(status_code=422, detail="configure a station first; the longitudinal pass needs the record a station produces.")
        try:
            result = DesktopStation.build(settings, station_id=stations[0]).dream_once()
        except Exception as exc:  # noqa: BLE001 - reported as a clear client error
            raise HTTPException(status_code=422, detail=f"could not run the longitudinal pass: {exc}") from exc
        narration = bool(_active_llm_name(settings))
        note = "candidate patterns are hypotheses tagged 'dream', never findings"
        if not narration:
            note += "; the desktop language model is unavailable, so this pass ran its structural half only, with no narration"

        # When a pass emits no patterns, explain why from the record itself, so a
        # zero reads as "not enough evidence yet" rather than "broken". Every
        # figure here is counted, never asserted: the verified count is the exact
        # generative gate the pass applies, and the thresholds are the floors it
        # measured each candidate against.
        diagnostics = None
        if result.patterns_emitted == 0:
            verified_count = len(db.list_pass_eligible_observation_ids())
            total_events = db.count_observations()
            dream_thresholds = settings.thresholds_config()["dream"]
            if total_events == 0:
                reason = ("No events in the record yet, so the pass had nothing to reason over. "
                          "A pattern is proposed only once confirmed events have been captured.")
            elif verified_count == 0:
                reason = (f"None of your {total_events} event(s) are confirmed yet. The pass builds "
                          "its baseline from the whole record but proposes a pattern only from events "
                          "the desktop verifier cleared or an expert confirmed, so verify or expertly "
                          "identify some events in Detections, then run the pass again.")
            else:
                reason = (f"{verified_count} confirmed event(s) in the record (desktop-verified or "
                          f"expert-identified). A candidate needs at "
                          f"least {dream_thresholds['min_events_for_correlation']} events for a "
                          f"correlation, {dream_thresholds['min_events_for_co_occurrence']} for a "
                          f"co-occurrence, or {dream_thresholds['min_periods_for_trend']} periods for "
                          f"a trend, and must clear an effect-size floor of "
                          f"{dream_thresholds['min_abs_effect']} and a p-value ceiling of "
                          f"{dream_thresholds['max_p_value']}. With this much evidence a zero means "
                          "not enough evidence yet, not a fault.")
            diagnostics = {
                "verified_count": verified_count,
                "total_events": total_events,
                "reason": reason,
            }

        return {
            "ran": True,
            "dream_pass_id": result.dream_pass_id,
            "status": result.status,
            "cycles_completed": result.cycles_completed,
            "observations_consolidated": result.observations_consolidated,
            "salience_scored": result.salience_scored,
            "patterns_emitted": result.patterns_emitted,
            "narration_available": narration,
            "note": note,
            "diagnostics": diagnostics,
        }

    @app.post(f"{API_PREFIX}/reports/run")
    def reports_run(request: ReportRequest):
        """Generate a report now and return where it was written.

        This runs synchronously so the control can report exactly what it produced,
        unlike POST /reports which schedules generation in the background.
        """
        if settings.node_role != "desktop":
            raise HTTPException(status_code=403, detail="report generation runs on the desktop; this node is not the desktop.")
        from audtheia.reports.generate import generate_report

        try:
            result = generate_report(
                settings, db,
                station_id=request.station_id, start=request.start, end=request.end,
                formats=request.formats, generated_at=_utc_now_iso(),
            )
        except Exception as exc:  # noqa: BLE001 - reported as a clear client error
            raise HTTPException(status_code=422, detail=f"could not generate the report: {exc}") from exc
        reports_dir = Path(settings.path("reports_dir")).resolve()

        def _rel(p):
            try:
                return str(Path(p).resolve().relative_to(reports_dir))
            except (ValueError, OSError):
                return str(p)

        return {
            "ran": True,
            "formats": result.formats,
            "bundle": Path(result.bundle_dir).name,
            "pdf": _rel(result.pdf_path) if result.pdf_path else None,
            "csv": [_rel(p) for p in result.csv_paths],
            "note": ("wrote " + ", ".join(result.formats) + " to the reports folder"
                     if result.formats else "no output format was produced"),
        }

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
        # Serve WAV clips as the widely-supported "audio/wav" rather than the
        # "audio/x-wav" mimetypes guesses, which some browsers refuse to play in
        # an <audio> element (the clip appears as 0:00 and never starts).
        media_type = "audio/wav" if target.suffix.lower() == ".wav" else None
        return FileResponse(str(target), media_type=media_type)

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
            warnings = list(warnings) + list(_persist_settings(settings) or [])
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

    @app.post(f"{API_PREFIX}/settings/stations/{{station_id}}/target-species", status_code=201)
    def add_target_species(station_id, request: TargetSpeciesRequest):
        """Add a target species to a station.

        A target species is a name the station is looking for; the reference fetch
        covers it, and the field model is trained on it. The name is appended to a
        working copy and the whole configuration is validated before it is written,
        so a malformed configuration is refused with the file left untouched. A
        name already present is accepted without duplicating it.
        """
        if settings.node_role != "desktop":
            raise HTTPException(status_code=403, detail="stations are configured on the desktop; this node is not the desktop.")
        name = (request.name or "").strip()
        if not name:
            raise HTTPException(status_code=422, detail="a species name is required")
        draft, station = _draft_station(station_id)
        existing = station.setdefault("target_species", [])
        if any(isinstance(n, str) and n.strip().lower() == name.lower() for n in existing):
            return {"config": _redact(settings.raw), "note": f"{name} is already a target species."}
        existing.append(name)
        try:
            _commit_draft(settings, draft)
        except BackendError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"config": _redact(settings.raw), "note": f"added {name}; the reference fetch will cover it."}

    @app.delete(f"{API_PREFIX}/settings/stations/{{station_id}}/target-species/{{name}}")
    def remove_target_species(station_id, name: str):
        """Remove a target species from a station."""
        if settings.node_role != "desktop":
            raise HTTPException(status_code=403, detail="stations are configured on the desktop; this node is not the desktop.")
        draft, station = _draft_station(station_id)
        targets = station.get("target_species", []) or []
        wanted = name.strip().lower()
        kept = [n for n in targets if not (isinstance(n, str) and n.strip().lower() == wanted)]
        if len(kept) == len(targets):
            raise HTTPException(status_code=404, detail=f"no target species named {name}")
        station["target_species"] = kept
        try:
            _commit_draft(settings, draft)
        except BackendError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"config": _redact(settings.raw), "note": f"removed {name}."}

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

    @app.post(f"{API_PREFIX}/settings/data-directory")
    def set_data_directory(request: DataDirectoryRequest):
        """Choose where captured data is stored, for example an external drive.

        Sets the data folder and the detections and GPS folders under it. It
        applies to new captures; data already written is not moved, so a person
        who wants the old data alongside the new copies it over once by hand. An
        absolute path is honored as given, which is how a store lives on another
        drive.
        """
        if settings.node_role != "desktop":
            raise HTTPException(status_code=403, detail="storage is configured on the desktop; this node is not the desktop.")
        raw = (request.path or "").strip()
        if not raw:
            raise HTTPException(status_code=422, detail="a folder path is required")
        base = str(Path(raw).expanduser()).replace("\\", "/").rstrip("/")
        if not base:
            raise HTTPException(status_code=422, detail="a folder path is required")
        draft = copy.deepcopy(settings.raw)
        paths = draft.setdefault("paths", {})
        paths["data_dir"] = base
        paths["detections_visual_dir"] = base + "/detections/visual"
        paths["detections_audio_dir"] = base + "/detections/audio"
        paths["gps_dir"] = base + "/gps"
        try:
            _commit_draft(settings, draft)
        except BackendError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"config": _redact(settings.raw),
                "note": "new captures will be stored here; data already on disk is not moved."}

    @app.post(f"{API_PREFIX}/storage/archive")
    def archive_storage(request: ArchiveRequest):
        """Copy captured frames to a chosen folder, optionally freeing the originals.

        Exports each event in the window to the destination with a metadata
        sidecar, and, when reclaim is set, removes those frames from the active
        store only after the copy is confirmed. The observation record is never
        changed, so the science stays; only the images move. The destination must
        be outside the captured-data folder.
        """
        if settings.node_role != "desktop":
            raise HTTPException(status_code=403, detail="archiving runs on the desktop; this node is not the desktop.")
        from audtheia.storage.archive import archive_events, ArchiveError
        try:
            result = archive_events(
                db, settings,
                target_dir=request.target_dir or "",
                start=request.start, end=request.end,
                station_id=request.station_id,
                reclaim=bool(request.reclaim),
            )
        except ArchiveError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - reported as a clear client error
            raise HTTPException(status_code=500, detail=f"could not archive: {exc}") from exc
        freed_mb = round((result.get("bytes_freed") or 0) / (1024 * 1024), 1)
        note = (f"archived {result['archived']} event(s) to {result['target']}"
                + (f"; reclaimed {result['reclaimed']} event(s), freeing about {freed_mb} MB"
                   if request.reclaim else "; originals were kept"))
        return {"ran": True, **result, "note": note}

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

    @app.post(f"{API_PREFIX}/stations/{{station_id}}/push-config")
    def push_config(station_id):
        """Push the station's current configuration to its already-connected Pi.

        A config edit on the desktop, such as adding a target species, reaches a
        Pi field station when it is pushed down. This sends only the updated
        configuration over the station's already-authorized key, without
        re-sending code or models, and the station applies it on its next start.
        It needs the Pi to have been connected once already (so the key is in
        place and the address is known); a station that has never been connected
        is asked to run the full connect flow in Settings first.
        """
        if settings.node_role != "desktop":
            raise HTTPException(status_code=403, detail="configuration is pushed from the desktop; this node is not the desktop.")
        _require_station(station_id)
        station = next((s for s in settings.stations() if s.get("station_id") == station_id), None)
        prov = (station or {}).get("provisioning") or {}
        host = (prov.get("host") or "").strip()
        user = (prov.get("user") or "").strip()
        port = int(prov.get("port") or 22)
        if not host or not user:
            raise HTTPException(
                status_code=409,
                detail="this station's Pi has not been connected yet; use Connect Pi in Settings first, then changes can be pushed.",
            )
        existing = _PROVISION_JOBS.get(station_id)
        if existing and existing["proc"].poll() is None:
            raise HTTPException(status_code=409, detail="a connection or push is already in progress for this station")

        log_path = Path(tempfile.gettempdir()) / f"audtheia-provision-{station_id}.log"
        log = open(log_path, "w", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                [sys.executable, str(_pi_script()), "--station-id", station_id,
                 "--host", host, "--user", user, "--port", str(port), "--key-auth", "--settings-only"],
                stdout=log, stderr=subprocess.STDOUT, cwd=str(settings.repo_root),
            )
        finally:
            log.close()
        _PROVISION_JOBS[station_id] = {"proc": proc, "log": str(log_path), "started": _utc_now_iso()}
        return {"state": "running", "note": "pushing the updated configuration to the Pi; poll the status endpoint for progress."}

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

    @app.post(f"{API_PREFIX}/capture/{{station_id}}/audio/start")
    def start_audio_capture(station_id):
        """Start desktop acoustic capture for a station in a background thread."""
        if settings.node_role != "desktop":
            raise HTTPException(status_code=403, detail="capture runs on the desktop; this node is not the desktop.")
        station = _require_station(station_id)
        source = ((station.get("capture", {}) or {}).get("source", {}) or {}).get("audio")
        if not source:
            raise HTTPException(status_code=422, detail="set a desktop audio source for this station first (Audio, Set audio source)")
        job = _AUDIO_CAPTURE_JOBS.get(station_id)
        if job and job.running():
            raise HTTPException(status_code=409, detail="audio capture is already running for this station")
        try:
            from audtheia.app.orchestrator import DesktopStation

            desktop = DesktopStation.build_audio(settings, station_id=station_id)
            live = desktop.start_audio_background()
        except Exception as exc:  # noqa: BLE001 - reported as a clear client error
            raise HTTPException(status_code=422, detail=f"could not start audio capture: {exc}") from exc
        _AUDIO_CAPTURE_JOBS[station_id] = live
        return {"state": "running", "note": "listening to the source; acoustic detections appear as the model recognizes calls."}

    @app.post(f"{API_PREFIX}/capture/{{station_id}}/audio/stop")
    def stop_audio_capture(station_id):
        """Stop a station's running desktop acoustic capture."""
        if settings.node_role != "desktop":
            raise HTTPException(status_code=403, detail="capture runs on the desktop; this node is not the desktop.")
        job = _AUDIO_CAPTURE_JOBS.get(station_id)
        if not job:
            return {"state": "idle"}
        try:
            job.stop()
        finally:
            _AUDIO_CAPTURE_JOBS.pop(station_id, None)
        return {"state": "stopped"}

    @app.get(f"{API_PREFIX}/capture/status")
    def capture_status():
        """Which stations are currently capturing (vision and audio) on the desktop."""
        running = [sid for sid, job in _CAPTURE_JOBS.items() if job.running()]
        running_audio = [sid for sid, job in _AUDIO_CAPTURE_JOBS.items() if job.running()]
        return {"running": running, "running_audio": running_audio}

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

    # Self-heal the schema on every launch, so a database created by an earlier
    # version gains any table added since (for example the per-frame review
    # table) with no manual migration step. This only ever creates a missing
    # table; it never alters or drops an existing one, so data is untouched. It
    # is best-effort: a schema that cannot be read must not stop the server from
    # starting on a database that is already complete.
    try:
        schema = settings.schema_path()
        if schema and Path(schema).is_file():
            database.ensure_schema(schema)
    except Exception as exc:  # noqa: BLE001 - startup must not hinge on a self-heal
        print(f"Note: could not auto-apply the schema ({exc}); "
              "an existing database still runs, but a newly added table may be missing.", file=sys.stderr)

    app = create_app(settings, database)
    server = settings.raw.get("server", {})
    host = server.get("host", "127.0.0.1")
    port = int(server.get("port", 8000))
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
