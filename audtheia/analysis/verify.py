"""Audtheia desktop verification and interpretation engine.

Path: audtheia/analysis/verify.py

The field station makes a fast screening call and hands the desktop a durable,
quality-controlled record. This module is the desktop step that re-checks that
call with the high-accuracy model, adds the interpretation the field tier is
forbidden from inventing, and opens the gate the longitudinal dream pass reads
before it is allowed to build any claim on a detection.

It is the occipital counterpart to the field tier: where the station runs a
deterministic reflex, the desktop looks again with a slower, more accurate eye
and, when warranted, overrides the reflex. The overriding verdict is written to
a desktop-owned table alongside the observation, never back onto the station's
own row, so the append-only pull from a station can never overwrite a desktop
result and the desktop never rewrites a value the station measured.

What the engine does to one record, in order:

  Re-score the event's saved frames with the verification model. A tracked
  encounter is many frames, and a single frame can carry a misclassification or
  a lucky-high confidence, so the verdict is the aggregate over every frame the
  verifier scored, not the reading of one representative frame. The aggregate
  names the taxon the verifier resolves, the confidence behind it, and whether
  it agrees with the field screening call, together with how many frames were
  scored and how many supported the resolved taxon.

  Decide the gate. A verification clears an observation only when the desktop
  model agrees with the field call and is confident enough to stand behind it.
  A disagreement or a weak re-score leaves the observation uncleared, which
  withholds it from the dream pass's generative phase while still letting it
  shape aggregate baselines. Either way the verdict is recorded, so a
  disagreement is preserved as evidence rather than silently dropped.

  Recompute authoritative salience. The desktop owns the authoritative salience
  slot and the ingredients the field tier could not compute against a full
  baseline. Until the salience combination formula is designed against real
  baseline statistics, the authoritative salience is set to the normalized
  verification confidence, and the remaining ingredients are retained wherever
  the interpreter can supply them, so the formula can be added later with no
  schema change and no rewrite here.

  Attach interpretation. Ecological role, rarity, behavioral context, and the
  other interpretive points are the desktop's to add, and every one of them is
  labelled as inference, never as measurement. The interpretation itself is
  produced by an injected interpreter (the desktop language model, and any
  interpretive-tier skills the deployment has defined); this engine only
  records what that interpreter returns, and refuses any point that is not a
  recognized interpretive point or that claims a provenance the desktop
  verification step is not allowed to write.

The measured-versus-inferred firewall is enforced here as a structural
property. This engine writes only desktop-owned rows: the verification verdict
and the interpretation points. It never writes an observation column, a child
detection, or an environmental reading, all of which the station owns, so a
desktop re-score can never masquerade as a field measurement. An interpretation
row is always inference; a verification verdict is a model fact kept distinct
from both the station's measurement and the dream tier's downstream patterns.

The engine reads a record by its identifier from the durable database and is
idempotent: an observation that already carries a verification result is left
exactly as it is unless a caller explicitly asks for a re-run, so a re-queued
identifier is harmless and interpretation rows are never duplicated. The model
and the interpreter are injected rather than imported, so the module loads and
its verification runs with no accelerator runtime, no language-model runtime,
and no frames on disk; the real model libraries live behind those seams and drop
in unchanged.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from audtheia.storage.database import (
    Database,
    Interpretation,
    ObservationVerification,
    new_id,
    utc_now_iso,
)

__all__ = [
    "FrameDetection",
    "VerificationVerdict",
    "InterpretationPoint",
    "VerifyResult",
    "FrameVerifier",
    "Interpreter",
    "VerifyEngine",
    "VerifyWorker",
    "QC_PASSED",
    "QC_DEFERRED",
    "ELIGIBLE_QC_STATES",
    "DATA_SOURCE_LLM_INFERRED",
    "PRODUCED_BY_VERIFY",
    "PRODUCED_BY_SKILL",
    "PRODUCED_BY_DREAM",
    "INTERPRETATION_POINT_TYPES",
    "POINT_TYPE_RARITY_SCORE",
    "DEFAULT_VERIFY_CLEAR_CONFIDENCE",
    "DEFAULT_MAX_FRAMES_SCORED",
    "DEFAULT_QUEUE_MAXSIZE",
]

logger = logging.getLogger("audtheia.analysis.verify")


# ---------------------------------------------------------------------------
# Which records this desktop step is allowed to verify. A record must already
# have been through field quality control: one the field tier passed, or one it
# could not classify and deferred to the desktop for exactly this adjudication.
# A still-pending record has not been quality-controlled yet and is not this
# step's input.
# ---------------------------------------------------------------------------
QC_PASSED = "qc_passed"
QC_DEFERRED = "qc_deferred"
ELIGIBLE_QC_STATES = frozenset({QC_PASSED, QC_DEFERRED})

# The acoustic clearance floor, used only for pure audio events. An audio event
# has no frame for the visual verifier to re-score, so it cannot be cross-checked
# by a second model; it is cleared for the dream pass's generative phase when its
# strongest acoustic detection meets this confidence floor. This is a weaker gate
# than the visual two-model check (it rests on the model's own confidence) and
# is set higher than the visual floor so only a strong call earns generative use.
# Overridable via analysis.thresholds.verification.acoustic_clear_confidence.
DEFAULT_ACOUSTIC_CLEAR_CONFIDENCE = 0.7

# The only provenance an interpretation may carry: it is inference, always.
DATA_SOURCE_LLM_INFERRED = "llm_inferred"

# Who produced an interpretation point. The desktop verification step writes its
# own analysis as 'verify' and an interpretive skill's output as 'skill'. A
# 'dream' point belongs to the downstream longitudinal pass and is refused here,
# so the verification step cannot write a dream-tier claim by mistake.
PRODUCED_BY_VERIFY = "verify"
PRODUCED_BY_SKILL = "skill"
PRODUCED_BY_DREAM = "dream"
_PRODUCED_BY_ALLOWED = frozenset({PRODUCED_BY_VERIFY, PRODUCED_BY_SKILL})

# The controlled set of interpretive point types the storage contract accepts.
# The engine records only these and never coins a new one, so a reader always
# sees interpretation drawn from a fixed, predictable vocabulary.
POINT_TYPE_RARITY_SCORE = "rarity_score"
INTERPRETATION_POINT_TYPES = frozenset(
    {
        "ecological_role",
        POINT_TYPE_RARITY_SCORE,
        "anomaly_flag",
        "cross_modal_attribution",
        "behavioral_context",
        "seasonal_assessment",
        "habitat_quality_flag",
        "interaction_pattern",
        "skill_note",
    }
)

# The confidence at or above which an agreeing re-score clears the verification
# gate. It is deliberately a middle value: a confident agreement clears, while a
# weak or contradicted re-score leaves the observation uncleared so the dream
# pass's generative phase never rests on it. This is a documented starting value
# with no configuration home yet; it is surfaced as a named constant here and
# can be promoted to a setting later with no change to the rest of this engine.
DEFAULT_VERIFY_CLEAR_CONFIDENCE = 0.50

# How many frames of one event the verifier is asked to score at most. Every
# stored frame of a long encounter can be thousands of inferences, so the set is
# bounded: the representative frame plus an even spread across the rest of the
# track, which catches a per-frame misclassification without re-scoring every
# frame. Also a documented starting value with no configuration home yet.
DEFAULT_MAX_FRAMES_SCORED = 32

# How many record identifiers may wait for the verification worker. Desktop
# verification is batch work, so the queue is only a convenience for feeding
# identifiers as records arrive; a sweep re-processes anything a full queue
# skipped, because every eligible record is durably stored.
DEFAULT_QUEUE_MAXSIZE = 512


# ===========================================================================
# Value types the engine consumes and produces
# ===========================================================================


@dataclass
class FrameDetection:
    """One frame's top detection as returned by the injected verifier.

    A frame with no detection is represented by a taxon of None; such a frame
    counts toward how many frames were scored but supports no taxon. The taxon
    key is the backbone usage key when the verifier resolved one, with the
    scientific name as the human-legible label.
    """

    gbif_usage_key: Optional[str] = None
    scientific_name: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class VerificationVerdict:
    """The aggregate of the verifier's per-frame detections for one event.

    resolved_gbif_usage_key and resolved_scientific_name name the taxon the
    verifier settled on across the scored frames; aggregate_confidence is the
    mean confidence of the frames that supported it, on the zero-to-one scale.
    agrees_with_field is True or False when there is a field label to compare
    against, and None when there is none (for example a pure audio event with no
    visual detection). frames_scored and frames_in_agreement make the verdict
    auditable: how many frames were looked at, and how many supported the
    resolved taxon.
    """

    resolved_gbif_usage_key: Optional[str] = None
    resolved_scientific_name: Optional[str] = None
    aggregate_confidence: Optional[float] = None
    agrees_with_field: Optional[bool] = None
    frames_scored: int = 0
    frames_in_agreement: int = 0


@dataclass
class InterpretationPoint:
    """One interpretive point as returned by the injected interpreter.

    point_type must be one of the recognized interpretive point types; value is
    the human-legible interpretation. produced_by is 'verify' for the desktop
    model's own analysis or 'skill' for an interpretive skill's output, in which
    case skill_id identifies the skill. numeric_value is an optional
    machine-usable number that travels with a point which also has one: a rarity
    point can carry both a legible characterization and the number the salience
    ingredient slot retains, so the qualitative point and the numeric ingredient
    stay in step without the engine inventing either.
    """

    point_type: str
    value: str
    produced_by: str = PRODUCED_BY_VERIFY
    confidence: Optional[float] = None
    skill_id: Optional[str] = None
    model_version: Optional[str] = None
    numeric_value: Optional[float] = None


@dataclass
class VerifyResult:
    """The outcome of running the engine on one record.

    outcome is one of: verified (the gate was opened), unverified (a verdict was
    recorded but the gate stayed closed, including an override that contradicted
    the field call), skipped (nothing to do, for example the record was not
    found, was not eligible, or was already verified). verdict is the aggregate
    re-score when one was produced, and interpretations_written counts the
    interpretive points recorded.
    """

    observation_id: str
    outcome: str  # "verified" | "unverified" | "skipped"
    verified: int = 0
    verdict: Optional[VerificationVerdict] = None
    interpretations_written: int = 0
    detail: Optional[str] = None


@dataclass
class VerificationContext:
    """Everything the interpreter is given about one event.

    The interpreter reads the observation core, its per-taxon detections, its
    environmental readings, and the verification verdict, and returns a list of
    interpretation points. Passing the verdict lets an interpreter reason about
    a disagreement between the field call and the desktop re-score.
    """

    observation: dict
    child_detections: list[dict]
    environmental_readings: list[dict]
    verdict: VerificationVerdict
    field_gbif_usage_key: Optional[str] = None
    field_scientific_name: Optional[str] = None
    # The interpretive-tier skills the deployment has defined. The interpreter
    # applies each one and returns its output as a point tagged produced_by
    # 'skill' with its skill_id, which the engine records as labelled inference.
    # Empty when none are defined; the desktop model still runs its own analysis.
    interpretive_skills: list = field(default_factory=list)


# A frame verifier turns a list of resolved frame paths into one detection per
# frame. It is injected, so the accelerator runtime that actually runs the model
# is never imported by this module and the engine runs against a stand-in with
# no model present. An interpreter turns the verification context into a list of
# interpretation points; it is injected for the same reason, so no language-model
# runtime is imported here either.
FrameVerifier = Callable[..., list]
Interpreter = Callable[..., list]


# ===========================================================================
# The engine
# ===========================================================================


class VerifyEngine:
    """Desktop re-verification, authoritative salience, and interpretation.

    Construct it with the validated configuration, the storage layer, an
    injected frame verifier, and an injected interpreter. Call process with a
    record identifier to re-score the event, record the verdict, recompute the
    authoritative salience, set the verification gate, and attach the
    interpreter's points. The engine holds no per-record state, so one instance
    safely serves a worker draining a queue.
    """

    def __init__(
        self,
        *,
        settings,
        db: Database,
        verifier,
        interpreter,
        clear_confidence: Optional[float] = None,
        max_frames_scored: Optional[int] = None,
        acoustic_clear_confidence: Optional[float] = None,
    ) -> None:
        self._settings = settings
        self._db = db
        self._verifier = verifier
        self._interpreter = interpreter
        # These thresholds come from configuration, with an explicit constructor
        # argument taking precedence when one is passed. The configured defaults
        # reproduce the values these held before they became configurable, so an
        # unset configuration changes nothing.
        verification_thresholds = settings.thresholds_config()["verification"]
        self._clear_confidence = float(
            clear_confidence if clear_confidence is not None
            else verification_thresholds["clear_confidence"]
        )
        self._max_frames_scored = int(
            max_frames_scored if max_frames_scored is not None
            else verification_thresholds["max_frames_scored"]
        )
        self._acoustic_clear_confidence = float(
            acoustic_clear_confidence if acoustic_clear_confidence is not None
            else verification_thresholds.get("acoustic_clear_confidence", DEFAULT_ACOUSTIC_CLEAR_CONFIDENCE)
        )

        # Visible counters, so a worker's progress is observable without reaching
        # into the engine.
        self.processed = 0
        self.verified = 0
        self.unverified = 0
        self.skipped = 0
        self.interpretations_written = 0

    # -- public entry point ----------------------------------------------

    def process(self, observation_id: str, *, force: bool = False) -> VerifyResult:
        """Verify one record, by identifier.

        Reads the record, re-scores its frames, records the verdict and the
        authoritative salience, sets the verification gate, and attaches the
        interpreter's points. Doing nothing is a valid outcome: a record that
        is missing, not eligible for verification, or already verified is left
        exactly as it is. Passing force re-runs a record that already has a
        verification result, replacing the verdict and adding interpretation
        again, which a caller uses only to deliberately re-verify.
        """
        self.processed += 1

        obs = self._db.get_observation(observation_id)
        if obs is None:
            return self._skip(observation_id, "record not found")

        qc_state = obs.get("qc_state")
        if qc_state not in ELIGIBLE_QC_STATES:
            return self._skip(
                observation_id,
                f"record is {qc_state!r}, not eligible for verification",
            )

        existing = self._db.get_observation_verification(observation_id)
        if existing is not None and not force:
            return self._skip(observation_id, "record already verified")

        children = self._db.list_child_detections(observation_id)

        # A pure acoustic event has no frame for the visual verifier to re-score,
        # so it takes the acoustic-confidence gate instead of a second-model
        # cross-check. Handled and returned here before any frame work.
        if str(obs.get("trigger_source")) == "audio":
            return self._process_acoustic(observation_id, children)

        readings = self._db.list_environmental_readings(observation_id)

        field_key, field_name = self._field_label(children)

        # Re-score the event's frames and aggregate the per-frame detections into
        # one verdict. The verifier does the model work behind the seam; the
        # engine only resolves which frames to hand it and how to combine what
        # comes back.
        frame_paths = self._resolve_frames(obs)
        frame_detections = self._score_frames(frame_paths)
        verdict = self._aggregate(frame_detections, field_key, field_name)

        cleared = self._gate(verdict)

        # Ask the interpreter for the desktop's interpretive points. The engine
        # records what the interpreter returns and never parses a skill's text
        # itself, so an interpretive claim can only enter as a labelled
        # inference, never as a measured value.
        context = VerificationContext(
            observation=obs,
            child_detections=children,
            environmental_readings=readings,
            verdict=verdict,
            field_gbif_usage_key=field_key,
            field_scientific_name=field_name,
            interpretive_skills=self._db.list_skills(tier="interpretive"),
        )
        points = self._interpret(context)

        rarity_numeric = self._rarity_ingredient(points)

        self._write_verification(
            observation_id,
            verdict=verdict,
            cleared=cleared,
            rarity_score=rarity_numeric,
        )
        written = self._write_interpretations(observation_id, points)

        return self._finalize(
            observation_id, cleared=cleared, verdict=verdict, written=written
        )

    def _process_acoustic(self, observation_id: str, children: list[dict]) -> VerifyResult:
        """Clear a pure acoustic event by the acoustic-confidence gate.

        An audio event has no frame to re-score, so it cannot be cross-checked by
        a second model the way a visual event is. It is cleared when its strongest
        acoustic detection meets the configured confidence floor, an honest,
        weaker gate than the visual two-model check, recorded as an acoustic
        clearance with every RF-DETR field left null because no RF-DETR ran. The
        peak confidence is not copied into the verdict's rfdetr_* columns, which
        are reserved for the visual verifier; it already lives on the child
        acoustic detections and in the audio audit.
        """
        confidences = [
            float(c["confidence"]) for c in children
            if c.get("modality") == "audio" and c.get("confidence") is not None
        ]
        peak = max(confidences) if confidences else None
        cleared = peak is not None and peak >= self._acoustic_clear_confidence
        self._write_acoustic_clearance(observation_id, cleared=cleared)
        verdict = VerificationVerdict(
            aggregate_confidence=peak, frames_scored=0, frames_in_agreement=0
        )
        return self._finalize(observation_id, cleared=cleared, verdict=verdict, written=0)

    def _write_acoustic_clearance(self, observation_id: str, *, cleared: bool) -> None:
        """Record an acoustic clearance in the desktop-owned verification table.

        `verified` is the only gate the dream pass reads, and it is set from the
        acoustic confidence gate. Every rfdetr_* field is null because no RF-DETR
        ran, and the authoritative salience is left unset so the field-provisional
        salience stands until a baseline recompute exists, matching how a
        frameless event is handled elsewhere.
        """
        now = utc_now_iso()
        verification = ObservationVerification(
            observation_id=observation_id,
            created_at=now,
            verified=1 if cleared else 0,
            rfdetr_version=None,
            rfdetr_gbif_usage_key=None,
            rfdetr_scientific_name=None,
            rfdetr_confidence=None,
            rfdetr_agrees_with_field=None,
            frames_scored=0,
            frames_in_agreement=0,
            salience_authoritative=None,
            rarity_score=None,
            baseline_deviation=None,
            anomaly_magnitude_authoritative=None,
            verified_at=now if cleared else None,
        )
        self._db.upsert_observation_verification(verification)

    def sweep(
        self, *, station_id: Optional[str] = None, limit: Optional[int] = None
    ) -> int:
        """Verify every eligible record that is not verified yet.

        Walks the observations (optionally for one station), processing each
        that has passed or been deferred by field quality control and does not
        already carry a verification result. Returns how many records were
        advanced to a verified or unverified outcome this call. Safe to call at
        any time: the per-record idempotency guard skips anything already done.
        """
        advanced = 0
        for row in self._db.list_observations(station_id=station_id, limit=limit):
            if row.get("qc_state") not in ELIGIBLE_QC_STATES:
                continue
            result = self.process(row["id"])
            if result.outcome in ("verified", "unverified"):
                advanced += 1
        return advanced

    # -- frames ----------------------------------------------------------

    def _resolve_frames(self, obs: dict) -> list[Path]:
        """Choose which of an event's saved frames to re-score.

        Always includes the representative frame when there is one. When the
        event's on-disk frame folder is present, the remaining frames are spread
        evenly and added up to the frame cap, so a long track is sampled across
        its length rather than only at its strongest moment. Paths are resolved
        through the configuration so the same relative layout works on every
        operating system; the verifier, not this engine, opens them.
        """
        paths: list[Path] = []
        seen: set[str] = set()

        representative = obs.get("representative_frame")
        if representative:
            rep_path = self._settings.resolve_path(representative)
            paths.append(rep_path)
            seen.add(str(rep_path))

        folder = self._event_frame_folder(obs)
        if folder is not None and folder.exists():
            try:
                fmt = self._settings.image_encoding().get("format", "jpg")
            except Exception:  # noqa: BLE001 - a missing media block falls back to a common format
                fmt = "jpg"
            others = sorted(
                p for p in folder.glob(f"*.{fmt}") if str(p) not in seen
            )
            for p in self._even_spread(others, self._max_frames_scored - len(paths)):
                paths.append(p)
                seen.add(str(p))

        return paths

    def _event_frame_folder(self, obs: dict) -> Optional[Path]:
        """The event's frame folder, keyed by the event's systematic name.

        Each event stores its frames under the visual-detections directory in a
        folder named for the event, so the desktop can find every retained frame
        without a per-frame database row. Returns None when the configuration
        does not define the directory.
        """
        event_name = obs.get("event_name")
        if not event_name:
            return None
        try:
            base = Path(self._settings.path("detections_visual_dir"))
        except Exception:  # noqa: BLE001 - no configured directory means no folder to scan
            return None
        return base / event_name

    @staticmethod
    def _even_spread(items: list, count: int) -> list:
        """Pick at most count items spread evenly across the list.

        An even spread samples the whole track rather than a contiguous run, so
        a misclassification part way through an encounter is as likely to be
        seen as one at the start.
        """
        if count <= 0 or not items:
            return []
        if len(items) <= count:
            return list(items)
        step = len(items) / count
        return [items[int(i * step)] for i in range(count)]

    def _score_frames(self, frame_paths: list[Path]) -> list[FrameDetection]:
        """Run the injected verifier and normalize its output to detections.

        The verifier is duck-typed: it exposes a callable that takes the frame
        paths and returns one detection per frame. Whatever shape it returns is
        coerced to FrameDetection here, so the aggregation logic downstream sees
        a single consistent type.
        """
        raw = self._verifier.verify_frames(frame_paths)
        out: list[FrameDetection] = []
        for item in raw or []:
            if isinstance(item, FrameDetection):
                out.append(item)
            elif isinstance(item, dict):
                out.append(
                    FrameDetection(
                        gbif_usage_key=item.get("gbif_usage_key"),
                        scientific_name=item.get("scientific_name"),
                        confidence=item.get("confidence"),
                    )
                )
            else:
                out.append(
                    FrameDetection(
                        gbif_usage_key=getattr(item, "gbif_usage_key", None),
                        scientific_name=getattr(item, "scientific_name", None),
                        confidence=getattr(item, "confidence", None),
                    )
                )
        return out

    # -- aggregation and the gate ----------------------------------------

    @staticmethod
    def _field_label(children: list[dict]) -> tuple[Optional[str], Optional[str]]:
        """The field screening call's taxon: the strongest visual detection.

        The event-level resolved taxon is the highest-confidence visual child
        detection. Returns the taxon key and scientific name, or a pair of None
        when the event carries no visual detection to compare against.
        """
        best = None
        best_conf = None
        for c in children:
            if c.get("modality") != "vision":
                continue
            conf = c.get("confidence")
            if conf is None:
                continue
            if best_conf is None or conf > best_conf:
                best_conf = conf
                best = c
        if best is None:
            return None, None
        return best.get("gbif_usage_key"), best.get("scientific_name")

    def _aggregate(
        self,
        detections: list[FrameDetection],
        field_key: Optional[str],
        field_name: Optional[str],
    ) -> VerificationVerdict:
        """Combine per-frame detections into one verdict for the event.

        The resolved taxon is the one the most frames supported, with total
        confidence breaking a tie; the aggregate confidence is the mean over the
        frames that supported it. Agreement compares the resolved taxon to the
        field call by backbone key when both have one, otherwise by scientific
        name; it is None when there is no field label to compare against.
        """
        frames_scored = len(detections)

        # Group supporting frames by taxon. A frame with no taxon is counted as
        # scored but supports nothing.
        by_taxon: dict[tuple, dict] = {}
        for d in detections:
            key = d.gbif_usage_key
            name = d.scientific_name
            if key is None and name is None:
                continue
            taxon = (key, name)
            slot = by_taxon.setdefault(
                taxon, {"count": 0, "confidence_sum": 0.0, "confidences": []}
            )
            slot["count"] += 1
            if d.confidence is not None:
                slot["confidence_sum"] += d.confidence
                slot["confidences"].append(d.confidence)

        if not by_taxon:
            return VerificationVerdict(
                frames_scored=frames_scored, frames_in_agreement=0
            )

        winner = max(
            by_taxon.items(),
            key=lambda kv: (kv[1]["count"], kv[1]["confidence_sum"]),
        )
        (res_key, res_name), stats = winner
        confidences = stats["confidences"]
        aggregate_confidence = (
            sum(confidences) / len(confidences) if confidences else None
        )

        agrees = self._agrees(res_key, res_name, field_key, field_name)

        return VerificationVerdict(
            resolved_gbif_usage_key=res_key,
            resolved_scientific_name=res_name,
            aggregate_confidence=aggregate_confidence,
            agrees_with_field=agrees,
            frames_scored=frames_scored,
            frames_in_agreement=stats["count"],
        )

    @staticmethod
    def _agrees(
        res_key: Optional[str],
        res_name: Optional[str],
        field_key: Optional[str],
        field_name: Optional[str],
    ) -> Optional[bool]:
        if field_key is None and field_name is None:
            return None
        if res_key is not None and field_key is not None:
            return res_key == field_key
        if res_name is not None and field_name is not None:
            return res_name.strip().lower() == field_name.strip().lower()
        return False

    def _gate(self, verdict: VerificationVerdict) -> bool:
        """Whether this verdict clears the observation for the dream pass.

        A verification clears an observation only when the desktop model agreed
        with the field call and was confident enough to stand behind it. A
        disagreement, an uncertain re-score, or an event with no frame to score
        leaves the observation uncleared.
        """
        if verdict.agrees_with_field is not True:
            return False
        if verdict.aggregate_confidence is None:
            return False
        return verdict.aggregate_confidence >= self._clear_confidence

    # -- interpretation --------------------------------------------------

    def _interpret(self, context: VerificationContext) -> list[InterpretationPoint]:
        """Run the injected interpreter and normalize its output to points.

        The interpreter is duck-typed and returns the desktop's interpretive
        points for the event. Whatever shape it returns is coerced to
        InterpretationPoint here; validation of each point happens at write time.
        """
        raw = self._interpreter.interpret(context)
        out: list[InterpretationPoint] = []
        for item in raw or []:
            if isinstance(item, InterpretationPoint):
                out.append(item)
            elif isinstance(item, dict):
                out.append(
                    InterpretationPoint(
                        point_type=item.get("point_type"),
                        value=item.get("value"),
                        produced_by=item.get("produced_by", PRODUCED_BY_VERIFY),
                        confidence=item.get("confidence"),
                        skill_id=item.get("skill_id"),
                        model_version=item.get("model_version"),
                        numeric_value=item.get("numeric_value"),
                    )
                )
            else:
                out.append(
                    InterpretationPoint(
                        point_type=getattr(item, "point_type", None),
                        value=getattr(item, "value", None),
                        produced_by=getattr(item, "produced_by", PRODUCED_BY_VERIFY),
                        confidence=getattr(item, "confidence", None),
                        skill_id=getattr(item, "skill_id", None),
                        model_version=getattr(item, "model_version", None),
                        numeric_value=getattr(item, "numeric_value", None),
                    )
                )
        return out

    @staticmethod
    def _rarity_ingredient(points: list[InterpretationPoint]) -> Optional[float]:
        """The numeric rarity a rarity point carries, if any.

        Rarity has two homes on purpose: the labelled interpretive point that a
        report shows, and the numeric ingredient the authoritative salience will
        consume once its formula is designed. The engine copies the number from
        whatever rarity point supplies one, and invents nothing when none does.
        """
        for p in points:
            if p.point_type == POINT_TYPE_RARITY_SCORE and p.numeric_value is not None:
                return float(p.numeric_value)
        return None

    def _interpreter_version(self) -> Optional[str]:
        version = getattr(self._interpreter, "version", None)
        if version:
            return version
        return self._desktop_model_version("llm")

    def _verifier_version(self) -> Optional[str]:
        version = getattr(self._verifier, "version", None)
        if version:
            return version
        return self._desktop_model_version("visual_rfdetr")

    def _desktop_model_version(self, key: str) -> Optional[str]:
        try:
            return (
                self._settings.raw.get("desktop_models", {}).get(key, {}).get("version")
            )
        except Exception:  # noqa: BLE001 - a missing block simply leaves the version unstamped
            return None

    # -- writes ----------------------------------------------------------

    def _write_verification(
        self,
        observation_id: str,
        *,
        verdict: VerificationVerdict,
        cleared: bool,
        rarity_score: Optional[float],
    ) -> None:
        """Record the verdict, the gate, and the authoritative salience.

        Everything written here lives in the desktop-owned verification table,
        so a station-to-desktop pull can never overwrite it and this write never
        touches a station-owned column. The authoritative salience is set to the
        normalized verification confidence as a documented interim, with the
        remaining ingredients retained where they are available, so the salience
        combination formula can be added later against real baseline statistics
        with no schema change.
        """
        now = utc_now_iso()
        agrees = verdict.agrees_with_field
        verification = ObservationVerification(
            observation_id=observation_id,
            created_at=now,
            verified=1 if cleared else 0,
            rfdetr_version=self._verifier_version(),
            rfdetr_gbif_usage_key=verdict.resolved_gbif_usage_key,
            rfdetr_scientific_name=verdict.resolved_scientific_name,
            rfdetr_confidence=verdict.aggregate_confidence,
            rfdetr_agrees_with_field=(None if agrees is None else (1 if agrees else 0)),
            frames_scored=verdict.frames_scored,
            frames_in_agreement=verdict.frames_in_agreement,
            salience_authoritative=verdict.aggregate_confidence,
            rarity_score=rarity_score,
            baseline_deviation=None,
            anomaly_magnitude_authoritative=None,
            verified_at=now if cleared else None,
        )
        self._db.upsert_observation_verification(verification)

    def _write_interpretations(
        self, observation_id: str, points: list[InterpretationPoint]
    ) -> int:
        """Write each valid interpretation point as a labelled inference.

        A point is written only when it names a recognized interpretive point
        type and a provenance the verification step is allowed to write; a skill
        point must identify its skill. Every stored point is inference, so its
        provenance is fixed to the inferred vocabulary and can never be recorded
        as a measurement. An invalid point is refused and logged rather than
        stored, which keeps a malformed interpreter response from contaminating
        the record.
        """
        written = 0
        for p in points:
            reason = self._point_rejection(p)
            if reason is not None:
                logger.error(
                    "refusing an interpretation point for observation %s: %s",
                    observation_id,
                    reason,
                )
                continue
            interpretation = Interpretation(
                id=new_id(),
                observation_id=observation_id,
                point_type=p.point_type,
                value=str(p.value),
                produced_by=p.produced_by,
                created_at=utc_now_iso(),
                data_source=DATA_SOURCE_LLM_INFERRED,
                confidence=p.confidence,
                model_version=p.model_version or self._interpreter_version(),
                skill_id=p.skill_id if p.produced_by == PRODUCED_BY_SKILL else None,
            )
            self._db.insert_interpretation(interpretation)
            written += 1
        self.interpretations_written += written
        return written

    @staticmethod
    def _point_rejection(p: InterpretationPoint) -> Optional[str]:
        """Why a point may not be written, or None when it is valid."""
        if p.point_type not in INTERPRETATION_POINT_TYPES:
            return f"unrecognized interpretive point type {p.point_type!r}"
        if p.value is None:
            return "interpretation has no value"
        if p.produced_by not in _PRODUCED_BY_ALLOWED:
            return (
                f"produced_by {p.produced_by!r} is not one the verification step "
                f"may write"
            )
        if p.produced_by == PRODUCED_BY_SKILL and not p.skill_id:
            return "a skill point must identify its skill"
        return None

    # -- finalize --------------------------------------------------------

    def _finalize(
        self,
        observation_id: str,
        *,
        cleared: bool,
        verdict: VerificationVerdict,
        written: int,
    ) -> VerifyResult:
        if cleared:
            self.verified += 1
            outcome = "verified"
        else:
            self.unverified += 1
            outcome = "unverified"
        return VerifyResult(
            observation_id=observation_id,
            outcome=outcome,
            verified=1 if cleared else 0,
            verdict=verdict,
            interpretations_written=written,
        )

    def _skip(self, observation_id: str, detail: str) -> VerifyResult:
        self.skipped += 1
        return VerifyResult(
            observation_id=observation_id,
            outcome="skipped",
            verified=0,
            detail=detail,
        )


# ===========================================================================
# The bounded queue and worker
# ===========================================================================


class VerifyWorker:
    """A bounded queue of record identifiers drained by a background worker.

    A caller submits an identifier when a record is ready for desktop
    verification, for example as a station's records arrive on sync. The worker
    thread drains the queue and runs the engine on each identifier. Submission
    never blocks the caller: a full queue drops the identifier and counts it,
    because the record is safely in the database and a sweep re-processes any
    eligible record a full queue skipped. One bad record never stops the worker;
    its failure is logged and the worker moves on.
    """

    _SENTINEL = object()

    def __init__(self, engine: VerifyEngine, *, maxsize: int = DEFAULT_QUEUE_MAXSIZE) -> None:
        self._engine = engine
        self._queue: "queue.Queue[object]" = queue.Queue(maxsize=maxsize)
        self._thread: Optional[threading.Thread] = None

        self.submitted = 0
        self.queue_saturation_events = 0
        self.processed = 0
        self.failed = 0

    def submit(self, observation_id: str) -> bool:
        """Queue one record for verification without ever blocking the caller."""
        try:
            self._queue.put_nowait(observation_id)
            self.submitted += 1
            return True
        except queue.Full:
            self.queue_saturation_events += 1
            logger.warning(
                "verification queue full; record %s stays eligible for a sweep "
                "(saturation count %d)",
                observation_id,
                self.queue_saturation_events,
            )
            return False

    def start(self) -> None:
        """Start the background worker draining the queue."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="audtheia-verify-worker", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the worker after it finishes the queued work."""
        if self._thread is None:
            return
        self._queue.put(self._SENTINEL)
        self._thread.join()
        self._thread = None

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._SENTINEL:
                break
            try:
                self._engine.process(item)
                self.processed += 1
            except Exception:  # noqa: BLE001 - one bad record must not stop the worker
                self.failed += 1
                logger.exception("verification failed for record %s", item)
