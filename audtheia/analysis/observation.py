"""Audtheia field-station quality-control and consolidation engine.

Path: audtheia/analysis/observation.py

Capture writes a raw observation the instant an encounter closes: the event
row, its per-taxon detections, and whatever sensor channels were gathered, all
stamped as pending quality control. This module is the step that takes such a
pending record and makes it complete, well-formed, and ready for the desktop.
It is the cerebellar forward model made literal: a deterministic predict,
compare, correct loop in plain Python that runs on the station's own processor,
never a language model. The same record always yields the same result, so the
outcome is reproducible and its marginal energy cost is near zero, which is
exactly why the field tier carries no model on its hot path.

What the engine does to one record, in order:

  Predict what a complete record should look like from the station's own
  configuration: the set of sensor channels the deployment enabled is the
  manifest every observation is measured against.

  Compare the captured record against that prediction and against the storage
  contract: is the event window coherent, does a visual event carry a visual
  detection, does an audio event carry an audio detection, is every enabled
  channel accounted for, and is every stored value labelled with a provenance
  that belongs at the field tier.

  Correct the gaps without ever inventing a measurement. A channel the
  configuration expected but that produced no reading is filled with an
  explicit "not measured" status rather than left as a silent hole, so a reader
  can always tell the difference between a value that was taken and one that was
  not. The provisional salience the desktop will later recompute is confirmed,
  or written from the detection's own confidence if capture left it empty,
  always to the provisional slot alone.

A record the engine cannot classify is never guessed at. It is stamped as
deferred, with a controlled reason code that tells the desktop why, and routed
onward. Interpretation lives on the desktop; the field engine only ever states
what was measured.

The measured-versus-inferred firewall is enforced here as a structural
property, not a hope. A field observation may only carry field-capture
provenance; a record that arrives tagged as inferred or dream-derived content
is refused and routed to the desktop rather than accepted as a measurement.
User-defined skills run here only when they are the deterministic kind, whose
trigger and output are pure functions of the measured values already in the
record; an interpretive skill is never executed at the field tier, and a skill
that tries to emit anything other than a plain measured or derived flag has its
output rejected. The engine has no path that writes interpretation, so it
cannot emit a free-form ecological claim as authoritative data even by mistake.

The engine reads a record by its identifier from the durable database rather
than being handed the record in memory. Capture has already stored the record,
so re-reading it is safe and idempotent: a crash partway through quality
control leaves the record pending and it is simply processed again, with no
double write. A bounded queue feeds identifiers to a worker so that capture and
detection never wait on quality control; if the queue is momentarily full the
identifier is dropped from the queue rather than blocking capture, because the
record is still safely in the database and a sweep re-processes any pending
record that a full queue skipped.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

from audtheia.config import ConfigError
from audtheia.storage.database import (
    Database,
    EnvironmentalReading,
    SkillFlagRow,
    new_id,
    utc_now_iso,
)

__all__ = [
    "SkillFlag",
    "ConsolidatedSnapshot",
    "QCResult",
    "QCEngine",
    "QCWorker",
    "FlagEvaluator",
    "compile_condition",
    "parse_condition",
    "CONDITION_SOURCES",
    "CONDITION_OBSERVATION_FIELDS",
    "CONDITION_DETECTION_FIELDS",
    "CONDITION_TIME_FIELDS",
    "CONDITION_AGGREGATES",
    "CONDITION_OPS",
    "QC_PASSED",
    "QC_DEFERRED",
    "QC_PENDING",
    "QC_VERIFIED",
    "REASON_SCHEMA_NOVEL_SHAPE",
    "REASON_INCOMPLETE_RECORD",
    "REASON_SENSOR_CONFLICT",
    "REASON_LOW_CONFIDENCE",
    "REASON_MANUAL_REVIEW",
    "REASON_FIREWALL_VIOLATION",
    "STATUS_MEASURED",
    "STATUS_NOT_MEASURED",
    "STATUS_BELOW_DETECTION_LIMIT",
    "STATUS_SENSOR_ERROR",
    "STATUS_NOT_APPLICABLE",
    "QARTOD_MISSING",
    "TIER_DETERMINISTIC_FLAG",
    "TIER_INTERPRETIVE",
    "FIELD_CAPTURE_PROVENANCE",
    "ANALYSIS_LOCATION_PI",
    "DEFAULT_FIELD_PASS_CONFIDENCE",
    "DEFAULT_QUEUE_MAXSIZE",
]

logger = logging.getLogger("audtheia.analysis.observation")


# ---------------------------------------------------------------------------
# Record lifecycle states, mirroring the storage contract's qc_state values.
# A record starts pending; the engine advances it to passed, or defers it for
# the desktop; verified is reached later, only on the desktop.
# ---------------------------------------------------------------------------
QC_PENDING = "qc_pending"
QC_PASSED = "qc_passed"
QC_DEFERRED = "qc_deferred"
QC_VERIFIED = "verified"

# The controlled reason codes recorded when a record is deferred. The engine
# selects exactly one of these and never coins a new term, so desktop triage
# reads from a fixed, predictable set. schema_novel_shape means the record's
# shape could not be validated; incomplete_record means required event detail
# was missing or inconsistent and could not be repaired deterministically;
# sensor_conflict means a channel's stored value and its status contradict each
# other; low_confidence_unclassified means the detection was too weak for the
# field tier to pass on its own and the desktop should adjudicate;
# manual_review_requested means a rule explicitly asked for review; and
# firewall_violation means content that does not belong at the field tier was
# found on the record or a skill tried to emit inference as measured data.
REASON_SCHEMA_NOVEL_SHAPE = "schema_novel_shape"
REASON_INCOMPLETE_RECORD = "incomplete_record"
REASON_SENSOR_CONFLICT = "sensor_conflict"
REASON_LOW_CONFIDENCE = "low_confidence_unclassified"
REASON_MANUAL_REVIEW = "manual_review_requested"
REASON_FIREWALL_VIOLATION = "firewall_violation"

# The missing-data vocabulary, one term per outcome, matching the values the
# storage contract accepts and the capture stage assigns. The engine uses
# not_measured when it fills a channel the configuration expected but that
# produced no reading: the channel exists on this station, and no value was
# captured for it at this event.
STATUS_MEASURED = "measured"
STATUS_NOT_MEASURED = "not_measured"
STATUS_BELOW_DETECTION_LIMIT = "below_detection_limit"
STATUS_SENSOR_ERROR = "sensor_error"
STATUS_NOT_APPLICABLE = "not_applicable"

# The set of statuses that assert no usable value is present. A row bearing one
# of these must not also carry a value, and a row that measured a value must
# not bear one of these; either combination is an internal contradiction.
_ABSENT_STATUSES = frozenset(
    {STATUS_NOT_MEASURED, STATUS_SENSOR_ERROR, STATUS_NOT_APPLICABLE}
)

# The oceanographic quality flag for a marine channel with no value to judge.
# A filled marine channel has no measurement to evaluate, so it carries the
# missing flag rather than a pass or fail.
QARTOD_MISSING = 9

# Skill placement tiers. The engine runs only the deterministic-flag tier and
# never the interpretive tier, deciding purely by this tag rather than by
# reading a skill's free text.
TIER_DETERMINISTIC_FLAG = "deterministic_flag"
TIER_INTERPRETIVE = "interpretive"

# The only provenance values a field-captured observation may carry. A record
# that arrives tagged as inferred or dream-derived is inference wearing the
# clothes of a measurement, so it is refused at the field tier.
FIELD_CAPTURE_PROVENANCE = frozenset({"model", "sensor"})

# When per-observation quality control runs on the station. Any other value
# means a power-critical deployment chose to defer quality control to the
# desktop at sync time, so the field engine leaves the record untouched.
ANALYSIS_LOCATION_PI = "pi"

# The confidence floor below which the field tier declines to pass a detection
# on its own and defers it to the desktop instead. It is deliberately low, so
# only detections near the noise floor defer while the desktop verification
# stream re-checks everything else anyway. This is a documented starting value
# with no configuration home yet; it is surfaced as a named constant here and
# can be promoted to a per-station setting later with no change to the rest of
# this engine.
DEFAULT_FIELD_PASS_CONFIDENCE = 0.10

# How many pending record identifiers may wait for the quality-control worker.
# The queue is only a hint: because every record is durably stored before its
# identifier is queued, a full queue drops the hint rather than blocking
# capture, and a sweep re-processes any record the queue skipped.
DEFAULT_QUEUE_MAXSIZE = 512


# ===========================================================================
# Value types the engine produces
# ===========================================================================


@dataclass
class SkillFlag:
    """One deterministic-flag skill's output for a record.

    A deterministic-flag skill is a pure function of the measured values
    already in the record, and its output is a plain measured or derived flag,
    never a sentence of interpretation. value is therefore a boolean or a
    number, never free text, which is what keeps a field-tier skill incapable
    of emitting an ecological claim as data.
    """

    skill_id: str
    skill_title: str
    name: str
    value: object  # a bool or a number; any other type is rejected as a firewall breach


@dataclass
class ConsolidatedSnapshot:
    """One coherent view of a whole multimodal observation after quality control.

    The observation core, its per-taxon detections, and its environmental
    readings (including any the engine filled) are assembled into one object,
    alongside the manifest coverage the engine checked and any deterministic
    flags it derived. This is the single snapshot the live feed and the desktop
    read; it is returned to the caller rather than stored, because the record
    itself, spread across its tables, is the persisted form.
    """

    observation_id: str
    event_name: str
    station_id: str
    trigger_source: str
    observation: dict
    child_detections: list[dict]
    environmental_readings: list[dict]

    # Manifest coverage: the channels the configuration expected, the ones the
    # record already had, and the ones the engine filled with a missing status.
    expected_channels: list[str] = field(default_factory=list)
    present_channels: list[str] = field(default_factory=list)
    filled_channels: list[str] = field(default_factory=list)

    # Deterministic-flag skill outputs derived from the measured values above.
    flags: list[SkillFlag] = field(default_factory=list)

    # Filled in once the outcome is decided.
    qc_state: str = QC_PENDING
    qc_reason: Optional[str] = None


@dataclass
class QCResult:
    """The outcome of running the engine on one record.

    outcome is one of: passed (advanced to qc_passed), deferred (advanced to
    qc_deferred with a reason), skipped (nothing to do, for example the record
    was already finalized, had been removed, or this node defers quality
    control to the desktop). snapshot is the consolidated view when one was
    built.
    """

    observation_id: str
    outcome: str  # "passed" | "deferred" | "skipped"
    qc_state: str
    qc_reason: Optional[str] = None
    snapshot: Optional[ConsolidatedSnapshot] = None
    detail: Optional[str] = None


# A deterministic-flag evaluator turns the consolidated snapshot into one flag,
# or into nothing when the skill does not fire for this record. It is a pure
# function of the measured values in the snapshot. Evaluators are supplied by
# the runtime, keyed by skill identifier; the engine never parses a skill's
# free text into behaviour, because doing so would be interpretation.
FlagEvaluator = Callable[["ConsolidatedSnapshot"], Optional[SkillFlag]]


# ===========================================================================
# Structured skill conditions
#
# A field-tier skill may carry a small, checkable condition alongside the free
# text a person wrote. The engine compiles that condition into a pure function
# of the measured values already in the record, which is what lets a skill
# someone authored actually run without the engine ever interpreting prose.
#
# The vocabulary is deliberately narrow. Everything a condition can name is a
# number the record already holds, and everything it can do is compare that
# number, so a skill cannot reach beyond measurement into inference.
# ===========================================================================

CONDITION_SOURCES = ("observation", "detection", "channel", "time")

# Numeric columns of the observation a condition may compare.
CONDITION_OBSERVATION_FIELDS = (
    "screening_confidence",
    "duration",
    "frame_count",
    "audio_true_duration_seconds",
    "salience_provisional",
)
CONDITION_DETECTION_FIELDS = ("confidence",)
CONDITION_TIME_FIELDS = ("hour_utc",)
CONDITION_AGGREGATES = ("max", "min")
CONDITION_OPS = ("lt", "lte", "gt", "gte", "between", "outside")


def _as_number(value) -> Optional[float]:
    """The value as a number, or nothing when it is absent or not numeric."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _measured_for_condition(snapshot: "ConsolidatedSnapshot", source: str,
                            field_name: str, aggregate: str) -> Optional[float]:
    """The one measured number a condition compares, or nothing when absent.

    Nothing is invented here. When the record does not carry the value a
    condition names, the skill simply does not fire.
    """
    if source == "observation":
        return _as_number((snapshot.observation or {}).get(field_name))

    if source == "detection":
        values = [
            _as_number(det.get(field_name))
            for det in (snapshot.child_detections or [])
        ]
        values = [v for v in values if v is not None]
        if not values:
            return None
        return min(values) if aggregate == "min" else max(values)

    if source == "channel":
        for reading in (snapshot.environmental_readings or []):
            if reading.get("channel") == field_name:
                return _as_number(reading.get("value"))
        return None

    if source == "time":
        # Timestamps are stored as UTC ISO8601, so the hour sits at a fixed offset.
        stamp = (snapshot.observation or {}).get("first_seen") or ""
        try:
            return float(int(str(stamp)[11:13]))
        except (TypeError, ValueError):
            return None

    return None


def _condition_holds(measured: float, op: str, value) -> bool:
    """Whether one measured number satisfies a comparison."""
    if op == "lt":
        return measured < value
    if op == "lte":
        return measured <= value
    if op == "gt":
        return measured > value
    if op == "gte":
        return measured >= value

    low, high = value[0], value[1]
    if low <= high:
        inside = low <= measured <= high
    else:
        # A range that wraps past midnight, for example 20:00 through 04:00.
        inside = measured >= low or measured <= high
    return inside if op == "between" else not inside


def _flag_name(title: str) -> str:
    """A short, stable name for the flag a skill produces."""
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in (title or ""))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "skill_flag"


def parse_condition(raw) -> Optional[dict]:
    """Read a stored condition into a validated specification, or nothing.

    Anything malformed returns nothing rather than raising, so one bad condition
    can never stop the engine from finishing the rest of a record.
    """
    if not raw:
        return None
    try:
        spec = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return None
    if not isinstance(spec, dict):
        return None

    source = spec.get("source")
    field_name = spec.get("field")
    op = spec.get("op")
    value = spec.get("value")
    aggregate = spec.get("aggregate") or "max"

    if source not in CONDITION_SOURCES or op not in CONDITION_OPS:
        return None
    if aggregate not in CONDITION_AGGREGATES:
        return None
    if not isinstance(field_name, str) or not field_name.strip():
        return None
    if source == "observation" and field_name not in CONDITION_OBSERVATION_FIELDS:
        return None
    if source == "detection" and field_name not in CONDITION_DETECTION_FIELDS:
        return None
    if source == "time" and field_name not in CONDITION_TIME_FIELDS:
        return None

    if op in ("between", "outside"):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        bounds = [_as_number(v) for v in value]
        if any(b is None for b in bounds):
            return None
        value = bounds
    else:
        value = _as_number(value)
        if value is None:
            return None

    return {"source": source, "field": field_name.strip(), "op": op,
            "value": value, "aggregate": aggregate}


def compile_condition(skill: dict) -> Optional[FlagEvaluator]:
    """Turn a skill's stored condition into a pure evaluator, or nothing.

    This is not an interpretation of what the skill says. It reads a structured
    comparison a person chose from a fixed vocabulary and returns a function that
    compares one measured number, so the output stays a plain flag and the field
    tier stays incapable of asserting anything it did not measure.
    """
    spec = parse_condition(skill.get("condition"))
    if spec is None:
        return None

    skill_id = str(skill.get("id") or "")
    title = str(skill.get("title") or "")
    name = _flag_name(title)

    def _evaluate(snapshot: "ConsolidatedSnapshot") -> Optional[SkillFlag]:
        measured = _measured_for_condition(snapshot, spec["source"], spec["field"], spec["aggregate"])
        if measured is None:
            return None
        try:
            fired = _condition_holds(measured, spec["op"], spec["value"])
        except (TypeError, ValueError):
            return None
        if not fired:
            return None
        return SkillFlag(skill_id=skill_id, skill_title=title, name=name, value=True)

    return _evaluate


# ===========================================================================
# The engine
# ===========================================================================


class QCEngine:
    """Deterministic field-tier quality control and consolidation for one record.

    Construct it with the validated configuration, the storage layer, and any
    deterministic-flag evaluators the deployment has defined. Call process with
    a record identifier to validate, complete, consolidate, and finalize that
    record. The engine holds no per-record state, so one instance safely serves
    a worker draining a queue.
    """

    def __init__(
        self,
        *,
        settings,
        db: Database,
        flag_evaluators: Optional[dict] = None,
        field_pass_confidence: Optional[float] = None,
    ) -> None:
        self._settings = settings
        self._db = db
        # Evaluators are keyed by skill identifier. A deterministic-flag skill
        # with no evaluator is one the field tier has no pure function for yet,
        # so it is skipped rather than guessed at.
        self._flag_evaluators = dict(flag_evaluators or {})
        # The field-pass confidence floor comes from configuration, with an
        # explicit constructor argument taking precedence when one is passed. The
        # configured default reproduces the value it held before it became
        # configurable, so an unset configuration changes nothing.
        self._field_pass_confidence = float(
            field_pass_confidence if field_pass_confidence is not None
            else settings.thresholds_config()["field_qc"]["pass_confidence"]
        )

        # Visible counters, so a worker's progress and any pressure are
        # observable without reaching into the engine.
        self.processed = 0
        self.passed = 0
        self.deferred = 0
        self.skipped = 0

    # -- public entry point ----------------------------------------------

    def process(self, observation_id: str) -> QCResult:
        """Run quality control on one record, by identifier.

        Reads the record from the database, validates and completes it, runs
        the deterministic-flag skills, and advances its lifecycle state to
        passed or deferred. Returns a result carrying the consolidated snapshot
        and the outcome. Doing nothing is a valid outcome: a record that is
        already finalized, that has been removed, or that this node defers to
        the desktop is left exactly as it is.
        """
        self.processed += 1

        # A deployment can choose to defer per-observation quality control to
        # the desktop; on such a node the field engine advances nothing.
        if self._analysis_location() != ANALYSIS_LOCATION_PI:
            self.skipped += 1
            return QCResult(
                observation_id=observation_id,
                outcome="skipped",
                qc_state=QC_PENDING,
                detail="per-observation quality control is deferred to the desktop on this node",
            )

        obs = self._db.get_observation(observation_id)
        if obs is None:
            # The record is gone (for example a buffer clean removed it after a
            # sync). There is nothing to process and nothing to repair.
            self.skipped += 1
            return QCResult(
                observation_id=observation_id,
                outcome="skipped",
                qc_state=QC_PENDING,
                detail="record not found",
            )

        current_state = obs.get("qc_state")
        if current_state != QC_PENDING:
            # Already finalized by an earlier run. Re-processing must not rewrite
            # a decision, which is what makes a re-queued identifier harmless.
            self.skipped += 1
            return QCResult(
                observation_id=observation_id,
                outcome="skipped",
                qc_state=current_state,
                detail="record already finalized",
            )

        children = self._db.list_child_detections(observation_id)
        readings = self._db.list_environmental_readings(observation_id)

        # Resolve the station manifest. Without it the engine cannot predict
        # what a complete record should look like, so it defers rather than
        # guessing.
        try:
            station = self._settings.station(obs["station_id"])
        except ConfigError:
            logger.error(
                "observation %s references station %s, which is not in the "
                "configuration; deferring to the desktop",
                observation_id,
                obs.get("station_id"),
            )
            return self._finalize(
                observation_id,
                QC_DEFERRED,
                REASON_SCHEMA_NOVEL_SHAPE,
                snapshot=None,
            )

        expected_channels = [
            c for c in station.get("channels", []) if c.get("enabled", False)
        ]

        # The measured-versus-inferred firewall, at the record level. A field
        # observation and everything captured with it must carry only
        # field-capture provenance. Content tagged as inference or dream output
        # is quarantined untouched and routed to the desktop; the engine does
        # not complete or re-score a record whose provenance it cannot trust.
        provenance_breach = self._record_provenance_breach(obs, children, readings)
        if provenance_breach is not None:
            logger.error(
                "observation %s failed the field-tier provenance firewall: %s; "
                "deferring untouched to the desktop",
                observation_id,
                provenance_breach,
            )
            snapshot = self._build_snapshot(
                obs, children, readings, expected_channels, [], []
            )
            return self._finalize(
                observation_id,
                QC_DEFERRED,
                REASON_FIREWALL_VIOLATION,
                snapshot=snapshot,
                detail=provenance_breach,
            )

        # Correct completeness gaps: fill any expected channel that produced no
        # reading with an explicit missing status. This never invents a value.
        present_channels = [r["channel"] for r in readings]
        filled_channels = self._fill_missing_channels(
            observation_id, expected_channels, present_channels
        )
        if filled_channels:
            # Re-read so the snapshot reflects the filled rows.
            readings = self._db.list_environmental_readings(observation_id)

        # Confirm the provisional salience, or write it from the detection's own
        # confidence if capture left it empty. Always the provisional slot only.
        self._confirm_provisional_salience(obs, children)
        obs = self._db.get_observation(observation_id) or obs

        snapshot = self._build_snapshot(
            obs, children, readings, expected_channels, present_channels, filled_channels
        )

        # Run the deterministic-flag skills over the consolidated snapshot. This
        # also enforces the skill-tier half of the firewall: an interpretive
        # skill is never executed, and a skill whose output is not a plain flag
        # has that output rejected.
        flags, skill_breach = self.evaluate_flag_skills(
            snapshot, self._db.list_skills(tier=TIER_DETERMINISTIC_FLAG)
        )
        snapshot.flags = flags
        # Persist each fired flag as a durable derived reading, so a skill's
        # effect on this event outlives the in-memory snapshot and is visible on
        # the desktop. This is the only field-tier write about a skill, and it
        # goes to the flag table alone, never to the measured record.
        self._persist_flags(observation_id, flags)

        if skill_breach:
            return self._finalize(
                observation_id,
                QC_DEFERRED,
                REASON_FIREWALL_VIOLATION,
                snapshot=snapshot,
                detail="a field-tier skill attempted to emit content that is not a measured flag",
            )

        # Validate the record's shape and internal consistency.
        reason = self._structural_reason(obs, children)
        if reason is not None:
            return self._finalize(
                observation_id, QC_DEFERRED, reason, snapshot=snapshot
            )

        # A detection too weak for the field tier to pass on its own is deferred
        # for the desktop to adjudicate.
        if self._below_confidence_floor(obs, children):
            return self._finalize(
                observation_id,
                QC_DEFERRED,
                REASON_LOW_CONFIDENCE,
                snapshot=snapshot,
            )

        return self._finalize(observation_id, QC_PASSED, None, snapshot=snapshot)

    # -- firewall --------------------------------------------------------

    @staticmethod
    def _record_provenance_breach(
        obs: dict, children: list[dict], readings: list[dict]
    ) -> Optional[str]:
        """Report the first field-tier provenance breach found, or nothing.

        A field observation may only carry model or sensor provenance; each
        detection is a model output; each environmental reading is a sensor
        measurement. Anything else is inferred content on a record that claims
        to be a measurement, which the field tier refuses.
        """
        if obs.get("data_source") not in FIELD_CAPTURE_PROVENANCE:
            return f"observation data_source {obs.get('data_source')!r} is not a field-capture provenance"
        for child in children:
            if child.get("data_source") != "model":
                return (
                    f"child detection {child.get('id')!r} carries data_source "
                    f"{child.get('data_source')!r}, not a model output"
                )
        for reading in readings:
            if reading.get("data_source") != "sensor":
                return (
                    f"environmental reading {reading.get('id')!r} carries data_source "
                    f"{reading.get('data_source')!r}, not a sensor measurement"
                )
        return None

    # -- completeness ----------------------------------------------------

    def _fill_missing_channels(
        self,
        observation_id: str,
        expected_channels: list[dict],
        present_channels: list[str],
    ) -> list[str]:
        """Insert a missing-status row for each expected channel that has none.

        A channel the configuration enabled but that produced no reading this
        event is recorded with a not-measured status and no value, so its
        absence is a stated fact rather than a silent gap. A marine channel with
        no value to judge carries the oceanographic missing flag. No value is
        ever invented.
        """
        present = set(present_channels)
        filled: list[str] = []
        for channel in expected_channels:
            channel_id = channel.get("id")
            if channel_id in present:
                continue
            reading = EnvironmentalReading(
                id=new_id(),
                observation_id=observation_id,
                channel=channel_id,
                status=STATUS_NOT_MEASURED,
                created_at=utc_now_iso(),
                data_source="sensor",
                value=None,
                unit=channel.get("unit"),
                qartod_flag=QARTOD_MISSING if channel.get("marine", False) else None,
            )
            self._db.insert_environmental_reading(reading)
            filled.append(channel_id)
        return filled

    # -- salience --------------------------------------------------------

    def _confirm_provisional_salience(self, obs: dict, children: list[dict]) -> None:
        """Confirm the provisional salience, or write it if capture left it empty.

        Capture normally writes the provisional salience from the detection's
        own confidence. The engine confirms it is present and in range and
        leaves it untouched. If it is missing, the engine writes it from the
        screening confidence, or from the strongest child confidence when there
        is no screening confidence (an audio event), always to the provisional
        slot alone and never to the authoritative slot, which belongs to the
        desktop.
        """
        current = obs.get("salience_provisional")
        if current is not None:
            # Present already; the storage contract guarantees it is in range,
            # so capture's value stands and the engine does not rewrite it.
            return

        source = obs.get("screening_confidence")
        if source is None:
            child_confidences = [
                c["confidence"] for c in children if c.get("confidence") is not None
            ]
            source = max(child_confidences) if child_confidences else None
        if source is None:
            # Nothing to derive a provisional value from; leaving it empty is
            # truthful, and the desktop computes the authoritative value later.
            return

        value = min(max(float(source), 0.0), 1.0)
        self._db.set_observation_provisional_salience(obs["id"], value, None)

    # -- skills ----------------------------------------------------------

    def evaluate_flag_skills(
        self, snapshot: ConsolidatedSnapshot, skills: list[dict]
    ) -> tuple[list[SkillFlag], bool]:
        """Run the deterministic-flag skills over one snapshot.

        Returns the flags derived and whether a firewall breach was found. A
        skill is executed only when its tier is the deterministic-flag tier and
        the deployment supplied a pure-function evaluator for it; a skill of any
        other tier is refused and counted as a breach, and a skill with no
        evaluator is skipped without guessing. An evaluator that returns
        anything other than a plain measured or derived flag has its output
        rejected as a breach, so a mis-tagged interpretive skill can never emit
        inference as measured data.
        """
        flags: list[SkillFlag] = []
        breach = False
        for skill in skills:
            skill_id = skill.get("id")
            if skill.get("tier") != TIER_DETERMINISTIC_FLAG:
                # The engine decides tier by the tag, so a skill that reaches the
                # field run path without the field tier is a breach, not a skill
                # to interpret.
                logger.error(
                    "skill %s has tier %r and must not run at the field tier; refusing it",
                    skill_id,
                    skill.get("tier"),
                )
                breach = True
                continue
            evaluator = self._flag_evaluators.get(skill_id)
            if evaluator is None:
                # A skill the deployment did not hand a purpose-built function to
                # can still run when it carries a structured condition, which is
                # compiled into one here. A skill with neither is skipped rather
                # than guessed at.
                evaluator = compile_condition(skill)
            if evaluator is None:
                logger.info(
                    "deterministic-flag skill %s has neither a field evaluator nor a "
                    "checkable condition; skipping",
                    skill_id,
                )
                continue
            try:
                flag = evaluator(snapshot)
            except Exception:  # noqa: BLE001 - a skill fault is isolated and logged, never fatal
                logger.exception(
                    "deterministic-flag skill %s raised during evaluation; skipping",
                    skill_id,
                )
                continue
            if flag is None:
                continue
            if not self._is_measured_flag(flag):
                logger.error(
                    "deterministic-flag skill %s returned a non-flag value %r; "
                    "rejecting it as a firewall breach",
                    skill_id,
                    getattr(flag, "value", flag),
                )
                breach = True
                continue
            flags.append(flag)
        return flags, breach

    @staticmethod
    def _is_measured_flag(flag: object) -> bool:
        """True only for a genuine measured or derived flag.

        A field-tier flag is a boolean or a number. A boolean is an integer in
        Python, so the numeric check already accepts it; a string or any other
        object is not a measured flag and is rejected.
        """
        if not isinstance(flag, SkillFlag):
            return False
        # bool is a subclass of int, so both booleans and numbers pass here,
        # while a string or other object does not.
        return isinstance(flag.value, (bool, int, float))

    def _persist_flags(self, observation_id: str, flags: list[SkillFlag]) -> None:
        """Write each fired flag as a durable derived reading, idempotently.

        The unique key on (observation_id, skill_id) makes a repeated write
        harmless, so re-running the field engine over a record never duplicates
        a flag. Nothing here touches the measured record.
        """
        for flag in flags:
            self._db.insert_skill_flag(SkillFlagRow(
                id=new_id(),
                observation_id=observation_id,
                skill_id=flag.skill_id,
                skill_title=flag.skill_title,
                flag_name=flag.name,
                created_at=utc_now_iso(),
            ))

    def rescan_flag_skills(self, observation_id: str) -> int:
        """Re-evaluate the deterministic-flag skills over one finalized record.

        The live path runs skills during quality control, but a record captured
        before a skill existed is already finalized and quality control will not
        touch it again. This lets the desktop apply the current field skills to
        such a record: it consolidates a read-only snapshot from the stored
        rows, runs the same evaluation and firewall as the live path, records
        any flag that now fires, and clears any earlier flag whose skill no
        longer fires (for example after the skill's condition was edited). It
        never changes the record or its quality-control decision. Returns the
        number of skills that fired on this record.
        """
        obs = self._db.get_observation(observation_id)
        if obs is None:
            return 0
        children = self._db.list_child_detections(observation_id)
        readings = self._db.list_environmental_readings(observation_id)
        snapshot = self._build_snapshot(obs, children, readings, [], [], [])

        skills = self._db.list_skills(tier=TIER_DETERMINISTIC_FLAG)
        flags, _breach = self.evaluate_flag_skills(snapshot, skills)
        fired_skill_ids = {flag.skill_id for flag in flags}

        # A skill that no longer fires must not leave a stale flag behind.
        for skill in skills:
            if skill.get("id") not in fired_skill_ids:
                self._db.clear_skill_flag(observation_id, skill.get("id"))
        self._persist_flags(observation_id, flags)
        return len(flags)

    # -- validation ------------------------------------------------------

    def _structural_reason(self, obs: dict, children: list[dict]) -> Optional[str]:
        """Return a defer reason if the record's shape is invalid, else nothing.

        The storage layer has already enforced the column-level contract, so
        this checks the consistency the database cannot: that the event window
        runs forward, that the duration is not negative, that a visual event
        carries a visual detection and an audio event an audio detection, and
        that no channel's value contradicts its own status.
        """
        trigger = obs.get("trigger_source")
        if trigger not in ("vision", "audio"):
            # The sensor-threshold trigger is reserved and has no field capture
            # path yet, so a record of that shape is one the engine cannot
            # validate and hands to the desktop.
            return REASON_SCHEMA_NOVEL_SHAPE

        first_seen = obs.get("first_seen")
        last_seen = obs.get("last_seen")
        # Timestamps are stored in one fixed UTC format, zero-padded, so a plain
        # comparison orders them correctly without parsing.
        if first_seen is None or last_seen is None or last_seen < first_seen:
            return REASON_INCOMPLETE_RECORD

        duration = obs.get("duration")
        if duration is None or duration < 0:
            return REASON_INCOMPLETE_RECORD

        modalities = {c.get("modality") for c in children}
        if trigger == "vision" and "vision" not in modalities:
            return REASON_INCOMPLETE_RECORD
        if trigger == "audio" and "audio" not in modalities:
            return REASON_INCOMPLETE_RECORD

        if self._readings_conflict(obs):
            return REASON_SENSOR_CONFLICT

        return None

    def _readings_conflict(self, obs: dict) -> bool:
        """True if any stored channel value contradicts its own status.

        A row that reports a measurement must carry a value; a row that reports
        an absence must not carry one. Either contradiction is a conflict the
        field tier cannot silently accept, so the record goes to the desktop.
        """
        for reading in self._db.list_environmental_readings(obs["id"]):
            status = reading.get("status")
            value = reading.get("value")
            if status == STATUS_MEASURED and value is None:
                return True
            if status in _ABSENT_STATUSES and value is not None:
                return True
        return False

    def _below_confidence_floor(self, obs: dict, children: list[dict]) -> bool:
        """True if the detection is too weak for the field tier to pass.

        The effective confidence is the screening confidence for a visual event,
        or the strongest child confidence when there is no screening confidence.
        A record with no confidence at all is not deferred on this ground; only
        a present confidence below the floor defers.
        """
        confidence = obs.get("screening_confidence")
        if confidence is None:
            child_confidences = [
                c["confidence"] for c in children if c.get("confidence") is not None
            ]
            confidence = max(child_confidences) if child_confidences else None
        if confidence is None:
            return False
        return confidence < self._field_pass_confidence

    # -- assembly and finalize -------------------------------------------

    @staticmethod
    def _build_snapshot(
        obs: dict,
        children: list[dict],
        readings: list[dict],
        expected_channels: list[dict],
        present_channels: list[str],
        filled_channels: list[str],
    ) -> ConsolidatedSnapshot:
        return ConsolidatedSnapshot(
            observation_id=obs["id"],
            event_name=obs["event_name"],
            station_id=obs["station_id"],
            trigger_source=obs["trigger_source"],
            observation=obs,
            child_detections=children,
            environmental_readings=readings,
            expected_channels=[c.get("id") for c in expected_channels],
            present_channels=list(present_channels),
            filled_channels=list(filled_channels),
        )

    def _finalize(
        self,
        observation_id: str,
        qc_state: str,
        qc_reason: Optional[str],
        *,
        snapshot: Optional[ConsolidatedSnapshot],
        detail: Optional[str] = None,
    ) -> QCResult:
        """Write the lifecycle outcome and return the result.

        Only the station-owned lifecycle columns are written here; the verified
        state and everything interpretive are the desktop's to write. The write
        is a single update, so a record moves from pending to its outcome in one
        step.
        """
        self._db.set_observation_qc(observation_id, qc_state, qc_reason)
        if snapshot is not None:
            snapshot.qc_state = qc_state
            snapshot.qc_reason = qc_reason
        if qc_state == QC_PASSED:
            self.passed += 1
            outcome = "passed"
        else:
            self.deferred += 1
            outcome = "deferred"
        return QCResult(
            observation_id=observation_id,
            outcome=outcome,
            qc_state=qc_state,
            qc_reason=qc_reason,
            snapshot=snapshot,
            detail=detail,
        )

    # -- helpers ---------------------------------------------------------

    def _analysis_location(self) -> str:
        try:
            return self._settings.analysis_location()
        except Exception:  # noqa: BLE001 - a missing setting defaults to running here
            return ANALYSIS_LOCATION_PI

    def sweep_pending(
        self, *, station_id: Optional[str] = None, limit: Optional[int] = None
    ) -> int:
        """Process every still-pending record, catching up after a restart.

        A record whose identifier a full queue dropped, or that was captured
        while the worker was down, stays pending in the database. This finds
        those records and runs quality control on each, which the per-record
        idempotency guard makes safe to call at any time. Returns the number of
        records advanced out of the pending state.
        """
        advanced = 0
        for row in self._db.list_observations(station_id=station_id, limit=limit):
            if row.get("qc_state") != QC_PENDING:
                continue
            result = self.process(row["id"])
            if result.outcome in ("passed", "deferred"):
                advanced += 1
        return advanced


# ===========================================================================
# The bounded queue and worker
# ===========================================================================


class QCWorker:
    """A bounded queue of record identifiers drained by a background worker.

    Capture submits an identifier the moment it has stored a record. The worker
    thread drains the queue and runs the engine on each identifier. Submission
    never blocks capture: if the queue is momentarily full, the identifier is
    dropped from the queue and counted, because the record is safely in the
    database and a sweep re-processes any pending record a full queue skipped.
    One bad record never stops the worker; its failure is logged and the worker
    moves on.
    """

    _SENTINEL = object()

    def __init__(self, engine: QCEngine, *, maxsize: int = DEFAULT_QUEUE_MAXSIZE) -> None:
        self._engine = engine
        self._queue: "queue.Queue[object]" = queue.Queue(maxsize=maxsize)
        self._thread: Optional[threading.Thread] = None

        # Visible counters mirroring the pipeline's back-pressure reporting.
        self.submitted = 0
        self.queue_saturation_events = 0
        self.processed = 0
        self.failed = 0

    def submit(self, observation_id: str) -> bool:
        """Queue one record for quality control without ever blocking the caller.

        Returns True when the identifier was queued, or False when the queue was
        full and the identifier was dropped from the queue. A dropped identifier
        is not lost work: the record stays pending in the database and a sweep
        will process it.
        """
        try:
            self._queue.put_nowait(observation_id)
            self.submitted += 1
            return True
        except queue.Full:
            self.queue_saturation_events += 1
            logger.warning(
                "quality-control queue full; record %s stays pending for a sweep "
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
            target=self._run, name="audtheia-qc-worker", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the worker after it finishes the queued work.

        A sentinel is placed after the identifiers already queued, so the worker
        processes everything that was submitted and then exits cleanly.
        """
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
                logger.exception("quality control failed for record %s", item)
