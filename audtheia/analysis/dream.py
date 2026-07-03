"""Audtheia longitudinal dream pass.

Path: audtheia/analysis/dream.py

This module is the desktop's longitudinal analysis stage. It runs after events
have been captured in the field, synced up, and verified, and its job is to
find the regularities across the whole record that no single event reveals: how
a signal drifts across the seasons, which taxa turn up together, how one
environmental channel moves with another, and which occurrences form a new
grouping the rest of the record has not seen.

The stage is built as a two-phase cycle, and the order of the phases is not
optional.

  Consolidate first. Each cycle takes the next batch of newly-arrived events and
  folds their environmental readings into a permanent, compact summary of the
  site: for each recurring period (for example each calendar month) and each
  signal, a running robust center and spread. This summary is the baseline the
  rest of the system reads. It is extracted before anything is pruned, because a
  longitudinal regularity is a shape spread across many events, and pruning the
  events before summarizing them would destroy the shape.

  Downscale second, and only the working memory. After consolidation, the
  derived working set the generative phase reasons over is capped to a bounded,
  salience-ranked selection. This touches only that working set. The event
  archive and the permanent baseline are never pruned, so the scientific record
  stays whole.

  Integrate last, and only over confirmed events. The generative phase proposes
  candidate patterns, and every candidate it proposes rests only on events the
  desktop verification stage has cleared. Consolidation tolerates an occasional
  unconfirmed field call because an aggregate absorbs it; a proposed claim must
  not, so the generative phase is gated to verified events.

Everything the generative phase emits is a candidate hypothesis, tagged as a
dream output, carried with an effect size and the span of data behind it, and
never presented as an established finding. This is deliberate: speculative
recombination belongs here, downstream, over the complete and labeled record,
and nowhere upstream.

The work is measured in cycles, not wall-clock time. A cycle is one
consolidate-downscale-integrate traversal and is the unit that commits, so a
pass can be asked to stop after any cycle and resume later from exactly where it
left off, with no event consolidated twice and none skipped. Because the
generative phase reasons over a bounded working set rather than the whole
archive, a pass over a years-old station costs no more per cycle than a pass
over a young one.

The language model that narrates a candidate in plain words, and any clustering
backend used to find novel groupings, are optional collaborators supplied by the
caller. With neither present the stage still runs end to end: candidates are
described by a built-in template, and the novel-grouping detector simply
contributes nothing. Only the Python standard library is imported here.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Iterable, Optional

from audtheia.storage.database import (
    Database,
    DreamPass,
    Pattern,
    SiteBaseline,
    new_id,
    utc_now_iso,
)

__all__ = [
    "DreamEngine",
    "PatternFragment",
    "PassResult",
    "PHASE_NREM_A",
    "PHASE_NREM_B",
    "PHASE_REM",
    "PHASE_COMPLETE",
    "PATTERN_TEMPORAL_SHIFT",
    "PATTERN_CO_OCCURRENCE",
    "PATTERN_ENVELOPE_CORRELATION",
    "PATTERN_NOVEL_CLUSTER",
]

logger = logging.getLogger("audtheia.analysis.dream")


# ---------------------------------------------------------------------------
# Controlled strings, mirroring the database vocabularies so a write can never
# drift from what the schema accepts.
# ---------------------------------------------------------------------------

DATA_SOURCE_DREAM = "dream"

PHASE_NREM_A = "nrem_a"   # consolidate
PHASE_NREM_B = "nrem_b"   # downscale
PHASE_REM = "rem"         # integrate
PHASE_COMPLETE = "complete"

STATUS_RUNNING = "running"
STATUS_PAUSED = "paused"
STATUS_COMPLETE = "complete"
STATUS_ERROR = "error"

DREAM_PHASE_NREM = "nrem"
DREAM_PHASE_REM = "rem"

PATTERN_TEMPORAL_SHIFT = "temporal_shift"
PATTERN_CO_OCCURRENCE = "co_occurrence"
PATTERN_ENVELOPE_CORRELATION = "envelope_correlation"
PATTERN_NOVEL_CLUSTER = "novel_cluster"

EFFECT_R = "r"
EFFECT_LOG_ODDS = "log_odds"

STAT_MANN_KENDALL = "mann_kendall"
STAT_SPEARMAN = "spearman_rho"
STAT_LOG_ODDS = "log_odds_ratio"

# A qualifying environmental reading is a real measurement that also passed its
# quality flag. The QARTOD scale is marine-only, so a non-marine channel carries
# no flag and qualifies on its measured status alone, while a marine channel must
# additionally carry the pass flag. Anything a station could not actually measure
# is excluded, so a sensor fault can never manufacture an anomaly.
_STATUS_MEASURED = "measured"
_QARTOD_PASS = 1

# The consistency constant that makes a median absolute deviation an unbiased
# estimator of the standard deviation under a normal distribution.
_MAD_TO_SD = 1.4826


# ---------------------------------------------------------------------------
# Defaults. Starting values only; the runtime configuration overrides them.
# Effect-size and support floors keep the generative phase from proposing a
# candidate that rests on too little data to mean anything. They ship as named
# constants for now and read cleanly from a settings home when one is added.
# ---------------------------------------------------------------------------

DEFAULT_MIN_PERIODS_FOR_TREND = 4      # a trend needs several periods to be a trend
DEFAULT_MIN_EVENTS_FOR_CORRELATION = 8  # a correlation needs enough paired events
DEFAULT_MIN_EVENTS_FOR_CO_OCCURRENCE = 8
DEFAULT_MIN_ABS_EFFECT = 0.2           # ignore a candidate whose effect is negligible
DEFAULT_MAX_P_VALUE = 0.05             # ignore a candidate that is not even nominally notable

DEFAULT_EPOCH_BATCH_SIZE = 500
DEFAULT_SUBSTRATE_EXEMPLAR_CAP = 5000
DEFAULT_SUBSTRATE_CANDIDATE_CAP = 1000


# ===========================================================================
# Small statistical helpers. Standard-library only, so the module imports and
# runs with no third-party numerical stack present.
# ===========================================================================


def _median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _mad_scaled(values: list[float], center: float) -> Optional[float]:
    if not values:
        return None
    deviations = [abs(v - center) for v in values]
    mad = _median(deviations)
    if mad is None:
        return None
    return _MAD_TO_SD * mad


def _mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _sd(values: list[float], mean: float) -> Optional[float]:
    if len(values) < 2:
        return None
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def _regularized_gamma_p(a: float, x: float) -> float:
    """The regularized lower incomplete gamma function P(a, x).

    This is the chi-square cumulative distribution in disguise, computed with
    the standard series expansion below the transition point and the standard
    continued fraction above it. It needs no third-party library.
    """
    if x <= 0 or a <= 0:
        return 0.0
    if x < a + 1.0:
        # Series expansion.
        term = 1.0 / a
        total = term
        n = a
        for _ in range(1000):
            n += 1.0
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    # Continued fraction for the complement, then subtract from one.
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    q = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
    return 1.0 - q


def _chi2_cdf(x: float, dof: int) -> float:
    """The chi-square cumulative distribution for a non-negative statistic.

    Rises from zero at no deviation toward one as the combined deviation grows,
    and its scale does not depend on how many channels contributed, because the
    degrees of freedom account for the channel count.
    """
    if dof <= 0 or x <= 0:
        return 0.0
    return _regularized_gamma_p(dof / 2.0, x / 2.0)


def _normal_sf(z: float) -> float:
    """The upper-tail probability of the standard normal at z."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _average_ranks(values: list[float]) -> list[float]:
    """Average ranks, so ties share the mean of the ranks they span."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> Optional[tuple[float, float]]:
    """Spearman rank correlation and a two-sided p-value, or nothing.

    Returns nothing when there are too few pairs or no variation to correlate.
    """
    n = len(xs)
    if n < 3 or len(ys) != n:
        return None
    rx = _average_ranks(xs)
    ry = _average_ranks(ys)
    mrx = sum(rx) / n
    mry = sum(ry) / n
    num = sum((rx[i] - mrx) * (ry[i] - mry) for i in range(n))
    denx = math.sqrt(sum((rx[i] - mrx) ** 2 for i in range(n)))
    deny = math.sqrt(sum((ry[i] - mry) ** 2 for i in range(n)))
    if denx == 0 or deny == 0:
        return None
    rho = num / (denx * deny)
    rho = max(-1.0, min(1.0, rho))
    if abs(rho) >= 1.0:
        return rho, 0.0
    # t approximation for the significance of a rank correlation.
    t = rho * math.sqrt((n - 2) / (1.0 - rho * rho))
    p = 2.0 * _normal_sf(abs(t))
    return rho, min(1.0, p)


def _mann_kendall(series: list[float]) -> Optional[tuple[float, float]]:
    """The Mann-Kendall trend test: Kendall's tau and a two-sided p-value.

    A nonparametric monotonic-trend test that assumes no particular
    distribution, which suits environmental series. Returns nothing when the
    series is too short.
    """
    n = len(series)
    if n < 4:
        return None
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += (series[j] > series[i]) - (series[j] < series[i])
    denom = n * (n - 1) / 2.0
    if denom == 0:
        return None
    tau = s / denom
    var = n * (n - 1) * (2 * n + 5) / 18.0
    if var <= 0:
        return tau, 1.0
    if s > 0:
        z = (s - 1) / math.sqrt(var)
    elif s < 0:
        z = (s + 1) / math.sqrt(var)
    else:
        z = 0.0
    p = 2.0 * _normal_sf(abs(z))
    return tau, min(1.0, p)


def _period_key(first_seen: str, granularity: str) -> Optional[str]:
    """The recurring-period bin an event falls into, from its UTC timestamp.

    A month bin groups every January together; an ISO-week bin groups week 23 of
    every year; a day-of-year bin groups the 172nd day of every year. Grouping
    by a recurring period is what lets a summer reading be judged against other
    summers rather than against a year-round average the season would dominate.
    """
    dt = _parse_iso(first_seen)
    if dt is None:
        return None
    if granularity == "month":
        return f"{dt.month:02d}"
    if granularity == "iso_week":
        return f"{dt.isocalendar()[1]:02d}"
    if granularity == "doy":
        return f"{dt.timetuple().tm_yday:03d}"
    return None


def _parse_iso(value: str) -> Optional[datetime]:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        # Fall back to a date-only or second-precision form.
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(value.replace("Z", ""), fmt)
            except ValueError:
                continue
    return None


def _chronological_key(first_seen: str, granularity: str) -> Optional[str]:
    """A real-time-ordered bin label, unlike the recurring-period key.

    The recurring-period key pools every year's same season into one cell, which
    is right for a seasonal baseline but wrong for a trend: a trend must be
    measured across real time, not across the twelve months of a single
    idealized year. This key prefixes the year, so the labels order
    chronologically and a trend detected across them is a change over time rather
    than the ordinary seasonal march.
    """
    dt = _parse_iso(first_seen)
    if dt is None:
        return None
    period = _period_key(first_seen, granularity)
    if period is None:
        return None
    if granularity == "iso_week":
        # Anchor an ISO week to its ISO year, which can differ from the calendar
        # year at a year boundary.
        return f"{dt.isocalendar()[0]:04d}-{period}"
    return f"{dt.year:04d}-{period}"


# The unit-separator control character, unused in any timestamp or identifier,
# packs the composite arrival cursor into the single checkpoint column.
_CURSOR_SEP = "\x1f"


def _encode_cursor(synced_at: Optional[str], obs_id: Optional[str]) -> Optional[str]:
    if synced_at is None:
        return None
    if obs_id is None:
        return synced_at
    return f"{synced_at}{_CURSOR_SEP}{obs_id}"


def _decode_cursor(token: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if token is None:
        return None, None
    if _CURSOR_SEP in token:
        synced_at, obs_id = token.split(_CURSOR_SEP, 1)
        return synced_at, obs_id
    return token, None


# ===========================================================================
# Working types
# ===========================================================================


@dataclass
class PatternFragment:
    """A candidate pattern before it becomes a stored row.

    A detector returns these; the engine turns each into a persisted candidate
    with its supporting event links. dedup_key lets the engine drop a duplicate
    a later cycle would otherwise propose again within the same pass.
    """
    pattern_type: str
    dream_phase: str
    description: str
    data_span_start: str
    data_span_end: str
    observation_ids: list[str]
    dedup_key: str
    effect_size: Optional[float] = None
    effect_size_type: Optional[str] = None
    statistic: Optional[str] = None
    p_value: Optional[float] = None
    q_value: Optional[float] = None
    confidence: Optional[float] = None
    model_version: Optional[str] = None


@dataclass
class PassResult:
    """The outcome of running (or resuming) one pass."""
    dream_pass_id: str
    status: str
    cycles_completed: int
    observations_consolidated: int
    salience_scored: int
    patterns_emitted: int
    checkpoint_watermark: Optional[str]


@dataclass
class _Exemplar:
    """One verified event in the bounded working set the generative phase reads."""
    observation_id: str
    station_id: str
    first_seen: str
    salience: float
    species: list[str] = field(default_factory=list)          # gbif usage keys present
    readings: dict = field(default_factory=dict)               # channel -> qualifying value


# ===========================================================================
# Engine
# ===========================================================================


class DreamEngine:
    """The longitudinal pass over one desktop database.

    Construct it with the validated configuration and the storage layer, and
    optionally a narrator that renders a candidate in plain words and a clusterer
    that proposes novel groupings. Neither optional collaborator is required: the
    pass runs to completion without them.

    The engine holds no per-record state between passes; each call to run_pass or
    resume_pass carries its own bounded working memory, so one instance can serve
    repeated passes safely.
    """

    def __init__(
        self,
        *,
        settings,
        db: Database,
        narrator=None,
        clusterer=None,
        min_periods_for_trend: Optional[int] = None,
        min_events_for_correlation: Optional[int] = None,
        min_events_for_co_occurrence: Optional[int] = None,
        min_abs_effect: Optional[float] = None,
        max_p_value: Optional[float] = None,
    ) -> None:
        self._settings = settings
        self._db = db
        self._narrator = narrator
        self._clusterer = clusterer

        # The generative-phase floors come from configuration, with an explicit
        # constructor argument taking precedence when one is passed. The
        # configured defaults reproduce the values these floors held before they
        # became configurable, so an unset configuration changes nothing.
        dream_thresholds = settings.thresholds_config()["dream"]
        self._min_periods_for_trend = int(
            min_periods_for_trend if min_periods_for_trend is not None
            else dream_thresholds["min_periods_for_trend"]
        )
        self._min_events_for_correlation = int(
            min_events_for_correlation if min_events_for_correlation is not None
            else dream_thresholds["min_events_for_correlation"]
        )
        self._min_events_for_co_occurrence = int(
            min_events_for_co_occurrence if min_events_for_co_occurrence is not None
            else dream_thresholds["min_events_for_co_occurrence"]
        )
        self._min_abs_effect = float(
            min_abs_effect if min_abs_effect is not None
            else dream_thresholds["min_abs_effect"]
        )
        self._max_p_value = float(
            max_p_value if max_p_value is not None
            else dream_thresholds["max_p_value"]
        )

        # Read the tunable analysis configuration once per engine.
        baseline_cfg = settings.baseline_config()
        self._granularity = baseline_cfg["period_granularity"]

        salience_cfg = settings.salience_config()
        self._weights = salience_cfg["weights"]
        self._min_effective_n = int(salience_cfg["anomaly"]["min_effective_n"])

        budget = settings.dream_budget()
        self._epoch_batch_size = int(budget.get("epoch_batch_size", DEFAULT_EPOCH_BATCH_SIZE)) or DEFAULT_EPOCH_BATCH_SIZE
        self._max_cycles_per_pass = int(budget.get("max_cycles_per_pass", 0))
        self._exemplar_cap = int(budget.get("substrate_exemplar_cap", DEFAULT_SUBSTRATE_EXEMPLAR_CAP)) or DEFAULT_SUBSTRATE_EXEMPLAR_CAP
        self._candidate_cap = int(budget.get("substrate_candidate_pattern_cap", DEFAULT_SUBSTRATE_CANDIDATE_CAP)) or DEFAULT_SUBSTRATE_CANDIDATE_CAP

        # Visible counters so a caller can observe progress without reaching in.
        self.observations_consolidated = 0
        self.salience_scored = 0
        self.patterns_emitted = 0

    # -- public entry points ---------------------------------------------

    def run_pass(
        self,
        *,
        station_scope: Optional[str] = None,
        should_pause: Optional[Callable[[], bool]] = None,
    ) -> PassResult:
        """Start a new pass over everything that has arrived since the last one.

        The new pass begins from the furthest point any earlier pass reached, so
        an event is never consolidated twice across passes. It runs cycle by
        cycle until the backlog is drained, the per-pass cycle limit is reached,
        or a pause is requested after a cycle. It returns as soon as it commits
        the cycle on which it stops.
        """
        start_watermark = self._latest_prior_watermark(station_scope)
        dp = DreamPass(
            id=new_id(),
            phase_reached=PHASE_NREM_A,
            status=STATUS_RUNNING,
            started_at=utc_now_iso(),
            created_at=utc_now_iso(),
            station_scope=station_scope,
            checkpoint_watermark=start_watermark,
        )
        self._db.create_dream_pass(dp)
        return self._drive(dp.id, station_scope, start_watermark, should_pause)

    def resume_pass(
        self,
        dream_pass_id: str,
        *,
        should_pause: Optional[Callable[[], bool]] = None,
    ) -> PassResult:
        """Continue a paused pass from exactly where it stopped.

        The pass resumes from its committed watermark, so no event it already
        consolidated is seen again and none in the remaining backlog is skipped.
        """
        row = self._db.get_dream_pass(dream_pass_id)
        if row is None:
            raise ValueError(f"no dream pass with id {dream_pass_id!r}")
        station_scope = row.get("station_scope")
        watermark = row.get("checkpoint_watermark")
        self._db.update_dream_pass(dream_pass_id, status=STATUS_RUNNING)
        return self._drive(dream_pass_id, station_scope, watermark, should_pause)

    # -- the cycle loop ---------------------------------------------------

    def _drive(
        self,
        dream_pass_id: str,
        station_scope: Optional[str],
        watermark: Optional[str],
        should_pause: Optional[Callable[[], bool]],
    ) -> PassResult:
        cycles = self._current_cycles(dream_pass_id)
        work_consumed = self._current_work(dream_pass_id)
        emitted_keys: set[str] = set()
        cursor_token = watermark

        while True:
            if self._max_cycles_per_pass and cycles >= self._max_cycles_per_pass:
                # The per-pass cycle limit is a work budget, not the end of the
                # backlog, so the pass pauses rather than declaring completion.
                self._commit_cycle(dream_pass_id, PHASE_REM, STATUS_PAUSED, cycles, work_consumed, cursor_token)
                return self._result(dream_pass_id, STATUS_PAUSED, cycles, cursor_token)

            after_synced, after_id = _decode_cursor(cursor_token)
            batch = self._db.list_synced_since(
                after_synced, after_id, limit=self._epoch_batch_size, station_id=station_scope
            )
            if not batch:
                # Nothing new to consolidate: the pass has cleared its backlog.
                self._commit_cycle(dream_pass_id, PHASE_COMPLETE, STATUS_COMPLETE, cycles, work_consumed, cursor_token, ended=True)
                return self._result(dream_pass_id, STATUS_COMPLETE, cycles, cursor_token)

            # NREM-A: consolidate the batch into the permanent gist, scoring each
            # event against the gist with its own readings held out, so an event
            # is never compared against a baseline that already contains it.
            self._db.update_dream_pass(dream_pass_id, phase_reached=PHASE_NREM_A)
            self._consolidate_and_score(batch, station_scope)

            # NREM-B: cap the derived working set. This prunes only working
            # memory; the archive and the gist are untouched.
            self._db.update_dream_pass(dream_pass_id, phase_reached=PHASE_NREM_B)
            exemplars = self._build_working_set(station_scope)

            # REM: propose candidates over the bounded, verified-only working
            # set, and persist those that clear the support and effect floors.
            self._db.update_dream_pass(dream_pass_id, phase_reached=PHASE_REM)
            self._integrate(dream_pass_id, exemplars, emitted_keys)

            cycles += 1
            work_consumed += len(batch)
            cursor_token = _encode_cursor(batch[-1]["synced_at"], batch[-1]["id"])
            self._commit_cycle(dream_pass_id, PHASE_REM, STATUS_RUNNING, cycles, work_consumed, cursor_token)

            if should_pause is not None and should_pause():
                self._db.update_dream_pass(dream_pass_id, status=STATUS_PAUSED)
                return self._result(dream_pass_id, STATUS_PAUSED, cycles, cursor_token)

    # -- NREM-A: consolidation and salience ------------------------------

    def _consolidate_and_score(self, batch: list[dict], station_scope: Optional[str]) -> None:
        """Fold a batch into the gist, then score its verified events.

        Consolidation runs first, so the gist reflects the full record before any
        event is scored against it. The gist is refreshed for exactly the cells
        the batch touches, recomputed from each cell's full membership so the
        robust center and spread are exact rather than approximated, and only the
        touched cells are recomputed, so the cost of a cycle follows the batch,
        not the length of the record.

        Scoring then measures each verified event against the gist with the
        event's own readings held out of the cells it is compared to, so an event
        is never judged against a baseline that already contains it. Holding a
        single event out of a mature cell barely moves the robust center, but the
        exclusion is exact rather than assumed.
        """
        touched: set[tuple[str, str, str]] = set()
        for obs in batch:
            self.observations_consolidated += 1
            period = _period_key(obs["first_seen"], self._granularity)
            if period is None:
                continue
            for reading in self._db.list_environmental_readings(obs["id"]):
                if not self._reading_qualifies(reading):
                    continue
                touched.add((obs["station_id"], reading["channel"], period))

        for station_id, channel, period in touched:
            self._recompute_cell(station_id, channel, period)

        # Score verified events against the now-current gist, each held out of its
        # own cells. A per-cell membership cache is built once and reused, so the
        # hold-out costs no extra queries per event.
        verified_ids = set(self._db.list_verified_observation_ids(station_id=station_scope))
        membership_cache: dict[tuple[str, str], dict[str, list[tuple[str, float]]]] = {}
        station_taxon_cache: dict[str, dict] = {}
        for obs in batch:
            if obs["id"] in verified_ids:
                self._score_salience(obs, membership_cache, station_taxon_cache)
                self.salience_scored += 1

    def _cell_membership(
        self,
        station_id: str,
        channel: str,
        cache: dict[tuple[str, str], dict[str, list[tuple[str, float]]]],
    ) -> dict[str, list[tuple[str, float]]]:
        """The qualifying (observation_id, value) readings of a channel, by period.

        Built once per station-and-channel and cached for the scoring sweep, so
        the leave-one-out hold-out reads from memory rather than re-querying.
        """
        key = (station_id, channel)
        if key in cache:
            return cache[key]
        by_period: dict[str, list[tuple[str, float]]] = {}
        for r in self._db.list_environmental_readings_for_baseline(station_id, channel):
            if not self._reading_qualifies(r):
                continue
            period = _period_key(r["first_seen"], self._granularity)
            if period is None:
                continue
            by_period.setdefault(period, []).append((r["observation_id"], float(r["value"])))
        cache[key] = by_period
        return by_period

    def _recompute_cell(self, station_id: str, channel: str, period: str) -> None:
        """Rebuild one baseline cell exactly from its full qualifying membership."""
        rows = self._db.list_environmental_readings_for_baseline(station_id, channel)
        values: list[float] = []
        span_start: Optional[str] = None
        span_end: Optional[str] = None
        for r in rows:
            if _period_key(r["first_seen"], self._granularity) != period:
                continue
            if not self._reading_qualifies(r):
                continue
            values.append(float(r["value"]))
            if span_start is None or r["first_seen"] < span_start:
                span_start = r["first_seen"]
            if span_end is None or r["first_seen"] > span_end:
                span_end = r["first_seen"]

        if not values or span_start is None or span_end is None:
            return

        center = _median(values)
        cell = SiteBaseline(
            id=new_id(),
            station_id=station_id,
            period_granularity=self._granularity,
            period_key=period,
            group_type="all",
            group_key="ALL",
            signal=channel,
            data_span_start=span_start,
            data_span_end=span_end,
            updated_at=utc_now_iso(),
            created_at=utc_now_iso(),
            n=len(values),
            median=center,
            mad_scaled=_mad_scaled(values, center) if center is not None else None,
            mean=_mean(values),
            sd=_sd(values, _mean(values)) if len(values) > 1 else None,
            min_value=min(values),
            max_value=max(values),
        )
        self._db.upsert_site_baseline(cell)

    def _score_salience(
        self,
        obs: dict,
        membership_cache: dict[tuple[str, str], dict[str, list[tuple[str, float]]]],
        station_taxon_cache: dict[str, dict],
    ) -> None:
        """Compute and store one event's authoritative salience.

        The confidence ingredient prefers the desktop verification confidence and
        falls back to the field screening confidence. The anomaly ingredient is a
        calibrated combination of the event's robust deviations across its
        qualifying channels against the matching baseline cells. The rarity
        ingredient, included only when it carries weight, is the taxon's local
        novelty: how rarely it appears in this station's own record. The
        ingredients are blended by the configured weights, renormalized over
        whichever ingredients are present, so a missing ingredient drops out
        cleanly and an event with neither is left unscored rather than assigned a
        false zero.
        """
        confidence = self._confidence_ingredient(obs)
        anomaly, signed_dev = self._anomaly_ingredient(obs, membership_cache)

        # Rarity is computed only when it is weighted, so a deployment that does
        # not weight rarity pays nothing for it and the score is unchanged.
        rarity_weight = float(self._weights.get("rarity", 0.0))
        rarity = self._rarity_ingredient(obs, station_taxon_cache) if rarity_weight > 0 else None

        present: list[tuple[float, float]] = []  # (weight, value)
        if confidence is not None:
            present.append((float(self._weights.get("confidence", 0.0)), confidence))
        if anomaly is not None:
            present.append((float(self._weights.get("anomaly", 0.0)), anomaly))
        if rarity is not None:
            present.append((rarity_weight, rarity))

        weight_sum = sum(w for w, _ in present)
        if not present or weight_sum <= 0:
            salience = None
        else:
            salience = sum(w * v for w, v in present) / weight_sum
            salience = max(0.0, min(1.0, salience))

        self._db.set_authoritative_salience(
            obs["id"],
            salience_authoritative=salience,
            baseline_deviation=signed_dev,
            anomaly_magnitude_authoritative=anomaly,
        )

    def _confidence_ingredient(self, obs: dict) -> Optional[float]:
        v = self._db.get_observation_verification(obs["id"])
        if v is not None and v.get("rfdetr_confidence") is not None:
            return float(v["rfdetr_confidence"])
        if obs.get("screening_confidence") is not None:
            return float(obs["screening_confidence"])
        return None

    def _anomaly_ingredient(
        self,
        obs: dict,
        membership_cache: dict[tuple[str, str], dict[str, list[tuple[str, float]]]],
    ) -> tuple[Optional[float], Optional[float]]:
        """The event's calibrated anomaly and the signed deviation behind it.

        Each qualifying channel is standardized against its baseline cell's
        robust center and spread, computed from the cell's membership with this
        event's own reading held out. The squared standardized deviations are
        combined through the chi-square distribution, whose degrees of freedom
        equal the number of contributing channels, so an event at a station with
        more sensors is not judged more anomalous for that reason alone. The
        signed deviation of the single largest-magnitude channel is returned for
        the record, so a reviewer can see which channel drove the score.
        """
        period = _period_key(obs["first_seen"], self._granularity)
        if period is None:
            return None, None

        squared_sum = 0.0
        dof = 0
        strongest_signed = None
        strongest_mag = -1.0

        for reading in self._db.list_environmental_readings(obs["id"]):
            if not self._reading_qualifies(reading):
                continue
            by_period = self._cell_membership(obs["station_id"], reading["channel"], membership_cache)
            members = by_period.get(period, [])
            # Hold this event's own reading out of the baseline it is scored
            # against, so the comparison is never against a cell containing it.
            others = [value for (oid, value) in members if oid != obs["id"]]
            if len(others) < self._min_effective_n:
                continue
            center = _median(others)
            scale = _mad_scaled(others, center) if center is not None else None
            if center is None or scale is None or scale <= 0:
                continue
            z = (float(reading["value"]) - float(center)) / float(scale)
            squared_sum += z * z
            dof += 1
            if abs(z) > strongest_mag:
                strongest_mag = abs(z)
                strongest_signed = z

        if dof == 0:
            return None, None
        anomaly = _chi2_cdf(squared_sum, dof)
        return max(0.0, min(1.0, anomaly)), strongest_signed

    def _rarity_ingredient(self, obs: dict, station_taxon_cache: dict[str, dict]) -> Optional[float]:
        """The event's local rarity: how rarely its taxa appear at this station.

        A taxon seen in few of a station's events is rare there, and a rare taxon
        should draw more attention, which is the taxonomic-channel analogue of the
        anomaly term's novelty over the environmental channels. Each taxon's local
        frequency is the share of the station's events it appears in; its rarity
        is one minus that share. When an event carries more than one taxon, the
        rarest present taxon drives the score, mirroring how the anomaly term
        follows the strongest-deviating channel. The per-station counts are read
        once and cached for the scoring sweep. An event with no taxon yields no
        rarity rather than a fabricated value.
        """
        station_id = obs["station_id"]
        stats = station_taxon_cache.get(station_id)
        if stats is None:
            stats = self._db.taxon_event_counts(station_id)
            station_taxon_cache[station_id] = stats
        total = stats["total_events"]
        counts = stats["taxon_events"]
        if total <= 0:
            return None
        taxa = [
            (c.get("gbif_usage_key") or c.get("common_name"))
            for c in self._db.list_child_detections(obs["id"])
        ]
        taxa = [t for t in taxa if t]
        if not taxa:
            return None
        rarest: Optional[float] = None
        for taxon in taxa:
            share = counts.get(taxon, 0) / total
            rarity = max(0.0, min(1.0, 1.0 - share))
            if rarest is None or rarity > rarest:
                rarest = rarity
        return rarest

    @staticmethod
    def _reading_qualifies(reading: dict) -> bool:
        if reading.get("value") is None:
            return False
        if reading.get("status") != _STATUS_MEASURED:
            return False
        flag = reading.get("qartod_flag")
        # A marine channel must carry the pass flag; a non-marine channel carries
        # no flag and qualifies on its measured status alone.
        return flag is None or flag == _QARTOD_PASS

    # -- NREM-B: the bounded, verified-only working set ------------------

    def _build_working_set(self, station_scope: Optional[str]) -> list[_Exemplar]:
        """Assemble the bounded, salience-ranked set the generative phase reads.

        Only verified events are eligible, which is the gate that keeps a
        proposed claim from resting on an unconfirmed field call. The set is
        ranked by authoritative salience, falling back to the field provisional
        salience when the authoritative value is not yet set, so an early-record
        event is ranked by a real value rather than buried by a missing one, and
        capped to the configured size so the generative phase's cost stays
        bounded no matter how large the archive grows.
        """
        verified_ids = self._db.list_verified_observation_ids(station_id=station_scope)
        scored: list[tuple[float, dict]] = []
        for oid in verified_ids:
            obs = self._db.get_observation(oid)
            if obs is None:
                continue
            v = self._db.get_observation_verification(oid)
            salience = None
            if v is not None and v.get("salience_authoritative") is not None:
                salience = v["salience_authoritative"]
            elif obs.get("salience_provisional") is not None:
                salience = obs["salience_provisional"]
            rank = salience if salience is not None else -1.0
            scored.append((rank, obs))

        # Keep the strongest by salience, up to the cap. This is the downscaling
        # of derived working memory: it selects what the generative phase reasons
        # over and never removes anything from the archive.
        scored.sort(key=lambda pair: pair[0], reverse=True)
        kept = scored[: self._exemplar_cap]

        exemplars: list[_Exemplar] = []
        for rank, obs in kept:
            species = [
                c["gbif_usage_key"]
                for c in self._db.list_child_detections(obs["id"])
                if c.get("gbif_usage_key")
            ]
            readings = {
                r["channel"]: float(r["value"])
                for r in self._db.list_environmental_readings(obs["id"])
                if self._reading_qualifies(r)
            }
            exemplars.append(
                _Exemplar(
                    observation_id=obs["id"],
                    station_id=obs["station_id"],
                    first_seen=obs["first_seen"],
                    salience=rank if rank >= 0 else 0.0,
                    species=species,
                    readings=readings,
                )
            )
        # Present the working set in time order, which every detector expects.
        exemplars.sort(key=lambda e: e.first_seen)
        return exemplars

    # -- REM: candidate integration --------------------------------------

    def _integrate(
        self,
        dream_pass_id: str,
        exemplars: list[_Exemplar],
        emitted_keys: set[str],
    ) -> int:
        """Run every detector over the working set and persist fresh candidates."""
        fragments: list[PatternFragment] = []
        fragments.extend(self._detect_temporal_shift(exemplars))
        fragments.extend(self._detect_envelope_correlation(exemplars))
        fragments.extend(self._detect_co_occurrence(exemplars))
        fragments.extend(self._detect_novel_cluster(exemplars))

        # Benjamini-Hochberg false-discovery adjustment across every test this
        # cycle produced. It runs over the full set of p-values, before the
        # effect and p floors select survivors, so the adjustment reflects the
        # true number of comparisons the cycle made rather than only the ones
        # that passed. A candidate carrying no p-value is left unadjusted.
        self._assign_bh_q_values(fragments)

        emitted = 0
        for frag in fragments:
            if len(emitted_keys) >= self._candidate_cap:
                break
            if frag.dedup_key in emitted_keys:
                continue
            if not self._passes_floors(frag):
                continue
            self._persist(dream_pass_id, frag)
            emitted_keys.add(frag.dedup_key)
            emitted += 1
            self.patterns_emitted += 1
        return emitted

    def _passes_floors(self, frag: PatternFragment) -> bool:
        if frag.effect_size is not None and abs(frag.effect_size) < self._min_abs_effect:
            return False
        if frag.p_value is not None and frag.p_value > self._max_p_value:
            return False
        return True

    @staticmethod
    def _assign_bh_q_values(fragments: list[PatternFragment]) -> None:
        """Attach Benjamini-Hochberg q-values across a cycle's tests.

        A q-value is the false-discovery-adjusted p-value: the smallest false
        discovery rate at which a candidate would still be called. Only
        candidates that carry a p-value take part, and the number of tests is
        exactly that set, so the adjustment is honest about how many comparisons
        were made. The step-up pass from the largest p-value down keeps the
        adjusted values monotone.
        """
        indexed = [(i, f.p_value) for i, f in enumerate(fragments) if f.p_value is not None]
        m = len(indexed)
        if m == 0:
            return
        order = sorted(indexed, key=lambda pair: pair[1])
        running_min = 1.0
        for rank in range(m, 0, -1):
            frag_index, p = order[rank - 1]
            candidate = p * m / rank
            if candidate < running_min:
                running_min = candidate
            fragments[frag_index].q_value = running_min if running_min < 1.0 else 1.0

    def _persist(self, dream_pass_id: str, frag: PatternFragment) -> None:
        pattern = Pattern(
            id=new_id(),
            dream_pass_id=dream_pass_id,
            dream_phase=frag.dream_phase,
            data_span_start=frag.data_span_start,
            data_span_end=frag.data_span_end,
            n=len(frag.observation_ids),
            description=frag.description,
            created_at=utc_now_iso(),
            pattern_type=frag.pattern_type,
            confidence=frag.confidence,
            effect_size=frag.effect_size,
            effect_size_type=frag.effect_size_type,
            statistic=frag.statistic,
            p_value=frag.p_value,
            q_value=frag.q_value,
            autocorr_adjusted=0,
            model_version=frag.model_version,
        )
        self._db.insert_pattern(pattern, observation_ids=frag.observation_ids)

    # -- detectors --------------------------------------------------------

    def _detect_temporal_shift(self, exemplars: list[_Exemplar]) -> list[PatternFragment]:
        """A monotonic trend in a signal's central value across real time.

        For each signal, the verified events are grouped into chronological bins
        that carry their year, the median value per bin forms a time-ordered
        series, and a Mann-Kendall test asks whether that series trends. The
        chronological binning is deliberate: grouping by season-of-year instead
        would pool every year's same months together and measure the ordinary
        seasonal march rather than a change over time. Because the bins carry the
        year, a trend found over a span shorter than a year still reflects
        movement within that window, which the stored data span makes explicit;
        distinguishing a genuine multi-year trend from the seasonal cycle needs a
        span of more than one year. The claim rests only on the verified events
        that fed the series.
        """
        by_signal: dict[str, list[_Exemplar]] = {}
        for ex in exemplars:
            for channel in ex.readings:
                by_signal.setdefault(channel, []).append(ex)

        out: list[PatternFragment] = []
        for channel, members in by_signal.items():
            per_bin: dict[str, list[float]] = {}
            bin_events: dict[str, list[_Exemplar]] = {}
            for ex in members:
                bin_key = _chronological_key(ex.first_seen, self._granularity)
                if bin_key is None:
                    continue
                per_bin.setdefault(bin_key, []).append(ex.readings[channel])
                bin_events.setdefault(bin_key, []).append(ex)
            if len(per_bin) < self._min_periods_for_trend:
                continue
            ordered_bins = sorted(per_bin.keys())
            series = [_median(per_bin[b]) for b in ordered_bins]
            if any(v is None for v in series):
                continue
            result = _mann_kendall([float(v) for v in series])
            if result is None:
                continue
            tau, p = result
            supporting = [ex for b in ordered_bins for ex in bin_events[b]]
            span = sorted(ex.first_seen for ex in supporting)
            direction = "rising" if tau > 0 else "falling"
            desc = self._describe(
                PATTERN_TEMPORAL_SHIFT,
                f"{channel} shows a {direction} trend across {len(ordered_bins)} time bins "
                f"(Kendall tau {tau:.2f})",
            )
            out.append(
                PatternFragment(
                    pattern_type=PATTERN_TEMPORAL_SHIFT,
                    dream_phase=DREAM_PHASE_REM,
                    description=desc,
                    data_span_start=span[0],
                    data_span_end=span[-1],
                    observation_ids=[ex.observation_id for ex in supporting],
                    dedup_key=f"{PATTERN_TEMPORAL_SHIFT}:{channel}",
                    effect_size=tau,
                    effect_size_type=EFFECT_R,
                    statistic=STAT_MANN_KENDALL,
                    p_value=p,
                )
            )
        return out

    def _detect_envelope_correlation(self, exemplars: list[_Exemplar]) -> list[PatternFragment]:
        """A monotonic coupling between two environmental signals across events.

        For each pair of channels, the events that measured both contribute a
        paired point, and a Spearman correlation asks whether the two move
        together. This surfaces a coupling in the site's environmental envelope
        that no single reading reveals.
        """
        channels: set[str] = set()
        for ex in exemplars:
            channels.update(ex.readings.keys())
        ordered = sorted(channels)

        out: list[PatternFragment] = []
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a, b = ordered[i], ordered[j]
                xs: list[float] = []
                ys: list[float] = []
                supporting: list[_Exemplar] = []
                for ex in exemplars:
                    if a in ex.readings and b in ex.readings:
                        xs.append(ex.readings[a])
                        ys.append(ex.readings[b])
                        supporting.append(ex)
                if len(supporting) < self._min_events_for_correlation:
                    continue
                result = _spearman(xs, ys)
                if result is None:
                    continue
                rho, p = result
                span = sorted(ex.first_seen for ex in supporting)
                desc = self._describe(
                    PATTERN_ENVELOPE_CORRELATION,
                    f"{a} and {b} move together across {len(supporting)} events "
                    f"(Spearman rho {rho:.2f})",
                )
                out.append(
                    PatternFragment(
                        pattern_type=PATTERN_ENVELOPE_CORRELATION,
                        dream_phase=DREAM_PHASE_REM,
                        description=desc,
                        data_span_start=span[0],
                        data_span_end=span[-1],
                        observation_ids=[ex.observation_id for ex in supporting],
                        dedup_key=f"{PATTERN_ENVELOPE_CORRELATION}:{a}:{b}",
                        effect_size=rho,
                        effect_size_type=EFFECT_R,
                        statistic=STAT_SPEARMAN,
                        p_value=p,
                    )
                )
        return out

    def _detect_co_occurrence(self, exemplars: list[_Exemplar]) -> list[PatternFragment]:
        """Taxa that appear together across events more than chance would predict.

        Each event is a presence set of taxa. For every pair of taxa, a two-by-two
        table over the events records how often each is present with and without
        the other, and a log odds ratio measures the strength of association. The
        claim rests on the events in which either taxon appeared.
        """
        events: list[tuple[_Exemplar, set[str]]] = [
            (ex, set(ex.species)) for ex in exemplars if ex.species
        ]
        if len(events) < self._min_events_for_co_occurrence:
            return []

        taxa: set[str] = set()
        for _, present in events:
            taxa.update(present)
        ordered = sorted(taxa)

        out: list[PatternFragment] = []
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a, b = ordered[i], ordered[j]
                n11 = n10 = n01 = n00 = 0
                supporting: list[_Exemplar] = []
                for ex, present in events:
                    has_a = a in present
                    has_b = b in present
                    if has_a and has_b:
                        n11 += 1
                    elif has_a and not has_b:
                        n10 += 1
                    elif has_b and not has_a:
                        n01 += 1
                    else:
                        n00 += 1
                    if has_a or has_b:
                        supporting.append(ex)
                if n11 == 0 or len(supporting) < self._min_events_for_co_occurrence:
                    continue
                # Log odds ratio with a continuity correction so an empty cell
                # does not make the ratio undefined.
                a_ = n11 + 0.5
                b_ = n10 + 0.5
                c_ = n01 + 0.5
                d_ = n00 + 0.5
                log_or = math.log((a_ * d_) / (b_ * c_))
                se = math.sqrt(1.0 / a_ + 1.0 / b_ + 1.0 / c_ + 1.0 / d_)
                p = 2.0 * _normal_sf(abs(log_or) / se) if se > 0 else 1.0
                span = sorted(ex.first_seen for ex in supporting)
                desc = self._describe(
                    PATTERN_CO_OCCURRENCE,
                    f"taxa {a} and {b} co-occur across {len(supporting)} events "
                    f"(log odds {log_or:.2f})",
                )
                out.append(
                    PatternFragment(
                        pattern_type=PATTERN_CO_OCCURRENCE,
                        dream_phase=DREAM_PHASE_REM,
                        description=desc,
                        data_span_start=span[0],
                        data_span_end=span[-1],
                        observation_ids=[ex.observation_id for ex in supporting],
                        dedup_key=f"{PATTERN_CO_OCCURRENCE}:{a}:{b}",
                        effect_size=log_or,
                        effect_size_type=EFFECT_LOG_ODDS,
                        statistic=STAT_LOG_ODDS,
                        p_value=min(1.0, p),
                    )
                )
        return out

    def _detect_novel_cluster(self, exemplars: list[_Exemplar]) -> list[PatternFragment]:
        """Novel groupings among the events, when a clustering backend is present.

        This detector contributes nothing unless the caller supplied a clusterer,
        which keeps the module fully functional with no clustering library
        installed. When a clusterer is present it is handed the working set and
        may return groupings, each of which becomes a candidate.
        """
        if self._clusterer is None:
            return []
        try:
            groupings = self._clusterer.cluster(exemplars)
        except Exception:  # noqa: BLE001 - an optional backend fault is isolated, never fatal
            logger.exception("clustering backend failed; contributing no groupings this cycle")
            return []

        out: list[PatternFragment] = []
        for group in groupings or []:
            member_ids = list(group.get("observation_ids", []))
            if not member_ids:
                continue
            members = [ex for ex in exemplars if ex.observation_id in set(member_ids)]
            if not members:
                continue
            span = sorted(ex.first_seen for ex in members)
            label = group.get("label", "unlabelled grouping")
            desc = self._describe(
                PATTERN_NOVEL_CLUSTER,
                f"a novel grouping of {len(members)} events: {label}",
            )
            out.append(
                PatternFragment(
                    pattern_type=PATTERN_NOVEL_CLUSTER,
                    dream_phase=DREAM_PHASE_REM,
                    description=desc,
                    data_span_start=span[0],
                    data_span_end=span[-1],
                    observation_ids=member_ids,
                    dedup_key=f"{PATTERN_NOVEL_CLUSTER}:{group.get('id', label)}",
                    effect_size=group.get("effect_size"),
                    effect_size_type=group.get("effect_size_type"),
                    statistic=group.get("statistic"),
                    confidence=group.get("confidence"),
                    model_version=group.get("model_version"),
                )
            )
        return out

    # -- narration --------------------------------------------------------

    def _describe(self, pattern_type: str, template: str) -> str:
        """Render a candidate in plain words.

        The built-in template always produces a complete, factual description. A
        narrator, when supplied, may rewrite it more readably, but a narrator
        fault or absence never blocks a candidate: the template stands on its own.
        """
        if self._narrator is None:
            return template
        try:
            narrated = self._narrator.narrate(pattern_type=pattern_type, template=template)
            if narrated:
                return str(narrated)
        except Exception:  # noqa: BLE001 - narration is a convenience, never load-bearing
            logger.exception("narrator failed; using the built-in description")
        return template

    # -- pass bookkeeping -------------------------------------------------

    def _latest_prior_watermark(self, station_scope: Optional[str]) -> Optional[str]:
        """The furthest arrival cursor any earlier pass reached, or nothing.

        A fresh pass resumes the record where the last one ended, so no event is
        consolidated twice across passes.
        """
        best: Optional[str] = None
        for row in self._db.list_dream_passes():
            if station_scope is not None and row.get("station_scope") != station_scope:
                continue
            wm = row.get("checkpoint_watermark")
            if wm is not None and (best is None or wm > best):
                best = wm
        return best

    def _current_cycles(self, dream_pass_id: str) -> int:
        row = self._db.get_dream_pass(dream_pass_id)
        return int(row["cycles_completed"]) if row else 0

    def _current_work(self, dream_pass_id: str) -> int:
        row = self._db.get_dream_pass(dream_pass_id)
        return int(row["work_budget_consumed"]) if row else 0

    def _commit_cycle(
        self,
        dream_pass_id: str,
        phase_reached: str,
        status: str,
        cycles: int,
        work_consumed: int,
        watermark: Optional[str],
        *,
        ended: bool = False,
    ) -> None:
        """Commit a cycle's progress: the watermark, counts, phase, and status.

        This is the only place a pass records forward progress, and it records it
        all at once, so a pass is never left half-advanced: either a cycle's work
        is fully committed or it is not, which is what makes a pause safe and a
        resume exact.
        """
        self._db.update_dream_pass(
            dream_pass_id,
            phase_reached=phase_reached,
            status=status,
            cycles_completed=cycles,
            work_budget_consumed=work_consumed,
            checkpoint_watermark=watermark,
            ended_at=utc_now_iso() if ended else None,
        )

    def _result(
        self,
        dream_pass_id: str,
        status: str,
        cycles: int,
        watermark: Optional[str],
    ) -> PassResult:
        return PassResult(
            dream_pass_id=dream_pass_id,
            status=status,
            cycles_completed=cycles,
            observations_consolidated=self.observations_consolidated,
            salience_scored=self.salience_scored,
            patterns_emitted=self.patterns_emitted,
            checkpoint_watermark=watermark,
        )
