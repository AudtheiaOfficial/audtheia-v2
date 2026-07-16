"""Provisional salience for a freshly-captured observation.

This implements the salience score defined in `docs/salience.md` (decisions #106
and #107) as the station-side *provisional* value written at capture time:

    C_eff = visual detection confidence if a visual detection is present, else 0
    A_eff = acoustic detection confidence if an acoustic detection is matched,
            else 0
    D     = 1 - (1 - C_eff)(1 - A_eff)          multimodal detection evidence
    S     = D * (wN*N + wR*R + wE*E)            importance,  wN+wR+wE = 1

with `N` local novelty at the station, `R` rarity across the whole record, and
`E` an optional environmental-anomaly term that is off by standing decision
(`wE = 0`). Every term is in `[0, 1]`, so `S` is in `[0, 1]`.

Novelty and rarity are the **Shannon information content** (self-information, or
"surprisal") of the observed species, Laplace-smoothed over the model's known
species universe of size `k`. The self-information of an event of probability
`p` is `-log2(p)`: the less probable a species is in the record, the more
information (surprise) its detection carries. A never-seen species carries the
most information (normalized to 1); a species that comes to dominate a large
record carries progressively less (toward 0), but never abruptly zero. This
replaces the earlier `1 - frequency` estimator, which collapsed to exactly 0 for
the only/dominant species after a single sighting (decision #107).

The counts are read from the record *before* the current observation is written,
so an event never contributes to its own baseline. This is the provisional slot
only; the desktop computes the authoritative salience later and this value is
never treated as final.
"""
from __future__ import annotations

import math

# #106 default weights: wN (novelty), wR (rarity), wE (environment, off).
DEFAULT_WEIGHTS = (0.5, 0.5, 0.0)

# Laplace (additive) smoothing pseudocount, so an unseen or barely-seen species
# is treated as rare rather than undefined.
LAPLACE_ALPHA = 1.0


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def detection_evidence(c_eff: float, a_eff: float) -> float:
    """Noisy-OR of the two detection confidences: ``1 - (1 - C)(1 - A)``.

    A modality that did not fire contributes 0, leaving the other unchanged, so
    ``D`` always equals the evidence of whichever modality actually triggered the
    event, and corroboration by both raises it above either alone.
    """
    c = _clamp01(float(c_eff))
    a = _clamp01(float(a_eff))
    return 1.0 - (1.0 - c) * (1.0 - a)


def _normalized_surprisal(count_s: int, total: int, k: int, alpha: float = LAPLACE_ALPHA) -> float:
    """Return the species' Shannon information content, normalized to ``[0, 1]``.

    The species' Laplace-smoothed probability in the record is
    ``p_s = (count_s + alpha) / (total + alpha*k)`` over a universe of ``k``
    known species. Its self-information is ``-log2(p_s)`` bits. Dividing by the
    self-information of the rarest possible (never-seen) species,
    ``-log2(alpha / (total + alpha*k))``, maps it to ``[0, 1]``: an unseen
    species yields 1, and a species that dominates a large record tends toward 0.
    """
    k = max(2, int(k))
    denom = float(total) + alpha * k
    if denom <= 0:
        return 1.0
    p_s = (count_s + alpha) / denom
    p_min = alpha / denom
    max_surprisal = -math.log2(p_min)
    if max_surprisal <= 0:
        return 1.0
    return _clamp01((-math.log2(p_s)) / max_surprisal)


def novelty(n_s: int, t_station: int, k: int) -> float:
    """Local novelty: the species' normalized surprisal at *this station*.

    ``n_s`` is the count of prior observations of this species at the station and
    ``t_station`` the station's total observations. A species new to the station
    is maximally novel; one that dominates the station's record is progressively
    less so.
    """
    return _normalized_surprisal(n_s, t_station, k)


def rarity(count_s: int, count_total: int, k: int) -> float:
    """Global rarity: the species' normalized surprisal across the *whole* record.

    ``count_s`` is the count of observations of this species across all stations
    and ``count_total`` the total number of observations.
    """
    return _normalized_surprisal(count_s, count_total, k)


def compute_salience(
    c_eff: float,
    a_eff: float,
    counts: dict,
    *,
    k: int,
    weights: tuple = DEFAULT_WEIGHTS,
    e: float = 0.0,
) -> float:
    """Return the provisional salience `S` in ``[0, 1]``.

    ``counts`` is the mapping returned by ``Database.salience_counts``:
    ``n_s``, ``t_station``, ``count_s``, ``count_total``. ``k`` is the size of
    the model's known-species universe (its label count). ``e`` is the optional
    environmental-anomaly term, unused while ``wE = 0``.
    """
    wN, wR, wE = weights
    D = detection_evidence(c_eff, a_eff)
    N = novelty(int(counts.get("n_s", 0)), int(counts.get("t_station", 0)), k)
    R = rarity(int(counts.get("count_s", 0)), int(counts.get("count_total", 0)), k)
    S = D * (wN * N + wR * R + wE * _clamp01(float(e)))
    return _clamp01(S)
