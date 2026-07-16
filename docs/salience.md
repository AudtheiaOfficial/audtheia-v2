# Salience

The salience score `S` is Audtheia's single, bounded measure of how much a given
observation matters. It is the quantity that decides which observations are
surfaced first, and — through a firing threshold — which ones are promoted into
the longitudinal NREM→REM dream pass for deeper consolidation. This document is
the authoritative definition of `S`: its meaning, its equation, how each term is
computed from the record, the assumptions it rests on, and the roadmap for the
terms that upgrade as more data and credentials become available.

## Meaning

> `S ∈ [0, 1]` is the importance of a multimodally-confirmed observation of an
> individual — how reliably it was detected visually and, when applicable,
> acoustically — scaled by how ecologically distinctive it is: locally novel at
> its station and rare across the whole record, at the moment of capture.

`S` is dimensionless and bounded on `[0, 1]`, so it is comparable across
species, stations, deployments, and studies. It is a **derived (inferred)**
quantity, not a measurement: it is computed from measured inputs but is itself a
model of importance, and it is tagged accordingly so it never blends with
measured data (the measured-versus-inferred provenance firewall).

## The equation

```
C_eff = visual-detection confidence if a visual detection is present for the
        event, otherwise 0
A_eff = acoustic-detection confidence if an acoustic detection is matched to
        the event, otherwise 0

D = 1 − (1 − C_eff)(1 − A_eff)             (multimodal detection evidence)

S = D · ( wN·N + wR·R + wE·E )             (importance)

with   wN + wR + wE = 1,   default  wN = wR = 0.5,  wE = 0
and    C_eff, A_eff, N, R, E, D, S  all ∈ [0, 1]
```

With the default weights (`wE = 0`) this reduces to `S = D · (0.5·N + 0.5·R)`.
For a silent or unheard taxon (`A_eff = 0`) it reduces to `S = C_eff · (0.5·N +
0.5·R)`; for an audio-only event (`C_eff = 0`, e.g. a call heard with no visual
capture) it reduces symmetrically to `S = A_eff · (0.5·N + 0.5·R)`.

## Novelty and rarity as Shannon information

Novelty (`N`) and rarity (`R`) are built on **Shannon information content**, also
called self-information or *surprisal*. For an event of probability `p`, its
information content is `−log₂(p)`, measured in bits: the less likely something
is, the more information — the more *surprise* — its occurrence carries. A
certainty (`p = 1`) carries 0 bits; a rare event carries many, and independent
events' information adds. Applied here: if a species makes up a large share of
the record it is expected, so detecting it is unsurprising and carries little
information (low `N`, `R`); a species that is new or rare is surprising and
information-rich (high `N`, `R`).

In plain English, **novelty and rarity measure how much a detection tells us we
did not already know** — locally (at the station) and globally (across the whole
record). The logarithm is what makes this graceful: a species' surprisal falls
*smoothly* as it is seen more often, rather than snapping to zero after the first
sighting. So a confident re-detection of a common species stays low-but-nonzero
(routine, not worthless — and its confidence is still shown beside `S`), while
the first or rare sighting stands out.

## Variables

- **C_eff — visual detection confidence (conditional).**
  `C_eff = maxₜ(confidenceₜ)` over the frames of the tracked event (the
  pipeline's `best_confidence`, stored as `screening_confidence`) when a visual
  detection is present, otherwise `0`. Measured model output. Available now.

- **A_eff — acoustic detection confidence (conditional).** The confidence of an
  acoustic detection matched to the same event, or `0` when none is matched.
  Measured model output (BirdNET or the marine PAM model).

  Both detection terms are conditional and symmetric: whichever modality did not
  fire contributes `0`. A silent or unheard taxon has `A_eff = 0`; an
  audio-only event (a call with no visual capture) has `C_eff = 0`. Neither
  case penalizes the observation — the missing channel simply adds no evidence.

- **D — multimodal detection evidence.** `D = 1 − (1 − C_eff)(1 − A_eff)`. The
  probability that at least one modality is correct, under the assumption the two
  modalities are independent. `D ≥ max(C_eff, A_eff)`, so corroboration by both
  channels strengthens the evidence, and because a non-firing channel contributes
  `0`, `D` always equals the evidence of the modality (or modalities) that
  actually triggered the event. The degenerate `C_eff = A_eff = 0` cannot occur:
  an observation exists only because at least one modality fired (the pipeline is
  detection-triggered).

- **N — local novelty.** The species' normalized Shannon surprisal *at this
  station*. With `nₛ` prior observations of the species at the station out of
  `t_station` total, its Laplace-smoothed probability over the `K` known species
  is `pₛ = (nₛ + α) / (t_station + α·K)` (`α = 1`). Novelty is the self-information
  normalized against the rarest possible (never-seen) species:
  `N = −log₂(pₛ) / −log₂(pₘᵢₙ)`, with `pₘᵢₙ = α / (t_station + α·K)`. A species new
  to the station → `N ≈ 1`; one that comes to dominate the station's record → `N`
  toward 0. Derived from the local SQLite record; no external service. Counts are
  taken over all history (not a rolling window).

- **R — species rarity.** The same normalized surprisal across the *entire*
  record: with `countₛ` observations of the species out of `count_total`,
  `pₛ = (countₛ + α) / (count_total + α·K)` and `R = −log₂(pₛ) / −log₂(pₘᵢₙ)`. Rare
  in the record → `R ≈ 1`; a species that dominates a large record → `R` toward 0.
  Derived from SQLite. This smoothed-surprisal estimator replaces the earlier
  `1 − frequency` form, which collapsed to exactly 0 for the only/dominant species
  after a single sighting (decision #107). See the roadmap for the
  effort-normalized (GBIF) upgrade.

- **K — known-species universe.** The number of species the model can recognize
  (its label count; 130 for the reference Porifera model). It sets the smoothing
  prior: before evidence each species is assumed to occur with probability `1/K`,
  so a species seen once out of `K` possible reads as rare. Read from the loaded
  model at capture; treated as `≥ 2`.

- **E — environmental anomaly (optional, off by standing decision).**
  `E = clamp( mean over channels of |x − μ| / (k·σ), 0, 1 )`, the mean
  standardized deviation of the capture-time environmental readings from the
  station's per-channel baseline (`μ`, `σ` from `site_baselines`; `k` a
  tolerance constant). By standing decision `wE = 0`: environmental conditions
  are reported in the written longitudinal (dream-pass) analysis as behavioral
  context — so a scientist can weigh how, say, a heat anomaly may have influenced
  the organism's behavior — rather than folded into `S`, which keeps *unusual
  conditions* from being conflated with *unusual biology*. The `E` term remains
  an optional, weight-gated hook for deployments that later choose to include it,
  with no schema or formula change.

- **wN, wR, wE — weights.** Non-negative, sum to 1. Defaults `0.5, 0.5, 0`. They
  are a deployment-level tuning knob and live in `settings.json`, never in code.

## Design rationale

**Why the detection terms multiply and the ecological terms add.** `C` and
`A_eff` answer *"is the organism really there, and how sure are we?"* `N`, `R`,
and `E` answer *"given it is there, how noteworthy is it?"* These are different
questions, so the detection evidence `D` **gates** the importance
multiplicatively: an unreliable detection (`D ≈ 0`) drives `S` toward 0 no
matter how rare or novel the putative species, which is the scientifically
conservative behavior. The ecological terms **add**, because local novelty,
global rarity, and environmental anomaly are *independent* reasons an
observation matters — any one of them alone should be able to raise `S`, so they
are summed rather than required jointly.

**Why noisy-OR for the two modalities.** Averaging `C` and `A_eff` would let a
weak acoustic call drag down a strong visual detection, which is wrong: two
detectors of the same event provide *more* evidence together, not less. The
noisy-OR `D = 1 − (1 − C)(1 − A_eff)` is the standard combination of independent
evidence and is monotonic in both inputs. It also makes the acoustic channel
strictly conditional: with no matched call, `A_eff = 0` and `D = C` exactly, so
silent taxa are neither penalized nor require a separate "is this species vocal"
flag.

**Why a full product (`C·N·R`) was rejected.** A pure product zeroes `S`
whenever any single factor is ~0 — so a confidently-detected, genuinely rare
species would score ~0 merely because it happens to be common at that one
station. Gating by `D` while summing the ecological drivers preserves that
signal.

## Boundary behavior

- Unreliable detection: `D → 0` ⇒ `S → 0`.
- Common, expected species: as the species comes to dominate the record, `N` and
  `R` fall smoothly toward (but not to) 0, so `S` declines toward a small value.
  A routine re-sighting is low-salience, not zero — and its confidence stays
  visible alongside `S`. Only a species that dominates a *large* record approaches
  `S ≈ 0`.
- Confident, locally novel, globally rare, multimodally confirmed: every factor
  → 1 ⇒ `S → 1`.
- Silent or unheard taxon (e.g. Porifera): `A_eff = 0` ⇒ `D = C_eff`; the
  formula degrades gracefully to the visual-only case.
- Audio-only event (heard, not seen): `C_eff = 0` ⇒ `D = A_eff`; the formula
  degrades symmetrically to the acoustic-only case.
- `C_eff = A_eff = 0` is excluded by construction — an observation exists only
  because at least one modality fired.

## Worked examples

**Silent taxon (a sponge), `K = 130`.** `C_eff = 0.44`, `A_eff = 0` ⇒ `D = 0.44`.
Suppose it is the 4th observation and the species already accounts for all 3
prior records (a single-species record so far). Its smoothed probability is
`pₛ = (3 + 1)/(3 + 1·130) = 4/133 = 0.030`, and the rarest possible is
`pₘᵢₙ = 1/133 = 0.0075`:

```
N = R = −log₂(0.030) / −log₂(0.0075) = 5.06 / 7.06 = 0.72
S     = 0.44 · (0.5·0.72 + 0.5·0.72) = 0.44 · 0.72 = 0.32
```

Even though it is the only species seen, salience stays meaningful (0.32) rather
than collapsing to 0. It sits below the confidence (0.44) because a
repeatedly-seen species is less surprising — and the confidence is shown next to
`S`, so the detection's reliability is never hidden. The very first sighting of a
species (empty baseline) gives `N = R = 1`, so `S = D` exactly.

**Vocal taxon, multimodally confirmed.** A bird seen (`C_eff = 0.70`) and heard
(`A_eff = 0.80`):

```
D = 1 − (1 − 0.70)(1 − 0.80) = 1 − 0.30·0.20 = 0.94
```

Detection evidence 0.94 exceeds either modality alone; the acoustic
corroboration is quantified rather than assumed.

**Audio-only event (heard, not seen).** A call is matched but no visual
detection was captured: `C_eff = 0`, `A_eff = 0.80` ⇒ `D = 1 − (1)(0.20) =
0.80`. The event is scored on its acoustic evidence alone, with no penalty for
the absent visual channel.

## The action-potential framing and the dream pass

Salience is designed as a graded "membrane potential" for an observation. A
firing threshold `θ` closes the analogy: observations with `S ≥ θ` *fire* into
the NREM→REM dream pass for longitudinal consolidation, while subthreshold
observations remain in the record but are not prioritized for deeper analysis.
`θ` is a `settings.json` value so a deployment can tune how selective the dream
pass is, exactly as a firing threshold tunes a neuron's excitability.

## Assumptions and limitations (state these when publishing)

1. **Modality independence.** `D` assumes the visual and acoustic detectors are
   independent evidence of the same event. Correlated errors would make `D`
   optimistic. Cross-modal taxonomic *conflict* is a quality-control concern and
   is resolved there, not in `S`.
2. **Frequency as an abundance proxy.** `N` and `R` are the Shannon surprisal of
   the species' Laplace-smoothed *observation frequency*, which is
   sampling-effort dependent, not true abundance. This is a standard and
   acceptable operational choice in ecology **provided it is disclosed**; the
   effort-normalized (GBIF) upgrade for `R` is on the roadmap. The smoothing prior
   `K` (the model's species count) also shapes early-record values, when few
   observations exist — everything is relatively surprising until the record is
   large enough to establish what is common.
3. **Detection-evidence bias.** Multimodally-confirmed detections receive higher
   `D` than single-modality ones. This is intended (more evidence ⇒ higher
   reliability) but means salience is not modality-neutral; the ecological
   drivers `N`, `R`, `E`, by contrast, are modality-independent.
4. **Environment off by default.** With `wE = 0`, environmental conditions do
   not yet influence `S`; enabling `E` before stable site baselines exist would
   add noise.

## Roadmap (no schema or formula change required)

- **R → effort-normalized rarity.** Once the GBIF occurrence and IUCN fetches
  are configured (a one-time setup step), `R` can graduate from dataset relative
  frequency to inverse GBIF occurrence density, optionally weighted by IUCN
  threat category (LC→CR). The schema already reserves `gbif_snapshot_date` and
  `iucn_fetch_date`. GBIF/IUCN accounts are **not** required for the current
  local-proxy `R`.
- **E → active.** When `site_baselines` hold enough per-channel history, set
  `wE > 0` and re-normalize the weights.
- **Spatial novelty.** For mobile or transect deployments, `N` can be computed
  per spatial cell using the GPS fix, so a species in a new location reads as
  novel. For a fixed station this is negligible and GPS carries no separate
  salience term.

## Validation plan

Before publication, defend `S` with: (1) a sensitivity analysis over `wN, wR,
wE`, reporting how rankings shift with the weights; (2) calibration against
expert-judged importance on a labeled subset, tuning the weights to that
reference; and (3) an ablation showing each term's contribution. Because every
input (`C`, `A_eff`, `N`, `R`, `E`) and the composite `S` are stored with their
provenance, any published score is fully reconstructable from the record.

## Data dependencies

| Term | Source | Kind | Available now |
|------|--------|------|---------------|
| `C_eff` | screening model (`best_confidence`), else 0 | measured | yes (0 if none) |
| `A_eff` | acoustic model matched to event, else 0 | measured | yes (0 if none) |
| `N` | surprisal of station counts (SQLite) + `K` | derived | yes |
| `R` | surprisal of dataset counts (SQLite) + `K` | derived | yes (proxy) |
| `K` | model label count | measured | yes |
| `E` | `site_baselines` deviation | derived | when baselines exist |
| `S` | composite of the above | inferred | yes (`wE = 0`) |
