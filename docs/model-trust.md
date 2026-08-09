# Model accuracy and model trust

This document is the authoritative definition of two derived quantities that
answer a question salience and the longitudinal pass do not: how good is a given
model, per species, judged against expert review, and how much should a single
detection be believed. Both are additive. Neither alters the salience formula in
[`docs/salience.md`](salience.md), the longitudinal pass in
[`docs/dream-pass.md`](dream-pass.md), the database of measured facts, or any
shipped behaviour. They read the record that already exists and produce a new,
clearly inferred layer on top of it.

## What they are, and are not

Two questions are kept separate. Salience asks how much an observation matters
(importance). Model trust asks how much a detection should be believed
(reliability of the model that produced it). They share the measured detection
evidence `D` but are orthogonal, so Model trust is a new parallel quantity and
does not touch salience.

Every value defined here is **derived and inferred**, not a measurement. It is
computed from measured model confidences and human expert verdicts, but it is
itself a model of reliability. It is tagged as inference, keyed to the exact
model version that produced the underlying detections, and never written into the
measured record. It changes no screening confidence, no salience value, and no
longitudinal-pass result. When an input is missing, the value is reported as not
computable rather than as a false zero.

## Per-species accuracy of a model

For a model version `M` and a species `s`, consider every expert-reviewed
detection where `M` predicted `s`. Let `c` be the confirms, `r` the relabels (an
expert changed it to another species), and `x` the rejects (no organism). The
reviewed total is `n = c + r + x`.

```
Acc(s, M) = (c + 1) / (n + 2)          Laplace-smoothed precision, in [0, 1]
```

The `+1 / +2` is a weak uniform Beta prior, so a single review does not read as 0
or 100 percent. This is per-species precision as judged by experts, so a low
`Acc(s, M)` marks that species as a fine-tuning target. When `n = 0`, `Acc(s, M)`
is not computed and the interface says "no expert reviews yet".

A conservative secondary figure, the Wilson score lower bound of the raw
precision `c / n`, is offered beside the primary number. It answers a more
cautious question: given how little a species has been reviewed, how low could the
model's precision on it plausibly be. It is never the primary metric.

### Which model and which species a verdict counts against

A verdict is keyed to the model of the modality that produced the reviewed call.
A vision detection is counted against the screening model version
(`observations.screening_model_version`); an audio detection against the acoustic
model version (`observations.acoustic_model_version`). The desktop verification
model (`observation_verification.rfdetr_version`) is a separate re-scorer and is
shown apart, never folded into a screening or acoustic accuracy.

A correction that names a specific detection judges that detection's predicted
species. A correction on the whole event names no species, so it is applied only
when the event has a single detection, where the target is unambiguous. On a
multi-taxon event an event-level verdict is left uncounted rather than guessed
onto a species, because a fabricated attribution is worse than a missing one. A
detection-level verdict always takes precedence over an event-level one for the
same detection. A detection with no resolved species contributes to nothing,
since an accuracy per species needs a species.

Accuracy is computed over the whole expert-reviewed record and is cumulative
rather than limited to any date window, because per-species precision is a
property of a model, not of a range.

## Per-model rollups

```
micro (event-weighted)   = sum(c) / sum(n)              overall correctness
macro (species-averaged) = mean over s of Acc(s, M)     fair across species
```

Micro weights every reviewed detection equally. Macro weights every species
equally, so it exposes a model that is weak on a rare species it seldom predicts.
Both are reported. From the relabel targets, a small confusion view records which
species `M` is confused with when it calls `s`.

## Model trust

Model trust is distinct from the per-frame "Event trust" shown in a detection's
frame-curation view, which is the share of an event's frames kept after per-frame
review. That is a curation weight over frames; model trust is a reliability score
for the model's identification. They are deliberately named apart. In the stored
response the model-trust value travels under the field name `event_trust`.

For an event that model `M` labelled species `s`, with visual confidence `C` and
acoustic confidence `A` (each 0 when that channel did not fire):

```
D  = 1 - (1 - C) * (1 - A)             detection evidence, reused from salience
ET = D * Acc(s, M)                     in [0, 1]
```

Read plainly: how strongly it was detected, times how reliably this model gets
this species right. `D` is the exact detection evidence salience uses, imported
rather than re-derived, so Model trust is consistent with salience without
altering it. It is naturally multimodal, since visual and acoustic corroborate
through `D`, and it degrades gracefully to a single modality. When `Acc(s, M)` is
not computable (no expert reviews for that species under that model), Model trust
is not computable either and is shown as "not yet rated", never as a number.

The species and model for an event's Model trust are taken from the event's
highest-confidence identified detection, which is the call a card headlines and a
reviewer judges. `C` is the strongest visual confidence in the event and `A` the
strongest acoustic one.

## Where each value appears

Per-species accuracy, the two rollups, and the confusion view are shown under
Brain, in the Learning and Auditing surface, beside the retraining export they
inform, with the lowest-accuracy species first. Model trust appears on each
Detections and Audio card as a compact chip, and in a detection's derivation
panel beside salience, always labelled as inference and tagged with the model.
The generated report carries a per-species accuracy section (a CSV table with a
companion rollups table, and a matching PDF section), labelled inferred and
model-keyed, computed over the whole reviewed record rather than the report
window.

## Rules that make it defensible

Every accuracy, Model trust, and rollup is keyed to a specific model version and
displays that model as a provenance tag; a different model yields different
numbers, and no trust figure is ever shown without saying which model it belongs
to. Every value is derived and inferred, tagged as such, and never written into
the measured record. Nothing is fabricated: a missing input produces an explicit
not-computable state, not a zero.

## Roadmap

Two refinements are deliberately deferred so the first build stays simple and
defensible. Desktop verification agreement may later feed Model trust, either by
substituting the verifier's confidence for `C` on a verified event or by
multiplying Model trust by an agreement factor. A fuller recall and
confusion-matrix view may be added; the current build ships precision plus the
relabel confusion counts, because recall needs reviews of events the model called
something else.
