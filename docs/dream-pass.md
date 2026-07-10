# The Dream Pass: Longitudinal Analysis Explained

This guide explains what Audtheia's dream pass does, for a research audience deciding whether and how to trust its outputs. The short version is that the dream pass is where Audtheia looks across the whole record of a site, over weeks and months, to surface regularities and anomalies that no one specified in advance, and that it presents everything it finds as a candidate hypothesis to investigate, never as an established finding.

## Why it exists

A single observation tells you what was in front of the sensor at one moment. The scientific value of a fixed station is that those moments accumulate: the longer a station runs, the more its record can reveal about how a place changes. The dream pass is the part of Audtheia that reads that accumulated record and asks what patterns it holds, population trends, seasonal timing, shifting baselines, co-occurrences between species and environmental conditions, and anomalies that stand out against a site's own history. Its explicit purpose is to surface patterns a person would be unlikely to specify or notice by hand, not as a side effect but as the point.

## Where it runs, and what it is not

The dream pass runs only on the desktop hub, over the full longitudinal record, on a schedule you set (daily, weekly, biweekly, or on demand). It is one of only two scheduled activities in the whole system; everything else is triggered by real events. The field station does no generative work at all: its quality-control step is a deterministic engine, and no language model runs on it. This separation is deliberate. Speculative, pattern-finding inference is kept out of data capture entirely and concentrated, downstream and clearly labeled, in the dream pass, where it belongs and where it can be checked against the complete record.

## How a pass is structured: consolidate, downscale, integrate

A dream pass is not a single sweep over all the data. Re-scanning the entire archive on every pass would scale with the age of the station and become the power-hungry failure a long-running deployment must avoid. Instead each pass runs a two-phase cycle, drawn as an organizing analogy from how sleeping brains consolidate memory, in a fixed order.

- **Consolidate.** The pass first replays new and salient observations into a permanent, compact statistical summary of the site: running statistics per time period, per species, and per signal. This is cheap, deterministic, and uses no language model. Crucially, the regularity is extracted into this durable summary *before* anything is pruned, because a longitudinal pattern is a correlated structure spread across many records, and discarding the records before summarizing them would destroy it.
- **Downscale.** The pass then prunes its own derived working memory, the store of exemplars and candidate patterns, toward a bounded, salience-ranked set: it keeps the strongest and lets weakly-supported candidates decay. This never touches the underlying observation archive, which stays append-only and complete. Only the derived scratch memory is trimmed.
- **Integrate.** Finally, the pass runs generative recombination over the compressed summary and the bounded set of exemplars, producing novel candidate patterns, de-duplicated against what it already knows. This is the expensive, claim-producing phase, and it is the only one that is gated.

Because the order is fixed and the integration phase works over a bounded summary rather than the raw archive, the cost of a pass does not grow with the age of the station: a station running for three years dreams as quickly as one running for three months.

## The gate: only confirmed detections seed new claims

Not every observation is allowed to shape a generated hypothesis. The consolidation phase draws on all synced observations, verified or not, because aggregate statistics tolerate the occasional stray false positive from the field. The generative integration phase, however, runs only over observations that the desktop verification step has cleared. In other words, a detection shapes the site's baselines immediately, but it does not help seed a new generated pattern until the high-accuracy desktop model has confirmed it. This keeps every generated candidate anchored to detections that have been checked, the same discipline the field tier already enforces at capture, applied again at the point where the system starts to reason.

## Budget and safe interruption

Work is measured in units of work, not wall-clock time: a consolidation batch is an epoch, one full consolidate-and-integrate traversal is a cycle, and a pass is however many cycles are needed to clear the backlog since the last pass. A pass is checkpointed at the end of each cycle, so asking it to stop means finish the current cycle, commit, and stop, never a half-written result. A pass can be paused and resumed from where it left off, and the progress display in the interface reflects the phase, the cycle, how many observations have been consolidated, and how many candidate patterns have been proposed so far.

## How to read the outputs

Every output of the dream pass is a candidate pattern, a hypothesis, not a conclusion. Each one is tagged as a dream-derived result, and each carries the evidence you need to judge it:

- an **effect size**, so you can see how large the pattern is, not just whether it cleared a threshold,
- the **data span** behind it, so a claim made over three weeks is never mistaken for one made over three years, and
- a **multiple-comparison-adjusted significance value** (a Benjamini-Hochberg q-value computed across the pass's candidates), so the many comparisons a broad search performs do not admit chance patterns at a naive threshold.

Treat these as leads worth investigating with proper analysis and, where possible, more data, not as findings to report directly. Several statistical cautions apply and are worth keeping in mind: environmental time series are autocorrelated, so nominal significance can overstate confidence; a trend claimed within a single year can reflect the seasonal cycle rather than a genuine interannual change, which the data span makes explicit; and a minimum span and sample size gate the strongest claims. The dream pass is built to be honest about these limits rather than to hide them, which is what makes its candidates useful as a starting point for real inquiry.

## Provenance is the guarantee, not the metaphor

The consolidate, downscale, and integrate structure is an organizing analogy for how the software is built, and the names it borrows from neuroscience are a way to reason about where work belongs. They are never a claim that any part of Audtheia is literally a brain, nor that it runs on special neuromorphic hardware. The scientific guarantee is not the metaphor; it is the provenance system. Every value the platform stores is labeled by its source, a sensor reading, a reference-database lookup, a model output, a downstream interpretation, or a dream-derived pattern, and measured facts and inferred claims never blend. A dream-derived pattern is always labeled as such, always traceable back to the confirmed observations that produced it, and always offered as a question rather than an answer.

## Where to go next

- To set up the desktop and run a pass, see the [README](../README.md).
- To understand the models behind the detections a pass reasons over, see the [custom models guide](custom-models.md).
- To build a station that feeds the record a pass reads, see the [hardware guide](hardware.md).
