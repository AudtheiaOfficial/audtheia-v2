# Contributing to Audtheia V2

Thank you for your interest in improving Audtheia. This project is built for
researchers and conservation practitioners, so contributions are reviewed for
scientific correctness as much as for code quality. This guide explains how to
report problems, propose changes, and work within the project's core design
rules.

## Ways to contribute

### Report a bug

Open a [GitHub Issue](https://github.com/AudtheiaOfficial/audtheia-v2/issues) and
include: what you did, what you expected, what happened instead, the platform you
ran on (the desktop hub on Windows, macOS, or Linux, or a Raspberry Pi field
station), and any message the application or setup printed. A minimal way to
reproduce the problem is the single most helpful thing you can provide. Do not
attach captured field data, a database file, or any credential.

### Request a feature

Open an issue describing the problem you are trying to solve and how it fits a
real monitoring workflow. Proposals grounded in a concrete field or research need
are easiest to evaluate.

### Propose code

1. Fork the repository and create a branch for your change.
2. Keep each change focused on one thing.
3. Make sure the test suite passes (see below).
4. Open a pull request that explains what changed and why, and note how you
   verified it.

## The design rules a contribution must respect

These are not style preferences. They are the guarantees that make Audtheia's
data trustworthy, and a change that violates one will be asked for revision.

- **The provenance firewall.** Every value written to the database carries a
  provenance tag and a quality-control status. A measured value, a value looked
  up from a reference database, a model's classification, an inference, and a
  candidate pattern proposed by the longitudinal pass must all remain permanently
  distinguishable. Do not write a value in a way that blurs a measurement with an
  inference, and do not present an inferred value as if it were measured.
- **Detection is the trigger.** Capture is event-driven, not scheduled. Only
  reports and the longitudinal pass run on a schedule. Do not add timer-based
  capture logic.
- **The platform is taxon-neutral.** Do not name a specific model family or
  assume a single taxon in interface copy, a configuration key, or a validation
  rule. The system is indifferent to what is being studied, and its language and
  its rules should stay that way. Describe a detector by its runtime, not by an
  architecture that implies one kind of subject.
- **No enforced format on acoustic models.** A visual slot may require a specific
  file format because its runtime fixes it. An acoustic slot must not, because the
  supported acoustic models differ.
- **Nothing personal or private in the repository.** No absolute machine paths,
  no credentials, no station identifiers, no captured data, and no personal
  information in any tracked file. Configuration that varies by machine belongs in
  the git-ignored local overrides file, not in the tracked configuration.
- **No fabricated ecological or taxonomic claims.** Cite a source or mark a claim
  as uncertain.

## Running the tests

The suite runs on mocked hardware, so no camera, accelerator, or Raspberry Pi is
required. From the repository root:

```
python tests/run_all.py
```

`run_all.py` discovers every `tests/test_*.py` automatically. Please run it before
opening a pull request, and add or update tests for the behavior you change.

## Style and scope

Configuration that varies between deployments belongs in `config/settings.json`
and is documented in `config/README.md`; never hardcode model names, species
names, credentials, or file paths. Prefer complete, self-consistent changes over
partial ones, and check a change against the database schema and the settings
before proposing it.

By contributing, you agree that your contributions are licensed under the MIT
License that covers this project.
