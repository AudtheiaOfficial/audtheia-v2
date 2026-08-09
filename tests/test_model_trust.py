"""Verification for per-species model accuracy and model trust.

Path: tests/test_model_trust.py

Proves the settled mathematics of the model-trust layer against a small synthetic
record, never the live database. The properties checked are the ones the design
depends on:

  - the Laplace-smoothed precision is exactly (c + 1) / (n + 2), and is NOT a
    number at n = 0 but an explicit "not computable",
  - the Wilson lower bound is a conservative secondary figure and is also not
    computable at n = 0,
  - model trust is D * Acc, is naturally multimodal through D, degrades to a
    single modality, and is not computable when Acc is not,
  - the per-species table counts confirms, relabels and rejects correctly, keys
    each species to its model version, records what a relabel confused it with,
    and sorts the fine-tuning targets first,
  - the micro and macro rollups differ exactly as intended on a species-imbalanced
    model, so a weakness on a rarely-predicted species is not hidden by a strong
    common one,
  - the modality-to-model and single-taxon event-level mapping rules resolve a
    real event's detections into the right records, and refuse to guess a species
    onto an event-level verdict when the event is multi-taxon.

Standard library only. It imports the pure math module and never opens a database
or touches the record, because the mathematics is meant to be provable on its own.

Run: python tests/test_model_trust.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from audtheia.analysis.model_trust import (  # noqa: E402
    accuracy_index,
    accuracy_table,
    event_review_records,
    event_trust,
    laplace_accuracy,
    latest_verdicts,
    model_rollups,
    wilson_lower_bound,
)
from audtheia.pipeline.salience import detection_evidence  # noqa: E402

CHECKS = {"passed": 0, "failed": 0}


def check(label: str, condition: bool) -> None:
    if condition:
        CHECKS["passed"] += 1
        print(f"  PASS  {label}")
    else:
        CHECKS["failed"] += 1
        print(f"  FAIL  {label}")


def _close(a, b, tol=1e-9) -> bool:
    return a is not None and b is not None and abs(a - b) <= tol


def test_laplace_accuracy() -> None:
    print("\nLaplace-smoothed precision, with an honest zero")
    check("no reviews is not computable, never 0.0", laplace_accuracy(0, 0) is None)
    check("one confirm reads as 2/3, not 1.0", _close(laplace_accuracy(1, 1), 2 / 3))
    check("one relabel or reject reads as 1/3, not 0.0", _close(laplace_accuracy(0, 1), 1 / 3))
    check("five of five reads as 6/7, tempered from 1.0", _close(laplace_accuracy(5, 5), 6 / 7))
    check("zero of ten reads as 1/12, tempered from 0.0", _close(laplace_accuracy(0, 10), 1 / 12))
    check("the smoothed value stays inside [0, 1]",
          0.0 <= laplace_accuracy(3, 4) <= 1.0)


def test_wilson_lower_bound() -> None:
    print("\nWilson lower bound as a conservative secondary figure")
    check("no reviews is not computable", wilson_lower_bound(0, 0) is None)
    # Eight confirms of ten at z = 1.96 has a known lower bound near 0.490.
    lb = wilson_lower_bound(8, 10)
    check("eight of ten has a lower bound near 0.490", _close(lb, 0.4902, tol=5e-3))
    check("the lower bound sits below the raw precision", lb < 0.8)
    check("the lower bound stays inside [0, 1]", 0.0 <= lb <= 1.0)


def test_event_trust() -> None:
    print("\nmodel trust is D times accuracy, multimodal, and honestly missing")
    # Both channels fired: D = 1 - (1 - 0.8)(1 - 0.5) = 0.9, so ET = 0.9 * 0.9.
    both = event_trust(0.8, 0.5, 0.9)
    check("both modalities corroborate through D", _close(both, 0.81))
    check("ET reuses salience's exact detection evidence",
          _close(detection_evidence(0.8, 0.5), 0.9))
    # One channel only: the silent channel contributes 0, so D is the live one.
    single = event_trust(0.6, 0.0, 0.5)
    check("a single modality degrades gracefully", _close(single, 0.30))
    check("ET is not computable when accuracy is not",
          event_trust(0.9, 0.9, None) is None)
    check("ET stays inside [0, 1]", 0.0 <= event_trust(1.0, 1.0, 1.0) <= 1.0)


def _records():
    """A small synthetic set of resolved review records across two models.

    Model 'screen-v1' predicting 'Sciurus carolinensis' is reviewed four times:
    three confirms and one relabel to 'Sciurus niger', so it is mostly right and
    is confused with the fox squirrel once. The same model predicting a rarely
    seen 'Tamias striatus' is reviewed once and rejected, a weak, small-n case.
    A different model, 'acoustic-v2', predicting 'Cardinalis cardinalis' is
    reviewed twice, both confirmed.
    """
    grey = {"model_version": "screen-v1", "species_key": "gray",
            "species_label": "Sciurus carolinensis"}
    records = [
        {**grey, "verdict": "confirm"},
        {**grey, "verdict": "confirm"},
        {**grey, "verdict": "confirm"},
        {**grey, "verdict": "relabel", "corrected_label": "Sciurus niger",
         "corrected_key": "fox"},
        {"model_version": "screen-v1", "species_key": "chip",
         "species_label": "Tamias striatus", "verdict": "reject"},
        {"model_version": "acoustic-v2", "species_key": "card",
         "species_label": "Cardinalis cardinalis", "verdict": "confirm"},
        {"model_version": "acoustic-v2", "species_key": "card",
         "species_label": "Cardinalis cardinalis", "verdict": "confirm"},
    ]
    return records


def test_accuracy_table() -> None:
    print("\nThe per-species table counts, keys, confuses, and orders correctly")
    table = accuracy_table(_records())
    by_key = {(r["model_version"], r["species_key"]): r for r in table}

    grey = by_key[("screen-v1", "gray")]
    check("gray squirrel has three confirms", grey["confirms"] == 3)
    check("gray squirrel has one relabel", grey["relabels"] == 1)
    check("gray squirrel has four reviews", grey["reviewed"] == 4)
    check("gray squirrel accuracy is 4/6", _close(grey["accuracy"], 4 / 6))
    check("gray squirrel confusion names the fox squirrel",
          grey["confused_with"] == {"Sciurus niger": 1})

    chip = by_key[("screen-v1", "chip")]
    check("chipmunk has one reject and no confirm", chip["rejects"] == 1 and chip["confirms"] == 0)
    check("chipmunk accuracy is 1/3", _close(chip["accuracy"], 1 / 3))

    card = by_key[("acoustic-v2", "card")]
    check("the cardinal is keyed to its own model, not merged",
          card["model_version"] == "acoustic-v2" and card["reviewed"] == 2)
    check("the cardinal accuracy is 3/4", _close(card["accuracy"], 3 / 4))

    check("every row is tagged as inference", all(r["provenance"] == "inferred" for r in table))
    # Low accuracy first: chipmunk (1/3) before gray (2/3) before cardinal (3/4).
    order = [(r["model_version"], r["species_key"]) for r in table]
    check("the fine-tuning target sorts to the top",
          order[0] == ("screen-v1", "chip"))
    check("accuracy is non-decreasing down the table",
          all(table[i]["accuracy"] <= table[i + 1]["accuracy"] for i in range(len(table) - 1)))


def test_rollups_micro_vs_macro() -> None:
    print("\nMicro and macro rollups diverge on an imbalanced model")
    # 'imb' predicts a common species reviewed ten times (nine right) and a rare
    # species reviewed once (wrong). Micro is dragged up by the common species;
    # macro, weighting species equally, exposes the rare-species weakness.
    common = {"model_version": "imb", "species_key": "common",
              "species_label": "Common sp."}
    rare = {"model_version": "imb", "species_key": "rare", "species_label": "Rare sp."}
    records = [{**common, "verdict": "confirm"} for _ in range(9)]
    records.append({**common, "verdict": "reject"})
    records.append({**rare, "verdict": "reject"})

    table = accuracy_table(records)
    rollups = model_rollups(table)
    imb = rollups["imb"]
    # micro = sum(c) / sum(n) = 9 / 11
    check("micro is event-weighted (9/11)", _close(imb["micro"], 9 / 11))
    # macro = mean(Acc_common, Acc_rare) = mean(10/12, 1/3)
    check("macro is species-averaged (mean of 10/12 and 1/3)",
          _close(imb["macro"], (10 / 12 + 1 / 3) / 2))
    check("micro reads higher than macro here", imb["micro"] > imb["macro"])
    check("the rollup carries its species and review counts",
          imb["species"] == 2 and imb["reviewed"] == 11)
    check("the rollup is tagged as inference", imb["macro"] is not None and rollups["imb"]["provenance"] == "inferred")


def test_accuracy_index() -> None:
    print("\nThe accuracy index feeds model trust by model and species")
    table = accuracy_table(_records())
    index = accuracy_index(table)
    check("a reviewed pair resolves to its accuracy",
          _close(index[("screen-v1", "gray")], 4 / 6))
    check("an unreviewed pair is absent, so its model trust is not computable",
          ("screen-v1", "never-seen") not in index)


def test_event_review_mapping() -> None:
    print("\nThe modality-to-model and event-level rules resolve real events")

    # A two-taxon vision event. A detection-level relabel on one box, and an
    # event-level reject that must NOT be guessed onto either species.
    multi_children = [
        {"id": "d1", "modality": "vision", "gbif_usage_key": "gray",
         "scientific_name": "Sciurus carolinensis"},
        {"id": "d2", "modality": "vision", "gbif_usage_key": "fox",
         "scientific_name": "Sciurus niger"},
    ]
    multi_corrections = [
        {"detection_id": "d1", "verdict": "relabel",
         "corrected_scientific_name": "Sciurus niger", "corrected_gbif_usage_key": "fox",
         "corrected_at": "2026-01-02"},
        {"detection_id": None, "verdict": "reject", "corrected_at": "2026-01-01"},
    ]
    recs = event_review_records(
        multi_children, multi_corrections,
        screening_model_version="screen-v1", acoustic_model_version="acoustic-v2",
    )
    check("a multi-taxon event yields only its box-level verdict", len(recs) == 1)
    check("the box relabel is keyed to the screening model",
          recs[0]["model_version"] == "screen-v1" and recs[0]["verdict"] == "relabel")
    check("the event-level reject is not guessed onto a species",
          all(r["verdict"] != "reject" for r in recs))

    # A single-taxon audio event with only an event-level confirm: it applies,
    # and is keyed to the acoustic model.
    single_children = [
        {"id": "a1", "modality": "audio", "gbif_usage_key": "card",
         "scientific_name": "Cardinalis cardinalis"},
    ]
    single_corrections = [
        {"detection_id": None, "verdict": "confirm", "corrected_at": "2026-01-01"},
    ]
    recs2 = event_review_records(
        single_children, single_corrections,
        screening_model_version="screen-v1", acoustic_model_version="acoustic-v2",
    )
    check("a single-taxon event-level verdict applies to the one detection", len(recs2) == 1)
    check("an audio detection is keyed to the acoustic model",
          recs2[0]["model_version"] == "acoustic-v2")

    # Detection-level precedence: a box confirm overrides an older event-level
    # reject on the same single-taxon event.
    precedence_corrections = [
        {"detection_id": "a1", "verdict": "confirm", "corrected_at": "2026-02-01"},
        {"detection_id": None, "verdict": "reject", "corrected_at": "2026-01-01"},
    ]
    recs3 = event_review_records(
        single_children, precedence_corrections,
        screening_model_version="screen-v1", acoustic_model_version="acoustic-v2",
    )
    check("a box verdict takes precedence over an event-level one",
          len(recs3) == 1 and recs3[0]["verdict"] == "confirm")

    # A detection with no species at all is skipped, since accuracy is per species.
    nameless = [{"id": "x1", "modality": "vision", "gbif_usage_key": None,
                 "scientific_name": None}]
    recs4 = event_review_records(
        nameless, [{"detection_id": "x1", "verdict": "confirm", "corrected_at": "2026-01-01"}],
        screening_model_version="screen-v1", acoustic_model_version=None,
    )
    check("a detection with no species contributes nothing", recs4 == [])

    # A class-label-only detection (a model class name that never matched the
    # backbone, so no gbif key and no scientific name) still names a taxon and is
    # counted, keyed and labelled by that class label. This is the case that made
    # reviewed example events read "not yet rated" before the identity fallback.
    class_label = [{"id": "b1", "modality": "vision", "gbif_usage_key": None,
                    "scientific_name": None, "common_name": "annas-hummingbird"}]
    recs5 = event_review_records(
        class_label, [{"detection_id": None, "verdict": "confirm", "corrected_at": "2026-01-01"}],
        screening_model_version="screen-v1", acoustic_model_version=None,
    )
    check("a class-label-only detection is counted", len(recs5) == 1)
    check("the class label is its species key and label",
          recs5[0]["species_key"] == "annas-hummingbird" and recs5[0]["species_label"] == "annas-hummingbird")


def test_latest_verdicts() -> None:
    print("\nThe current verdict per target is the newest row")
    # Rows arrive newest first, exactly as the storage layer returns them.
    rows = [
        {"detection_id": "d1", "verdict": "reject", "corrected_at": "2026-03-01"},
        {"detection_id": "d1", "verdict": "confirm", "corrected_at": "2026-01-01"},
        {"detection_id": None, "verdict": "relabel", "corrected_at": "2026-02-01"},
    ]
    latest = latest_verdicts(rows)
    check("a changed mind collapses to the newest verdict",
          latest["d1"]["verdict"] == "reject")
    check("the event-level target is kept separately",
          latest[None]["verdict"] == "relabel")


def main() -> int:
    print("=" * 72)
    print("Model accuracy and model trust")
    print("=" * 72)
    test_laplace_accuracy()
    test_wilson_lower_bound()
    test_event_trust()
    test_accuracy_table()
    test_rollups_micro_vs_macro()
    test_accuracy_index()
    test_event_review_mapping()
    test_latest_verdicts()
    print("\n" + "=" * 72)
    print(f"RESULT: {CHECKS['passed']} passed, {CHECKS['failed']} failed")
    print("=" * 72)
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
