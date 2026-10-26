"""Verification for choosing training candidates from the record.

Path: tests/test_training_candidates.py

Builds a small record in a temporary folder through the real storage layer and
checks that the three criteria, the label an event ends up with, and the steps
still missing are worked out exactly as the corrections, frame review and
verification rules say, and that reading candidates changes nothing. Covers
confirm, relabel to a different species, reject, a mixed-species event judged
per detection, an event-level verdict that applies only to a single-detection
event, the newest verdict winning, a verifier disagreement an expert resolved,
every combination of the three criteria, and a database that predates the review
tables. A genuine query fault must surface, never be swallowed as "no reviews".

Uses the standard library, NumPy and Pillow, plus the project itself.
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from training_fixture import World  # noqa: E402
from audtheia.training import candidates as rules  # noqa: E402

CHECKS = {"passed": 0, "failed": 0}
A = ("Aplysina fistularis", 100)
B = ("Aplysina lacunosa", 102)
C = ("Poecile carolinensis", 200)


def check(label: str, condition: bool) -> None:
    CHECKS["passed" if condition else "failed"] += 1
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")


def by_id(rows: list) -> dict:
    return {r["observation_id"]: r for r in rows}


def digest(world: World) -> dict:
    out = {}
    with world.db.connect() as conn:
        for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall():
            rows = conn.execute(f'SELECT * FROM "{name}"').fetchall()
            out[name] = (len(rows), hashlib.sha256(repr(sorted(repr(tuple(r)) for r in rows)).encode("utf-8")).hexdigest())
    return out


def test_labels(work: Path) -> None:
    print("\n[1] The label an event ends up with")
    w = World(work / "labels")
    plain = w.vision_event([A])
    confirmed = w.vision_event([A]); w.confirm(confirmed)
    rejected = w.vision_event([A]); w.reject(rejected)
    relabelled = w.vision_event([A]); w.relabel(relabelled, B[0], B[1])
    changed_mind = w.vision_event([A]); w.confirm(changed_mind); w.reject(changed_mind)
    rows = by_id(w.db.training_candidates("vision"))

    check("an event nobody judged carries the model's own label", rows[plain["id"]]["label_state"] == rules.LABEL_MODEL)
    check("an unjudged event is not expert confirmed", rows[plain["id"]]["criteria"]["expert_confirmed"] is False)
    check("an unjudged event is told to wait for a review", "review" in rows[plain["id"]]["missing_steps"])
    check("a confirmed event is positive", rows[confirmed["id"]]["label_state"] == rules.LABEL_POSITIVE)
    check("a confirmed event is expert confirmed", rows[confirmed["id"]]["criteria"]["expert_confirmed"] is True)
    check("a confirmed event keeps the model's species", rows[confirmed["id"]]["taxa"][0]["class_key"] == "gbif:100")
    check("a confirmed detection is labelled by an expert", rows[confirmed["id"]]["taxa"][0]["source"] == "expert")
    check("a rejected event is negative", rows[rejected["id"]]["label_state"] == rules.LABEL_NEGATIVE)
    check("a rejected event counts as expert confirmed", rows[rejected["id"]]["criteria"]["expert_confirmed"] is True)

    rel = rows[relabelled["id"]]["taxa"][0]
    check("a relabel takes the expert's species", rel["class_key"] == "gbif:102" and rel["class_name"] == B[0])
    check("a relabel is labelled by an expert", rel["source"] == "expert" and rel["verdict"] == "relabel")
    check("a relabelled detection can still be found by the name the model used", "aplysina fistularis" in rel["match_names"])
    check("the newest verdict wins when an expert changed their mind", rows[changed_mind["id"]]["label_state"] == rules.LABEL_NEGATIVE)


def test_scope_of_verdicts(work: Path) -> None:
    print("\n[2] Which detection a verdict applies to")
    w = World(work / "scope")
    mixed = w.vision_event([A, C])
    w.confirm(mixed, detection=mixed["children"][0])
    w.reject(mixed, detection=mixed["children"][1])
    only_event_level = w.vision_event([A, C])
    w.confirm(only_event_level)
    single = w.vision_event([A]); w.confirm(single)
    detection_beats_event = w.vision_event([A])
    w.confirm(detection_beats_event)
    w.relabel(detection_beats_event, B[0], B[1], detection=detection_beats_event["children"][0])
    half = w.vision_event([A, C]); w.confirm(half, detection=half["children"][0])
    rows = by_id(w.db.training_candidates("vision"))

    m = rows[mixed["id"]]
    check("a mixed event judged per detection is positive for the confirmed species", [t["class_key"] for t in m["positive_taxa"]] == ["gbif:100"])
    check("the rejected species of a mixed event is not a positive label", all(t["class_key"] != "gbif:200" for t in m["positive_taxa"]))
    check("a mixed event with every detection judged is expert confirmed", m["criteria"]["expert_confirmed"] is True)
    check("an event-level verdict does not speak for a two-detection event", rows[only_event_level["id"]]["label_state"] == rules.LABEL_MODEL)
    check("an event-level verdict speaks for a one-detection event", rows[single["id"]]["label_state"] == rules.LABEL_POSITIVE)
    check("a detection-level verdict beats an event-level one", rows[detection_beats_event["id"]]["taxa"][0]["class_key"] == "gbif:102")
    check("an event with one of two detections judged is only partly judged", rows[half["id"]]["label_state"] == rules.LABEL_PARTIAL)
    check("a partly judged event is not expert confirmed", rows[half["id"]]["criteria"]["expert_confirmed"] is False)


def test_combinations(work: Path) -> None:
    print("\n[3] Every combination of the three criteria")
    w = World(work / "combos")
    made = {}
    for expert in (False, True):
        for qc in (False, True):
            for verified in (False, True):
                ev = w.vision_event([A], qc="qc_passed" if qc else "qc_pending", verified=verified)
                if expert:
                    w.confirm(ev)
                made[(expert, qc, verified)] = ev
    rows = by_id(w.db.training_candidates("vision"))
    names = set()
    for key, ev in made.items():
        names.add(rules._combination_name(rows[ev["id"]]["criteria"]))
    check("the eight events fall in eight distinct combinations", len(names) == 8)
    check("the combination names are exactly the documented eight", names == set(rules.COMBINATIONS))

    everything = frozenset(rules.CRITERIA)
    eligible = [k for k, ev in made.items() if rules.is_eligible(rows[ev["id"]], everything)]
    check("requiring all three keeps only the event that has all three", eligible == [(True, True, True)])
    only_expert = [k for k, ev in made.items() if rules.is_eligible(rows[ev["id"]], frozenset({"expert_confirmed"}))]
    check("requiring only Expert Confirmed keeps four events", len(only_expert) == 4 and all(k[0] for k in only_expert))
    none_required = [k for k, ev in made.items() if rules.is_eligible(rows[ev["id"]], frozenset())]
    check("requiring nothing keeps all eight", len(none_required) == 8)
    check("an unknown criterion is refused, not ignored", _raises(lambda: rules.required_criteria(["expert_confirmed", "nonsense"])))

    summary = rules.summarize(list(rows.values()), everything)
    check("the summary counts every event", summary["candidates"] == 8)
    check("the summary counts the events that qualify", summary["eligible"] == 1)
    check("each criterion is counted over all eight events", all(summary["per_criterion"][c] == 4 for c in rules.CRITERIA))
    check("the combination counts add up to the total", sum(summary["combinations"].values()) == 8)

    waiting = rows[made[(False, True, True)]["id"]]
    check("an event short of review says exactly which step is missing", waiting["missing_steps"] == ["review"])
    waiting = rows[made[(True, False, False)]["id"]]
    check("an event short of two steps lists both", sorted(waiting["missing_steps"]) == ["qc", "verify"])


def test_rejections_and_verifier(work: Path) -> None:
    print("\n[4] False alarms and verifier disagreements")
    w = World(work / "verifier")
    rejected_unverified = w.vision_event([A], qc="qc_pending", verified=False); w.reject(rejected_unverified)
    disagreed = w.vision_event([A], agrees=0); w.relabel(disagreed, B[0], B[1])
    disagreed_unresolved = w.vision_event([A], agrees=0)
    rows = by_id(w.db.training_candidates("vision"))
    everything = frozenset(rules.CRITERIA)
    check("a rejected event is eligible whatever the other two criteria say", rules.is_eligible(rows[rejected_unverified["id"]], everything))
    check("a verifier disagreement an expert resolved is flagged", rows[disagreed["id"]]["disagreement_resolved"] is True)
    check("a verifier disagreement nobody resolved is not flagged", rows[disagreed_unresolved["id"]]["disagreement_resolved"] is False)
    check("an event with no judgement at all is never a negative", rows[disagreed_unresolved["id"]]["label_state"] != rules.LABEL_NEGATIVE)
    nothing = {"observation": {"id": "x", "qc_state": "qc_passed"}, "children": [], "corrections": []}
    check("an event with no detection of the kind is unresolved and never eligible",
          not rules.is_eligible(rules.evaluate_candidate(nothing, "vision"), frozenset()))


def test_frames_and_modality(work: Path) -> None:
    print("\n[5] Frame verdicts, modality and station filters")
    w = World(work / "frames")
    ev = w.vision_event([A]); w.confirm(ev)
    w.frame(ev, 0, "accurate"); w.frame(ev, 1, "accurate"); w.frame(ev, 2, "inaccurate")
    w.frame(ev, 1, "cleared")
    au = w.audio_event([C], station="ReefB"); w.confirm(au)
    rows = by_id(w.db.training_candidates("vision"))
    check("the latest verdict per frame is counted", rows[ev["id"]]["frame_verdicts"] == {"accurate": 1, "inaccurate": 1})
    check("an audio event is not a visual candidate", au["id"] not in rows)
    audio = by_id(w.db.training_candidates("audio"))
    check("an audio event is an audio candidate", au["id"] in audio and ev["id"] not in audio)
    check("a station filter keeps only that station", list(by_id(w.db.training_candidates("audio", station_id=au["station_id"]))) == [au["id"]])
    check("a station with nothing returns nothing", w.db.training_candidates("vision", station_id=au["station_id"]) == [])
    check("an unknown modality is refused", _raises(lambda: w.db.training_candidates("video")))


def test_read_only_and_degradation(work: Path) -> None:
    print("\n[6] Reading changes nothing, and failures are not hidden")
    w = World(work / "readonly")
    for _ in range(3):
        e = w.vision_event([A]); w.confirm(e); w.frame(e, 0, "accurate")
    before = digest(w)
    w.db.training_candidates("vision")
    w.db.training_candidates("audio")
    check("reading candidates leaves every table exactly as it was", digest(w) == before)

    old = World(work / "old")
    e = old.vision_event([A])
    with old.db.connect() as conn:
        conn.execute("DROP TABLE frame_review")
        conn.execute("DROP TABLE observation_corrections")
    rows = old.db.training_candidates("vision")
    check("a database that predates the review tables still reads", len(rows) == 1)
    check("without those tables an event simply waits for a review", rows[0]["label_state"] == rules.LABEL_MODEL and rows[0]["frame_verdicts"] == {"accurate": 0, "inaccurate": 0})

    broken = World(work / "broken")
    broken.vision_event([A])
    with broken.db.connect() as conn:
        conn.execute("ALTER TABLE observation_corrections RENAME COLUMN corrected_at TO moved_away")
    check("a real query fault is raised, not mistaken for 'no reviews'", _raises(lambda: broken.db.training_candidates("vision")))


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:  # noqa: BLE001
        return True
    return False


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        test_labels(work)
        test_scope_of_verdicts(work)
        test_combinations(work)
        test_rejections_and_verifier(work)
        test_frames_and_modality(work)
        test_read_only_and_degradation(work)
    print(f"\nResult: {CHECKS['passed']} passed, {CHECKS['failed']} failed")
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
