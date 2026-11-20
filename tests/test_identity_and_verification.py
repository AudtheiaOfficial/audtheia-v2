"""Checks for one identity per species and for honest verification wording.

Path: tests/test_identity_and_verification.py

Builds a miniature record in a temporary repository, with its own settings,
database, data folder, exports folder and a tiny taxonomic backbone, so nothing
here can reach a real installation's data, database, reports or models. Proves,
in one run:

  - Verification is classified per gate. An event with frames is checked by a
    second model; a pure sound event passes only an acoustic confidence gate. A
    verdict recorded while no verification model was configured reads as not
    checked, never as a model that looked and recognised nothing, and the two
    gates are counted apart, never pooled.
  - The wording names the gate, the floor actually in force, the verifier version,
    and, when the verifier's label set lacks the field taxon, says so.
  - With no verification model configured the engine writes no verdict, and
    re-verifying is refused. With one configured, re-verifying replaces only the
    desktop verdict and leaves the station's observation row exactly as it was.
  - A field label and a verifier label that name the same species agree, even
    when one is a synonym and the other the accepted name.
  - Labels naming one species resolve to one identity, the raw label is kept, and
    an unresolved label is flagged and never merged by guessing. A mapping made
    through the shared label endpoint after the fact is honoured in the dataset
    plan, in the interface, and in a report, without changing the record.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from training_fixture import World, fake_rollup, make_backbone  # noqa: E402
from audtheia.analysis import taxa, verdicts  # noqa: E402
from audtheia.analysis.verify import FrameDetection, VerifyEngine  # noqa: E402
from audtheia.app.orchestrator import NullInterpreter, NullVerifier  # noqa: E402
from audtheia.storage.database import (  # noqa: E402
    ChildDetection, Observation, ObservationVerification, new_id, utc_now_iso,
)
from audtheia.training import dataset as ds  # noqa: E402

_checks = 0


def check(label: str, condition: bool) -> None:
    global _checks
    if not condition:
        raise AssertionError(label)
    _checks += 1
    print(f"  ok  {label}")


class FixedVerifier:
    """A configured verifier that resolves the same taxon on every frame."""

    configured = True

    def __init__(self, detection: FrameDetection, version: str = "verifier-test-1"):
        self._detection = detection
        self.version = version

    def verify_frames(self, frame_paths):
        return [self._detection for _ in frame_paths]


def world(root: Path) -> World:
    w = World(root)
    make_backbone(Path(w.settings.path("gbif_backbone_path")))
    return w


def vision(w: World, label: str, *, station="ReefA", day=1, key=None, confirm=True) -> dict:
    """A visual event whose label sits where a screening model writes it: the common name."""
    ev = w.vision_event([(label, key)], station=station, first_seen=f"2026-03-{day:02d}T10:00:00Z",
                        verified=None)
    with w.db.connect() as conn:
        conn.execute("UPDATE child_detections SET scientific_name = NULL WHERE observation_id = ?", (ev["id"],))
    if confirm:
        w.confirm(ev)
    return ev


def stand_in(w: World, ev: dict, frames: int = 6) -> None:
    """The verdict shape the stand-in verifier wrote when no model was configured."""
    w.db.upsert_observation_verification(ObservationVerification(
        observation_id=ev["id"], created_at=utc_now_iso(), verified=0, frames_scored=frames,
        frames_in_agreement=0))


def audio(w: World, label: str, confidence: float, *, cleared: bool, day=1) -> dict:
    """A pure sound event with a BirdNET-style "Scientific_Common" label."""
    sid = w.station("ReefA")
    oid = new_id()
    now = utc_now_iso()
    w.db.insert_observation(Observation(
        id=oid, event_name=f"au{day:02d}_{oid[:6]}", station_id=sid, trigger_source="audio",
        first_seen=f"2026-03-{day:02d}T11:00:00Z", last_seen=f"2026-03-{day:02d}T11:00:00Z", duration=3.0,
        data_source="model", created_at=now, qc_state="qc_passed"),
        children=[ChildDetection(id=new_id(), observation_id=oid, modality="audio", created_at=now,
                                 common_name=label, confidence=confidence)])
    w.db.upsert_observation_verification(ObservationVerification(
        observation_id=oid, created_at=now, verified=1 if cleared else 0, frames_scored=0, frames_in_agreement=0,
        verified_at=now if cleared else None))
    return {"id": oid}


def pair(w: World, oid: str) -> tuple:
    return w.db.get_observation(oid), w.db.get_observation_verification(oid)


def test_gates(root: Path) -> None:
    print("\n[1] Verification is classified and counted per gate")
    w = world(root / "g")
    a = vision(w, "Aplysina fistularis", day=1)
    stand_in(w, a)
    b = vision(w, "Aplysina fistularis", day=2)
    c = audio(w, "Poecile carolinensis_Carolina Chickadee", 0.998, cleared=True, day=3)
    d = audio(w, "Poecile carolinensis_Carolina Chickadee", 0.40, cleared=False, day=4)

    check("a stand-in verdict reads as not checked by a second model",
          verdicts.classify(*pair(w, a["id"]))["state"] == verdicts.NOT_RUN)
    check("an event with frames and no verdict is pending", verdicts.classify(*pair(w, b["id"]))["state"] == verdicts.PENDING)
    check("a cleared sound event is cleared by the acoustic gate",
          verdicts.classify(*pair(w, c["id"])) == {"gate": verdicts.GATE_ACOUSTIC, "state": verdicts.CLEARED, "reason": None})
    split = verdicts.split_counts([pair(w, x["id"]) for x in (a, b, c, d)])
    check("the two gates are counted apart and never pooled",
          split["visual"] == {"total": 2, "verified": 0, "not_verified": 0, "not_run": 1, "pending": 1}
          and split["audio"] == {"total": 2, "cleared": 1, "not_cleared": 1, "pending": 0})
    sentence = verdicts.summary_sentence(split)
    check("the summary sentence states both gates separately",
          "0 of 2 visual events by the two-model check" in sentence
          and "1 of 2 audio events by the acoustic confidence gate" in sentence)

    configured = w.settings.thresholds_config()["verification"]
    floors = verdicts.floors(w.settings)
    check("the floors come from the configuration in force",
          floors["clear_confidence"] == float(configured["clear_confidence"])
          and floors["acoustic_clear_confidence"] == float(configured.get("acoustic_clear_confidence", 0.7)))
    obs, v = pair(w, c["id"])
    text = verdicts.describe(obs, v, children=w.db.list_child_detections(c["id"]),
                             floors_in_force={"clear_confidence": 0.5, "acoustic_clear_confidence": 0.65})
    check("an acoustic clearance names the gate, the strongest call and the floor given",
          text == "cleared by the acoustic confidence gate: strongest call 0.998 against a floor of 0.65; "
                  "not cross-checked by a second model")
    obs, v = pair(w, a["id"])
    check("a stand-in verdict is described as no model having checked a frame",
          "no frame was checked by a second model" in verdicts.describe(obs, v))

    # A real verifier that scored frames and recognised none, and whose label set
    # does not include the field taxon.
    e = vision(w, "Aplysina fistularis", day=5)
    w.db.upsert_observation_verification(ObservationVerification(
        observation_id=e["id"], created_at=utc_now_iso(), verified=0, rfdetr_version="cv-v1",
        frames_scored=32, frames_in_agreement=0))
    with taxa.TaxonBook.for_settings(w.settings, w.db) as book:
        obs, v = pair(w, e["id"])
        text = verdicts.describe(obs, v, children=w.db.list_child_detections(e["id"]), verifier_configured=True,
                                 verifier_version="cv-v1", verifier_identities={"gbif:200"}, book=book)
    check("a verifier that recognised nothing is said to have scored N frames and recognised none, by version",
          text.startswith("not verified: cv-v1 scored 32 frames and recognized none of them"))
    check("a label set without the field taxon is stated plainly",
          "its label set does not include the field call's taxon, Aplysina fistularis" in text)

    # A verdict stored with no agreement, from the screening model itself, over a
    # record whose field call names the same taxon.
    f = vision(w, "blue-jay", day=6)
    with w.db.connect() as conn:
        conn.execute("UPDATE observations SET screening_model_version = 'cv-v1' WHERE id = ?", (f["id"],))
    w.db.upsert_observation_verification(ObservationVerification(
        observation_id=f["id"], created_at=utc_now_iso(), verified=0, rfdetr_version="cv-v1",
        rfdetr_scientific_name="blue-jay", rfdetr_confidence=0.33, frames_scored=11, frames_in_agreement=11))
    with taxa.TaxonBook.for_settings(w.settings, w.db) as book:
        obs, v = pair(w, f["id"])
        text = verdicts.describe(obs, v, children=w.db.list_child_detections(f["id"]), book=book)
    check("with no stored agreement, the field call is compared when read and said to be",
          "the same taxon as the field call (compared when read; no agreement was stored with the verdict)" in text
          and "below the floor of 0.50" in text)
    check("a verifier that is the screening model is not called an independent check",
          text.endswith("the verifier is the same model that screened the event, so this is not an independent check"))


def test_engine(root: Path) -> None:
    print("\n[2] The engine writes no verdict without a model, and re-verifying never touches the station row")
    w = world(root / "e")
    ident = taxa.identity_function(w.settings)
    ev = vision(w, "Verongia fistularis", day=1)
    engine = VerifyEngine(settings=w.settings, db=w.db, verifier=NullVerifier(), interpreter=NullInterpreter(),
                          identify=ident)
    result = engine.process(ev["id"])
    check("with no verification model configured, no verdict is written",
          result.outcome == "skipped" and w.db.get_observation_verification(ev["id"]) is None)
    check("re-verifying with no verification model configured runs nothing",
          engine.reverify([ev["id"]])["not_configured"] is True)

    stand_in(w, ev)
    before = dict(w.db.get_observation(ev["id"]))
    children_before = [dict(c) for c in w.db.list_child_detections(ev["id"])]
    real = VerifyEngine(settings=w.settings, db=w.db, interpreter=NullInterpreter(), identify=ident,
                        verifier=FixedVerifier(FrameDetection(gbif_usage_key="100",
                                                              scientific_name="Aplysina fistularis",
                                                              confidence=0.91)))
    counts = real.reverify([ev["id"]])
    v = w.db.get_observation_verification(ev["id"])
    check("re-verifying replaces the stand-in verdict with a stamped one", counts["verified"] == 1
          and v["rfdetr_version"] == "verifier-test-1" and v["verified"] == 1)
    check("the synonym field label and the accepted verifier label agree", v["rfdetr_agrees_with_field"] == 1)
    check("the station's observation row is unchanged", dict(w.db.get_observation(ev["id"])) == before)
    check("the station's detections are unchanged",
          [dict(c) for c in w.db.list_child_detections(ev["id"])] == children_before)


def test_identity(root: Path) -> None:
    print("\n[3] One identity per species, raw labels kept, unresolved labels flagged")
    from fastapi.testclient import TestClient
    from audtheia.app import server as srv
    from audtheia.reports import generate as gen

    w = world(root / "i")
    for day in (1, 2):
        vision(w, "Aplysina fistularis", day=day)
    vision(w, "Verongia fistularis", day=3)
    for day in (4, 5):
        vision(w, "aplysina-slug", day=day)
    vision(w, "mystery-sponge", day=6)
    audio(w, "Poecile carolinensis_Carolina Chickadee", 0.95, cleared=True, day=7)

    with taxa.TaxonBook.for_settings(w.settings, w.db) as book:
        acc = book.identify_label("Aplysina fistularis")
        syn = book.identify_label("Verongia fistularis")
        slug = book.identify_label("aplysina-slug")
        bird = book.identify_label("Poecile carolinensis_Carolina Chickadee")
        check("a synonym and its accepted name are one identity", acc["key"] == syn["key"] == "gbif:100")
        check("a BirdNET label resolves to its species with the common name shown",
              book.display(bird) == "Poecile carolinensis (Carolina Chickadee)")
        check("a label no rule matches is flagged unresolved and is not merged by guessing",
              slug["unresolved"] is True and slug["key"] != "gbif:100")

    client = TestClient(srv.create_app(w.settings, w.db))
    a = client.get("/api/analytics").json()
    counts = a["taxon_event_counts"]
    check("analytics counts the synonym with its accepted species", counts.get("gbif:100") == 3)
    check("analytics keeps the unresolved label as its own flagged entry",
          any(v.get("unresolved") for k, v in a["taxa"].items() if k != "gbif:100"))
    check("analytics reports verification per gate and no pooled count",
          "verification_by_gate" in a and "verified_count" not in a)

    r = client.post("/api/train/identity/labels",
                    json={"label": "aplysina-slug", "action": "map", "gbif_usage_key": "100"})
    check("the shared label endpoint accepts the mapping", r.status_code == 200)
    a = client.get("/api/analytics").json()
    check("after mapping, analytics counts the label with its species", a["taxon_event_counts"].get("gbif:100") == 5)
    dets = client.get("/api/detections?limit=50").json()
    rows = dets["items"] if isinstance(dets, dict) else dets
    slug_children = [d for o in rows for d in o["vision_detections"] if d.get("common_name") == "aplysina-slug"]
    check("the interface shows the mapped identity while the label as written stays on the record",
          slug_children and all(d["identity"]["key"] == "gbif:100" and d["identity"]["method"] == "label_map"
                                for d in slug_children))

    req = {"modality": "vision", "required_criteria": ["expert_confirmed"], "min_examples_per_class": 1}
    plan = ds.summarize_plan(ds.plan_dataset(w.db, w.settings, ds.options_from_request(w.settings, req),
                                             rollup_lookup=fake_rollup({})))
    keys = sorted(c["class_key"] for c in plan["classes"])
    check("a mapping made after the fact is honoured at the next dataset build",
          "gbif:100" in keys and not any("aplysina-slug" in k for k in keys))

    out = Path(root / "i" / "repo" / "reports")
    result = gen.generate_report(w.settings, w.db, formats=["csv"], output_dir=out)
    csv_dir = Path(next(p for p in result.csv_paths if Path(p).name == "verification.csv")).parent
    with open(csv_dir / "verification.csv", encoding="utf-8") as fh:
        gates = {row["gate"] for row in csv.DictReader(fh)}
    check("the report's verification table names each event's gate",
          gates == {verdicts.GATE_TWO_MODEL, verdicts.GATE_ACOUSTIC})
    det_csv = csv_dir / "detections.csv"
    with open(det_csv, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    slug_rows = [r for r in rows if "aplysina-slug" in json.dumps(r)]
    check("the report keeps the raw label and adds the accepted identity beside it",
          slug_rows and all(r["identity_key"] == "gbif:100" and r["identity_scientific_name"] == "Aplysina fistularis"
                            for r in slug_rows))

    r = client.post("/api/verify/rerun", json={"scope": "not_run"})
    check("re-verify with nothing to re-verify reports that plainly", r.status_code == 200 and r.json()["events"] == 0)
    first_visual = next(o for o in w.db.list_observations() if o["trigger_source"] == "vision")
    stand_in(w, first_visual)
    r = client.post("/api/verify/rerun", json={"scope": "not_run"})
    check("re-verify is refused while no verification model is configured", r.status_code == 409)
    st = client.get("/api/verify/status").json()
    check("the status says no verification model is configured and counts what no model checked",
          st["verifier_configured"] is False and st["visual"]["not_run"] == 1)


def run() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        test_gates(root)
        test_engine(root)
        test_identity(root)
    print(f"\nAll {_checks} identity and verification checks passed.")


if __name__ == "__main__":
    try:
        run()
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
