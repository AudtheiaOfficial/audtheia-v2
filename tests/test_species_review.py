"""Review species by species on an event that holds more than one.

Path: tests/test_species_review.py

An acoustic event often holds two or more species heard in the same clip. A
single verdict on such an event cannot say which species it is about, so a
"Correct" would read as confirming every species at once. These checks drive the
real application over an in-process request cycle against a temporary database
and prove that:

  - A verdict on the whole event is refused when the event holds more than one
    detection, and still accepted on a single-species event.
  - A verdict naming one detection is accepted, and two species in the same
    event can receive different verdicts.
  - The Audio list carries the verdict that stands on each detection, so the
    page can show which species have been reviewed.
  - Downstream readers see each species on its own: model trust counts one
    record per reviewed species, and the longitudinal pass drops the species
    judged not present while keeping the confirmed one.

Everything is written under a temporary directory; nothing in the repository's
data, database or reports folders is touched.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import test_reports as tr  # noqa: E402  (reuse its real-storage seeding)
import test_server as ts  # noqa: E402  (reuse its temporary settings)
from audtheia.analysis import model_trust  # noqa: E402
from audtheia.app import server as srv  # noqa: E402
from audtheia.storage.database import ChildDetection, Observation, new_id  # noqa: E402

_checks = 0


def check(condition: bool, message: str) -> None:
    global _checks
    if not condition:
        raise AssertionError(message)
    _checks += 1


def seed_two_species(db) -> dict:
    """An acoustic event with two species heard in one clip."""
    obs = new_id()
    titmouse = new_id()
    chickadee = new_id()
    db.insert_observation(
        Observation(
            id=obs, event_name="ExampleForest_2026-06-05_two", station_id=tr.FOREST_ID,
            trigger_source="audio", first_seen="2026-06-05T06:00:00Z",
            last_seen="2026-06-05T06:00:18Z", duration=18.0, data_source="model",
            created_at="2026-06-05T06:00:19Z", qc_state="qc_passed",
            acoustic_model_version="birdnet-global-6k-v2.4",
            audio_clip_path="data/detections/audio/two.wav",
            audio_true_duration_seconds=18.0, audio_capped=0),
        children=[
            ChildDetection(id=titmouse, observation_id=obs, modality="audio",
                           created_at="2026-06-05T06:00:19Z", gbif_usage_key="2487789",
                           scientific_name="Baeolophus bicolor", common_name="Tufted Titmouse",
                           confidence=0.998),
            ChildDetection(id=chickadee, observation_id=obs, modality="audio",
                           created_at="2026-06-05T06:00:19Z", gbif_usage_key="2487906",
                           scientific_name="Poecile carolinensis", common_name="Carolina Chickadee",
                           confidence=0.91),
        ],
    )
    return {"obs": obs, "titmouse": titmouse, "chickadee": chickadee}


def run() -> None:
    from fastapi.testclient import TestClient

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        settings = ts.make_settings(tmp)
        db = tr.fresh_db(tmp)
        ids = tr.seed(db)
        two = seed_two_species(db)
        client = TestClient(srv.create_app(settings, db))
        url = f"/api/observations/{two['obs']}/correct"

        # -- a whole-event verdict is refused on a two-species event ----
        for verdict in ("confirm", "reject"):
            r = client.post(url, json={"verdict": verdict, "modality": "audio"})
            check(r.status_code == 400, f"a whole-event {verdict} on a two-species event was accepted")
            check("each species" in r.json().get("detail", ""),
                  "the refusal did not tell the reviewer to judge each species")
        check(db.corrections_for_observation(two["obs"]) == [],
              "a refused verdict still left a row behind")

        # -- each species takes its own verdict --------------------------
        r = client.post(url, json={"verdict": "confirm", "modality": "audio",
                                   "detection_id": two["titmouse"]})
        check(r.status_code == 201, f"confirming one species failed: {r.text}")
        check(r.json()["correction"]["detection_id"] == two["titmouse"],
              "the confirmation was not stored against the titmouse")
        r = client.post(url, json={"verdict": "reject", "modality": "audio",
                                   "detection_id": two["chickadee"]})
        check(r.status_code == 201, f"marking one species not present failed: {r.text}")

        # A detection from another event is still refused.
        other_child = db.list_child_detections(ids["obs3"])[0]["id"]
        r = client.post(url, json={"verdict": "confirm", "modality": "audio",
                                   "detection_id": other_child})
        check(r.status_code == 400, "a detection from another event was accepted")

        # -- a single-species event still takes a whole-event verdict ----
        r = client.post(f"/api/observations/{ids['obs3']}/correct",
                        json={"verdict": "confirm", "modality": "audio"})
        check(r.status_code == 201, "a whole-event verdict on a single-species event was refused")

        # -- the Audio list carries the verdict on each species ----------
        items = client.get("/api/audio").json()["items"]
        event = next((i for i in items if i["id"] == two["obs"]), None)
        check(event is not None, "the two-species event is missing from the Audio list")
        by_id = {d["id"]: d for d in event["audio_detections"]}
        check(by_id[two["titmouse"]]["correction"]["verdict"] == "confirm",
              "the titmouse did not carry its confirmation")
        check(by_id[two["chickadee"]]["correction"]["verdict"] == "reject",
              "the chickadee did not carry its not-present verdict")
        check(event["correction"] is None,
              "a per-species verdict was reported as a verdict on the whole event")
        single = next(i for i in items if i["id"] == ids["obs3"])
        check(single["correction"]["verdict"] == "confirm",
              "the single-species event lost its whole-event verdict")
        check(single["audio_detections"][0]["correction"] is None,
              "a whole-event verdict was copied onto a detection")

        # -- downstream readers see each species on its own --------------
        records = model_trust.event_review_records(
            db.list_child_detections(two["obs"]),
            db.corrections_for_observation(two["obs"]),
            screening_model_version=None,
            acoustic_model_version="birdnet-global-6k-v2.4",
        )
        verdicts = {r["species_key"]: r["verdict"] for r in records}
        check(verdicts == {"2487789": "confirm", "2487906": "reject"},
              f"model trust did not count each species separately: {verdicts}")
        kept = db.expert_resolved_species(two["obs"])
        check(kept == ["2487789"],
              f"the longitudinal pass should keep only the confirmed species, got {kept}")
        counts = db.correction_counts()
        check(counts["confirm"] == 2 and counts["reject"] == 1,
              f"the review counts did not count each species once: {counts}")

        print(f"ALL CHECKS PASSED ({_checks})")


if __name__ == "__main__":
    run()
