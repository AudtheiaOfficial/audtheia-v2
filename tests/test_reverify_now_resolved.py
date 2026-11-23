"""Re-verifying events whose disagreement came from an unmatched field label.

Path: tests/test_reverify_now_resolved.py

A screening model's own class name, such as tufted-titmouse, is not matched to a
species until someone maps it. An event verified before that mapping is recorded
as disagreeing with the second model even when both name the same bird. These
checks drive the real application against a temporary database and a tiny
taxonomic backbone and prove that:

  - Before the label is mapped, the event is not offered for this re-verify.
  - After the label is mapped to the species the verifier named, the status
    counts the event as one whose labels now match.
  - A genuine disagreement, where the verifier named a different species, is
    never offered.
  - The re-verify scope selects the event, and is refused honestly when no
    verification model is configured, so nothing is written.
  - An unknown scope is refused.

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

from training_fixture import World, make_backbone  # noqa: E402
from audtheia.app import server as srv  # noqa: E402
from audtheia.storage.database import ObservationVerification, utc_now_iso  # noqa: E402

_checks = 0


def check(label: str, condition: bool) -> None:
    global _checks
    if not condition:
        raise AssertionError(label)
    _checks += 1
    print(f"  ok  {label}")


def screened(w: World, label: str, day: int) -> dict:
    """A visual event labelled the way a screening model writes it."""
    ev = w.vision_event([(label, None)], station="ForestA", first_seen=f"2026-03-{day:02d}T10:00:00Z",
                        verified=None)
    with w.db.connect() as conn:
        conn.execute("UPDATE child_detections SET scientific_name = NULL WHERE observation_id = ?", (ev["id"],))
    return ev


def disagreed(w: World, ev: dict, name: str, key: str) -> None:
    """A second-model verdict that named a taxon and was stored as disagreeing."""
    w.db.upsert_observation_verification(ObservationVerification(
        observation_id=ev["id"], created_at=utc_now_iso(), verified=0, rfdetr_version="BioCLIP 2",
        rfdetr_scientific_name=name, rfdetr_gbif_usage_key=key, rfdetr_confidence=0.9,
        rfdetr_agrees_with_field=0, frames_scored=6, frames_in_agreement=0))


def run() -> None:
    from fastapi.testclient import TestClient

    with tempfile.TemporaryDirectory() as td:
        w = World(Path(td))
        make_backbone(Path(w.settings.path("gbif_backbone_path")))
        same_bird = screened(w, "chickadee-slug", 1)
        disagreed(w, same_bird, "Poecile carolinensis", "200")
        other_bird = screened(w, "chickadee-slug", 2)
        disagreed(w, other_bird, "Aplysina fistularis", "100")
        client = TestClient(srv.create_app(w.settings, w.db))

        st = client.get("/api/verify/status").json()
        check("before the label is mapped, no event is offered", st.get("now_resolved") == 0)

        r = client.post("/api/train/identity/labels",
                        json={"label": "chickadee-slug", "action": "map", "gbif_usage_key": "200"})
        check("the label is mapped to the species", r.status_code == 200)

        st = client.get("/api/verify/status").json()
        check("after mapping, the event whose verifier named the same species is counted",
              st.get("now_resolved") == 1)

        r = client.post("/api/verify/rerun", json={"scope": "now_resolved"})
        check("re-verifying those events is refused honestly when no verification model is configured",
              r.status_code == 409 and "no desktop verification model" in r.json().get("detail", ""))
        v = w.db.get_observation_verification(same_bird["id"])
        check("a refused re-verify leaves the stored verdict as it was",
              v["verified"] == 0 and v["rfdetr_agrees_with_field"] == 0)

        r = client.post("/api/verify/rerun", json={"scope": "everything"})
        check("an unknown scope is refused", r.status_code == 422)

    print(f"\nAll {_checks} re-verify checks passed.")


if __name__ == "__main__":
    try:
        run()
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
