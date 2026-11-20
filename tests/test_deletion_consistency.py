"""Checks that deleting an observation removes it from every derived view.

Path: tests/test_deletion_consistency.py

Builds a miniature record in a temporary repository, with its own settings,
database, data folder, exports folder and a tiny taxonomic backbone, so nothing
here can reach a real installation's data, database, reports or models. Deletes
one observation through the application's own delete route and proves that each
view derived from the record no longer counts it:

  - Analytics events, richness and events per taxon, and the per-gate
    verification split.
  - Species profiles, which are built from the detections listing.
  - Model trust, the per-species accuracy from expert reviews.
  - The Brain audit.
  - The candidates for an outside-labelling export of hard cases.
  - The candidates for a Train tab dataset.
  - A candidate pattern's supporting events, and the pattern itself, which is
    withdrawn with when and why rather than left as a live candidate.
  - The event's frame folder on disk.
  - The site-baseline cell its environmental reading fed, recomputed, and a
    cell left with no reading at all, removed.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from training_fixture import World, make_backbone  # noqa: E402
from audtheia.analysis.dream import DreamEngine  # noqa: E402
from audtheia.storage.database import (  # noqa: E402
    DreamPass, EnvironmentalReading, Pattern, new_id, utc_now_iso,
)

_checks = 0


def check(label: str, condition: bool) -> None:
    global _checks
    if not condition:
        raise AssertionError(label)
    _checks += 1
    print(f"  ok  {label}")


def reading(w: World, oid: str, value: float) -> None:
    w.db.insert_environmental_reading(EnvironmentalReading(
        id=new_id(), observation_id=oid, channel="water_temp_c", status="measured", created_at=utc_now_iso(),
        value=value, unit="C", qartod_flag=1))


def views(client, w: World, ev: dict) -> dict:
    a = client.get("/api/analytics").json()
    dets = client.get("/api/detections?limit=500").json()["items"]
    trust = client.get("/api/brain/model-trust").json()
    audit = client.get("/api/brain/audit").json()
    hard = client.get("/api/brain/retraining/candidates?confidence_below=0.45").json()
    train = client.get("/api/train/candidates/summary?modality=vision&criteria=expert_confirmed").json()
    cells = [c for c in w.db.list_site_baselines(station_id=ev["station_id"]) if c["signal"] == "water_temp_c"]
    return {
        "events": a["total_events"],
        "per_taxon": a["taxon_event_counts"],
        "visual_gate_total": a["verification_by_gate"]["visual"]["total"],
        "listed": [o["id"] for o in dets],
        "reviewed": sum(int(r.get("reviewed") or 0) for r in trust.get("species") or []),
        "audit_events": audit["events"],
        "hard_events": hard["vision"]["events"],
        "train_candidates": train["candidates"],
        "baseline_n": cells[0]["n"] if cells else 0,
        "baseline_max": cells[0]["max_value"] if cells else None,
    }


def run() -> None:
    from fastapi.testclient import TestClient
    from audtheia.app import server as srv

    with tempfile.TemporaryDirectory() as td:
        w = World(Path(td))
        make_backbone(Path(w.settings.path("gbif_backbone_path")))
        keep = []
        for day in (1, 2):
            ev = w.vision_event([("Aplysina fistularis", "100")], first_seen=f"2026-03-{day:02d}T10:00:00Z")
            w.confirm(ev)
            reading(w, ev["id"], 26.0 + day)
            keep.append(ev)
        doomed = w.vision_event([("Aplysina lacunosa", "102")], first_seen="2026-03-03T10:00:00Z")
        w.relabel(doomed, "Aplysina fistularis", gbif=100)
        reading(w, doomed["id"], 31.0)
        # The doomed event is the weakest call, so it is the hard case to export.
        with w.db.connect() as conn:
            conn.execute("UPDATE child_detections SET confidence = 0.2 WHERE observation_id = ?", (doomed["id"],))

        # A second event alone in its own month, so deleting it leaves its cell empty.
        lone = w.vision_event([("Aplysina fistularis", "100")], first_seen="2026-04-10T10:00:00Z")
        w.confirm(lone)
        reading(w, lone["id"], 25.0)

        engine = DreamEngine(settings=w.settings, db=w.db)
        engine.refresh_cells([(e["station_id"], "water_temp_c", "2026-03-01T10:00:00Z") for e in keep + [doomed]]
                             + [(lone["station_id"], "water_temp_c", "2026-04-10T10:00:00Z")])

        now = utc_now_iso()
        w.db.create_dream_pass(DreamPass(id="pass-1", phase_reached="complete", status="complete",
                                         started_at=now, created_at=now, ended_at=now))
        w.db.insert_pattern(Pattern(id="pat-1", dream_pass_id="pass-1", dream_phase="rem",
                                    data_span_start="2026-03-01T10:00:00Z", data_span_end="2026-03-03T10:00:00Z",
                                    n=3, description="three events", created_at=now),
                            observation_ids=[e["id"] for e in keep + [doomed]])

        client = TestClient(srv.create_app(w.settings, w.db))
        april = [c for c in w.db.list_site_baselines(station_id=lone["station_id"])
                 if c["signal"] == "water_temp_c" and c["n"] == 1 and c["max_value"] == 25.0]
        check("the lone event has a baseline cell of its own", len(april) == 1)
        r = client.post("/api/detections/delete", json={"ids": [lone["id"]]})
        check("deleting the lone event answers", r.status_code == 200)
        check("a baseline cell left with no reading is removed",
              not [c for c in w.db.list_site_baselines(station_id=lone["station_id"])
                   if c["signal"] == "water_temp_c" and c["period_key"] == april[0]["period_key"]])

        before = views(client, w, doomed)
        print("\n[1] The record before the deletion counts the event everywhere")
        check("analytics counts three events", before["events"] == 3 and before["visual_gate_total"] == 3)
        check("each species is counted", before["per_taxon"].get("gbif:102") == 1)
        check("the listing behind species profiles includes it", doomed["id"] in before["listed"])
        check("model trust counts its expert review", before["reviewed"] == 3)
        check("the audit counts it", before["audit_events"] == 3)
        check("it is a hard case to export", before["hard_events"] == 1)
        check("it is a Train tab candidate", before["train_candidates"] == 3)
        check("the baseline cell holds its reading", before["baseline_n"] == 3 and before["baseline_max"] == 31.0)
        check("the pattern links to it", len(w.db.list_pattern_observations("pat-1")) == 3)
        check("its frame folder is on disk", doomed["dir"].is_dir())

        r = client.post("/api/detections/delete", json={"ids": [doomed["id"]]})
        body = r.json()
        check("the delete route answers", r.status_code == 200)
        check("the delete route removed the frame folder and refreshed the baseline",
              body.get("frame_folders_removed") == 1 and body.get("baseline_cells_refreshed") == 1)
        check("the delete route reports the withdrawn pattern", body.get("patterns_withdrawn") == 1)

        after = views(client, w, doomed)
        print("\n[2] After the deletion no derived view counts it")
        check("analytics no longer counts it", after["events"] == 2 and after["visual_gate_total"] == 2)
        check("its species is no longer counted", "gbif:102" not in after["per_taxon"])
        check("species profiles no longer list it", doomed["id"] not in after["listed"])
        check("model trust no longer counts its review", after["reviewed"] == 2)
        check("the audit no longer counts it", after["audit_events"] == 2)
        check("it is no longer a hard case to export", after["hard_events"] == 0)
        check("it is no longer a Train tab candidate", after["train_candidates"] == 2)
        check("the baseline cell is recomputed without its reading",
              after["baseline_n"] == 2 and after["baseline_max"] == 28.0)
        check("the pattern no longer links to it", len(w.db.list_pattern_observations("pat-1")) == 2)
        check("its frame folder is gone", not doomed["dir"].exists())
        check("the other events' frame folders are untouched", all(e["dir"].is_dir() for e in keep))
        pat = w.db.get_pattern("pat-1")
        check("the pattern is withdrawn, not deleted, with when and why",
              pat["status"] == "withdrawn" and pat["withdrawn_at"]
              and pat["withdrawn_reason"] == "a supporting event was deleted; status before: candidate")
        listed = client.get("/api/brain/learning").json()["patterns"]
        check("Brain receives it marked withdrawn", [p["status"] for p in listed if p["id"] == "pat-1"] == ["withdrawn"])
        from audtheia.reports import generate as gen
        model = gen.ReportGenerator(w.settings, w.db)._gather(station_id=None, start=None, end=None,
                                                              generated_at=utc_now_iso(), tz=None)
        check("a report does not offer it as a hypothesis but keeps it in the withdrawn list",
              not model.patterns and [p.pattern["id"] for p in model.withdrawn_patterns] == ["pat-1"])

    print(f"\nAll {_checks} deletion consistency checks passed.")


if __name__ == "__main__":
    try:
        run()
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
