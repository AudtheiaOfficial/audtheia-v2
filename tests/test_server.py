"""Local request test for the desktop web backend.

Path: tests/test_server.py

Seeds a temporary database through the real storage layer (reusing the report
test's full record), builds the real FastAPI application, and drives it through
FastAPI's in-process test client, so every endpoint is exercised over a real
request cycle with no network, no hardware, and no models.

The checks prove, in one run:

  - Every listed surface serves from the database: stations, detections and
    their verification, audio, GPS, analytics, brain (models, memory, learning,
    skills), reports, and settings.
  - Provenance survives the wire: a detection keeps its data_source, an
    environmental reading keeps its status and QARTOD flag, an interpretation is
    labelled inferred, and every candidate pattern is returned under an explicit
    hypothesis framing with its supporting events.
  - The two desktop controls work: a running longitudinal pass can be paused and
    resumed, a finished pass refuses both, and an unknown identifier is a clean
    404.
  - A report is produced on request as a background task and then appears in the
    reports listing, and a file inside the reports directory can be fetched while
    a path that tries to escape it is refused.
  - The settings view is read-only and redacts secret-like fields.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import test_reports as tr  # noqa: E402  (reuse its real-storage seeding)
from audtheia.config import load_settings  # noqa: E402
from audtheia.storage.database import DreamPass, Skill, new_id  # noqa: E402
from audtheia.app import server as srv  # noqa: E402

_checks = 0


def check(condition: bool, message: str) -> None:
    global _checks
    if not condition:
        raise AssertionError(message)
    _checks += 1


def make_settings(tmp: Path) -> object:
    base = json.loads((REPO / "config" / "settings.json").read_text(encoding="utf-8"))
    base["node"]["role"] = "desktop"
    base["node"]["active_station_id"] = None
    base["localization"]["local_timezone"] = "America/Puerto_Rico"
    # An absolute reports directory in the temp area, so a generated report never
    # touches the repository.
    base["paths"]["reports_dir"] = str((tmp / "reports_out").resolve())
    # Point the configured database at the same temp file the harness seeds, so
    # endpoints that build a DesktopStation (which opens the configured db) act on
    # the seeded record rather than a separate empty file. tr.fresh_db uses
    # report.db in the same temp directory.
    base["paths"]["db_path"] = str((tmp / "report.db").resolve())
    path = tmp / "settings.server.json"
    path.write_text(json.dumps(base), encoding="utf-8")
    return load_settings(path)


def run() -> None:
    # The module itself must import without the web framework; that is confirmed
    # separately before the framework is installed. Here the framework is present.
    from fastapi.testclient import TestClient

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        settings = make_settings(tmp)
        db = tr.fresh_db(tmp)
        ids = tr.seed(db)

        # A running longitudinal pass, so pause and resume have a live target.
        running_id = new_id()
        db.create_dream_pass(DreamPass(
            id=running_id, phase_reached="nrem_a", status="running",
            started_at="2026-06-06T00:00:00Z", created_at="2026-06-06T00:00:00Z"))
        # A skill, so the brain skills surface returns something real.
        db.upsert_skill(Skill(
            id=new_id(), title="Flag rare sponge", trigger_condition="sponge with low local frequency",
            instruction="raise attention", tier="deterministic_flag",
            created_at="2026-06-01T00:00:00Z", updated_at="2026-06-01T00:00:00Z"))

        app = srv.create_app(settings, db)
        client = TestClient(app)

        # -- meta and stations ------------------------------------------
        r = client.get("/api/health")
        check(r.status_code == 200 and r.json()["status"] == "ok", "health did not report ok")
        r = client.get("/api/stations")
        check(r.status_code == 200 and len(r.json()) == 2, "stations did not return both seeded stations")
        check(client.get(f"/api/stations/{tr.REEF_ID}").status_code == 200, "known station not served")
        check(client.get("/api/stations/does-not-exist").status_code == 404, "unknown station not a 404")

        # -- detections with verification, provenance intact ------------
        r = client.get("/api/detections")
        check(r.status_code == 200, "detections did not return")
        # The list surfaces answer with a page of items alongside the total, so
        # the interface can show "showing 20 of 400" without a second request.
        #
        # This surface is visual events only. Acoustic events are served by the
        # audio endpoint checked below, so a station's four seeded events are
        # split across the two views rather than repeated in both.
        payload = r.json()
        det = payload["items"]
        check(payload["total"] == 2, f"expected 2 visual events, got {payload['total']}")
        check(len(det) == payload["total"], "the page did not carry every counted event")
        check(all(d["trigger_source"] == "vision" for d in det),
              "an acoustic event leaked into the visual detections surface")
        with_vision = [d for d in det if d["vision_detections"]]
        check(with_vision, "no event carried vision detections")
        sample_child = with_vision[0]["vision_detections"][0]
        check(sample_child["data_source"] == "model", "a detection lost its model provenance")
        verified_events = [d for d in det if d.get("verification") and d["verification"].get("verified")]
        check(verified_events, "no verified event surfaced its verification verdict")

        r = client.get(f"/api/detections/{ids['obs1']}")
        check(r.status_code == 200, "detection detail did not return")
        detail = r.json()
        env_statuses = {e["status"] for e in detail["environment"]}
        check("sensor_error" in env_statuses, "an environmental status was lost over the wire")
        check(any(e.get("qartod_flag") is not None for e in detail["environment"]), "QARTOD flag lost")
        check(detail["interpretations"] and detail["interpretations"][0]["data_source"] == "llm_inferred",
              "an interpretation was not labelled inferred")
        check(client.get("/api/detections/nope").status_code == 404, "unknown observation not a 404")

        # -- audio ------------------------------------------------------
        audio = client.get("/api/audio").json()["items"]
        names = {c["scientific_name"] for a in audio for c in a["audio_detections"]}
        check("Coereba flaveola" in names, "resolved audio taxon missing from audio surface")
        check(any(not a["audio_detections"] for a in audio), "unresolved audio capture missing from audio surface")

        # -- gps --------------------------------------------------------
        gps = client.get("/api/gps").json()
        check(any(g["observation_id"] == ids["obs1"] and g["gps_status"] == "measured" for g in gps),
              "a located event was missing or lost its gps status")

        # -- analytics (derived) ----------------------------------------
        a = client.get("/api/analytics").json()
        check(a["provenance"] == "derived", "analytics did not mark itself derived")
        # Three distinct taxa by detection: the yellow tube sponge, the
        # unresolved sponge in the disagreeing event, and the bananaquit.
        check(a["total_events"] == 4 and a["species_richness"] == 3 and a["verified_count"] == 1,
              f"analytics summary was wrong: {a['total_events']}, {a['species_richness']}, {a['verified_count']}")

        # -- brain ------------------------------------------------------
        models = client.get("/api/brain/models").json()
        brain_ids = {s.get("station_id") for s in models.get("stations", [])}
        check("desktop_models" in models and {tr.REEF_ID, tr.FOREST_ID} <= brain_ids,
              "brain models incomplete")
        check(all("deployment" in s for s in models.get("stations", [])),
              "brain models did not classify each station's deployment")
        memory = client.get("/api/brain/memory").json()
        check("site_baselines" in memory, "brain memory did not return the gist container")

        # -- brain audit and retraining surfaces show real, counted data ----
        audit = client.get("/api/brain/audit").json()
        check("verification" in audit and "versions" in audit and audit.get("events") == 4,
              "the audit view did not summarise the seeded record")
        cands = client.get("/api/brain/retraining/candidates").json()
        check("vision" in cands and "acoustic" in cands,
              "the retraining candidate summary did not return both modalities")
        learning = client.get("/api/brain/learning").json()
        check(len(learning["patterns"]) == 2, "brain learning did not return the candidate patterns")
        for p in learning["patterns"]:
            check(p["framing"] == "candidate_hypothesis", "a pattern was not framed as a hypothesis")
            check(p["data_source"] == "dream", "a pattern lost its dream provenance")
            check("supporting_observation_ids" in p and p["supporting_observation_ids"],
                  "a pattern did not carry its supporting events")
        skills = client.get("/api/brain/skills").json()
        check(len(skills) == 1 and skills[0]["tier"] == "deterministic_flag", "brain skills did not serve the skill")

        # -- skills write arc: create, edit, delete, and validation -----
        created = client.post("/api/brain/skills", json={
            "title": "Note sponge near coral",
            "trigger_condition": "a sponge is detected close to coral",
            "instruction": "note potential competition for substrate",
            "tier": "interpretive",
        })
        check(created.status_code == 201, "authoring a skill did not return created")
        new_skill = created.json()
        check(new_skill["tier"] == "interpretive" and new_skill["id"], "the authored skill came back malformed")
        check(len(client.get("/api/brain/skills").json()) == 2, "the authored skill was not stored")

        edited = client.put(f"/api/brain/skills/{new_skill['id']}", json={
            "title": "Note sponge near coral",
            "trigger_condition": "a sponge is detected within one metre of coral",
            "instruction": "note potential competition for substrate",
            "tier": "interpretive",
        })
        check(edited.status_code == 200 and "one metre" in edited.json()["trigger_condition"],
              "editing a skill did not take")
        check(edited.json()["created_at"] == new_skill["created_at"], "editing a skill changed its creation time")

        # A bad type and a missing field are refused before anything is stored.
        check(client.post("/api/brain/skills", json={
            "title": "x", "trigger_condition": "y", "instruction": "z", "tier": "not_a_tier",
        }).status_code == 422, "an invalid skill type was accepted")
        check(client.post("/api/brain/skills", json={
            "title": "", "trigger_condition": "y", "instruction": "z", "tier": "interpretive",
        }).status_code == 422, "an empty skill title was accepted")
        check(client.put("/api/brain/skills/none", json={
            "title": "a", "trigger_condition": "b", "instruction": "c", "tier": "interpretive",
        }).status_code == 404, "editing an unknown skill was not a 404")

        deleted = client.delete(f"/api/brain/skills/{new_skill['id']}")
        check(deleted.status_code == 200 and deleted.json()["status"] == "deleted", "deleting a skill did not take")
        check(len(client.get("/api/brain/skills").json()) == 1, "the deleted skill was not removed")
        check(client.delete("/api/brain/skills/none").status_code == 404, "deleting an unknown skill was not a 404")

        # -- skill execution: condition round-trip, fired counts, apply -----
        # A field skill authored with a structured condition stores it, and the
        # skills surface reports how many events each has flagged.
        first_skills = client.get("/api/brain/skills").json()
        check(all("flagged_events" in s for s in first_skills), "a skill did not report its flagged-event count")
        with_cond = client.post("/api/brain/skills", json={
            "title": "Flag weak screening",
            "trigger_condition": "the screening confidence is low",
            "instruction": "raise a low-confidence flag",
            "tier": "deterministic_flag",
            "condition": {"source": "observation", "field": "screening_confidence", "op": "lt", "value": 0.5},
        })
        check(with_cond.status_code == 201, "authoring a field skill with a condition was refused")
        stored_cond = with_cond.json().get("condition")
        check(stored_cond and json.loads(stored_cond)["field"] == "screening_confidence",
              "the structured condition did not round-trip")
        check(client.post("/api/brain/skills", json={
            "title": "bad", "trigger_condition": "x", "instruction": "y", "tier": "interpretive",
            "condition": {"source": "observation", "field": "screening_confidence", "op": "lt", "value": 0.5},
        }).status_code == 422, "an interpretive skill was wrongly allowed to carry a condition")
        applied = client.post("/api/brain/skills/apply")
        check(applied.status_code == 200 and "scanned" in applied.json() and "flags" in applied.json(),
              "applying skills to existing records did not report a result")

        # -- language model management: list, select, and guards --------
        llm0 = client.get("/api/brain/llm").json()
        check(all(k in llm0 for k in ("available", "active", "runtime_available", "directory",
                                      "status", "status_message", "remedy")),
              "the language model view was missing fields")
        # With no model configured yet, the status is honest about which step is
        # missing and carries an actionable remedy rather than a bare flag.
        check(llm0["status"] in ("runtime_missing", "no_model") and llm0["remedy"],
              "the language model status was not actionable when no model is present")

        # Point the language model folder at a temp directory holding two models,
        # so listing and selection act on something real.
        llm_dir = tmp / "llm_models"
        llm_dir.mkdir()
        (llm_dir / "alpha.gguf").write_bytes(b"a")
        (llm_dir / "beta.gguf").write_bytes(b"b")
        settings.raw["desktop_models"]["llm"]["path"] = str(llm_dir)

        listed = client.get("/api/brain/llm").json()
        check([m["name"] for m in listed["available"]] == ["alpha.gguf", "beta.gguf"],
              "installed models did not list in name order")
        check(listed["active"] == "alpha.gguf", "the first model was not reported active")

        chosen = client.post("/api/brain/llm/select", json={"name": "beta.gguf"})
        check(chosen.status_code == 200 and chosen.json()["active"] == "beta.gguf", "selecting a model did not take")
        check(settings.raw["desktop_models"]["llm"]["path"].endswith("beta.gguf"),
              "the selected model was not written into the configuration")

        check(client.post("/api/brain/llm/select", json={"name": ""}).status_code == 422,
              "an empty model name was accepted")
        check(client.post("/api/brain/llm/select", json={"name": "../escape.gguf"}).status_code == 400,
              "a model path escaping the folder was accepted")
        check(client.post("/api/brain/llm/select", json={"name": "missing.gguf"}).status_code == 404,
              "an unknown model name was accepted")

        # -- dream status and controls ----------------------------------
        status = client.get("/api/dream/status").json()
        check(status["active"] and status["active"]["id"] == running_id, "running pass not reported active")
        paused = client.post(f"/api/dream/{running_id}/pause")
        check(paused.status_code == 200 and paused.json()["status"] == "paused", "pause did not take")
        resumed = client.post(f"/api/dream/{running_id}/resume")
        check(resumed.status_code == 200 and resumed.json()["status"] == "running", "resume did not take")
        check(client.post(f"/api/dream/{ids['dream_pass']}/pause").status_code == 409,
              "a completed pass was allowed to be paused")
        check(client.post("/api/dream/none/pause").status_code == 404, "unknown pass not a 404")

        # A run-now longitudinal pass reports its counts, and when it emits no
        # pattern it carries a counted, plain-language reason rather than a blank.
        # No pass may be running when one is started, so the live pass is paused first.
        client.post(f"/api/dream/{running_id}/pause")
        ran = client.post("/api/dream/run")
        check(ran.status_code == 200, "the longitudinal pass did not run")
        rp = ran.json()
        check(all(k in rp for k in ("observations_consolidated", "salience_scored",
                                    "patterns_emitted", "narration_available", "diagnostics")),
              "the pass result was missing its reporting fields")
        if rp["patterns_emitted"] == 0:
            check(rp["diagnostics"] and rp["diagnostics"].get("reason"),
                  "a zero-pattern pass did not explain why")

        # -- reports: list empty, generate in background, list again ----
        first = client.get("/api/reports").json()
        check(first["bundles"] == [], "reports listing was not empty at the start")
        made = client.post("/api/reports", json={"formats": ["csv"]})
        check(made.status_code == 202 and made.json()["status"] == "scheduled", "report was not scheduled")
        after = client.get("/api/reports").json()
        check(len(after["bundles"]) == 1, "the generated report bundle did not appear")
        bundle = after["bundles"][0]
        csv_files = [f for f in bundle["files"] if f.endswith(".csv")]
        check(csv_files, "the report bundle held no CSV files")

        # -- report file download, with a traversal guard ---------------
        # Listing paths are relative to the bundle, so a link is the bundle name
        # plus the file path, exactly as the interface builds it.
        one = bundle["name"] + "/" + csv_files[0]
        direct = client.get("/api/reports/file", params={"path": one})
        check(direct.status_code == 200, "a real report file could not be fetched")
        check(client.get("/api/reports/file", params={"path": "../../etc/passwd"}).status_code == 400,
              "a path escaping the reports directory was not refused")

        # -- report deletion -------------------------------------------
        check(client.delete("/api/reports/no-such-bundle").status_code == 404,
              "deleting an unknown report bundle was not a 404")
        gone = client.delete(f"/api/reports/{bundle['name']}")
        check(gone.status_code == 200 and gone.json()["status"] == "deleted", "deleting a report did not take")
        check(client.get("/api/reports").json()["bundles"] == [], "the deleted report bundle still appears")

        # -- target-species editor and detected-species fetch scope -----
        reef = tr.REEF_ID
        added = client.post(f"/api/settings/stations/{reef}/target-species", json={"name": "Aplysina fistularis"})
        check(added.status_code == 201, "adding a target species failed")
        rstatus = client.get("/api/species/reference/status").json()
        check("Aplysina fistularis" in rstatus.get("target_species", []),
              "the added target species did not reach the fetch scope")
        check(isinstance(rstatus.get("detected_species"), list) and rstatus["detected_species"],
              "reference status did not report the species detected in the record")
        check(client.post(f"/api/settings/stations/{reef}/target-species", json={"name": "Aplysina fistularis"}).status_code == 201,
              "re-adding an existing target species was rejected")
        check(client.post(f"/api/settings/stations/{reef}/target-species", json={"name": "   "}).status_code == 422,
              "an empty target species was accepted")
        rem = client.delete(f"/api/settings/stations/{reef}/target-species/Aplysina%20fistularis")
        check(rem.status_code == 200, "removing a target species failed")
        check("Aplysina fistularis" not in client.get("/api/species/reference/status").json().get("target_species", []),
              "the removed target species is still in scope")
        check(client.delete(f"/api/settings/stations/{reef}/target-species/Nope").status_code == 404,
              "removing an unknown target species was not a 404")
        # Pushing config to a Pi that was never connected is refused with a clear
        # message rather than attempting an impossible connection.
        check(client.post(f"/api/stations/{reef}/push-config").status_code == 409,
              "pushing config to an unconnected station was not refused")

        # -- storage: configurable data directory and archive/reclaim ---
        newdir = str((tmp / "AudtheiaData").resolve()).replace("\\", "/")
        dd = client.post("/api/settings/data-directory", json={"path": newdir})
        check(dd.status_code == 200, "setting the data directory failed")
        cfg = client.get("/api/settings").json()["config"]["paths"]
        check(cfg["data_dir"] == newdir and cfg["detections_visual_dir"] == newdir + "/detections/visual",
              "the data directory and its subfolders were not set together")
        check(client.post("/api/settings/data-directory", json={"path": "  "}).status_code == 422,
              "an empty data directory was accepted")
        arch = client.post("/api/storage/archive", json={"target_dir": str((tmp / "arch").resolve()), "reclaim": False})
        check(arch.status_code == 200 and "archived" in arch.json(), "the archive action did not report a result")
        check(client.post("/api/storage/archive", json={"target_dir": ""}).status_code == 422,
              "archiving without a destination was accepted")

        # -- settings: read-only, secrets redacted ----------------------
        s = client.get("/api/settings").json()
        check(s["config"]["node"]["role"] == "desktop", "settings view did not return the config")
        redacted = srv._redact({"password": "hunter2", "nested": {"api_token": "abc"}, "keep": 1})
        check(redacted["password"] == "***redacted***" and redacted["keep"] == 1,
              "redaction did not blank a secret-like field")

        print(f"ALL CHECKS PASSED ({_checks})")


if __name__ == "__main__":
    run()
