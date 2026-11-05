"""Verification for the box tool.

Path: tests/test_training_boxes.py

Drives the box correction endpoints against a real record in a temporary folder.
Proves that an expert can adjust or remove the model's box, draw a box the model
missed with a taxon from the backbone, move that box again without losing its
taxon, and retract any of it; that a box outside the frame, a missing taxon or an
unknown box is refused; that every change is appended with data_source
human_expert while the captured annotation file and the model's detections stay
byte for byte the same; and that a dataset built afterwards uses the expert's
boxes and counts them as expert-drawn.

Uses the standard library, NumPy and Pillow, plus the project itself.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from training_fixture import IMG_H, IMG_W, World, fake_rollup, make_backbone  # noqa: E402
from audtheia.training import dataset as ds  # noqa: E402
from audtheia.training import evaluate as ev  # noqa: E402

CHECKS = {"passed": 0, "failed": 0}


def check(label: str, condition: bool) -> None:
    CHECKS["passed" if condition else "failed"] += 1
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        print("  SKIP  the web framework is not installed in this interpreter")
        return 0
    from audtheia.app import server as srv

    with tempfile.TemporaryDirectory(prefix="audtheia-boxes-") as tmp:
        w = World(Path(tmp))
        make_backbone(Path(w.settings.path("gbif_backbone_path")))
        events = []
        for i in range(5):
            e = w.vision_event([("Aplysina fistularis", 100)], station=f"S{i % 2}", first_seen=f"2026-03-0{i + 1}T10:00:00Z")
            w.confirm(e)
            events.append(e)
        ev0 = events[0]
        ann = ev0["dir"] / "annotations.jsonl"
        ann_before = digest(ann)
        with w.db.connect() as conn:
            children_before = conn.execute("SELECT * FROM child_detections ORDER BY id").fetchall()
        client = TestClient(srv.create_app(w.settings, w.db))
        base = f"/api/detections/{ev0['id']}/frames/0/boxes"

        print("\n[1] Reading a frame's boxes")
        got = client.get(base).json()
        check("the model's box and the frame size are given", got["model_box"] == [10.0, 8.0, 40.0, 30.0]
              and got["frame_width"] == IMG_W and got["frame_height"] == IMG_H and got["corrections"] == [])
        check("an unknown frame is refused", client.get(f"/api/detections/{ev0['id']}/frames/99/boxes").status_code == 404)
        check("an unknown event is refused", client.get("/api/detections/nope/frames/0/boxes").status_code == 404)

        print("\n[2] Refusals")
        post = lambda body, url=base: client.post(url, json=body)  # noqa: E731
        check("an unknown action is refused", post({"action": "move"}).status_code == 422)
        check("a box past the frame edge is refused", post({"action": "adjust", "xyxy": [0, 0, IMG_W + 1, 10]}).status_code == 422)
        check("a box with right before left is refused", post({"action": "adjust", "xyxy": [20, 0, 10, 10]}).status_code == 422)
        check("a box that is not four numbers is refused", post({"action": "adjust", "xyxy": [1, 2, 3]}).status_code == 422)
        check("a box with a value that is not a number is refused", post({"action": "adjust", "xyxy": [1, 2, "x", 9]}).status_code == 422)
        check("an added box without a taxon is refused", post({"action": "add", "xyxy": [1, 1, 9, 9]}).status_code == 422)
        check("an added box with a key not in the backbone is refused",
              post({"action": "add", "xyxy": [1, 1, 9, 9], "gbif_usage_key": "999999"}).status_code == 422)
        check("an unknown box is refused", post({"action": "remove", "box_key": "not-a-box"}).status_code == 422)
        check("nothing was written by a refused request", client.get(base).json()["corrections"] == [])

        print("\n[3] Adjusting, adding and retracting")
        r = post({"action": "adjust", "xyxy": [12, 9, 38, 28], "corrector": "Reviewer"})
        row = r.json()["box"]
        check("the model's box is adjusted as the expert's own record",
              r.status_code == 201 and row["box_key"] == "model" and row["action"] == "adjust"
              and row["data_source"] == "human_expert" and row["corrector"] == "Reviewer"
              and row["frame_width"] == IMG_W)
        r = post({"action": "add", "xyxy": [44, 2, 60, 20], "gbif_usage_key": "101"})
        added = r.json()["box"]
        check("a missed box is added with the accepted taxon of a synonym",
              r.status_code == 201 and added["corrected_gbif_usage_key"] == "100"
              and added["corrected_scientific_name"] == "Aplysina fistularis" and added["box_key"] != "model")
        r = post({"action": "adjust", "box_key": added["box_key"], "xyxy": [45, 3, 61, 21]})
        moved = r.json()["box"]
        check("moving the expert's own box keeps its lineage and its taxon",
              r.status_code == 201 and moved["box_key"] == added["box_key"] and moved["action"] == "add"
              and moved["corrected_gbif_usage_key"] == "100" and moved["x1"] == 45)
        state = {c["box_key"]: c for c in client.get(base).json()["corrections"]}
        check("the newest drawing of each box is the one that stands", state[added["box_key"]]["x1"] == 45
              and state["model"]["action"] == "adjust" and len(state) == 2)

        other = f"/api/detections/{events[1]['id']}/frames/0/boxes"
        check("removing the model's box on another frame is accepted", post({"action": "remove"}, other).status_code == 201)
        check("an expert box from another frame is not this frame's", post({"action": "remove", "box_key": added["box_key"]}, other).status_code == 422)
        r = post({"action": "remove", "box_key": added["box_key"]})
        check("removing an expert box is recorded", r.status_code == 201)
        check("a removed expert box cannot be moved, only drawn again",
              post({"action": "adjust", "box_key": added["box_key"], "xyxy": [1, 1, 9, 9]}).status_code == 422)
        r = post({"action": "add", "xyxy": [44, 2, 60, 20], "gbif_usage_key": "102"})
        second = r.json()["box"]

        print("\n[4] The record is unchanged")
        check("the captured annotation file is byte for byte the same", digest(ann) == ann_before)
        with w.db.connect() as conn:
            check("the model's detections are unchanged",
                  conn.execute("SELECT * FROM child_detections ORDER BY id").fetchall() == children_before)
            n = conn.execute("SELECT COUNT(*) FROM frame_box_correction").fetchone()[0]
        check("every change was appended, none overwritten", n == 6)

        print("\n[5] A dataset uses the expert's boxes")
        built = ds.build_dataset(w.db, w.settings, ds.options_from_request(w.settings, {
            "modality": "vision", "required_criteria": ["expert_confirmed"], "min_examples_per_class": 1,
            "max_frames_per_event": 6}), rollup_lookup=fake_rollup({}))
        manifest = json.loads((Path(built["path"]) / "manifest.json").read_text(encoding="utf-8"))
        frame0 = [i for i in manifest["items"] if i["event_id"] == ev0["id"] and i["frame_index"] == 0]
        sources = sorted(b["box_source"] for i in frame0 for b in i["boxes"])
        check("frame 0 carries the adjusted box and the added box, the removed one is gone",
              sources == ["expert_added", "expert_adjusted"])
        adjusted = [b for i in frame0 for b in i["boxes"] if b["box_source"] == "expert_adjusted"]
        check("the adjusted box has the expert's geometry", adjusted and adjusted[0]["xyxy"] == [12.0, 9.0, 38.0, 28.0])
        added_boxes = [b for i in frame0 for b in i["boxes"] if b["box_source"] == "expert_added"]
        check("the added box has its own taxon", added_boxes and added_boxes[0]["class"] == "Aplysina lacunosa")
        removed = [i for i in manifest["items"] if i["event_id"] == events[1]["id"] and i["frame_index"] == 0]
        check("a frame whose only box was removed is not used as a positive example",
              all(i["role"] != "positive" for i in removed))
        check("frames an expert edited are marked", all(i["box_expert_edited"] for i in frame0))
        check("the dataset counts the expert-drawn boxes", manifest["box_stats"]["expert_adjusted"] >= 1
              and manifest["box_stats"]["expert_added"] >= 1)
        if any(i["split"] == "test" for i in frame0):
            origin = ev.box_origin(Path(built["path"]))
            check("the results caveat counts them as expert-drawn", origin["expert_drawn"] >= 1)

        r = post({"action": "cleared", "box_key": second["box_key"]})
        r = post({"action": "cleared"})
        state = {c["box_key"]: c["action"] for c in r.json()["corrections"]}
        check("retracting returns the model's box and drops the expert box",
              state["model"] == "cleared" and state[second["box_key"]] == "cleared")
        w.settings.raw["node"]["role"] = "desktop"

    text = (REPO / "audtheia" / "app" / "static" / "app.js").read_text(encoding="utf-8")
    tool = text[text.index("// The box tool."):text.index("function openBoxEditor")]
    tool += text[text.index("function openBoxEditor"):text.index("// Model trust, an inferred")]
    check("the box tool's code is ASCII only and app.js has no em dash", tool.isascii() and "\u2014" not in text)
    print(f"\nResult: {CHECKS['passed']} passed, {CHECKS['failed']} failed")
    return 0 if CHECKS["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
