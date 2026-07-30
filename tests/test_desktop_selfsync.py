"""The longitudinal pass consolidates desktop-native records.

Path: tests/test_desktop_selfsync.py

The longitudinal pass reads the arrival stream, which only includes events whose
`synced_at` is set. A field station's events get that stamp when the desktop
imports them, but an event captured directly on a desktop station is
authoritative the moment it is written and was never imported, so without help it
has no `synced_at` and the pass never reaches it. This proves the desktop
self-sync closes that gap: unstamped events are marked as arrived, in capture
order, and the pass then consolidates them and builds the site gist.

Runs on the real storage layer and the real desktop orchestrator, with the
verifier and language model absent (the pass runs its structural half regardless).
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

logging.disable(logging.CRITICAL)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audtheia.config import load_settings  # noqa: E402
from audtheia.app.orchestrator import DesktopStation  # noqa: E402

import test_reports as tr  # noqa: E402


CHECKS = {"passed": 0, "failed": 0}


def check(label: str, condition: bool):
    if condition:
        CHECKS["passed"] += 1
        print(f"  PASS  {label}")
    else:
        CHECKS["failed"] += 1
        print(f"  FAIL  {label}")


def _settings(tmp: Path):
    base = json.loads((REPO / "config" / "settings.json").read_text(encoding="utf-8"))
    base["node"]["role"] = "desktop"
    base["node"]["active_station_id"] = None
    base["paths"]["db_path"] = str((tmp / "report.db").resolve())
    path = tmp / "settings.selfsync.json"
    path.write_text(json.dumps(base), encoding="utf-8")
    return load_settings(path)


def test_desktop_native_records_are_consolidated():
    print("\n[1] Desktop-native events reach the longitudinal pass")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        settings = _settings(tmp)
        db = tr.fresh_db(tmp)
        tr.seed(db)

        # Simulate desktop-native capture: the events exist but were never
        # imported, so none carry an arrival stamp.
        with db.connect() as conn:
            conn.execute("UPDATE observations SET synced_at = NULL")
        check("no event is on the arrival stream before self-sync",
              len(db.list_synced_since(None)) == 0)

        stamped = db.self_sync_local_observations()
        check("self-sync stamps every unstamped event", stamped == 4)
        check("every event is now on the arrival stream", len(db.list_synced_since(None)) == 4)

        # Idempotent: a second self-sync stamps nothing more.
        check("a second self-sync stamps nothing", db.self_sync_local_observations() == 0)

        # The pass, run through the desktop station, now consolidates the record
        # and builds the site gist instead of finding an empty stream.
        result = DesktopStation.build(settings, station_id=tr.REEF_ID).dream_once()
        check("the pass consolidated the desktop-native events", result.observations_consolidated > 0)
        check("the pass built at least one site baseline", len(db.list_site_baselines()) >= 1)


def main() -> int:
    test_desktop_native_records_are_consolidated()
    print(f"\n==== desktop self-sync: {CHECKS['passed']} passed, {CHECKS['failed']} failed ====")
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
