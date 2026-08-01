"""Reference snapshot dates are stamped independently, never overwritten.

Path: tests/test_snapshot_stamping.py

GBIF and IUCN reference data can be fetched at different times: a record can
carry a GBIF snapshot date from an early fetch while its IUCN date is still
missing because the conservation status could not be reached yet. This proves
that stamping fills each date on its own, only when that date is still unset, and
never overwrites a date already present.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_reports as tr  # noqa: E402


CHECKS = {"passed": 0, "failed": 0}


def check(label: str, condition: bool):
    if condition:
        CHECKS["passed"] += 1
        print(f"  PASS  {label}")
    else:
        CHECKS["failed"] += 1
        print(f"  FAIL  {label}")


def _first_obs_id(db):
    with db.connect() as conn:
        return conn.execute("SELECT id FROM observations LIMIT 1").fetchone()[0]


def _dates(db, oid):
    with db.connect() as conn:
        row = conn.execute(
            "SELECT gbif_snapshot_date, iucn_fetch_date FROM observations WHERE id = ?", (oid,)
        ).fetchone()
    return row[0], row[1]


def test_independent_stamping():
    print("\n[1] Each snapshot date fills on its own, and a set date is never overwritten")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        db = tr.fresh_db(tmp)
        tr.seed(db)
        oid = _first_obs_id(db)

        # Simulate an early GBIF-only fetch: a GBIF date is set, IUCN is not.
        with db.connect() as conn:
            conn.execute(
                "UPDATE observations SET gbif_snapshot_date = '2026-01-01', iucn_fetch_date = NULL WHERE id = ?",
                (oid,),
            )

        # A later fetch offers both dates. Only the missing (IUCN) one should fill.
        changed = db.stamp_observation_snapshot(oid, "2026-09-09", "2026-02-02")
        gbif, iucn = _dates(db, oid)
        check("a row was updated", changed is True)
        check("the existing GBIF date was NOT overwritten", gbif == "2026-01-01")
        check("the missing IUCN date was filled", iucn == "2026-02-02")

        # A record with both dates already set is left untouched.
        changed2 = db.stamp_observation_snapshot(oid, "2030-01-01", "2030-01-01")
        gbif2, iucn2 = _dates(db, oid)
        check("a fully-stamped record reports no change", changed2 is False)
        check("both dates stay exactly as they were", gbif2 == "2026-01-01" and iucn2 == "2026-02-02")


def test_both_from_empty():
    print("\n[2] A record with neither date gets both filled in one call")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        db = tr.fresh_db(tmp)
        tr.seed(db)
        oid = _first_obs_id(db)
        with db.connect() as conn:
            conn.execute(
                "UPDATE observations SET gbif_snapshot_date = NULL, iucn_fetch_date = NULL WHERE id = ?",
                (oid,),
            )
        changed = db.stamp_observation_snapshot(oid, "2026-03-03", "2026-03-04")
        gbif, iucn = _dates(db, oid)
        check("the record was stamped", changed is True and gbif == "2026-03-03" and iucn == "2026-03-04")


def main() -> int:
    test_independent_stamping()
    test_both_from_empty()
    print(f"\n==== snapshot stamping: {CHECKS['passed']} passed, {CHECKS['failed']} failed ====")
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
