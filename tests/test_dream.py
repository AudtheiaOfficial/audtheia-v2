"""Mocked end-to-end test for the longitudinal dream pass.

Path: tests/test_dream.py

Runs dream.py from end to end with no language-model runtime and no clustering
library present. A scripted narrator stands in for the desktop language model and
a scripted clusterer stands in for an optional novel-grouping backend, so the
module loads and its pass runs with neither library installed. The real
configuration is read through the real loader and every row is read back against
the real schema.

The checks prove, in one run:

  - Consolidation happens before scoring: the permanent baseline is populated,
    an event with a wildly deviant channel scores a high anomaly while a typical
    event in the same period scores a low one, and an immature cell yields no
    anomaly rather than a fabricated one.
  - The generative gate holds: every stored candidate rests only on verified
    events, while an unverified event still contributes to the baseline it helps
    define.
  - The injected regularities are recovered: a rising trend, a co-occurring
    taxon pair, and a pair of channels that move together, each stored as a
    dated, effect-sized candidate hypothesis and never as an established fact.
  - A pass is resumable: asked to pause after a cycle, it stops with a committed
    watermark and, on resume, drains the rest of the backlog with every event
    consumed exactly once.
  - The working set is bounded: with a small cap the generative phase reasons
    over no more than the cap allows, yet still emits candidates.
  - The archive firewall holds: no station-owned row and no verification verdict
    is altered by the pass, which writes only the authoritative salience columns.
  - A verified event with no scorable ingredients keeps a null authoritative
    salience yet still enters the working set, ranked by its field provisional
    value rather than buried.
  - An optional narrator and clusterer are used when present, and the pass runs
    with neither.
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

from audtheia.config import load_settings  # noqa: E402
from audtheia.storage.database import (  # noqa: E402
    Database,
    Station,
    Observation,
    ChildDetection,
    EnvironmentalReading,
    ObservationVerification,
    new_id,
    utc_now_iso,
)
from audtheia.analysis.dream import (  # noqa: E402
    DreamEngine,
    PATTERN_TEMPORAL_SHIFT,
    PATTERN_CO_OCCURRENCE,
    PATTERN_ENVELOPE_CORRELATION,
    PATTERN_NOVEL_CLUSTER,
    STATUS_PAUSED,
    STATUS_COMPLETE,
    _encode_cursor,
)

STATION_ID = "11111111-1111-1111-1111-111111111111"
SPECIES_A = "1001"
SPECIES_B = "1002"
SPECIES_C = "1003"


# ---------------------------------------------------------------------------
# Optional collaborators, both scripted so no real library is needed
# ---------------------------------------------------------------------------


class ScriptedNarrator:
    """Rewrites a candidate description, standing in for the desktop model."""

    def __init__(self):
        self.calls = 0

    def narrate(self, *, pattern_type, template):
        self.calls += 1
        return f"[narrated:{pattern_type}] {template}"


class ScriptedClusterer:
    """Returns one grouping over the working set, standing in for a backend."""

    def __init__(self):
        self.calls = 0

    def cluster(self, exemplars):
        self.calls += 1
        ids = [ex.observation_id for ex in exemplars[:3]]
        if not ids:
            return []
        return [{"id": "g1", "label": "test grouping", "observation_ids": ids}]


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def make_settings(tmp: Path, *, tag: str, budget_overrides=None):
    """Write a desktop copy of the real configuration and load it for real.

    A budget override lets a check exercise small batches or a small working-set
    cap without touching the shipped file.
    """
    base = json.loads((REPO / "config" / "settings.json").read_text(encoding="utf-8"))
    base["node"]["role"] = "desktop"
    base["node"]["active_station_id"] = None
    if budget_overrides:
        base["schedules"]["dream_pass"]["budget"].update(budget_overrides)
    path = tmp / f"settings.dream.{tag}.json"
    path.write_text(json.dumps(base), encoding="utf-8")
    return load_settings(path)


def fresh_db(tmp: Path, tag: str) -> Database:
    db = Database(str(tmp / f"dream.{tag}.db"))
    db.initialize_schema(REPO / "audtheia" / "storage" / "schema.sql")
    db.create_station(
        Station(
            id=STATION_ID,
            station_name=f"reef-{tag}",
            environment_type="marine",
            created_at=utc_now_iso(),
        )
    )
    return db


class _Seq:
    """A monotonic arrival-time generator, so sync order is deterministic."""

    def __init__(self):
        self._i = 0

    def next(self) -> str:
        v = f"2026-07-02T00:{self._i:02d}:00.000000Z"
        self._i += 1
        return v


def add_event(
    db: Database,
    seq: _Seq,
    *,
    month: int,
    day: int,
    temp=None,
    do=None,
    species=(),
    verified: bool,
    confidence=None,
    provisional=None,
    temp_status="measured",
) -> str:
    """Insert one synced event with its readings, children, and verification."""
    oid = new_id()
    first_seen = f"2026-{month:02d}-{day:02d}T12:00:00.000000Z"
    synced_at = seq.next()
    readings = []
    if temp is not None:
        readings.append(
            EnvironmentalReading(
                id=new_id(),
                observation_id=oid,
                channel="water_temp_c",
                value=temp,
                unit="C",
                data_source="sensor",
                status=temp_status,
                qartod_flag=1 if temp_status == "measured" else 4,
                created_at=utc_now_iso(),
            )
        )
    if do is not None:
        readings.append(
            EnvironmentalReading(
                id=new_id(),
                observation_id=oid,
                channel="dissolved_oxygen",
                value=do,
                unit="mg/L",
                data_source="sensor",
                status="measured",
                qartod_flag=1,
                created_at=utc_now_iso(),
            )
        )
    children = [
        ChildDetection(
            id=new_id(),
            observation_id=oid,
            modality="vision",
            gbif_usage_key=key,
            scientific_name=f"sp {key}",
            confidence=0.9,
            data_source="model",
            status="measured",
            created_at=utc_now_iso(),
        )
        for key in species
    ]
    obs = Observation(
        id=oid,
        event_name=f"reef_{first_seen}_{oid[:8]}",
        station_id=STATION_ID,
        trigger_source="vision",
        first_seen=first_seen,
        last_seen=first_seen,
        duration=1.0,
        data_source="model",
        created_at=utc_now_iso(),
        qc_state="verified" if verified else "qc_passed",
        screening_confidence=None,
        salience_provisional=provisional,
        synced_at=synced_at,
    )
    db.insert_observation(obs, children=children, environmental_readings=readings)
    # Every event the desktop has seen carries a verification row. rarity_score
    # is set here so a later check can prove the pass never disturbs it.
    db.upsert_observation_verification(
        ObservationVerification(
            observation_id=oid,
            created_at=utc_now_iso(),
            verified=1 if verified else 0,
            rfdetr_confidence=confidence,
            rarity_score=0.5,
        )
    )
    return oid


def build_dataset(db: Database):
    """Build one longitudinal record with known regularities and anomalies."""
    seq = _Seq()
    ids = {
        "verified": [],
        "unverified": [],
        "anomalous": None,
        "typical": None,
        "null_salience": None,
        "month6_temp_oids": [],
    }

    # Months 1..5: a rising temperature trend, three verified events each, each
    # with the co-occurring taxa and a correlated oxygen channel. These cells
    # stay immature (three members), so they exercise graceful degradation.
    bases = {1: 10.0, 2: 12.0, 3: 14.0, 4: 16.0, 5: 18.0}
    for month, base in bases.items():
        for k in range(3):
            temp = base + 0.3 * k
            oid = add_event(
                db, seq, month=month, day=1 + k, temp=temp, do=temp * 0.8 + 0.2 * k,
                species=(SPECIES_A, SPECIES_B), verified=True, confidence=0.9,
            )
            ids["verified"].append(oid)

    # Month 6: a mature cell. Twelve typical verified events give the baseline
    # enough members to score against, plus one wildly deviant event and one
    # ordinary event singled out for the anomaly checks.
    for k in range(12):
        temp = 19.0 + 0.2 * k
        oid = add_event(
            db, seq, month=6, day=1 + k, temp=temp, do=temp * 0.8,
            species=(SPECIES_A, SPECIES_B), verified=True, confidence=0.9,
        )
        ids["verified"].append(oid)
        ids["month6_temp_oids"].append(oid)
        if k == 5:
            ids["typical"] = oid

    anomalous = add_event(
        db, seq, month=6, day=20, temp=40.0, do=32.0,
        species=(SPECIES_A, SPECIES_B), verified=True, confidence=0.9,
    )
    ids["verified"].append(anomalous)
    ids["month6_temp_oids"].append(anomalous)
    ids["anomalous"] = anomalous

    # Three unverified events in month 6: they feed the baseline but must never
    # back a candidate.
    for k in range(3):
        oid = add_event(
            db, seq, month=6, day=24 + k, temp=20.0 + 0.1 * k, do=16.0,
            species=(SPECIES_A, SPECIES_B), verified=False,
        )
        ids["unverified"].append(oid)
        ids["month6_temp_oids"].append(oid)

    # Co-occurrence separation: a few single-taxon and off-pair events so the
    # association table is not degenerate.
    for _ in range(2):
        oid = add_event(db, seq, month=6, day=15, temp=20.1, do=16.1, species=(SPECIES_A,), verified=True, confidence=0.9)
        ids["verified"].append(oid); ids["month6_temp_oids"].append(oid)
    for _ in range(2):
        oid = add_event(db, seq, month=6, day=16, temp=20.2, do=16.2, species=(SPECIES_B,), verified=True, confidence=0.9)
        ids["verified"].append(oid); ids["month6_temp_oids"].append(oid)
    for _ in range(2):
        oid = add_event(db, seq, month=6, day=17, temp=20.3, do=16.3, species=(SPECIES_C,), verified=True, confidence=0.9)
        ids["verified"].append(oid); ids["month6_temp_oids"].append(oid)

    # A verified event with no scorable ingredient: its one channel is not a
    # measurement, and it carries no confidence, so its authoritative salience
    # must stay null. It carries a field provisional value for the ranking
    # fallback and both taxa so it is still a valid exemplar.
    null_oid = add_event(
        db, seq, month=6, day=28, temp=20.0, do=None,
        species=(SPECIES_A, SPECIES_B), verified=True, confidence=None,
        provisional=0.3, temp_status="not_measured",
    )
    ids["verified"].append(null_oid)
    ids["null_salience"] = null_oid

    return ids


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


PASSED = 0


def check(label: str, ok: bool):
    global PASSED
    if ok:
        PASSED += 1
    else:
        raise AssertionError(f"FAILED: {label}")


def patterns_by_type(db: Database):
    out = {}
    for row in db.list_dream_passes():
        pass
    all_patterns = []
    # There is no list-all-patterns method; gather through the passes' ids.
    with db.connect() as conn:
        all_patterns = [dict(r) for r in conn.execute("SELECT * FROM patterns").fetchall()]
    for p in all_patterns:
        out.setdefault(p["pattern_type"], []).append(p)
    return out, all_patterns


def run():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # ---- main pass: consolidation, scoring, gating, recovery ----
        db = fresh_db(tmp, "main")
        ids = build_dataset(db)
        settings = make_settings(tmp, tag="main")
        narrator = ScriptedNarrator()
        clusterer = ScriptedClusterer()
        engine = DreamEngine(settings=settings, db=db, narrator=narrator, clusterer=clusterer)

        # Snapshot every station-owned row and every verification verdict so the
        # firewall can be checked after the pass.
        with db.connect() as conn:
            obs_before = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM observations").fetchall()}
            verdict_before = {
                r["observation_id"]: (r["verified"], r["rfdetr_confidence"], r["rarity_score"])
                for r in conn.execute("SELECT * FROM observation_verification").fetchall()
            }

        result = engine.run_pass()
        check("pass completes in one run", result.status == STATUS_COMPLETE)
        check("every synced event consolidated once", result.observations_consolidated == len(ids["verified"]) + len(ids["unverified"]))

        # -- consolidation before scoring --
        cells = db.list_site_baselines(station_id=STATION_ID)
        check("baseline gist populated", len(cells) > 0)
        month6_temp = db.get_site_baseline(STATION_ID, "month", "06", "all", "ALL", "water_temp_c")
        check("month 6 temperature cell exists", month6_temp is not None)
        # The unverified events raise the cell's membership: it counts every
        # synced qualifying reading, verified or not.
        check(
            "unverified readings are inside the baseline",
            month6_temp["n"] == len(ids["month6_temp_oids"]),
        )

        v_anom = db.get_observation_verification(ids["anomalous"])
        v_typ = db.get_observation_verification(ids["typical"])
        check("deviant event scores a high anomaly", v_anom["anomaly_magnitude_authoritative"] > 0.9)
        check("typical event scores a low anomaly", v_typ["anomaly_magnitude_authoritative"] < 0.5)
        check("deviant event outranks typical on salience", v_anom["salience_authoritative"] > v_typ["salience_authoritative"])
        check("deviant event records its signed deviation", v_anom["baseline_deviation"] is not None and v_anom["baseline_deviation"] > 0)

        # An immature month-1 cell must yield no anomaly, only confidence.
        month1_verified = ids["verified"][0]
        v_m1 = db.get_observation_verification(month1_verified)
        check("immature cell yields no anomaly", v_m1["anomaly_magnitude_authoritative"] is None)
        check("immature-cell event still scored from confidence", v_m1["salience_authoritative"] is not None)

        # -- the generative gate --
        by_type, all_patterns = patterns_by_type(db)
        check("candidates were emitted", len(all_patterns) > 0)
        check("every candidate is a dream candidate", all(p["data_source"] == "dream" and p["status"] == "candidate" for p in all_patterns))
        check("every candidate carries a data span and support", all(p["data_span_start"] and p["data_span_end"] and p["n"] > 0 for p in all_patterns))

        verified_set = set(ids["verified"])
        unverified_set = set(ids["unverified"])
        linked_ids = set()
        for p in all_patterns:
            linked_ids.update(db.list_pattern_observations(p["id"]))
        check("no candidate rests on an unverified event", linked_ids.isdisjoint(unverified_set))
        check("every supporting event is verified", linked_ids.issubset(verified_set))

        # -- recovered regularities --
        check("a rising trend was found", PATTERN_TEMPORAL_SHIFT in by_type)
        temp_trend = [p for p in by_type.get(PATTERN_TEMPORAL_SHIFT, []) if "water_temp_c" in p["description"]]
        check("the temperature trend is the rising one", bool(temp_trend) and temp_trend[0]["effect_size"] > 0)
        check("the trend carries its test name", bool(temp_trend) and temp_trend[0]["statistic"] == "mann_kendall")
        check("a co-occurrence was found", PATTERN_CO_OCCURRENCE in by_type)
        check("the co-occurrence is positive", any(p["effect_size"] > 0 for p in by_type.get(PATTERN_CO_OCCURRENCE, [])))
        check("an envelope correlation was found", PATTERN_ENVELOPE_CORRELATION in by_type)
        check("the correlation is strong and positive", any(p["effect_size"] > 0.5 for p in by_type.get(PATTERN_ENVELOPE_CORRELATION, [])))

        # -- optional collaborators were used --
        check("the narrator was consulted", narrator.calls > 0)
        check("a narrated description was stored", any(p["description"].startswith("[narrated:") for p in all_patterns))
        check("the clusterer was consulted", clusterer.calls > 0)
        check("a novel grouping was stored", PATTERN_NOVEL_CLUSTER in by_type)

        # -- the null-salience exemplar --
        v_null = db.get_observation_verification(ids["null_salience"])
        check("an unscorable verified event keeps a null salience", v_null["salience_authoritative"] is None)
        working = engine._build_working_set(None)
        check("the null-salience event still enters the working set", any(e.observation_id == ids["null_salience"] for e in working))

        # -- the archive firewall --
        with db.connect() as conn:
            obs_after = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM observations").fetchall()}
            verdict_after = {
                r["observation_id"]: (r["verified"], r["rfdetr_confidence"], r["rarity_score"])
                for r in conn.execute("SELECT * FROM observation_verification").fetchall()
            }
        check("no station-owned observation row was altered", obs_before == obs_after)
        check("no verification verdict, confidence, or rarity was altered", verdict_before == verdict_after)

        # ---- resumability: pause after a cycle, then drain the rest ----
        db2 = fresh_db(tmp, "resume")
        ids2 = build_dataset(db2)
        total_synced = len(ids2["verified"]) + len(ids2["unverified"])
        settings_small = make_settings(tmp, tag="resume", budget_overrides={"epoch_batch_size": 10})
        engine2 = DreamEngine(settings=settings_small, db=db2)

        # Capture the composite cursor of the tenth event, which is where a
        # pause after the first ten-event cycle must land.
        first_ten = db2.list_synced_since(None, limit=10)
        tenth_token = _encode_cursor(first_ten[-1]["synced_at"], first_ten[-1]["id"])

        tripped = {"n": 0}

        def pause_after_first_cycle():
            tripped["n"] += 1
            return True  # request a pause at the first cycle boundary

        paused = engine2.run_pass(should_pause=pause_after_first_cycle)
        check("pass pauses when asked", paused.status == STATUS_PAUSED)
        check("pause lands after exactly one cycle", paused.cycles_completed == 1)
        check("pause commits the composite arrival cursor", paused.checkpoint_watermark == tenth_token)

        engine3 = DreamEngine(settings=settings_small, db=db2)
        resumed = engine3.resume_pass(paused.dream_pass_id)
        check("resume completes the pass", resumed.status == STATUS_COMPLETE)
        finished = db2.get_dream_pass(paused.dream_pass_id)
        check("every event was consumed exactly once across the pause", finished["work_budget_consumed"] == total_synced)
        # Every verified event that had a confidence ingredient must now be scored.
        with db2.connect() as conn:
            scored = conn.execute(
                "SELECT COUNT(*) AS c FROM observation_verification "
                "WHERE verified = 1 AND rfdetr_confidence IS NOT NULL AND salience_authoritative IS NOT NULL"
            ).fetchone()["c"]
            eligible = conn.execute(
                "SELECT COUNT(*) AS c FROM observation_verification "
                "WHERE verified = 1 AND rfdetr_confidence IS NOT NULL"
            ).fetchone()["c"]
        check("resume scored every eligible verified event", scored == eligible and eligible > 0)

        # ---- one sync stamps a shared arrival time: no row may be skipped ----
        # A real sync stamps one identical synced_at across every row it imports.
        # With a small epoch batch, the batch boundary falls inside that shared
        # arrival time, which a naive arrival-only cursor would skip past. This
        # proves the composite cursor consumes every row exactly once regardless.
        db_tie = fresh_db(tmp, "tie")
        ids_tie = build_dataset(db_tie)
        total_tie = len(ids_tie["verified"]) + len(ids_tie["unverified"])
        shared_stamp = "2026-07-02T09:00:00.000000Z"
        with db_tie.connect() as conn:
            conn.execute("UPDATE observations SET synced_at = ?", (shared_stamp,))
        settings_tie = make_settings(tmp, tag="tie", budget_overrides={"epoch_batch_size": 10})
        eng_tie_a = DreamEngine(settings=settings_tie, db=db_tie)
        paused_tie = eng_tie_a.run_pass(should_pause=lambda: True)
        check("shared-arrival pass pauses mid-batch", paused_tie.status == STATUS_PAUSED)
        eng_tie_b = DreamEngine(settings=settings_tie, db=db_tie)
        done_tie = eng_tie_b.resume_pass(paused_tie.dream_pass_id)
        check("shared-arrival pass resumes to completion", done_tie.status == STATUS_COMPLETE)
        fin_tie = db_tie.get_dream_pass(paused_tie.dream_pass_id)
        check("no row is skipped when a whole sync shares one arrival time", fin_tie["work_budget_consumed"] == total_tie)
        with db_tie.connect() as conn:
            scored_tie = conn.execute(
                "SELECT COUNT(*) AS c FROM observation_verification "
                "WHERE verified = 1 AND rfdetr_confidence IS NOT NULL AND salience_authoritative IS NOT NULL"
            ).fetchone()["c"]
            eligible_tie = conn.execute(
                "SELECT COUNT(*) AS c FROM observation_verification "
                "WHERE verified = 1 AND rfdetr_confidence IS NOT NULL"
            ).fetchone()["c"]
        check("every eligible event scored under a shared arrival time", scored_tie == eligible_tie and eligible_tie > 0)

        # ---- cost bounding: a small cap bounds the working set ----
        db3 = fresh_db(tmp, "cap")
        ids3 = build_dataset(db3)
        settings_cap = make_settings(tmp, tag="cap", budget_overrides={"substrate_exemplar_cap": 20})
        engine4 = DreamEngine(settings=settings_cap, db=db3)
        engine4.run_pass()
        capped_working = engine4._build_working_set(None)
        check("the working set is bounded by the cap", len(capped_working) <= 20)
        check("the cap actually bit (more verified than the cap)", len(ids3["verified"]) > 20 and len(capped_working) == 20)
        check("candidates are still emitted under the cap", engine4.patterns_emitted > 0)

        # ---- absent collaborators: the pass runs with neither ----
        db4 = fresh_db(tmp, "bare")
        build_dataset(db4)
        settings_bare = make_settings(tmp, tag="bare")
        bare = DreamEngine(settings=settings_bare, db=db4)  # narrator=None, clusterer=None
        bare_result = bare.run_pass()
        check("a pass runs with no narrator and no clusterer", bare_result.status == STATUS_COMPLETE)
        _, bare_patterns = patterns_by_type(db4)
        check("candidates are emitted with no collaborators", len(bare_patterns) > 0)
        check("no grouping is produced without a clusterer", all(p["pattern_type"] != PATTERN_NOVEL_CLUSTER for p in bare_patterns))

    print(f"All {PASSED} checks passed.")


if __name__ == "__main__":
    run()
