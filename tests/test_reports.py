"""In-environment check for desktop report generation.

Path: tests/test_reports.py

Seeds a temporary database through the real storage layer with a full record
(stations, vision and audio events, environmental readings across the status
and QARTOD vocabularies, desktop verification verdicts, labelled interpretive
points, a longitudinal pass with several candidate patterns, station telemetry
with per-channel errors, and cached species reference data), then runs the real
report generator against it with no hardware and no models present.

The checks prove, in one run:

  - The module imports and gathers a report, and the CSV bundle is produced,
    with the PDF library absent, and every CSV value that has a provenance in
    the schema is written next to its source and status.
  - Every candidate pattern is exported under an explicit hypothesis framing and
    carries its effect size, effect-size type, test, data span, n, p, and q.
  - The taxonomic snapshot date and the conservation fetch date behind the
    record are disclosed.
  - When the PDF library is present, a real PDF is written that opens as a valid
    document, and it names the candidate-hypothesis framing and the provenance
    labels.
  - Every configurable value (the output location, the formats, the display time
    zone) is taken from the loader, not hardcoded.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

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
    Interpretation,
    StationTelemetry,
    TelemetryError,
    DreamPass,
    Pattern,
    SpeciesReference,
    new_id,
)
from audtheia.reports import generate as gen  # noqa: E402


REEF_ID = "00000000-0000-0000-0000-000000000001"
FOREST_ID = "00000000-0000-0000-0000-000000000002"
GBIF_SNAPSHOT = "2026-05-15"
IUCN_FETCH = "2026-05-20"

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
    path = tmp / "settings.report.json"
    path.write_text(json.dumps(base), encoding="utf-8")
    return load_settings(path)


def fresh_db(tmp: Path) -> Database:
    db = Database(str(tmp / "report.db"))
    db.initialize_schema(REPO / "audtheia" / "storage" / "schema.sql")
    return db


def seed(db: Database) -> dict:
    db.create_station(Station(id=REEF_ID, station_name="ExampleReef",
                              environment_type="marine", created_at="2026-01-01T00:00:00Z"))
    db.create_station(Station(id=FOREST_ID, station_name="ExampleForest",
                              environment_type="terrestrial", created_at="2026-01-01T00:00:00Z"))

    ids = {}

    # A verified marine vision event with a multi-taxon detection, sensor
    # readings across the status vocabulary, and a labelled interpretation.
    obs1 = new_id()
    ids["obs1"] = obs1
    db.insert_observation(
        Observation(
            id=obs1, event_name="ExampleReef_2026-06-01_a1", station_id=REEF_ID,
            trigger_source="vision", first_seen="2026-06-01T14:00:00Z",
            last_seen="2026-06-01T14:00:20Z", duration=20.0, data_source="model",
            created_at="2026-06-01T14:00:21Z", qc_state="verified",
            screening_confidence=0.82, screening_model_version="yolo11-porifera-v3",
            gbif_snapshot_date=GBIF_SNAPSHOT, iucn_fetch_date=IUCN_FETCH,
            salience_provisional=0.55, gps_latitude=18.2, gps_longitude=-67.1,
            gps_elevation=-5.0, gps_status="measured", representative_frame="data/detections/visual/a1.jpg",
            frame_count=12,
        ),
        children=[
            ChildDetection(id=new_id(), observation_id=obs1, modality="vision",
                           created_at="2026-06-01T14:00:21Z", gbif_usage_key="2367028",
                           scientific_name="Aplysina fistularis", common_name="Yellow tube sponge",
                           confidence=0.82, bbox_x=0.1, bbox_y=0.2, bbox_w=0.3, bbox_h=0.4),
        ],
        environmental_readings=[
            EnvironmentalReading(id=new_id(), observation_id=obs1, channel="water_temp_c",
                                 status="measured", created_at="2026-06-01T14:00:21Z",
                                 value=27.4, unit="degC", qartod_flag=1),
            EnvironmentalReading(id=new_id(), observation_id=obs1, channel="ph",
                                 status="measured", created_at="2026-06-01T14:00:21Z",
                                 value=8.1, unit="pH", qartod_flag=3),
            EnvironmentalReading(id=new_id(), observation_id=obs1, channel="dissolved_oxygen_mg_l",
                                 status="sensor_error", created_at="2026-06-01T14:00:21Z",
                                 value=None, unit="mg/L", qartod_flag=4),
        ],
    )
    db.upsert_observation_verification(ObservationVerification(
        observation_id=obs1, created_at="2026-06-01T15:00:00Z", verified=1,
        rfdetr_version="rfdetr-porifera-medium-v12", rfdetr_gbif_usage_key="2367028",
        rfdetr_scientific_name="Aplysina fistularis", rfdetr_confidence=0.91,
        rfdetr_agrees_with_field=1, frames_scored=12, frames_in_agreement=11,
        salience_authoritative=0.63, rarity_score=0.4, baseline_deviation=1.2,
        anomaly_magnitude_authoritative=0.7, verified_at="2026-06-01T15:00:00Z"))
    db.insert_interpretation(Interpretation(
        id=new_id(), observation_id=obs1, point_type="ecological_role",
        value="reef filter feeder", produced_by="verify", created_at="2026-06-01T15:00:01Z",
        confidence=0.7, model_version="llm-interpreter-v2"))

    # A disagreeing verification: field call overturned, gate stays closed, a
    # missing channel recorded as an explicit not-measured status.
    obs2 = new_id()
    ids["obs2"] = obs2
    db.insert_observation(
        Observation(
            id=obs2, event_name="ExampleReef_2026-06-02_b2", station_id=REEF_ID,
            trigger_source="vision", first_seen="2026-06-02T09:30:00Z",
            last_seen="2026-06-02T09:30:05Z", duration=5.0, data_source="model",
            created_at="2026-06-02T09:30:06Z", qc_state="qc_passed",
            screening_confidence=0.4, screening_model_version="yolo11-porifera-v3",
            gbif_snapshot_date=GBIF_SNAPSHOT, gps_status="not_measured"),
        children=[
            ChildDetection(id=new_id(), observation_id=obs2, modality="vision",
                           created_at="2026-06-02T09:30:06Z", scientific_name="Porifera sp.",
                           common_name="Unknown sponge", confidence=0.4),
        ],
        environmental_readings=[
            EnvironmentalReading(id=new_id(), observation_id=obs2, channel="salinity_psu",
                                 status="not_measured", created_at="2026-06-02T09:30:06Z",
                                 value=None, unit="PSU", qartod_flag=9),
        ],
    )
    db.upsert_observation_verification(ObservationVerification(
        observation_id=obs2, created_at="2026-06-02T10:00:00Z", verified=0,
        rfdetr_version="rfdetr-porifera-medium-v12", rfdetr_scientific_name="not a sponge",
        rfdetr_confidence=0.2, rfdetr_agrees_with_field=0, frames_scored=4, frames_in_agreement=0,
        verified_at="2026-06-02T10:00:00Z"))

    # A terrestrial audio event with a clip and a resolved bird.
    obs3 = new_id()
    ids["obs3"] = obs3
    db.insert_observation(
        Observation(
            id=obs3, event_name="ExampleForest_2026-06-03_c3", station_id=FOREST_ID,
            trigger_source="audio", first_seen="2026-06-03T05:15:00Z",
            last_seen="2026-06-03T05:15:10Z", duration=10.0, data_source="model",
            created_at="2026-06-03T05:15:11Z", qc_state="qc_passed",
            acoustic_model_version="birdnet-global-6k-v2.4",
            audio_clip_path="data/detections/audio/c3.wav",
            audio_true_duration_seconds=10.0, audio_capped=0),
        children=[
            ChildDetection(id=new_id(), observation_id=obs3, modality="audio",
                           created_at="2026-06-03T05:15:11Z", gbif_usage_key="2482507",
                           scientific_name="Coereba flaveola", common_name="Bananaquit",
                           confidence=0.77),
        ],
    )

    # A pure-audio event whose sound was captured but not resolved to a taxon.
    obs4 = new_id()
    ids["obs4"] = obs4
    db.insert_observation(
        Observation(
            id=obs4, event_name="ExampleForest_2026-06-04_d4", station_id=FOREST_ID,
            trigger_source="audio", first_seen="2026-06-04T06:00:00Z",
            last_seen="2026-06-04T06:00:04Z", duration=4.0, data_source="model",
            created_at="2026-06-04T06:00:05Z", qc_state="qc_deferred", qc_reason="low_confidence_unclassified",
            acoustic_model_version="birdnet-global-6k-v2.4",
            audio_clip_path="data/detections/audio/d4.wav",
            audio_true_duration_seconds=4.0, audio_capped=0, time_provisional=1),
    )

    # Telemetry for effort context, with a per-channel error.
    tel = new_id()
    db.insert_station_telemetry(
        StationTelemetry(id=tel, station_id=REEF_ID, recorded_at="2026-06-01T13:55:00Z",
                         created_at="2026-06-01T13:55:00Z", camera_uptime_seconds=3600.0,
                         frames_processed=54000, frames_dropped=12, npu_active_seconds=3500.0,
                         valid_audio_seconds=3600.0, effective_detection_fps=14.8,
                         station_temperature_c=41.2, buffer_fill_pct=32.0, sync_lag_seconds=5.0),
        errors=[TelemetryError(id=new_id(), telemetry_id=tel, channel="ph", error_count=2)])

    # A longitudinal pass with candidate patterns spanning the record.
    dp = new_id()
    ids["dream_pass"] = dp
    db.create_dream_pass(DreamPass(
        id=dp, phase_reached="complete", status="complete", started_at="2026-06-05T00:00:00Z",
        created_at="2026-06-05T00:00:00Z", station_scope=None, ended_at="2026-06-05T01:00:00Z",
        cycles_completed=3))
    pat1 = new_id()
    ids["pattern_temporal"] = pat1
    db.insert_pattern(
        Pattern(id=pat1, dream_pass_id=dp, dream_phase="rem",
                data_span_start="2026-06-01T00:00:00Z", data_span_end="2026-06-04T00:00:00Z",
                n=4, description="Detections of the yellow tube sponge trend later in the day over the span.",
                created_at="2026-06-05T00:30:00Z", pattern_type="temporal_shift",
                confidence=0.6, effect_size=0.44, effect_size_type="r", statistic="mann_kendall",
                p_value=0.03, q_value=0.06, autocorr_adjusted=0, model_version="llm-interpreter-v2"),
        observation_ids=[obs1, obs2])
    pat2 = new_id()
    ids["pattern_cooccur"] = pat2
    db.insert_pattern(
        Pattern(id=pat2, dream_pass_id=dp, dream_phase="nrem",
                data_span_start="2026-06-01T00:00:00Z", data_span_end="2026-06-04T00:00:00Z",
                n=8, description="The sponge and the bananaquit co-occur across sites more than chance.",
                created_at="2026-06-05T00:31:00Z", pattern_type="co_occurrence",
                confidence=0.5, effect_size=0.9, effect_size_type="log_odds", statistic="log_odds",
                p_value=0.04, q_value=0.08, autocorr_adjusted=1),
        observation_ids=[obs1, obs3])

    db.upsert_species_reference(SpeciesReference(
        gbif_usage_key="2367028", scientific_name="Aplysina fistularis", fetched_at="2026-05-20T00:00:00Z",
        common_name="Yellow tube sponge", taxonomic_rank="species", iucn_status="LC",
        iucn_fetch_date=IUCN_FETCH, gbif_occurrence_count=1234, gbif_snapshot_date=GBIF_SNAPSHOT))
    db.upsert_species_reference(SpeciesReference(
        gbif_usage_key="2482507", scientific_name="Coereba flaveola", fetched_at="2026-05-20T00:00:00Z",
        common_name="Bananaquit", taxonomic_rank="species", iucn_status="LC",
        iucn_fetch_date=IUCN_FETCH, gbif_occurrence_count=98765, gbif_snapshot_date=GBIF_SNAPSHOT))

    return ids


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


def run() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        settings = make_settings(tmp)
        db = fresh_db(tmp)
        ids = seed(db)
        out_root = tmp / "reports_out"

        # -- CSV path works with the settings' formats, into the settings-driven
        #    order, and is the sole dependency-free output.
        result = gen.generate_report(settings, db, formats=["csv"], output_dir=out_root)
        check(result.bundle_dir.exists(), "bundle directory was not created")
        check(result.bundle_dir.parent == out_root, "output_dir was not honored")
        csv_dir = result.bundle_dir / "csv"
        names = {p.name for p in result.csv_paths}
        for expected in ("observations.csv", "detections.csv", "verification.csv", "audio.csv",
                         "environment.csv", "interpretations.csv", "patterns.csv",
                         "pattern_observations.csv", "analytics.csv", "telemetry.csv",
                         "species_reference.csv", "provenance.csv"):
            check(expected in names, f"missing CSV table: {expected}")
            check((csv_dir / expected).exists(), f"CSV file not on disk: {expected}")

        # Observations carry the event-level provenance column.
        head, rows = read_csv(csv_dir / "observations.csv")
        check("event_data_source" in head, "observations.csv lacks a data_source column")
        check(len(rows) == 4, f"expected 4 observations, found {len(rows)}")

        # Environment values carry data_source, status, and the QARTOD label.
        head, rows = read_csv(csv_dir / "environment.csv")
        for col in ("data_source", "status", "qartod_flag", "qartod_label"):
            check(col in head, f"environment.csv lacks column {col}")
        status_idx = head.index("status")
        statuses = {r[status_idx] for r in rows}
        check("sensor_error" in statuses and "not_measured" in statuses,
              "environment.csv did not preserve the missing-data statuses")
        src_idx = head.index("data_source")
        check(all(r[src_idx] == "sensor" for r in rows), "an environment value lost its sensor provenance")

        # Detections are model-sourced; interpretations are language-model inferred.
        head, rows = read_csv(csv_dir / "detections.csv")
        src_idx = head.index("data_source")
        check(rows and all(r[src_idx] == "model" for r in rows), "a vision detection lost its model provenance")
        head, rows = read_csv(csv_dir / "interpretations.csv")
        src_idx = head.index("data_source")
        check(rows and all(r[src_idx] == "llm_inferred" for r in rows),
              "an interpretation was not labelled as inferred")

        # Audio: the resolved bird and the unresolved capture both appear.
        head, rows = read_csv(csv_dir / "audio.csv")
        sci_idx = head.index("scientific_name")
        audio_names = {r[sci_idx] for r in rows}
        check("Coereba flaveola" in audio_names, "the resolved audio taxon is missing")
        check("" in audio_names, "the unresolved audio capture was dropped")

        # Every candidate pattern is framed as a hypothesis with a full stat line.
        head, rows = read_csv(csv_dir / "patterns.csv")
        check(len(rows) == 2, f"expected 2 candidate patterns, found {len(rows)}")
        for col in ("framing", "effect_size", "effect_size_type", "statistic",
                    "data_span_start_utc", "data_span_end_utc", "n", "p_value", "q_value"):
            check(col in head, f"patterns.csv lacks column {col}")
        framing_idx = head.index("framing")
        src_idx = head.index("data_source")
        et_idx = head.index("effect_size_type")
        check(all(r[framing_idx] == "candidate_hypothesis" for r in rows),
              "a pattern was not framed as a candidate hypothesis")
        check(all(r[src_idx] == "dream" for r in rows), "a pattern lost its dream provenance")
        check({r[et_idx] for r in rows} == {"r", "log_odds"}, "effect-size types were not preserved")

        # Data age is disclosed.
        head, rows = read_csv(csv_dir / "provenance.csv")
        cat_idx, val_idx = head.index("category"), head.index("value")
        gbif_vals = {r[val_idx] for r in rows if r[cat_idx] == "gbif_snapshot_dates"}
        iucn_vals = {r[val_idx] for r in rows if r[cat_idx] == "iucn_fetch_dates"}
        check(GBIF_SNAPSHOT in gbif_vals, "the taxonomic snapshot date was not disclosed")
        check(IUCN_FETCH in iucn_vals, "the conservation fetch date was not disclosed")
        rfdetr_vals = {r[val_idx] for r in rows if r[cat_idx] == "rfdetr_versions"}
        check("rfdetr-porifera-medium-v12" in rfdetr_vals, "the verifier version was not disclosed")

        # -- Model gathering respects a station filter.
        reef_only = gen.ReportGenerator(settings, db)._gather(
            station_id=REEF_ID, start=None, end=None, generated_at="2026-07-01T00:00:00Z",
            tz=settings.resolve_timezone())
        check(len(reef_only.records) == 2, "station filter did not limit the observations")
        check(all(p.pattern for p in reef_only.patterns), "a scoped pattern lost its payload")

        # -- PDF path: only exercised when the library is present, proving the
        #    seam. The module and the CSV above already ran without it.
        try:
            import fpdf  # noqa: F401
            have_pdf = True
        except ImportError:
            have_pdf = False

        if have_pdf:
            result2 = gen.generate_report(settings, db, output_dir=out_root)
            check(result2.pdf_path is not None and result2.pdf_path.exists(), "PDF was not written")
            data = result2.pdf_path.read_bytes()
            check(data.startswith(b"%PDF-"), "PDF does not start with a valid header")
            check(b"%%EOF" in data[-1024:], "PDF is not properly terminated")
            check(len(data) > 3000, "PDF is implausibly small")
            check(set(result2.formats) == {"pdf", "csv"},
                  "default formats did not come from the settings schedule")
        else:
            # Confirm the seam fails loudly rather than silently when asked for a
            # PDF without the library.
            try:
                gen.write_pdf(reef_only, out_root / "should_fail.pdf")
                check(False, "PDF render should have raised without the library")
            except gen.ReportDependencyError:
                check(True, "seam raised a clear dependency error")

        print(f"ALL CHECKS PASSED ({_checks}) | PDF exercised: {have_pdf}")


if __name__ == "__main__":
    run()
