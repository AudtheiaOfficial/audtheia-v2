#!/usr/bin/env python3
"""Checks for the desktop language-model adapter.

Path: tests/test_gguf_llm.py

These prove that the GGUF interpreter turns a model response into exactly the
interpretive points the verification engine accepts, that it refuses anything
outside the allowed interpretive vocabulary, that a malformed or empty response
yields no points rather than an error, and that the narrator rephrases a
description without ever being load-bearing. The model is a scripted completer
returning fixed text, so the adapter's own prompt-building and defensive parsing
run for real while only the model runtime is a stand-in. The final interpretation
check runs the real VerifyEngine interpret seam over the adapter and confirms
every point it produces passes the engine's own validation.

Run directly: python tests/test_gguf_llm.py
It needs no third-party package.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from audtheia.config import load_settings
from audtheia.storage.database import Database
from audtheia.analysis.verify import (
    VerifyEngine,
    VerificationContext,
    VerificationVerdict,
    InterpretationPoint,
    INTERPRETATION_POINT_TYPES,
    DATA_SOURCE_LLM_INFERRED,
)
from audtheia.inference.gguf_llm import (
    GGUFInterpreter,
    GGUFNarrator,
    INTERPRETER_POINT_TYPES,
    _extract_json_array,
    _resolve_model_file,
    LLMError,
)


class ScriptedCompleter:
    """A stand-in for a loaded GGUF model: returns fixed text for any prompt.

    It records every prompt it is given, so a check can confirm the adapter built
    the prompt it intended, and carries a version like the real completer so
    version stamping is exercised.
    """

    def __init__(self, text: str, *, version: str = "gguf-test"):
        self._text = text
        self.version = version
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, max_tokens: int = 256, temperature: float = 0.2) -> str:
        self.prompts.append(prompt)
        return self._text


def _context(resolved_name="Xestospongia muta", companions=None, readings=None):
    verdict = VerificationVerdict(resolved_scientific_name=resolved_name)
    children = []
    for name in companions or []:
        children.append({"scientific_name": name, "modality": "vision", "confidence": 0.8})
    return VerificationContext(
        observation={"id": "obs-1", "station_id": "st-1"},
        child_detections=children,
        environmental_readings=readings or [],
        verdict=verdict,
        field_gbif_usage_key=None,
        field_scientific_name=resolved_name,
    )


def test_interpreter_parses_accepted_points():
    response = json.dumps(
        [
            {"point_type": "ecological_role", "value": "A large barrel sponge that filters water and provides habitat."},
            {"point_type": "habitat_quality_flag", "value": "Its presence suggests an established reef community."},
        ]
    )
    interp = GGUFInterpreter(ScriptedCompleter(response))
    points = interp.interpret(_context())
    assert len(points) == 2, points
    assert all(p["produced_by"] == "verify" for p in points)
    assert {p["point_type"] for p in points} == {"ecological_role", "habitat_quality_flag"}
    assert all(p["point_type"] in INTERPRETER_POINT_TYPES for p in points)
    assert interp.version == "gguf-test"


def test_interpreter_refuses_out_of_vocabulary_and_measurements():
    # rarity_score and anomaly_flag are data-derived and must never come from the
    # model; an unknown type and a non-string value must be dropped too.
    response = json.dumps(
        [
            {"point_type": "rarity_score", "value": "very rare", "numeric_value": 0.97},
            {"point_type": "anomaly_flag", "value": "anomalous"},
            {"point_type": "not_a_real_type", "value": "something"},
            {"point_type": "ecological_role", "value": ""},
            {"point_type": "behavioral_context", "value": "Likely filter-feeding during the day."},
        ]
    )
    interp = GGUFInterpreter(ScriptedCompleter(response))
    points = interp.interpret(_context())
    assert len(points) == 1, points
    assert points[0]["point_type"] == "behavioral_context"
    # None of the excluded, data-derived types survived.
    assert all(p["point_type"] not in {"rarity_score", "anomaly_flag"} for p in points)


def test_interpreter_no_taxon_yields_nothing():
    interp = GGUFInterpreter(ScriptedCompleter("[]"))
    ctx = _context(resolved_name=None)
    assert interp.interpret(ctx) == []


def test_interpreter_malformed_response_yields_nothing():
    interp = GGUFInterpreter(ScriptedCompleter("I could not produce structured output."))
    assert interp.interpret(_context()) == []


def test_interpreter_caps_point_count():
    many = [{"point_type": "ecological_role", "value": f"reading number {i}"} for i in range(20)]
    interp = GGUFInterpreter(ScriptedCompleter(json.dumps(many)))
    points = interp.interpret(_context())
    assert len(points) <= 6, len(points)


def test_prompt_includes_taxon_and_measured_conditions():
    completer = ScriptedCompleter("[]")
    interp = GGUFInterpreter(completer)
    readings = [{"channel": "water_temp_c", "value": 29.1, "unit": "degC", "status": "measured"}]
    interp.interpret(_context(resolved_name="Panthera leo", companions=["Equus quagga"], readings=readings))
    prompt = completer.prompts[0]
    assert "Panthera leo" in prompt
    assert "water_temp_c" in prompt
    assert "Equus quagga" in prompt
    # A failed or missing reading must never be presented as a fact.
    assert "not restate any measurement" in prompt


def test_prompt_excludes_unmeasured_readings():
    completer = ScriptedCompleter("[]")
    interp = GGUFInterpreter(completer)
    readings = [
        {"channel": "ph", "value": None, "status": "sensor_error"},
        {"channel": "salinity_psu", "value": 35.0, "unit": "PSU", "status": "measured"},
    ]
    interp.interpret(_context(readings=readings))
    prompt = completer.prompts[0]
    assert "salinity_psu" in prompt
    assert "ph " not in prompt.replace("salinity_psu", "")  # the errored channel is absent


def test_extract_json_array_handles_surrounding_text():
    assert _extract_json_array('Here you go: [{"a": 1}] thanks') == [{"a": 1}]
    # A bracket inside a quoted string must not confuse the matcher.
    assert _extract_json_array('[{"value": "a ] b"}]') == [{"value": "a ] b"}]
    assert _extract_json_array("no array here") is None
    assert _extract_json_array("[not valid json]") is None


def test_narrator_rephrases_and_is_never_load_bearing():
    narrator = GGUFNarrator(ScriptedCompleter("  Water temperature is trending upward over time.  "))
    out = narrator.narrate(pattern_type="temporal_shift", template="water_temp_c shows a rising trend")
    assert out == "Water temperature is trending upward over time."
    assert narrator.version == "gguf-test"

    # An empty completion returns an empty string, which the dream pass reads as
    # "keep the built-in description".
    empty = GGUFNarrator(ScriptedCompleter("   "))
    assert empty.narrate(pattern_type="co_occurrence", template="taxa A and B co-occur") == ""


def test_resolve_model_file_reports_missing_clearly():
    settings = load_settings()
    tmp = Path(tempfile.mkdtemp(prefix="audtheia_gguf_"))
    # An empty folder has no GGUF file, which must be a clear, catchable error.
    settings.raw["desktop_models"]["llm"]["path"] = str(tmp)
    raised = False
    try:
        _resolve_model_file(settings)
    except LLMError:
        raised = True
    assert raised

    # A folder with a GGUF file resolves to that file.
    (tmp / "model.gguf").write_bytes(b"not a real model")
    resolved = _resolve_model_file(settings)
    assert resolved.name == "model.gguf"


def test_points_pass_the_real_verify_engine():
    settings = load_settings()
    tmp = Path(tempfile.mkdtemp(prefix="audtheia_gguf_"))
    settings.raw["paths"]["db_path"] = str(tmp / "audtheia.db")
    db = Database(settings.db_path(), **settings.database_kwargs())
    db.initialize_schema(settings.schema_path())

    response = json.dumps(
        [
            {"point_type": "ecological_role", "value": "A reef-building filter feeder."},
            {"point_type": "interaction_pattern", "value": "Often shares substrate with encrusting species."},
        ]
    )

    class _NoFrames:
        version = None

        def verify_frames(self, frame_paths):
            return []

    engine = VerifyEngine(
        settings=settings,
        db=db,
        verifier=_NoFrames(),
        interpreter=GGUFInterpreter(ScriptedCompleter(response, version="qwen-test")),
    )

    coerced = engine._interpret(_context())
    assert len(coerced) == 2
    assert all(isinstance(p, InterpretationPoint) for p in coerced)
    # Every produced point must pass the engine's own validation and be a
    # recognized interpretive type destined for the inferred provenance.
    for p in coerced:
        assert engine._point_rejection(p) is None, p
        assert p.point_type in INTERPRETATION_POINT_TYPES
    assert DATA_SOURCE_LLM_INFERRED == "llm_inferred"


def main() -> int:
    checks = [
        test_interpreter_parses_accepted_points,
        test_interpreter_refuses_out_of_vocabulary_and_measurements,
        test_interpreter_no_taxon_yields_nothing,
        test_interpreter_malformed_response_yields_nothing,
        test_interpreter_caps_point_count,
        test_prompt_includes_taxon_and_measured_conditions,
        test_prompt_excludes_unmeasured_readings,
        test_extract_json_array_handles_surrounding_text,
        test_narrator_rephrases_and_is_never_load_bearing,
        test_resolve_model_file_reports_missing_clearly,
        test_points_pass_the_real_verify_engine,
    ]
    failures = 0
    for check in checks:
        try:
            check()
            print(f"PASS  {check.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {check.__name__}: {exc}")
            import traceback
            traceback.print_exc()
    print(f"\n{len(checks) - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
