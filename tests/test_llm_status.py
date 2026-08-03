"""The language-model panel loads a folder model and never alarms a person.

Path: tests/test_llm_status.py

Two fixes are proven here. The loader falls back to the default language-model
folder when no specific model was pinned, so a model dropped in and shown active
also loads (the manager and the loader agree). And the readiness status is calm
and plain: a model that is present reads as present even if an earlier, now-stale
error was recorded, and the one failure a person cannot self-diagnose (the runtime
not matching this CPU) is stated gently, with the raw error code and build flags
kept out of the panel.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from audtheia.config import load_settings  # noqa: E402
from audtheia.inference.gguf_llm import _resolve_model_file  # noqa: E402
from audtheia.app import server as srv  # noqa: E402
import audtheia.app.orchestrator as orch  # noqa: E402

CHECKS = {"passed": 0, "failed": 0}


def check(label, cond):
    CHECKS["passed" if cond else "failed"] += 1
    print(("  PASS  " if cond else "  FAIL  ") + label)


def _settings(tmp: Path, *, llm_path=None):
    base = json.loads((REPO / "config" / "settings.json").read_text(encoding="utf-8"))
    base["node"]["role"] = "desktop"
    base["paths"]["models_dir"] = str(tmp / "models")
    base["desktop_models"]["llm"]["path"] = llm_path
    p = tmp / "s.json"
    p.write_text(json.dumps(base), encoding="utf-8")
    return load_settings(p)


def test_loader_falls_back_to_folder():
    print("\n[1] With no pinned path, the loader uses the default llm folder")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        settings = _settings(tmp, llm_path=None)
        llm_dir = tmp / "models" / "llm"
        llm_dir.mkdir(parents=True)
        model = llm_dir / "Some-Model.gguf"
        model.write_bytes(b"gguf")
        resolved = _resolve_model_file(settings)
        check("the folder model is resolved without a pinned path", Path(resolved) == model)


def test_status_is_calm():
    print("\n[2] The status is calm: present stays present; only a CPU mismatch is surfaced, gently")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        model = tmp / "m.gguf"; model.write_bytes(b"gguf")
        settings = _settings(tmp, llm_path=str(model))

        # Pretend the runtime is installed, so the status reaches the model logic.
        original = srv._llm_runtime_available
        srv._llm_runtime_available = lambda: True
        try:
            orch._LAST_LLM_ERROR = None
            s = srv._llm_status(settings)
            check("a present model with no error reads as present", s["status"] == "model_present")
            check("a present model shows no scary remedy", not s["remedy"])

            # A stale, non-CPU error must not turn the panel into a failure.
            orch._LAST_LLM_ERROR = "no desktop language model is configured under desktop_models.llm.path."
            s = srv._llm_status(settings)
            check("a stale config error does not alarm; still present", s["status"] == "model_present")

            # A real CPU-instruction failure is surfaced, but gently and without jargon.
            orch._LAST_LLM_ERROR = "could not be loaded: [WinError -1073741795] Windows Error 0xc000001d"
            s = srv._llm_status(settings)
            check("a CPU mismatch is reported as such", s["status"] == "cpu_incompatible")
            check("the message stays plain (no error code or CMAKE in it)",
                  "0xc000001d" not in s["message"] and "CMAKE" not in s["message"])
            check("the remedy points to the guide", "docs/language-model.md" in s["remedy"])
        finally:
            srv._llm_runtime_available = original
            orch._LAST_LLM_ERROR = None


def main() -> int:
    test_loader_falls_back_to_folder()
    test_status_is_calm()
    print(f"\n==== llm status: {CHECKS['passed']} passed, {CHECKS['failed']} failed ====")
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
