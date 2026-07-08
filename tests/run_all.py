#!/usr/bin/env python3
"""Run the whole test suite and report one result.

Path: tests/run_all.py

The suite is written to run entirely on mocked hardware, so no Raspberry Pi,
accelerator, camera, or hydrophone is needed. Every check module runs itself
through its own entry point and returns a clear exit code, so this runner simply
runs each one with the current Python, gathers the outcomes, and prints a single
summary. It exits non-zero if anything failed, which is what a continuous check
or a release gate reads.

One check module, the detection-loop check, needs the object-tracker package to
run. When that package is not installed, the runner reports the module as skipped
rather than failed, so a minimal machine still gets a clean pass on everything it
can run. Installing the tracker package makes that module run too.

Run it with the environment setup created (which has the test libraries), from
the repository root:

    .venv/bin/python tests/run_all.py            (macOS, Linux, Raspberry Pi OS)
    .venv\\Scripts\\python tests\\run_all.py       (Windows)
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TESTS = REPO / "tests"

# Check modules that need an optional package to run at all. When the package is
# absent the module is skipped, not failed.
OPTIONAL_REQUIREMENTS = {"test_monitor": "supervision"}

# The libraries a fully green run installs. The storage, configuration, and
# quality-control checks need none of these; the others use them, and the
# detection-loop check needs the tracker. This list is printed so a person knows
# exactly what to install for a complete run.
TEST_LIBRARIES = (
    "numpy (pipeline and analysis checks)",
    "fpdf2 (report PDF path)",
    "fastapi, uvicorn, httpx (interface checks)",
    "supervision (the detection-loop check)",
)


def _has_package(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # noqa: BLE001 - a broken optional package counts as absent
        return False


def discover() -> list[Path]:
    return sorted(p for p in TESTS.glob("test_*.py"))


def run_module(path: Path) -> tuple[str, str]:
    """Run one check module. Returns (status, captured output)."""
    needed = OPTIONAL_REQUIREMENTS.get(path.stem)
    if needed and not _has_package(needed):
        return "skip", f"the '{needed}' package is not installed"

    proc = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    return ("pass" if proc.returncode == 0 else "fail"), (proc.stdout + proc.stderr)


def main() -> int:
    print("Audtheia test suite")
    print("Libraries a full run uses: " + "; ".join(TEST_LIBRARIES))

    modules = discover()
    results: dict[str, str] = {}

    for path in modules:
        status, output = run_module(path)
        results[path.stem] = status
        mark = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}[status]
        print(f"\n[{mark}] {path.stem}")
        if status == "fail":
            tail = output.strip().splitlines()[-20:]
            print("\n".join("      " + line for line in tail))
        elif status == "skip":
            print(f"      skipped: {output}")

    passed = sum(v == "pass" for v in results.values())
    skipped = sum(v == "skip" for v in results.values())
    failed = sum(v == "fail" for v in results.values())

    print(f"\n==== {passed} passed, {skipped} skipped, {failed} failed ====")
    if skipped:
        skipped_names = ", ".join(n for n, v in results.items() if v == "skip")
        print(f"     skipped ({skipped_names}) can run once their optional package is installed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
