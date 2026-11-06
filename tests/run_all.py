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
run. When that package is not installed, or is installed but cannot be imported
(for example a broken or quarantined compiled dependency of its own), the runner
reports the module as skipped rather than failed, so a minimal or imperfect
machine still gets a clean pass on everything it can run. A working install of the
tracker package makes that module run too.

Run it with the environment setup created (which has the test libraries), from
the repository root:

    .venv/bin/python tests/run_all.py            (macOS, Linux, Raspberry Pi OS)
    .venv\\Scripts\\python tests\\run_all.py       (Windows)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TESTS = REPO / "tests"

# Check modules that need an optional package to run at all. When the package is
# absent, or present but unable to import, the module is skipped, not failed.
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


def _importable(name: str) -> bool:
    """Whether an optional package can actually be imported, not merely found.

    A package can be installed yet fail to import: a compiled dependency may be
    missing, quarantined by antivirus, or built for a different interpreter. Only
    locating the package (find_spec) would miss all of those and let a dependent
    check fail. Importing it in a throwaway subprocess instead means a broken
    package counts as unavailable, so its dependent check is skipped rather than
    failed, and a heavy or faulty import never touches this runner's own process.
    """
    proc = subprocess.run(
        [sys.executable, "-c", f"import {name}"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def discover() -> list[Path]:
    return sorted(p for p in TESTS.glob("test_*.py"))


def run_module(path: Path) -> tuple[str, str]:
    """Run one check module. Returns (status, captured output)."""
    needed = OPTIONAL_REQUIREMENTS.get(path.stem)
    if needed and not _importable(needed):
        return "skip", f"the '{needed}' package is not installed or could not be imported"

    proc = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    return ("pass" if proc.returncode == 0 else "fail"), (proc.stdout + proc.stderr)


def user_data_present() -> list[str]:
    """What in this folder shows it is someone's working installation, not a fresh checkout.

    Several tests derive their scenario configurations from config/settings.json
    and some write beside it, so they must run where that file is the committed
    one and no real record, machine path or installed training component can be
    reached. A working installation is refused rather than risked.
    """
    import json

    found = []
    for rel in ("config/settings.local.json", "database/audtheia.db"):
        if (REPO / rel).exists():
            found.append(rel)
    try:
        raw = json.loads((REPO / "config" / "settings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    for key, value in ((raw.get("training") or {}).get("paths") or {}).items():
        if isinstance(value, str) and (Path(value).is_absolute() or ":" in value[:3]):
            found.append(f"config/settings.json training.paths.{key} = {value}")
    return found


def main() -> int:
    print("Audtheia test suite")
    present = user_data_present()
    if present and "--allow-working-folder" not in sys.argv:
        print("\nRefusing to run: this folder is a working installation, not a fresh checkout.")
        for item in present:
            print(f"  found {item}")
        print("Run the tests in a fresh clone of the repository, so no test can reach your records,"
              " settings or installed training components. docs/setup-guide.md explains how.")
        return 2
    print("Libraries a full run uses: " + "; ".join(TEST_LIBRARIES))

    modules = discover()
    results: dict[str, str] = {}

    for path in modules:
        status, output = run_module(path)
        results[path.stem] = status
        mark = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}[status]
        print(f"\n[{mark}] {path.stem}")
        if status == "fail":
            lines = output.strip().splitlines()
            # Every failed check first, wherever it appears in the output, then the
            # end of the output for a traceback or a summary.
            failed_checks = [line for line in lines if line.lstrip().startswith("FAIL")]
            if failed_checks:
                print("      failed checks:")
                print("\n".join("        " + line.strip() for line in failed_checks))
                print("      end of the output:")
            tail = lines[-20:]
            print("\n".join("      " + line for line in tail))
        elif status == "skip":
            print(f"      skipped: {output}")

    passed = sum(v == "pass" for v in results.values())
    skipped = sum(v == "skip" for v in results.values())
    failed = sum(v == "fail" for v in results.values())

    print(f"\n==== {passed} passed, {skipped} skipped, {failed} failed ====")
    if skipped:
        skipped_names = ", ".join(n for n, v in results.items() if v == "skip")
        print(f"     skipped ({skipped_names}) can run once their optional package installs and imports cleanly.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
