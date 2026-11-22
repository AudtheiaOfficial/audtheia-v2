"""Every top-level function in the page script has a unique name.

Path: tests/test_app_js_names.py

The page script, audtheia/app/static/app.js, is one closure whose functions are
declared at its top level. JavaScript allows two declarations with the same name
there and silently keeps the later one, so a second function sharing a name
replaces the first everywhere it is called, without any error. This check reads
the script as text and fails when a top-level name is declared twice. It only
reads the file and writes nothing.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "audtheia" / "app" / "static" / "app.js"

# Top-level declarations inside the closure sit at an indent of two spaces.
TOP_LEVEL = re.compile(r"^  function\s+([A-Za-z_$][\w$]*)\s*\(", re.M)


def run() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    names = Counter(TOP_LEVEL.findall(text))
    if len(names) < 50:
        raise AssertionError(f"only {len(names)} top-level functions were found; the indent rule no longer matches the script")
    repeated = sorted(name for name, n in names.items() if n > 1)
    if repeated:
        raise AssertionError("top-level functions declared more than once: " + ", ".join(repeated))
    print(f"ALL CHECKS PASSED ({len(names)} unique top-level functions)")


if __name__ == "__main__":
    try:
        run()
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
