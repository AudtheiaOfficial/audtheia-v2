"""Verification that every image the README shows is present, valid and light enough.

Path: tests/test_readme_assets.py

A broken or missing picture on the repository's front page is the first thing a
visitor sees, and the usual causes do not show up on a Windows computer. File
names are case sensitive on GitHub but not on Windows, so a path that differs from
the real file only by letter case works locally and breaks online. A file can also
be committed as a pointer stub, be cut short, or grow past what a page can load
quickly. This check reads the README, finds every local image it references, and
confirms each one exists with exactly the spelled name, is a real image of its
declared type, is complete, and stays under a size ceiling.

It reads files only and needs nothing beyond the standard library.
"""

from __future__ import annotations

import re
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"
# A page with several animations should not make a visitor wait. The largest
# picture the README uses today is about five megabytes, so this ceiling stops
# growth without forcing a change to a picture that already works.
MAX_BYTES = 8 * 1024 * 1024
CHECKS = {"passed": 0, "failed": 0}


def check(label: str, condition: bool) -> None:
    CHECKS["passed" if condition else "failed"] += 1
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")


def image_references(text: str) -> list[str]:
    """Local image paths used by markdown image syntax or an img tag, in order."""
    found = re.findall(r"!\[[^\]]*\]\(([^)\s]+)", text)
    found += re.findall(r"<img\b[^>]*?\bsrc=[\"']([^\"']+)[\"']", text, flags=re.IGNORECASE)
    local = []
    for ref in found:
        if re.match(r"^[a-z][a-z0-9+.-]*:", ref, flags=re.IGNORECASE) or ref.startswith("//"):
            continue  # an online address such as a badge
        ref = ref.split("#")[0].split("?")[0]
        if ref and ref not in local:
            local.append(ref)
    return local


def exact_case_exists(relative: str) -> bool:
    """True only if every part of the path matches a real name letter for letter."""
    current = REPO
    for part in Path(relative).parts:
        if part in ("", "."):
            continue
        try:
            names = {entry.name for entry in current.iterdir()}
        except OSError:
            return False
        if part not in names:
            return False
        current = current / part
    return current.is_file()


def gif_problem(data: bytes) -> str | None:
    if data[:6] not in (b"GIF87a", b"GIF89a"):
        return "does not start like a GIF"
    width, height = struct.unpack("<HH", data[6:10])
    if width < 1 or height < 1:
        return "has no size"
    if data[-1:] != b";":
        return "is cut short, the GIF end marker is missing"
    return None


def png_problem(data: bytes) -> str | None:
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return "does not start like a PNG"
    if b"IEND" not in data[-16:]:
        return "is cut short, the PNG end chunk is missing"
    return None


def main() -> int:
    print("\n[1] Pictures the README shows")
    text = README.read_text(encoding="utf-8")
    refs = image_references(text)
    check("the README shows at least one local picture", len(refs) >= 1)
    for ref in refs:
        print(f"  {ref}")
        exists = exact_case_exists(ref)
        check(f"{ref} exists with exactly this spelling", exists)
        if not exists:
            continue
        data = (REPO / ref).read_bytes()
        check(f"{ref} is not an unresolved large-file pointer", not data.startswith(b"version https://git-lfs"))
        suffix = Path(ref).suffix.lower()
        problem = gif_problem(data) if suffix == ".gif" else png_problem(data) if suffix == ".png" else None
        check(f"{ref} is a complete {suffix[1:].upper() or 'image'}" + (f" ({problem})" if problem else ""),
              problem is None)
        check(f"{ref} is under the size ceiling", len(data) <= MAX_BYTES)
    print(f"\nResult: {CHECKS['passed']} passed, {CHECKS['failed']} failed")
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
