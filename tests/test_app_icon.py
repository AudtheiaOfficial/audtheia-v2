#!/usr/bin/env python3
"""Checks for the application icon and the desktop window's icon handling.

Path: tests/test_app_icon.py

The Windows icon file must carry the sizes the operating system asks for, each
frame must be a well-formed image, and the window module must import and behave
as a no-op for the Windows-only steps on any other platform, so the desktop
window can never fail to open because of its icon.

Run directly: python tests/test_app_icon.py
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

STATIC = REPO / "audtheia" / "app" / "static"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
REQUIRED_SIZES = {16, 32, 48, 256}


def _read_ico_frames(path: Path):
    """The (width, height, bits, data) of every frame in an .ico file."""
    data = path.read_bytes()
    reserved, kind, count = struct.unpack("<HHH", data[:6])
    assert reserved == 0 and kind == 1, "not an icon file"
    frames = []
    for i in range(count):
        width, height, _colors, _res, _planes, bits, size, offset = struct.unpack(
            "<BBBBHHII", data[6 + 16 * i: 22 + 16 * i]
        )
        width = width or 256
        height = height or 256
        assert offset + size <= len(data), "a frame runs past the end of the file"
        frames.append((width, height, bits, data[offset: offset + size]))
    return frames


def test_windows_icon_has_the_required_sizes_and_valid_frames():
    frames = _read_ico_frames(STATIC / "favicon.ico")
    sizes = {w for w, _h, _b, _d in frames}
    assert REQUIRED_SIZES <= sizes, f"the icon is missing sizes; has {sorted(sizes)}"
    for width, height, bits, blob in frames:
        assert width == height, f"a {width}x{height} frame is not square"
        assert bits == 32, f"the {width}px frame is not 32-bit with transparency"
        if blob.startswith(PNG_SIGNATURE):
            w, h = struct.unpack(">II", blob[16:24])
            assert (w, h) == (width, height), f"the {width}px PNG frame declares {w}x{h}"
        else:
            header_size, w, h2 = struct.unpack("<Iii", blob[:12])
            assert header_size == 40 and w == width and h2 == height * 2, (
                f"the {width}px bitmap frame has a malformed header"
            )


def test_png_icon_and_inline_page_icon_are_present():
    png = (STATIC / "favicon.png").read_bytes()
    assert png.startswith(PNG_SIGNATURE), "favicon.png is not a PNG"
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'rel="icon"' in html and "data:image/png;base64," in html, (
        "the interface page must carry its own icon so a browser tab is branded offline"
    )


def test_window_module_icon_steps_are_safe_off_windows():
    from audtheia.app import window

    assert window.WINDOWS_APP_ID
    if not sys.platform.startswith("win"):
        # Neither Windows-only step may do anything, or raise, elsewhere.
        window._set_windows_app_identity()
        window._apply_windows_window_icon(STATIC / "favicon.ico")


def main() -> int:
    checks = [
        test_windows_icon_has_the_required_sizes_and_valid_frames,
        test_png_icon_and_inline_page_icon_are_present,
        test_window_module_icon_steps_are_safe_off_windows,
    ]
    failures = 0
    for check in checks:
        try:
            check()
            print(f"PASS  {check.__name__}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"FAIL  {check.__name__}: {exc}")
    print(f"\n{len(checks) - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
