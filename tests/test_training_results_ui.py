"""Verification for the Results panel's colours and wiring.

Path: tests/test_training_results_ui.py

Parses the stylesheet and checks that every chart series colour reaches the WCAG
non-text contrast of 3:1 against both the surface and the page of all seven
themes, with a translucent surface composited over its page as the browser draws
it, and that neighbouring series stay apart. Then checks that the Results panel
is wired into the page, that the charts are coloured only from the series tokens,
and that the panel's text never words a verdict about which model is better.

Standard library only.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STATIC = REPO / "audtheia" / "app" / "static"
THEMES = ("ocean", "forest", "cloud-dancer", "cyberpunk", "retro", "glass", "neumorph")
SERIES = 8
CHECKS = {"passed": 0, "failed": 0}


def check(label: str, condition: bool) -> None:
    CHECKS["passed" if condition else "failed"] += 1
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")


def _channel(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(rgb) -> float:
    r, g, b = (_channel(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b) -> float:
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def colour(value: str, under=None):
    value = value.strip()
    if value.startswith("#"):
        h = value[1:]
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    m = re.match(r"rgba?\(([^)]*)\)", value)
    if not m:
        raise ValueError(f"not a colour: {value}")
    parts = [float(x) for x in m.group(1).split(",")]
    rgb = tuple(x / 255 for x in parts[:3])
    alpha = parts[3] if len(parts) > 3 else 1.0
    if alpha < 1.0:
        if under is None:
            raise ValueError("a translucent colour needs the colour under it")
        return tuple(alpha * c + (1 - alpha) * u for c, u in zip(rgb, under))
    return rgb


def theme_tokens(css: str) -> dict:
    tokens: dict = {}
    for _sel, name, body in re.findall(r'(\[data-theme="([a-z-]+)"\][^{]*)\{([^}]*)\}', css):
        tokens.setdefault(name, {}).update(dict(re.findall(r"(--[a-z0-9-]+):\s*([^;]+);", body)))
    merged = {}
    for name in THEMES:
        values = dict(tokens.get("ocean", {}))
        values.update(tokens.get(name, {}))
        merged[name] = values
    return merged


def test_contrast() -> None:
    print("\n[1] Every series colour is legible in every theme")
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    tokens = theme_tokens(css)
    check("all seven themes are found in the stylesheet", all(tokens[t].get("--bg") for t in THEMES))
    for name in THEMES:
        v = tokens[name]
        page = colour(v["--bg"])
        surface = colour(v["--surface"], page)
        worst = min(min(contrast(colour(v[f"--series-{i}"]), surface), contrast(colour(v[f"--series-{i}"]), page))
                    for i in range(1, SERIES + 1))
        check(f"{name}: every series colour reaches 3:1 on the surface and the page (lowest {worst:.2f})", worst >= 3.0)
        distinct = len({v[f"--series-{i}"].strip().lower() for i in range(1, SERIES + 1)})
        check(f"{name}: the eight series colours are all different", distinct == SERIES)


def test_wiring() -> None:
    print("\n[2] The panel is wired in and coloured only from the tokens")
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    check("the Results tab sits in the Train sections", 'data-subpanel="train-results"' in html
          and 'data-region="train-results"' in html)
    check("the Results tab has a loader", '"train-results": function () { loadTrainResults(); }' in js)
    start = js.index("// 8d. Results")
    end = js.index("// 9. Startup")
    section = js[start:end]
    check("the chart code names no colour of its own, apart from the fallback for a missing token",
          re.findall(r"#[0-9a-fA-F]{6}\b", section) == ["#888888"])
    check("every chart kind the server sends is drawn", all(k in section for k in ('"line"', '"bar"', "trainMatrixSvg")))
    check("a picture is made from the same drawing that is shown", "trainResolvedSvg(svg)" in section and "registry[chart.id] = svg" in section)
    lower = section.lower()
    check("the panel never words a verdict", not any(w in lower for w in ("better", "improved", "worse", "winner")))
    check("the sound run card names its base model", '"Mode and base model"' in js)


def main() -> int:
    test_contrast()
    test_wiring()
    print(f"\nResult: {CHECKS['passed']} passed, {CHECKS['failed']} failed")
    return 0 if CHECKS["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
