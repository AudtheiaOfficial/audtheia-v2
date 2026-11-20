"""Report charts (matplotlib, offline, imported lazily behind a seam).

Path: audtheia/reports/charts.py

The report's executive summary shows a few figures so a reader sees the shape of
the record at a glance, not only tables of numbers. Every figure is drawn from
the same measured record the tables are built from, so a chart never introduces a
claim the data does not support: a bar is a count, a histogram is a distribution
of measured confidences, and the verification figure is a count of gate outcomes.

matplotlib is imported lazily and uses the non-interactive Agg backend, so this
module loads with the library absent and needs no display. When the library is
not installed the renderer returns no charts and the report simply omits the
figures rather than failing, exactly as the PDF degrades without fpdf2.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("audtheia.reports.charts")

# The report palette, matching the interface's earth tones so a generated report
# and the live application read as one product.
_AMBER = "#B45309"
_TEAL = "#0F766E"
_SLATE = "#475569"
_VIOLET = "#6D28D9"
_GREEN = "#15803D"
_MUTED = "#6B7280"
_GRID = "#E5E7EB"
_INK = "#111827"

_SERIES = [_AMBER, _TEAL, _SLATE, _GREEN, _VIOLET]

# One consistent figure width so every chart sits cleanly in the PDF column.
_FIG_W = 6.6
_DPI = 150


def _load_matplotlib():
    """Import matplotlib on the Agg backend, or return None when it is absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:  # noqa: BLE001 - charts are enrichment, never load-bearing
        logger.info("matplotlib is not available; the report will omit its figures")
        return None


def _parse_utc(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z") or text.endswith("z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _style_axes(ax) -> None:
    """A clean, restrained axis style shared by every figure."""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_GRID)
    ax.tick_params(colors=_MUTED, labelsize=8, length=0)
    ax.title.set_color(_INK)
    ax.yaxis.label.set_color(_MUTED)
    ax.xaxis.label.set_color(_MUTED)


def render_charts(model, out_dir: str | Path) -> dict:
    """Render the report's figures into out_dir, returning {name: Path}.

    Returns an empty dict when matplotlib is absent, or omits any single figure
    the record cannot support (for example a confidence histogram when nothing
    was scored), so the report shows only figures that mean something.
    """
    plt = _load_matplotlib()
    if plt is None:
        return {}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    charts: dict = {}

    for name, fn in (
        ("species_composition", _species_composition),
        ("detection_timeline", _detection_timeline),
        ("confidence_distribution", _confidence_distribution),
        ("verification_summary", _verification_summary),
    ):
        try:
            path = fn(plt, model, out)
            if path is not None:
                charts[name] = path
        except Exception:  # noqa: BLE001 - one bad figure never blocks the report
            logger.exception("could not render the %s figure; omitting it", name)

    return charts


def _save(plt, fig, out: Path, name: str) -> Path:
    path = out / f"{name}.png"
    fig.savefig(str(path), dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _species_composition(plt, model, out: Path) -> Optional[Path]:
    """Top taxa by the number of events each was recorded in."""
    counts = (model.analytics or {}).get("taxon_event_counts") or {}
    names = (model.analytics or {}).get("taxon_display_names") or {}
    if not counts:
        return None
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    labels = [str(names.get(k, k))[:34] for k, _ in top][::-1]
    values = [v for _, v in top][::-1]

    fig, ax = plt.subplots(figsize=(_FIG_W, max(1.8, 0.42 * len(labels) + 0.8)))
    ax.barh(labels, values, color=_TEAL, height=0.66)
    ax.set_title("Species recorded, by number of events", fontsize=10, loc="left", pad=8)
    ax.set_xlabel("events")
    _style_axes(ax)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v}", va="center", ha="left", color=_MUTED, fontsize=8)
    return _save(plt, fig, out, "species_composition")


def _detection_timeline(plt, model, out: Path) -> Optional[Path]:
    """Events per day across the reporting window."""
    from collections import Counter
    days: Counter = Counter()
    for rec in model.records:
        dt = _parse_utc(rec.observation.get("first_seen"))
        if dt is not None:
            days[dt.date()] += 1
    if not days:
        return None
    ordered = sorted(days.items())
    xs = [d.strftime("%m-%d") for d, _ in ordered]
    ys = [c for _, c in ordered]

    fig, ax = plt.subplots(figsize=(_FIG_W, 2.4))
    ax.bar(xs, ys, color=_AMBER, width=0.7)
    ax.set_title("Detections over time", fontsize=10, loc="left", pad=8)
    ax.set_ylabel("events")
    _style_axes(ax)
    if len(xs) > 12:
        step = max(1, len(xs) // 12)
        ax.set_xticks(range(0, len(xs), step))
        ax.set_xticklabels([xs[i] for i in range(0, len(xs), step)], rotation=0)
    return _save(plt, fig, out, "detection_timeline")


def _confidence_distribution(plt, model, out: Path) -> Optional[Path]:
    """Distribution of the measured detection confidences."""
    values = []
    for rec in model.records:
        for det in list(rec.vision_detections) + list(rec.audio_detections):
            c = det.get("confidence")
            if isinstance(c, (int, float)):
                values.append(float(c))
    if not values:
        return None

    fig, ax = plt.subplots(figsize=(_FIG_W, 2.4))
    ax.hist(values, bins=10, range=(0, 1), color=_SLATE, edgecolor="white")
    ax.set_title("Detection confidence distribution", fontsize=10, loc="left", pad=8)
    ax.set_xlabel("confidence")
    ax.set_ylabel("detections")
    _style_axes(ax)
    return _save(plt, fig, out, "confidence_distribution")


def _verification_summary(plt, model, out: Path) -> Optional[Path]:
    """Verification per gate: the two-model check and the acoustic confidence gate.

    One bar per gate, never pooled, because a sound event can only pass a
    confidence floor while an event with frames is checked by a second model.
    Events no model checked are shown as their own segment, not as failures.
    """
    split = (model.analytics or {}).get("verification_by_gate") or {}
    vis = split.get("visual") or {}
    aud = split.get("audio") or {}
    rows = []
    if vis.get("total"):
        rows.append(("Visual: two-model check", [
            ("verified", int(vis.get("verified", 0)), _GREEN),
            ("not verified", int(vis.get("not_verified", 0)), _AMBER),
            ("not checked by a second model", int(vis.get("not_run", 0)) + int(vis.get("pending", 0)), _GRID),
        ], int(vis["total"])))
    if aud.get("total"):
        rows.append(("Audio: confidence gate", [
            ("cleared", int(aud.get("cleared", 0)), _TEAL),
            ("not cleared", int(aud.get("not_cleared", 0)), _AMBER),
            ("not yet checked", int(aud.get("pending", 0)), _GRID),
        ], int(aud["total"])))
    if not rows:
        return None

    fig, axes = plt.subplots(len(rows), 1, figsize=(_FIG_W, 0.9 + 0.75 * len(rows)), squeeze=False)
    for ax, (label, segments, total) in zip(axes[:, 0], rows):
        left = 0
        for name, value, color in segments:
            if value <= 0:
                continue
            ax.barh([label], [value], left=[left], color=color, height=0.5, label=f"{name} ({value})")
            left += value
        ax.set_xlim(0, total)
        _style_axes(ax)
        ax.legend(loc="lower right", frameon=False, fontsize=7, ncol=3, bbox_to_anchor=(1.0, -0.75))
    axes[0, 0].set_title("Verification by gate", fontsize=10, loc="left", pad=8)
    fig.tight_layout()
    return _save(plt, fig, out, "verification_summary")
