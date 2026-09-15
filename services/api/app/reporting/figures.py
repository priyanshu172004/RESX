"""Charts as images, for the exported formats.

On screen a chart is SVG drawn by the browser. A PDF or a Word file cannot run
that, so the same chart has to be drawn again here — and "again" is the whole
problem. A figure that is blue on screen and orange in the PDF is not the same
figure, and a reader comparing the two has to work out which one to believe.

So the palette below is copied from `apps/web/src/app/globals.css` rather than
chosen. The eight categorical steps are the validated series palette in slot
order; the four status steps are the reserved good/warning/serious/critical set.
`CATEGORICAL` is never indexed past its length by wrapping — a ninth series
folds, exactly as it does in the browser, because a cycled hue silently tells
the reader two different series are the same one.

The rules the on-screen charts obey apply here unchanged:

* **One axis.** The server already split value columns by role, so a chart
  arriving here holds one unit and needs one scale. Nothing here re-joins them.
* **A legend whenever there are two or more series**, because identity must
  never be carried by colour alone — in print that matters more, not less,
  since a reader cannot hover to check.
* **Status charts take the status palette**, per mark, driven by the slice
  label rather than its position.

Rendered at 160 DPI: sharp in print without making a 20-figure PDF enormous.
"""

from __future__ import annotations

import io
import math
from typing import Any

#: The eight categorical steps, in slot order, copied from the web palette
#: (`--chart-1` … `--chart-8`). Assigned by position and never cycled.
CATEGORICAL: tuple[str, ...] = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)

#: Reserved. Never used for "series 4".
STATUS: dict[str, str] = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

#: Which status a slice label means. Mirrors `STATUS_BY_LABEL` in
#: `report-charts.tsx`. An unrecognised label is a *warning*, never "good" —
#: guessing optimistically about a state we failed to parse is the one error
#: that would mislead rather than merely confuse.
STATUS_BY_LABEL: dict[str, str] = {
    "confirmed": "good",
    "unreviewed": "warning",
    "contested": "serious",
    "refuted": "critical",
}

INK = "#1c1c1c"
MUTED = "#6b6b6b"
GRID = "#e4e4e4"
SURFACE = "#ffffff"

DPI = 160
FIGSIZE = (7.2, 3.6)

#: Beyond this many marks the x labels collide whatever the rotation, so they
#: are thinned rather than overlapped into mush.
MAX_TICK_LABELS = 14


def _status_for(label: str) -> str:
    role = STATUS_BY_LABEL.get(label.strip().lower(), "warning")
    return STATUS[role]


def _colour(chart: Any, index: int, label: str) -> str:
    if str(getattr(chart, "palette", "categorical")) == "status":
        return _status_for(label)
    return CATEGORICAL[index % len(CATEGORICAL)]


def _series_of(chart: Any) -> list[Any]:
    return list(getattr(chart, "series", None) or [])


def _points_of(series: Any) -> list[dict[str, Any]]:
    return list(getattr(series, "points", None) or [])


def _thin(labels: list[str]) -> list[str]:
    """Keep every nth label so the axis stays readable.

    Dropping labels is honest — the marks are all still drawn — whereas
    overlapping them renders the axis unreadable while implying it is fine.
    """
    if len(labels) <= MAX_TICK_LABELS:
        return labels
    step = math.ceil(len(labels) / MAX_TICK_LABELS)
    return [label if i % step == 0 else "" for i, label in enumerate(labels)]


def render_chart_png(chart: Any) -> bytes | None:
    """One chart as a PNG, or None when it cannot be drawn.

    Returning None rather than raising is deliberate: a figure that fails to
    render must not take the whole export down with it. The caller falls back
    to the data table, which carries the same numbers.
    """
    series = _series_of(chart)
    if not series or not any(_points_of(s) for s in series):
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")  # headless; no display, no GUI backend
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except Exception:  # pragma: no cover - matplotlib is a hard dependency
        return None

    kind = str(getattr(chart, "kind", "bar"))

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    try:
        if kind == "donut":
            _draw_donut(ax, chart, series[0])
        elif kind == "line":
            _draw_line(ax, chart, series)
        else:
            _draw_bar(ax, chart, series)

        if kind != "donut":
            ax.set_xlabel(str(getattr(chart, "x_label", "") or ""), color=MUTED, fontsize=9)
            ax.set_ylabel(str(getattr(chart, "y_label", "") or ""), color=MUTED, fontsize=9)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _compact(v)))
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_color(GRID)
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)

        # No title drawn into the image. Every caller prints the chart's title
        # as a real heading beside it — searchable text, and what Word's
        # navigation pane and a screen reader read — so drawing it again here
        # printed it twice on the page.

        # A legend the moment there are two or more series: identity must never
        # rest on colour alone, and a printed page offers no hover to recover it.
        if len(series) >= 2 and kind != "donut":
            ax.legend(
                frameon=False,
                fontsize=8,
                labelcolor=MUTED,
                loc="upper left",
                bbox_to_anchor=(0, -0.22),
                ncol=min(len(series), 3),
            )

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
        return buffer.getvalue()
    except Exception:
        return None
    finally:
        plt.close(fig)


def _compact(value: float) -> str:
    """Axis ticks a reader can scan: 1.2M rather than 1200000.

    Fractions are kept. Rounding every tick to a whole number produced an axis
    reading 4, 4, 5, 5, 6, 6, 7 — matplotlib had chosen half-steps and the
    formatter collapsed each pair into the same label, so the scale silently
    lied about its own intervals. A duplicated tick is worse than a longer one.
    """
    magnitude = abs(value)
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if magnitude >= limit:
            scaled = value / limit
            text = f"{scaled:,.1f}"
            return f"{text[:-2] if text.endswith('.0') else text}{suffix}"
    if magnitude and magnitude < 1:
        return f"{value:,.2f}"
    if not float(value).is_integer():
        return f"{value:,.1f}"
    return f"{value:,.0f}"


def _draw_line(ax: Any, chart: Any, series: list[Any]) -> None:
    for index, one in enumerate(series):
        points = _points_of(one)
        labels = [str(p.get("x", "")) for p in points]
        values = [float(p.get("y") or 0.0) for p in points]
        ax.plot(
            labels,
            values,
            linewidth=2.0,
            marker="o",
            markersize=4.5,
            color=_colour(chart, index, str(getattr(one, "label", ""))),
            label=str(getattr(one, "label", "") or ""),
        )
    first = _points_of(series[0])
    ax.set_xticks(range(len(first)))
    ax.set_xticklabels(_thin([str(p.get("x", "")) for p in first]), rotation=0)


def _draw_bar(ax: Any, chart: Any, series: list[Any]) -> None:
    import numpy as np

    count = len(series)
    first = _points_of(series[0])
    positions = np.arange(len(first), dtype="float64")
    width = min(0.8 / max(count, 1), 0.38)

    for index, one in enumerate(series):
        points = _points_of(one)
        values = [float(p.get("y") or 0.0) for p in points]
        offset = (index - (count - 1) / 2) * width
        label = str(getattr(one, "label", "") or "")
        if str(getattr(chart, "palette", "categorical")) == "status":
            # Per-bar colour: these marks carry state, and the state is the
            # slice label rather than the series it sits in.
            colours = [_status_for(str(p.get("x", ""))) for p in points]
        else:
            colours = [_colour(chart, index, label)] * len(points)
        ax.bar(
            positions + offset,
            values,
            width=width * 0.92,  # a 2px-equivalent gap between adjacent fills
            color=colours,
            label=label,
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(_thin([str(p.get("x", "")) for p in first]), rotation=0)


def _draw_donut(ax: Any, chart: Any, series: Any) -> None:
    points = _points_of(series)
    labels = [str(p.get("x", "")) for p in points]
    values = [float(p.get("y") or 0.0) for p in points]
    if not any(values):
        raise ValueError("a donut of zeroes is not a chart")

    colours = [_colour(chart, i, label) for i, label in enumerate(labels)]
    total = sum(values)
    _wedges, _texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colours,
        autopct=lambda pct: f"{pct:.0f}%" if pct >= 4 else "",
        pctdistance=0.78,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.42, "edgecolor": SURFACE, "linewidth": 2},
        textprops={"color": INK, "fontsize": 8},
    )
    for text in autotexts:
        text.set_color(SURFACE)
        text.set_fontsize(8)
    ax.text(
        0,
        0,
        _compact(total),
        ha="center",
        va="center",
        color=INK,
        fontsize=13,
    )
    ax.set_aspect("equal")


def chart_table(chart: Any) -> tuple[list[str], list[list[str]]]:
    """A chart's numbers as a header and rows.

    Every export carries this beside the image. Not a fallback: a figure a
    reader cannot read the values off is half a figure, and colour-blind and
    print readers need the numbers rather than the shape.
    """
    series = _series_of(chart)
    if not series:
        return [], []

    order: list[str] = []
    seen: set[str] = set()
    for one in series:
        for point in _points_of(one):
            key = str(point.get("x", ""))
            if key not in seen:
                seen.add(key)
                order.append(key)

    header = [str(getattr(chart, "x_label", "") or "Category")]
    header += [str(getattr(one, "label", "") or one.key) for one in series]

    rows: list[list[str]] = []
    for key in order:
        row = [key]
        for one in series:
            found = next(
                (p for p in _points_of(one) if str(p.get("x", "")) == key),
                None,
            )
            row.append("" if found is None else _compact(float(found.get("y") or 0.0)))
        rows.append(row)
    return header, rows
