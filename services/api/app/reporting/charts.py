"""Charts derived from the extracted tables, in code.

The one design decision that matters here: **a chart is computed, never
written.** The Synthesizer is not asked for chart data and cannot supply it.

The reason is the same reason no figure in this system comes from a model. A
chart is a series of numbers wearing a visual claim to precision, and a model
asked for "revenue by quarter" will produce a plausible, smooth, entirely
invented series — and a reader who would interrogate a sentence will not
interrogate a line. So every point plotted here is read out of the CSV that was
extracted from the user's document, and every chart carries the dataset, the
document and the page it came from.

What that costs: charts only exist where a table was extracted. A document of
pure prose produces none, and that is the honest outcome rather than a
decorative one.

Three rules the derivation follows, from `dataviz`:

* **One axis, ever.** Numeric columns are grouped by role — currency with
  currency, percentages with percentages — and each group becomes its own
  chart. Revenue and margin on one plot needs two y-scales, which is the single
  most misleading thing a chart can do.
* **The form follows the data's job.** A period column means change-over-time,
  which is a line. A category column means comparison, which is a bar. A single
  positive measure over few categories is also a composition, which is where a
  donut is legitimate.
* **Colour follows the entity, never its rank**, so the series key — not its
  position — picks the palette slot. That is enforced on the client, which owns
  the palette; this module only emits stable keys for it to map.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal

#: Roles that can be plotted on a value axis, grouped so that two roles never
#: share one. `percentage` is kept apart from `currency` for the obvious
#: reason: a 12% margin and $12m revenue on one axis is a dual-axis chart with
#: the second axis hidden.
VALUE_ROLES = ("currency", "quantity", "percentage")

#: Roles that can label a point.
LABEL_ROLES = ("date", "period", "category")

#: Roles that mean "change over time" and therefore a line.
TIME_ROLES = ("date", "period")

#: The palette has eight validated slots. A ninth series is never a generated
#: colour, so extra series are dropped and the omission is stated.
MAX_SERIES = 8

#: Beyond this many categories a bar chart stops being readable. The largest
#: are kept, because "which are the big ones" is the question a bar chart is
#: usually asked.
MAX_CATEGORIES = 12

#: A donut is only honest for a small number of parts.
MAX_DONUT_SLICES = 6

#: Charts per run. A report with twenty charts has not prioritised anything.
MAX_CHARTS = 6


@dataclass
class ChartSeries:
    #: Stable identity for palette mapping. Derived from the column name, so
    #: the same column keeps the same colour across renders and across runs.
    key: str
    label: str
    points: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "label": self.label, "points": self.points}


@dataclass
class ReportChart:
    chart_id: str
    title: str
    #: "line" | "bar" | "donut"
    kind: str
    x_label: str
    y_label: str
    series: list[ChartSeries]

    # Provenance. A chart without it is an assertion; with it, it is evidence.
    dataset_id: str
    doc_id: str
    source_name: str
    source_page: int | None = None

    unit: str | None = None

    #: Which palette the renderer must use.
    #:
    #: "categorical" means the marks carry *identity* — a region, a quarter, a
    #: column — and take the validated eight-slot series palette. "status"
    #: means they carry *state*: confirmed, contested, refuted, unreviewed.
    #: Those steps are reserved and never reused for "series 4", because a
    #: reader who learns that red means refuted must not meet red meaning
    #: West Region on the next chart. Declared here rather than inferred from
    #: the title, which would be a guess.
    palette: Literal["categorical", "status"] = "categorical"

    caption: str = ""
    #: Extraction problems a reader must see *on the chart*. A footnote in a
    #: different panel does not reach someone looking at a line going up.
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chart_id": self.chart_id,
            "title": self.title,
            "kind": self.kind,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "unit": self.unit,
            "palette": self.palette,
            "caption": self.caption,
            "warnings": self.warnings,
            "dataset_id": self.dataset_id,
            "doc_id": self.doc_id,
            "source_name": self.source_name,
            "source_page": self.source_page,
            "series": [s.to_dict() for s in self.series],
        }


def _humanise(name: str) -> str:
    """Turn an internal table name into something a reader would accept.

    Names arrive from ingestion as identifiers -- `csv_quarterly`,
    `table_p3_1` -- and a chart titled "csv_quarterly" tells the reader about
    our filenames rather than about their data.
    """
    text = name.strip()
    for prefix in ("csv_", "xlsx_", "table_", "sheet_"):
        if text.lower().startswith(prefix):
            text = text[len(prefix) :]
            break
    text = text.replace("_", " ").replace("-", " ").strip()
    if not text:
        return "Extracted table"
    return text[0].upper() + text[1:]


def _slug(text: str) -> str:
    out = "".join(ch if ch.isalnum() else "_" for ch in text.strip().lower())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "series"


def _to_decimal(raw: str) -> Decimal | None:
    """Parse a cell, or decline.

    Deliberately narrow. `write_dataset` already normalised every numeric cell
    to a plain decimal string with the scale applied, so anything that fails
    here is genuinely not a number and must not be coerced into one — a cell
    read as 0 because it said "n/a" is a fabricated data point.
    """
    text = raw.strip().replace(",", "")
    if not text or text in {"-", "--", "n/a", "N/A", "NA", "nil"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    text = text.lstrip("$£€").rstrip("%").strip()
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    return -value if negative else value


def _read_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _warnings_for(dataset: dict[str, Any]) -> list[str]:
    """Extraction problems that change how a chart should be read."""
    notes: list[str] = []
    profile = dataset.get("profile") or {}

    if dataset.get("agreement") is False:
        notes.append(
            "Two extraction strategies disagreed on this table. Confirm these "
            "figures against the source before acting on them."
        )
    for issue in profile.get("issues") or []:
        kind = issue.get("kind")
        if kind == "undeclared_scale":
            notes.append(
                "The source declared no scale for its monetary columns, so "
                "these figures are plotted at face value."
            )
        elif kind == "high_null":
            notes.append(
                f"Column {issue.get('column')!r} is more than half empty; "
                "gaps are omitted rather than filled."
            )
        elif kind == "duplicate_rows":
            notes.append(
                f"{issue.get('count')} duplicate row(s) in the source table "
                "are included as extracted."
            )
    return notes


def _unit_for(dataset: dict[str, Any], role: str) -> str | None:
    if role == "percentage":
        return "%"
    if role == "currency":
        return dataset.get("currency") or None
    return None


def _axis_label(role: str, unit: str | None) -> str:
    base = {"currency": "Amount", "quantity": "Count", "percentage": "Share"}[role]
    return f"{base} ({unit})" if unit else base


def _pick_columns(
    profile: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, list[dict[str, Any]]]]:
    """The label column, and the value columns grouped by role.

    Grouping by role is what keeps a chart to one axis. A table holding
    revenue, cost and margin% yields a currency chart and a percentage chart,
    never one chart with two scales.
    """
    columns = profile.get("columns") or []

    label = None
    for role in (*TIME_ROLES, "category"):
        for column in columns:
            if column.get("candidate_role") == role:
                label = column
                break
        if label is not None:
            break

    grouped: dict[str, list[dict[str, Any]]] = {}
    for column in columns:
        role = column.get("candidate_role")
        if role in VALUE_ROLES:
            grouped.setdefault(role, []).append(column)
    return label, grouped


def _build_series(
    *,
    label_column: dict[str, Any],
    value_columns: list[dict[str, Any]],
    rows: list[list[str]],
) -> list[ChartSeries]:
    label_idx = int(label_column["index"])
    series: list[ChartSeries] = []

    for column in value_columns[:MAX_SERIES]:
        idx = int(column["index"])
        points: list[dict[str, Any]] = []
        for row in rows:
            if idx >= len(row) or label_idx >= len(row):
                continue
            value = _to_decimal(row[idx])
            name = row[label_idx].strip()
            if value is None or not name:
                # A missing cell is a gap, not a zero. Plotting it as zero
                # invents a data point and, on a line, invents a cliff.
                continue
            points.append({"x": name, "y": float(value)})
        if points:
            series.append(
                ChartSeries(
                    key=_slug(str(column.get("name") or f"col_{idx}")),
                    label=str(column.get("name") or f"col_{idx}"),
                    points=points,
                )
            )
    return series


def _trim_categories(series: list[ChartSeries]) -> tuple[list[ChartSeries], int]:
    """Keep the largest categories when there are too many to read."""
    if not series:
        return series, 0
    first = series[0]
    if len(first.points) <= MAX_CATEGORIES:
        return series, 0

    ranked = sorted(first.points, key=lambda p: abs(float(p["y"])), reverse=True)
    keep = {str(p["x"]) for p in ranked[:MAX_CATEGORIES]}
    omitted = len(first.points) - len(keep)
    trimmed = [
        ChartSeries(
            key=s.key,
            label=s.label,
            points=[p for p in s.points if str(p["x"]) in keep],
        )
        for s in series
    ]
    return trimmed, omitted


def charts_for_dataset(
    dataset: dict[str, Any], *, dataset_dir: Path, source_name: str
) -> list[ReportChart]:
    """Every chart one extracted table honestly supports."""
    path = Path(dataset.get("path") or "")
    if not path.is_absolute():
        path = dataset_dir / path.name
    if not path.exists():
        return []

    header, rows = _read_rows(path)
    if not rows or not header:
        return []

    profile = dataset.get("profile") or {}
    label_column, grouped = _pick_columns(profile)
    if label_column is None or not grouped:
        # No way to label a point, or nothing numeric to plot. A table of free
        # text is not a chart, and forcing one would mean inventing an axis.
        return []

    warnings = _warnings_for(dataset)
    is_time = label_column.get("candidate_role") in TIME_ROLES
    dataset_id = str(dataset.get("dataset_id"))
    page = dataset.get("source_page")
    table_name = _humanise(str(dataset.get("name") or "table"))

    out: list[ReportChart] = []

    for role, value_columns in grouped.items():
        series = _build_series(
            label_column=label_column, value_columns=value_columns, rows=rows
        )
        if not series:
            continue

        unit = _unit_for(dataset, role)
        y_label = _axis_label(role, unit)
        x_label = str(label_column.get("name") or "")
        omitted = 0

        if not is_time:
            series, omitted = _trim_categories(series)
            if not series:
                continue

        dropped_series = max(0, len(value_columns) - MAX_SERIES)
        caption_parts = [
            f"{len(series[0].points)} "
            f"{'periods' if is_time else 'categories'} from "
            f"{table_name}" + (f", page {page}" if page else ""),
        ]
        if omitted:
            caption_parts.append(
                f"the {MAX_CATEGORIES} largest shown, {omitted} smaller omitted"
            )
        if dropped_series:
            caption_parts.append(f"{dropped_series} further column(s) not plotted")

        out.append(
            ReportChart(
                chart_id=f"{dataset_id}_{role}",
                title=(f"{table_name}: {y_label.lower()}" if len(grouped) > 1 else table_name),
                kind="line" if is_time else "bar",
                x_label=x_label,
                y_label=y_label,
                unit=unit,
                caption="; ".join(caption_parts) + ".",
                warnings=list(warnings),
                dataset_id=dataset_id,
                doc_id=str(dataset.get("doc_id")),
                source_name=source_name,
                source_page=page,
                series=series,
            )
        )

        # A single positive measure across a few categories is a composition as
        # well as a comparison, and the composition is often the point. Added
        # only when it is genuinely parts-of-a-whole: one series, few slices,
        # nothing negative. A donut of mixed signs is meaningless.
        if (
            not is_time
            and len(series) == 1
            and role != "percentage"
            and 1 < len(series[0].points) <= MAX_DONUT_SLICES
            and all(float(p["y"]) > 0 for p in series[0].points)
            and not omitted
        ):
            out.append(
                ReportChart(
                    chart_id=f"{dataset_id}_{role}_share",
                    title=f"{table_name}: share of total",
                    kind="donut",
                    x_label=x_label,
                    y_label=y_label,
                    unit=unit,
                    caption=(
                        f"Each {x_label.lower() or 'category'} as a share of the "
                        f"total of {series[0].label}, from {table_name}"
                        + (f", page {page}" if page else "")
                        + "."
                    ),
                    warnings=list(warnings),
                    dataset_id=dataset_id,
                    doc_id=str(dataset.get("doc_id")),
                    source_name=source_name,
                    source_page=page,
                    series=series,
                )
            )

    return out


def build_charts(
    *,
    store: Any,
    workspace_id: str,
    corpus_ids: list[str] | None,
    dataset_dir: Path | str,
    max_charts: int = MAX_CHARTS,
) -> list[ReportChart]:
    """The charts this run's corpus supports, broadest coverage first.

    Ordering within a table is mechanical — more data points first, then more
    series — so it is reproducible rather than being whichever table was
    extracted first.

    **Across tables it is a round robin**, and that is the part that matters.
    Ranking every chart by size and taking the top `max_charts` sounds neutral
    and is not: a table with twelve points beats four tables with four points
    each, so a corpus of four uploaded files produced figures from one of them
    and the reader had no way to know the other three existed. Their charts
    were not judged worse; they were never compared.

    So each table contributes its best chart before any table contributes a
    second. A reader who uploads five files sees something from each, which is
    what "the figures for my corpus" means.
    """
    directory = Path(dataset_dir)
    documents = {
        d["doc_id"]: d.get("source_name") or d["doc_id"]
        for d in store.list_documents(workspace_id=workspace_id)
    }

    datasets = store.list_datasets(workspace_id=workspace_id)
    if corpus_ids:
        allowed = set(corpus_ids)
        datasets = [d for d in datasets if d.get("doc_id") in allowed]

    def substance(chart: ReportChart) -> tuple[int, int, str]:
        return (
            -sum(len(s.points) for s in chart.series),
            -len(chart.series),
            chart.chart_id,
        )

    # Grouped by table, each group in its own best-first order.
    by_dataset: list[list[ReportChart]] = []
    for dataset in datasets:
        produced = charts_for_dataset(
            dataset,
            dataset_dir=directory,
            source_name=str(documents.get(dataset.get("doc_id"), "document")),
        )
        if produced:
            by_dataset.append(sorted(produced, key=substance))

    # Tables with the most to show go first, so the strongest figure in the
    # corpus is still the first one the reader meets.
    by_dataset.sort(key=lambda group: substance(group[0]))

    # Then one from each, round and round, until the cap is reached.
    selected: list[ReportChart] = []
    depth = 0
    while len(selected) < max_charts and any(len(g) > depth for g in by_dataset):
        for group in by_dataset:
            if len(selected) >= max_charts:
                break
            if len(group) > depth:
                selected.append(group[depth])
        depth += 1
    return selected


# --------------------------------------------------------------------------- #
# Charts from claims
#
# A research-mode run has no datasets, so `build_charts` produces nothing and
# the report arrives with no figure in it at all. But the run does produce
# grounded numeric claims, each with a value, a unit and a citation that
# resolved — which is the same standard the dataset charts are held to, reached
# by a different route.
#
# So these are charts of *findings* rather than of a table. The distinction is
# kept visible: the caption says so, and each point carries the source it came
# from, because a bar chart of "what four analysts said the market is worth" is
# a useful thing and a misleading thing to mistake for a measurement.
# --------------------------------------------------------------------------- #

#: Fewest comparable claims worth plotting. Two bars is a sentence.
MIN_CLAIMS_FOR_CHART = 3


def charts_from_claims(claims: list[Any], *, max_charts: int = 3) -> list[ReportChart]:
    """Charts derived from grounded numeric claims.

    Grouped by unit, because that is the only grouping that keeps one axis
    honest: a chart mixing "USD bn" and "%" is a dual-axis chart with the
    second axis hidden, and mixing "USD bn" with "GWh" is meaningless.
    """
    by_unit: dict[str, list[Any]] = {}
    for claim in claims:
        if claim.value is None:
            continue
        unit = (claim.unit or "").strip() or "value"
        by_unit.setdefault(unit, []).append(claim)

    out: list[ReportChart] = []
    for unit, group in sorted(by_unit.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(group) < MIN_CLAIMS_FOR_CHART:
            continue

        points: list[dict[str, Any]] = []
        sources: list[str] = []
        for claim in group[:MAX_CATEGORIES]:
            # The label is the finding, shortened. A chart axis cannot carry a
            # sentence, and truncating mid-word is worse than truncating at one.
            label = str(claim.payload.get("figure") or claim.statement).strip()
            if len(label) > 46:
                cut = label[:46]
                space = cut.rfind(" ")
                label = (cut[:space] if space > 20 else cut) + "…"
            points.append({"x": label, "y": float(claim.value)})
            citation = claim.citations[0] if claim.citations else None
            if citation is not None:
                sources.append(
                    str(getattr(citation, "publisher", None) or citation.url or citation.doc_id)
                )

        unique_sources = list(dict.fromkeys(s for s in sources if s))
        out.append(
            ReportChart(
                chart_id=f"claims_{_slug(unit)}",
                title=f"Findings in {unit}",
                kind="bar",
                x_label="Finding",
                y_label=unit,
                unit=None if unit == "value" else unit,
                caption=(
                    f"{len(points)} grounded findings measured in {unit}. Each bar "
                    f"is a figure an agent established and a citation resolved — "
                    f"these are reported findings, not one measured series, so "
                    f"read them as comparable claims rather than a trend."
                ),
                warnings=(
                    [
                        "Sourced from: "
                        + ", ".join(unique_sources[:6])
                        + ("…" if len(unique_sources) > 6 else "")
                    ]
                    if unique_sources
                    else []
                ),
                dataset_id="",
                doc_id="",
                source_name="the run's grounded claims",
                source_page=None,
                series=[ChartSeries(key=_slug(unit), label=unit, points=points)],
            )
        )
        if len(out) >= max_charts:
            break
    return out
