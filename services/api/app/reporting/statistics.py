"""Charts of the analysis itself.

`charts.py` plots the *subject*: revenue by quarter, findings in USD bn. This
plots the *run* — how much each specialist contributed, how much of it survived
review, how confident the claims were, and where the evidence came from.

Both matter and they answer different questions. "What did the numbers say" is
the subject; "how much of this can I trust, and who established it" is the
analysis. The second was entirely invisible: it existed as claims, verdicts and
citations in the store and had no representation anywhere a reader could see.

Every series here is counted from the run's own record, so there is nothing to
fabricate and no model call to make. That also means these charts are honest
about a thin run: three claims and no verdicts produces a chart that says so,
rather than an encouraging shape.

One rule is load-bearing. The review-outcome chart encodes **state**, not
identity, so it uses the reserved status palette rather than the categorical
one — a confirmed finding is not "series 1". And every slice carries its label,
because a reader must never have to distinguish "refuted" from "confirmed" by
colour alone.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from app.agents.catalog import AGENT_PROFILES
from app.reporting.charts import ChartSeries, ReportChart

#: Confidence buckets. Coarse on purpose: the interesting question is "how much
#: of this is the model hedging" and finer bins invite reading noise as signal.
CONFIDENCE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("<70%", 0.0, 0.70),
    ("70-79%", 0.70, 0.80),
    ("80-89%", 0.80, 0.90),
    ("90-94%", 0.90, 0.95),
    ("95-100%", 0.95, 1.01),
)

#: Beyond this many sources a bar chart of them is unreadable. The largest are
#: kept and the remainder folded into one bar rather than dropped, because
#: "how concentrated is the evidence" is the question being asked and silently
#: discarding the tail would answer it wrongly.
MAX_SOURCES = 8


def _agent_title(agent: Any) -> str:
    profile = AGENT_PROFILES.get(agent) or {}
    return str(profile.get("title") or getattr(agent, "value", str(agent)).title())


def findings_per_agent(claims: list[Any]) -> ReportChart | None:
    """How much each specialist actually contributed.

    Worth seeing because "all five agents ran" and "all five agents found
    something" are different statements, and only the second one shows up here.
    An agent that ran and produced nothing is absent from this chart, which is
    the honest representation of that.
    """
    counts = Counter(claim.agent for claim in claims)
    if len(counts) < 2:
        # One contributor is a sentence, not a chart.
        return None

    points = [
        {"x": _agent_title(agent), "y": float(count)}
        for agent, count in sorted(counts.items(), key=lambda kv: -kv[1])
    ]
    return ReportChart(
        chart_id="stats_per_agent",
        title="Findings per specialist",
        kind="bar",
        x_label="Specialist",
        y_label="Findings",
        unit=None,
        caption=(
            f"{sum(counts.values())} grounded findings across "
            f"{len(counts)} specialists. A specialist that ran but established "
            f"nothing does not appear here."
        ),
        dataset_id="",
        doc_id="",
        source_name="this run's grounded claims",
        source_page=None,
        series=[ChartSeries(key="findings", label="Findings", points=points)],
    )


def review_outcomes(claims: list[Any], verdicts: list[Any]) -> ReportChart | None:
    """How much of the analysis survived review, including what went unreviewed.

    "Unreviewed" is a category rather than an omission, and showing it is the
    point. A run where the Critic was rate-limited produces findings nobody
    checked, and a chart that quietly counted only the reviewed ones would
    report a cleaner result than the run earned.
    """
    if not claims:
        return None

    latest: dict[str, Any] = {}
    for verdict in verdicts:
        current = latest.get(verdict.claim_id)
        if current is None or verdict.debate_round >= current.debate_round:
            latest[verdict.claim_id] = verdict

    tally = Counter[str]()
    for claim in claims:
        found = latest.get(claim.claim_id)
        tally[found.verdict.value if found is not None else "unreviewed"] += 1

    # Fixed order, worst last, so the same run always reads the same way and
    # the eye lands on the problems.
    order = ["confirmed", "unreviewed", "contested", "refuted"]
    points = [{"x": name.title(), "y": float(tally[name])} for name in order if tally[name]]
    if len(points) < 2:
        return None

    reviewed = sum(v for k, v in tally.items() if k != "unreviewed")
    return ReportChart(
        chart_id="stats_review",
        title="Review outcome",
        kind="donut",
        x_label="Outcome",
        y_label="Findings",
        unit=None,
        # `status` rather than the categorical palette: these are states, and a
        # refuted finding is not "series 4". The renderer reads this.
        palette="status",
        caption=(
            f"{reviewed} of {len(claims)} findings were independently reviewed. "
            f"Anything unreviewed was not checked by the Critic — usually "
            f"because the run hit a rate limit — and should be treated as "
            f"unverified rather than as passed."
        ),
        warnings=(
            [
                f"{tally['unreviewed']} finding(s) were not reviewed. They are "
                f"grounded in a source but nothing independently checked them."
            ]
            if tally["unreviewed"]
            else []
        ),
        dataset_id="",
        doc_id="",
        source_name="this run's verdicts",
        source_page=None,
        series=[ChartSeries(key="outcome", label="Findings", points=points)],
    )


def confidence_distribution(claims: list[Any]) -> ReportChart | None:
    """Where the stated confidence sits.

    A run where everything is 95%+ is not a confident run, it is an
    uncalibrated one — the shared rules require a reason for anything below
    0.9, which makes the low end the honest end. Seeing the shape is the only
    way to notice that every claim arrived at the same number.
    """
    if len(claims) < 3:
        return None

    tally = Counter[str]()
    for claim in claims:
        for label, low, high in CONFIDENCE_BANDS:
            if low <= float(claim.confidence) < high:
                tally[label] += 1
                break

    points = [
        {"x": label, "y": float(tally[label])}
        for label, _, _ in CONFIDENCE_BANDS
        if tally[label]
    ]
    if len(points) < 2:
        # Every claim in one band. Stated in the report's prose instead of
        # drawn as a single bar pretending to be a distribution.
        return None

    return ReportChart(
        chart_id="stats_confidence",
        title="Stated confidence",
        kind="bar",
        x_label="Confidence",
        y_label="Findings",
        unit=None,
        caption=(
            "How confident each finding claims to be. A distribution clustered "
            "at the top is a calibration problem rather than a strong result — "
            "any claim below 90% has to state its reason, so the low end is "
            "the examined end."
        ),
        dataset_id="",
        doc_id="",
        source_name="this run's grounded claims",
        source_page=None,
        series=[ChartSeries(key="findings", label="Findings", points=points)],
    )


def evidence_by_source(claims: list[Any]) -> ReportChart | None:
    """Which sources the conclusions actually rest on.

    Concentration is the finding here. Eight findings that all cite one page
    are one finding with eight sentences, and that is invisible in a list of
    claims but obvious as a bar.
    """
    tally = Counter[str]()
    for claim in claims:
        for citation in claim.citations:
            name = (
                getattr(citation, "publisher", None)
                or getattr(citation, "url", None)
                or getattr(citation, "doc_id", None)
            )
            if name:
                tally[str(name)[:44]] += 1

    if len(tally) < 2:
        return None

    ranked = tally.most_common()
    points = [{"x": name, "y": float(count)} for name, count in ranked[:MAX_SOURCES]]
    folded = ranked[MAX_SOURCES:]
    if folded:
        # Folded rather than dropped: the tail is part of the answer to "how
        # concentrated is this".
        points.append({"x": f"{len(folded)} others", "y": float(sum(c for _, c in folded))})

    return ReportChart(
        chart_id="stats_sources",
        title="Citations per source",
        kind="bar",
        x_label="Source",
        y_label="Citations",
        unit=None,
        caption=(
            f"{sum(tally.values())} citations across {len(tally)} distinct "
            f"sources. Findings concentrated on a single source are more "
            f"fragile than their count suggests."
        ),
        dataset_id="",
        doc_id="",
        source_name="this run's citations",
        source_page=None,
        series=[ChartSeries(key="citations", label="Citations", points=points)],
    )


def statistics_charts(claims: list[Any], verdicts: list[Any]) -> list[ReportChart]:
    """Every statistic chart this run supports.

    Each builder returns None when the run is too thin to draw it honestly, so
    a two-claim run gets no charts rather than four misleading ones.
    """
    built = [
        findings_per_agent(claims),
        review_outcomes(claims, verdicts),
        confidence_distribution(claims),
        evidence_by_source(claims),
    ]
    return [chart for chart in built if chart is not None]
