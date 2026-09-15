"""Charts of the analysis, not of the subject.

`charts.py` plots what the numbers said. This plots the run: who contributed,
how much survived review, how confident the claims were, how concentrated the
sources are. All of that existed as claims, verdicts and citations in the store
and had no representation a reader could see.

Two properties are worth holding onto.

**They stay honest about a thin run.** Every builder declines rather than
drawing an encouraging shape over two data points, because a donut of one
slice and a "distribution" of one bar both read as findings.

**The review chart uses the reserved status palette.** Its slices are states,
not identities, and a reader who has learnt that red means *refuted* must not
meet red meaning *West Region* on the next chart.
"""

from __future__ import annotations

from decimal import Decimal

from app.agents.schemas import AgentName, Citation, Claim, Verdict, VerdictKind
from app.reporting.statistics import (
    confidence_distribution,
    evidence_by_source,
    findings_per_agent,
    review_outcomes,
    statistics_charts,
)


def claim(
    agent: AgentName,
    index: int,
    *,
    confidence: float = 0.95,
    publisher: str = "example",
    value: str | None = None,
) -> Claim:
    return Claim(
        claim_id=f"clm_{index}",
        agent=agent,
        statement=f"Finding {index} was established from a cited source.",
        value=Decimal(value) if value is not None else None,
        unit="USD bn" if value is not None else None,
        confidence=confidence,
        confidence_reason=None if confidence >= 0.9 else "thin sourcing",
        citations=[
            Citation(
                url=f"https://{publisher}.com/a",
                publisher=publisher,
                quote="Q" * 30,
            )
        ],
    )


def verdict(claim_id: str, kind: VerdictKind, *, round_: int = 0) -> Verdict:
    return Verdict(
        claim_id=claim_id,
        verdict=kind,
        rationale="independently checked against the cited source",
        # An adverse verdict needs evidence, so supply one.
        independent_value=(Decimal("1") if kind is not VerdictKind.CONFIRMED else None),
        debate_round=round_,
    )


# --------------------------------------------------------------------------- #
# Findings per specialist
# --------------------------------------------------------------------------- #


def test_findings_per_agent_counts_each_specialist() -> None:
    claims = [
        claim(AgentName.MARKET, 1),
        claim(AgentName.MARKET, 2),
        claim(AgentName.RISK, 3),
    ]
    chart = findings_per_agent(claims)
    assert chart is not None
    assert chart.kind == "bar"
    counts = {p["x"]: p["y"] for p in chart.series[0].points}
    assert counts == {"Market": 2.0, "Risk": 1.0}


def test_a_single_contributor_is_not_a_chart() -> None:
    """One bar is a sentence. The report says it in prose instead."""
    assert findings_per_agent([claim(AgentName.RISK, 1)]) is None


def test_an_agent_that_found_nothing_is_absent() -> None:
    """ "All five agents ran" and "all five agents found something" are
    different statements, and only the second one is what this chart shows."""
    chart = findings_per_agent([claim(AgentName.RISK, 1), claim(AgentName.NEWS, 2)])
    assert chart is not None
    labels = {p["x"] for p in chart.series[0].points}
    assert "Finance" not in labels


# --------------------------------------------------------------------------- #
# Review outcome
# --------------------------------------------------------------------------- #


def test_unreviewed_findings_are_counted_not_omitted() -> None:
    """The point of the chart. A run where the Critic was rate-limited leaves
    findings nobody checked, and counting only the reviewed ones would report a
    cleaner result than the run earned."""
    claims = [claim(AgentName.RISK, i) for i in range(1, 5)]
    chart = review_outcomes(claims, [verdict("clm_1", VerdictKind.CONFIRMED)])
    assert chart is not None
    counts = {p["x"]: p["y"] for p in chart.series[0].points}
    assert counts == {"Confirmed": 1.0, "Unreviewed": 3.0}
    assert any("not reviewed" in w for w in chart.warnings)


def test_the_review_chart_uses_the_reserved_status_palette() -> None:
    """Its slices are states, so they must not take a categorical slot."""
    claims = [claim(AgentName.RISK, i) for i in range(1, 4)]
    chart = review_outcomes(claims, [verdict("clm_1", VerdictKind.CONFIRMED)])
    assert chart is not None
    assert chart.palette == "status"


def test_only_the_last_debate_round_counts() -> None:
    """A claim contested in round 0 and confirmed in round 2 is confirmed. The
    conclusion is the last word, not the first."""
    claims = [claim(AgentName.RISK, 1), claim(AgentName.RISK, 2)]
    chart = review_outcomes(
        claims,
        [
            verdict("clm_1", VerdictKind.CONTESTED, round_=0),
            verdict("clm_1", VerdictKind.CONFIRMED, round_=2),
        ],
    )
    assert chart is not None
    counts = {p["x"]: p["y"] for p in chart.series[0].points}
    assert counts.get("Confirmed") == 1.0
    assert "Contested" not in counts


def test_an_all_confirmed_run_draws_nothing() -> None:
    """One slice is not a breakdown."""
    claims = [claim(AgentName.RISK, 1), claim(AgentName.RISK, 2)]
    chart = review_outcomes(
        claims,
        [verdict("clm_1", VerdictKind.CONFIRMED), verdict("clm_2", VerdictKind.CONFIRMED)],
    )
    assert chart is None


# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #


def test_confidence_is_banded() -> None:
    claims = [
        claim(AgentName.RISK, 1, confidence=0.99),
        claim(AgentName.RISK, 2, confidence=0.96),
        claim(AgentName.RISK, 3, confidence=0.85),
        claim(AgentName.RISK, 4, confidence=0.72),
    ]
    chart = confidence_distribution(claims)
    assert chart is not None
    counts = {p["x"]: p["y"] for p in chart.series[0].points}
    assert counts == {"95-100%": 2.0, "80-89%": 1.0, "70-79%": 1.0}


def test_a_run_where_everything_is_the_same_confidence_draws_nothing() -> None:
    """A single bar is not a distribution, and drawing one would let an
    uncalibrated run look like a confident one."""
    claims = [claim(AgentName.RISK, i, confidence=0.95) for i in range(1, 6)]
    assert confidence_distribution(claims) is None


def test_too_few_claims_for_a_distribution() -> None:
    assert confidence_distribution([claim(AgentName.RISK, 1)]) is None


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #


def test_citations_are_counted_per_source() -> None:
    """Concentration is the finding: eight claims citing one page are one claim
    with eight sentences, which is invisible in a list and obvious as a bar."""
    claims = [
        claim(AgentName.RISK, 1, publisher="ibef"),
        claim(AgentName.RISK, 2, publisher="ibef"),
        claim(AgentName.NEWS, 3, publisher="reuters"),
    ]
    chart = evidence_by_source(claims)
    assert chart is not None
    counts = {p["x"]: p["y"] for p in chart.series[0].points}
    assert counts == {"ibef": 2.0, "reuters": 1.0}


def test_a_long_tail_of_sources_is_folded_not_dropped() -> None:
    """The tail is part of the answer to "how concentrated is this", so it is
    summarised rather than discarded."""
    claims = [claim(AgentName.RISK, i, publisher=f"site{i}") for i in range(1, 13)]
    chart = evidence_by_source(claims)
    assert chart is not None
    labels = [p["x"] for p in chart.series[0].points]
    assert any("others" in label for label in labels)
    total = sum(p["y"] for p in chart.series[0].points)
    assert total == 12.0


def test_a_single_source_is_not_a_chart() -> None:
    assert evidence_by_source([claim(AgentName.RISK, 1)]) is None


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def test_a_full_run_produces_every_statistic() -> None:
    claims = [
        claim(AgentName.MARKET, 1, confidence=0.96, publisher="ibef", value="3.71"),
        claim(AgentName.MARKET, 2, confidence=0.95, publisher="imarc", value="3712.2"),
        claim(AgentName.RISK, 3, confidence=0.88, publisher="cornell"),
        claim(AgentName.RISK, 4, confidence=0.95, publisher="ibef"),
        claim(AgentName.NEWS, 5, confidence=0.99, publisher="reuters"),
    ]
    charts = statistics_charts(claims, [verdict("clm_1", VerdictKind.CONFIRMED)])
    titles = {c.title for c in charts}
    assert titles == {
        "Findings per specialist",
        "Review outcome",
        "Stated confidence",
        "Citations per source",
    }
    # Every one of them says where its numbers came from, like every other
    # chart in the system.
    assert all(c.source_name for c in charts)
    assert all(c.caption for c in charts)


def test_a_thin_run_produces_none_rather_than_four_misleading_ones() -> None:
    assert statistics_charts([claim(AgentName.RISK, 1)], []) == []


def test_no_claims_produces_no_statistics() -> None:
    assert statistics_charts([], []) == []


def test_statistics_charts_carry_no_dataset_provenance() -> None:
    """They are counted from the run rather than read from a document, and the
    renderer keys its provenance line off exactly this: claiming "plotted from
    the extracted table" would be a false provenance on a page whose whole
    purpose is making provenance checkable."""
    claims = [claim(AgentName.RISK, i) for i in range(1, 4)]
    for chart in statistics_charts(claims, [verdict("clm_1", VerdictKind.CONFIRMED)]):
        assert chart.dataset_id == ""
        assert chart.doc_id == ""


# --------------------------------------------------------------------------- #
# The document carries them too
# --------------------------------------------------------------------------- #


def test_the_document_renders_each_chart_as_a_table() -> None:
    """Markdown cannot hold a plot, and a downloaded report that silently drops
    every figure is a different document from the one on screen.

    This also caught a real bug: the section joined a variable left over from
    an earlier block, so it rendered 180 characters of the wrong content while
    looking present. mypy's `no-redef` warning was pointing straight at it.
    """
    from app.agents.schemas import ExecutiveReport, Insight
    from app.reporting.document import build_document

    claims = [
        claim(AgentName.MARKET, 1, publisher="ibef", value="3.71"),
        claim(AgentName.MARKET, 2, publisher="imarc", value="54.41"),
        claim(AgentName.RISK, 3, confidence=0.88, publisher="cornell"),
        claim(AgentName.NEWS, 4, confidence=0.99, publisher="reuters"),
    ]
    verdicts = [verdict("clm_1", VerdictKind.CONFIRMED)]
    charts = [c.to_dict() for c in statistics_charts(claims, verdicts)]
    assert charts, "the fixture must produce charts for this to test anything"

    report = ExecutiveReport(
        question="q",
        executive_summary=(
            "A summary long enough to clear the schema floor, which exists to "
            "rule out a one-line report rather than to force padding here."
        ),
        insights=[
            Insight(
                headline="A market headline",
                so_what=(
                    "An implication stated at enough length to clear the "
                    "ninety character floor that rules out a restated headline."
                ),
                confidence=0.95,
                suggested_owner="CEO",
                effort="M",
            )
        ],
        charts=charts,
    )
    document = build_document(question="q", report=report, claims=claims, verdicts=verdicts)

    section = next((s for s in document.sections if s.slug == "figures"), None)
    assert section is not None, "no figures section"
    # Every chart's title, and its numbers as a table.
    for chart in charts:
        assert chart["title"] in section.body
    assert "| Specialist | Findings |" in section.body
    # And the caveat travels with it, not into a footnote elsewhere.
    assert "not reviewed" in section.body


def test_the_figures_section_states_its_own_provenance() -> None:
    """These are counted from the run rather than read from a document, and the
    report has to say which — it is the same claim the renderer makes."""
    from app.agents.schemas import ExecutiveReport, Insight
    from app.reporting.document import build_document

    claims = [claim(AgentName.RISK, i, publisher=f"s{i}") for i in range(1, 4)]
    charts = [
        c.to_dict()
        for c in statistics_charts(claims, [verdict("clm_1", VerdictKind.CONFIRMED)])
    ]
    report = ExecutiveReport(
        question="q",
        executive_summary=(
            "A summary long enough to clear the schema floor, which exists to "
            "rule out a one-line report rather than to force padding here."
        ),
        insights=[
            Insight(
                headline="A headline",
                so_what=(
                    "An implication stated at enough length to clear the "
                    "ninety character floor that rules out a restated headline."
                ),
                confidence=0.95,
                suggested_owner="CEO",
                effort="M",
            )
        ],
        charts=charts,
    )
    document = build_document(question="q", report=report, claims=claims, verdicts=[])
    section = next(s for s in document.sections if s.slug == "figures")
    assert "this run's" in section.body
