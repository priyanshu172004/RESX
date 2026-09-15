"""Recommendations: what to do about the findings, held to the same standard.

An insight says what is true; a recommendation says what to do about it. They
fail differently -- an insight is wrong when it misreads the evidence, a
recommendation is wrong when it does not follow from evidence that is perfectly
correct -- so the second gets its own gate rather than riding on the first.

The gate is `supporting_claim_ids` checked against the run's accepted claims.
Without it, "diversify into adjacent markets" is a sentence a model will
produce indefinitely, for any input, and nobody can act on.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from app.agents.schemas import (
    AgentName,
    Citation,
    Claim,
    ExecutiveReport,
    ExecutiveReportDraft,
    InsightDraft,
    Recommendation,
)

#: Long enough to clear the schema's floor, which exists to rule out a
#: one-line report rather than to force padding.
SUMMARY = (
    "Supplier concentration is the largest single exposure the analysis "
    "found, with three suppliers carrying the majority of spend."
)


def a_recommendation(**overrides: Any) -> Recommendation:
    base: dict[str, Any] = {
        "action": "Renegotiate the top three supplier contracts before Q3 renewal",
        "kind": "mitigate",
        "rationale": "Supplier concentration was found to be the largest single "
        "exposure in the analysis.",
        "metric": "Share of spend with the top three suppliers",
        "horizon": "quarter",
        "owner": "Procurement",
        "effort": "M",
        "supporting_claim_ids": ["clm_1"],
    }
    base.update(overrides)
    return Recommendation(**base)


# --------------------------------------------------------------------------- #
# The shape
# --------------------------------------------------------------------------- #


def test_a_recommendation_must_cite_at_least_one_claim() -> None:
    """Ungrounded advice is the failure mode. An empty list is not a claim of
    "no evidence needed", it is advice with nothing behind it."""
    with pytest.raises(ValidationError):
        a_recommendation(supporting_claim_ids=[])


def test_a_recommendation_must_carry_a_metric() -> None:
    """Advice with no observable cannot be reviewed later, which makes it an
    opinion rather than a plan."""
    with pytest.raises(ValidationError):
        a_recommendation(metric="")


def test_an_action_must_be_substantial_enough_to_act_on() -> None:
    with pytest.raises(ValidationError):
        a_recommendation(action="Do better")


def test_the_kind_is_constrained_to_the_four_that_mean_something() -> None:
    for kind in ("mitigate", "grow", "monitor", "investigate"):
        assert a_recommendation(kind=kind).kind == kind
    with pytest.raises(ValidationError):
        a_recommendation(kind="synergise")


def test_expected_effect_is_free_text_because_a_figure_would_be_a_new_number() -> None:
    """The Synthesizer may not introduce numbers no agent produced, so this
    field cannot be numeric without contradicting that rule."""
    rec = a_recommendation(expected_effect="Materially lower renewal exposure")
    assert isinstance(rec.expected_effect, str)
    assert a_recommendation().expected_effect is None


# --------------------------------------------------------------------------- #
# The ordering is mechanical, like the insight ranking
# --------------------------------------------------------------------------- #


def test_recommendations_are_ordered_soonest_then_cheapest() -> None:
    """Reproducible and arguable, rather than whichever the model wrote first."""
    now_large = a_recommendation(horizon="now", effort="L")
    now_small = a_recommendation(horizon="now", effort="S")
    year_small = a_recommendation(horizon="year", effort="S")
    quarter_small = a_recommendation(horizon="quarter", effort="S")

    report = ExecutiveReport(
        question="q",
        executive_summary=SUMMARY,
        insights=[],
        recommendations=[year_small, now_large, quarter_small, now_small],
    )
    ordered = report.ranked_recommendations()
    assert [r.horizon for r in ordered] == ["now", "now", "quarter", "year"]
    # Within the same horizon, the cheaper one comes first.
    assert ordered[0].effort == "S"
    assert ordered[1].effort == "L"


def test_the_report_caps_the_list_because_twenty_next_steps_is_not_a_plan() -> None:
    with pytest.raises(ValidationError):
        ExecutiveReportDraft(
            question="q",
            executive_summary=SUMMARY,
            insights=[],
            recommendations=[a_recommendation() for _ in range(9)],
        )


def test_a_report_with_no_recommendations_is_valid() -> None:
    """ "The evidence carries nothing here" must remain expressible."""
    assert (
        ExecutiveReportDraft(
            question="q", executive_summary=SUMMARY, insights=[]
        ).recommendations
        == []
    )


# --------------------------------------------------------------------------- #
# The gate: advice citing a claim the run never produced
# --------------------------------------------------------------------------- #


def _claim(claim_id: str) -> Claim:
    return Claim(
        claim_id=claim_id,
        agent=AgentName.RISK,
        statement="Supplier concentration is the largest single exposure.",
        confidence=0.95,
        citations=[Citation(url="https://example.com/a", quote="A" * 20)],
    )


def _report_with(*recs: Recommendation) -> ExecutiveReportDraft:
    return ExecutiveReportDraft(
        question="q",
        executive_summary=(
            "Supplier concentration is the largest single exposure the "
            "analysis found. Three suppliers carry the majority of spend, and "
            "one failing would halt production for weeks."
        ),
        insights=[
            InsightDraft(
                headline="Supplier concentration is the largest exposure",
                so_what=(
                    "Three suppliers carry the majority of spend, so one "
                    "failing would stop production for weeks with no "
                    "qualified alternative in place."
                ),
                confidence=0.95,
                suggested_owner="Procurement",
                effort="M",
            )
        ],
        recommendations=list(recs),
    )


def test_a_recommendation_citing_an_unknown_claim_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point. A claim_id that is not in the run means the
    justification was invented, which is the same failure as a quote that
    resolves to nothing."""
    from app.graph import nodes

    good = a_recommendation(supporting_claim_ids=["clm_real"])
    bad = a_recommendation(
        action="Enter three adjacent markets within the next two quarters",
        kind="grow",
        supporting_claim_ids=["clm_invented"],
    )
    report = _report_with(good, bad)
    events: list[tuple[str, dict[str, Any]]] = []
    state = _run_state([_claim("clm_real")])
    deps = _stub_deps(monkeypatch, nodes, report, events)

    result = nodes.synthesize_node(state, deps)

    kept = result["report"].recommendations
    assert [r.supporting_claim_ids for r in kept] == [["clm_real"]]
    assert any(k == "recommendation_dropped" for k, _ in events)


def test_dropping_a_recommendation_is_disclosed_in_the_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reader given four recommendations has no way to know two were
    removed, so the report has to say it."""
    from app.graph import nodes

    report = _report_with(a_recommendation(supporting_claim_ids=["clm_nope"]))
    events: list[tuple[str, dict[str, Any]]] = []
    state = _run_state([_claim("clm_real")])
    deps = _stub_deps(monkeypatch, nodes, report, events)

    result = nodes.synthesize_node(state, deps)

    assert result["report"].recommendations == []
    assert any(
        "recommendation" in lim and "dropped" in lim for lim in result["report"].limitations
    )


def test_recommendations_all_grounded_are_all_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.graph import nodes

    report = _report_with(
        a_recommendation(supporting_claim_ids=["clm_real"]),
        a_recommendation(
            action="Track supplier concentration monthly against a 40% ceiling",
            kind="monitor",
            supporting_claim_ids=["clm_real", "clm_other"],
        ),
    )
    events: list[tuple[str, dict[str, Any]]] = []
    state = _run_state([_claim("clm_real"), _claim("clm_other")])
    deps = _stub_deps(monkeypatch, nodes, report, events)

    result = nodes.synthesize_node(state, deps)

    assert len(result["report"].recommendations) == 2
    assert not any(
        "recommendation" in lim and "dropped" in lim for lim in result["report"].limitations
    )


# --------------------------------------------------------------------------- #
# Fixtures for driving synthesize_node without a model
# --------------------------------------------------------------------------- #


def _run_state(claims: list[Claim]) -> dict[str, Any]:
    from app.agents.schemas import Budget

    return {
        "run_id": "run_test",
        "workspace_id": "ws_test",
        "question": "What is our largest exposure?",
        "corpus_ids": ["doc_1"],
        "research_mode": False,
        "claims": claims,
        "verdicts": [],
        "dropped_claims": [],
        "computations": [],
        "degraded": [],
        "errors": [],
        "injection_attempts": [],
        "budget": Budget(),
        "spend": [],
    }


def _stub_deps(
    monkeypatch: pytest.MonkeyPatch,
    nodes: Any,
    report: ExecutiveReportDraft,
    events: list[tuple[str, dict[str, Any]]],
) -> Any:
    """A GraphDeps whose model returns exactly the report under test."""
    from app.agents.llm import LLMResult, LLMUsage

    class _LLM:
        def structured(self, **kwargs: Any) -> LLMResult:
            return LLMResult(
                text="{}",
                parsed=report,
                usage=LLMUsage(model="stub", provider="stub"),
                stop_reason="stop",
            )

    class _Store:
        """Charts are derived from extracted tables; this run has none."""

        @staticmethod
        def list_documents(**_kwargs: Any) -> list[dict[str, Any]]:
            return []

        @staticmethod
        def list_datasets(**_kwargs: Any) -> list[dict[str, Any]]:
            return []

    class _Deps:
        llm = _LLM()
        store = _Store()
        embedder = None
        dataset_dir = "./storage/datasets"

        def event(self, kind: str, payload: dict[str, Any]) -> None:
            events.append((kind, payload))

    monkeypatch.setattr(nodes, "is_exhausted", lambda _state: False)
    return _Deps()
