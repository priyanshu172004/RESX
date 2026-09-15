"""Assembling the final report from the model's draft.

The Synthesizer used to be asked for `ExecutiveReport` directly, citation
objects included, and it transcribed them badly: it copied a claim's quote and
doc_id but dropped the page. `Citation` rejects that — correctly, because a
corpus citation with no page cannot be resolved — so the whole report failed
validation twice and the run died with everything else about it correct.

It is now asked for `ExecutiveReportDraft`, which carries claim ids and no
citation objects at all, and the citations are attached here from the claims
themselves. Re-transcribing a verified citation through a language model can
only lose or corrupt fields, so it is not attempted.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.agents.llm import LLMResult, LLMUsage
from app.agents.schemas import (
    AgentName,
    Budget,
    Citation,
    Claim,
    ExecutiveReportDraft,
    InsightDraft,
    SwotDraft,
    SwotItemDraft,
)

WORKSPACE = "ws_assembly"

SUMMARY = (
    "Revenue rose across 2024 while gross margin moved with it, and the "
    "regional split concentrates a majority of revenue in one territory."
)


def corpus_claim(claim_id: str, page: int = 3) -> Claim:
    """A claim citing an uploaded document — the case that used to break."""
    return Claim(
        claim_id=claim_id,
        agent=AgentName.FINANCE,
        statement="Revenue for the period was reported in the quarterly table.",
        confidence=0.95,
        citations=[
            Citation(
                doc_id="doc_quarterly",
                page=page,
                char_start=0,
                char_end=60,
                quote="Q1 2024 | 1200000 | 780000 | 35.0",
                resolution="ok",
            )
        ],
    )


def a_draft(**overrides: Any) -> ExecutiveReportDraft:
    base: dict[str, Any] = {
        "question": "How did revenue develop?",
        "executive_summary": SUMMARY,
        "insights": [
            InsightDraft(
                headline="Revenue rose through 2024 while margin held",
                so_what=(
                    "Revenue climbed across the four quarters and gross margin "
                    "moved with it rather than against it, so the growth was "
                    "not bought with discounting."
                ),
                confidence=0.95,
                suggested_owner="CFO",
                effort="M",
                supporting_claim_ids=["clm_a"],
            )
        ],
        "swot": SwotDraft(
            strengths=[
                SwotItemDraft(
                    text="Margin improved alongside revenue growth.",
                    supporting_claim_ids=["clm_a"],
                )
            ]
        ),
    }
    base.update(overrides)
    return ExecutiveReportDraft(**base)


def run_synthesis(
    monkeypatch: pytest.MonkeyPatch,
    draft: ExecutiveReportDraft,
    claims: list[Claim],
) -> Any:
    from app.graph import nodes

    class _LLM:
        def structured(self, **_kwargs: Any) -> LLMResult:
            return LLMResult(
                text="{}",
                parsed=draft,
                usage=LLMUsage(model="stub", provider="stub"),
                stop_reason="stop",
            )

    class _Store:
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
            return None

    monkeypatch.setattr(nodes, "is_exhausted", lambda _state: False)
    state = {
        "run_id": "r",
        "workspace_id": WORKSPACE,
        "question": "How did revenue develop?",
        "corpus_ids": ["doc_quarterly"],
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
    return nodes.synthesize_node(state, _Deps())["report"]


def test_the_model_is_never_asked_for_a_citation_object() -> None:
    """The structural fix. It cannot mis-transcribe a field it is not given."""
    schema = ExecutiveReportDraft.model_json_schema()
    assert "Citation" not in str(schema.get("$defs", {}).keys())
    assert "evidence" not in InsightDraft.model_fields
    assert "citations" not in SwotItemDraft.model_fields


def test_an_insights_evidence_comes_from_the_claim_it_cites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = corpus_claim("clm_a", page=3)
    report = run_synthesis(monkeypatch, a_draft(), [claim])

    evidence = report.insights[0].evidence
    assert len(evidence) == 1
    # Identical object, not a copy the model retyped: same page, same quote.
    assert evidence[0].page == 3
    assert evidence[0].quote == claim.citations[0].quote
    assert evidence[0].citation_id == claim.citations[0].citation_id


def test_a_corpus_citation_keeps_its_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact failure: the page was dropped in transcription and the report
    was rejected by `Citation`'s validator."""
    report = run_synthesis(monkeypatch, a_draft(), [corpus_claim("clm_a", page=7)])
    assert report.insights[0].evidence[0].page == 7


def test_swot_items_are_cited_too(monkeypatch: pytest.MonkeyPatch) -> None:
    report = run_synthesis(monkeypatch, a_draft(), [corpus_claim("clm_a")])
    strength = report.swot.strengths[0]
    assert strength.citations
    assert strength.citations[0].doc_id == "doc_quarterly"


def test_an_insight_citing_an_unknown_claim_gets_no_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It must not invent one, and must not crash the report either. An
    insight with no evidence is visible for what it is."""
    draft = a_draft(
        insights=[
            InsightDraft(
                headline="A finding with no basis in this run",
                so_what=(
                    "This insight references a claim identifier that this run "
                    "never produced, so there is nothing behind it at all."
                ),
                confidence=0.95,
                suggested_owner="CFO",
                effort="S",
                supporting_claim_ids=["clm_invented"],
            )
        ]
    )
    report = run_synthesis(monkeypatch, draft, [corpus_claim("clm_a")])
    assert report.insights[0].evidence == []


def test_duplicate_claim_ids_do_not_duplicate_the_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two references to one claim are one piece of evidence, not two. Showing
    it twice reads as corroboration, which is what citations exist to
    establish."""
    draft = a_draft(
        insights=[
            InsightDraft(
                headline="One claim referenced twice over",
                so_what=(
                    "The same claim identifier appears twice in the supporting "
                    "list, which must not present one source as two."
                ),
                confidence=0.95,
                suggested_owner="CFO",
                effort="S",
                supporting_claim_ids=["clm_a", "clm_a"],
            )
        ]
    )
    report = run_synthesis(monkeypatch, draft, [corpus_claim("clm_a")])
    assert len(report.insights[0].evidence) == 1


def test_the_summary_and_the_rest_survive_assembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assembly must carry every field through, not just the cited ones."""
    draft = a_draft(
        financial_summary="Revenue 1.2m in Q1 rising to 1.52m in Q4.",
        risk_register=["Revenue concentrated in one region"],
    )
    report = run_synthesis(monkeypatch, draft, [corpus_claim("clm_a")])
    assert report.executive_summary == SUMMARY
    assert report.financial_summary.startswith("Revenue 1.2m")
    assert report.risk_register == ["Revenue concentrated in one region"]
    assert report.question == "How did revenue develop?"
