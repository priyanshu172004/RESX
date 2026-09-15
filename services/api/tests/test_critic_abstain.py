"""The Critic's error handler.

Found on a live run against Atlas. The Critic returned `contested` with no
contradicting citation and no independently derived value. The `Verdict` schema
forbids that — an adverse verdict with no evidence is just an assertion.

The node caught the `ValidationError` and then, inside the handler, built

    Verdict(verdict=CONTESTED, contradicting_citations=[], independent_value=None)

which breaks the identical rule. The handler raised the exception it was
handling, escaped `critic_node`, and killed the entire graph — so one unusable
verdict turned into a failed run that produced nothing at all. The user saw
"status: failed" and no report.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from app.agents.llm import LLMResult, LLMUsage
from app.agents.schemas import (
    AgentName,
    Citation,
    Claim,
    Verdict,
    VerdictKind,
)
from app.analysis.sandbox import LocalSubprocessSandbox
from app.graph.nodes import CriticBatch, CriticOutput, critic_node
from app.graph.state import GraphDeps, initial_state
from app.rag.embeddings import HashingEmbedder
from app.store.local import LocalStore

WORKSPACE = "ws_critic"


class ScriptedLLM:
    """Returns one prepared `CriticBatch`, whatever it is asked."""

    provider = "test"

    def __init__(self, batch: CriticBatch) -> None:
        self.batch = batch

    def structured(self, **_: Any) -> LLMResult:
        return LLMResult(
            text="",
            parsed=self.batch,
            usage=LLMUsage(input_tokens=10, output_tokens=5, model="test"),
            stop_reason="stop",
        )

    def text(self, **_: Any) -> LLMResult:  # pragma: no cover - unused here
        return LLMResult(text="", usage=LLMUsage(model="test"))


@pytest.fixture
def scene(tmp_path: Path):
    """A store with one claim, and the deps a node needs."""
    store = LocalStore(tmp_path / "resx.db")
    events: list[tuple[str, dict[str, Any]]] = []

    claim = Claim(
        claim_id="clm_target",
        agent=AgentName.FINANCE,
        statement="Total revenue was 1,765,000.",
        value=Decimal("1765000"),
        computation_id="cmp_x",
        confidence=0.95,
        citations=[Citation(doc_id="doc_x", page=1, quote="Revenue | 450000 | 367000")],
    )

    def make(batch: CriticBatch):
        deps = GraphDeps(
            llm=ScriptedLLM(batch),
            store=store,
            embedder=HashingEmbedder(dimensions=64),
            sandbox=LocalSubprocessSandbox(wall_timeout_seconds=10),
            dataset_dir=str(tmp_path / "datasets"),
            tavily_api_key="",
            egress_allowlist=frozenset(),
            top_k=4,
            max_debate_rounds=3,
            emit=lambda kind, payload: events.append((kind, payload)),
        )
        state = initial_state(
            run_id="run_critic",
            workspace_id=WORKSPACE,
            question="What is total revenue?",
            corpus_ids=["doc_x"],
        )
        state["claims"] = [claim]
        return deps, state

    yield make, events, claim
    store.close()


# --------------------------------------------------------------------------- #
# The crash
# --------------------------------------------------------------------------- #


def test_an_unevidenced_contested_verdict_does_not_crash_the_run(scene) -> None:
    """The bug, exactly as it happened."""
    make, _events, _ = scene
    deps, state = make(
        CriticBatch(
            verdicts=[
                CriticOutput(
                    claim_id="clm_target",
                    verdict=VerdictKind.CONTESTED,
                    rationale="I am not convinced by this figure at all.",
                    # No contradiction, no independent value: the schema will
                    # reject the Verdict built from this.
                )
            ]
        )
    )

    # Must not raise. Before the fix this propagated a ValidationError out of
    # the node and LangGraph failed the whole run.
    out = critic_node(state, deps)
    assert isinstance(out, dict)


def test_the_unevidenced_verdict_is_not_recorded(scene) -> None:
    """A verdict the schema rejects is not a verdict."""
    make, _, _ = scene
    deps, state = make(
        CriticBatch(
            verdicts=[
                CriticOutput(
                    claim_id="clm_target",
                    verdict=VerdictKind.CONTESTED,
                    rationale="I am not convinced by this figure at all.",
                )
            ]
        )
    )
    out = critic_node(state, deps)
    assert out["verdicts"] == []


def test_it_is_never_silently_upgraded_to_confirmed(scene) -> None:
    """The worst possible direction to fail in.

    An adverse judgement becoming a favourable one would let the Critic's
    doubt read as its approval.
    """
    make, _, _ = scene
    deps, state = make(
        CriticBatch(
            verdicts=[
                CriticOutput(
                    claim_id="clm_target",
                    verdict=VerdictKind.REFUTED,
                    rationale="This number looks wrong to me, on reflection.",
                )
            ]
        )
    )
    out = critic_node(state, deps)
    kinds = [v.verdict for v in out["verdicts"]]
    assert VerdictKind.CONFIRMED not in kinds
    assert out["verdicts"] == []


def test_the_abstention_is_reported_on_the_run(scene) -> None:
    """A reader has to be able to tell an unreviewed claim from a confirmed one."""
    make, events, _ = scene
    deps, state = make(
        CriticBatch(
            verdicts=[
                CriticOutput(
                    claim_id="clm_target",
                    verdict=VerdictKind.CONTESTED,
                    rationale="I am not convinced by this figure at all.",
                )
            ]
        )
    )
    out = critic_node(state, deps)

    assert out.get("errors"), "an abstention must reach the run record"
    assert "unreviewed" in out["errors"][0]["error"]

    kinds = [kind for kind, _ in events]
    assert "critic_abstained" in kinds

    payload = next(p for k, p in events if k == "critic_abstained")
    assert payload["claim_id"] == "clm_target"
    assert payload["attempted"] == "contested"
    assert payload["reason"], "the reason must say why it was rejected"


# --------------------------------------------------------------------------- #
# The valid paths still work
# --------------------------------------------------------------------------- #


def test_a_confirmed_verdict_needs_no_evidence(scene) -> None:
    """Confirming a claim asserts nothing new, so it carries no evidence burden."""
    make, _, _ = scene
    deps, state = make(
        CriticBatch(
            verdicts=[
                CriticOutput(
                    claim_id="clm_target",
                    verdict=VerdictKind.CONFIRMED,
                    rationale="Re-derived independently and it matches exactly.",
                )
            ]
        )
    )
    out = critic_node(state, deps)
    assert len(out["verdicts"]) == 1
    assert out["verdicts"][0].verdict is VerdictKind.CONFIRMED
    assert not out.get("errors")


def test_a_contested_verdict_with_an_independent_value_is_kept(scene) -> None:
    """Evidence can be the Critic's own re-derivation, not only a quote."""
    make, _, _ = scene
    deps, state = make(
        CriticBatch(
            verdicts=[
                CriticOutput(
                    claim_id="clm_target",
                    verdict=VerdictKind.CONTESTED,
                    independent_value="1755000",
                    rationale="My own re-derivation gives a different total.",
                )
            ]
        )
    )
    out = critic_node(state, deps)
    assert len(out["verdicts"]) == 1
    assert out["verdicts"][0].independent_value == Decimal("1755000")


def test_a_verdict_on_an_unknown_claim_is_discarded(scene) -> None:
    """A verdict about a claim that does not exist cannot be about anything."""
    make, _, _ = scene
    deps, state = make(
        CriticBatch(
            verdicts=[
                CriticOutput(
                    claim_id="clm_does_not_exist",
                    verdict=VerdictKind.CONFIRMED,
                    rationale="Confirming something that was never claimed.",
                )
            ]
        )
    )
    out = critic_node(state, deps)
    assert out["verdicts"] == []


def test_one_bad_verdict_does_not_discard_the_good_ones(scene) -> None:
    """The failure has to be per-verdict, not per-batch.

    Before the fix a single unusable verdict destroyed the entire run, so every
    other claim's review was lost with it.
    """
    make, _, claim = scene
    second = Claim(
        claim_id="clm_second",
        agent=AgentName.FINANCE,
        statement="North contributed 450,000 of revenue.",
        value=Decimal("450000"),
        computation_id="cmp_y",
        confidence=0.9,
        citations=[Citation(doc_id="doc_x", page=1, quote="North | 1200 | 450000")],
    )

    deps, state = make(
        CriticBatch(
            verdicts=[
                CriticOutput(
                    claim_id="clm_target",
                    verdict=VerdictKind.CONTESTED,
                    rationale="Unevidenced doubt, which the schema will reject.",
                ),
                CriticOutput(
                    claim_id="clm_second",
                    verdict=VerdictKind.CONFIRMED,
                    rationale="Re-derived and it matches the source exactly.",
                ),
            ]
        )
    )
    state["claims"] = [claim, second]

    out = critic_node(state, deps)
    assert len(out["verdicts"]) == 1
    assert out["verdicts"][0].claim_id == "clm_second"
    assert out.get("errors"), "the abstention is still reported"


def test_the_verdict_schema_still_rejects_unevidenced_adverse_verdicts() -> None:
    """The constraint itself must remain. The bug was the handler, not the rule."""
    with pytest.raises(Exception, match="contradicting citations"):
        Verdict(
            claim_id="clm_x",
            verdict=VerdictKind.CONTESTED,
            rationale="No evidence offered here at all.",
            contradicting_citations=[],
            independent_value=None,
        )
