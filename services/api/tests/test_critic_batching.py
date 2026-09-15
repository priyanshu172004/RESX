"""The Critic reviews in batches, and a failed batch costs only its own claims.

Seen on live runs. The Critic sent every claim in one request. Each claim
carries its statement plus its quoted evidence, so twenty claims make a large
prompt, and against a free tier's 8,000 tokens-per-minute ceiling the request
was rejected outright. The node's `except LLMError` then degraded the *whole*
Critic, so a report with twenty perfectly good findings came back with all
twenty marked "not reviewed".

Nothing was wrong with the findings. One request was too big.

Two properties are asserted here, and the second is the one that matters:

1. Claims are split into requests that fit.
2. **A failure is partial.** If the third of four batches is rejected, the
   other three batches' verdicts survive and only those claims go unreviewed —
   with the report naming them. All-or-nothing was the actual defect.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from app.agents.llm import LLMError, LLMResult, LLMUsage
from app.agents.schemas import AgentName, Citation, Claim, VerdictKind
from app.analysis.sandbox import LocalSubprocessSandbox
from app.graph.nodes import (
    CRITIC_BATCH_CHARS,
    CRITIC_BATCH_CLAIMS,
    CriticBatch,
    CriticOutput,
    _critic_batches,
    critic_node,
)
from app.graph.state import GraphDeps, initial_state
from app.rag.embeddings import HashingEmbedder
from app.store.local import LocalStore

WORKSPACE = "ws_batching"


def claim_at(index: int, *, quote_chars: int = 40) -> Claim:
    return Claim(
        claim_id=f"clm_{index:03d}",
        agent=AgentName.FINANCE,
        statement=f"Finding number {index} about the figures.",
        value=Decimal(str(1000 + index)),
        computation_id=f"cmp_{index}",
        confidence=0.9,
        citations=[
            Citation(doc_id="doc_x", page=1, quote="R" * max(quote_chars, 8)),
        ],
    )


class BatchingLLM:
    """Answers each request with a verdict per claim id it was shown.

    `fail_on` names 1-based request numbers to reject with an `LLMError`, which
    is how a rate limit arrives.
    """

    provider = "test"

    def __init__(self, fail_on: set[int] | None = None) -> None:
        self.fail_on = fail_on or set()
        self.calls: list[str] = []

    def structured(self, **kwargs: Any) -> LLMResult:
        user = str(kwargs.get("user") or "")
        self.calls.append(user)
        if len(self.calls) in self.fail_on:
            raise LLMError("429 rate_limit_exceeded: tokens per minute")

        ids = [
            line.split("claim_id:", 1)[1].strip()
            for line in user.splitlines()
            if line.startswith("claim_id:")
        ]
        return LLMResult(
            text="",
            parsed=CriticBatch(
                verdicts=[
                    CriticOutput(
                        claim_id=cid,
                        verdict=VerdictKind.CONFIRMED,
                        rationale="Re-derived independently and it matches.",
                    )
                    for cid in ids
                ]
            ),
            usage=LLMUsage(input_tokens=10, output_tokens=5, model="test"),
            stop_reason="stop",
        )


@pytest.fixture
def scene(tmp_path: Path):
    store = LocalStore(tmp_path / "resx.db")
    events: list[tuple[str, dict[str, Any]]] = []

    def make(claims: list[Claim], llm: BatchingLLM):
        deps = GraphDeps(
            llm=llm,
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
            run_id="run_batching",
            workspace_id=WORKSPACE,
            question="What do the figures say?",
            corpus_ids=["doc_x"],
        )
        state["claims"] = claims
        return deps, state

    yield make, events
    store.close()


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #


def test_a_small_run_is_still_one_request() -> None:
    """Batching must not cost a round trip on runs that never had the problem."""
    claims = [claim_at(i) for i in range(3)]
    assert len(_critic_batches(claims, lambda c: c.statement)) == 1


def test_many_claims_are_split() -> None:
    claims = [claim_at(i) for i in range(30)]
    batches = _critic_batches(claims, lambda c: "x" * 500)
    assert len(batches) > 1
    # Nothing is lost and nothing is reviewed twice.
    flat = [c.claim_id for batch in batches for c in batch]
    assert flat == [c.claim_id for c in claims]


def test_a_batch_never_exceeds_the_schema_limit() -> None:
    """`CriticBatch.verdicts` caps at 20, so a larger batch could not be
    answered in full even if the request were accepted."""
    claims = [claim_at(i) for i in range(40)]
    for batch in _critic_batches(claims, lambda c: "x"):
        assert len(batch) <= CRITIC_BATCH_CLAIMS


def test_batches_are_sized_by_text_not_by_count() -> None:
    """Claims are not the same size. One carrying a 900-character quoted table
    is worth six carrying a sentence, and splitting by count alone makes some
    requests trivially small and others too large to send.

    The size is measured on what is actually sent, so the fake `describe` here
    is what decides — not the claim object.
    """
    fat = [claim_at(i) for i in range(4)]
    batches = _critic_batches(fat, lambda c: "x" * CRITIC_BATCH_CHARS)
    assert len(batches) == 4, "each oversized claim needs its own request"


def test_one_oversized_claim_still_gets_sent() -> None:
    """Dropping it would be worse: the request may succeed anyway, and if it
    does not, only that claim is lost."""
    batches = _critic_batches([claim_at(0)], lambda c: "x" * (CRITIC_BATCH_CHARS * 3))
    assert len(batches) == 1
    assert batches[0][0].claim_id == "clm_000"


# --------------------------------------------------------------------------- #
# The property that matters
# --------------------------------------------------------------------------- #


def test_every_claim_is_reviewed_across_the_batches(scene) -> None:
    make, _events = scene
    claims = [claim_at(i) for i in range(20)]
    llm = BatchingLLM()
    deps, state = make(claims, llm)

    out = critic_node(state, deps)

    assert len(llm.calls) > 1, "20 claims should not go in one request"
    reviewed = {v.claim_id for v in out["verdicts"]}
    assert reviewed == {c.claim_id for c in claims}
    assert "critic" not in (out.get("degraded") or [])


def test_a_failed_batch_does_not_lose_the_others(scene) -> None:
    """The defect, stated as a test.

    Before batching, one rejected request meant every finding in the report
    read "not reviewed". Now the second batch's claims are the only casualties.
    """
    make, _events = scene
    claims = [claim_at(i) for i in range(20)]
    llm = BatchingLLM(fail_on={2})
    deps, state = make(claims, llm)

    out = critic_node(state, deps)

    verdicts = out.get("verdicts") or []
    assert verdicts, "a single failed batch must not wipe out every verdict"
    # Some claims reviewed, some not — which is the honest outcome.
    assert len(verdicts) < len(claims)


def test_the_unreviewed_claims_are_named(scene) -> None:
    """A reader has to be able to tell which figures nobody checked. "Some
    were not reviewed" is not actionable; a list of claim ids is."""
    make, _events = scene
    claims = [claim_at(i) for i in range(20)]
    deps, state = make(claims, BatchingLLM(fail_on={2}))

    out = critic_node(state, deps)

    errors = out.get("errors") or []
    assert errors, "a rejected batch must be reported"
    named = {cid for e in errors for cid in (e.get("claims_unreviewed") or [])}
    assert named, "the error must say which claims went unreviewed"
    reviewed = {v.claim_id for v in out["verdicts"]}
    assert not (named & reviewed), "a claim cannot be both reviewed and not"


def test_the_critic_is_only_degraded_when_nothing_succeeded(scene) -> None:
    """Partial success is not degradation. Marking the Critic degraded after
    one failed batch out of four would tell the report to distrust fifteen
    verdicts that are perfectly good."""
    make, _events = scene
    claims = [claim_at(i) for i in range(20)]

    deps, state = make(claims, BatchingLLM(fail_on={1}))
    partial = critic_node(state, deps)
    assert "critic" not in (partial.get("degraded") or [])

    deps, state = make(claims, BatchingLLM(fail_on={1, 2, 3, 4, 5, 6, 7, 8}))
    total = critic_node(state, deps)
    assert "critic" in (total.get("degraded") or [])
    assert not total.get("verdicts")


def test_progress_is_reported_per_batch(scene) -> None:
    """A Critic that takes four requests on a rate-limited tier is slow, and a
    run page showing nothing for two minutes reads as a hang."""
    make, events = scene
    deps, state = make([claim_at(i) for i in range(20)], BatchingLLM())

    critic_node(state, deps)

    batched = [payload for kind, payload in events if kind == "critic_batch"]
    assert len(batched) > 1
    assert all("of" in p and "claims" in p for p in batched)
