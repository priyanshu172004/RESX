"""Graph state.

State is the contract between nodes. Evidence channels are **append-only**:
`Annotated[list[...], operator.add]` means a node adds claims and never edits
another node's claims. Two consequences, both load-bearing:

  * The five specialists can run as parallel branches and have their outputs
    merged without a write conflict.
  * The debate is auditable, because nothing that was said can be quietly
    rewritten afterwards.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, TypedDict

from app.agents.llm import LLM
from app.agents.schemas import (
    AgentName,
    Budget,
    Claim,
    Corroboration,
    ExecutiveReport,
    Plan,
    Verdict,
)
from app.analysis.sandbox import Sandbox
from app.rag.embeddings import Embedder

if TYPE_CHECKING:
    # Type-only: `app.store.factory` imports the Mongo backend, and a
    # SQLite-only install must not need pymongo to build a graph.
    from app.store.factory import Store

EmitFn = Callable[[str, dict[str, Any]], None]


class RunState(TypedDict, total=False):
    """The shared state a run carries through the graph."""

    run_id: str
    workspace_id: str
    question: str
    corpus_ids: list[str]
    #: True when there is no corpus and evidence comes from the web.
    #: Read by the Manager, to route to the agents that can actually
    #: search, and by the Synthesizer, to disclose the weaker sourcing.
    research_mode: bool

    #: Run every specialist regardless of what the Manager planned.
    #:
    #: The Manager is instructed to plan the *minimum* set, which is the right
    #: default for cost: a question about process waste should not spend tokens
    #: on market sizing. But a full report is a different product -- the reader
    #: wants each specialist's angle on the question, and an agent that never
    #: ran has no section. This makes that a choice rather than a rewrite.
    #:
    #: It is expensive and the cost is not hidden: five specialists at two
    #: phases each, plus the Critic and any debate rounds, is roughly five
    #: times a focused run.
    all_agents: bool

    plan: Plan | None

    # --- append-only evidence channels ---
    claims: Annotated[list[Claim], operator.add]
    verdicts: Annotated[list[Verdict], operator.add]
    corroborations: Annotated[list[Corroboration], operator.add]
    computations: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[dict[str, Any]], operator.add]
    degraded: Annotated[list[str], operator.add]
    dropped_claims: Annotated[list[dict[str, Any]], operator.add]
    injection_attempts: Annotated[list[dict[str, Any]], operator.add]

    #: Spend is append-only for the same reason evidence is: the five
    #: specialists run as parallel branches, and a single-value channel can
    #: only accept one write per superstep. Each node appends what it spent and
    #: the total is derived, rather than nodes mutating one shared object —
    #: which would both conflict and double-count.
    spend: Annotated[list[dict[str, Any]], operator.add]

    # --- control ---
    debate_rounds: int
    reingest_count: int
    identity_report: dict[str, Any] | None
    #: Caps only. Never written by a node.
    budget: Budget
    report: ExecutiveReport | None
    status: str


class RunCancelledError(RuntimeError):
    """Raised at a node boundary when the run's owner cancelled it.

    A distinct type rather than a generic exception, so the runner can record
    "cancelled" instead of "failed". They are not the same outcome: a failure is
    something to investigate, a cancellation is something the user asked for,
    and a run page reporting the second as the first sends people looking for a
    bug that is not there.
    """


@dataclass(slots=True)
class GraphDeps:
    """Everything the nodes need, injected once when the graph is built.

    LangGraph node functions take only `state`, so dependencies are closed over
    rather than threaded through state. Keeping them out of state also keeps
    state serialisable for checkpointing — a live sandbox handle is not
    something you can write to Postgres.
    """

    llm: LLM
    store: Store
    embedder: Embedder
    sandbox: Sandbox | None = None
    dataset_dir: str = "./storage/datasets"
    tavily_api_key: str = ""
    egress_allowlist: frozenset[str] = field(default_factory=frozenset)
    top_k: int = 8
    # See `Settings.retrieval_context_chars`: the retrieved-text ceiling has to
    # follow the model's request limit, not the store's idea of a big page.
    context_chars: int = 7_000
    max_debate_rounds: int = 3
    max_reingest: int = 1
    emit: EmitFn | None = None
    #: Asked at every node boundary. Returns True once the run has been
    #: cancelled by its owner.
    should_cancel: Callable[[], bool] | None = None

    def event(self, kind: str, payload: dict[str, Any]) -> None:
        """Emit a progress event, and stop the run if it has been cancelled.

        The cancellation check lives here because this is the one place every
        node passes through — each begins by emitting `node_start` — so a
        single check covers the whole graph instead of eight that drift apart
        as nodes are added.

        Cancellation therefore takes effect at the next node boundary, not
        instantly. `graph.invoke()` is one blocking call and a node in flight
        cannot be interrupted from outside without killing the thread, which
        would leave the store half-written. Waiting for the boundary costs at
        most one node and leaves every record consistent.
        """
        if self.should_cancel is not None and self.should_cancel():
            raise RunCancelledError("the run was cancelled")
        if self.emit is not None:
            self.emit(kind, payload)


def initial_state(
    *,
    run_id: str,
    workspace_id: str,
    question: str,
    corpus_ids: Sequence[str],
    budget: Budget | None = None,
    all_agents: bool = False,
) -> RunState:
    return RunState(
        run_id=run_id,
        workspace_id=workspace_id,
        question=question,
        corpus_ids=list(corpus_ids),
        # Derived, never passed in: the mode *is* the absence of a
        # corpus, and two sources of truth for that would drift.
        research_mode=not list(corpus_ids),
        all_agents=all_agents,
        plan=None,
        claims=[],
        verdicts=[],
        corroborations=[],
        computations=[],
        errors=[],
        degraded=[],
        dropped_claims=[],
        injection_attempts=[],
        spend=[],
        debate_rounds=0,
        reingest_count=0,
        identity_report=None,
        budget=budget or Budget(),
        report=None,
        status="planning",
    )


def spend_entry(
    *, agent: str, usd: float = 0.0, tokens: int = 0, nodes: int = 0
) -> dict[str, Any]:
    return {"agent": agent, "usd": usd, "tokens": tokens, "nodes": nodes}


def totals(state: RunState) -> tuple[float, int, int]:
    """Sum the spend channel into `(usd, tokens, nodes)`."""
    usd = tokens = nodes = 0.0
    for entry in state.get("spend", []):
        usd += float(entry.get("usd", 0.0))
        tokens += int(entry.get("tokens", 0))
        nodes += int(entry.get("nodes", 0))
    return usd, int(tokens), int(nodes)


def is_exhausted(state: RunState) -> bool:
    """Has this run hit any of its ceilings?

    Checked before every model call. An agent loop without a budget is a
    self-inflicted denial of service, and it is the failure mode most likely to
    occur with no attacker present at all.
    """
    caps = state.get("budget") or Budget()
    usd, tokens, nodes = totals(state)
    return usd >= caps.usd_cap or tokens >= caps.token_cap or nodes >= caps.node_cap


def budget_reason(state: RunState) -> str:
    caps = state.get("budget") or Budget()
    usd, tokens, nodes = totals(state)
    if usd >= caps.usd_cap:
        return f"spend cap reached (${usd:.4f} of ${caps.usd_cap:.2f})"
    if tokens >= caps.token_cap:
        return f"token cap reached ({tokens:,} of {caps.token_cap:,})"
    if nodes >= caps.node_cap:
        return f"node cap reached ({nodes} of {caps.node_cap})"
    return "within budget"


def spend_summary(state: RunState) -> dict[str, Any]:
    caps = state.get("budget") or Budget()
    usd, tokens, nodes = totals(state)
    return {
        "usd_used": round(usd, 6),
        "usd_cap": caps.usd_cap,
        "tokens_used": tokens,
        "token_cap": caps.token_cap,
        "nodes_run": nodes,
        "node_cap": caps.node_cap,
        "exhausted": is_exhausted(state),
    }


def claims_by_agent(state: RunState, agent: AgentName) -> list[Claim]:
    return [c for c in state.get("claims", []) if c.agent == agent]


def verdict_for(state: RunState, claim_id: str) -> Verdict | None:
    """The most recent verdict on a claim.

    Verdicts accumulate across debate rounds, so the latest round wins — but
    the earlier ones stay in state, which is what makes the debate reviewable.
    """
    matching = [v for v in state.get("verdicts", []) if v.claim_id == claim_id]
    if not matching:
        return None
    return max(matching, key=lambda v: v.debate_round)
