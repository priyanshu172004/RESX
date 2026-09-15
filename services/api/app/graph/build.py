"""The LangGraph wiring.

LangGraph is the manager: it decides who works, when, and what happens on
failure. Three routing decisions carry the design, and they are the reason this
is a graph rather than a pipeline:

  * **`fan_out`** — the Manager's *plan* decides which specialists run, not a
    static edge list. A question about process waste should not spend tokens on
    market sizing.
  * **`reingest`** — a failed accounting identity routes *backwards* to the
    Manager. This is the "if the Finance agent fails, go back to the Data
    agent" loop, and it exists because a broken identity means the document was
    misread; the fix is to re-read it, never to reconcile the number.
  * **the debate loop** — critic → debate → critic, bounded at three rounds,
    with `CONTESTED` as a legitimate exit rather than a failure.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from functools import partial
from typing import Any, cast

from langgraph.graph import END, StateGraph

from app.agents.schemas import SPECIALISTS, AgentName, VerdictKind
from app.graph.nodes import (
    critic_node,
    debate_node,
    identity_node,
    make_specialist_node,
    manager_node,
    synthesize_node,
)
from app.graph.state import GraphDeps, RunState, is_exhausted

IDENTITY_NODE = "identity_check"


def _bind(fn: Callable[..., dict[str, Any]], deps: GraphDeps):  # type: ignore[no-untyped-def]
    """Close a node over its dependencies.

    LangGraph node functions receive only `state`, so deps are bound here
    rather than smuggled through state — which also keeps state serialisable
    for checkpointing.
    """
    bound = partial(fn, deps=deps)
    bound.__name__ = getattr(fn, "__name__", "node")  # type: ignore[attr-defined]
    return bound


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


def fan_out(state: RunState) -> list[str]:
    """Which specialists to run, from the Manager's plan.

    Returning a list makes LangGraph run those nodes as parallel branches; the
    append-only evidence channels merge their outputs without conflict.
    """
    if state.get("all_agents"):
        # Every specialist that can actually gather evidence for this run. In
        # research mode Finance and Workflow hold only filesystem and sandbox
        # tools, so running them would spend two model calls each to return
        # nothing -- "all agents" means all agents that can act, not all
        # agents that exist.
        from app.agents.tools import ALLOWLIST

        if state.get("research_mode"):
            return [a.value for a in SPECIALISTS if "search.web_search" in ALLOWLIST[a]]
        return [a.value for a in SPECIALISTS]

    plan = state.get("plan")
    if plan is None:
        # No plan means the Manager failed. Skip straight to the identity check
        # and on to synthesis, which will produce a report that says so rather
        # than silently returning nothing.
        return [IDENTITY_NODE]

    chosen = [a.value for a in plan.agents if a in SPECIALISTS]
    return chosen or [IDENTITY_NODE]


def route_after_critic(state: RunState, *, max_debate_rounds: int, max_reingest: int) -> str:
    """The decision that makes this a graph.

    Order matters: a failed identity is checked *first*, because there is no
    point debating the interpretation of a figure that was misread from the
    page in the first place.
    """
    report = state.get("identity_report")
    if (
        report is not None
        and not report.get("passed", True)
        and state.get("reingest_count", 0) < max_reingest
    ):
        return "reingest"

    rounds = state.get("debate_rounds", 0)
    disputed = any(
        v.verdict in (VerdictKind.REFUTED, VerdictKind.CONTESTED)
        for v in state.get("verdicts", [])
    )
    if disputed and rounds < max_debate_rounds and not is_exhausted(state):
        return "debate"

    return "synthesize"


def reingest_node(state: RunState) -> dict[str, Any]:
    """Record that we are going back to re-read the source.

    Deliberately thin: it increments the counter and annotates the run, then
    control returns to the Manager. The counter is what stops a document with a
    genuinely inconsistent statement from looping forever.
    """
    count = state.get("reingest_count", 0) + 1
    return {
        "reingest_count": count,
        "errors": [
            {
                "node": "reingest",
                "error": (
                    "an accounting identity failed, indicating a misread source; "
                    f"re-reading the documents (attempt {count})"
                ),
            }
        ],
        "status": "analyzing",
    }


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def build_graph(
    deps: GraphDeps,
    *,
    agents: Sequence[AgentName] | None = None,
    interrupt_before_synthesis: bool = False,
    checkpointer: Any = None,
) -> Any:
    """Compile the run graph.

    `interrupt_before_synthesis` exposes the optional human-in-the-loop
    checkpoint for regulated or high-stakes use: the run pauses with all
    evidence gathered and verdicts recorded, before the report is written.
    """
    specialists = list(agents) if agents else list(SPECIALISTS)

    graph = StateGraph(RunState)

    graph.add_node("manager", _bind(manager_node, deps))
    for agent in specialists:
        graph.add_node(agent.value, _bind(make_specialist_node(agent), deps))
    graph.add_node(IDENTITY_NODE, _bind(identity_node, deps))
    graph.add_node("critic", _bind(critic_node, deps))
    graph.add_node("debate", _bind(debate_node, deps))
    graph.add_node("synthesize", _bind(synthesize_node, deps))
    graph.add_node("reingest", reingest_node)

    graph.set_entry_point("manager")

    # The plan chooses the branches; every branch converges on the identity
    # check, which runs once after all specialists have finished.
    graph.add_conditional_edges(
        "manager",
        fan_out,
        # `dict[Hashable, str]` is what LangGraph declares; the keys here
        # are `str`, which is Hashable, but the invariance of dict rules
        # out the narrower type without an explicit annotation.
        cast(
            "dict[Hashable, str]",
            {agent.value: agent.value for agent in specialists}
            | {IDENTITY_NODE: IDENTITY_NODE},
        ),
    )
    for agent in specialists:
        graph.add_edge(agent.value, IDENTITY_NODE)

    graph.add_edge(IDENTITY_NODE, "critic")

    graph.add_conditional_edges(
        "critic",
        partial(
            route_after_critic,
            max_debate_rounds=deps.max_debate_rounds,
            max_reingest=deps.max_reingest,
        ),
        {
            "debate": "debate",
            "synthesize": "synthesize",
            "reingest": "reingest",
        },
    )

    graph.add_edge("debate", "critic")
    graph.add_edge("reingest", "manager")
    graph.add_edge("synthesize", END)

    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    if interrupt_before_synthesis:
        compile_kwargs["interrupt_before"] = ["synthesize"]

    return graph.compile(**compile_kwargs)
