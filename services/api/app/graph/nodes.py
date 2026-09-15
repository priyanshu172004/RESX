"""Graph nodes.

Every specialist runs the same **two-phase** cycle, and the shape of it is the
enforcement mechanism for "the LLM never does arithmetic":

  Phase A — the agent reads retrieved context and dataset profiles, then asks
            for the computations it needs. It returns *code*, not numbers.
  Phase B — the sandbox executes that code. The agent then writes claims and
            must reference a real `computation_id` for any figure.

A single-phase design would let a model emit a number and a plausible-looking
computation id in the same breath. Splitting the phases means the id cannot
exist unless the sandbox actually ran, and the `Claim` validator rejects a
numeric claim without one. The rule is therefore structural rather than
instructional.

Every claim then passes the grounding gate before it enters state. A claim that
fails is dropped and recorded — never downgraded to "low confidence", because a
low-confidence fabrication is still a fabrication in a report.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import Field, ValidationError

from app.agents import prompts
from app.agents.llm import LLMError
from app.agents.schemas import (
    AgentName,
    Citation,
    Claim,
    ExecutiveReport,
    ExecutiveReportDraft,
    Insight,
    Plan,
    Strict,
    Swot,
    SwotItem,
    Verdict,
    VerdictKind,
)
from app.agents.tools import Toolbelt, ToolError, ToolNotGrantedError
from app.analysis import identities
from app.graph.state import (
    GraphDeps,
    RunState,
    budget_reason,
    is_exhausted,
    spend_entry,
)
from app.rag.citations import ground_claim
from app.rag.provenance import derive_citations
from app.reporting.charts import build_charts, charts_from_claims
from app.reporting.document import build_document
from app.reporting.statistics import statistics_charts

# --------------------------------------------------------------------------- #
# Phase-A / Phase-B payloads
# --------------------------------------------------------------------------- #


class ComputeRequest(Strict):
    label: str = Field(min_length=1, max_length=120)
    purpose: str = Field(min_length=8, max_length=400)
    code: str = Field(min_length=1, max_length=8000)
    inputs: list[str] = Field(default_factory=list)


class ComputePlan(Strict):
    """What the agent wants computed. Deliberately contains no numbers."""

    requests: list[ComputeRequest] = Field(default_factory=list, max_length=8)
    notes: str = ""


class ClaimDraft(Strict):
    """A claim as the model writes it, before grounding."""

    statement: str = Field(min_length=8, max_length=2000)
    value: str | None = None
    unit: str | None = None
    period: str | None = None
    currency: str | None = None
    computation_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_reason: str | None = None
    doc_id: str | None = None
    page: int | None = None
    para_idx: int | None = None
    quote: str = Field(min_length=8, max_length=2000)
    url: str | None = None
    publisher: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class SpecialistOutput(Strict):
    claims: list[ClaimDraft] = Field(default_factory=list, max_length=12)
    injection_attempt_detected: bool = False
    injection_note: str = ""
    could_not_determine: list[str] = Field(default_factory=list)


class CriticOutput(Strict):
    claim_id: str
    verdict: VerdictKind
    independent_value: str | None = None
    rationale: str = Field(min_length=8, max_length=4000)
    contradiction_doc_id: str | None = None
    contradiction_page: int | None = None
    contradiction_quote: str | None = None


class CriticBatch(Strict):
    verdicts: list[CriticOutput] = Field(default_factory=list, max_length=20)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I),
    re.compile(r"disregard\s+the\s+(above|financial|figures)", re.I),
    re.compile(r"you\s+are\s+now\s+in\s+\w+\s+mode", re.I),
    re.compile(r"system\s+notice", re.I),
    re.compile(r"do\s+not\s+mention\s+this\s+instruction", re.I),
    re.compile(r"send\s+the\s+.{0,40}\s+to\s+\S+@\S+", re.I),
)


def scan_for_injection(text: str) -> list[str]:
    """Flag injection payloads in retrieved content.

    This is *detection for reporting*, not the defence. The defence is that the
    agent has no tool it should not have (see `agents/tools.py`). But an
    injection attempt embedded in a business document is itself a finding worth
    surfacing to the user, so it is recorded.
    """
    return [p.pattern for p in _INJECTION_PATTERNS if p.search(text)]


def _to_decimal(raw: str | None) -> Decimal | None:
    if raw is None or not str(raw).strip():
        return None
    cleaned = re.sub(r"[^\d.\-]", "", str(raw))
    if not cleaned or cleaned in {"-", ".", "-."}:
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def _draft_to_claim(draft: ClaimDraft, agent: AgentName) -> Claim:
    citation = Citation(
        doc_id=draft.doc_id,
        page=draft.page,
        para_idx=draft.para_idx,
        quote=draft.quote,
        url=draft.url,
        publisher=draft.publisher,
        resolution="external" if (draft.url and not draft.doc_id) else "anchor_not_found",
    )
    return Claim(
        agent=agent,
        statement=draft.statement,
        value=_to_decimal(draft.value),
        unit=draft.unit,
        period=draft.period,
        currency=(draft.currency or None),
        computation_id=draft.computation_id,
        citations=[citation],
        confidence=draft.confidence,
        confidence_reason=draft.confidence_reason,
        payload=draft.payload,
    )


def _gather_context(
    belt: Toolbelt, question: str, *, research: bool = False
) -> tuple[str, list[str]]:
    """Retrieve what the agent needs and report any injection found in it.

    `research` means there is no corpus: the run was started from a question
    alone. It changes which tools are worth calling, and getting that wrong is
    what made research-mode runs return nothing.

    The earlier version chose web search only when the corpus tool was
    *absent* from the allowlist. Market and Risk hold both tools, so they never
    searched the web at all — in research mode they queried an empty corpus,
    received "no matching content", and had nothing whatsoever to cite. Every
    claim they made was correctly dropped for being uncitable, and the report
    came back empty. Only News, which happens not to hold the corpus tool, ever
    reached the internet.

    That directly contradicted the Manager's own instructions, which say
    "Only News, Market and Risk can gather evidence in this mode". The prompt
    described the intended design; the wiring denied it.

    So the rule is about the *run*, not the allowlist: with no corpus, every
    agent granted web search uses it. With a corpus, the corpus stays primary
    and web search is the fallback for agents that cannot read it.
    """
    parts: list[str] = []
    injections: list[str] = []

    if research:
        # Nothing on the filesystem is in scope. Listing datasets and
        # documents here would describe a corpus this run is not permitted to
        # read -- `manager_node` already withholds it for exactly that reason,
        # and the specialists were quietly disagreeing with it.
        if "search.web_search" in belt.granted:
            results = belt.call("search.web_search", query=question)
            # Web pages are untrusted input in precisely the way documents are,
            # so they get the same injection scan. Omitting it here would have
            # left the internet as the one unscanned path into the prompt.
            injections = scan_for_injection(results)
            parts.append("## Web search\n" + results)
        return "\n\n".join(parts), injections

    if "filesystem.list_datasets" in belt.granted:
        parts.append("## Registered datasets\n" + belt.call("filesystem.list_datasets"))
    if "filesystem.list_documents" in belt.granted:
        parts.append("## Documents\n" + belt.call("filesystem.list_documents"))
    if "filesystem.search_corpus" in belt.granted:
        retrieved = belt.call("filesystem.search_corpus", query=question)
        injections = scan_for_injection(retrieved)
        parts.append("## Retrieved spans\n" + retrieved)
    if "search.web_search" in belt.granted and "filesystem.search_corpus" not in belt.granted:
        results = belt.call("search.web_search", query=question)
        injections = injections + scan_for_injection(results)
        parts.append("## Web search\n" + results)

    return "\n\n".join(parts), injections


def _run_compute_plan(
    belt: Toolbelt, plan: ComputePlan, deps: GraphDeps, state: RunState
) -> tuple[str, list[dict[str, Any]]]:
    """Execute Phase A's requests and format the results for Phase B."""
    lines: list[str] = []
    records: list[dict[str, Any]] = []

    for request in plan.requests:
        if "sandbox.run_python" not in belt.granted:
            lines.append(f"- {request.label}: no sandbox granted to this agent")
            continue
        try:
            # The return value is discarded on purpose: the sandbox
            # appends a ComputationRecord and that record -- not the
            # formatted string -- is what the rest of this function uses.
            belt.call("sandbox.run_python", code=request.code, inputs=request.inputs)
        except (ToolError, ToolNotGrantedError) as exc:
            lines.append(f"- {request.label}: tool error: {exc}")
            continue

        record = belt.computations[-1]
        records.append(record.to_dict())
        deps.event(
            "computation",
            {
                "computation_id": record.computation_id,
                "agent": belt.agent.value,
                "label": request.label,
                # The generated code is deliberately NOT streamed. It stays
                # retrievable per computation from `/computations/{id}`, which
                # is authenticated and workspace-scoped: auditing one number is
                # a deliberate act, whereas broadcasting every line of
                # generated Python reveals dataset ids, paths and the sandbox's
                # API surface to anyone merely watching a run.
                "ok": record.ok,
                "result": record.result,
                "duration_ms": record.duration_ms,
            },
        )
        lines.append(
            f"- label={request.label!r} computation_id={record.computation_id} "
            f"ok={record.ok} result={record.result!r} error={record.error!r}"
        )

    if not lines:
        return "No computations were requested or run.", records
    return "\n".join(lines), records


def _ground_and_collect(
    drafts: Sequence[ClaimDraft],
    *,
    agent: AgentName,
    deps: GraphDeps,
    state: RunState,
    computation_results: dict[str, Any],
) -> tuple[list[Claim], list[dict[str, Any]]]:
    """Validate, ground, and keep only claims that survive."""
    kept: list[Claim] = []
    dropped: list[dict[str, Any]] = []

    for draft in drafts:
        try:
            claim = _draft_to_claim(draft, agent)
        except Exception as exc:
            # A schema violation is itself a drop reason worth recording: most
            # often it is a numeric claim with no computation_id, which is the
            # rule working.
            dropped.append(
                {
                    "agent": agent.value,
                    "statement": draft.statement[:300],
                    "reason": f"schema rejected: {exc}",
                }
            )
            continue

        verdict = ground_claim(
            claim,
            store=deps.store,
            workspace_id=state["workspace_id"],
            computation_result=computation_results.get(claim.computation_id or ""),
        )

        if not verdict.accepted and claim.computation_id:
            # The agent's quote did not resolve, but the claim is backed by a
            # computation — and a computation records which datasets it read.
            # That is provenance the system holds already, so the quote is
            # sliced from the stored source text instead of relying on the
            # agent to have transcribed a table row exactly.
            #
            # This is why it is not a loosening: a derived quote *is* the
            # source, and a fabricated computation yields no datasets to walk
            # back from, so it produces nothing.
            derived = derive_citations(
                claim,
                store=deps.store,
                workspace_id=state["workspace_id"],
                citation_factory=Citation,
            )
            if derived:
                claim.citations = derived
                verdict = ground_claim(
                    claim,
                    store=deps.store,
                    workspace_id=state["workspace_id"],
                    computation_result=computation_results.get(claim.computation_id or ""),
                )
                if verdict.accepted:
                    deps.event(
                        "citation_derived",
                        {
                            "agent": agent.value,
                            "claim_id": claim.claim_id,
                            "computation_id": claim.computation_id,
                            "spans": len(derived),
                        },
                    )

        if not verdict.accepted:
            dropped.append(
                {
                    "agent": agent.value,
                    "claim_id": claim.claim_id,
                    "statement": claim.statement[:300],
                    "reason": verdict.reason,
                }
            )
            deps.event(
                "claim_dropped",
                {
                    "agent": agent.value,
                    "statement": claim.statement[:200],
                    "reason": verdict.reason,
                },
            )
            continue

        # Record how each citation resolved, so citation validity is measurable
        # rather than assumed.
        # One result per citation by construction in `ground_claim`, so a
        # length mismatch is a bug worth failing on rather than silently
        # dropping the tail.
        for citation, result in zip(claim.citations, verdict.citation_results, strict=True):
            # A derived citation resolves as "ok" -- it is sliced from the
            # source, so of course it matches. Overwriting the marker would
            # erase the distinction between a quote the agent produced and one
            # the system produced, which is exactly what a reader needs to
            # know and what the scorecard measures separately.
            if citation.resolution != "derived_from_computation":
                citation.resolution = result.resolution.value
            citation.match_score = result.score

        kept.append(claim)
        deps.event(
            "claim",
            {
                "claim_id": claim.claim_id,
                "agent": agent.value,
                "statement": claim.statement,
                "value": str(claim.value) if claim.value is not None else None,
                "confidence": claim.confidence,
                "citation": claim.citations[0].label,
            },
        )

    return kept, dropped


# --------------------------------------------------------------------------- #
# Manager
# --------------------------------------------------------------------------- #


def manager_node(state: RunState, deps: GraphDeps) -> dict[str, Any]:
    deps.event("node_start", {"node": "manager"})

    if is_exhausted(state):
        return {"status": "budget_exhausted", "degraded": ["manager"]}

    research = bool(state.get("research_mode"))
    documents = (
        []
        if research
        # Listing documents in research mode would describe a corpus the run is
        # not allowed to read, and the Manager would plan agents to read it.
        else deps.store.list_documents(workspace_id=state["workspace_id"])
    )
    corpus_summary = (
        "\n".join(
            f"{d['doc_id']} | {d['source_name']} | {d['page_count']} page(s)"
            for d in documents
            if not state.get("corpus_ids") or d["doc_id"] in state["corpus_ids"]
        )
        or "no documents"
    )

    user = (
        f"User question:\n{state['question']}\n\n"
        f"Available documents:\n{corpus_summary}\n\n"
        "Emit the minimum plan that answers the question."
    )

    try:
        result = deps.llm.structured(
            agent=AgentName.MANAGER,
            system=prompts.system_prompt(AgentName.MANAGER, research=research),
            user=user,
            schema=Plan,
            effort="medium",
        )
    except LLMError as exc:
        return {
            "errors": [{"node": "manager", "error": str(exc)}],
            "status": "failed",
            "degraded": ["manager"],
        }

    charged = [
        spend_entry(
            agent="manager",
            usd=result.usage.cost_usd,
            tokens=result.usage.total_tokens,
            nodes=1,
        )
    ]

    if result.refused or result.parsed is None:
        return {
            "errors": [
                {
                    "node": "manager",
                    "error": f"no plan produced (stop_reason={result.stop_reason})",
                }
            ],
            "status": "failed",
            "degraded": ["manager"],
            "spend": charged,
        }

    plan: Plan = result.parsed
    deps.event(
        "node_end",
        {
            "node": "manager",
            "agents": [a.value for a in plan.agents],
            "rationale": plan.rationale[:300],
            "cost_usd": round(result.usage.cost_usd, 6),
            "tokens": result.usage.total_tokens,
        },
    )
    return {"plan": plan, "status": "analyzing", "spend": charged}


# --------------------------------------------------------------------------- #
# Specialists
# --------------------------------------------------------------------------- #


def make_specialist_node(agent: AgentName):  # type: ignore[no-untyped-def]
    """Build a node function for one specialist."""

    def node(state: RunState, deps: GraphDeps) -> dict[str, Any]:
        deps.event("node_start", {"node": agent.value})

        if is_exhausted(state):
            deps.event(
                "node_end",
                {"node": agent.value, "skipped": True, "reason": budget_reason(state)},
            )
            return {"degraded": [agent.value]}

        charged: list[dict[str, Any]] = []

        plan: Plan | None = state.get("plan")
        sub_question = state["question"]
        if plan is not None:
            for task in plan.tasks:
                if task.agent == agent:
                    sub_question = task.sub_question
                    break

        belt = Toolbelt(
            agent,
            store=deps.store,
            workspace_id=state["workspace_id"],
            embedder=deps.embedder,
            sandbox=deps.sandbox,
            dataset_dir=deps.dataset_dir,
            tavily_api_key=deps.tavily_api_key,
            egress_allowlist=deps.egress_allowlist,
            corpus_ids=state.get("corpus_ids"),
            top_k=deps.top_k,
            context_chars=deps.context_chars,
        )

        # Read before gathering, not after: with no corpus the agent must
        # search the web instead of querying a corpus that does not exist.
        research = bool(state.get("research_mode"))

        try:
            context, injections = _gather_context(belt, sub_question, research=research)
        except (ToolError, ToolNotGrantedError) as exc:
            return {
                "errors": [{"node": agent.value, "error": f"context gathering: {exc}"}],
                "degraded": [agent.value],
            }

        for invocation in belt.invocations:
            deps.event("tool_call", invocation.to_dict())

        injection_records: list[dict[str, Any]] = []
        if injections:
            injection_records.append(
                {
                    "agent": agent.value,
                    "patterns": injections,
                    "note": (
                        "injection payload found in retrieved document content; "
                        "treated as data and reported as a finding"
                    ),
                }
            )
            deps.event(
                "injection_detected",
                {"agent": agent.value, "patterns": injections},
            )

        system = (
            prompts.system_prompt(agent, research=research) + "\n\n" + belt.describe_granted()
        )

        # --- Phase A: ask for computations, not numbers ---
        phase_a_user = (
            f"Sub-question:\n{sub_question}\n\n"
            f"{context}\n\n"
            "PHASE 1 of 2. Do not state any figures yet.\n"
            "List the computations you need in order to answer. Each must be "
            "Python that loads a registered dataset with resx.load(dataset_id), "
            "uses Decimal for money, and assigns its answer to `result`.\n"
            "If you need no computation (a purely qualitative finding), return "
            "an empty request list."
        )

        computation_results: dict[str, Any] = {}
        compute_summary = "No computations were requested."
        records: list[dict[str, Any]] = []

        try:
            phase_a = deps.llm.structured(
                agent=agent,
                system=system,
                user=phase_a_user,
                schema=ComputePlan,
                cache_prefix=prompts.SHARED_RULES,
            )
            charged.append(
                spend_entry(
                    agent=agent.value,
                    usd=phase_a.usage.cost_usd,
                    tokens=phase_a.usage.total_tokens,
                )
            )
            if phase_a.parsed is not None:
                compute_summary, records = _run_compute_plan(belt, phase_a.parsed, deps, state)
                for record in belt.computations:
                    computation_results[record.computation_id] = record.result
        except LLMError as exc:
            return {
                "errors": [{"node": agent.value, "error": f"phase A: {exc}"}],
                "degraded": [agent.value],
                "spend": charged,
            }

        # --- Phase B: write claims against real computation ids ---
        phase_b_user = (
            f"Sub-question:\n{sub_question}\n\n"
            f"{context}\n\n"
            f"## Computation results\n{compute_summary}\n\n"
            "PHASE 2 of 2. Write your claims now.\n"
            "Every claim needs a verbatim quote plus its doc_id and page. Any "
            "claim with a numeric value MUST carry the computation_id that "
            "produced it, taken from the results above — a value without one "
            "will be rejected.\n"
            "If you could not determine something, list it in "
            "could_not_determine rather than guessing."
        )

        try:
            phase_b = deps.llm.structured(
                agent=agent,
                system=system,
                user=phase_b_user,
                schema=SpecialistOutput,
                cache_prefix=prompts.SHARED_RULES,
            )
            charged.append(
                spend_entry(
                    agent=agent.value,
                    usd=phase_b.usage.cost_usd,
                    tokens=phase_b.usage.total_tokens,
                    nodes=1,
                )
            )
        except LLMError as exc:
            return {
                "errors": [{"node": agent.value, "error": f"phase B: {exc}"}],
                "degraded": [agent.value],
                "spend": charged,
            }

        if phase_b.refused or phase_b.parsed is None:
            return {
                "errors": [
                    {
                        "node": agent.value,
                        "error": f"no output (stop_reason={phase_b.stop_reason})",
                    }
                ],
                "degraded": [agent.value],
                "spend": charged,
            }

        output: SpecialistOutput = phase_b.parsed
        if output.injection_attempt_detected and output.injection_note:
            injection_records.append(
                {
                    "agent": agent.value,
                    "patterns": ["agent-reported"],
                    "note": output.injection_note[:400],
                }
            )

        kept, dropped = _ground_and_collect(
            output.claims,
            agent=agent,
            deps=deps,
            state=state,
            computation_results=computation_results,
        )

        deps.event(
            "node_end",
            {
                "node": agent.value,
                "claims_kept": len(kept),
                "claims_dropped": len(dropped),
                "computations": len(records),
                "cost_usd": round(phase_a.usage.cost_usd + phase_b.usage.cost_usd, 6),
            },
        )

        update: dict[str, Any] = {
            "claims": kept,
            "computations": records,
            "dropped_claims": dropped,
            "spend": charged,
        }
        if injection_records:
            update["injection_attempts"] = injection_records
        return update

    node.__name__ = f"{agent.value}_node"
    return node


# --------------------------------------------------------------------------- #
# Identity check (deterministic, no model)
# --------------------------------------------------------------------------- #


def _financial_figures(state: RunState) -> dict[str, Decimal]:
    """Pull named financial figures out of the accepted claims.

    Keyed off the claim payload the Finance agent supplies, so this stays
    deterministic — no model reads or re-derives anything here.
    """
    figures: dict[str, Decimal] = {}
    for claim in state.get("claims", []):
        if claim.agent is not AgentName.FINANCE or claim.value is None:
            continue
        key = str(claim.payload.get("figure", "")).strip().lower()
        if key:
            figures[key] = claim.value
    return figures


def identity_node(state: RunState, deps: GraphDeps) -> dict[str, Any]:
    """Assert the accounting identities in code.

    A failure routes *backwards* to ingestion rather than forwards to the
    Critic: a broken identity means the document was misread, and the correct
    response is to re-extract, never to reconcile a number until it closes.
    """
    deps.event("node_start", {"node": "identity_check"})
    figures = _financial_figures(state)

    statements: dict[str, dict[str, Any]] = {}
    if {"revenue", "cogs", "gross_profit"} <= figures.keys():
        income: dict[str, Any] = {
            "revenue": figures["revenue"],
            "cogs": figures["cogs"],
            "gross_profit": figures["gross_profit"],
        }
        if {"opex", "operating_income"} <= figures.keys():
            income["opex"] = figures["opex"]
            income["operating_income"] = figures["operating_income"]
        statements["income_statement"] = income

    if {"assets", "liabilities", "equity"} <= figures.keys():
        statements["balance_sheet"] = {
            "assets": figures["assets"],
            "liabilities": figures["liabilities"],
            "equity": figures["equity"],
        }

    if {"opening_cash", "net_cash_flow", "closing_cash"} <= figures.keys():
        statements["cash_flow"] = {
            "opening_cash": figures["opening_cash"],
            "net_cash_flow": figures["net_cash_flow"],
            "closing_cash": figures["closing_cash"],
        }

    if not statements:
        deps.event(
            "node_end",
            {
                "node": "identity_check",
                "checked": 0,
                "note": "no checkable statement set was extracted",
            },
        )
        return {
            "identity_report": {
                "passed": True,
                "checks": [],
                "note": "no identities were checkable from the extracted figures",
            }
        }

    report = identities.check_all(statements)
    deps.event(
        "node_end",
        {
            "node": "identity_check",
            "checked": len(report.checks),
            "passed": report.passed,
            "summary": report.summary(),
        },
    )
    return {"identity_report": report.to_dict()}


# --------------------------------------------------------------------------- #
# Critic
# --------------------------------------------------------------------------- #

#: Characters of claim text one Critic request may carry.
#:
#: Derived rather than picked. `LLMClient.max_request_tokens` defaults to 6,000
#: and `CHARS_PER_TOKEN` is 3.2, so a request has roughly 19,000 characters to
#: spend — and the claims are only part of it. The system prompt, the shared
#: rules, the dataset listing and the response schema take the rest, so the
#: claim block gets a third of the budget. Under-filling a request costs an
#: extra round trip; over-filling it costs the whole batch.
CRITIC_BATCH_CHARS = 6_000

#: However much fits by characters, never more claims than the response schema
#: can hold. `CriticBatch.verdicts` caps at 20, so a 25-claim batch could not
#: be answered in full even if it were accepted.
CRITIC_BATCH_CLAIMS = 8


def _critic_batches(claims: list[Any], describe: Callable[[Any], str]) -> list[list[Any]]:
    """Split the claims into requests that each fit the token budget.

    Sized by the *rendered* text rather than by claim count, because claims are
    not the same size: one carrying a 900-character quoted table is worth six
    carrying a single sentence, and a fixed count would make some batches
    trivially small and others too large to send.

    A single claim that exceeds the budget on its own still gets its own batch.
    Splitting it further is not possible, and dropping it would be worse — the
    request may still succeed, and if it does not, only that one claim is lost.
    """
    batches: list[list[Any]] = []
    current: list[Any] = []
    size = 0

    for claim in claims:
        length = len(describe(claim))
        too_long = current and size + length > CRITIC_BATCH_CHARS
        too_many = len(current) >= CRITIC_BATCH_CLAIMS
        if too_long or too_many:
            batches.append(current)
            current = []
            size = 0
        current.append(claim)
        size += length

    if current:
        batches.append(current)
    return batches


def critic_node(state: RunState, deps: GraphDeps) -> dict[str, Any]:
    deps.event("node_start", {"node": "critic"})
    claims = state.get("claims", [])

    if not claims:
        deps.event("node_end", {"node": "critic", "note": "no claims to review"})
        return {"status": "synthesizing"}

    if is_exhausted(state):
        deps.event(
            "node_end",
            {"node": "critic", "skipped": True, "reason": budget_reason(state)},
        )
        return {"degraded": ["critic"], "status": "synthesizing"}

    round_number = state.get("debate_rounds", 0)
    already = {v.claim_id for v in state.get("verdicts", []) if v.debate_round == round_number}
    to_review = [c for c in claims if c.claim_id not in already]
    if not to_review:
        return {"status": "synthesizing"}

    belt = Toolbelt(
        AgentName.CRITIC,
        store=deps.store,
        workspace_id=state["workspace_id"],
        embedder=deps.embedder,
        sandbox=deps.sandbox,
        dataset_dir=deps.dataset_dir,
        corpus_ids=state.get("corpus_ids"),
        top_k=deps.top_k,
        context_chars=deps.context_chars,
    )

    def describe(claim: Any) -> str:
        return (
            f"claim_id: {claim.claim_id}\n"
            f"agent: {claim.agent.value}\n"
            f"statement: {claim.statement}\n"
            f"value: {claim.value if claim.value is not None else '(none)'} "
            f"{claim.unit or ''}\n"
            f"cited: {claim.citations[0].doc_id} {claim.citations[0].label}\n"
            f'quote: "{claim.citations[0].quote}"\n'
            f"stated_confidence: {claim.confidence}"
        )

    datasets = belt.call("filesystem.list_datasets")
    identity_note = ""
    report = state.get("identity_report")
    if report and not report.get("passed", True):
        identity_note = (
            "\n\nThe deterministic identity check FAILED:\n"
            + "\n".join(
                f"  - {c['message']}" for c in report.get("checks", []) if not c["passed"]
            )
            + "\nTreat any claim depending on these figures as suspect."
        )

    research = bool(state.get("research_mode"))
    system = (
        prompts.system_prompt(AgentName.CRITIC, research=research)
        + "\n\n"
        + belt.describe_granted()
    )
    # Reviewed in batches rather than all at once.
    #
    # One request carrying every claim is the natural way to write this, and on
    # a free tier it fails in the worst possible way. Twenty claims, each with
    # its statement and its quoted evidence, make a large prompt; against an
    # 8,000 tokens-per-minute ceiling it is rejected outright. The handler below
    # then degraded the *entire* Critic, so every finding in the report came
    # back "not reviewed" — not because anything was wrong with any of them,
    # but because one request was too big.
    #
    # Batching also turns a total failure into a partial one. If the fourth
    # batch fails, the first three are already collected and keep their value;
    # only those claims go unreviewed, and the report names them. A run that
    # reviews 15 of 20 claims and says which 5 it missed is worth far more than
    # one that reviews none.
    batches = _critic_batches(to_review, describe)
    collected: list[CriticOutput] = []
    charged: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for index, batch in enumerate(batches, start=1):
        claim_block = "\n\n".join(describe(c) for c in batch)
        part = f" (part {index} of {len(batches)})" if len(batches) > 1 else ""
        user = (
            f"Original question:\n{state['question']}\n\n"
            f"## Registered datasets\n{datasets}\n"
            f"{identity_note}\n\n"
            f"## Claims to falsify (round {round_number}){part}\n{claim_block}\n\n"
            "Re-derive every figure independently using sandbox.run_python — do not "
            "reuse the original computation. Search the corpus for contradictions. "
            "Then return one verdict per claim_id.\n"
            "A refuted or contested verdict must carry a contradicting quote or your "
            "own independently derived value."
        )

        try:
            result = deps.llm.structured(
                agent=AgentName.CRITIC,
                system=system,
                user=user,
                schema=CriticBatch,
                cache_prefix=prompts.SHARED_RULES,
                effort="high",
            )
        except LLMError as exc:
            # This batch only. Its claims stay unreviewed and are disclosed as
            # such; everything already collected is kept.
            errors.append(
                {
                    "node": "critic",
                    "error": f"batch {index}/{len(batches)}: {exc}",
                    "claims_unreviewed": [c.claim_id for c in batch],
                }
            )
            continue

        charged.append(
            spend_entry(
                agent="critic",
                usd=result.usage.cost_usd,
                tokens=result.usage.total_tokens,
                nodes=1,
            )
        )
        if result.parsed is None:
            errors.append(
                {
                    "node": "critic",
                    "error": f"batch {index}/{len(batches)}: no verdicts produced",
                    "claims_unreviewed": [c.claim_id for c in batch],
                }
            )
            continue

        collected.extend(result.parsed.verdicts)
        deps.event(
            "critic_batch",
            {
                "batch": index,
                "of": len(batches),
                "claims": len(batch),
                "verdicts": len(result.parsed.verdicts),
            },
        )

    if not collected:
        return {
            "errors": errors or [{"node": "critic", "error": "no verdicts produced"}],
            "degraded": ["critic"],
            "status": "synthesizing",
            "spend": charged,
        }

    verdicts: list[Verdict] = []
    # Claims the Critic could not return a schema-valid verdict on. Reported
    # rather than dropped: a reader needs to know which figures went
    # unreviewed.
    abstentions: list[dict[str, Any]] = []
    known = {c.claim_id for c in claims}
    for item in collected:
        if item.claim_id not in known:
            continue  # a verdict on a claim that does not exist is discarded

        contradicting: list[Citation] = []
        if item.contradiction_quote and item.contradiction_doc_id and item.contradiction_page:
            try:
                contradicting.append(
                    Citation(
                        doc_id=item.contradiction_doc_id,
                        page=item.contradiction_page,
                        quote=item.contradiction_quote,
                    )
                )
            except Exception:
                contradicting = []

        try:
            verdict = Verdict(
                claim_id=item.claim_id,
                verdict=item.verdict,
                independent_value=_to_decimal(item.independent_value),
                rationale=item.rationale,
                contradicting_citations=contradicting,
                debate_round=round_number,
            )
        except ValidationError as exc:
            # The schema requires evidence for an adverse verdict, and the
            # Critic supplied none. Two tempting responses are both wrong:
            #
            #   * Upgrading to CONFIRMED accepts an adverse judgement as a
            #     favourable one, which is the worst direction to fail in.
            #   * Forcing a CONTESTED with empty evidence violates the same
            #     constraint -- which is exactly what the previous version of
            #     this handler did, so it raised from inside the `except` and
            #     took the whole run down with it.
            #
            # A verdict the schema rejects is not a verdict. The claim stays
            # unreviewed, the abstention is recorded, and the report discloses
            # it. An unreviewed claim that says so is honest; a manufactured
            # verdict is not.
            reason = str(exc).splitlines()[0][:200]
            abstentions.append(
                {
                    "claim_id": item.claim_id,
                    "attempted": getattr(item.verdict, "value", str(item.verdict)),
                    "reason": reason,
                }
            )
            deps.event(
                "critic_abstained",
                {
                    "claim_id": item.claim_id,
                    "attempted": getattr(item.verdict, "value", str(item.verdict)),
                    "reason": reason,
                },
            )
            continue

        verdicts.append(verdict)
        deps.event(
            "verdict",
            {
                "claim_id": verdict.claim_id,
                "verdict": verdict.verdict.value,
                "round": round_number,
                "rationale": verdict.rationale[:300],
            },
        )

    for invocation in belt.invocations:
        deps.event("tool_call", invocation.to_dict())

    computations = [r.to_dict() for r in belt.computations]
    disputed = sum(
        1 for v in verdicts if v.verdict in (VerdictKind.REFUTED, VerdictKind.CONTESTED)
    )
    deps.event(
        "node_end",
        {
            "node": "critic",
            "verdicts": len(verdicts),
            "disputed": disputed,
            "abstained": len(abstentions),
            "batches": len(batches),
            "batches_failed": len(errors),
            # Summed across the batches. Reading it off the last `result` was a
            # crash waiting to happen: if the final batch was the one rejected,
            # that name is unbound.
            "cost_usd": round(sum(float(c.get("usd") or 0.0) for c in charged), 6),
        },
    )

    out: dict[str, Any] = {
        "verdicts": verdicts,
        "computations": computations,
        "spend": charged,
    }

    # Both kinds of gap in one place, because they mean the same thing to a
    # reader: this figure was not checked. One comes from a request that never
    # landed, the other from a verdict that arrived without evidence.
    reported = list(errors)
    if abstentions:
        reported.append(
            {
                "node": "critic",
                "error": (
                    f"abstained on {len(abstentions)} claim(s): the verdict "
                    f"returned had no supporting evidence, so those figures are "
                    f"unreviewed"
                ),
                "claims_unreviewed": [a["claim_id"] for a in abstentions],
            }
        )
    if reported:
        # Recorded on the run so the Synthesizer can disclose it. A claim the
        # Critic could not review is a weaker claim than one it confirmed, and
        # the reader has to be able to tell them apart.
        out["errors"] = reported
    return out


# --------------------------------------------------------------------------- #
# Debate
# --------------------------------------------------------------------------- #


def debate_node(state: RunState, deps: GraphDeps) -> dict[str, Any]:
    """One bounded round of author-versus-critic.

    Capped at three rounds because disagreements that survive three
    evidence-backed exchanges are almost always genuine ambiguity in the
    source, and a fourth round produces rhetoric rather than evidence.
    """
    round_number = state.get("debate_rounds", 0) + 1
    deps.event("node_start", {"node": "debate", "round": round_number})

    disputed_ids = [
        v.claim_id
        for v in state.get("verdicts", [])
        if v.verdict in (VerdictKind.REFUTED, VerdictKind.CONTESTED)
    ]
    claims_by_id = {c.claim_id: c for c in state.get("claims", [])}
    disputed = [claims_by_id[cid] for cid in dict.fromkeys(disputed_ids) if cid in claims_by_id]

    if not disputed or is_exhausted(state) or round_number > deps.max_debate_rounds:
        deps.event(
            "node_end",
            {"node": "debate", "round": round_number, "note": "nothing left to debate"},
        )
        return {"debate_rounds": round_number}

    belt = Toolbelt(
        AgentName.FINANCE,
        store=deps.store,
        workspace_id=state["workspace_id"],
        embedder=deps.embedder,
        sandbox=deps.sandbox,
        dataset_dir=deps.dataset_dir,
        corpus_ids=state.get("corpus_ids"),
        context_chars=deps.context_chars,
    )

    latest = {
        v.claim_id: v for v in sorted(state.get("verdicts", []), key=lambda v: v.debate_round)
    }
    block = "\n\n".join(
        f"claim_id: {c.claim_id}\n"
        f"your claim: {c.statement}\n"
        f"value: {c.value if c.value is not None else '(none)'}\n"
        f"critic verdict: {latest[c.claim_id].verdict.value}\n"
        f"critic rationale: {latest[c.claim_id].rationale}"
        for c in disputed
        if c.claim_id in latest
    )

    system = prompts.SHARED_RULES + "\n\n---\n\n" + prompts.DEBATE_AUTHOR
    user = (
        f"{belt.describe_granted()}\n\n"
        f"## Disputed claims (round {round_number} of {deps.max_debate_rounds})\n"
        f"{block}\n\n"
        "For each claim, concede, defend with NEW evidence, or narrow it. Then "
        "restate the claims you still stand behind, with citations and "
        "computation ids as usual."
    )

    try:
        result = deps.llm.structured(
            agent=AgentName.FINANCE,
            system=system,
            user=user,
            schema=SpecialistOutput,
            cache_prefix=prompts.SHARED_RULES,
        )
        charged = [
            spend_entry(
                agent="debate",
                usd=result.usage.cost_usd,
                tokens=result.usage.total_tokens,
                nodes=1,
            )
        ]
    except LLMError as exc:
        return {
            "errors": [{"node": "debate", "error": str(exc)}],
            "debate_rounds": round_number,
        }

    revised: list[Claim] = []
    dropped: list[dict[str, Any]] = []
    if result.parsed is not None:
        revised, dropped = _ground_and_collect(
            result.parsed.claims,
            agent=AgentName.FINANCE,
            deps=deps,
            state=state,
            computation_results={},
        )

    deps.event(
        "debate_round",
        {
            "round": round_number,
            "disputed": len(disputed),
            "revised": len(revised),
        },
    )
    return {
        "claims": revised,
        "dropped_claims": dropped,
        "debate_rounds": round_number,
        "spend": charged,
    }


# --------------------------------------------------------------------------- #
# Synthesizer
# --------------------------------------------------------------------------- #


def _verdict_label(claim_id: str, verdicts: dict[str, Verdict]) -> str:
    """How the Critic ruled on a claim, for the Synthesizer's claim block."""
    found = verdicts.get(claim_id)
    return found.verdict.value if found is not None else "not reviewed"


def synthesize_node(state: RunState, deps: GraphDeps) -> dict[str, Any]:
    deps.event("node_start", {"node": "synthesize"})
    claims = state.get("claims", [])

    contested = [
        v.claim_id
        for v in state.get("verdicts", [])
        if v.verdict in (VerdictKind.CONTESTED, VerdictKind.REFUTED)
    ]

    limitations = _build_limitations(state)

    if not claims:
        # Written here rather than asked of a model, because there is nothing
        # for a model to summarise and asking would invite it to fill the
        # space. It says what happened and what to do about it -- the failure
        # is nearly always that the sources did not carry the answer, and a
        # reader needs to know that rather than seeing an empty page.
        dropped = state.get("dropped_claims", []) or []
        detail = (
            f"{len(dropped)} finding(s) were produced but none could be traced "
            f"to a source, so all were dropped rather than reported unverified."
            if dropped
            else "No agent produced a finding that could be traced to a source."
        )
        report = ExecutiveReport(
            question=state["question"],
            executive_summary=(
                f"This analysis could not answer the question from the "
                f"evidence available. {detail} Nothing is reported here "
                f"because every figure in a RESX report has to resolve to a "
                f"quote in a source, and an unverifiable answer is worse than "
                f"none. The limitations below say what stood in the way; "
                f"a narrower question, or a source that states the figure "
                f"directly, is usually what closes the gap."
            ),
            insights=[],
            financial_summary="",
            limitations=[
                *limitations,
                "No claim could be grounded, so no insight can be offered.",
            ],
            contested_claim_ids=contested,
            degraded_agents=[],
        )
        deps.event("node_end", {"node": "synthesize", "insights": 0})
        return {"report": report, "status": "done"}

    verdict_by_claim = {v.claim_id: v for v in state.get("verdicts", [])}
    claim_block = "\n\n".join(
        f"claim_id: {c.claim_id}\n"
        f"agent: {c.agent.value}\n"
        f"statement: {c.statement}\n"
        f"value: {c.value if c.value is not None else '(none)'} {c.unit or ''}\n"
        f"confidence: {c.confidence}\n"
        f"verdict: {_verdict_label(c.claim_id, verdict_by_claim)}\n"
        # `doc_id` alone rendered a web source as the literal "None", which is
        # the only thing the Synthesizer was told about where a research-mode
        # finding came from. The URL is the anchor in that mode, so show it.
        f"cited: {c.citations[0].doc_id or c.citations[0].url or 'unknown'} "
        f"({c.citations[0].label})\n"
        # 160 rather than 300. The quote is here for judgement — is this claim
        # about what the question asked? — not for transcription: the citation
        # is attached in code from the claim id now, so a full quote buys
        # nothing and the claim block has ~4,900 characters to fit every claim
        # in. Halving it is the difference between all the claims reaching the
        # Synthesizer and the middle of the list being trimmed away.
        f'quote: "{c.citations[0].quote[:160]}"'
        for c in claims
    )

    user = (
        f"Question:\n{state['question']}\n\n"
        f"## Grounded claims and their verdicts\n{claim_block}\n\n"
        f"## Known limitations to carry into the report\n"
        + ("\n".join(f"- {item}" for item in limitations) or "- none")
        + "\n\nWrite the report. At most five insights. Every figure must trace "
        "to a claim above; introduce no new numbers. Mark contested claims as "
        "contested.\n"
        "Reference evidence by claim_id only. Do NOT write citation objects, "
        "quotes, doc_ids or page numbers — the citations already exist on the "
        "claims and are attached for you."
    )

    try:
        result = deps.llm.structured(
            agent=AgentName.SYNTHESIZER,
            system=prompts.system_prompt(
                AgentName.SYNTHESIZER, research=bool(state.get("research_mode"))
            ),
            user=user,
            # The draft, not the final report: it carries claim ids instead of
            # citation objects, so the model cannot mis-transcribe one.
            schema=ExecutiveReportDraft,
            effort="high",
        )
        charged = [
            spend_entry(
                agent="synthesizer",
                usd=result.usage.cost_usd,
                tokens=result.usage.total_tokens,
                nodes=1,
            )
        ]
    except LLMError as exc:
        return {
            "errors": [{"node": "synthesize", "error": str(exc)}],
            "status": "failed",
        }

    if result.parsed is None:
        return {
            "errors": [{"node": "synthesize", "error": "no report produced"}],
            "status": "failed",
        }

    draft: ExecutiveReportDraft = result.parsed

    # Attach the real citations, taken from the claims the model referenced.
    #
    # The model used to be asked for citation objects directly, and it copied a
    # claim's quote and doc_id while dropping the page — which `Citation`
    # rejects, correctly, because a corpus citation without a page cannot be
    # resolved. The whole report then failed validation twice and the run died
    # with everything else about it correct.
    #
    # Re-transcription was never worth attempting: the citation already exists,
    # already verified, attached to the claim, and passing it through a
    # language model can only lose or corrupt fields. The evidence on an
    # insight is now the evidence that was actually checked.
    by_id = {c.claim_id: c for c in claims}

    def citations_for(claim_ids: Sequence[str]) -> list[Citation]:
        out: list[Citation] = []
        seen: set[str] = set()
        for claim_id in claim_ids:
            claim = by_id.get(claim_id)
            if claim is None:
                # A claim id that is not in this run. Silently skipped here
                # rather than raised: the recommendation gate below reports
                # invented ids, and an insight with no evidence is visible on
                # its own.
                continue
            for citation in claim.citations:
                if citation.citation_id in seen:
                    continue
                seen.add(citation.citation_id)
                out.append(citation)
        return out

    report = ExecutiveReport(
        question=draft.question or state["question"],
        executive_summary=draft.executive_summary,
        insights=[
            Insight(
                headline=item.headline,
                so_what=item.so_what,
                evidence=citations_for(item.supporting_claim_ids),
                confidence=item.confidence,
                suggested_owner=item.suggested_owner,
                effort=item.effort,
                expected_impact=item.expected_impact,
                impact_unit=item.impact_unit,
                contested=item.contested,
                supporting_claim_ids=list(item.supporting_claim_ids),
            )
            for item in draft.insights
        ],
        recommendations=list(draft.recommendations),
        swot=Swot(
            **{
                quadrant: [
                    SwotItem(
                        text=item.text,
                        citations=citations_for(item.supporting_claim_ids),
                        supporting_claim_ids=list(item.supporting_claim_ids),
                    )
                    for item in getattr(draft.swot, quadrant)
                ]
                for quadrant in (
                    "strengths",
                    "weaknesses",
                    "opportunities",
                    "threats",
                )
            }
        ),
        financial_summary=draft.financial_summary,
        risk_register=list(draft.risk_register),
        limitations=list(draft.limitations),
        contested_claim_ids=list(draft.contested_claim_ids),
    )

    # Advice is grounded on the same terms a figure is.
    #
    # A recommendation cites the claims it follows from, and a claim_id that is
    # not in this run is the same failure as a quote that resolves to nothing:
    # it means the justification was invented. Checked in code rather than
    # trusted to the prompt, because "cite a claim" is exactly the instruction
    # a model satisfies by writing something claim-shaped.
    known_claims = {c.claim_id for c in claims}
    grounded_recommendations = []
    ungrounded = []
    for rec in report.recommendations:
        unknown = [cid for cid in rec.supporting_claim_ids if cid not in known_claims]
        if unknown:
            ungrounded.append({"action": rec.action[:200], "unknown_claim_ids": unknown})
            deps.event(
                "recommendation_dropped",
                {"action": rec.action[:200], "unknown_claim_ids": unknown},
            )
            continue
        grounded_recommendations.append(rec)

    if ungrounded:
        # Said in the report, not just in the log. A reader who is given six
        # recommendations has no way to know two were removed.
        limitations = [
            *limitations,
            f"{len(ungrounded)} recommendation(s) were dropped for citing "
            f"findings this run did not produce.",
        ]

    # Charts are derived from the extracted tables, never taken from the
    # model. Overwritten unconditionally rather than merged: anything the
    # model emitted here is invented data, and the whole point of plotting
    # from the CSV is that every point traces to the document.
    derived = build_charts(
        store=deps.store,
        workspace_id=state["workspace_id"],
        corpus_ids=list(state.get("corpus_ids") or []),
        dataset_dir=deps.dataset_dir,
    )
    if not derived:
        # A research-mode run has no datasets, so the report would carry no
        # figure at all. It does have grounded numeric claims, held to the same
        # standard by a different route, so those are plotted instead -- and
        # labelled as findings rather than as a measured series.
        derived = charts_from_claims(list(claims))

    # Then the analysis itself: who contributed, how much survived review, how
    # confident it claimed to be, and how concentrated the sources are. All
    # counted from this run's own record, so they are honest about a thin run
    # rather than drawing an encouraging shape over three claims.
    derived = [*derived, *statistics_charts(list(claims), list(state.get("verdicts", [])))]
    derived_charts = [chart.to_dict() for chart in derived]

    # The synthesizer is told to carry limitations and degraded agents; they are
    # also merged in here so a model that forgets cannot produce a report that
    # silently omits a missing branch.
    merged = list(dict.fromkeys([*report.limitations, *limitations]))
    report = report.model_copy(
        update={
            "recommendations": grounded_recommendations,
            "charts": derived_charts,
            "limitations": merged,
            "contested_claim_ids": list(
                dict.fromkeys([*report.contested_claim_ids, *contested])
            ),
            "degraded_agents": [
                AgentName(a)
                for a in dict.fromkeys(state.get("degraded", []))
                if a in {x.value for x in AgentName}
            ],
        }
    )

    # The full document, assembled last so it sees the finished report.
    #
    # Built from the run's own claims and verdicts rather than asked of a
    # model: a section per specialist cannot be silently omitted by a loop, no
    # figure can be restated slightly wrong, and it costs no tokens on a tier
    # where the run is already waiting out rate limits. See
    # `app.reporting.document`.
    report = report.model_copy(
        update={
            "document": build_document(
                question=state["question"],
                report=report,
                claims=list(claims),
                verdicts=list(state.get("verdicts", [])),
                computations=list(state.get("computations", [])),
            ).to_dict()
        }
    )

    deps.event(
        "node_end",
        {
            "node": "synthesize",
            "insights": len(report.insights),
            "recommendations": len(report.recommendations),
            "recommendations_dropped": len(ungrounded),
            "contested": len(report.contested_claim_ids),
            "cost_usd": round(result.usage.cost_usd, 6),
        },
    )
    return {"report": report, "status": "done", "spend": charged}


def _build_limitations(state: RunState) -> list[str]:
    """Assemble everything the report must disclose about itself."""
    limitations: list[str] = []

    degraded = list(dict.fromkeys(state.get("degraded", [])))
    if degraded:
        limitations.append(
            "These agents did not complete and their perspective is missing: "
            + ", ".join(degraded)
        )

    dropped = state.get("dropped_claims", [])
    if dropped:
        limitations.append(
            f"{len(dropped)} claim(s) were dropped for failing the grounding gate "
            "and are excluded from this report."
        )

    report = state.get("identity_report")
    if report and not report.get("passed", True):
        limitations.append(
            "An accounting identity did not hold, which indicates a misread "
            "source. Figures dependent on it should not be relied upon."
        )

    injections = state.get("injection_attempts", [])
    if injections:
        limitations.append(
            f"{len(injections)} prompt-injection attempt(s) were found embedded in "
            "the source documents. They were treated as data and not obeyed."
        )

    if is_exhausted(state):
        limitations.append(f"The run stopped early: {budget_reason(state)}.")

    if state.get("research_mode"):
        # First in the list, in every research-mode report. Web evidence is
        # weaker than a document the reader supplied: it can be stale, wrong,
        # or written to rank rather than to inform — and the reader cannot
        # audit the corpus, because there is not one.
        limitations.insert(
            0,
            "This analysis had no uploaded corpus. Every finding is sourced "
            "from public web pages, which are weaker evidence than a document "
            "you supply: they may be out of date, incorrect, or promotional. "
            "Check the cited sources before acting on any figure.",
        )

    return limitations
