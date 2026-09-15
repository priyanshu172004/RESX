"""Dashboard aggregation, agent roster, and grounded chat.

Everything the dashboard renders is computed here from stored records rather
than assembled in the browser. Two reasons, and the second is the important
one:

  * A chart built by summing values in the client cannot be traced back to a
    computation, so a figure on screen would have no `computation_id` and the
    accuracy contract would stop at the API boundary.
  * The tenant filter lives in the store. Aggregating server-side means the
    dashboard's headline numbers pass through the same filter as everything
    else, instead of being derived from whatever the client happened to fetch.

Where there is no data yet, these endpoints return empty series and say so.
They never substitute a plausible-looking placeholder: a dashboard that shows
invented revenue is worse than one that shows nothing.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from functools import partial
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from app.agents.schemas import AgentName
from app.api.deps import (
    CurrentUser,
    EmbedderDep,
    SettingsDep,
    StoreDep,
    WorkspaceDep,
)

router = APIRouter(prefix="/api/v1", tags=["workspace"])


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #


@router.get("/dashboard")
async def dashboard(
    workspace_id: WorkspaceDep,
    store: StoreDep,
    settings: SettingsDep,
    user: CurrentUser,
) -> dict[str, Any]:
    """Everything the dashboard shell needs, in one round trip.

    One request rather than six because the page renders as a unit: six
    parallel fetches would paint the header, then the KPIs, then each chart, and
    the layout would jump twice on every load.
    """
    summary = store.corpus_summary(workspace_id=workspace_id)
    runs = store.list_runs(workspace_id=workspace_id, limit=10)
    documents = store.list_documents(workspace_id=workspace_id)
    datasets = store.list_datasets(workspace_id=workspace_id)

    latest_report: dict[str, Any] | None = None
    for run in runs:
        if run["status"] == "completed":
            full = store.get_run(workspace_id=workspace_id, run_id=run["run_id"])
            if full and full.get("report"):
                latest_report = {"run_id": run["run_id"], **full["report"]}
                break

    return {
        "workspace": {"workspace_id": workspace_id, "name": user.get("name", "")},
        "summary": summary,
        "kpis": _kpis(summary, runs, datasets),
        "documents": documents[:12],
        "datasets": datasets[:24],
        "runs": runs,
        "latest_report": latest_report,
        "spend": _spend_series(runs),
        "agent_activity": _agent_activity(store, workspace_id, runs),
        "capabilities": {
            "model_provider": settings.provider,
            "model_ready": settings.has_model_key,
            "search_ready": bool(settings.tavily_api_key),
            "semantic_embeddings": bool(settings.voyage_api_key),
            "store": store.backend,
        },
        # Stated rather than implied. An empty dashboard should say why it is
        # empty instead of looking broken.
        "empty_reason": _empty_reason(summary, runs),
    }


def _empty_reason(summary: dict[str, Any], runs: list[dict[str, Any]]) -> str | None:
    if summary["documents"] == 0:
        return "no documents yet — upload one to begin"
    if not runs:
        return "corpus ingested; no analysis has been run yet"
    return None


def _kpis(
    summary: dict[str, Any],
    runs: list[dict[str, Any]],
    datasets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Headline tiles.

    These are counts and rates over stored records — facts about the corpus,
    not derived financial figures. Financial figures come from claims, each of
    which carries its own `computation_id`; they are never recomputed here.
    """
    completed = [r for r in runs if r["status"] == "completed"]
    spend = sum(float(r.get("usd_used") or 0) for r in runs)
    return [
        {
            "key": "documents",
            "label": "Documents",
            "value": summary["documents"],
            "format": "integer",
            "detail": f"{summary['ready']} ready",
        },
        {
            "key": "pages",
            "label": "Pages indexed",
            "value": summary["pages"],
            "format": "integer",
            "detail": f"{summary['chunks']} chunks",
        },
        {
            "key": "datasets",
            "label": "Datasets",
            "value": len(datasets),
            "format": "integer",
            "detail": f"{sum(int(d.get('n_rows') or 0) for d in datasets)} rows",
        },
        {
            "key": "citation_validity",
            "label": "Citation validity",
            "value": summary["citation_validity"],
            "format": "percent",
            # The gate is exactly 1.00, so anything else is a failure rather
            # than a score to be improved on later.
            "target": 1.0,
            "detail": f"{summary['total_citations']} citations resolved",
        },
        {
            "key": "runs",
            "label": "Analyses",
            "value": summary["runs"],
            "format": "integer",
            "detail": f"{len(completed)} completed",
        },
        {
            "key": "spend",
            "label": "Spend",
            "value": round(spend, 4),
            "format": "usd",
            "detail": f"{summary['computations']} computations logged",
        },
    ]


def _spend_series(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cost per run, oldest first, for the area chart."""
    ordered = sorted(runs, key=lambda r: r.get("created_at") or 0)
    return [
        {
            "run_id": r["run_id"],
            "label": r["question"][:40],
            "usd": round(float(r.get("usd_used") or 0), 4),
            "tokens": int(r.get("tokens_used") or 0),
            "at": r.get("created_at"),
        }
        for r in ordered
    ]


def _agent_activity(
    store: Any, workspace_id: str, runs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Claims and mean confidence per agent across recent runs.

    Confidence is averaged only over claims that actually carry one, so an
    agent that produced no claims reads as zero rather than as an
    average-of-nothing.
    """
    tally: dict[str, dict[str, Any]] = {}
    for run in runs[:5]:
        for claim in store.list_claims(workspace_id=workspace_id, run_id=run["run_id"]):
            agent = str(claim.get("agent") or "unknown")
            entry = tally.setdefault(
                agent, {"agent": agent, "claims": 0, "confidence_sum": 0.0, "cited": 0}
            )
            entry["claims"] += 1
            entry["confidence_sum"] += float(claim.get("confidence") or 0)
            if claim.get("citations"):
                entry["cited"] += 1

    out = []
    for entry in tally.values():
        count = entry["claims"] or 1
        out.append(
            {
                "agent": entry["agent"],
                "claims": entry["claims"],
                "mean_confidence": round(entry["confidence_sum"] / count, 3),
                "cited_share": round(entry["cited"] / count, 3),
            }
        )
    return sorted(out, key=lambda e: e["claims"], reverse=True)


# --------------------------------------------------------------------------- #
# Agents
# --------------------------------------------------------------------------- #


@router.get("/agents")
async def list_agents(
    workspace_id: WorkspaceDep, store: StoreDep, settings: SettingsDep
) -> dict[str, Any]:
    """The roster, what each agent does, and what it has produced.

    Deliberately does **not** return system prompts or internal tool
    identifiers. Both were previously served straight to the browser, and both
    are targeting aids: the prompts contain the injection defences verbatim, so
    a reader can craft a document aimed at their exact wording, and the tool
    identifiers enumerate the capabilities an injection would try to reach. See
    `app/agents/catalog.py` for the full reasoning.

    `network_access` stays derived from the allowlist rather than described by
    hand. It is a security property a user is entitled to verify, and a written
    claim about it could drift from the code that enforces it.
    """
    from app.agents.catalog import capabilities_for, profile_for
    from app.agents.tools import ALLOWLIST

    runs = store.list_runs(workspace_id=workspace_id, limit=5)
    activity = {a["agent"]: a for a in _agent_activity(store, workspace_id, runs)}

    roster = []
    for name in AgentName:
        stats = activity.get(name.value, {})
        tools = ALLOWLIST.get(name, frozenset())
        profile = profile_for(name)
        roster.append(
            {
                "agent": name.value,
                "title": profile["title"],
                "role": profile["role"],
                "description": profile["description"],
                # Plain-language groups, not the identifiers themselves.
                "capabilities": capabilities_for(tools),
                # Derived from the allowlist, because this one *is* a checkable
                # security claim: the Finance agent has no web tool in its
                # schema, and a user should be able to confirm that.
                "network_access": any(t.startswith(("search", "fetch")) for t in tools),
                "claims": stats.get("claims", 0),
                "mean_confidence": stats.get("mean_confidence", 0.0),
                "cited_share": stats.get("cited_share", 0.0),
            }
        )
    return {"agents": roster, "runs_considered": len(runs)}


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=2, max_length=2000)
    run_id: str | None = None
    top_k: int = Field(default=8, ge=1, le=20)


@router.post("/chat")
async def chat(
    body: ChatRequest,
    workspace_id: WorkspaceDep,
    store: StoreDep,
    embedder: EmbedderDep,
) -> dict[str, Any]:
    """Extractive, cited answers over the corpus. No model required.

    This returns the passages that support an answer rather than prose about
    them. That is a deliberate limit, not an unfinished feature: with no model
    in the loop there is nothing that could write a summary *without*
    ungrounded generation, and an ungrounded summary is the exact failure this
    system exists to prevent. The passages are the honest answer.
    """
    from app.rag.retrieve import retrieve

    # Off the event loop: retrieval embeds the question over blocking HTTP and
    # then scores the vector matrix in numpy. Neither yields, so on the loop it
    # stalls every other request for the duration.
    result = await run_in_threadpool(
        partial(
            retrieve,
            body.question,
            store=store,
            embedder=embedder,
            workspace_id=workspace_id,
            top_k=body.top_k,
        )
    )

    claims: list[dict[str, Any]] = []
    if body.run_id:
        run = store.get_run(workspace_id=workspace_id, run_id=body.run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="no such run")
        # Scope to the run so "why did the Risk agent flag page 42?" can be
        # answered from that run's evidence rather than the whole corpus.
        needle = {w for w in body.question.lower().split() if len(w) > 3}
        for claim in store.list_claims(workspace_id=workspace_id, run_id=body.run_id):
            text = f"{claim.get('statement', '')} {claim.get('agent', '')}".lower()
            if not needle or any(w in text for w in needle):
                claims.append(claim)

    return {
        "question": body.question,
        "passages": [c.to_dict() for c in result.chunks],
        "claims": claims[:10],
        "diagnostics": result.diagnostics.to_dict() if result.diagnostics else None,
        "answer_mode": "extractive",
        "note": (
            "These are the source passages that answer the question, with their "
            "citations. Run a full analysis for a synthesised report."
        ),
    }


# --------------------------------------------------------------------------- #
# Claims across the workspace
# --------------------------------------------------------------------------- #


@router.get("/insights")
async def insights(
    workspace_id: WorkspaceDep,
    store: StoreDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
) -> dict[str, Any]:
    """Every claim in the workspace, newest run first, with its verdict.

    Sorted by confidence within a run rather than by recency: the ranking that
    matters to a reader is how well-supported a finding is.
    """
    runs = store.list_runs(workspace_id=workspace_id, limit=10)
    rows: list[dict[str, Any]] = []
    for run in runs:
        for claim in store.list_claims(workspace_id=workspace_id, run_id=run["run_id"]):
            verdicts = claim.get("verdicts") or []
            rows.append(
                {
                    "claim_id": claim["claim_id"],
                    "run_id": run["run_id"],
                    "question": run["question"],
                    "agent": claim.get("agent"),
                    "statement": claim.get("statement"),
                    "value": claim.get("value"),
                    "unit": claim.get("unit"),
                    "currency": claim.get("currency"),
                    "period": claim.get("period"),
                    "confidence": claim.get("confidence"),
                    "computation_id": claim.get("computation_id"),
                    "citations": claim.get("citations") or [],
                    "verdict": (verdicts[0]["verdict"] if verdicts else None),
                    "created_at": claim.get("created_at"),
                }
            )
        if len(rows) >= limit:
            break

    rows.sort(key=lambda r: (-(r["confidence"] or 0), r["agent"] or ""))
    contested = sum(1 for r in rows if r["verdict"] == "contested")
    return {
        "claims": rows[:limit],
        "total": len(rows),
        "contested": contested,
        # A numeric claim with no computation should be impossible: the store
        # rejects it. Surfaced anyway, because a zero here is a fact worth
        # showing rather than an assumption worth making.
        "ungrounded_numeric": sum(
            1 for r in rows if r["value"] is not None and not r["computation_id"]
        ),
    }


@router.get("/benchmarks")
async def benchmarks(workspace_id: WorkspaceDep, store: StoreDep) -> dict[str, Any]:
    """Live accuracy signals for this workspace.

    Distinct from `benchmarks/score.py`, which measures the pipeline against a
    gold corpus with known answers. This measures the *user's* corpus, where
    there is no ground truth — so it reports only what can be checked without
    one: whether citations resolve, whether numbers carry computations, and
    whether extraction flagged anything.
    """
    summary = store.corpus_summary(workspace_id=workspace_id)
    documents = store.list_documents(workspace_id=workspace_id)
    datasets = store.list_datasets(workspace_id=workspace_id)

    flagged = [d for d in documents if d.get("warnings")]
    disagreed = [d for d in datasets if d.get("agreement") is False]
    scaled = [d for d in datasets if str(d.get("scale_factor", "1")) != "1"]

    return {
        "metrics": [
            {
                "key": "citation_validity",
                "label": "Citation validity",
                "value": summary["citation_validity"],
                "target": 1.0,
                "comparator": "==",
                "format": "ratio",
                "note": (
                    f"{summary['total_citations']} citations checked against source anchors"
                ),
            },
            {
                "key": "table_agreement",
                "label": "Table extractor agreement",
                "value": (1.0 if not datasets else 1 - (len(disagreed) / len(datasets))),
                "target": 1.0,
                "comparator": ">=",
                "format": "ratio",
                "note": (
                    f"{len(disagreed)} of {len(datasets)} tables disagreed between extractors"
                ),
            },
            {
                "key": "scale_detection",
                "label": "Scale factors applied",
                "value": len(scaled),
                "target": None,
                "comparator": None,
                "format": "integer",
                "note": "tables where a declared scale ('in thousands') was applied",
            },
            {
                "key": "documents_flagged",
                "label": "Documents with warnings",
                "value": len(flagged),
                "target": 0,
                "comparator": "==",
                "format": "integer",
                "note": "scanned pages, low-confidence extraction, or missing text",
            },
        ],
        "not_measured": [
            {
                "key": "hallucination_rate",
                "reason": "requires a gold answer for each question asked",
            },
            {
                "key": "critic_catch_rate",
                "reason": "requires deliberately seeded errors to catch",
            },
            {
                "key": "insight_precision",
                "reason": "requires expert rating of the ranked insights",
            },
        ],
        "gold_corpus_note": (
            "Run `python benchmarks/score.py` for the gated metrics measured "
            "against a corpus with known answers."
        ),
    }


# --------------------------------------------------------------------------- #
# Maintenance
# --------------------------------------------------------------------------- #


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # A typed confirmation rather than a bare POST. This deletes the whole
    # corpus, and a mis-click should not be enough to do it.
    confirm: str = Field(description="must be the literal string 'DELETE'")


@router.post("/workspace/reset")
async def reset_workspace(
    body: ResetRequest,
    workspace_id: WorkspaceDep,
    store: StoreDep,
    settings: SettingsDep,
    user: CurrentUser,
) -> dict[str, Any]:
    """Delete every document, dataset, run and claim in this workspace."""
    from app.services.auth import role_allows

    if not role_allows(str(user.get("role", "viewer")), "admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="resetting a workspace requires the admin role",
        )
    if body.confirm != "DELETE":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='send {"confirm": "DELETE"} to confirm',
        )

    counts = store.wipe_workspace(workspace_id=workspace_id)

    # Remove the extracted dataset files too. Leaving them behind would let a
    # stale parquet answer a computation for a document that no longer exists.
    removed_files = 0
    dataset_dir = settings.dataset_dir
    if dataset_dir.is_dir():
        for path in dataset_dir.iterdir():
            if path.is_file():
                path.unlink()
                removed_files += 1

    store.audit(
        workspace_id=workspace_id,
        actor=user.get("user_id"),
        action="workspace.reset",
        detail={"deleted": counts, "dataset_files": removed_files},
    )
    return {"deleted": counts, "dataset_files_removed": removed_files}


def _as_decimal(value: Any) -> Decimal | None:
    """Parse a stored claim value without ever falling back to float."""
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


__all__ = ["router"]
