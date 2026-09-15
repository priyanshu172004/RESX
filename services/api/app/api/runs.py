"""Run routes, including the SSE event stream.

The stream is **resumable**. A run takes minutes and outlives the tab that
started it, so a reconnecting client sends `Last-Event-ID` and gets everything
after that sequence number replayed from the store. Buffering events in memory
would make a reconnect lose the middle of the run.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from functools import partial
from typing import Annotated, Any
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.api.deps import (
    RequireAnalyst,
    RunServiceDep,
    SettingsDep,
    StoreDep,
    WorkspaceDep,
    require_model,
)
from app.reporting.export import ExportError, export_report

router = APIRouter(prefix="/api/v1", tags=["runs"])

#: Kept well under typical proxy idle timeouts so a quiet stretch mid-run does
#: not silently drop the connection.
HEARTBEAT_SECONDS = 15


class CreateRunRequest(BaseModel):
    model_config = {"extra": "forbid"}

    question: str = Field(min_length=3, max_length=2000)
    corpus_ids: list[str] = Field(default_factory=list, max_length=200)
    #: Answer from public web sources instead of an uploaded corpus. Needs a
    #: search key. Set explicitly rather than inferred from an empty
    #: `corpus_ids`: "analyse everything I uploaded" and "go and research this"
    #: are different requests, and guessing between them would silently run the
    #: wrong one.
    research: bool = False
    #: Run every specialist rather than the minimum set the Manager plans.
    #:
    #: Defaults on, because the product is a full report and an agent that
    #: never ran cannot have a section in one. Turning it off gives the older,
    #: cheaper behaviour: the Manager decides, and a narrow question may use a
    #: single agent.
    #:
    #: The cost is real and not hidden. Five specialists at two model calls
    #: each, plus the Critic and any debate rounds, is roughly five times a
    #: focused run -- which on a free tier is the difference between a run
    #: finishing and a run waiting out rate limits for several minutes.
    all_agents: bool = True


class CreateRunResponse(BaseModel):
    run_id: str
    status: str
    stream_url: str


@router.post(
    "/runs",
    response_model=CreateRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start an analysis run",
)
async def create_run(
    body: CreateRunRequest,
    workspace_id: WorkspaceDep,
    runs: RunServiceDep,
    store: StoreDep,
    settings: SettingsDep,
) -> CreateRunResponse:
    require_model(settings)

    ready = [
        d["doc_id"]
        for d in store.list_documents(workspace_id=workspace_id)
        if d["status"].startswith("ready")
    ]

    if body.research:
        # Research mode: no corpus, evidence from the web. The citation rule is
        # unchanged -- a web claim carries its URL, publisher and retrieval
        # date -- but the sourcing is weaker, and every report says so.
        if not settings.tavily_api_key:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "research mode needs a web search key. Set TAVILY_API_KEY "
                    "and restart the API, or upload a document and analyse "
                    "that instead."
                ),
            )
        requested: list[str] = []
    elif not ready:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "no ingested documents in this workspace. Upload one, or send "
                '{"research": true} to answer from public web sources instead.'
            ),
        )
    else:
        requested = body.corpus_ids or ready

    unknown = sorted(set(requested) - set(ready))
    if unknown:
        # 404 rather than silently analysing a subset: a run over fewer
        # documents than asked for would produce a quietly wrong answer.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"unknown or unready documents: {unknown}",
        )

    ctx = runs.create(
        workspace_id=workspace_id,
        question=body.question,
        corpus_ids=requested,
        all_agents=body.all_agents,
    )
    runs.start_background(ctx)

    return CreateRunResponse(
        run_id=ctx.run_id,
        status="queued",
        stream_url=f"/api/v1/runs/{ctx.run_id}/events",
    )


@router.get("/runs")
async def list_runs(
    workspace_id: WorkspaceDep,
    store: StoreDep,
    response: Response,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[dict[str, Any]]:
    """One page of runs, newest first.

    The total goes in `X-Total-Count` rather than wrapping the array in an
    envelope. Changing the response body would break every existing caller for
    the sake of a number most of them do not read, and a header carries it
    without touching the shape.
    """
    response.headers["X-Total-Count"] = str(store.count_runs(workspace_id=workspace_id))
    return store.list_runs(workspace_id=workspace_id, limit=limit, offset=offset)


@router.get("/runs/{run_id}")
async def get_run(
    run_id: str, workspace_id: WorkspaceDep, runs: RunServiceDep
) -> dict[str, Any]:
    snapshot = runs.snapshot(workspace_id=workspace_id, run_id=run_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="no such run")
    return snapshot


@router.get("/runs/{run_id}/report")
async def get_report(
    run_id: str, workspace_id: WorkspaceDep, store: StoreDep
) -> dict[str, Any]:
    run = store.get_run(workspace_id=workspace_id, run_id=run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="no such run")
    if run.get("report") is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"this run has no report yet (status: {run['status']})",
        )
    report: dict[str, Any] = run["report"]
    return report


@router.post("/runs/{run_id}/cancel", summary="Stop a run that is still going")
async def cancel_run(
    run_id: str,
    workspace_id: WorkspaceDep,
    runs: RunServiceDep,
    store: StoreDep,
    actor: RequireAnalyst,
) -> dict[str, Any]:
    """Ask a running analysis to stop.

    It stops at the next node boundary rather than immediately. Killing the
    worker mid-node would leave claims written without their citations, and a
    corpus that half-remembers a cancelled run is worse than one that finished
    it — so the run is asked to stop, and it does so within one node.

    Everything the run established before stopping stays readable. A cancelled
    run is not a discarded one.
    """
    run = store.get_run(workspace_id=workspace_id, run_id=run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="no such run")

    if not runs.cancel(workspace_id=workspace_id, run_id=run_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"this run has already finished (status: {run['status']}), so "
                f"there is nothing to cancel"
            ),
        )

    store.audit(
        workspace_id=workspace_id,
        actor=actor.get("user_id"),
        action="run.cancel",
        target=run_id,
        detail={"question": run.get("question")},
    )
    return {
        "run_id": run_id,
        "status": "cancelling",
        "note": "the run will stop at its next node boundary",
    }


@router.get(
    "/runs/{run_id}/export",
    summary="The report as a PDF, Word document, workbook or Markdown file",
)
async def export_run_report(
    run_id: str,
    workspace_id: WorkspaceDep,
    store: StoreDep,
    # Aliased: the query parameter stays `?format=` because that is what a
    # caller would guess, while the Python name avoids shadowing the builtin.
    fmt: Annotated[str, Query(alias="format", description="pdf | docx | xlsx | md")] = "pdf",
) -> Response:
    """Download the finished report.

    Built from the *stored* report rather than re-running anything, so an
    export costs no model tokens and returns the same document the run page is
    showing — a second run would produce a different report and the file would
    no longer match what the reader had on screen.
    """
    run = store.get_run(workspace_id=workspace_id, run_id=run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="no such run")
    if run.get("report") is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"this run has no report yet (status: {run['status']}), so "
                f"there is nothing to export"
            ),
        )

    try:
        # Rendering a dozen charts with matplotlib and laying out a PDF is
        # seconds of CPU. On the event loop that would stall every other
        # request for the duration, which is the same mistake uploads made.
        export = await run_in_threadpool(
            partial(export_report, run["report"], fmt, run_id=run_id)
        )
    except ExportError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return Response(
        content=export.content,
        media_type=export.media_type,
        headers={
            # `filename*` carries the UTF-8 name for a question with non-ASCII
            # characters; `filename` stays as the ASCII fallback for older
            # clients that would otherwise save a mojibake name.
            "Content-Disposition": (
                f'attachment; filename="{export.filename}"; '
                f"filename*=UTF-8''{quote(export.filename)}"
            ),
            "Content-Length": str(len(export.content)),
        },
    )


@router.get("/runs/{run_id}/events", summary="SSE event stream for a run")
async def stream_events(
    run_id: str,
    request: Request,
    workspace_id: WorkspaceDep,
    store: StoreDep,
    runs: RunServiceDep,
) -> StreamingResponse:
    run = store.get_run(workspace_id=workspace_id, run_id=run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="no such run")

    # Resume from where the client left off. Falls back to 0, which replays the
    # whole run — correct for a fresh connection.
    last_seen = 0
    header = request.headers.get("last-event-id")
    if header and header.isdigit():
        last_seen = int(header)

    async def generator() -> AsyncIterator[str]:
        cursor = last_seen
        idle = 0.0

        while True:
            if await request.is_disconnected():
                break

            events = store.read_events(
                workspace_id=workspace_id, run_id=run_id, after_seq=cursor
            )
            for event in events:
                cursor = event["seq"]
                idle = 0.0
                payload = json.dumps(event["payload"], default=str)
                yield f"id: {cursor}\nevent: {event['kind']}\ndata: {payload}\n\n"
                if event["kind"] == "done":
                    return

            current = store.get_run(workspace_id=workspace_id, run_id=run_id)
            terminal = {"done", "failed", "cancelled", "budget_exhausted"}
            finished = current is not None and current["status"] in terminal
            if current is not None and finished and not runs.is_running(run_id) and not events:
                # Emit a terminal event even if the worker died before writing
                # one, so a client is never left waiting on a dead run.
                yield (
                    f"id: {cursor + 1}\nevent: done\n"
                    f"data: {json.dumps({'status': current['status']})}\n\n"
                )
                return

            await asyncio.sleep(0.35)
            idle += 0.35
            if idle >= HEARTBEAT_SECONDS:
                idle = 0.0
                yield ": heartbeat\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            # Stops nginx buffering the stream into uselessness.
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/claims/{claim_id}")
async def get_claim(
    claim_id: str,
    workspace_id: WorkspaceDep,
    store: StoreDep,
    run_id: Annotated[str, Query()],
) -> dict[str, Any]:
    claims = store.list_claims(workspace_id=workspace_id, run_id=run_id)
    for claim in claims:
        if claim["claim_id"] == claim_id:
            return claim
    raise HTTPException(status_code=404, detail="no such claim")


@router.get("/computations/{computation_id}")
async def get_computation(
    computation_id: str, workspace_id: WorkspaceDep, store: StoreDep
) -> dict[str, Any]:
    """The full audit record for one computation.

    Part of the audit surface: with this and the raw upload, any figure in a
    report can be re-derived by a user, not just by an engineer.
    """
    record = store.get_computation(workspace_id=workspace_id, computation_id=computation_id)
    if record is None:
        raise HTTPException(status_code=404, detail="no such computation")
    return record


@router.get("/metrics/grounding")
async def grounding_metrics(workspace_id: WorkspaceDep, store: StoreDep) -> dict[str, Any]:
    """Citation validity for the workspace.

    `docs/04` requires exactly 1.00, so it is a queryable number rather than an
    assumption.
    """
    return store.citation_validity(workspace_id=workspace_id)
