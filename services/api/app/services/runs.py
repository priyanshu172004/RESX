"""Run orchestration.

Owns the lifecycle of one analysis: build the graph, stream its events into the
store, persist the evidence, and record why it stopped.

Events are written to the store as they happen rather than being buffered until
the end. That is what makes the SSE stream resumable — a run outlives the
browser tab that started it, and a reconnecting client replays from the last
sequence number it saw.
"""

from __future__ import annotations

import threading
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.agents.llm import LLM, LLMError, build_llm
from app.agents.schemas import Budget, VerdictKind
from app.analysis.sandbox import Sandbox, build_sandbox
from app.core.config import Settings, get_settings
from app.graph.build import build_graph
from app.graph.state import GraphDeps, RunCancelledError, initial_state, spend_summary
from app.rag.embeddings import Embedder, build_embedder

if TYPE_CHECKING:
    # A `TYPE_CHECKING`-only import: `app.store.factory` pulls in the Mongo
    # backend, and a SQLite-only install must not need pymongo at import time.
    # `from __future__ import annotations` makes every annotation a string, so
    # this costs nothing at runtime and states the truth — these functions
    # accept either backend, and annotating `LocalStore` was the inaccuracy.
    from app.store.factory import Store


@dataclass(slots=True)
class RunContext:
    run_id: str
    workspace_id: str
    question: str
    corpus_ids: list[str]
    #: Run every specialist rather than the minimum set the Manager plans.
    #: Defaults on: a full report is what the product is for, and an agent
    #: that never ran cannot have a section in it. See `fan_out`.
    all_agents: bool = True


#: Statuses after which there is nothing left to cancel. Named rather than
#: repeated so "is this run over" has one answer.
TERMINAL_STATUSES = frozenset({"done", "failed", "cancelled"})

#: A flag for a run that was never started here, so `should_cancel` can ask
#: without a dictionary lookup that might insert.
_NEVER = threading.Event()


class RunService:
    """Starts runs and records everything they produce."""

    def __init__(
        self,
        *,
        store: Store,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        sandbox: Sandbox | None = None,
        llm: LLM | None = None,
    ) -> None:
        self.store = store
        self.settings = settings or get_settings()
        self._embedder = embedder
        self._sandbox = sandbox
        self._llm = llm
        self._threads: dict[str, threading.Thread] = {}
        #: Set when a run is cancelled. Read at every node boundary.
        self._cancelled: dict[str, threading.Event] = {}

    # -- lazily built dependencies ---------------------------------------

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = build_embedder(self.settings)
        return self._embedder

    @property
    def sandbox(self) -> Sandbox:
        if self._sandbox is None:
            self._sandbox = build_sandbox(self.settings)
        return self._sandbox

    @property
    def llm(self) -> LLM:
        # Built on demand so ingestion, retrieval and the benchmark harness all
        # work with no API key configured. Only an actual run needs a model.
        if self._llm is None:
            self._llm = build_llm(self.settings)
        return self._llm

    # -- lifecycle -------------------------------------------------------

    def create(
        self,
        *,
        workspace_id: str,
        question: str,
        corpus_ids: Sequence[str],
        all_agents: bool = True,
    ) -> RunContext:
        run_id = self.store.create_run(
            workspace_id=workspace_id,
            question=question,
            corpus_ids=list(corpus_ids),
            usd_cap=self.settings.run_max_usd,
        )
        return RunContext(
            run_id=run_id,
            workspace_id=workspace_id,
            question=question,
            corpus_ids=list(corpus_ids),
            all_agents=all_agents,
        )

    def start_background(self, ctx: RunContext) -> None:
        """Run the graph on a worker thread.

        A run takes minutes, so it must not occupy the request. In production
        this is an `arq` job on Redis; a thread is the single-node equivalent
        and keeps the same interface.
        """
        thread = threading.Thread(
            target=self._execute_guarded, args=(ctx,), name=f"resx-{ctx.run_id}", daemon=True
        )
        self._threads[ctx.run_id] = thread
        thread.start()

    def is_running(self, run_id: str) -> bool:
        thread = self._threads.get(run_id)
        return bool(thread and thread.is_alive())

    def cancel(self, *, workspace_id: str, run_id: str) -> bool:
        """Ask a run to stop at its next node boundary.

        Returns False when there is nothing running to cancel — an unknown run,
        or one that already finished. The caller turns that into a 404 or a 409
        rather than reporting a cancellation that did not happen.

        The thread is not killed. A node mid-write would leave claims stored
        without their citations, and a corpus that half-remembers a cancelled
        run is worse than one that finished it. Cancelling costs at most one
        more node.
        """
        run = self.store.get_run(workspace_id=workspace_id, run_id=run_id)
        if run is None:
            return False
        if run.get("status") in TERMINAL_STATUSES:
            return False

        self._cancelled.setdefault(run_id, threading.Event()).set()
        self.store.update_run(workspace_id=workspace_id, run_id=run_id, status="cancelling")
        self.store.append_event(
            workspace_id=workspace_id,
            run_id=run_id,
            kind="cancelling",
            payload={"message": "cancelling at the next node boundary"},
        )
        return True

    def _execute_guarded(self, ctx: RunContext) -> None:
        try:
            self.execute(ctx)
        except Exception as exc:
            self.store.append_event(
                workspace_id=ctx.workspace_id,
                run_id=ctx.run_id,
                kind="error",
                payload={
                    "node": "runner",
                    "message": str(exc),
                    "type": type(exc).__name__,
                },
            )
            self.store.update_run(
                workspace_id=ctx.workspace_id,
                run_id=ctx.run_id,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            self.store.append_event(
                workspace_id=ctx.workspace_id,
                run_id=ctx.run_id,
                kind="done",
                payload={"status": "failed"},
            )

    # -- execution -------------------------------------------------------

    def execute(self, ctx: RunContext) -> dict[str, Any]:
        store, ws, run_id = self.store, ctx.workspace_id, ctx.run_id

        def emit(kind: str, payload: dict[str, Any]) -> None:
            store.append_event(workspace_id=ws, run_id=run_id, kind=kind, payload=payload)

        store.update_run(workspace_id=ws, run_id=run_id, status="planning")
        emit("run_start", {"question": ctx.question, "corpus_ids": ctx.corpus_ids})

        try:
            llm = self.llm
            # Check the configured models exist before the graph spends
            # anything. Without this a withdrawn model id is discovered on the
            # Manager node, and the run then completes "successfully" with an
            # empty report and a degraded branch -- which looks like a corpus
            # problem rather than a one-line fix in .env.
            preflight = getattr(llm, "preflight", None)
            if callable(preflight):
                preflight()
        except LLMError as exc:
            # An honest, specific failure. Everything up to the agents works
            # without a key, so the message says exactly what is missing.
            emit("error", {"node": "runner", "message": str(exc)})
            store.update_run(workspace_id=ws, run_id=run_id, status="failed", error=str(exc))
            emit("done", {"status": "failed"})
            return {"status": "failed", "error": str(exc)}

        deps = GraphDeps(
            llm=llm,
            store=store,
            embedder=self.embedder,
            sandbox=self.sandbox,
            dataset_dir=str(self.settings.dataset_dir),
            tavily_api_key=self.settings.tavily_api_key,
            egress_allowlist=self.settings.egress_allowlist,
            top_k=self.settings.retrieval_top_k,
            context_chars=self.settings.retrieval_context_chars,
            max_debate_rounds=self.settings.run_max_debate_rounds,
            emit=emit,
            should_cancel=lambda: self._cancelled.get(run_id, _NEVER).is_set(),
        )

        graph = build_graph(deps)
        state = initial_state(
            run_id=run_id,
            workspace_id=ws,
            question=ctx.question,
            corpus_ids=ctx.corpus_ids,
            all_agents=ctx.all_agents,
            budget=Budget(
                usd_cap=self.settings.run_max_usd,
                token_cap=self.settings.run_max_tokens,
                node_cap=self.settings.run_max_nodes,
                debate_round_cap=self.settings.run_max_debate_rounds,
            ),
        )

        store.update_run(workspace_id=ws, run_id=run_id, status="analyzing")

        try:
            final = graph.invoke(state, {"recursion_limit": 80})
        except RunCancelledError:
            # Asked for, not broken. Recorded as its own status so the run page
            # does not send the reader hunting for a failure that never
            # happened. Whatever the run established before stopping is already
            # persisted and stays readable.
            store.update_run(workspace_id=ws, run_id=run_id, status="cancelled")
            emit("done", {"status": "cancelled"})
            return {"status": "cancelled", "run_id": run_id}
        except Exception as exc:
            emit(
                "error",
                {
                    "node": "graph",
                    "message": str(exc),
                    # Enough frames to reach the actual failure. `limit=3` showed only
                    # graph.invoke -> pregel -> runner.tick, which is the plumbing
                    # rather than the code that broke -- and made a BSON encoding
                    # bug take four runs to locate. Trimmed to the innermost
                    # frames, which are the informative ones.
                    "trace": "".join(traceback.format_exc().splitlines(keepends=True)[-28:]),
                },
            )
            store.update_run(
                workspace_id=ws,
                run_id=run_id,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            emit("done", {"status": "failed"})
            raise

        self._persist(ctx, final)

        spend = spend_summary(final)
        emit("budget", spend)

        report = final.get("report")
        status = final.get("status", "done")
        degraded = list(dict.fromkeys(final.get("degraded", [])))

        # Node failures accumulate in `state["errors"]`. Emitting and recording
        # them is what makes a degraded branch explicable: without this the run
        # reports `error: None` beside `degraded: ["manager"]`, and the reason
        # exists only in a channel nobody reads.
        node_errors = [dict(entry) for entry in final.get("errors", []) if entry]
        for entry in node_errors:
            emit(
                "node_error",
                {
                    "node": entry.get("node", "unknown"),
                    "message": str(entry.get("error", ""))[:2000],
                },
            )

        summary: str | None = None
        if node_errors:
            summary = "; ".join(
                f"{entry.get('node', 'unknown')}: {str(entry.get('error', ''))[:300]}"
                for entry in node_errors
            )[:2000]

        store.update_run(
            workspace_id=ws,
            run_id=run_id,
            status=status,
            report=report.model_dump(mode="json") if report is not None else None,
            degraded=degraded,
            tokens_used=spend["tokens_used"],
            usd_used=spend["usd_used"],
            # Recorded even when the graph completed. A run that produced no
            # insights *because a node failed* is a different outcome from one
            # whose corpus genuinely supported no claims, and the two must not
            # be indistinguishable in the record.
            error=summary,
        )
        if report is not None:
            emit("report", report.model_dump(mode="json"))
        emit("done", {"status": status, "degraded": degraded, "errors": len(node_errors)})

        return {
            "status": status,
            "claims": len(final.get("claims", [])),
            "dropped": len(final.get("dropped_claims", [])),
            "verdicts": len(final.get("verdicts", [])),
            "degraded": degraded,
            "errors": node_errors,
            "spend": spend,
        }

    def _persist(self, ctx: RunContext, final: dict[str, Any]) -> None:
        """Write claims, citations, verdicts and computations to the store.

        Computations are written *before* claims, because a claim carries a
        foreign key to its computation and the store's CHECK constraint means a
        numeric claim cannot land without one.
        """
        ws, run_id = ctx.workspace_id, ctx.run_id

        for record in final.get("computations", []):
            self.store.add_computation(
                workspace_id=ws,
                run_id=run_id,
                agent=str(record.get("agent", "unknown")),
                record=_AsRecord(record),
            )

        for claim in final.get("claims", []):
            self.store.add_claim(workspace_id=ws, run_id=run_id, claim=claim)

        known = {c.claim_id for c in final.get("claims", [])}
        for verdict in final.get("verdicts", []):
            if verdict.claim_id in known:
                self.store.add_verdict(workspace_id=ws, verdict=verdict)

    # -- reads -----------------------------------------------------------

    def snapshot(self, *, workspace_id: str, run_id: str) -> dict[str, Any] | None:
        run = self.store.get_run(workspace_id=workspace_id, run_id=run_id)
        if run is None:
            return None
        claims = self.store.list_claims(workspace_id=workspace_id, run_id=run_id)
        contested = [
            c["claim_id"]
            for c in claims
            if any(v["verdict"] != VerdictKind.CONFIRMED.value for v in c["verdicts"])
        ]
        return {
            **run,
            "running": self.is_running(run_id),
            "claims": claims,
            "contested_claim_ids": contested,
            "grounding": self.store.citation_validity(workspace_id=workspace_id),
        }


class _AsRecord:
    """Adapt a computation dict back to the attribute shape the store expects."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.computation_id = data.get("computation_id", "")
        self.code = data.get("code", "")
        self.inputs = data.get("inputs", [])
        self.stdout = data.get("stdout", "")
        self.stderr = data.get("stderr", "")
        self.result = data.get("result")
        self.duration_ms = data.get("duration_ms", 0)
        self.image_digest = data.get("image_digest", "unknown")
        self.ok = bool(data.get("ok", False))
        self.error = data.get("error")
