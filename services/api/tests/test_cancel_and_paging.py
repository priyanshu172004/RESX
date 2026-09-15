"""Stopping a run, and reading a corpus that no longer fits on one screen.

Two gaps that only show up once the system is actually used. A run costs real
tokens against a rate-limited tier, so starting the wrong one and having no way
to stop it burns the quota for everything else that day. And a list route with
no limit is fine at two documents and useless at two hundred.

The cancellation rule worth stating: a cancelled run is **not** a failed one.
A failure is something to investigate; a cancellation is something the user
asked for. Reporting the second as the first sends people hunting for a bug
that was never there.
"""

from __future__ import annotations

import io
from typing import Any

import pytest

from app.graph.state import GraphDeps, RunCancelledError
from app.rag.embeddings import HashingEmbedder


def workspace_of(client: Any) -> str:
    return str(client.get("/api/v1/auth/me").json()["workspace"]["workspace_id"])


# --------------------------------------------------------------------------- #
# The checkpoint
# --------------------------------------------------------------------------- #


def test_a_cancelled_run_stops_at_the_next_node_boundary() -> None:
    """Every node begins by emitting `node_start`, so one check there covers
    the whole graph rather than eight that drift apart as nodes are added."""
    deps = GraphDeps(
        llm=None,  # type: ignore[arg-type]
        store=None,  # type: ignore[arg-type]
        embedder=HashingEmbedder(dimensions=32),
        should_cancel=lambda: True,
    )
    with pytest.raises(RunCancelledError):
        deps.event("node_start", {"node": "finance"})


def test_an_uncancelled_run_emits_normally() -> None:
    """The check must not cost the ordinary path anything."""
    seen: list[tuple[str, dict[str, Any]]] = []
    deps = GraphDeps(
        llm=None,  # type: ignore[arg-type]
        store=None,  # type: ignore[arg-type]
        embedder=HashingEmbedder(dimensions=32),
        should_cancel=lambda: False,
        emit=lambda kind, payload: seen.append((kind, payload)),
    )
    deps.event("node_start", {"node": "finance"})
    assert seen == [("node_start", {"node": "finance"})]


def test_a_run_with_no_cancel_hook_is_unaffected() -> None:
    """The benchmark harness and the tests build deps without one."""
    seen: list[str] = []
    deps = GraphDeps(
        llm=None,  # type: ignore[arg-type]
        store=None,  # type: ignore[arg-type]
        embedder=HashingEmbedder(dimensions=32),
        emit=lambda kind, _payload: seen.append(kind),
    )
    deps.event("node_start", {"node": "risk"})
    assert seen == ["node_start"]


# --------------------------------------------------------------------------- #
# The route
# --------------------------------------------------------------------------- #


def test_cancelling_a_finished_run_is_a_conflict_not_a_lie(client: Any) -> None:
    """Returning 200 for a run that already finished would report a
    cancellation that never happened."""
    from app.api import deps

    store = deps.get_store()
    workspace = workspace_of(client)
    store.create_run(workspace_id=workspace, run_id="run_over", question="q", corpus_ids=[])
    store.update_run(workspace_id=workspace, run_id="run_over", status="done")

    response = client.post("/api/v1/runs/run_over/cancel")
    assert response.status_code == 409
    assert "already finished" in str(response.json())


def test_cancelling_an_unknown_run_is_a_404(client: Any) -> None:
    assert client.post("/api/v1/runs/run_nope/cancel").status_code == 404


def test_cancelling_marks_the_run_and_says_what_happens_next(client: Any) -> None:
    from app.api import deps

    store = deps.get_store()
    workspace = workspace_of(client)
    store.create_run(workspace_id=workspace, run_id="run_live", question="q", corpus_ids=[])
    store.update_run(workspace_id=workspace, run_id="run_live", status="analyzing")

    response = client.post("/api/v1/runs/run_live/cancel")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "cancelling"
    # The delay is real, so it is stated rather than left to surprise anyone.
    assert "node boundary" in body["note"]

    stored = store.get_run(workspace_id=workspace, run_id="run_live")
    assert stored is not None
    assert stored["status"] == "cancelling"


def test_another_workspace_cannot_cancel_this_run(client: Any) -> None:
    from app.api import deps

    store = deps.get_store()
    store.create_run(
        workspace_id=workspace_of(client),
        run_id="run_mine",
        question="q",
        corpus_ids=[],
    )
    store.update_run(workspace_id=workspace_of(client), run_id="run_mine", status="analyzing")

    other = client.post(
        "/api/v1/auth/register",
        json={
            "email": "canceller@example.com",
            "password": "correct-horse-battery-staple-9",
            "name": "Other",
        },
    )
    assert other.status_code == 201, other.text
    client.headers["Authorization"] = f"Bearer {other.json()['access_token']}"
    assert client.post("/api/v1/runs/run_mine/cancel").status_code == 404


# --------------------------------------------------------------------------- #
# Paging
# --------------------------------------------------------------------------- #


def upload(client: Any, index: int) -> Any:
    body = f"region,revenue\nR{index},{100 + index}\n".encode()
    return client.post(
        "/api/v1/documents",
        files={"file": (f"doc{index}.csv", io.BytesIO(body), "text/csv")},
    )


def test_documents_page_and_report_the_total(client: Any) -> None:
    for i in range(5):
        assert upload(client, i).status_code == 201

    page = client.get("/api/v1/documents?limit=2")
    assert page.status_code == 200
    assert len(page.json()) == 2
    # The total is what tells a reader there is more; without it a short page
    # is indistinguishable from the end of the list.
    assert page.headers["x-total-count"] == "5"

    second = client.get("/api/v1/documents?limit=2&offset=2")
    assert len(second.json()) == 2
    first_ids = {d["doc_id"] for d in page.json()}
    second_ids = {d["doc_id"] for d in second.json()}
    assert not (first_ids & second_ids), "pages must not overlap"


def test_the_last_page_is_short_rather_than_padded(client: Any) -> None:
    for i in range(3):
        upload(client, i)
    tail = client.get("/api/v1/documents?limit=2&offset=2")
    assert len(tail.json()) == 1
    assert tail.headers["x-total-count"] == "3"


def test_an_offset_past_the_end_is_empty_not_an_error(client: Any) -> None:
    upload(client, 0)
    response = client.get("/api/v1/documents?limit=10&offset=99")
    assert response.status_code == 200
    assert response.json() == []


def test_runs_page_and_report_the_total(client: Any) -> None:
    from app.api import deps

    store = deps.get_store()
    workspace = workspace_of(client)
    for i in range(4):
        store.create_run(
            workspace_id=workspace,
            run_id=f"run_{i}",
            question=f"question {i}",
            corpus_ids=[],
        )

    page = client.get("/api/v1/runs?limit=2")
    assert len(page.json()) == 2
    assert page.headers["x-total-count"] == "4"


def test_internal_callers_still_see_the_whole_corpus(client: Any) -> None:
    """The dedup guard and the chart builder need every document, not a page.

    Defaulting the store to a page would make them quietly wrong — a duplicate
    upload would sail past the guard because its earlier copy was on page two.
    Only the HTTP route pages, because only a reader has a screen.
    """
    from app.api import deps

    for i in range(7):
        upload(client, i)
    everything = deps.get_store().list_documents(workspace_id=workspace_of(client))
    assert len(everything) == 7
