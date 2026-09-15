"""API integration tests.

Exercises the routes a client actually uses: upload, list, search, and the
guards around starting a run. Upload validation gets the most attention here
because it is the one place untrusted bytes enter the system.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings

GOLD = Path(__file__).resolve().parents[3] / "benchmarks" / "gold" / "synthetic-pnl"


# --------------------------------------------------------------------------- #
# The guard itself
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/api/v1/documents"),
        ("get", "/api/v1/datasets"),
        ("get", "/api/v1/search?q=revenue"),
        ("get", "/api/v1/runs"),
        ("get", "/api/v1/metrics/grounding"),
        ("get", "/api/v1/auth/me"),
        ("post", "/api/v1/runs"),
    ],
)
def test_every_data_route_refuses_an_anonymous_caller(
    anon: TestClient, method: str, path: str
) -> None:
    # Tenancy is not optional. A route that answers without a session would be
    # serving one workspace's documents to anybody who asks.
    response = getattr(anon, method)(path)
    assert response.status_code == 401, f"{method.upper()} {path} -> {response.status_code}"


def test_a_forged_token_is_rejected(anon: TestClient) -> None:
    # Signed with the wrong key entirely: RS256 verification must fail closed.
    anon.headers["Authorization"] = "Bearer not.a.real.token"
    assert anon.get("/api/v1/documents").status_code == 401


def test_one_workspace_cannot_read_another(client: TestClient) -> None:
    """The property that matters most, asserted rather than assumed."""
    csv = b"region,revenue\nnorth,100\n"
    upload = client.post(
        "/api/v1/documents",
        files={"file": ("mine.csv", io.BytesIO(csv), "text/csv")},
    )
    assert upload.status_code in (201, 202), upload.text
    mine = upload.json()["doc_id"]

    # A second account gets its own workspace at registration.
    other = client.post(
        "/api/v1/auth/register",
        json={
            "email": "intruder@example.com",
            "password": "another-long-password",
            "name": "Intruder",
        },
    )
    assert other.status_code == 201
    intruder_token = other.json()["access_token"]

    headers = {"Authorization": f"Bearer {intruder_token}"}
    assert client.get("/api/v1/documents", headers=headers).json() == []
    assert client.get(f"/api/v1/documents/{mine}", headers=headers).status_code == 404


# --------------------------------------------------------------------------- #
# Health and shape
# --------------------------------------------------------------------------- #


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_empty_workspace_lists_nothing(client: TestClient) -> None:
    assert client.get("/api/v1/documents").json() == []


# --------------------------------------------------------------------------- #
# Upload validation — magic bytes, not filenames
# --------------------------------------------------------------------------- #


def test_upload_rejects_an_executable_renamed_as_a_pdf(client: TestClient) -> None:
    # A PE header with a .pdf extension. Trusting the filename or the
    # client-supplied content type would accept this.
    payload = b"MZ\x90\x00\x03\x00\x00\x00" + b"\x00" * 512
    response = client.post(
        "/api/v1/documents",
        files={"file": ("invoice.pdf", io.BytesIO(payload), "application/pdf")},
    )
    assert response.status_code == 415
    assert "signature" in response.json()["error"]["message"].lower()


def test_upload_rejects_a_zip_without_an_office_extension(client: TestClient) -> None:
    payload = b"PK\x03\x04" + b"\x00" * 256
    response = client.post(
        "/api/v1/documents",
        files={"file": ("archive.pdf", io.BytesIO(payload), "application/pdf")},
    )
    assert response.status_code == 415
    assert "zip" in response.json()["error"]["message"].lower()


def test_upload_rejects_an_empty_file(client: TestClient) -> None:
    response = client.post(
        "/api/v1/documents", files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
    )
    assert response.status_code == 400


def test_upload_rejects_binary_masquerading_as_csv(client: TestClient) -> None:
    payload = bytes(range(256)) * 4  # not decodable as text
    response = client.post(
        "/api/v1/documents",
        files={"file": ("data.csv", io.BytesIO(payload), "text/csv")},
    )
    assert response.status_code == 415


def test_upload_accepts_a_csv_and_registers_a_dataset(client: TestClient) -> None:
    csv = b"month,revenue\n2025-01,100\n2025-02,200\n2025-03,300\n"
    response = client.post(
        "/api/v1/documents", files={"file": ("months.csv", io.BytesIO(csv), "text/csv")}
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["chunks"] >= 1
    assert body["anchor_completeness"] == 1.0
    assert body["datasets"], "a table should be registered as a dataset"

    datasets = client.get("/api/v1/datasets").json()
    assert len(datasets) == 1
    assert datasets[0]["n_rows"] == 3


def test_duplicate_upload_reuses_the_existing_document(client: TestClient) -> None:
    """The same bytes are the same document, so the second upload is not an
    error — it resolves to the copy already there.

    It must not re-ingest: a second set of chunks for one document would put
    the same text in retrieval twice, which is what content-addressing exists
    to prevent.
    """
    csv = b"month,revenue\n2025-01,100\n2025-02,200\n"
    first = client.post(
        "/api/v1/documents", files={"file": ("a.csv", io.BytesIO(csv), "text/csv")}
    )
    assert first.status_code == 201

    # Same bytes, different name.
    second = client.post(
        "/api/v1/documents", files={"file": ("b.csv", io.BytesIO(csv), "text/csv")}
    )
    assert second.status_code == 201, second.text
    body = second.json()
    assert body["doc_id"] == first.json()["doc_id"]
    assert any("already ingested" in w for w in body["warnings"])
    # Nothing was processed, and the counts are the real ones rather than
    # zeroes — "0 chunks" beside a usable document reads as a failure.
    assert body["duration_ms"] == 0
    assert body["chunks"] == first.json()["chunks"]
    assert len(client.get("/api/v1/documents").json()) == 1


def test_filename_traversal_is_neutralised(client: TestClient) -> None:
    csv = b"a,b\n1,2\n3,4\n"
    response = client.post(
        "/api/v1/documents",
        files={"file": ("../../../etc/passwd.csv", io.BytesIO(csv), "text/csv")},
    )
    assert response.status_code == 201
    # The stored name must not contain a path.
    name = response.json()["source_name"]
    assert "/" not in name and "\\" not in name and ".." not in name


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #


def test_upload_pdf_then_search_returns_cited_spans(client: TestClient) -> None:
    # Checked at run time, not with `skipif`. A module-level `skipif` is
    # evaluated during collection, which is before the session fixture that
    # generates this corpus has run -- so the test skipped even when the file
    # was about to exist.
    source = GOLD / "synthetic_annual_report_fy2025.pdf"
    if not source.exists():
        pytest.skip(f"gold corpus missing at {source}")
    pdf = source.read_bytes()
    upload = client.post(
        "/api/v1/documents",
        files={"file": ("annual.pdf", io.BytesIO(pdf), "application/pdf")},
    )
    assert upload.status_code == 201, upload.text
    doc_id = upload.json()["doc_id"]
    assert upload.json()["pages"] == 5

    search = client.get("/api/v1/search", params={"q": "total revenue FY2025", "top_k": 5})
    assert search.status_code == 200
    body = search.json()
    assert body["chunks"], "expected retrieved spans"

    # Every result must carry a resolvable anchor — that is what makes it
    # citable.
    for chunk in body["chunks"]:
        assert chunk["doc_id"] == doc_id
        assert chunk["page"] >= 1
        assert chunk["char_end"] > chunk["char_start"]

    # And the diagnostics must say whether retrieval was semantic, so a recall
    # number is never read out of context.
    assert body["diagnostics"]["semantic"] in (True, False)

    page = client.get(f"/api/v1/documents/{doc_id}/pages/2")
    assert page.status_code == 200
    assert "48,920" in page.json()["text"]


def test_search_requires_a_real_query(client: TestClient) -> None:
    assert client.get("/api/v1/search", params={"q": "a"}).status_code == 422


def test_missing_page_is_404(client: TestClient) -> None:
    csv = b"a,b\n1,2\n"
    doc_id = client.post(
        "/api/v1/documents", files={"file": ("x.csv", io.BytesIO(csv), "text/csv")}
    ).json()["doc_id"]
    assert client.get(f"/api/v1/documents/{doc_id}/pages/99").status_code == 404


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #


def test_run_without_documents_is_a_conflict(client: TestClient) -> None:
    response = client.post("/api/v1/runs", json={"question": "Is this profitable?"})
    # No key configured, so the model guard fires first — and it says so
    # specifically rather than returning a generic 500.
    assert response.status_code in (409, 503)
    message = response.json()["error"]["message"]
    assert "GROQ_API_KEY" in message or "no ingested documents" in message


def test_run_rejects_an_unknown_document(client: TestClient) -> None:
    csv = b"a,b\n1,2\n"
    client.post("/api/v1/documents", files={"file": ("y.csv", io.BytesIO(csv), "text/csv")})
    response = client.post(
        "/api/v1/runs",
        json={"question": "Is this profitable?", "corpus_ids": ["doc_does_not_exist"]},
    )
    assert response.status_code in (404, 503)


def test_run_validates_its_body(client: TestClient) -> None:
    assert client.post("/api/v1/runs", json={"question": "hi"}).status_code == 422
    # extra="forbid" is mass-assignment prevention.
    assert (
        client.post(
            "/api/v1/runs", json={"question": "valid question", "usd_cap": 9999}
        ).status_code
        == 422
    )


def test_unknown_run_is_404(client: TestClient) -> None:
    assert client.get("/api/v1/runs/run_nope").status_code == 404
    assert client.get("/api/v1/computations/cmp_nope").status_code == 404


def test_grounding_metric_is_queryable(client: TestClient) -> None:
    body = client.get("/api/v1/metrics/grounding").json()
    assert body["validity"] == 1.0
    assert body["total_citations"] == 0


# --------------------------------------------------------------------------- #
# Security headers on real routes
# --------------------------------------------------------------------------- #


def test_security_headers_present_on_api_routes(client: TestClient) -> None:
    response = client.get("/api/v1/documents")
    for header in (
        "Content-Security-Policy",
        "X-Content-Type-Options",
        "X-Frame-Options",
        "Referrer-Policy",
        "X-Request-Id",
    ):
        assert header in response.headers, f"missing {header}"

    csp = response.headers["Content-Security-Policy"]
    assert "unsafe-inline" not in csp
    assert "unsafe-eval" not in csp


# --------------------------------------------------------------------------- #
# A failed ingest must reach a terminal state
#
# Found in use: Voyage rate-limited the embedding call, `ingest_file` raised
# between setting "extracting" and setting "ready", and nothing reset it. The
# document sat at "extracting" with a NULL failure_reason indefinitely — work
# that was already dead, with no way to find out why.
# --------------------------------------------------------------------------- #


def test_an_embedding_failure_degrades_rather_than_losing_the_document(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Text and tables survive; only semantic retrieval is lost."""
    from app.rag.embeddings import EmbeddingError, HashingEmbedder

    def refuse(self, texts):
        raise EmbeddingError("rate limited (429)")

    monkeypatch.setattr(HashingEmbedder, "embed_documents", refuse)

    csv = b"region,revenue\nnorth,1000\nsouth,2000\n"
    response = client.post(
        "/api/v1/documents",
        files={"file": ("degrade.csv", io.BytesIO(csv), "text/csv")},
    )
    assert response.status_code == 201, response.text
    body = response.json()

    # The document is usable, and says what it lost.
    assert body["status"] == "ready_with_warnings"
    assert body["embedded"] is False
    assert body["chunks"] > 0
    assert any("lexical only" in w for w in body["warnings"]), body["warnings"]

    # And it is listed rather than stranded.
    listed = client.get("/api/v1/documents").json()
    assert listed[0]["status"] == "ready_with_warnings"


def test_a_hard_ingest_failure_marks_the_document_failed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never left in a transient state, whatever the failure."""
    import app.ingest.pipeline as pipeline

    def explode(**kwargs):
        raise RuntimeError("chunker exploded")

    monkeypatch.setattr(pipeline, "_ingest_body", explode)

    csv = b"a,b\n1,2\n"
    response = client.post(
        "/api/v1/documents",
        files={"file": ("boom.csv", io.BytesIO(csv), "text/csv")},
    )
    # A named 502, not the generic 500 handler's "An unexpected error
    # occurred" — which would tell the caller neither what broke nor that
    # the same file can simply be uploaded again.
    assert response.status_code == 502, response.text
    body = str(response.json())
    assert "chunker exploded" in body
    assert "again to retry" in body

    listed = client.get("/api/v1/documents").json()
    assert len(listed) == 1
    # "failed", not "extracting": a reader can act on a failure.
    assert listed[0]["status"] == "failed"

    detail = client.get(f"/api/v1/documents/{listed[0]['doc_id']}").json()
    assert "chunker exploded" in (detail["failure_reason"] or "")


def test_no_document_is_ever_left_extracting(client: TestClient) -> None:
    csv = b"x,y\n3,4\n"
    client.post(
        "/api/v1/documents",
        files={"file": ("fine.csv", io.BytesIO(csv), "text/csv")},
    )
    for doc in client.get("/api/v1/documents").json():
        assert doc["status"] != "extracting", (
            "a transient status with no owner is worse than a failure: the work "
            "is dead and nothing says so"
        )


# --------------------------------------------------------------------------- #
# Starting a run with no corpus: the two request shapes
#
# The bug these lock down was in the browser, not here, and it was invisible
# from either side alone. The New Run page derives the mode it *displays* from
# `effectiveMode` -- which becomes "research" automatically when the workspace
# is empty -- while the submit handler read the raw `mode` state, still on its
# "corpus" default. So the page showed research mode, the user asked a web
# question, and the request went out as `corpus_ids: []`.
#
# The API then refused it, correctly and confusingly:
#
#     no ingested documents in this workspace. Upload one, or send
#     {"research": true} to answer from public web sources instead.
#
# Which reads like a broken product. The screen and the request disagreed, and
# only the request counted. These pin the contract the frontend has to satisfy.
# --------------------------------------------------------------------------- #


@contextmanager
def with_run_keys(client: TestClient) -> Iterator[None]:
    """Give the app the keys a run needs, for the duration of one test.

    The default fixture has none on purpose, and that matters here: the route
    checks the model key first and the search key second, both *before* the
    corpus guard. So a test about the corpus guard written against the bare
    fixture never reaches it — it asserts on a 503 about the model key, or a
    409 about the search key, and passes or fails for the wrong reason. Both
    of those happened while writing these.
    """
    current = client.app.dependency_overrides[get_settings]()
    patched = current.model_copy(
        update={
            "groq_api_key": "gsk_test_not_real",
            "tavily_api_key": "tvly_test_not_real",
        }
    )
    client.app.dependency_overrides[get_settings] = lambda: patched
    try:
        yield
    finally:
        client.app.dependency_overrides[get_settings] = lambda: current


def test_an_empty_corpus_without_research_is_refused(client: TestClient) -> None:
    """The exact response the user saw. Kept as a test so the wording stays
    actionable and the status stays 409 rather than a 500."""
    with with_run_keys(client):
        response = client.post(
            "/api/v1/runs",
            json={"question": "What is the outlook for the Indian EV market in 2025?"},
        )
    assert response.status_code == 409
    detail = response.json().get("detail") or response.json().get("error", {}).get(
        "message", ""
    )
    assert "no ingested documents" in detail
    # The message has to name the fix, because it is the only thing the caller
    # can act on.
    assert "research" in detail


def test_research_true_is_accepted_with_no_corpus(client: TestClient) -> None:
    """What the page should have been sending all along.

    The fixture has no model key, so the run is refused for *that* reason —
    which is the point: it gets past the corpus guard. A 409 about documents
    would mean the guard still rejected it.
    """
    with with_run_keys(client):
        response = client.post(
            "/api/v1/runs",
            json={
                "question": "What is the outlook for the Indian EV market in 2025?",
                "research": True,
            },
        )
    # 202: the run is accepted and executes on a worker, so the response is
    # the handle rather than the result.
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["run_id"]
    assert body["stream_url"].endswith("/events")


def test_research_and_corpus_ids_are_different_requests(client: TestClient) -> None:
    """They are not interchangeable, which is why the mode is sent explicitly
    rather than inferred from an empty `corpus_ids`. Inferring it is what would
    have hidden the frontend bug instead of surfacing it."""
    with with_run_keys(client):
        inferred = client.post(
            "/api/v1/runs",
            json={"question": "a question long enough", "corpus_ids": []},
        )
        explicit = client.post(
            "/api/v1/runs",
            json={"question": "a question long enough", "research": True},
        )
    assert inferred.status_code == 409
    assert "no ingested documents" in str(inferred.json())
    assert explicit.status_code == 202


def test_all_agents_defaults_on(client: TestClient) -> None:
    """A full report is the product, and an agent that never ran cannot have a
    section in one. The field is accepted explicitly so the choice can be
    turned off, but the default is every agent."""
    from app.api.runs import CreateRunRequest

    assert CreateRunRequest(question="a question long enough").all_agents is True
    assert (
        CreateRunRequest(question="a question long enough", all_agents=False).all_agents
        is False
    )


def test_an_unknown_field_is_rejected(client: TestClient) -> None:
    """`extra="forbid"` on the request model: a typo in a client is a 422 here
    rather than a silently ignored option. `all_agents` misspelt as
    `allAgents` would otherwise run the cheap path while the caller believed
    it had asked for the full one."""
    response = client.post(
        "/api/v1/runs",
        json={
            "question": "a question long enough",
            "research": True,
            "allAgents": True,
        },
    )
    assert response.status_code == 422
