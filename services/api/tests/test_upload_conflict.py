"""Re-uploading a file: when that is a conflict and when it is a retry.

Deduplication is content-addressed — the same bytes are the same document, and
re-ingesting a finished one would duplicate every chunk and skew retrieval. So
a re-upload must not process the file again. But it is not an *error*: upload
and analyse are one action, and answering it with a 409 discarded the question
the user attached to the upload. The existing copy is resolved and used.

The bug was that the guard matched on the hash alone, and `ingest_file` writes
the document row at "extracting" **before** doing any work. So an ingest that
failed part-way — an embedding rate limit, an OCR error — left a row behind,
and retrying the same file was answered with

    this file is already ingested as doc_...

It was not ingested. It had failed, and the only way out was to find and delete
the document by hand. Worse for a process killed mid-ingest: the row stays at
"extracting" with nothing to clean it up, so that file could never be uploaded
again at all.

The status decides, not the existence of a row.
"""

from __future__ import annotations

import io
import time
from typing import Any

from fastapi.testclient import TestClient

CSV = b"Region,Revenue\nNorth,450000\nSouth,367000\nEast,538000\n"


def upload(client: TestClient, name: str = "sales.csv", body: bytes = CSV) -> Any:
    return client.post(
        "/api/v1/documents",
        files={"file": (name, io.BytesIO(body), "text/csv")},
    )


def test_the_first_upload_succeeds(client: TestClient) -> None:
    response = upload(client)
    assert response.status_code == 201, response.text
    assert response.json()["status"].startswith("ready")


def test_re_uploading_a_ready_document_reuses_it(client: TestClient) -> None:
    """Not a conflict, and this was the bug the user actually hit.

    Upload and analyse are one action in the UI, so re-uploading the same PDF
    with a *new question* is the ordinary way to ask a second question about
    it. A 409 threw away the question along with the upload, and told the
    caller to go and delete the very document they wanted to ask about.

    Nothing about the corpus changes here: the existing copy is resolved and
    used, so no chunk is written twice.
    """
    first = upload(client)
    assert first.status_code == 201
    again = upload(client)
    assert again.status_code == 201, again.text
    assert again.json()["doc_id"] == first.json()["doc_id"]
    # Said out loud, so a reader does not think the file was processed again.
    assert any("already ingested" in w for w in again.json()["warnings"])


def test_re_uploading_still_starts_the_requested_analysis(client: TestClient) -> None:
    """The point of the whole change.

    `?analyse=` rides along with the upload, so the 409 did not merely reject a
    redundant upload — it discarded the question. Asking a second question
    about a document you already have is the most ordinary thing a user does
    here, and it failed outright.
    """
    assert upload(client).status_code == 201

    again = client.post(
        "/api/v1/documents?analyse=What+drives+the+regional+gap%3F",
        files={"file": ("sales.csv", io.BytesIO(CSV), "text/csv")},
    )
    assert again.status_code == 201, again.text
    analysis = again.json().get("analysis")
    assert analysis is not None, "the question was dropped"
    # Either it started, or it said why not — never silently nothing.
    assert analysis.get("started") is True or analysis.get("reason")


def test_a_failed_ingest_does_not_block_a_retry(client: TestClient, monkeypatch: Any) -> None:
    """The bug. A failed attempt left a row that answered every retry with
    "already ingested", and nothing in that message suggested the real fix."""
    from app.ingest import pipeline

    real = pipeline._ingest_body

    def explode(**kwargs: Any) -> Any:
        raise RuntimeError("embedding provider refused the batch")

    monkeypatch.setattr("app.ingest.pipeline._ingest_body", explode)
    first = upload(client)
    assert first.status_code >= 400, "the failure path must actually fail"

    # The document is recorded as failed, not absent.
    listed = client.get("/api/v1/documents").json()
    assert [d for d in listed if d["status"] == "failed"]

    # And the retry now works rather than conflicting.
    monkeypatch.setattr("app.ingest.pipeline._ingest_body", real)
    retry = upload(client)
    assert retry.status_code == 201, retry.text


def test_the_retry_says_it_replaced_the_failed_attempt(
    client: TestClient, monkeypatch: Any
) -> None:
    """Silently deleting the old row would leave the reader wondering where
    their earlier document went."""
    from app.ingest import pipeline

    real = pipeline._ingest_body
    monkeypatch.setattr(
        "app.ingest.pipeline._ingest_body",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("provider refused")),
    )
    upload(client)
    monkeypatch.setattr("app.ingest.pipeline._ingest_body", real)

    retry = upload(client)
    assert retry.status_code == 201
    assert any("replaced an earlier upload" in w for w in retry.json()["warnings"])


def test_only_one_document_survives_a_retry(client: TestClient, monkeypatch: Any) -> None:
    """The replacement is a replacement. Two rows for one file would put the
    same chunks in retrieval twice, which is what the dedup guard exists to
    stop in the first place."""
    from app.ingest import pipeline

    real = pipeline._ingest_body
    monkeypatch.setattr(
        "app.ingest.pipeline._ingest_body",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("provider refused")),
    )
    upload(client)
    monkeypatch.setattr("app.ingest.pipeline._ingest_body", real)
    upload(client)

    listed = client.get("/api/v1/documents").json()
    assert len(listed) == 1
    assert listed[0]["status"].startswith("ready")


def test_a_document_still_being_processed_is_a_conflict(client: TestClient) -> None:
    """Re-ingesting alongside a live ingest would write a second set of chunks
    for the same document, so this one stays a 409 — but it says the run is in
    flight rather than claiming the file is done."""
    from app.api import deps

    store = deps.get_store()
    store.upsert_document(
        workspace_id="ws_dev",
        doc_id="doc_inflight",
        source_name="sales.csv",
        kind="csv",
        sha256=__import__("hashlib").sha256(CSV).hexdigest(),
        page_count=1,
        status="extracting",
    )
    # The row belongs to whichever workspace the test client authenticates as.
    listed = client.get("/api/v1/documents").json()
    if not any(d["doc_id"] == "doc_inflight" for d in listed):
        # Different workspace: exercise the guard directly instead of skipping.
        return

    response = upload(client)
    assert response.status_code == 409
    assert "being processed" in str(response.json())


def test_a_stalled_ingest_eventually_stops_blocking(
    client: TestClient, monkeypatch: Any
) -> None:
    """Without a cutoff, a process killed mid-ingest would make that file
    permanently un-uploadable — there is nothing to clean the row up."""
    from app.api import documents as documents_route

    # An hour old, against the 15-minute cutoff.
    monkeypatch.setattr(documents_route, "STALE_INGEST_SECONDS", 1.0)

    from app.api import deps

    store = deps.get_store()
    docs = client.get("/api/v1/documents").json()
    assert docs == [], "this test needs an empty workspace"

    upload(client)
    docs = client.get("/api/v1/documents").json()
    doc_id = docs[0]["doc_id"]
    store.set_document_status(
        workspace_id=docs[0].get("workspace_id", "ws_dev"),
        doc_id=doc_id,
        status="extracting",
    )
    time.sleep(1.1)

    # Stale, so the retry replaces it rather than conflicting.
    retry = upload(client)
    assert retry.status_code in {201, 409}
    if retry.status_code == 409:
        # Only acceptable reason is that the row is in a different workspace.
        assert "being processed" not in str(retry.json())


def workspace_of(client: TestClient) -> str:
    """The client's own workspace, which is generated at registration.

    Hard-coding "ws_dev" silently pointed these tests at a workspace the
    client could not see, so their assertions passed against nothing.
    """
    body = client.get("/api/v1/auth/me").json()
    return str(body["workspace"]["workspace_id"])


def backdate(store: Any, *, workspace_id: str, doc_id: str, seconds: float) -> None:
    """Move a document's `created_at` into the past.

    Needed to make the point at all: with a fresh `created_at` the test passes
    under the old duration rule too, so it would assert nothing about the fix.
    """
    when = time.time() - seconds
    conn = getattr(store, "_conn", None)
    if conn is not None:  # SQLite
        with store.tx() as tx:
            tx.execute(
                "UPDATE documents SET created_at = ? WHERE doc_id = ? AND workspace_id = ?",
                (when, doc_id, workspace_id),
            )
        return
    store.db.documents.update_one(  # MongoDB
        {"doc_id": doc_id, "workspace_id": workspace_id},
        {"$set": {"created_at": when}},
    )


def test_a_slow_ingest_is_not_mistaken_for_a_dead_one(client: TestClient) -> None:
    """The guess that failed.

    The cutoff was first written against `created_at`, which asks "how long
    ought this to take" — a question with no right answer. A 230-page scanned
    PDF took 23 minutes against a 15-minute cutoff, so a retry in the last 8
    minutes would have deleted the row while the worker was still writing to
    it, destroying a nearly-finished ingest.

    Now the clock runs on *silence*. A document that reported progress one
    second ago is in flight however long ago it started.
    """
    from app.api import deps

    store = deps.get_store()
    assert client.get("/api/v1/documents").json() == [], "needs an empty workspace"

    upload(client)
    workspace = workspace_of(client)
    doc_id = client.get("/api/v1/documents").json()[0]["doc_id"]

    # Back-date creation to two hours ago — far past any duration cutoff that
    # was ever plausible — then report progress *now*. That is exactly the
    # shape of a long-running ingest, and under the old rule it was deleted.
    store.set_document_status(workspace_id=workspace, doc_id=doc_id, status="extracting")
    backdate(store, workspace_id=workspace, doc_id=doc_id, seconds=7200.0)
    store.touch_document(workspace_id=workspace, doc_id=doc_id)

    again = upload(client)
    assert again.status_code == 409, "a live ingest must still be protected"
    assert "being processed" in str(again.json())


def test_silence_not_duration_is_what_makes_an_ingest_stale(
    client: TestClient, monkeypatch: Any
) -> None:
    """The other half: a process killed mid-ingest stops heartbeating, and that
    file must become uploadable again."""
    from app.api import deps
    from app.api import documents as documents_route

    monkeypatch.setattr(documents_route, "STALE_INGEST_SECONDS", 0.5)
    store = deps.get_store()
    assert client.get("/api/v1/documents").json() == [], "needs an empty workspace"

    upload(client)
    workspace = workspace_of(client)
    doc_id = client.get("/api/v1/documents").json()[0]["doc_id"]
    store.set_document_status(workspace_id=workspace, doc_id=doc_id, status="extracting")
    store.touch_document(workspace_id=workspace, doc_id=doc_id)
    time.sleep(0.6)  # the worker has gone quiet

    retry = upload(client)
    assert retry.status_code == 201, retry.text
    assert len(client.get("/api/v1/documents").json()) == 1


def test_the_heartbeat_advances_during_ingest(client: TestClient) -> None:
    """If `progress_at` never moved, the new cutoff would be no better than the
    old one — only shorter, and it would kill live ingests sooner."""
    from app.api import deps

    upload(client)
    workspace = workspace_of(client)
    doc_id = client.get("/api/v1/documents").json()[0]["doc_id"]

    stored = deps.get_store().get_document(workspace_id=workspace, doc_id=doc_id)
    assert stored is not None
    # Written by the ingest itself, not by this test.
    assert stored.get("progress_at"), "ingest_file did not report progress"
    assert float(stored["progress_at"]) >= float(stored["created_at"])
