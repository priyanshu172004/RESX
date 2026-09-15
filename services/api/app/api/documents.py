"""Document and dataset routes.

Upload validation follows `docs/05-SECURITY.md` §4.7: the type decision comes
from **magic bytes**, never the filename or the client's `Content-Type`, and
the body is read in bounded chunks so a lying `Content-Length` cannot exhaust
memory.
"""

from __future__ import annotations

import hashlib
import re
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, HTTPException, Query, Response, UploadFile, status
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.api.deps import (
    EmbedderDep,
    RequireAnalyst,
    RunServiceDep,
    SettingsDep,
    StoreDep,
    WorkspaceDep,
)
from app.ingest.extract import ExtractionError
from app.ingest.pipeline import ingest_file

router = APIRouter(prefix="/api/v1", tags=["documents"])

#: Magic-byte signatures. A PDF that does not begin with %PDF- is not a PDF,
#: whatever it is called.
_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", "application/pdf"),
    (b"PK\x03\x04", "application/zip"),  # xlsx and docx are zip containers
    (b"\xd0\xcf\x11\xe0", "application/vnd.ms-excel"),  # legacy OLE
)

#: Statuses that mean the document is usable. Only these block a re-upload.
READY_STATUSES = frozenset({"ready", "ready_with_warnings"})

#: The status `ingest_file` writes before it starts work.
INGESTING_STATUS = "extracting"

#: How long an in-flight ingest may go **without reporting progress** before
#: it is treated as abandoned rather than slow.
#:
#: This was first written as a cutoff on total duration, which is a guess about
#: how long a document ought to take — and a 230-page PDF with OCR disproved
#: the guess immediately by taking 23 minutes against a 15-minute cutoff. A
#: retry during those last 8 minutes would have deleted the row while the
#: worker was still writing to it.
#:
#: So the clock runs on silence, not on elapsed time. `ingest_file` heartbeats
#: after each phase and after every embedding batch, so a slow ingest stays
#: alive indefinitely while a killed process goes quiet at once. Five minutes
#: is comfortably longer than the gap between two heartbeats even when the
#: embedding provider is making it wait out a 429.
STALE_INGEST_SECONDS = 300.0


def _reuse_payload(store: Any, *, workspace_id: str, doc: dict[str, Any]) -> dict[str, Any]:
    """The ingest response for a document that was already ingested.

    Shaped exactly like `IngestReport.to_dict()` so the caller cannot tell the
    two paths apart — the same fields, read back from the store instead of
    from a run that just happened. The counts are the real ones rather than
    zeroes, because "0 chunks" beside a usable document reads as a failure.

    `duration_ms` is 0 and that is accurate: nothing was processed.
    """
    doc_id = str(doc["doc_id"])
    chunks = store.iter_chunks(workspace_id=workspace_id, doc_ids=[doc_id])
    datasets = store.list_datasets(workspace_id=workspace_id, doc_id=doc_id)
    return {
        "doc_id": doc_id,
        "source_name": str(doc.get("source_name") or ""),
        "kind": str(doc.get("kind") or ""),
        "pages": int(doc.get("page_count") or 0),
        "chunks": len(chunks),
        "table_chunks": sum(1 for c in chunks if c.kind == "table"),
        "datasets": [str(d.get("dataset_id")) for d in datasets],
        # Not re-measured: anchors were verified when this copy was ingested,
        # and re-deriving them here would report a number this request did not
        # establish.
        "anchor_completeness": 1.0,
        "anchor_problems": [],
        "low_confidence_pages": list(doc.get("low_conf_pages") or []),
        "warnings": list(doc.get("warnings") or []),
        "embedded": bool(store.load_vectors(workspace_id=workspace_id, doc_ids=[doc_id])[0]),
        "embedder": str(doc.get("embedder") or ""),
        "duration_ms": 0,
        "status": str(doc.get("status") or "ready"),
    }


_TEXTUAL_SUFFIXES = frozenset({".csv", ".tsv", ".txt", ".md", ".json"})
_ZIP_SUFFIXES = frozenset({".xlsx", ".xlsm", ".docx"})
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class IngestResponse(BaseModel):
    doc_id: str
    source_name: str
    kind: str
    pages: int
    chunks: int
    datasets: list[str]
    anchor_completeness: float
    low_confidence_pages: list[int]
    warnings: list[str]
    embedded: bool
    embedder: str
    duration_ms: int
    status: str
    #: Present only when `?analyse=` was supplied. `started: false` carries the
    #: reason, so a missing model key is reported rather than looking like a
    #: silent failure.
    analysis: dict[str, Any] | None = None


class DocumentSummary(BaseModel):
    doc_id: str
    source_name: str
    kind: str
    page_count: int
    status: str
    low_conf_pages: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


#: Bytes that legitimately appear in text: printable ASCII plus tab, newline,
#: carriage return and form feed.
_TEXT_BYTES = frozenset(bytes(range(0x20, 0x7F)) + b"\t\n\r\f")


def _looks_like_text(head: bytes) -> bool:
    """Does this actually look like text?

    A strict UTF-8 decode is the first and best signal. The tempting fallback —
    "try latin-1" — is worthless as a check, because latin-1 maps every one of
    the 256 byte values and therefore *never* raises: every binary file on
    earth passes it. So the fallback is a printable-byte ratio instead, which
    lets a mis-encoded CSV through while still rejecting a renamed executable.
    """
    if not head:
        return False
    try:
        head.decode("utf-8")
        return True
    except UnicodeDecodeError:
        pass

    if b"\x00" in head:  # NUL is a reliable binary tell
        return False

    printable = sum(1 for byte in head if byte in _TEXT_BYTES)
    return (printable / len(head)) >= 0.90


def _sniff(head: bytes, suffix: str) -> str:
    for signature, mime in _SIGNATURES:
        if head.startswith(signature):
            if mime == "application/zip":
                if suffix not in _ZIP_SUFFIXES:
                    raise HTTPException(
                        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                        detail=(
                            "file is a zip container but does not have an xlsx or "
                            "docx extension; refusing to guess its contents"
                        ),
                    )
                return mime
            return mime

    # No binary signature. Accept only if the content actually looks like text
    # AND the extension is one we can parse — that combination is what keeps a
    # renamed binary out.
    if suffix in _TEXTUAL_SUFFIXES:
        if not _looks_like_text(head):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=(
                    "file has a text extension but its content is binary; "
                    "refusing to parse it as text"
                ),
            )
        return "text/plain"

    raise HTTPException(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        detail=(
            "unsupported file type: the content does not match any accepted format signature"
        ),
    )


def _safe_filename(name: str) -> str:
    cleaned = _SAFE_NAME.sub("_", Path(name).name).strip("._")
    return cleaned or "upload"


@router.post(
    "/documents",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document",
)
async def upload_document(
    workspace_id: WorkspaceDep,
    store: StoreDep,
    embedder: EmbedderDep,
    settings: SettingsDep,
    runs: RunServiceDep,
    # Uploading mutates the corpus and costs embedding calls, so `viewer` is
    # not enough. Declared as a dependency rather than checked in the body:
    # a missing `if` in a handler is invisible in review.
    actor: RequireAnalyst,
    file: Annotated[UploadFile, File(description="pdf, csv, tsv, xlsx, docx or txt")],
    analyse: Annotated[
        str | None,
        Query(
            description=(
                "A question to analyse immediately after ingestion. The upload "
                "returns as soon as the corpus is indexed; the run streams "
                "separately at /runs/{run_id}/events."
            ),
            max_length=2000,
        ),
    ] = None,
) -> IngestResponse:
    original = _safe_filename(file.filename or "upload")
    suffix = Path(original).suffix.lower()

    head = await file.read(4096)
    if not head:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="file is empty")
    mime = _sniff(head, suffix)

    if mime not in settings.upload_allowed_mime and mime not in {
        "application/zip",
        "text/plain",
    }:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"{mime} is not an accepted type",
        )

    # Stream to disk with a hard ceiling. Content-Length is a hint, so the
    # limit is enforced on bytes actually read.
    digest = hashlib.sha256()
    written = 0
    tmp_dir = Path(tempfile.mkdtemp(prefix="resx-upload-"))
    tmp_path = tmp_dir / original

    try:
        with tmp_path.open("wb") as out:
            chunk = head
            while chunk:
                written += len(chunk)
                if written > settings.upload_max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(f"upload exceeds the {settings.upload_max_bytes} byte limit"),
                    )
                digest.update(chunk)
                out.write(chunk)
                chunk = await file.read(1024 * 1024)

        existing = [
            d
            for d in store.list_documents(workspace_id=workspace_id)
            if d["sha256"] == digest.hexdigest()
        ]
        replaced: str | None = None
        reused: dict[str, Any] | None = None
        if existing:
            # Content-addressed, so the same bytes are the same document, and
            # re-ingesting a *finished* one would duplicate every chunk and
            # skew retrieval.
            #
            # But the status decides that, not the mere existence of a row.
            # `ingest_file` writes the row at "extracting" BEFORE doing any
            # work, so a failed ingest — an embedding rate limit, an OCR
            # error — leaves a row behind, and this guard then answered a
            # retry of the same file with "this file is already ingested". It
            # was not. It had failed, and the only way out was to find and
            # delete the document by hand.
            #
            # Worse for a process killed mid-ingest: the row stays at
            # "extracting" forever and that file can never be uploaded again.
            doc = existing[0]
            doc_status = str(doc.get("status") or "")
            # Time since the last sign of life, falling back to creation for a
            # row written before heartbeats existed.
            last_seen = float(doc.get("progress_at") or doc.get("created_at") or 0.0)
            age_seconds = max(0.0, time.time() - last_seen)

            if doc_status in READY_STATUSES:
                # Already ingested and usable, so there is nothing to do to
                # the corpus — but that is not an error, and answering it with
                # a 409 was the bug.
                #
                # Upload and analyse are one action in the UI. Re-uploading the
                # same PDF with a *new question* is the ordinary way to ask a
                # second question about it, and the conflict threw away the
                # question along with the upload. Nothing about the document
                # changes here; the analysis below runs against the copy that
                # is already there.
                reused = doc

            elif doc_status == INGESTING_STATUS and age_seconds < STALE_INGEST_SECONDS:
                # The one conflict that survives. Ingesting alongside a live
                # ingest would write a second set of chunks for the same
                # document, and the first run is about to produce the copy the
                # caller wants anyway.
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=(
                        f"this file is being processed right now as "
                        f"{doc['doc_id']} (last progress {int(age_seconds)}s "
                        f"ago). A large scanned PDF can take 20 minutes or "
                        f"more. Wait for it to finish, then ask your question "
                        f"against it."
                    ),
                )

            else:
                # Failed, or abandoned mid-extraction. The previous attempt
                # left nothing usable, so it is cleared and this upload
                # replaces it.
                store.delete_document(workspace_id=workspace_id, doc_id=doc["doc_id"])
                replaced = doc["doc_id"]

        # Off the event loop, not on it.
        #
        # `ingest_file` is synchronous and CPU-bound: extraction, OCR at 300
        # DPI, chunking, then blocking HTTP to the embedding provider. Called
        # directly from an `async def` handler it blocked the *whole* event
        # loop for the duration, so every other request queued behind it —
        # which is why uploading a PDF made the dashboard time out. The upload
        # was not slow because of the dashboard; the dashboard was starved by
        # the upload.
        #
        # Safe to thread now: the SQLite store serialises its writes behind an
        # RLock. Before that lock existed this would have raced.
        if reused is not None:
            payload = _reuse_payload(store, workspace_id=workspace_id, doc=reused)
            report = None
        else:
            try:
                report = await run_in_threadpool(
                    partial(
                        ingest_file,
                        tmp_path,
                        workspace_id=workspace_id,
                        store=store,
                        embedder=embedder,
                        dataset_dir=settings.dataset_dir,
                    )
                )
            except ExtractionError as exc:
                # The file itself could not be read. This happens before any
                # row is written, so there is nothing to clean up and nothing
                # to retry until the file changes.
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"{original} could not be read: {exc}",
                ) from exc
            except HTTPException:
                raise
            except Exception as exc:
                # Something downstream of extraction failed — an embedding
                # rate limit is the common one. `ingest_file` has already
                # marked the document "failed" with the reason, so say what
                # happened and that the same file can simply be uploaded again;
                # the generic 500 handler would have said "An unexpected error
                # occurred", which tells the caller neither.
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=(
                        f"ingesting {original} failed after extraction "
                        f"({type(exc).__name__}: {exc}). The document is "
                        f"recorded as failed; upload the same file again to "
                        f"retry."
                    ),
                ) from exc
            payload = report.to_dict()
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
            tmp_dir.rmdir()
        except OSError:
            pass

    if reused is not None:
        # The document was already there, so the upload changed nothing about
        # the corpus. Said out loud, because a silent 201 would imply the file
        # had just been ingested and the reader would wonder why it reports a
        # duration of zero.
        payload["warnings"] = [
            *(payload.get("warnings") or []),
            f"this file was already ingested as {reused['doc_id']}, so it was "
            f"not processed again; the existing copy is used as-is",
        ]

    if replaced is not None:
        # Said out loud. The upload succeeded *and* cleared a previous attempt
        # for the same file, and a reader who does not know that will wonder
        # why the earlier document has vanished from their list.
        payload["warnings"] = [
            *(payload.get("warnings") or []),
            f"replaced an earlier upload of this file ({replaced}) that had not "
            f"finished ingesting",
        ]

    if analyse and analyse.strip():
        # Analyse on the spot. Started here rather than by a second client call
        # so the document cannot be uploaded and then silently left unanalysed
        # because the follow-up request was lost.
        if not settings.has_model_key:
            payload["analysis"] = {
                "started": False,
                "reason": (
                    f"no {settings.model_key_env_var} is configured, so the "
                    "document was ingested but not analysed. Search and "
                    "retrieval work without a model key."
                ),
            }
        else:
            ctx = runs.create(
                workspace_id=workspace_id,
                question=analyse.strip(),
                corpus_ids=[str(payload["doc_id"])],
            )
            runs.start_background(ctx)
            payload["analysis"] = {"started": True, "run_id": ctx.run_id}

    store.audit(
        workspace_id=workspace_id,
        actor=actor.get("user_id"),
        action="document.reuse" if reused is not None else "document.upload",
        target=str(payload["doc_id"]),
        detail={
            "source_name": payload.get("source_name"),
            "pages": payload.get("pages"),
        },
    )
    return IngestResponse(**payload)


@router.delete("/documents/{doc_id}", status_code=status.HTTP_200_OK)
async def delete_document(
    doc_id: str,
    workspace_id: WorkspaceDep,
    store: StoreDep,
    settings: SettingsDep,
    actor: RequireAnalyst,
) -> dict[str, Any]:
    """Delete a document and everything derived from it.

    Chunks, pages and datasets go with it. An orphaned chunk is worse than a
    missing document: it stays retrievable, so it would be cited against a
    source the user believes they deleted.
    """
    document = store.get_document(workspace_id=workspace_id, doc_id=doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="no such document")

    datasets = store.list_datasets(workspace_id=workspace_id, doc_id=doc_id)
    store.delete_document(workspace_id=workspace_id, doc_id=doc_id)

    removed = 0
    for dataset in datasets:
        target = Path(dataset.get("path") or "")
        # Only inside the configured dataset directory. A stored path is not
        # user input today, but deleting by a path from the database without
        # bounding it is one schema change away from being a traversal.
        root = settings.dataset_dir.resolve()
        try:
            resolved = target.resolve()
            if resolved.is_file() and resolved.is_relative_to(root):
                resolved.unlink()
                removed += 1
        except (OSError, ValueError):
            continue

    store.audit(
        workspace_id=workspace_id,
        actor=actor.get("user_id"),
        action="document.delete",
        target=doc_id,
        detail={"source_name": document.get("source_name"), "datasets": len(datasets)},
    )
    return {
        "deleted": doc_id,
        "datasets_removed": len(datasets),
        "files_removed": removed,
    }


@router.get("/documents", response_model=list[DocumentSummary])
async def list_documents(
    workspace_id: WorkspaceDep,
    store: StoreDep,
    response: Response,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[DocumentSummary]:
    """One page of documents, newest first.

    The total goes in `X-Total-Count` rather than wrapping the array, so
    existing callers keep working and a reader can still tell there is more.
    """
    response.headers["X-Total-Count"] = str(store.count_documents(workspace_id=workspace_id))
    return [
        DocumentSummary(**d)
        for d in store.list_documents(workspace_id=workspace_id, limit=limit, offset=offset)
    ]


@router.get("/documents/{doc_id}")
async def get_document(
    doc_id: str, workspace_id: WorkspaceDep, store: StoreDep
) -> dict[str, Any]:
    document = store.get_document(workspace_id=workspace_id, doc_id=doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="no such document")
    return {
        **document,
        "datasets": store.list_datasets(workspace_id=workspace_id, doc_id=doc_id),
    }


@router.get("/documents/{doc_id}/pages/{page}")
async def get_page(
    doc_id: str, page: int, workspace_id: WorkspaceDep, store: StoreDep
) -> dict[str, Any]:
    """Extracted page text — what the citation viewer renders."""
    text = store.get_page_text(workspace_id=workspace_id, doc_id=doc_id, page=page)
    if text is None:
        raise HTTPException(status_code=404, detail="no such page")
    return {"doc_id": doc_id, "page": page, "text": text}


@router.get("/datasets")
async def list_datasets(
    workspace_id: WorkspaceDep,
    store: StoreDep,
    doc_id: Annotated[str | None, Query()] = None,
) -> list[dict[str, Any]]:
    return store.list_datasets(workspace_id=workspace_id, doc_id=doc_id)


@router.get("/search")
async def search_corpus(
    workspace_id: WorkspaceDep,
    store: StoreDep,
    embedder: EmbedderDep,
    q: Annotated[str, Query(min_length=2, max_length=1000)],
    top_k: Annotated[int, Query(ge=1, le=20)] = 8,
) -> dict[str, Any]:
    """Hybrid retrieval, exposed directly.

    Useful on its own and it is the honest demonstration of Phase 3: grounded,
    cited retrieval that works with no model in the loop at all.
    """
    from app.rag.retrieve import retrieve

    # Threaded for the same reason as ingestion: retrieval embeds the
    # query over blocking HTTP and then scores a matrix in numpy, neither
    # of which yields to the event loop.
    result = await run_in_threadpool(
        partial(
            retrieve,
            q,
            store=store,
            embedder=embedder,
            workspace_id=workspace_id,
            top_k=top_k,
        )
    )
    return {
        "query": q,
        "diagnostics": result.diagnostics.to_dict() if result.diagnostics else None,
        "chunks": [c.to_dict() for c in result.chunks],
    }
