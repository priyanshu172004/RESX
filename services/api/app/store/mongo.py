"""MongoDB persistence.

This is the **production backend**, specified in `docs/08-DATA-MODEL.md`. The
interface is identical to `LocalStore` (SQLite), which remains the
zero-dependency development fallback, so the two are interchangeable and the
choice is configuration rather than code.

Four properties are carried across from the relational design deliberately,
because each of them is load-bearing rather than incidental:

  * **Every query is tenant-filtered.** `workspace_id` is a required keyword on
    every read and write, and the helper refuses to build a filter without one.
    MongoDB has no row-level security to fall back on, so the application
    filter is the only line of defence — which is exactly why it is enforced by
    the method signatures instead of by convention.
  * **The math rule is a database constraint, not a convention.** A claim
    carrying a value with no `computation_id` is rejected by a collection-level
    `$jsonSchema` validator, mirroring the SQL `CHECK`. If a bug ever bypasses
    the Pydantic validator, the write still fails.
  * **Event sequence numbers are allocated atomically.** `find_one_and_update`
    on a counter document, not `MAX(seq) + 1`. Two agents emitting at once
    would otherwise be handed the same sequence number, and the SSE stream's
    `Last-Event-ID` resumption depends on those being unique.
  * **Embeddings stay bytes.** A float32 buffer in BSON `Binary`, exactly as it
    was a SQLite BLOB. Storing them as BSON arrays of doubles would double the
    storage and make every load a Python-level decode of a million floats.

On vector search: similarity is scored with numpy, exactly as the SQLite
backend does. The tenant filter is part of the query that loads the matrix, so
isolation holds during the search rather than being applied to its results.

`$vectorSearch`, Atlas's own aggregation stage, is **not** used, and the reason
is the bullet above: the stage requires the indexed field to be a BSON array of
numbers, and embeddings here are a float32 buffer in `Binary`. The two choices
are incompatible and only one of them can be had. Binary storage is kept
because it is a quarter of the size and needs no per-float decode, and numpy
scoring is comfortably fast at the scale one workspace reaches — an Atlas
vector index over these documents would match nothing, however correct the
index definition looked.

Adopting `$vectorSearch` means storing vectors as arrays and re-ingesting every
document. Worth doing when a workspace outgrows loading its whole matrix; not
before, and not silently.
"""

from __future__ import annotations

import contextlib
import logging
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import numpy as np

from app.ingest.chunk import Chunk
from app.store.local import RetrievedChunk, StoreError, _require_workspace

log = logging.getLogger("resx.store")

#: What every array in this system is: float32, because that is what the store
#: persists and what the providers return. Named rather than repeated so the
#: dtype is stated once.
FloatArray = np.ndarray[Any, np.dtype[np.float32]]

# --------------------------------------------------------------------------- #
# Collection validators
# --------------------------------------------------------------------------- #

#: The storage-level expression of "the LLM never does arithmetic". A claim may
#: have no value (a qualitative finding), or it may have a value *and* the id of
#: the computation that produced it. It may never have a value alone.
CLAIMS_VALIDATOR: dict[str, Any] = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["claim_id", "workspace_id", "run_id", "agent", "statement", "confidence"],
        "properties": {
            "confidence": {"bsonType": ["double", "int"], "minimum": 0, "maximum": 1},
        },
    },
    "$or": [
        {"value": {"$in": [None]}},
        {"value": {"$exists": False}},
        {"computation_id": {"$type": "string", "$ne": None}},
        # The web case. A figure quoted from a page is quoted, not computed --
        # there is no dataset to compute over in research mode -- and the
        # number must appear verbatim in the cited quote, which `ground_claim`
        # enforces. Omitting this arm made every research-mode numeric claim
        # pass validation and the grounding gate and then fail the INSERT,
        # which killed the run.
        {"source_kind": "external"},
    ],
}

CITATIONS_VALIDATOR: dict[str, Any] = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["citation_id", "workspace_id", "claim_id", "quote", "resolution"],
        "properties": {
            "resolution": {
                "enum": [
                    "ok",
                    "derived_from_computation",
                    "quote_mismatch",
                    "anchor_not_found",
                    "external",
                ],
            },
        },
    }
}

VERDICTS_VALIDATOR: dict[str, Any] = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["verdict_id", "workspace_id", "claim_id", "verdict", "rationale"],
        "properties": {"verdict": {"enum": ["confirmed", "refuted", "contested"]}},
    }
}

VALIDATORS: dict[str, dict[str, Any]] = {
    "claims": CLAIMS_VALIDATOR,
    "citations": CITATIONS_VALIDATOR,
    "verdicts": VERDICTS_VALIDATOR,
}


def bson_safe(value: Any) -> Any:
    """Convert a value into something BSON can encode, losing no precision.

    `Decimal` becomes a string rather than `bson.Decimal128`. Decimal128 is
    limited to 34 significant digits, so a full-precision ratio would be
    silently rounded — and every other `Decimal` in this system is already
    stored as a string (`claims.value`, `datasets.scale_factor`) for exactly
    that reason. One representation, not two.

    Dictionary keys are stringified because BSON requires string keys, and a
    pandas `to_dict()` routinely produces integer ones.
    """
    if value is None or isinstance(value, (str, bool, int, bytes)):
        return value

    if isinstance(value, float):
        # NaN and infinity are valid BSON doubles but meaningless as data, and
        # they compare false against themselves — which makes them very hard to
        # notice once stored.
        if value != value or value in (float("inf"), float("-inf")):
            return str(value)
        return value

    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Mapping):
        return {str(k): bson_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set, frozenset)):
        return [bson_safe(v) for v in value]

    # An unknown object. `str()` rather than dropping it: a value that vanishes
    # is harder to debug than one that arrives as text.
    return str(value)


class MongoStore:
    """MongoDB-backed corpus, run and identity store."""

    #: Mirrors `LocalStore.is_security_boundary` in spirit: callers that care
    #: whether they are on the production backend can ask rather than guess.
    backend = "mongo"

    def __init__(
        self,
        url: str = "mongodb://localhost:27017",
        *,
        database: str = "resx",
        server_selection_timeout_ms: int = 5_000,
        ensure_schema: bool = True,
    ) -> None:
        from pymongo import MongoClient

        self.url = url
        self.database_name = database
        self._client: MongoClient[dict[str, Any]] = MongoClient(
            url,
            serverSelectionTimeoutMS=server_selection_timeout_ms,
            tz_aware=False,
            appname="resx-api",
        )
        self.db = self._client[database]
        if ensure_schema:
            self.ensure_schema()

    # -- plumbing ---------------------------------------------------------

    def ping(self) -> bool:
        try:
            self._client.admin.command("ping")
        except Exception:
            return False
        return True

    def ensure_schema(self) -> None:
        """Create indexes and install the collection validators.

        Idempotent, so it is safe to call on every boot. Index creation is the
        cheap part; the validators are the reason this is not optional — a
        collection created implicitly by a first insert has no validator at
        all, and the math rule would then hold only in Python.
        """
        from pymongo import ASCENDING, DESCENDING
        from pymongo.errors import CollectionInvalid, OperationFailure

        existing = set(self.db.list_collection_names())
        for name, validator in VALIDATORS.items():
            if name in existing:
                try:
                    self.db.command(
                        "collMod", name, validator=validator, validationAction="error"
                    )
                except OperationFailure as exc:  # pragma: no cover - server variance
                    raise StoreError(
                        f"could not install the {name} validator: {exc}. The claims "
                        "validator is what enforces 'no value without a computation' "
                        "at the storage layer, so RESX refuses to run without it."
                    ) from exc
            else:
                # `CollectionInvalid` here means another process created it
                # between the listing and this call, which is fine: the next
                # boot installs the validator via collMod.
                with contextlib.suppress(CollectionInvalid):
                    self.db.create_collection(
                        name, validator=validator, validationAction="error"
                    )

        self.db.documents.create_index(
            [("workspace_id", ASCENDING), ("sha256", ASCENDING)], unique=True
        )
        self.db.documents.create_index([("doc_id", ASCENDING)], unique=True)
        self.db.documents.create_index(
            [("workspace_id", ASCENDING), ("created_at", DESCENDING)]
        )
        self.db.pages.create_index([("doc_id", ASCENDING), ("page", ASCENDING)], unique=True)
        self.db.chunks.create_index([("chunk_id", ASCENDING)], unique=True)
        self.db.chunks.create_index(
            [("workspace_id", ASCENDING), ("doc_id", ASCENDING), ("page", ASCENDING)]
        )
        self.db.datasets.create_index([("dataset_id", ASCENDING)], unique=True)
        self.db.datasets.create_index([("workspace_id", ASCENDING), ("doc_id", ASCENDING)])
        self.db.runs.create_index([("run_id", ASCENDING)], unique=True)
        self.db.runs.create_index([("workspace_id", ASCENDING), ("created_at", DESCENDING)])
        self.db.computations.create_index([("computation_id", ASCENDING)], unique=True)
        self.db.claims.create_index([("claim_id", ASCENDING)], unique=True)
        self.db.claims.create_index([("workspace_id", ASCENDING), ("run_id", ASCENDING)])
        self.db.citations.create_index([("citation_id", ASCENDING)], unique=True)
        self.db.citations.create_index([("claim_id", ASCENDING)])
        self.db.verdicts.create_index([("claim_id", ASCENDING)])
        # Unique on (run_id, seq): the SSE stream resumes from Last-Event-ID, so
        # a duplicate sequence number would silently drop or replay an event.
        self.db.run_events.create_index(
            [("run_id", ASCENDING), ("seq", ASCENDING)], unique=True
        )

        # --- identity ---
        self.db.users.create_index([("user_id", ASCENDING)], unique=True)
        self.db.users.create_index([("email", ASCENDING)], unique=True)
        self.db.refresh_tokens.create_index([("token_hash", ASCENDING)], unique=True)
        self.db.refresh_tokens.create_index([("family_id", ASCENDING)])
        self.db.refresh_tokens.create_index([("user_id", ASCENDING)])
        self.db.invitations.create_index([("token_hash", ASCENDING)], unique=True)
        self.db.invitations.create_index([("invite_id", ASCENDING)], unique=True)
        self.db.invitations.create_index([("workspace_id", ASCENDING), ("email", ASCENDING)])
        self.db.audit_log.create_index([("workspace_id", ASCENDING), ("ts", DESCENDING)])

    @contextmanager
    def tx(self) -> Iterator[Any]:
        """Interface parity with `LocalStore.tx`.

        A real multi-document transaction needs a replica set, which a
        single-node development MongoDB is not. Rather than pretend, this yields
        the database and the writes that must be atomic are expressed as single
        documents instead — which is why a claim embeds nothing and citations
        carry their own `claim_id`.
        """
        yield self.db

    def close(self) -> None:
        self._client.close()

    # -- documents --------------------------------------------------------

    def upsert_document(
        self,
        *,
        workspace_id: str,
        doc_id: str,
        source_name: str,
        kind: str,
        sha256: str,
        page_count: int,
        status: str = "ready",
        low_conf_pages: Sequence[int] = (),
        warnings: Sequence[str] = (),
    ) -> None:
        _require_workspace(workspace_id)
        self.db.documents.update_one(
            {"doc_id": doc_id},
            {
                "$set": {
                    "workspace_id": workspace_id,
                    "source_name": source_name,
                    "kind": kind,
                    "sha256": sha256,
                    "page_count": page_count,
                    "status": status,
                    "low_conf_pages": list(low_conf_pages),
                    "warnings": list(warnings),
                },
                "$setOnInsert": {"created_at": time.time(), "failure_reason": None},
            },
            upsert=True,
        )

    def set_document_status(
        self,
        *,
        workspace_id: str,
        doc_id: str,
        status: str,
        failure_reason: str | None = None,
    ) -> None:
        _require_workspace(workspace_id)
        self.db.documents.update_one(
            {"doc_id": doc_id, "workspace_id": workspace_id},
            {"$set": {"status": status, "failure_reason": failure_reason}},
        )

    def touch_document(self, *, workspace_id: str, doc_id: str) -> None:
        """Record that this ingest is still alive.

        A stale-ingest cutoff measured from `created_at` is a guess about how
        long a document *ought* to take, and a 230-page PDF with OCR
        disproved the first guess by taking 23 minutes. Progress is the honest
        signal: an ingest that is still embedding batches is not abandoned
        however long it has been running, and one whose process was killed
        stops calling this immediately.
        """
        _require_workspace(workspace_id)
        self.db.documents.update_one(
            {"doc_id": doc_id, "workspace_id": workspace_id},
            {"$set": {"progress_at": time.time()}},
        )

    def count_documents(self, *, workspace_id: str) -> int:
        _require_workspace(workspace_id)
        return int(self.db.documents.count_documents({"workspace_id": workspace_id}))

    def list_documents(
        self, *, workspace_id: str, limit: int | None = None, offset: int = 0
    ) -> list[dict[str, Any]]:
        """The workspace's documents, newest first.

        `limit` defaults to None — every document — because the dedup guard and
        the chart builder both need the whole corpus, and silently handing them
        a page would make them wrong rather than slow. Only the HTTP route
        pages, because only a reader has a screen.
        """
        _require_workspace(workspace_id)
        cursor = self.db.documents.find({"workspace_id": workspace_id}, {"_id": 0}).sort(
            "created_at", -1
        )
        if limit is not None:
            cursor = cursor.skip(max(0, offset)).limit(min(limit, 200))
        return [self._document_doc(d) for d in cursor]

    def get_document(self, *, workspace_id: str, doc_id: str) -> dict[str, Any] | None:
        _require_workspace(workspace_id)
        doc = self.db.documents.find_one(
            {"doc_id": doc_id, "workspace_id": workspace_id}, {"_id": 0}
        )
        return self._document_doc(doc) if doc else None

    def delete_document(self, *, workspace_id: str, doc_id: str) -> bool:
        """Remove a document and everything derived from it.

        Cascades explicitly. MongoDB has no `ON DELETE CASCADE`, and orphaned
        chunks are worse than a missing document: they stay retrievable and
        would be cited against a source the user believes they deleted.
        """
        _require_workspace(workspace_id)
        scope = {"workspace_id": workspace_id, "doc_id": doc_id}
        result = self.db.documents.delete_one(scope)
        for collection in (self.db.pages, self.db.chunks, self.db.datasets):
            collection.delete_many(scope)
        return result.deleted_count > 0

    @staticmethod
    def _document_doc(doc: dict[str, Any]) -> dict[str, Any]:
        data = dict(doc)
        data.pop("_id", None)
        data.setdefault("low_conf_pages", [])
        data.setdefault("warnings", [])
        data.setdefault("failure_reason", None)
        return data

    # -- pages ------------------------------------------------------------

    def add_pages(
        self,
        *,
        workspace_id: str,
        doc_id: str,
        pages: Sequence[tuple[int, str, str | None]],
    ) -> None:
        _require_workspace(workspace_id)
        if not pages:
            return
        from pymongo import ReplaceOne

        self.db.pages.bulk_write(
            [
                ReplaceOne(
                    {"doc_id": doc_id, "page": page},
                    {
                        "doc_id": doc_id,
                        "workspace_id": workspace_id,
                        "page": page,
                        "text": text,
                        "section": section,
                    },
                    upsert=True,
                )
                for page, text, section in pages
            ],
            ordered=False,
        )

    def get_page_text(self, *, workspace_id: str, doc_id: str, page: int) -> str | None:
        _require_workspace(workspace_id)
        doc = self.db.pages.find_one(
            {"doc_id": doc_id, "page": page, "workspace_id": workspace_id},
            {"text": 1},
        )
        return doc["text"] if doc else None

    # -- chunks -----------------------------------------------------------

    def add_chunks(
        self,
        *,
        workspace_id: str,
        chunks: Sequence[Chunk],
        embeddings: FloatArray | None = None,
    ) -> int:
        _require_workspace(workspace_id)
        if embeddings is not None and len(embeddings) != len(chunks):
            raise StoreError("embeddings and chunks must be the same length")
        if not chunks:
            return 0

        from bson.binary import Binary
        from pymongo import ReplaceOne

        operations = []
        for i, chunk in enumerate(chunks):
            blob = (
                Binary(np.asarray(embeddings[i], dtype="float32").tobytes())
                if embeddings is not None
                else None
            )
            operations.append(
                ReplaceOne(
                    {"chunk_id": chunk.chunk_id},
                    {
                        "chunk_id": chunk.chunk_id,
                        "workspace_id": workspace_id,
                        "doc_id": chunk.doc_id,
                        "page": chunk.page,
                        "para_idx": chunk.para_idx,
                        "section": chunk.section,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                        "bbox": list(chunk.bbox) if chunk.bbox else None,
                        "text": chunk.text,
                        "token_count": chunk.token_count,
                        "kind": chunk.kind,
                        "table_name": chunk.table_name,
                        "embedding": blob,
                    },
                    upsert=True,
                )
            )
        self.db.chunks.bulk_write(operations, ordered=False)
        return len(operations)

    def get_chunk(self, *, workspace_id: str, chunk_id: str) -> RetrievedChunk | None:
        _require_workspace(workspace_id)
        doc = self.db.chunks.find_one(
            {"chunk_id": chunk_id, "workspace_id": workspace_id}, {"embedding": 0}
        )
        return self._to_retrieved(doc) if doc else None

    def iter_chunks(
        self, *, workspace_id: str, doc_ids: Sequence[str] | None = None
    ) -> list[RetrievedChunk]:
        _require_workspace(workspace_id)
        query: dict[str, Any] = {"workspace_id": workspace_id}
        if doc_ids:
            query["doc_id"] = {"$in": list(doc_ids)}
        cursor = self.db.chunks.find(query, {"embedding": 0}).sort(
            [("doc_id", 1), ("page", 1), ("char_start", 1)]
        )
        return [self._to_retrieved(d) for d in cursor]

    def load_vectors(
        self,
        *,
        workspace_id: str,
        doc_ids: Sequence[str] | None = None,
        dimensions: int | None = None,
    ) -> tuple[list[str], FloatArray]:
        """Load the workspace's embedding matrix for similarity search.

        The tenant filter is part of the query, so isolation holds *inside* the
        vector search rather than being applied to its results afterwards.

        `dimensions` drops vectors of any other width. That is not defensive
        programming for a case that cannot happen — a workspace really can hold
        two widths at once. Changing `EMBEDDING_PROVIDER` or
        `EMBEDDING_DIMENSIONS` and re-ingesting *some* documents leaves the rest
        on the old model, and this store has held 1024-wide vectors from Voyage
        beside 768-wide ones from Gemini in the same database.

        Both outcomes of not filtering are bad: `np.vstack` raises on mismatched
        widths and takes retrieval down, or — worse, if the widths happen to
        line up — the scores are computed across two unrelated embedding spaces
        and are meaningless while looking completely normal. A vector only means
        anything inside its own model's space.

        So the stale ones are skipped and counted. The caller reports the count;
        the remedy is to re-ingest, and nothing here can do that silently.
        """
        _require_workspace(workspace_id)
        query: dict[str, Any] = {
            "workspace_id": workspace_id,
            "embedding": {"$ne": None},
        }
        if doc_ids:
            query["doc_id"] = {"$in": list(doc_ids)}

        ids: list[str] = []
        buffers: list[FloatArray] = []
        skipped = 0
        for doc in self.db.chunks.find(query, {"chunk_id": 1, "embedding": 1}):
            blob = doc.get("embedding")
            if not blob:
                continue
            vector = np.frombuffer(bytes(blob), dtype="float32")
            if dimensions is not None and vector.shape[0] != dimensions:
                skipped += 1
                continue
            ids.append(doc["chunk_id"])
            buffers.append(vector)

        if skipped:
            log.warning(
                "skipped %d chunk(s) embedded at a different width than the "
                "current model (%s); re-ingest those documents to make them "
                "searchable again",
                skipped,
                dimensions,
            )

        if not buffers:
            return [], np.zeros((0, 0), dtype="float32")

        widths = {b.shape[0] for b in buffers}
        if len(widths) > 1:
            # Reached only when the caller did not say which width it wanted.
            # Keeping the largest consistent group beats raising, and beats
            # mixing far more.
            keep = max(widths, key=lambda w: sum(1 for b in buffers if b.shape[0] == w))
            log.warning(
                "workspace holds vectors of widths %s; using %d and ignoring "
                "the rest. Re-ingest so every document shares one model.",
                sorted(widths),
                keep,
            )
            pairs = [(i, b) for i, b in zip(ids, buffers, strict=True) if b.shape[0] == keep]
            ids = [i for i, _ in pairs]
            buffers = [b for _, b in pairs]

        return ids, np.vstack(buffers)

    def set_document_embedder(
        self, *, workspace_id: str, doc_id: str, embedder: str, dimensions: int
    ) -> None:
        """Record which model produced this document's vectors.

        Without it there is no way to answer "is this document searchable with
        the current model" except by measuring a stored vector, and no way at
        all to tell a reader which of their documents went stale when they
        changed provider.
        """
        _require_workspace(workspace_id)
        self.db.documents.update_one(
            {"doc_id": doc_id, "workspace_id": workspace_id},
            {"$set": {"embedder": embedder, "embedding_dimensions": int(dimensions)}},
        )

    @staticmethod
    def _to_retrieved(
        doc: dict[str, Any], score: float = 0.0, source: str = ""
    ) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=doc["chunk_id"],
            doc_id=doc["doc_id"],
            page=doc["page"],
            text=doc["text"],
            char_start=doc["char_start"],
            char_end=doc["char_end"],
            para_idx=doc.get("para_idx"),
            section=doc.get("section"),
            kind=doc.get("kind", "prose"),
            score=score,
            source=source,
        )

    def hydrate(
        self, *, workspace_id: str, scored: Sequence[tuple[str, float, str]]
    ) -> list[RetrievedChunk]:
        """Turn `(chunk_id, score, source)` triples into full chunks, in order."""
        _require_workspace(workspace_id)
        if not scored:
            return []
        ids = [s[0] for s in scored]
        by_id = {
            d["chunk_id"]: d
            for d in self.db.chunks.find(
                {"workspace_id": workspace_id, "chunk_id": {"$in": ids}},
                {"embedding": 0},
            )
        }
        out: list[RetrievedChunk] = []
        for chunk_id, score, source in scored:
            doc = by_id.get(chunk_id)
            if doc is not None:
                out.append(self._to_retrieved(doc, score=score, source=source))
        return out

    # -- datasets ---------------------------------------------------------

    def add_dataset(
        self,
        *,
        workspace_id: str,
        doc_id: str,
        name: str,
        path: str,
        n_rows: int,
        n_cols: int,
        source_page: int | None = None,
        profile: dict[str, Any] | None = None,
        scale_factor: str = "1",
        currency: str | None = None,
        agreement: bool | None = None,
        dataset_id: str | None = None,
    ) -> str:
        _require_workspace(workspace_id)
        resolved = dataset_id or f"ds_{uuid.uuid4().hex[:10]}"
        self.db.datasets.replace_one(
            {"dataset_id": resolved},
            bson_safe(
                {
                    "dataset_id": resolved,
                    "workspace_id": workspace_id,
                    "doc_id": doc_id,
                    "name": name,
                    "source_page": source_page,
                    "path": path,
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "profile": bson_safe(profile or {}),
                    # Kept as a string: the scale factor is exact, and a float would
                    # reintroduce exactly the rounding this system exists to avoid.
                    "scale_factor": str(scale_factor),
                    "currency": currency,
                    "agreement": agreement,
                    "created_at": time.time(),
                }
            ),
            upsert=True,
        )
        return resolved

    def list_datasets(
        self, *, workspace_id: str, doc_id: str | None = None
    ) -> list[dict[str, Any]]:
        _require_workspace(workspace_id)
        query: dict[str, Any] = {"workspace_id": workspace_id}
        if doc_id:
            query["doc_id"] = doc_id
        out = []
        for doc in self.db.datasets.find(query, {"_id": 0}):
            data = dict(doc)
            data.setdefault("profile", {})
            out.append(data)
        return out

    # -- runs -------------------------------------------------------------

    def create_run(
        self,
        *,
        workspace_id: str,
        question: str,
        corpus_ids: Sequence[str],
        usd_cap: float = 5.0,
        run_id: str | None = None,
    ) -> str:
        _require_workspace(workspace_id)
        resolved = run_id or f"run_{uuid.uuid4().hex[:10]}"
        self.db.runs.insert_one(
            bson_safe(
                {
                    "run_id": resolved,
                    "workspace_id": workspace_id,
                    "question": question,
                    "corpus_ids": list(corpus_ids),
                    "status": "queued",
                    "plan": None,
                    "report": None,
                    "degraded": [],
                    "tokens_used": 0,
                    "usd_used": 0.0,
                    "usd_cap": usd_cap,
                    "error": None,
                    "created_at": time.time(),
                    "finished_at": None,
                }
            )
        )
        return resolved

    def update_run(self, *, workspace_id: str, run_id: str, **fields: Any) -> None:
        _require_workspace(workspace_id)
        if not fields:
            return
        allowed = {
            "status",
            "plan",
            "report",
            "degraded",
            "tokens_used",
            "usd_used",
            "error",
            "finished_at",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise StoreError(f"cannot update unknown run fields: {sorted(unknown)}")
        # `bson_safe` rather than `dict`: `report` and `plan` arrive here as
        # nested structures full of computed `Decimal`s, which BSON cannot
        # encode. This was the one that failed in production.
        self.db.runs.update_one(
            {"run_id": run_id, "workspace_id": workspace_id},
            {"$set": bson_safe(dict(fields))},
        )

    def get_run(self, *, workspace_id: str, run_id: str) -> dict[str, Any] | None:
        _require_workspace(workspace_id)
        doc = self.db.runs.find_one(
            {"run_id": run_id, "workspace_id": workspace_id}, {"_id": 0}
        )
        if not doc:
            return None
        data = dict(doc)
        data.setdefault("corpus_ids", [])
        data.setdefault("degraded", [])
        return data

    def count_runs(self, *, workspace_id: str) -> int:
        """How many runs exist, so a caller can page without guessing."""
        _require_workspace(workspace_id)
        return int(self.db.runs.count_documents({"workspace_id": workspace_id}))

    def list_runs(
        self, *, workspace_id: str, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        _require_workspace(workspace_id)
        cursor = (
            self.db.runs.find(
                {"workspace_id": workspace_id},
                {
                    "_id": 0,
                    "run_id": 1,
                    "question": 1,
                    "status": 1,
                    "tokens_used": 1,
                    "usd_used": 1,
                    "created_at": 1,
                    "finished_at": 1,
                },
            )
            .sort("created_at", -1)
            .skip(max(0, offset))
            .limit(min(limit, 100))
        )
        return [dict(d) for d in cursor]

    # -- evidence ---------------------------------------------------------

    def add_computation(
        self, *, workspace_id: str, run_id: str | None, agent: str, record: Any
    ) -> str:
        _require_workspace(workspace_id)
        self.db.computations.replace_one(
            {"computation_id": record.computation_id},
            bson_safe(
                {
                    "computation_id": record.computation_id,
                    "workspace_id": workspace_id,
                    "run_id": run_id,
                    "agent": agent,
                    "code": record.code,
                    "inputs": bson_safe(list(record.inputs)),
                    "stdout": record.stdout,
                    "stderr": record.stderr,
                    "result": bson_safe(record.result),
                    "duration_ms": record.duration_ms,
                    "image_digest": record.image_digest,
                    "ok": bool(record.ok),
                    "error": record.error,
                    "created_at": time.time(),
                }
            ),
            upsert=True,
        )
        computation_id: str = record.computation_id
        return computation_id

    def get_computation(
        self, *, workspace_id: str, computation_id: str
    ) -> dict[str, Any] | None:
        _require_workspace(workspace_id)
        doc = self.db.computations.find_one(
            {"computation_id": computation_id, "workspace_id": workspace_id},
            {"_id": 0},
        )
        return dict(doc) if doc else None

    def add_claim(self, *, workspace_id: str, run_id: str, claim: Any) -> str:
        """Persist a claim and its citations.

        The collection validator is what actually enforces "the LLM never does
        arithmetic" at the storage layer — a numeric claim with no computation
        is rejected by MongoDB here, not merely by the Pydantic model upstream.
        """
        _require_workspace(workspace_id)
        from pymongo.errors import WriteError

        value = None if claim.value is None else str(claim.value)
        try:
            self.db.claims.replace_one(
                {"claim_id": claim.claim_id},
                bson_safe(
                    {
                        "claim_id": claim.claim_id,
                        "workspace_id": workspace_id,
                        "run_id": run_id,
                        "agent": claim.agent,
                        "statement": claim.statement,
                        "value": value,
                        "unit": claim.unit,
                        "period": claim.period,
                        "currency": claim.currency,
                        "scale_factor": str(getattr(claim, "scale_factor", "1")),
                        "computation_id": claim.computation_id,
                        "source_kind": (
                            "external" if claim.is_externally_sourced else "corpus"
                        ),
                        "confidence": float(claim.confidence),
                        "confidence_reason": claim.confidence_reason,
                        "payload": bson_safe(getattr(claim, "payload", {}) or {}),
                        "created_at": time.time(),
                    }
                ),
                upsert=True,
            )
        except WriteError as exc:
            raise StoreError(
                f"claim {claim.claim_id} was rejected by the database: {exc}. "
                "A claim carrying a value must carry the computation_id "
                "that produced it, unless every one of its citations is an "
                "external web source."
            ) from exc

        for citation in claim.citations:
            self.db.citations.replace_one(
                {"citation_id": citation.citation_id},
                bson_safe(
                    {
                        "citation_id": citation.citation_id,
                        "workspace_id": workspace_id,
                        "claim_id": claim.claim_id,
                        "chunk_id": citation.chunk_id,
                        "doc_id": citation.doc_id,
                        "page": citation.page,
                        "para_idx": citation.para_idx,
                        "char_start": citation.char_start,
                        "char_end": citation.char_end,
                        "quote": citation.quote,
                        "url": citation.url,
                        "publisher": citation.publisher,
                        "retrieved_at": time.time(),
                        "resolution": citation.resolution,
                        "match_score": citation.match_score,
                    }
                ),
                upsert=True,
            )
        claim_id: str = claim.claim_id
        return claim_id

    def list_claims(self, *, workspace_id: str, run_id: str) -> list[dict[str, Any]]:
        _require_workspace(workspace_id)
        claims = list(
            self.db.claims.find(
                {"workspace_id": workspace_id, "run_id": run_id}, {"_id": 0}
            ).sort("created_at", 1)
        )
        if not claims:
            return []

        ids = [c["claim_id"] for c in claims]
        # Two queries for the whole page rather than two per claim: the N+1 was
        # invisible on SQLite and would be a round trip each over the network.
        cites: dict[str, list[dict[str, Any]]] = {}
        for doc in self.db.citations.find(
            {"workspace_id": workspace_id, "claim_id": {"$in": ids}}, {"_id": 0}
        ):
            cites.setdefault(doc["claim_id"], []).append(dict(doc))

        verdicts: dict[str, list[dict[str, Any]]] = {}
        for doc in self.db.verdicts.find(
            {"workspace_id": workspace_id, "claim_id": {"$in": ids}}, {"_id": 0}
        ).sort("debate_round", -1):
            verdicts.setdefault(doc["claim_id"], []).append(dict(doc))

        out = []
        for claim in claims:
            data = dict(claim)
            data.setdefault("payload", {})
            data["citations"] = cites.get(data["claim_id"], [])
            data["verdicts"] = verdicts.get(data["claim_id"], [])
            out.append(data)
        return out

    def add_verdict(self, *, workspace_id: str, verdict: Any) -> str:
        _require_workspace(workspace_id)
        verdict_id = f"vd_{uuid.uuid4().hex[:10]}"
        self.db.verdicts.insert_one(
            bson_safe(
                {
                    "verdict_id": verdict_id,
                    "workspace_id": workspace_id,
                    "claim_id": verdict.claim_id,
                    "verdict": verdict.verdict,
                    "independent_value": (
                        None
                        if verdict.independent_value is None
                        else str(verdict.independent_value)
                    ),
                    "rationale": verdict.rationale,
                    "debate_round": verdict.debate_round,
                    "created_at": time.time(),
                }
            )
        )
        return verdict_id

    # -- events -----------------------------------------------------------

    def append_event(
        self, *, workspace_id: str, run_id: str, kind: str, payload: dict[str, Any]
    ) -> int:
        """Append an event, allocating its sequence number atomically.

        `find_one_and_update` rather than `MAX(seq) + 1`: specialists run in
        parallel and emit concurrently, so a read-then-write would hand two of
        them the same number, and the unique index would then reject one event
        outright.
        """
        _require_workspace(workspace_id)
        from pymongo import ReturnDocument

        counter = self.db.event_counters.find_one_and_update(
            {"_id": run_id},
            {"$inc": {"seq": 1}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        if counter is None:  # pragma: no cover - upsert=True guarantees a doc
            raise StoreError(f"could not allocate an event sequence for {run_id}")
        seq = int(counter["seq"])
        self.db.run_events.insert_one(
            bson_safe(
                {
                    "run_id": run_id,
                    "workspace_id": workspace_id,
                    "seq": seq,
                    "kind": kind,
                    "payload": bson_safe(payload),
                    "ts": time.time(),
                }
            )
        )
        return seq

    def read_events(
        self, *, workspace_id: str, run_id: str, after_seq: int = 0
    ) -> list[dict[str, Any]]:
        _require_workspace(workspace_id)
        cursor = self.db.run_events.find(
            {
                "run_id": run_id,
                "workspace_id": workspace_id,
                "seq": {"$gt": after_seq},
            },
            {"_id": 0, "seq": 1, "kind": 1, "payload": 1, "ts": 1},
        ).sort("seq", 1)
        return [dict(d) for d in cursor]

    # -- metrics ----------------------------------------------------------

    def citation_validity(self, *, workspace_id: str) -> dict[str, Any]:
        """Grounding health for the workspace.

        `docs/04` requires citation validity of exactly 1.00, so it is a
        queryable number rather than an assumption.
        """
        _require_workspace(workspace_id)
        total = self.db.citations.count_documents({"workspace_id": workspace_id})
        resolved = self.db.citations.count_documents(
            {
                "workspace_id": workspace_id,
                "resolution": {"$in": ["ok", "derived_from_computation", "external"]},
            }
        )
        return {
            "total_citations": total,
            "resolved": resolved,
            "validity": 1.0 if total == 0 else resolved / total,
        }

    def corpus_summary(self, *, workspace_id: str) -> dict[str, Any]:
        """Counts for the dashboard header, in one place.

        The frontend previously derived these from a fixture. Computing them
        here keeps the dashboard's headline figures on the same tenant filter
        as everything else.
        """
        _require_workspace(workspace_id)
        documents = list(
            self.db.documents.find(
                {"workspace_id": workspace_id}, {"_id": 0, "page_count": 1, "status": 1}
            )
        )
        grounding = self.citation_validity(workspace_id=workspace_id)
        return {
            "documents": len(documents),
            "ready": sum(1 for d in documents if str(d.get("status", "")).startswith("ready")),
            "pages": sum(int(d.get("page_count") or 0) for d in documents),
            "chunks": self.db.chunks.count_documents({"workspace_id": workspace_id}),
            "datasets": self.db.datasets.count_documents({"workspace_id": workspace_id}),
            "runs": self.db.runs.count_documents({"workspace_id": workspace_id}),
            "claims": self.db.claims.count_documents({"workspace_id": workspace_id}),
            "computations": self.db.computations.count_documents(
                {"workspace_id": workspace_id}
            ),
            "citation_validity": grounding["validity"],
            "total_citations": grounding["total_citations"],
        }

    # -- identity ---------------------------------------------------------
    #
    # Users live in the same store as the corpus so that a workspace and its
    # members cannot drift apart across two databases.

    def create_user(
        self,
        *,
        email: str,
        password_hash: str,
        name: str,
        workspace_id: str,
        role: str = "owner",
        verified: bool = False,
    ) -> dict[str, Any]:
        from pymongo.errors import DuplicateKeyError

        user = {
            "user_id": f"usr_{uuid.uuid4().hex[:12]}",
            "email": email.strip().lower(),
            "password_hash": password_hash,
            "name": name.strip(),
            "workspace_id": workspace_id,
            "role": role,
            "verified": verified,
            "failed_logins": 0,
            "locked_until": None,
            "totp_secret": None,
            "created_at": time.time(),
            "last_login_at": None,
        }
        try:
            self.db.users.insert_one(dict(user))
        except DuplicateKeyError as exc:
            raise StoreError("email already registered") from exc
        return {k: v for k, v in user.items() if k != "password_hash"}

    def get_user_by_email(self, email: str) -> dict[str, Any] | None:
        doc = self.db.users.find_one({"email": email.strip().lower()}, {"_id": 0})
        return dict(doc) if doc else None

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        doc = self.db.users.find_one({"user_id": user_id}, {"_id": 0})
        return dict(doc) if doc else None

    def update_user(self, *, user_id: str, **fields: Any) -> None:
        allowed = {
            "password_hash",
            "name",
            "verified",
            "failed_logins",
            "locked_until",
            "totp_secret",
            "pending_totp_secret",
            "recovery_codes",
            "last_login_at",
            "role",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise StoreError(f"cannot update unknown user fields: {sorted(unknown)}")
        if fields:
            self.db.users.update_one({"user_id": user_id}, {"$set": bson_safe(dict(fields))})

    def count_users(self) -> int:
        return self.db.users.count_documents({})

    # -- refresh tokens ---------------------------------------------------

    # -- invitations ------------------------------------------------------

    def create_invitation(
        self,
        *,
        invite_id: str,
        workspace_id: str,
        email: str,
        role: str,
        token_hash: bytes,
        invited_by: str,
        expires_at: float,
    ) -> None:
        _require_workspace(workspace_id)
        self.db.invitations.insert_one(
            {
                "invite_id": invite_id,
                "workspace_id": workspace_id,
                "email": email.strip().lower(),
                "role": role,
                "token_hash": token_hash,
                "invited_by": invited_by,
                "created_at": time.time(),
                "expires_at": expires_at,
                "accepted_at": None,
            }
        )

    def get_invitation(self, token_hash: bytes) -> dict[str, Any] | None:
        """Looked up by the hash of the token, never by id.

        The id is visible in a URL; the token is the secret. Looking up by id
        and comparing afterwards would mean a timing-observable compare on a
        row an attacker chose.
        """
        doc = self.db.invitations.find_one({"token_hash": token_hash}, {"_id": 0})
        return dict(doc) if doc else None

    def list_invitations(self, *, workspace_id: str) -> list[dict[str, Any]]:
        _require_workspace(workspace_id)
        cursor = self.db.invitations.find(
            {"workspace_id": workspace_id}, {"_id": 0, "token_hash": 0}
        ).sort("created_at", -1)
        return [dict(d) for d in cursor]

    def mark_invitation_accepted(self, *, invite_id: str) -> None:
        self.db.invitations.update_one(
            {"invite_id": invite_id}, {"$set": {"accepted_at": time.time()}}
        )

    def revoke_invitation(self, *, workspace_id: str, invite_id: str) -> bool:
        """Deleted outright rather than flagged.

        A revoked-but-present row is one forgotten filter away from still
        working, and this row grants membership of a workspace.
        """
        _require_workspace(workspace_id)
        result = self.db.invitations.delete_one(
            {
                "invite_id": invite_id,
                "workspace_id": workspace_id,
                "accepted_at": None,
            }
        )
        return bool(result.deleted_count)

    def store_refresh_token(
        self,
        *,
        token_hash: bytes,
        user_id: str,
        family_id: str,
        expires_at: float,
        parent_hash: bytes | None = None,
    ) -> None:
        self.db.refresh_tokens.insert_one(
            {
                "token_hash": token_hash,
                "user_id": user_id,
                "family_id": family_id,
                "parent_hash": parent_hash,
                "expires_at": expires_at,
                "used_at": None,
                "revoked": False,
                "created_at": time.time(),
            }
        )

    def get_refresh_token(self, token_hash: bytes) -> dict[str, Any] | None:
        doc = self.db.refresh_tokens.find_one({"token_hash": token_hash}, {"_id": 0})
        return dict(doc) if doc else None

    def mark_refresh_token_used(self, token_hash: bytes) -> None:
        self.db.refresh_tokens.update_one(
            {"token_hash": token_hash}, {"$set": {"used_at": time.time()}}
        )

    def revoke_token_family(self, family_id: str) -> int:
        """Revoke every token in a family.

        Called on reuse detection. A replayed refresh token means the token was
        captured, and the only safe response is to invalidate the whole chain
        rather than the single token that was replayed.
        """
        result = self.db.refresh_tokens.update_many(
            {"family_id": family_id}, {"$set": {"revoked": True}}
        )
        return result.modified_count

    def purge_expired_refresh_tokens(self) -> int:
        result = self.db.refresh_tokens.delete_many({"expires_at": {"$lt": time.time()}})
        return result.deleted_count

    # -- audit ------------------------------------------------------------

    def audit(
        self,
        *,
        workspace_id: str,
        actor: str | None,
        action: str,
        target: str | None = None,
        detail: dict[str, Any] | None = None,
        ip: str | None = None,
    ) -> None:
        """Append-only. Nothing in the application ever updates or deletes."""
        self.db.audit_log.insert_one(
            bson_safe(
                {
                    "audit_id": f"aud_{uuid.uuid4().hex[:12]}",
                    "workspace_id": workspace_id,
                    "actor": actor,
                    "action": action,
                    "target": target,
                    "detail": bson_safe(detail or {}),
                    "ip": ip,
                    "ts": time.time(),
                }
            )
        )

    def read_audit(self, *, workspace_id: str, limit: int = 100) -> list[dict[str, Any]]:
        _require_workspace(workspace_id)
        cursor = (
            self.db.audit_log.find({"workspace_id": workspace_id}, {"_id": 0})
            .sort("ts", -1)
            .limit(min(limit, 500))
        )
        return [dict(d) for d in cursor]

    # -- maintenance ------------------------------------------------------

    def wipe_workspace(self, *, workspace_id: str) -> dict[str, int]:
        """Delete every corpus artefact for a workspace, keeping its users.

        Used by `resx.py reset`. Identity is deliberately untouched: wiping the
        corpus should not log the operator out of their own workspace.
        """
        _require_workspace(workspace_id)
        counts: dict[str, int] = {}
        for name in (
            "documents",
            "pages",
            "chunks",
            "datasets",
            "runs",
            "computations",
            "claims",
            "citations",
            "verdicts",
            "run_events",
        ):
            counts[name] = (
                self.db[name].delete_many({"workspace_id": workspace_id}).deleted_count
            )
        self.db.event_counters.delete_many({})
        return counts
