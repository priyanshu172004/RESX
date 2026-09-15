"""Local persistence (SQLite + numpy vectors).

This is the **development and single-node backend**. The production target is
MongoDB, specified in `docs/08-DATA-MODEL.md` and implemented in
`app/store/mongo.py`; the interface here mirrors it exactly so the choice of
backend is configuration rather than a rewrite.

Keeping this backend is deliberate rather than legacy: the test suite and the
benchmark harness run against it with no server of any kind, which is what
makes `pytest` and `benchmarks/score.py` work on a clean checkout.

One property is carried over exactly, because it is the unrecoverable failure
mode: **every query is tenant-filtered.** `workspace_id` is a required
parameter on every read and write, not an optional filter, and the helpers
refuse to build a query without it. SQLite has no row-level security to fall
back on, so at this layer the application filter is the *only* line of defence
— which is precisely why it is enforced by the method signatures instead of by
convention.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.ingest.chunk import Chunk

log = logging.getLogger("resx.store")

#: What every array in this system is: float32, because that is what the store
#: persists and what the providers return. Named rather than repeated so the
#: dtype is stated once.
FloatArray = np.ndarray[Any, np.dtype[np.float32]]

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    doc_id        TEXT PRIMARY KEY,
    workspace_id  TEXT NOT NULL,
    source_name   TEXT NOT NULL,
    kind          TEXT NOT NULL,
    sha256        TEXT NOT NULL,
    page_count    INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'queued',
    failure_reason TEXT,
    low_conf_pages TEXT NOT NULL DEFAULT '[]',
    warnings      TEXT NOT NULL DEFAULT '[]',
    created_at    REAL NOT NULL,
    -- Last sign of life from an in-flight ingest. NULL until it reports.
    progress_at   REAL,
    -- Which model produced this document's vectors. A vector only means
    -- anything inside its own model's space, so a document embedded by a
    -- different one is not searchable alongside the rest.
    embedder      TEXT,
    embedding_dimensions INTEGER,
    UNIQUE (workspace_id, sha256)
);
CREATE INDEX IF NOT EXISTS idx_documents_ws ON documents (workspace_id, status);

CREATE TABLE IF NOT EXISTS pages (
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL,
    page         INTEGER NOT NULL,
    text         TEXT NOT NULL,
    section      TEXT,
    PRIMARY KEY (doc_id, page)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    page         INTEGER NOT NULL,
    para_idx     INTEGER,
    section      TEXT,
    char_start   INTEGER NOT NULL,
    char_end     INTEGER NOT NULL,
    bbox         TEXT,
    text         TEXT NOT NULL,
    token_count  INTEGER NOT NULL,
    kind         TEXT NOT NULL DEFAULT 'prose',
    table_name   TEXT,
    embedding    BLOB,
    CHECK (char_end > char_start)
);
CREATE INDEX IF NOT EXISTS idx_chunks_ws ON chunks (workspace_id, doc_id, page);

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id    TEXT PRIMARY KEY,
    workspace_id  TEXT NOT NULL,
    doc_id        TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    source_page   INTEGER,
    path          TEXT NOT NULL,
    n_rows        INTEGER NOT NULL,
    n_cols        INTEGER NOT NULL,
    profile       TEXT NOT NULL DEFAULT '{}',
    scale_factor  TEXT NOT NULL DEFAULT '1',
    currency      TEXT,
    agreement     INTEGER,
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_datasets_ws ON datasets (workspace_id, doc_id);

CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    workspace_id  TEXT NOT NULL,
    question      TEXT NOT NULL,
    corpus_ids    TEXT NOT NULL DEFAULT '[]',
    status        TEXT NOT NULL DEFAULT 'queued',
    plan          TEXT,
    report        TEXT,
    degraded      TEXT NOT NULL DEFAULT '[]',
    tokens_used   INTEGER NOT NULL DEFAULT 0,
    usd_used      REAL NOT NULL DEFAULT 0,
    usd_cap       REAL NOT NULL DEFAULT 5.0,
    error         TEXT,
    created_at    REAL NOT NULL,
    finished_at   REAL
);
CREATE INDEX IF NOT EXISTS idx_runs_ws ON runs (workspace_id, created_at DESC);

CREATE TABLE IF NOT EXISTS computations (
    computation_id TEXT PRIMARY KEY,
    workspace_id   TEXT NOT NULL,
    run_id         TEXT,
    agent          TEXT NOT NULL,
    code           TEXT NOT NULL,
    inputs         TEXT NOT NULL DEFAULT '[]',
    stdout         TEXT,
    stderr         TEXT,
    result         TEXT,
    duration_ms    INTEGER,
    image_digest   TEXT NOT NULL,
    ok             INTEGER NOT NULL DEFAULT 0,
    error          TEXT,
    created_at     REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id       TEXT PRIMARY KEY,
    workspace_id   TEXT NOT NULL,
    run_id         TEXT NOT NULL,
    agent          TEXT NOT NULL,
    statement      TEXT NOT NULL,
    value          TEXT,
    unit           TEXT,
    period         TEXT,
    currency       TEXT,
    scale_factor   TEXT DEFAULT '1',
    computation_id TEXT,
    confidence     REAL NOT NULL,
    confidence_reason TEXT,
    payload        TEXT NOT NULL DEFAULT '{}',
    -- 'corpus' or 'external'. Persisted rather than derived because the CHECK
    -- below has to see it: SQLite cannot join to the citations table from a
    -- constraint.
    source_kind    TEXT NOT NULL DEFAULT 'corpus',
    created_at     REAL NOT NULL,
    -- The math rule, expressed as a storage constraint: a claim carrying a
    -- value with no computation cannot be persisted, even if a bug bypasses
    -- the Pydantic validator. Mirrors docs/08-DATA-MODEL.md.
    --
    -- The `source_kind` arm is the web case, and leaving it out was a live
    -- bug: `Claim` was relaxed to allow a quoted web figure with no
    -- computation (there is no dataset to compute over in research mode), the
    -- grounding gate accepted it, and then the INSERT failed and took the
    -- whole run down. Two layers stating the same rule is the design; the two
    -- of them stating *different* rules is a crash.
    CHECK (
        value IS NULL
        OR computation_id IS NOT NULL
        OR source_kind = 'external'
    ),
    CHECK (confidence BETWEEN 0 AND 1)
);
CREATE INDEX IF NOT EXISTS idx_claims_run ON claims (workspace_id, run_id);

CREATE TABLE IF NOT EXISTS citations (
    citation_id  TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    claim_id     TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    chunk_id     TEXT,
    doc_id       TEXT,
    page         INTEGER,
    para_idx     INTEGER,
    char_start   INTEGER,
    char_end     INTEGER,
    quote        TEXT NOT NULL,
    url          TEXT,
    publisher    TEXT,
    retrieved_at REAL,
    resolution   TEXT NOT NULL,
    match_score  REAL,
    CHECK (resolution IN ('ok','derived_from_computation','quote_mismatch',
                          'anchor_not_found','external'))
);
CREATE INDEX IF NOT EXISTS idx_citations_claim ON citations (claim_id);

CREATE TABLE IF NOT EXISTS verdicts (
    verdict_id        TEXT PRIMARY KEY,
    workspace_id      TEXT NOT NULL,
    claim_id          TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    verdict           TEXT NOT NULL,
    independent_value TEXT,
    rationale         TEXT NOT NULL,
    debate_round      INTEGER NOT NULL DEFAULT 0,
    created_at        REAL NOT NULL,
    CHECK (verdict IN ('confirmed','refuted','contested'))
);

CREATE TABLE IF NOT EXISTS run_events (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id   TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    seq      INTEGER NOT NULL,
    kind     TEXT NOT NULL,
    payload  TEXT NOT NULL,
    ts       REAL NOT NULL,
    UNIQUE (run_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_events_run ON run_events (run_id, seq);

CREATE TABLE IF NOT EXISTS users (
    user_id       TEXT PRIMARY KEY,
    email         TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    name          TEXT NOT NULL,
    workspace_id  TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'owner',
    verified      INTEGER NOT NULL DEFAULT 0,
    failed_logins INTEGER NOT NULL DEFAULT 0,
    locked_until  REAL,
    totp_secret   TEXT,
    -- A secret generated but not yet proven. Kept apart from `totp_secret` so
    -- that login, which reads only the active column, cannot start demanding
    -- codes from a secret the user has not confirmed they can generate.
    pending_totp_secret TEXT,
    -- Single-use codes for getting back in without the authenticator, stored
    -- hashed exactly like passwords, because that is what they are.
    recovery_codes TEXT,
    created_at    REAL NOT NULL,
    last_login_at REAL,
    CHECK (role IN ('owner','admin','analyst','viewer'))
);
CREATE INDEX IF NOT EXISTS idx_users_ws ON users (workspace_id);

-- An invitation to join an existing workspace.
--
-- The token is stored hashed for exactly the reason refresh tokens are: this
-- row grants membership of a workspace, so a dump of the table must not be
-- usable to join one. The plaintext is shown to the inviter once and never
-- again, which is also why `accepted_at` matters -- a link that still works
-- after it was used is a permanent back door into the workspace.
CREATE TABLE IF NOT EXISTS invitations (
    invite_id    TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    email        TEXT NOT NULL,
    role         TEXT NOT NULL DEFAULT 'analyst',
    token_hash   BLOB NOT NULL UNIQUE,
    invited_by   TEXT NOT NULL,
    created_at   REAL NOT NULL,
    expires_at   REAL NOT NULL,
    accepted_at  REAL,
    -- An owner cannot be invited. Ownership transfers deliberately, not by
    -- someone emailing a link.
    CHECK (role IN ('admin','analyst','viewer'))
);
CREATE INDEX IF NOT EXISTS idx_invitations_ws ON invitations (workspace_id, email);

-- Refresh tokens are stored hashed. A dump of this table must not be usable
-- to mint sessions, which is the whole point of not storing the plaintext.
CREATE TABLE IF NOT EXISTS refresh_tokens (
    token_hash  BLOB PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    family_id   TEXT NOT NULL,
    parent_hash BLOB,
    expires_at  REAL NOT NULL,
    used_at     REAL,
    revoked     INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_refresh_family ON refresh_tokens (family_id);

-- Append-only. Nothing in the application updates or deletes a row here.
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id     TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    actor        TEXT,
    action       TEXT NOT NULL,
    target       TEXT,
    detail       TEXT NOT NULL DEFAULT '{}',
    ip           TEXT,
    ts           REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_ws ON audit_log (workspace_id, ts DESC);
"""


class StoreError(RuntimeError):
    pass


def _require_workspace(workspace_id: str) -> str:
    """Refuse to build a query without a tenant scope.

    An empty or missing `workspace_id` would silently widen a query to every
    tenant, which is the one bug class that is unrecoverable. Making it raise
    means the mistake shows up in a test rather than in a leak.
    """
    if not workspace_id or not workspace_id.strip():
        raise StoreError("workspace_id is required on every store operation")
    return workspace_id


def _json(value: Any) -> str:
    return json.dumps(value, default=str)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    page: int
    text: str
    char_start: int
    char_end: int
    para_idx: int | None
    section: str | None
    kind: str
    score: float = 0.0
    source: str = ""

    def citation_label(self) -> str:
        label = f"p.{self.page}"
        if self.para_idx is not None:
            label += f" ¶{self.para_idx}"
        return label

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "page": self.page,
            "para_idx": self.para_idx,
            "section": self.section,
            "kind": self.kind,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "text": self.text,
            "score": self.score,
            "source": self.source,
        }


class LocalStore:
    """SQLite-backed corpus, run and identity store."""

    backend = "sqlite"

    def __init__(self, path: str | Path = "./storage/resx.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()
        # One connection is shared across threads, and the graph runs the
        # specialists in parallel. See `tx`.
        self._lock = threading.RLock()
        self._closed = False

    def _migrate(self) -> None:
        """Bring an existing database up to the current schema.

        `CREATE TABLE IF NOT EXISTS` does nothing to a table that already
        exists, so a schema change is invisible to any database created before
        it. That is fine for a new column and fatal for a changed CHECK
        constraint, because SQLite has no `ALTER TABLE ... DROP CONSTRAINT`:
        the old rule stays enforced forever.

        Which is exactly what happened. `claims` carried
        `CHECK (value IS NULL OR computation_id IS NOT NULL)`; the application
        was later relaxed to allow a figure quoted from a web page with no
        computation, and every research-mode run then died on the INSERT with
        the whole graph half-finished. Editing the SCHEMA string above fixed
        new databases and left every existing one broken.

        So the table is rebuilt when the old constraint is found. SQLite's
        documented procedure: create the new shape, copy, drop, rename.
        """
        # `documents.progress_at`: a plain nullable column, so ALTER TABLE is
        # enough and no rebuild is needed.
        doc_cols = {
            r["name"] for r in self._conn.execute("PRAGMA table_info(documents)").fetchall()
        }
        user_cols = {
            r["name"] for r in self._conn.execute("PRAGMA table_info(users)").fetchall()
        }
        for column, ddl in (
            ("recovery_codes", "recovery_codes TEXT"),
            ("pending_totp_secret", "pending_totp_secret TEXT"),
        ):
            if user_cols and column not in user_cols:
                with self.tx() as conn:
                    conn.execute(f"ALTER TABLE users ADD COLUMN {ddl}")

        for column, ddl in (
            ("progress_at", "progress_at REAL"),
            ("embedder", "embedder TEXT"),
            ("embedding_dimensions", "embedding_dimensions INTEGER"),
        ):
            if doc_cols and column not in doc_cols:
                with self.tx() as conn:
                    conn.execute(f"ALTER TABLE documents ADD COLUMN {ddl}")

        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'claims'"
        ).fetchone()
        if row is None or "source_kind" in (row["sql"] or ""):
            return

        columns = [
            "claim_id",
            "workspace_id",
            "run_id",
            "agent",
            "statement",
            "value",
            "unit",
            "period",
            "currency",
            "scale_factor",
            "computation_id",
            "confidence",
            "confidence_reason",
            "payload",
            "created_at",
        ]
        joined = ", ".join(columns)
        # The current `claims` definition, taken from SCHEMA rather than
        # written out again here: two copies of a table definition drift, and
        # the drift would be a migration that produces the wrong shape.
        claims_ddl = SCHEMA[
            SCHEMA.index("CREATE TABLE IF NOT EXISTS claims (") : SCHEMA.index(
                "CREATE INDEX IF NOT EXISTS idx_claims_run"
            )
        ]
        # Foreign keys off for the swap: `citations` references `claims`, and
        # dropping the old table with them on would cascade the citations away.
        self._conn.executescript("PRAGMA foreign_keys = OFF;")
        # Every interpolated fragment is a literal from this module -- the
        # column list above and SCHEMA itself. No caller input reaches it, and
        # a table definition cannot be parameterised.
        migration = f"""
            BEGIN;
            CREATE TABLE claims_migrated AS SELECT {joined} FROM claims;
            DROP TABLE claims;
            {claims_ddl}
            INSERT INTO claims ({joined}, source_kind)
                SELECT {joined},
                       -- Anything already stored satisfied the old, stricter
                       -- rule, so 'corpus' is correct for all of it.
                       'corpus'
                FROM claims_migrated;
            DROP TABLE claims_migrated;
            CREATE INDEX IF NOT EXISTS idx_claims_run ON claims (workspace_id, run_id);
            COMMIT;
            """  # noqa: S608
        self._conn.executescript(migration)
        self._conn.executescript("PRAGMA foreign_keys = ON;")

    # -- plumbing ---------------------------------------------------------

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """A write transaction, serialised across threads.

        The lock is not defensive tidiness; without it the graph crashes. This
        store holds a single connection opened `check_same_thread=False`, and
        LangGraph runs the specialist nodes in parallel threads. Any
        read-modify-write spanning two statements can therefore interleave, and
        `append_event` did exactly that — `SELECT MAX(seq)` then `INSERT` — so
        two agents emitting a tool call at the same moment both computed the
        same sequence number and the second lost:

            sqlite3.IntegrityError: UNIQUE constraint failed:
                run_events.run_id, run_events.seq

        That exception propagates out of the emit callback, through the node,
        and kills the entire run. It is also a race, so it appeared
        intermittently and looked like a different bug each time.

        The Mongo store never had this: it allocates sequences with
        `find_one_and_update`, which is atomic server-side. The SQLite
        fallback simply never got the equivalent.
        """
        with self._lock:
            self._assert_open()
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def close(self) -> None:
        """Close the connection, and refuse every use after it.

        The flag is not tidiness. A run executes on a worker thread and writes
        events as it goes, while the API's shutdown closes the store — so a
        run in flight when the process stops will reach a closed connection.
        `sqlite3` does not answer that with an exception: on Windows it is an
        access violation that kills the interpreter, taking the rest of the
        shutdown with it. Observed as

            Windows fatal exception: access violation
              File "app/store/local.py", ... in append_event
              File "app/services/runs.py", ... in _execute_guarded

        A `StoreError` is caught by the worker's own guard, logged, and the run
        is marked failed. A segfault is not catchable by anything.
        """
        with self._lock:
            self._closed = True
            self._conn.close()

    def _assert_open(self) -> None:
        if self._closed:
            raise StoreError(
                "the store has been closed; this usually means a run was still "
                "in flight when the process began shutting down"
            )

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
        with self.tx() as conn:
            conn.execute(
                """
                INSERT INTO documents (doc_id, workspace_id, source_name, kind, sha256,
                                       page_count, status, low_conf_pages, warnings, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    page_count = excluded.page_count,
                    status = excluded.status,
                    low_conf_pages = excluded.low_conf_pages,
                    warnings = excluded.warnings
                """,
                (
                    doc_id,
                    workspace_id,
                    source_name,
                    kind,
                    sha256,
                    page_count,
                    status,
                    _json(list(low_conf_pages)),
                    _json(list(warnings)),
                    time.time(),
                ),
            )

    def set_document_status(
        self, *, workspace_id: str, doc_id: str, status: str, failure_reason: str | None = None
    ) -> None:
        _require_workspace(workspace_id)
        with self.tx() as conn:
            conn.execute(
                "UPDATE documents SET status = ?, failure_reason = ? "
                "WHERE doc_id = ? AND workspace_id = ?",
                (status, failure_reason, doc_id, workspace_id),
            )

    def touch_document(self, *, workspace_id: str, doc_id: str) -> None:
        """Record that this ingest is still alive.

        A stale-ingest cutoff measured from `created_at` is a guess about how
        long a document *ought* to take, and a 230-page PDF with OCR disproved
        the first guess by taking 23 minutes. Progress is the honest signal: an
        ingest still embedding batches is not abandoned however long it has
        run, and one whose process was killed stops calling this immediately.
        """
        _require_workspace(workspace_id)
        with self.tx() as conn:
            conn.execute(
                "UPDATE documents SET progress_at = ? WHERE doc_id = ? AND workspace_id = ?",
                (time.time(), doc_id, workspace_id),
            )

    def count_documents(self, *, workspace_id: str) -> int:
        _require_workspace(workspace_id)
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM documents WHERE workspace_id = ?", (workspace_id,)
        ).fetchone()
        return int(row["n"]) if row else 0

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
        sql = "SELECT * FROM documents WHERE workspace_id = ? ORDER BY created_at DESC"
        params: list[Any] = [workspace_id]
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([min(limit, 200), max(0, offset)])
        rows = self._conn.execute(sql, params).fetchall()
        return [self._document_row(r) for r in rows]

    def get_document(self, *, workspace_id: str, doc_id: str) -> dict[str, Any] | None:
        _require_workspace(workspace_id)
        row = self._conn.execute(
            "SELECT * FROM documents WHERE doc_id = ? AND workspace_id = ?",
            (doc_id, workspace_id),
        ).fetchone()
        return self._document_row(row) if row else None

    @staticmethod
    def _document_row(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["low_conf_pages"] = json.loads(data.get("low_conf_pages") or "[]")
        data["warnings"] = json.loads(data.get("warnings") or "[]")
        return data

    # -- pages ------------------------------------------------------------

    def add_pages(
        self, *, workspace_id: str, doc_id: str, pages: Sequence[tuple[int, str, str | None]]
    ) -> None:
        _require_workspace(workspace_id)
        with self.tx() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO pages (doc_id, workspace_id, page, text, section) "
                "VALUES (?, ?, ?, ?, ?)",
                [(doc_id, workspace_id, p, text, section) for p, text, section in pages],
            )

    def get_page_text(self, *, workspace_id: str, doc_id: str, page: int) -> str | None:
        _require_workspace(workspace_id)
        row = self._conn.execute(
            "SELECT text FROM pages WHERE doc_id = ? AND page = ? AND workspace_id = ?",
            (doc_id, page, workspace_id),
        ).fetchone()
        return row["text"] if row else None

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

        rows = []
        for i, chunk in enumerate(chunks):
            blob = (
                np.asarray(embeddings[i], dtype="float32").tobytes()
                if embeddings is not None
                else None
            )
            rows.append(
                (
                    chunk.chunk_id,
                    workspace_id,
                    chunk.doc_id,
                    chunk.page,
                    chunk.para_idx,
                    chunk.section,
                    chunk.char_start,
                    chunk.char_end,
                    _json(list(chunk.bbox)) if chunk.bbox else None,
                    chunk.text,
                    chunk.token_count,
                    chunk.kind,
                    chunk.table_name,
                    blob,
                )
            )

        with self.tx() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunks
                    (chunk_id, workspace_id, doc_id, page, para_idx, section,
                     char_start, char_end, bbox, text, token_count, kind,
                     table_name, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def get_chunk(self, *, workspace_id: str, chunk_id: str) -> RetrievedChunk | None:
        _require_workspace(workspace_id)
        row = self._conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ? AND workspace_id = ?",
            (chunk_id, workspace_id),
        ).fetchone()
        return self._to_retrieved(row) if row else None

    def iter_chunks(
        self, *, workspace_id: str, doc_ids: Sequence[str] | None = None
    ) -> list[RetrievedChunk]:
        _require_workspace(workspace_id)
        sql = "SELECT * FROM chunks WHERE workspace_id = ?"
        params: list[Any] = [workspace_id]
        if doc_ids:
            # Placeholders are generated from the *count*, never from the
            # values, so nothing user-supplied is interpolated into SQL.
            sql += f" AND doc_id IN ({','.join('?' for _ in doc_ids)})"
            params.extend(doc_ids)
        sql += " ORDER BY doc_id, page, char_start"
        rows = self._conn.execute(sql, params).fetchall()
        return [self._to_retrieved(r) for r in rows]

    def load_vectors(
        self,
        *,
        workspace_id: str,
        doc_ids: Sequence[str] | None = None,
        dimensions: int | None = None,
    ) -> tuple[list[str], FloatArray]:
        """Load the workspace's embedding matrix for similarity search.

        Tenant filtering happens in SQL, so isolation holds *inside* the vector
        search rather than being applied to its results afterwards.

        `dimensions` drops vectors of any other width. A workspace really can
        hold two widths at once: changing `EMBEDDING_PROVIDER` or
        `EMBEDDING_DIMENSIONS` and re-ingesting only some documents leaves the
        rest on the old model. Not filtering fails in one of two ways and the
        second is worse — `np.vstack` raises on mismatched widths and takes
        retrieval down, or the widths coincidentally match and the scores are
        computed across two unrelated embedding spaces while looking entirely
        normal. A vector only means anything inside its own model's space.
        """
        _require_workspace(workspace_id)
        sql = (
            "SELECT chunk_id, embedding FROM chunks "
            "WHERE workspace_id = ? AND embedding IS NOT NULL"
        )
        params: list[Any] = [workspace_id]
        if doc_ids:
            sql += f" AND doc_id IN ({','.join('?' for _ in doc_ids)})"
            params.extend(doc_ids)

        rows = self._conn.execute(sql, params).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype="float32")

        ids: list[str] = []
        buffers: list[FloatArray] = []
        skipped = 0
        for row in rows:
            vector = np.frombuffer(row["embedding"], dtype="float32")
            if dimensions is not None and vector.shape[0] != dimensions:
                skipped += 1
                continue
            ids.append(row["chunk_id"])
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
            # mixing by a wide margin.
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
        the current model" except by measuring a stored vector, and no way to
        tell a reader which documents went stale when they changed provider.
        """
        _require_workspace(workspace_id)
        with self.tx() as conn:
            conn.execute(
                "UPDATE documents SET embedder = ?, embedding_dimensions = ? "
                "WHERE doc_id = ? AND workspace_id = ?",
                (embedder, int(dimensions), doc_id, workspace_id),
            )

    @staticmethod
    def _to_retrieved(row: sqlite3.Row, score: float = 0.0, source: str = "") -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            page=row["page"],
            text=row["text"],
            char_start=row["char_start"],
            char_end=row["char_end"],
            para_idx=row["para_idx"],
            section=row["section"],
            kind=row["kind"],
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
        rows = self._conn.execute(
            # Only the "?" placeholders are interpolated; every id is
            # bound as a parameter below. An IN clause cannot be
            # parameterised any other way.
            f"SELECT * FROM chunks WHERE workspace_id = ? "  # noqa: S608
            f"AND chunk_id IN ({','.join('?' for _ in ids)})",
            [workspace_id, *ids],
        ).fetchall()
        by_id = {r["chunk_id"]: r for r in rows}
        out: list[RetrievedChunk] = []
        for chunk_id, score, source in scored:
            row = by_id.get(chunk_id)
            if row is not None:
                out.append(self._to_retrieved(row, score=score, source=source))
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
        with self.tx() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO datasets
                    (dataset_id, workspace_id, doc_id, name, source_page, path,
                     n_rows, n_cols, profile, scale_factor, currency, agreement, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved,
                    workspace_id,
                    doc_id,
                    name,
                    source_page,
                    path,
                    n_rows,
                    n_cols,
                    _json(profile or {}),
                    scale_factor,
                    currency,
                    None if agreement is None else int(agreement),
                    time.time(),
                ),
            )
        return resolved

    def list_datasets(
        self, *, workspace_id: str, doc_id: str | None = None
    ) -> list[dict[str, Any]]:
        _require_workspace(workspace_id)
        if doc_id:
            rows = self._conn.execute(
                "SELECT * FROM datasets WHERE workspace_id = ? AND doc_id = ?",
                (workspace_id, doc_id),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM datasets WHERE workspace_id = ?", (workspace_id,)
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["profile"] = json.loads(d.get("profile") or "{}")
            d["agreement"] = None if d["agreement"] is None else bool(d["agreement"])
            out.append(d)
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
        with self.tx() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, workspace_id, question, corpus_ids, "
                "status, usd_cap, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    resolved,
                    workspace_id,
                    question,
                    _json(list(corpus_ids)),
                    "queued",
                    usd_cap,
                    time.time(),
                ),
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

        # Column names come from the allowlist above, never from the caller's
        # keys directly — identifiers cannot be parameterised, so they must be
        # validated rather than interpolated.
        assignments = ", ".join(f"{k} = ?" for k in fields)
        values = [
            _json(v) if k in {"plan", "report", "degraded"} and not isinstance(v, str) else v
            for k, v in fields.items()
        ]
        with self.tx() as conn:
            conn.execute(
                # `assignments` holds only column names from the allowlist
                # validated above; identifiers cannot be parameterised.
                f"UPDATE runs SET {assignments} "  # noqa: S608
                "WHERE run_id = ? AND workspace_id = ?",
                [*values, run_id, workspace_id],
            )

    def get_run(self, *, workspace_id: str, run_id: str) -> dict[str, Any] | None:
        _require_workspace(workspace_id)
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ? AND workspace_id = ?",
            (run_id, workspace_id),
        ).fetchone()
        if not row:
            return None
        data = dict(row)
        for key in ("corpus_ids", "degraded"):
            data[key] = json.loads(data.get(key) or "[]")
        for key in ("plan", "report"):
            data[key] = json.loads(data[key]) if data.get(key) else None
        return data

    def count_runs(self, *, workspace_id: str) -> int:
        """How many runs exist, so a caller can page without guessing."""
        _require_workspace(workspace_id)
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM runs WHERE workspace_id = ?", (workspace_id,)
        ).fetchone()
        return int(row["n"]) if row else 0

    def list_runs(
        self, *, workspace_id: str, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        _require_workspace(workspace_id)
        rows = self._conn.execute(
            "SELECT run_id, question, status, tokens_used, usd_used, created_at, "
            "finished_at FROM runs WHERE workspace_id = ? ORDER BY created_at DESC "
            "LIMIT ? OFFSET ?",
            (workspace_id, min(limit, 100), max(0, offset)),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- evidence ---------------------------------------------------------

    def add_computation(
        self, *, workspace_id: str, run_id: str | None, agent: str, record: Any
    ) -> str:
        _require_workspace(workspace_id)
        with self.tx() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO computations
                    (computation_id, workspace_id, run_id, agent, code, inputs,
                     stdout, stderr, result, duration_ms, image_digest, ok, error, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.computation_id,
                    workspace_id,
                    run_id,
                    agent,
                    record.code,
                    _json(record.inputs),
                    record.stdout,
                    record.stderr,
                    _json(record.result),
                    record.duration_ms,
                    record.image_digest,
                    int(record.ok),
                    record.error,
                    time.time(),
                ),
            )
        computation_id: str = record.computation_id
        return computation_id

    def get_computation(
        self, *, workspace_id: str, computation_id: str
    ) -> dict[str, Any] | None:
        _require_workspace(workspace_id)
        row = self._conn.execute(
            "SELECT * FROM computations WHERE computation_id = ? AND workspace_id = ?",
            (computation_id, workspace_id),
        ).fetchone()
        if not row:
            return None
        data = dict(row)
        data["inputs"] = json.loads(data.get("inputs") or "[]")
        data["result"] = json.loads(data["result"]) if data.get("result") else None
        data["ok"] = bool(data["ok"])
        return data

    def add_claim(self, *, workspace_id: str, run_id: str, claim: Any) -> str:
        """Persist a claim and its citations.

        The CHECK constraint on the table is what actually enforces "the LLM
        never does arithmetic" at the storage layer — a numeric claim with no
        computation and no external source raises here.
        """
        _require_workspace(workspace_id)
        with self.tx() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO claims
                    (claim_id, workspace_id, run_id, agent, statement, value, unit,
                     period, currency, scale_factor, computation_id, confidence,
                     confidence_reason, payload, source_kind, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    claim.claim_id,
                    workspace_id,
                    run_id,
                    claim.agent,
                    claim.statement,
                    None if claim.value is None else str(claim.value),
                    claim.unit,
                    claim.period,
                    claim.currency,
                    str(getattr(claim, "scale_factor", "1")),
                    claim.computation_id,
                    claim.confidence,
                    claim.confidence_reason,
                    _json(getattr(claim, "payload", {})),
                    # Read from the claim rather than inferred here, so the
                    # column and the Pydantic rule cannot disagree.
                    "external" if claim.is_externally_sourced else "corpus",
                    time.time(),
                ),
            )
            for citation in claim.citations:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO citations
                        (citation_id, workspace_id, claim_id, chunk_id, doc_id, page,
                         para_idx, char_start, char_end, quote, url, publisher,
                         retrieved_at, resolution, match_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        citation.citation_id,
                        workspace_id,
                        claim.claim_id,
                        citation.chunk_id,
                        citation.doc_id,
                        citation.page,
                        citation.para_idx,
                        citation.char_start,
                        citation.char_end,
                        citation.quote,
                        citation.url,
                        citation.publisher,
                        time.time(),
                        citation.resolution,
                        citation.match_score,
                    ),
                )
        claim_id: str = claim.claim_id
        return claim_id

    def list_claims(self, *, workspace_id: str, run_id: str) -> list[dict[str, Any]]:
        _require_workspace(workspace_id)
        rows = self._conn.execute(
            "SELECT * FROM claims WHERE workspace_id = ? AND run_id = ? ORDER BY created_at",
            (workspace_id, run_id),
        ).fetchall()
        claims = []
        for row in rows:
            data = dict(row)
            data["payload"] = json.loads(data.get("payload") or "{}")
            cites = self._conn.execute(
                "SELECT * FROM citations WHERE claim_id = ? AND workspace_id = ?",
                (data["claim_id"], workspace_id),
            ).fetchall()
            data["citations"] = [dict(c) for c in cites]
            verdicts = self._conn.execute(
                "SELECT * FROM verdicts WHERE claim_id = ? AND workspace_id = ? "
                "ORDER BY debate_round DESC",
                (data["claim_id"], workspace_id),
            ).fetchall()
            data["verdicts"] = [dict(v) for v in verdicts]
            claims.append(data)
        return claims

    def add_verdict(self, *, workspace_id: str, verdict: Any) -> str:
        _require_workspace(workspace_id)
        verdict_id = f"vd_{uuid.uuid4().hex[:10]}"
        with self.tx() as conn:
            conn.execute(
                "INSERT INTO verdicts (verdict_id, workspace_id, claim_id, verdict, "
                "independent_value, rationale, debate_round, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    verdict_id,
                    workspace_id,
                    verdict.claim_id,
                    verdict.verdict,
                    None
                    if verdict.independent_value is None
                    else str(verdict.independent_value),
                    verdict.rationale,
                    verdict.debate_round,
                    time.time(),
                ),
            )
        return verdict_id

    # -- events -----------------------------------------------------------

    def append_event(
        self, *, workspace_id: str, run_id: str, kind: str, payload: dict[str, Any]
    ) -> int:
        _require_workspace(workspace_id)
        with self.tx() as conn:
            # One statement, so the sequence cannot be allocated twice even if
            # the lock in `tx` is ever removed or a second process attaches.
            # The subquery is evaluated inside the same INSERT.
            cursor = conn.execute(
                "INSERT INTO run_events (run_id, workspace_id, seq, kind, payload, ts) "
                "SELECT ?, ?, COALESCE(MAX(seq), 0) + 1, ?, ?, ? "
                "FROM run_events WHERE run_id = ?",
                (run_id, workspace_id, kind, _json(payload), time.time(), run_id),
            )
            row = conn.execute(
                "SELECT seq FROM run_events WHERE rowid = ?",
                (cursor.lastrowid,),
            ).fetchone()
        return int(row["seq"]) if row else 0

    def read_events(
        self, *, workspace_id: str, run_id: str, after_seq: int = 0
    ) -> list[dict[str, Any]]:
        _require_workspace(workspace_id)
        rows = self._conn.execute(
            "SELECT seq, kind, payload, ts FROM run_events WHERE run_id = ? "
            "AND workspace_id = ? AND seq > ? ORDER BY seq",
            (run_id, workspace_id, after_seq),
        ).fetchall()
        return [
            {
                "seq": r["seq"],
                "kind": r["kind"],
                "payload": json.loads(r["payload"]),
                "ts": r["ts"],
            }
            for r in rows
        ]

    # -- metrics ----------------------------------------------------------

    def citation_validity(self, *, workspace_id: str) -> dict[str, Any]:
        """Grounding health for the workspace.

        `docs/04` requires citation validity of exactly 1.00, so it is a
        queryable number rather than an assumption.
        """
        _require_workspace(workspace_id)
        row = self._conn.execute(
            "SELECT COUNT(*) AS total, "
            "SUM(CASE WHEN resolution IN "
            "('ok','derived_from_computation','external') "
            "THEN 1 ELSE 0 END) AS resolved "
            "FROM citations WHERE workspace_id = ?",
            (workspace_id,),
        ).fetchone()
        total = int(row["total"] or 0)
        resolved = int(row["resolved"] or 0)
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
        docs = self._conn.execute(
            "SELECT page_count, status FROM documents WHERE workspace_id = ?",
            (workspace_id,),
        ).fetchall()

        def count(table: str) -> int:
            row = self._conn.execute(
                # `table` comes from the literal tuple below, never a caller.
                f"SELECT COUNT(*) AS n FROM {table} WHERE workspace_id = ?",  # noqa: S608
                (workspace_id,),
            ).fetchone()
            return int(row["n"] or 0)

        grounding = self.citation_validity(workspace_id=workspace_id)
        return {
            "documents": len(docs),
            "ready": sum(1 for d in docs if str(d["status"] or "").startswith("ready")),
            "pages": sum(int(d["page_count"] or 0) for d in docs),
            "chunks": count("chunks"),
            "datasets": count("datasets"),
            "runs": count("runs"),
            "claims": count("claims"),
            "computations": count("computations"),
            "citation_validity": grounding["validity"],
            "total_citations": grounding["total_citations"],
        }

    def delete_document(self, *, workspace_id: str, doc_id: str) -> bool:
        """Remove a document and everything derived from it."""
        _require_workspace(workspace_id)
        with self.tx() as conn:
            cursor = conn.execute(
                "DELETE FROM documents WHERE doc_id = ? AND workspace_id = ?",
                (doc_id, workspace_id),
            )
            deleted = cursor.rowcount > 0
            # Explicit, even though the foreign keys cascade: the pragma is set
            # per-connection, and a cascade that depends on a pragma being on is
            # a cascade that silently stops working.
            for table in ("pages", "chunks", "datasets"):
                conn.execute(
                    f"DELETE FROM {table} WHERE doc_id = ? AND workspace_id = ?",  # noqa: S608
                    (doc_id, workspace_id),
                )
        return deleted

    def wipe_workspace(self, *, workspace_id: str) -> dict[str, int]:
        """Delete every corpus artefact for a workspace, keeping its users.

        Used by `resx.py reset`. Identity is deliberately untouched: wiping the
        corpus should not log the operator out of their own workspace.
        """
        _require_workspace(workspace_id)
        counts: dict[str, int] = {}
        with self.tx() as conn:
            for table in (
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
                cursor = conn.execute(
                    f"DELETE FROM {table} WHERE workspace_id = ?",  # noqa: S608
                    (workspace_id,),
                )
                counts[table] = cursor.rowcount
        return counts

    # -- identity ---------------------------------------------------------

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
        user_id = f"usr_{uuid.uuid4().hex[:12]}"
        normalised = email.strip().lower()
        try:
            with self.tx() as conn:
                conn.execute(
                    "INSERT INTO users (user_id, email, password_hash, name, "
                    "workspace_id, role, verified, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        user_id,
                        normalised,
                        password_hash,
                        name.strip(),
                        workspace_id,
                        role,
                        int(verified),
                        time.time(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise StoreError("email already registered") from exc
        created = self.get_user(user_id)
        if created is None:
            # `assert` would be stripped under `python -O`, turning this into a
            # `None` dereference in exactly the deployment that optimises.
            raise StoreError(f"user {user_id} vanished immediately after insert")
        return {k: v for k, v in created.items() if k != "password_hash"}

    def get_user_by_email(self, email: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM users WHERE email = ?", (email.strip().lower(),)
        ).fetchone()
        return self._user_row(row) if row else None

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return self._user_row(row) if row else None

    @staticmethod
    def _user_row(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["verified"] = bool(data.get("verified"))
        return data

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
        if not fields:
            return
        values = [int(v) if isinstance(v, bool) else v for v in fields.values()]
        # Column names come from the `allowed` set checked above, never from
        # the caller's keys directly: SQL identifiers cannot be parameterised,
        # so they have to be validated against an allowlist instead. Every
        # *value* is still bound.
        assignments = ", ".join(f"{k} = ?" for k in fields)
        with self.tx() as conn:
            conn.execute(
                f"UPDATE users SET {assignments} WHERE user_id = ?",  # noqa: S608
                [*values, user_id],
            )

    def count_users(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()
        return int(row["n"] or 0)

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
        with self.tx() as conn:
            conn.execute(
                "INSERT INTO invitations (invite_id, workspace_id, email, role, "
                "token_hash, invited_by, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    invite_id,
                    workspace_id,
                    email.strip().lower(),
                    role,
                    token_hash,
                    invited_by,
                    time.time(),
                    expires_at,
                ),
            )

    def get_invitation(self, token_hash: bytes) -> dict[str, Any] | None:
        """Looked up by the hash of the token, never by id.

        The id is in the URL a reader can see; the token is the secret. Looking
        up by id and then comparing would mean a timing-observable compare on
        a row an attacker chose.
        """
        row = self._conn.execute(
            "SELECT * FROM invitations WHERE token_hash = ?", (token_hash,)
        ).fetchone()
        return dict(row) if row else None

    def list_invitations(self, *, workspace_id: str) -> list[dict[str, Any]]:
        _require_workspace(workspace_id)
        rows = self._conn.execute(
            "SELECT invite_id, workspace_id, email, role, invited_by, created_at, "
            "expires_at, accepted_at FROM invitations WHERE workspace_id = ? "
            "ORDER BY created_at DESC",
            (workspace_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_invitation_accepted(self, *, invite_id: str) -> None:
        with self.tx() as conn:
            conn.execute(
                "UPDATE invitations SET accepted_at = ? WHERE invite_id = ?",
                (time.time(), invite_id),
            )

    def revoke_invitation(self, *, workspace_id: str, invite_id: str) -> bool:
        """Delete it outright rather than flagging it.

        A revoked-but-present row is one forgotten `WHERE` clause away from
        still working, and this row grants membership of a workspace.
        """
        _require_workspace(workspace_id)
        with self.tx() as conn:
            cursor = conn.execute(
                "DELETE FROM invitations WHERE invite_id = ? AND workspace_id = ? "
                "AND accepted_at IS NULL",
                (invite_id, workspace_id),
            )
            return bool(cursor.rowcount)

    def store_refresh_token(
        self,
        *,
        token_hash: bytes,
        user_id: str,
        family_id: str,
        expires_at: float,
        parent_hash: bytes | None = None,
    ) -> None:
        with self.tx() as conn:
            conn.execute(
                "INSERT INTO refresh_tokens (token_hash, user_id, family_id, "
                "parent_hash, expires_at, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (token_hash, user_id, family_id, parent_hash, expires_at, time.time()),
            )

    def get_refresh_token(self, token_hash: bytes) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM refresh_tokens WHERE token_hash = ?", (token_hash,)
        ).fetchone()
        if not row:
            return None
        data = dict(row)
        data["revoked"] = bool(data.get("revoked"))
        return data

    def mark_refresh_token_used(self, token_hash: bytes) -> None:
        with self.tx() as conn:
            conn.execute(
                "UPDATE refresh_tokens SET used_at = ? WHERE token_hash = ?",
                (time.time(), token_hash),
            )

    def revoke_token_family(self, family_id: str) -> int:
        """Revoke every token in a family.

        Called on reuse detection. A replayed refresh token means the token was
        captured, and the only safe response is to invalidate the whole chain
        rather than the single token that was replayed.
        """
        with self.tx() as conn:
            cursor = conn.execute(
                "UPDATE refresh_tokens SET revoked = 1 WHERE family_id = ?",
                (family_id,),
            )
        return cursor.rowcount

    def purge_expired_refresh_tokens(self) -> int:
        with self.tx() as conn:
            cursor = conn.execute(
                "DELETE FROM refresh_tokens WHERE expires_at < ?", (time.time(),)
            )
        return cursor.rowcount

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
        with self.tx() as conn:
            conn.execute(
                "INSERT INTO audit_log (audit_id, workspace_id, actor, action, "
                "target, detail, ip, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"aud_{uuid.uuid4().hex[:12]}",
                    workspace_id,
                    actor,
                    action,
                    target,
                    _json(detail or {}),
                    ip,
                    time.time(),
                ),
            )

    def read_audit(self, *, workspace_id: str, limit: int = 100) -> list[dict[str, Any]]:
        _require_workspace(workspace_id)
        rows = self._conn.execute(
            "SELECT * FROM audit_log WHERE workspace_id = ? ORDER BY ts DESC LIMIT ?",
            (workspace_id, min(limit, 500)),
        ).fetchall()
        out = []
        for row in rows:
            data = dict(row)
            data["detail"] = json.loads(data.get("detail") or "{}")
            out.append(data)
        return out
