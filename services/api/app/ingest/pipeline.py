"""The ingestion pipeline.

    extract -> chunk -> verify anchors -> embed -> persist
            -> tables to typed datasets (the numeric path)

Two things are worth calling out.

**Anchors are verified at ingestion, not at click time.** If a chunk's recorded
span does not contain its own opening text, that is a bug in extraction, and
the right moment to find it is while the document is being processed — not when
a user clicks a citation in a finished report.

**Tables become datasets.** A financial table is written out as CSV and
registered so the sandbox can load it with `resx.load(dataset_id)` and sum a
`Decimal` column exactly. This is the difference between an agent reading
"48,920" out of a sentence and an agent summing a column.
"""

from __future__ import annotations

import contextlib
import csv
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.core.money import parse_amount
from app.ingest.chunk import ChunkingStats, chunk_document, verify_anchors
from app.ingest.extract import ExtractedTable, ExtractionResult, extract
from app.rag.embeddings import Embedder, EmbeddingError

if TYPE_CHECKING:
    # A `TYPE_CHECKING`-only import: `app.store.factory` pulls in the Mongo
    # backend, and a SQLite-only install must not need pymongo at import time.
    # `from __future__ import annotations` makes every annotation a string, so
    # this costs nothing at runtime and states the truth — these functions
    # accept either backend, and annotating `LocalStore` was the inaccuracy.
    from app.store.factory import Store


@dataclass(slots=True)
class IngestReport:
    doc_id: str
    source_name: str
    kind: str
    pages: int
    chunks: int
    table_chunks: int
    datasets: list[str] = field(default_factory=list)
    anchor_completeness: float = 1.0
    anchor_problems: list[str] = field(default_factory=list)
    low_confidence_pages: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    embedded: bool = False
    embedder: str = ""
    duration_ms: int = 0
    status: str = "ready"

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source_name": self.source_name,
            "kind": self.kind,
            "pages": self.pages,
            "chunks": self.chunks,
            "table_chunks": self.table_chunks,
            "datasets": self.datasets,
            "anchor_completeness": self.anchor_completeness,
            "anchor_problems": self.anchor_problems,
            "low_confidence_pages": self.low_confidence_pages,
            "warnings": self.warnings,
            "embedded": self.embedded,
            "embedder": self.embedder,
            "duration_ms": self.duration_ms,
            "status": self.status,
        }

    def summary(self) -> str:
        bits = [
            f"{self.source_name}: {self.pages} page(s), {self.chunks} chunk(s)",
            f"{len(self.datasets)} dataset(s)",
            f"anchors {self.anchor_completeness:.0%}",
        ]
        if self.low_confidence_pages:
            bits.append(f"{len(self.low_confidence_pages)} page(s) need OCR")
        if self.warnings:
            bits.append(f"{len(self.warnings)} warning(s)")
        return " · ".join(bits)


# --------------------------------------------------------------------------- #
# Dataset profiling
# --------------------------------------------------------------------------- #

_DATE = re.compile(r"^\d{4}[-/]\d{1,2}([-/]\d{1,2})?$|^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$")
_YEAR_LABEL = re.compile(r"^(FY)?\s?(19|20)\d{2}$", re.I)


def _infer_role(header: str, values: Sequence[str]) -> str:
    """Classify a column so the LLM reads a profile, not 50,000 raw rows.

    A faithful summary is both cheaper and *more accurate* than a truncated
    sample, which invites the model to generalise from the first twenty rows.

    **Contents decide the role; the header only disambiguates.** Getting this
    backwards is a real hazard in financial statements, where the columns are
    headed by periods: a column headed "FY2025" holding 48,920 / 28,471 /
    20,449 is a column of *money* labelled by a period, not a column of
    periods. Classifying it off the header marked it non-numeric, which meant
    the thousands separators were never stripped and the scale factor was
    never applied — reintroducing the 1000x error the scale machinery exists
    to prevent.
    """
    lowered = header.lower()
    non_empty = [v for v in values if v.strip()]
    if not non_empty:
        return "empty"

    sample = non_empty[:50]
    numeric = sum(1 for value in sample if parse_amount(value) is not None)
    numeric_ratio = numeric / len(sample)

    # --- figure columns, decided by the cells ---
    if numeric_ratio > 0.8:
        # A date-shaped cell is a date even though it parses as a number.
        date_like = sum(1 for v in sample[:20] if _DATE.match(v.strip()))
        if date_like > len(sample[:20]) * 0.6:
            return "date"

        if "%" in lowered or any("%" in v for v in sample[:10]):
            return "percentage"

        money_words = (
            "revenue",
            "cost",
            "cash",
            "profit",
            "income",
            "expense",
            "amount",
            "total",
            "price",
            "sales",
            "margin",
            "asset",
            "liabilit",
            "equity",
            "debt",
            "spend",
            "cogs",
            "opex",
            "$",
        )
        if any(word in lowered for word in money_words):
            return "currency"

        # A period-labelled column of figures in a financial table is money.
        if _YEAR_LABEL.match(header.strip()):
            return "currency"

        # Currency symbols or accounting negatives in the cells themselves.
        if any(
            symbol in value for value in sample[:10] for symbol in ("$", "£", "€", "¥", "₹")
        ) or any(value.strip().startswith("(") for value in sample[:10]):
            return "currency"

        return "quantity"

    # --- non-numeric columns ---
    if any(k in lowered for k in ("date", "month", "period", "quarter", "year")):
        return "date"
    if sum(1 for v in sample[:20] if _DATE.match(v.strip())) > len(sample[:20]) * 0.6:
        return "date"
    if _YEAR_LABEL.match(header.strip()):
        return "period"

    distinct = set(non_empty)
    unique_ratio = len(distinct) / len(non_empty)
    if unique_ratio > 0.95 and len(non_empty) > 5:
        return "id"
    if unique_ratio < 0.5:
        return "category"

    # Few distinct short values is a category even when every one is unique.
    #
    # The ratio test alone requires repetition, so a breakdown table -- one row
    # per region, per segment, per cost line -- scored 1.0 and was classified
    # "text": the most common shape of business table in existence, and the
    # profile said it held prose. Nothing could label a chart or group a
    # comparison from it.
    if len(distinct) <= 25 and all(len(v.strip()) <= 60 for v in list(distinct)[:25]):
        return "category"
    return "text"


def profile_table(table: ExtractedTable) -> dict[str, Any]:
    """A structured profile of one table. See docs/01 §1.2."""
    header = table.header or [f"col_{i}" for i in range(table.n_cols)]
    columns: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for idx, name in enumerate(header):
        values = [row[idx] if idx < len(row) else "" for row in table.rows]
        non_empty = [v for v in values if v.strip()]
        null_pct = 0.0 if not values else 1.0 - (len(non_empty) / len(values))
        role = _infer_role(name, values)

        columns.append(
            {
                "name": name or f"col_{idx}",
                "index": idx,
                "candidate_role": role,
                "null_pct": round(null_pct, 4),
                "unique_pct": round(
                    (len(set(non_empty)) / len(non_empty)) if non_empty else 0.0, 4
                ),
                "sample": non_empty[:3],
            }
        )

        if null_pct > 0.5:
            issues.append(
                {
                    "kind": "high_null",
                    "column": name,
                    "severity": "warning",
                    "count": len(values) - len(non_empty),
                }
            )

    seen: set[tuple[str, ...]] = set()
    duplicates = 0
    for row in table.rows:
        key = tuple(row)
        if key in seen:
            duplicates += 1
        seen.add(key)

    if duplicates:
        issues.append(
            {
                "kind": "duplicate_rows",
                "column": None,
                "severity": "warning",
                "count": duplicates,
            }
        )

    if table.agreement is False:
        issues.append(
            {
                "kind": "extractor_disagreement",
                "column": None,
                "severity": "critical",
                "count": 1,
                "note": (
                    "two extraction strategies disagreed on this table; figures "
                    "require confirmation before backing a financial claim"
                ),
            }
        )

    if table.scale_phrase is None and any(c["candidate_role"] == "currency" for c in columns):
        issues.append(
            {
                "kind": "undeclared_scale",
                "column": None,
                "severity": "warning",
                "count": 1,
                "note": (
                    "monetary columns with no scale declaration found nearby; "
                    "figures are read at face value and should be confirmed"
                ),
            }
        )

    return {
        "n_rows": table.n_rows,
        "n_cols": table.n_cols,
        "columns": columns,
        "duplicate_row_pct": round(duplicates / table.n_rows, 4) if table.n_rows else 0.0,
        "scale_factor": table.scale_factor,
        "scale_phrase": table.scale_phrase,
        "currency": table.currency,
        "extractor": table.extractor,
        "extractor_agreement": table.agreement,
        "issues": issues,
    }


# --------------------------------------------------------------------------- #
# The numeric path
# --------------------------------------------------------------------------- #


def _normalise_numeric_cell(value: str, scale: Decimal) -> str:
    """Write a machine-readable value where the cell holds a figure.

    The scale is applied **here**, once, at the boundary — so everything
    downstream (the sandbox, the identity checks) works in absolute units and
    nothing has to remember to multiply by a thousand.
    """
    parsed = parse_amount(value, scale=scale)
    if parsed is None:
        return value.strip()
    return str(parsed)


def write_dataset(
    table: ExtractedTable,
    *,
    dataset_dir: Path,
    dataset_id: str,
) -> tuple[Path, int, int]:
    """Persist a table as CSV the sandbox can load."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_dir / f"{dataset_id}.csv"

    try:
        scale = Decimal(table.scale_factor)
    except InvalidOperation:
        scale = Decimal(1)

    header = table.header or [f"col_{i}" for i in range(table.n_cols)]
    safe_header = [(h.strip() or f"col_{i}").replace(",", " ") for i, h in enumerate(header)]

    numeric_columns = {
        c["index"]
        for c in profile_table(table)["columns"]
        if c["candidate_role"] in {"currency", "quantity", "percentage"}
    }

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(safe_header)
        for row in table.rows:
            padded = list(row) + [""] * (len(safe_header) - len(row))
            writer.writerow(
                [
                    _normalise_numeric_cell(cell, scale)
                    if i in numeric_columns
                    else cell.strip()
                    for i, cell in enumerate(padded[: len(safe_header)])
                ]
            )

    return path, table.n_rows, len(safe_header)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


log = logging.getLogger("resx.ingest")


def ingest_file(
    path: str | Path,
    *,
    workspace_id: str,
    store: Store,
    embedder: Embedder | None = None,
    dataset_dir: str | Path = "./storage/datasets",
    doc_id: str | None = None,
) -> IngestReport:
    """Ingest one document end to end."""
    started = time.perf_counter()
    result: ExtractionResult = extract(path, doc_id=doc_id)

    store.upsert_document(
        workspace_id=workspace_id,
        doc_id=result.doc_id,
        source_name=result.source_name,
        kind=result.kind,
        sha256=result.sha256,
        page_count=result.page_count,
        status="extracting",
        low_conf_pages=result.low_confidence_pages,
        warnings=result.warnings,
    )

    # From here on, every failure path must leave the document in a *terminal*
    # state. A row stuck at "extracting" is unreadable and unactionable: the
    # work is dead and nothing says so.
    try:
        return _ingest_body(
            result=result,
            workspace_id=workspace_id,
            store=store,
            embedder=embedder,
            dataset_dir=dataset_dir,
            started=started,
        )
    except Exception as exc:
        store.set_document_status(
            workspace_id=workspace_id,
            doc_id=result.doc_id,
            status="failed",
            failure_reason=f"{type(exc).__name__}: {exc}"[:2000],
        )
        raise


def _ingest_body(
    *,
    result: ExtractionResult,
    workspace_id: str,
    store: Any,
    embedder: Embedder | None,
    dataset_dir: str | Path,
    started: float,
) -> IngestReport:
    """The rest of the ingest. Split out so the caller above owns the status."""

    def heartbeat() -> None:
        """Say the ingest is alive, so the upload guard can tell a slow run
        from a dead one. Best-effort: a failed heartbeat must never be the
        thing that kills an otherwise healthy ingest."""
        touch = getattr(store, "touch_document", None)
        if touch is None:
            return
        try:
            touch(workspace_id=workspace_id, doc_id=result.doc_id)
        except Exception:
            # Liveness reporting, not the work itself.
            log.debug("heartbeat failed for %s", result.doc_id, exc_info=True)

    heartbeat()
    store.add_pages(
        workspace_id=workspace_id,
        doc_id=result.doc_id,
        pages=[(p.page, p.text, p.section) for p in result.pages],
    )

    chunks, stats = chunk_document(result)
    problems = verify_anchors(chunks, result)
    warnings = list(result.warnings)
    heartbeat()

    embeddings = None
    embedder_name = ""
    if embedder is not None and chunks:
        # Embedding is the long pole — 595 chunks through a free tier took 23
        # minutes — so this is where liveness has to be reported. The embedders
        # already expose a per-batch progress hook; it is borrowed rather than
        # replaced, so a caller that set its own still gets called.
        previous = getattr(embedder, "progress", None)

        def on_batch(done: int, total: int) -> None:
            heartbeat()
            if previous is not None:
                previous(done, total)

        try:
            with contextlib.suppress(AttributeError):
                embedder.progress = on_batch  # type: ignore[attr-defined]
            embeddings = embedder.embed_documents([c.text for c in chunks])
            embedder_name = embedder.model_name
        except EmbeddingError as exc:
            # Degrade rather than fail. The text is extracted, the tables are
            # usable, and BM25 lexical retrieval still works — throwing all of
            # that away because an embedding provider was rate-limited would
            # be a far worse outcome than a document with weaker recall.
            #
            # The warning is not cosmetic: without it, retrieval quality would
            # be silently halved and every recall figure would be a lie.
            log.warning("embeddings unavailable for %s: %s", result.doc_id, exc)
            warnings.append(
                f"semantic embeddings unavailable, so retrieval for this "
                f"document is lexical only: {exc}"
            )

        finally:
            with contextlib.suppress(AttributeError):
                embedder.progress = previous  # type: ignore[attr-defined]

    store.add_chunks(workspace_id=workspace_id, chunks=chunks, embeddings=embeddings)
    heartbeat()

    if embeddings is not None and embedder is not None:
        # Recorded so a later run can tell whether this document's vectors are
        # searchable with the model it is holding. A vector only means anything
        # inside its own model's space, and without this the only way to find
        # out is to measure one.
        record = getattr(store, "set_document_embedder", None)
        if record is not None:
            record(
                workspace_id=workspace_id,
                doc_id=result.doc_id,
                embedder=embedder_name,
                dimensions=int(embeddings.shape[1]),
            )

    # --- the numeric path ---
    dataset_ids: list[str] = []
    ds_dir = Path(dataset_dir)
    for table in result.tables:
        # Skip trivia: a one-column "table" is almost always a mis-detected
        # text block, and registering it as a dataset is noise the agent would
        # have to reason past.
        if table.n_rows < 2 or table.n_cols < 2:
            continue

        dataset_id = f"ds_{result.doc_id.removeprefix('doc_')}_{table.name}"
        csv_path, n_rows, n_cols = write_dataset(
            table, dataset_dir=ds_dir, dataset_id=dataset_id
        )
        store.add_dataset(
            workspace_id=workspace_id,
            doc_id=result.doc_id,
            name=table.name,
            path=str(csv_path),
            n_rows=n_rows,
            n_cols=n_cols,
            source_page=table.page,
            profile=profile_table(table),
            scale_factor=table.scale_factor,
            currency=table.currency,
            agreement=table.agreement,
            dataset_id=dataset_id,
        )
        dataset_ids.append(dataset_id)

    status = "ready" if not problems and not warnings else "ready_with_warnings"
    # Re-written because warnings can be *added* after extraction — a missing
    # embedding is discovered here, not by the extractor.
    store.upsert_document(
        workspace_id=workspace_id,
        doc_id=result.doc_id,
        source_name=result.source_name,
        kind=result.kind,
        sha256=result.sha256,
        page_count=result.page_count,
        status=status,
        low_conf_pages=result.low_confidence_pages,
        warnings=warnings,
    )

    return IngestReport(
        doc_id=result.doc_id,
        source_name=result.source_name,
        kind=result.kind,
        pages=result.page_count,
        chunks=stats.chunks,
        table_chunks=stats.table_chunks,
        datasets=dataset_ids,
        anchor_completeness=stats.anchor_completeness,
        anchor_problems=problems,
        low_confidence_pages=result.low_confidence_pages,
        warnings=warnings,
        embedded=embeddings is not None,
        embedder=embedder_name,
        duration_ms=int((time.perf_counter() - started) * 1000),
        status=status,
    )


def ingest_directory(
    directory: str | Path,
    *,
    workspace_id: str,
    store: Store,
    embedder: Embedder | None = None,
    dataset_dir: str | Path = "./storage/datasets",
    patterns: Sequence[str] = ("*.pdf", "*.csv", "*.xlsx", "*.docx", "*.txt"),
) -> list[IngestReport]:
    """Ingest every supported file in a directory, newest-first order stable."""
    root = Path(directory)
    reports: list[IngestReport] = []
    seen: set[Path] = set()

    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            reports.append(
                ingest_file(
                    path,
                    workspace_id=workspace_id,
                    store=store,
                    embedder=embedder,
                    dataset_dir=dataset_dir,
                )
            )
    return reports


def stats_by_chunking(chunks_stats: ChunkingStats) -> dict[str, Any]:
    return chunks_stats.to_dict()
