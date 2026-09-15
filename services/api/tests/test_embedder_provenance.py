"""A vector only means anything inside its own model's space.

Found in live data. One workspace's `sales_sample.csv` carried 1024-wide
vectors from Voyage while a document ingested after the provider changed
carried 768-wide ones from Gemini. They happened to sit in different
workspaces, so nothing broke — but nothing prevented them sharing one either,
and `load_vectors` stacked whatever it found.

Two ways that ends, and the second is worse:

  * `np.vstack` raises on mismatched widths and retrieval dies outright. Loud,
    recoverable, obvious.
  * The widths coincide — 1024 is a common default across providers — and the
    dot products are computed across two unrelated spaces. The scores look
    entirely normal and are meaningless. Nothing reports anything.

So the width is filtered at load, the model is recorded per document, and the
skipped chunks are counted rather than quietly dropped.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from app.ingest.chunk import Chunk
from app.store.local import LocalStore

WORKSPACE = "ws_provenance"


def chunk_at(index: int) -> Chunk:
    return Chunk(
        chunk_id=f"chk_{index:03d}",
        doc_id="doc_a",
        page=1,
        para_idx=index,
        section=None,
        char_start=index * 10,
        char_end=index * 10 + 9,
        text=f"sentence number {index}",
        token_count=4,
        kind="prose",
    )


@pytest.fixture
def store(tmp_path: Path):
    store = LocalStore(tmp_path / "resx.db")
    store.upsert_document(
        workspace_id=WORKSPACE,
        doc_id="doc_a",
        source_name="a.csv",
        kind="csv",
        sha256="a" * 64,
        page_count=1,
    )
    yield store
    store.close()


def test_vectors_of_another_width_are_left_out(store: LocalStore) -> None:
    """The filter. Asking for 768 must not return the 1024-wide ones."""
    store.add_chunks(
        workspace_id=WORKSPACE,
        chunks=[chunk_at(0), chunk_at(1)],
        embeddings=np.ones((2, 768), dtype="float32"),
    )
    store.add_chunks(
        workspace_id=WORKSPACE,
        chunks=[chunk_at(2)],
        embeddings=np.ones((1, 1024), dtype="float32"),
    )

    ids, matrix = store.load_vectors(workspace_id=WORKSPACE, dimensions=768)
    assert matrix.shape == (2, 768)
    assert set(ids) == {"chk_000", "chk_001"}


def test_a_mixed_workspace_does_not_crash_without_a_width(store: LocalStore) -> None:
    """A caller that does not say which width it wants still gets a usable
    matrix rather than a `ValueError` out of `np.vstack`."""
    store.add_chunks(
        workspace_id=WORKSPACE,
        chunks=[chunk_at(0), chunk_at(1)],
        embeddings=np.ones((2, 768), dtype="float32"),
    )
    store.add_chunks(
        workspace_id=WORKSPACE,
        chunks=[chunk_at(2)],
        embeddings=np.ones((1, 1024), dtype="float32"),
    )

    ids, matrix = store.load_vectors(workspace_id=WORKSPACE)
    # The larger consistent group wins; nothing is stacked across widths.
    assert matrix.shape == (2, 768)
    assert len(ids) == matrix.shape[0]


def test_skipped_vectors_are_reported_not_swallowed(
    store: LocalStore, caplog: pytest.LogCaptureFixture
) -> None:
    """Silently returning fewer results is how a corpus appears to have lost
    documents. The remedy is to re-ingest, and the log has to say so."""
    store.add_chunks(
        workspace_id=WORKSPACE,
        chunks=[chunk_at(0)],
        embeddings=np.ones((1, 768), dtype="float32"),
    )
    store.add_chunks(
        workspace_id=WORKSPACE,
        chunks=[chunk_at(1)],
        embeddings=np.ones((1, 1024), dtype="float32"),
    )

    with caplog.at_level(logging.WARNING):
        store.load_vectors(workspace_id=WORKSPACE, dimensions=768)

    assert any("re-ingest" in r.message.lower() for r in caplog.records)


def test_a_uniform_workspace_is_unaffected(store: LocalStore) -> None:
    """The common case must not pay for the rare one."""
    store.add_chunks(
        workspace_id=WORKSPACE,
        chunks=[chunk_at(i) for i in range(5)],
        embeddings=np.ones((5, 768), dtype="float32"),
    )
    ids, matrix = store.load_vectors(workspace_id=WORKSPACE, dimensions=768)
    assert matrix.shape == (5, 768)
    assert len(ids) == 5


def test_the_model_is_recorded_on_the_document(store: LocalStore) -> None:
    """So a later run can answer "is this searchable with what I am holding"
    without measuring a stored vector."""
    store.set_document_embedder(
        workspace_id=WORKSPACE,
        doc_id="doc_a",
        embedder="gemini-embedding-001",
        dimensions=768,
    )
    doc = store.get_document(workspace_id=WORKSPACE, doc_id="doc_a")
    assert doc is not None
    assert doc["embedder"] == "gemini-embedding-001"
    assert doc["embedding_dimensions"] == 768


def test_ingest_records_the_model_it_used(tmp_path: Path) -> None:
    """End to end: the pipeline writes it, not just the store method."""
    from app.ingest.pipeline import ingest_file
    from app.rag.embeddings import HashingEmbedder

    source = tmp_path / "sales.csv"
    source.write_text("region,revenue\nNorth,100\nSouth,200\n", encoding="utf-8")

    store = LocalStore(tmp_path / "resx.db")
    try:
        report = ingest_file(
            source,
            workspace_id=WORKSPACE,
            store=store,
            embedder=HashingEmbedder(dimensions=64),
            dataset_dir=tmp_path / "datasets",
        )
        doc = store.get_document(workspace_id=WORKSPACE, doc_id=report.doc_id)
        assert doc is not None
        assert doc["embedding_dimensions"] == 64
        assert doc["embedder"], "the model name must be recorded, not left blank"
    finally:
        store.close()
