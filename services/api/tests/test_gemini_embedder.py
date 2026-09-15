"""The Gemini embedding provider.

Added for the rate limit rather than the quality. Voyage's free tier allows 3
requests and 10,000 tokens per minute and the token ceiling binds first, so a
real PDF spends most of its ingest waiting out 429s; Gemini's free tier is
materially more generous. It is still a hosted API — nothing here runs locally.

Nothing in this file touches the network: the transport is stubbed, because
what needs testing is our request shaping, ordering guarantees and error
handling rather than Google's uptime.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pytest

from app.core.config import Settings
from app.rag.embeddings import (
    EmbeddingError,
    GeminiEmbedder,
    VoyageEmbedder,
    build_embedder,
)


def stub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: int = 200,
    body: Any = None,
    capture: list[dict[str, Any]] | None = None,
    text: str = "",
) -> None:
    import httpx

    class _Response:
        status_code = status
        # `retry-after` is read from here; an empty mapping means "no hint".
        headers: ClassVar[dict[str, str]] = {}

        @staticmethod
        def json() -> Any:
            return body

        @property
        def text(self) -> str:
            return text

    def fake_post(url: str, **kwargs: Any) -> Any:
        if capture is not None:
            capture.append({"url": url, **kwargs})
        return _Response()

    monkeypatch.setattr(httpx, "post", fake_post)


def vectors(count: int, dims: int = 8) -> dict[str, Any]:
    return {"embeddings": [{"values": [0.1] * dims} for _ in range(count)]}


# --------------------------------------------------------------------------- #
# Request shaping
# --------------------------------------------------------------------------- #


def test_documents_and_queries_use_different_task_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini's names for the same idea Voyage calls `input_type`. Getting them
    the wrong way round costs real recall and raises no error."""
    calls: list[dict[str, Any]] = []
    stub(monkeypatch, body=vectors(1), capture=calls)
    embedder = GeminiEmbedder("k", dimensions=8)

    embedder.embed_documents(["a document"])
    embedder.embed_query("a question")

    assert calls[0]["json"]["requests"][0]["taskType"] == "RETRIEVAL_DOCUMENT"
    assert calls[1]["json"]["requests"][0]["taskType"] == "RETRIEVAL_QUERY"


def test_the_api_key_is_sent_as_a_header_not_in_the_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A URL is logged by proxies and written to shell history."""
    calls: list[dict[str, Any]] = []
    stub(monkeypatch, body=vectors(1), capture=calls)
    GeminiEmbedder("secret-key", dimensions=8).embed_query("q")

    assert "secret-key" not in calls[0]["url"]
    assert calls[0]["headers"]["x-goog-api-key"] == "secret-key"


def test_the_model_path_is_normalised(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemini wants `models/<name>`; a bare name and a prefixed one must both
    work rather than one of them 404ing."""
    calls: list[dict[str, Any]] = []
    stub(monkeypatch, body=vectors(1), capture=calls)
    GeminiEmbedder("k", model="gemini-embedding-001", dimensions=8).embed_query("q")
    GeminiEmbedder("k", model="models/gemini-embedding-001", dimensions=8).embed_query("q")

    assert calls[0]["url"] == calls[1]["url"]
    assert calls[0]["url"].endswith("models/gemini-embedding-001:batchEmbedContents")


def test_batches_split_on_tokens_as_well_as_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A count-only split ignores the limit that actually rejects a request."""
    calls: list[dict[str, Any]] = []
    stub(monkeypatch, body=vectors(1), capture=calls)
    embedder = GeminiEmbedder("k", dimensions=8, token_budget=100, batch_size=100)

    # Each text is ~300 tokens at 3.2 chars/token, so no two fit in one batch.
    embedder.embed_documents(["x" * 1000, "y" * 1000, "z" * 1000])
    assert len(calls) == 3


# --------------------------------------------------------------------------- #
# The guarantees that matter
# --------------------------------------------------------------------------- #


def test_a_length_mismatch_is_an_error_not_a_silent_mismapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini returns results in request order with no index to sort by, so a
    length mismatch means the text-to-vector mapping is unknown. Accepting it
    would attach the wrong vector to a chunk and quietly wreck retrieval —
    with no error anywhere to explain the bad results."""
    stub(monkeypatch, body=vectors(2))
    with pytest.raises(EmbeddingError, match="order cannot be trusted"):
        GeminiEmbedder("k", dimensions=8).embed_documents(["a", "b", "c"])


def test_vectors_are_unit_length(monkeypatch: pytest.MonkeyPatch) -> None:
    """`gemini-embedding-001` is Matryoshka-trained, so a truncated vector is
    no longer unit length. This system treats a dot product as cosine
    similarity, which is only true for unit vectors."""
    stub(monkeypatch, body={"embeddings": [{"values": [3.0, 4.0]}]})
    vector = GeminiEmbedder("k", dimensions=2).embed_query("q")
    assert np.isclose(float(np.linalg.norm(vector)), 1.0)


def test_a_rejected_key_says_what_to_do(monkeypatch: pytest.MonkeyPatch) -> None:
    stub(monkeypatch, status=403, body={}, text="forbidden")
    with pytest.raises(EmbeddingError, match="GEMINI_API_KEY"):
        GeminiEmbedder("bad", dimensions=8).embed_query("q")


def test_a_rejected_dimensionality_names_the_supported_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub(
        monkeypatch,
        status=400,
        body={},
        text="Invalid value for outputDimensionality",
    )
    with pytest.raises(EmbeddingError, match="3072"):
        GeminiEmbedder("k", dimensions=99_999).embed_query("q")


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def test_the_provider_is_selected_from_settings() -> None:
    embedder = build_embedder(
        Settings(embedding_provider="gemini", gemini_api_key="k", embedding_model="")
    )
    assert isinstance(embedder, GeminiEmbedder)
    assert embedder.model_name == "gemini-embedding-001"
    assert embedder.is_semantic


def test_voyage_is_unaffected() -> None:
    embedder = build_embedder(
        Settings(
            embedding_provider="voyage",
            voyage_api_key="k",
            embedding_model="voyage-3",
        )
    )
    assert isinstance(embedder, VoyageEmbedder)


def test_switching_provider_without_the_model_is_refused() -> None:
    """The trap this guard exists for, caught the first time it was tried:
    `EMBEDDING_PROVIDER` and `EMBEDDING_MODEL` are set separately, so a switch
    leaves the old model name behind and Gemini is asked for "voyage-3". The
    provider answers 400 with nothing explanatory."""
    with pytest.raises(EmbeddingError, match="not a gemini model"):
        build_embedder(
            Settings(
                embedding_provider="gemini",
                gemini_api_key="k",
                embedding_model="voyage-3",
            )
        )


def test_the_refusal_says_the_corpus_must_be_re_ingested() -> None:
    """An embedding is only meaningful inside its own model's space, so a
    corpus embedded with one provider and queried with another returns
    confident nonsense rather than an error."""
    with pytest.raises(EmbeddingError, match="re-ingested"):
        build_embedder(
            Settings(
                embedding_provider="gemini",
                gemini_api_key="k",
                embedding_model="voyage-3",
            )
        )
