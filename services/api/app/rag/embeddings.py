"""Embeddings.

Two backends behind one interface.

`VoyageEmbedder` is the production path. `HashingEmbedder` is a deterministic,
offline fallback so the whole pipeline is runnable and testable without an API
key or a network call — but it is **lexical, not semantic**: it captures
character-n-gram overlap, so it will match "gross profit" to "gross profits"
and will *not* match it to "margin". `is_semantic` is `False` on it, and
retrieval reports which backend produced a result, because a recall number
measured on the fallback says nothing about production recall.

That honesty matters more than it might seem: the hybrid retriever leans on
BM25 for exactly the queries financial analysis cares about most (specific
figures, rare identifiers), so the fallback is genuinely useful for testing the
plumbing — it just cannot stand in for semantic recall.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np

#: What every array in this system is: float32, because that is what the store
#: persists and what the providers return. Named rather than repeated so the
#: dtype is stated once.
FloatArray = np.ndarray[Any, np.dtype[np.float32]]


_TOKEN = re.compile(r"[a-z0-9]+")


log = logging.getLogger("resx.embeddings")


class EmbeddingError(RuntimeError):
    pass


class Embedder(ABC):
    dimensions: int
    model_name: str
    is_semantic: bool

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> FloatArray: ...

    @abstractmethod
    def embed_query(self, text: str) -> FloatArray: ...

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "dimensions": self.dimensions,
            "is_semantic": self.is_semantic,
        }


def _l2_normalise(matrix: FloatArray) -> FloatArray:
    """Unit-normalise rows so a dot product is cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalised: FloatArray = matrix / norms
    return normalised


# --------------------------------------------------------------------------- #
# Offline deterministic fallback
# --------------------------------------------------------------------------- #


class HashingEmbedder(Embedder):
    """Hashed character-n-gram embedding. Deterministic and offline.

    Not semantic — see the module docstring. Used for local development and for
    tests that need to exercise retrieval end to end without a network call.
    """

    is_semantic = False

    def __init__(self, dimensions: int = 1024, ngram: int = 4) -> None:
        if dimensions < 32:
            raise EmbeddingError("dimensions must be at least 32")
        self.dimensions = dimensions
        self.ngram = ngram
        self.model_name = f"hashing-ngram{ngram}-{dimensions}d"

    def _vector(self, text: str) -> FloatArray:
        vec = np.zeros(self.dimensions, dtype="float32")
        lowered = text.lower()

        # Word unigrams carry most of the signal for figure lookups.
        for token in _TOKEN.findall(lowered):
            idx = (
                int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
                % self.dimensions
            )
            vec[idx] += 1.0

        # Character n-grams add robustness to morphology and to the digit
        # groupings that appear in financial tables.
        squeezed = re.sub(r"\s+", " ", lowered)
        for i in range(max(0, len(squeezed) - self.ngram + 1)):
            gram = squeezed[i : i + self.ngram]
            idx = (
                int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=8).digest(), "big")
                % self.dimensions
            )
            vec[idx] += 0.35

        return vec

    def embed_documents(self, texts: Sequence[str]) -> FloatArray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype="float32")
        matrix = np.vstack([self._vector(t) for t in texts])
        return _l2_normalise(matrix)

    def embed_query(self, text: str) -> FloatArray:
        vector: FloatArray = _l2_normalise(self._vector(text).reshape(1, -1))[0]
        return vector


# --------------------------------------------------------------------------- #
# Production backend
# --------------------------------------------------------------------------- #


class VoyageEmbedder(Embedder):
    """Voyage AI embeddings over HTTP.

    Documents and queries are embedded with different `input_type` values,
    which is what the model expects and materially improves retrieval quality
    over embedding both identically.
    """

    is_semantic = True

    #: Voyage's free tier allows 3 requests per minute AND 10,000 tokens per
    #: minute, and the token ceiling is the binding one. That is the trap: a
    #: 128-chunk batch is roughly 14,000 tokens, so every request is rejected
    #: no matter how carefully the *requests* are paced. Batching therefore has
    #: to be by estimated tokens, not by count.
    DEFAULT_BATCH = 128

    #: Token budget per request, kept under the free tier's per-minute ceiling
    #: so that a single batch cannot exhaust the whole window.
    DEFAULT_TOKEN_BUDGET = 7_000

    #: Characters per token. Deliberately pessimistic: financial tables are
    #: dense with digits and punctuation and tokenize worse than prose, and
    #: under-estimating here is what produces the 429 this exists to avoid.
    CHARS_PER_TOKEN = 3.2

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "voyage-3",
        dimensions: int = 1024,
        batch_size: int | None = None,
        token_budget: int | None = None,
        timeout: float = 60.0,
        max_retries: int = 5,
        min_interval_seconds: float = 0.0,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        if not api_key:
            raise EmbeddingError("VoyageEmbedder requires an API key")
        self.api_key = api_key
        self.model_name = model
        self.dimensions = dimensions
        self.batch_size = batch_size or self.DEFAULT_BATCH
        self.token_budget = token_budget or self.DEFAULT_TOKEN_BUDGET
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        # A client-side floor between requests. Set from RPM in `build_embedder`
        # so the first request of a batch does not have to *discover* the limit
        # by being rejected.
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self.progress = progress
        self._last_request_at = 0.0

    def _throttle(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if self._last_request_at and elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)

    def _request_batch(self, batch: list[str], input_type: str) -> list[list[float]]:
        """One batch, retried through rate limits.

        A 429 here is expected rather than exceptional: the free tier is three
        requests a minute, and an ingest of any real document exceeds that in
        seconds. Failing the whole upload because the *fourth* batch was
        throttled would make the free tier unusable, so this waits it out --
        preferring the server's `Retry-After` over a guess, because the server
        knows when the window resets.
        """
        import httpx

        last = ""
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                response = httpx.post(
                    "https://api.voyageai.com/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "input": batch,
                        "model": self.model_name,
                        "input_type": input_type,
                        "output_dimension": self.dimensions,
                    },
                    timeout=self.timeout,
                )
            except httpx.TimeoutException as exc:
                last = f"timeout: {exc}"
                if attempt < self.max_retries:
                    time.sleep(min(2.0 * (2**attempt), 60.0))
                continue
            except httpx.HTTPError as exc:
                raise EmbeddingError(f"could not reach Voyage: {exc}") from exc
            finally:
                self._last_request_at = time.monotonic()

            if response.status_code == 200:
                payload = response.json()
                # Results carry an index; never trust positional order.
                ordered = sorted(payload["data"], key=lambda d: d["index"])
                return [d["embedding"] for d in ordered]

            last = f"{response.status_code}: {response.text[:200]}"

            if response.status_code == 401:
                raise EmbeddingError(
                    "Voyage rejected the API key (401). Check VOYAGE_API_KEY, "
                    "or unset it to fall back to the non-semantic embedder."
                )
            if response.status_code not in (408, 429, 500, 502, 503, 504):
                raise EmbeddingError(f"embedding request failed ({last})")

            wait = 2.0 * (2**attempt)
            header = response.headers.get("retry-after")
            if header:
                with contextlib.suppress(ValueError):
                    wait = max(wait, float(header))
            if response.status_code == 429 and not header:
                # No hint given, and the free-tier window is a minute. Waiting
                # four seconds and failing would be worse than waiting.
                wait = max(wait, 20.0)
            if attempt < self.max_retries:
                log.info(
                    "voyage %s; waiting %.0fs before retry %d/%d",
                    response.status_code,
                    wait,
                    attempt + 1,
                    self.max_retries,
                )
                time.sleep(min(wait, 90.0))

        raise EmbeddingError(
            f"embedding request failed after {self.max_retries + 1} attempts "
            f"({last}). Voyage's free tier allows 3 requests per minute; unset "
            "VOYAGE_API_KEY to use the offline embedder instead."
        )

    def _batches(self, texts: list[str]) -> Iterator[list[str]]:
        """Split by estimated tokens as well as by count.

        A count-only split ignores the limit that actually rejects the request.
        A single text over budget still goes on its own — the provider will
        reject an over-long input either way, and splitting it here would
        change the embedding.
        """
        batch: list[str] = []
        budget = 0
        for text in texts:
            cost = max(1, int(len(text) / self.CHARS_PER_TOKEN))
            over_tokens = batch and budget + cost > self.token_budget
            over_count = len(batch) >= self.batch_size
            if over_tokens or over_count:
                yield batch
                batch, budget = [], 0
            batch.append(text)
            budget += cost
        if batch:
            yield batch

    def _post(self, texts: list[str], input_type: str) -> FloatArray:
        rows: list[list[float]] = []
        total = len(texts)
        done = 0
        for batch in self._batches(texts):
            rows.extend(self._request_batch(batch, input_type))
            done += len(batch)
            if self.progress is not None:
                self.progress(done, total)

        return _l2_normalise(np.asarray(rows, dtype="float32"))

    def embed_documents(self, texts: Sequence[str]) -> FloatArray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype="float32")
        return self._post(list(texts), "document")

    def embed_query(self, text: str) -> FloatArray:
        vector: FloatArray = self._post([text], "query")[0]
        return vector


class GeminiEmbedder(Embedder):
    """Google's `gemini-embedding-001` over HTTP.

    Added because the *rate limit*, not the quality, is what makes ingestion
    painful on a free tier. Voyage allows 3 requests and 10,000 tokens per
    minute, and the token ceiling binds first: a real PDF exceeds it in
    seconds, so an ingest spends most of its time waiting out 429s. Gemini's
    free tier is far more generous, which is the whole reason this exists.

    Two differences from Voyage worth knowing:

    * **Task type, not input type.** `RETRIEVAL_DOCUMENT` and
      `RETRIEVAL_QUERY` are the same idea under a different name, and getting
      them the wrong way round costs real recall.
    * **One text per request part.** `batchEmbedContents` takes a list of
      requests rather than a list of strings, so the batch is assembled
      differently even though the batching *policy* is shared.

    The model is Matryoshka-trained, so `output_dimensionality` genuinely
    truncates rather than degrading arbitrarily — but a truncated vector is no
    longer unit-length, so it is re-normalised. Skipping that silently breaks
    cosine similarity, because the dot product this system uses as cosine only
    equals cosine for unit vectors.
    """

    is_semantic = True

    #: Requests per batch. Gemini accepts up to 100 parts per
    #: `batchEmbedContents` call.
    DEFAULT_BATCH = 100

    #: Token budget per request. Higher than Voyage's because the per-minute
    #: ceiling is higher, but still bounded: one enormous request that fails
    #: costs the whole batch.
    DEFAULT_TOKEN_BUDGET = 18_000

    CHARS_PER_TOKEN = 3.2

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gemini-embedding-001",
        dimensions: int = 1024,
        batch_size: int | None = None,
        token_budget: int | None = None,
        timeout: float = 60.0,
        max_retries: int = 5,
        min_interval_seconds: float = 0.0,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        if not api_key:
            raise EmbeddingError("GeminiEmbedder requires an API key")
        self.api_key = api_key
        self.model_name = model
        self.dimensions = dimensions
        self.batch_size = batch_size or self.DEFAULT_BATCH
        self.token_budget = token_budget or self.DEFAULT_TOKEN_BUDGET
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self.progress = progress
        self._last_request_at = 0.0

    def _throttle(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if self._last_request_at and elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)

    def _batches(self, texts: list[str]) -> Iterator[list[str]]:
        """Split by estimated tokens as well as by count.

        Same policy as Voyage and for the same reason: a count-only split
        ignores the limit that actually rejects the request.
        """
        batch: list[str] = []
        budget = 0
        for text in texts:
            cost = max(1, int(len(text) / self.CHARS_PER_TOKEN))
            if batch and (budget + cost > self.token_budget or len(batch) >= self.batch_size):
                yield batch
                batch, budget = [], 0
            batch.append(text)
            budget += cost
        if batch:
            yield batch

    def _request_batch(self, batch: list[str], task_type: str) -> list[list[float]]:
        import httpx

        model_path = (
            self.model_name
            if self.model_name.startswith("models/")
            else f"models/{self.model_name}"
        )
        payload = {
            "requests": [
                {
                    "model": model_path,
                    "content": {"parts": [{"text": text}]},
                    "taskType": task_type,
                    "outputDimensionality": self.dimensions,
                }
                for text in batch
            ]
        }

        last = ""
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                response = httpx.post(
                    f"{self.BASE_URL}/{model_path}:batchEmbedContents",
                    # The key goes in a header, not the query string: a URL is
                    # logged by proxies and written to shell history, and a
                    # secret in one is a secret leaked.
                    headers={
                        "x-goog-api-key": self.api_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
            except httpx.TimeoutException as exc:
                last = f"timeout: {exc}"
                if attempt < self.max_retries:
                    time.sleep(min(2.0 * (2**attempt), 60.0))
                continue
            except httpx.HTTPError as exc:
                raise EmbeddingError(f"could not reach Gemini: {exc}") from exc
            finally:
                self._last_request_at = time.monotonic()

            if response.status_code == 200:
                body = response.json()
                embeddings = body.get("embeddings") or []
                if len(embeddings) != len(batch):
                    # Gemini returns results in request order with no index to
                    # sort by, so a length mismatch means the mapping from text
                    # to vector is unknown. Silently accepting it would attach
                    # the wrong vector to a chunk and quietly wreck retrieval.
                    raise EmbeddingError(
                        f"Gemini returned {len(embeddings)} embeddings for "
                        f"{len(batch)} inputs; the order cannot be trusted"
                    )
                return [e["values"] for e in embeddings]

            last = f"{response.status_code}: {response.text[:200]}"

            if response.status_code in (401, 403):
                raise EmbeddingError(
                    "Gemini rejected the API key. Check GEMINI_API_KEY, or set "
                    "EMBEDDING_PROVIDER=hashing to use the offline embedder."
                )
            if response.status_code == 400 and "outputDimensionality" in response.text:
                raise EmbeddingError(
                    f"{self.model_name} rejected output dimensionality "
                    f"{self.dimensions}. gemini-embedding-001 supports 3072, "
                    f"1536 and 768 natively; set EMBEDDING_DIMENSIONS to one "
                    f"of those."
                )
            if response.status_code not in (408, 429, 500, 502, 503, 504):
                raise EmbeddingError(f"embedding request failed ({last})")

            wait = 2.0 * (2**attempt)
            header = response.headers.get("retry-after")
            if header:
                with contextlib.suppress(ValueError):
                    wait = max(wait, float(header))
            if attempt < self.max_retries:
                log.info(
                    "gemini %s; waiting %.0fs before retry %d/%d",
                    response.status_code,
                    wait,
                    attempt + 1,
                    self.max_retries,
                )
                time.sleep(min(wait, 90.0))

        raise EmbeddingError(
            f"embedding request failed after {self.max_retries + 1} attempts ({last})"
        )

    def _post(self, texts: list[str], task_type: str) -> FloatArray:
        rows: list[list[float]] = []
        done = 0
        for batch in self._batches(texts):
            rows.extend(self._request_batch(batch, task_type))
            done += len(batch)
            if self.progress is not None:
                self.progress(done, len(texts))
        # Re-normalised because a Matryoshka-truncated vector is not unit
        # length, and this system treats a dot product as cosine similarity.
        return _l2_normalise(np.asarray(rows, dtype="float32"))

    def embed_documents(self, texts: Sequence[str]) -> FloatArray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype="float32")
        return self._post(list(texts), "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> FloatArray:
        vector: FloatArray = self._post([text], "RETRIEVAL_QUERY")[0]
        return vector


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


#: Model-name prefixes that belong to each provider. Used to catch a switch of
#: `EMBEDDING_PROVIDER` that left `EMBEDDING_MODEL` pointing at the old
#: provider's model -- which is easy to do, since the two are set separately,
#: and produces a 400 from the provider rather than anything explanatory.
_PROVIDER_MODEL_PREFIXES: dict[str, tuple[str, ...]] = {
    "voyage": ("voyage",),
    "gemini": ("gemini-", "models/gemini-", "text-embedding-"),
}


def _model_for(provider: str, configured: str, default: str) -> str:
    """The model to use, or an explanation of why the configured one is wrong."""
    if not configured:
        return default
    expected = _PROVIDER_MODEL_PREFIXES.get(provider, ())
    if expected and not configured.startswith(expected):
        raise EmbeddingError(
            f"EMBEDDING_MODEL is {configured!r}, which is not a {provider} "
            f"model. Switching EMBEDDING_PROVIDER to {provider!r} means "
            f"EMBEDDING_MODEL has to change too — set it to {default!r} or "
            f"leave it empty to take the provider's default. Note that "
            f"switching provider also invalidates every stored vector, so the "
            f"corpus has to be re-ingested."
        )
    return configured


def build_embedder(settings: Any) -> Embedder:
    """Pick a backend from settings, preferring the real one when configured.

    In production a missing key is an error rather than a silent downgrade: a
    system that quietly falls back to lexical-only embeddings would report
    plausible-looking recall while having lost semantic retrieval entirely.
    """
    provider = getattr(settings, "embedding_provider", "voyage")
    dimensions = int(getattr(settings, "embedding_dimensions", 1024))
    is_production = bool(getattr(settings, "is_production", False))
    rpm = int(getattr(settings, "embedding_requests_per_minute", 0) or 0)
    batch_size = int(getattr(settings, "embedding_batch_size", 0) or 0) or None
    token_budget = int(getattr(settings, "embedding_token_budget", 0) or 0) or None
    # Spread requests over the window rather than firing them all and absorbing
    # 429s: the retry path works, but it costs a full minute per rejection.
    interval = (60.0 / rpm) if rpm > 0 else 0.0

    if provider == "voyage":
        key = getattr(settings, "voyage_api_key", "") or ""
        if key:
            return VoyageEmbedder(
                key,
                model=_model_for(
                    "voyage", getattr(settings, "embedding_model", "") or "", "voyage-3"
                ),
                dimensions=dimensions,
                batch_size=batch_size,
                token_budget=token_budget,
                min_interval_seconds=interval,
            )

    if provider == "gemini":
        key = getattr(settings, "gemini_api_key", "") or ""
        if key:
            return GeminiEmbedder(
                key,
                model=_model_for(
                    "gemini",
                    getattr(settings, "embedding_model", "") or "",
                    "gemini-embedding-001",
                ),
                dimensions=dimensions,
                batch_size=batch_size,
                token_budget=token_budget,
                min_interval_seconds=interval,
            )

    if is_production:
        raise EmbeddingError(
            "no embedding provider configured; refusing to fall back to the "
            "non-semantic hashing embedder in production"
        )

    return HashingEmbedder(dimensions=dimensions)
