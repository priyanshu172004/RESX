"""Hybrid retrieval.

BM25 u vector → reciprocal rank fusion → rerank → top-k.

Hybrid is not hedging. Pure vector search reliably fails on the things a
financial question is actually about — an exact figure, a note number, a
covenant ratio — because embeddings map them into a neighbourhood of
"similar-looking numbers". BM25 finds them exactly. Conversely BM25 misses
"how profitable is this business" when the document says "margin". Each covers
the other's blind spot, and RRF combines them without needing their scores to
be on a comparable scale.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from app.rag.bm25 import build_index, normalise_number, tokenize
from app.rag.embeddings import Embedder
from app.store.local import RetrievedChunk

if TYPE_CHECKING:
    # A `TYPE_CHECKING`-only import: `app.store.factory` pulls in the Mongo
    # backend, and a SQLite-only install must not need pymongo at import time.
    # `from __future__ import annotations` makes every annotation a string, so
    # this costs nothing at runtime and states the truth — these functions
    # accept either backend, and annotating `LocalStore` was the inaccuracy.
    from app.store.factory import Store


# RRF's smoothing constant. 60 is the value from the original paper and it is
# deliberately large relative to the number of results: it flattens the
# contribution of rank position so a document ranked 3rd by both retrievers
# beats one ranked 1st by only one of them.
RRF_K = 60

_NUMBER_IN_QUERY = re.compile(r"\d[\d,]*\.?\d*")


@dataclass(slots=True)
class RetrievalDiagnostics:
    """What the retriever did. Surfaced because retrieval quality is a metric.

    `semantic` records whether the embedding backend was the real one. A recall
    figure measured with the offline hashing fallback is not a statement about
    production recall, and conflating the two would make the benchmark lie.
    """

    query: str
    lexical_hits: int = 0
    vector_hits: int = 0
    fused_candidates: int = 0
    returned: int = 0
    semantic: bool = False
    embedder: str = ""
    numeric_query: bool = False
    reranker: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "lexical_hits": self.lexical_hits,
            "vector_hits": self.vector_hits,
            "fused_candidates": self.fused_candidates,
            "returned": self.returned,
            "semantic": self.semantic,
            "embedder": self.embedder,
            "numeric_query": self.numeric_query,
            "reranker": self.reranker,
        }


@dataclass(slots=True)
class RetrievalResult:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    diagnostics: RetrievalDiagnostics | None = None

    def __len__(self) -> int:
        return len(self.chunks)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.chunks)

    def context_block(self, *, max_chars: int = 7_000) -> str:
        """Assemble retrieved text for a prompt, wrapped as untrusted content.

        Every chunk is delimited and labelled with its anchor. Two jobs:
        the agent can cite precisely, and the standing system rule about
        `<untrusted_document_content>` has something concrete to apply to — a
        document cannot issue instructions if its text never arrives
        undelimited (docs/05-SECURITY.md §4.4).

        `max_chars` is the caller's entire retrieval budget and has to be sized
        from the *model's* request limit, not from what looks like a reasonable
        amount of text. It was 24,000 — a fair figure for a 200k-token Claude
        context, and about 7,500 tokens, which on its own overflowed Groq's
        8,000-token request ceiling and killed every specialist with HTTP 413
        before the model ran. `Toolbelt` passes it from settings; this default
        is the conservative one, for any caller that forgets to.
        """
        parts: list[str] = []
        budget = max_chars
        dropped = 0
        for chunk in self.chunks:
            body = chunk.text
            header = (
                f'<untrusted_document_content doc_id="{chunk.doc_id}" '
                f'page="{chunk.page}" '
                f'para_idx="{chunk.para_idx if chunk.para_idx is not None else ""}" '
                f'chunk_id="{chunk.chunk_id}" kind="{chunk.kind}">'
            )
            block = f"{header}\n{body}\n</untrusted_document_content>"
            if len(block) > budget:
                # Skip and keep going rather than stopping here. Chunks arrive
                # ranked, so stopping at the first oversized one discarded
                # every later chunk that would have fitted — and one outsized
                # table at rank 1 returned an empty context.
                #
                # A chunk is never cut in half. The delimiters are what tell
                # the model where untrusted text begins and ends, and half a
                # wrapper leaves ambiguous exactly the boundary the wrapper
                # exists to make unambiguous.
                dropped += 1
                continue
            parts.append(block)
            budget -= len(block)

        if dropped:
            # Outside the wrappers: this line is ours, not the document's.
            # Said out loud because silence reads as "this is everything", and
            # a confident finding drawn from a silently truncated corpus is
            # the failure this whole pipeline is built to avoid.
            parts.append(
                f"[{dropped} further retrieved passage(s) did not fit this "
                f"request and were omitted. Say so if what remains is "
                f"insufficient rather than answering from a partial view.]"
            )
        return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]], *, k: int = RRF_K
) -> list[tuple[str, float]]:
    """Fuse ranked ID lists by 1/(k + rank).

    Rank-based rather than score-based on purpose: BM25 scores are unbounded
    and corpus-dependent while cosine similarity is bounded in [-1, 1], so any
    weighted sum of the two raw scores is arbitrary. Ranks are comparable.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


# --------------------------------------------------------------------------- #
# Reranking
# --------------------------------------------------------------------------- #


class LexicalReranker:
    """A cheap, dependency-free reranker.

    **Why not a cross-encoder.** It is the better instrument and it slots in
    behind this same interface, and it also means pulling a transformer runtime
    — roughly half a gigabyte of torch — into a service that currently has no
    ML runtime at all, plus a model download before the first query and seconds
    of CPU per rerank on a machine with no GPU. That is a real trade, not an
    oversight, and it is not worth making until retrieval quality is what is
    actually limiting answers. Right now the embedding provider is.

    It scores four things that matter specifically for financial retrieval:

      * **IDF-weighted term coverage.** Weighted, because unweighted coverage
        was the defect here: every query term counted equally, so a chunk
        matching "the", "of" and "in" out of "what was the operating margin of
        the West region" scored the same as one matching "operating margin
        West". The corpus's own document frequencies already exist in the BM25
        index, so the weights are measured rather than guessed.
      * **Exact numeric match** — a chunk containing the queried figure is
        almost always the right chunk.
      * **Proximity** — query terms appearing near each other, not merely
        present. A 900-word page that mentions "margin" at the top and "West"
        at the bottom is not about the West region's margin, and bag-of-words
        coverage cannot tell the difference.
      * **Table affinity** — a numeric question is usually answered by a table,
        not by prose describing one.
    """

    name = "lexical-coverage-v2"

    #: Terms this far apart in the chunk contribute nothing to proximity. Set
    #: at roughly a long sentence: closer than this and the two terms are
    #: plausibly about each other, further and they are two separate facts.
    PROXIMITY_WINDOW = 40

    def __init__(self, index: Any | None = None) -> None:
        #: The BM25 index, for its document frequencies. Optional so the
        #: reranker still works standalone — without it every term weighs the
        #: same, which is the old behaviour and an honest fallback rather than
        #: a crash.
        self.index = index

    def _weights(self, terms: list[str]) -> dict[str, float]:
        idf = getattr(self.index, "_idf", None)
        if idf is None:
            return dict.fromkeys(terms, 1.0)
        weights = {term: max(0.0, float(idf(term))) for term in terms}
        # A query of nothing but corpus-wide words would weigh zero and make
        # coverage undefined. Falling back to uniform is better than dividing
        # by zero or silently scoring everything identically.
        return weights if any(weights.values()) else dict.fromkeys(terms, 1.0)

    def _proximity(self, query_terms: set[str], chunk_tokens: list[str]) -> float:
        positions = [i for i, t in enumerate(chunk_tokens) if t in query_terms]
        if len(positions) < 2:
            # One match cannot be near anything. Neutral rather than zero: a
            # single rare-term hit is still a good signal and the coverage
            # term already counts it.
            return 0.0
        spread = min(positions[i + 1] - positions[i] for i in range(len(positions) - 1))
        return max(0.0, 1.0 - spread / self.PROXIMITY_WINDOW)

    def score(self, query: str, chunk: RetrievedChunk, *, numeric_query: bool) -> float:
        query_tokens = [normalise_number(t) for t in tokenize(query)]
        query_terms = set(query_tokens)
        if not query_terms:
            return 0.0

        chunk_tokens = [normalise_number(t) for t in tokenize(chunk.text)]
        chunk_terms = set(chunk_tokens)

        weights = self._weights(query_tokens)
        total = sum(weights[t] for t in query_terms)
        matched = sum(weights[t] for t in query_terms & chunk_terms)
        coverage = matched / total if total else 0.0

        query_numbers = {n.replace(",", "") for n in _NUMBER_IN_QUERY.findall(query)}
        chunk_numbers = {n.replace(",", "") for n in _NUMBER_IN_QUERY.findall(chunk.text)}
        numeric_hit = 1.0 if query_numbers and (query_numbers & chunk_numbers) else 0.0

        proximity = self._proximity(query_terms, chunk_tokens)
        table_affinity = 1.0 if (numeric_query and chunk.kind == "table") else 0.0

        return 0.50 * coverage + 0.25 * numeric_hit + 0.15 * proximity + 0.10 * table_affinity


def looks_numeric(query: str) -> bool:
    """Is this a question about a figure?

    Used to bias toward table chunks. Deliberately keyword-based and
    inspectable rather than a model call — a wrong guess here costs a little
    ranking quality, and it must never cost a request.
    """
    lowered = query.lower()
    signals = (
        "how much",
        "how many",
        "what was",
        "what is the",
        "total",
        "revenue",
        "profit",
        "margin",
        "cost",
        "cash",
        "burn",
        "runway",
        "ratio",
        "percentage",
        "percent",
        "%",
        "growth",
        "sum",
        "average",
        "eps",
        "ebitda",
        "debt",
        "assets",
        "liabilities",
        "equity",
    )
    return bool(_NUMBER_IN_QUERY.search(query)) or any(s in lowered for s in signals)


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #


def retrieve(
    query: str,
    *,
    store: Store,
    embedder: Embedder,
    workspace_id: str,
    doc_ids: Sequence[str] | None = None,
    top_k: int = 8,
    candidates: int = 50,
    reranker: LexicalReranker | None = None,
) -> RetrievalResult:
    """Retrieve the top-k chunks for a query, with anchors attached."""
    if not query.strip():
        return RetrievalResult(
            chunks=[], diagnostics=RetrievalDiagnostics(query=query, returned=0)
        )

    numeric = looks_numeric(query)
    diagnostics = RetrievalDiagnostics(
        query=query,
        semantic=embedder.is_semantic,
        embedder=embedder.model_name,
        numeric_query=numeric,
        reranker=(reranker or LexicalReranker()).name,
    )

    # --- lexical half ---
    all_chunks = store.iter_chunks(workspace_id=workspace_id, doc_ids=doc_ids)
    if not all_chunks:
        diagnostics.returned = 0
        return RetrievalResult(chunks=[], diagnostics=diagnostics)

    index = build_index([(c.chunk_id, c.text) for c in all_chunks])

    # Built here rather than at the top of the function because it needs the
    # index: term weights come from the corpus's own document frequencies, so
    # that "the" and "EBITDA" are not treated as equally informative. A caller
    # that supplied its own reranker keeps it.
    reranker = reranker or LexicalReranker(index=index)

    lexical = index.search(query, limit=candidates)
    lexical_ids = [chunk_id for chunk_id, _ in lexical]
    diagnostics.lexical_hits = len(lexical_ids)

    # --- vector half ---
    vector_ids: list[str] = []
    # The width is passed so vectors from a previous embedding model are left
    # out rather than silently scored against this one. Without it a workspace
    # that changed provider mid-corpus either crashed on `np.vstack` or — when
    # the widths happened to agree — produced confident scores computed across
    # two unrelated embedding spaces.
    ids, matrix = store.load_vectors(
        workspace_id=workspace_id,
        doc_ids=doc_ids,
        dimensions=getattr(embedder, "dimensions", None),
    )
    if len(ids) and matrix.size:
        query_vec = embedder.embed_query(query)
        if query_vec.shape[0] == matrix.shape[1]:
            # Both sides are L2-normalised, so a dot product is cosine.
            similarities = matrix @ query_vec
            order = np.argsort(-similarities)[:candidates]
            vector_ids = [ids[i] for i in order]
        # A dimension mismatch means the corpus was embedded with a different
        # model. Silently returning lexical-only results would look like a
        # working system with quietly halved recall, so it is surfaced.
        else:
            diagnostics.embedder += " (DIMENSION MISMATCH - vector half skipped)"
    diagnostics.vector_hits = len(vector_ids)

    # --- fuse ---
    fused = reciprocal_rank_fusion([lexical_ids, vector_ids])
    diagnostics.fused_candidates = len(fused)
    if not fused:
        diagnostics.returned = 0
        return RetrievalResult(chunks=[], diagnostics=diagnostics)

    shortlist_ids = [chunk_id for chunk_id, _ in fused[: max(top_k * 4, 20)]]
    by_id = {c.chunk_id: c for c in all_chunks}
    shortlist = [by_id[i] for i in shortlist_ids if i in by_id]

    # --- rerank ---
    rrf_scores = dict(fused)
    reranked = sorted(
        shortlist,
        key=lambda c: (
            -(
                # Rerank score leads; RRF breaks ties. Keeping RRF in the key
                # means a chunk both retrievers liked is not discarded just
                # because it shares few literal terms with the question.
                0.75 * reranker.score(query, c, numeric_query=numeric)
                + 0.25 * min(1.0, rrf_scores.get(c.chunk_id, 0.0) * RRF_K)
            ),
            c.chunk_id,
        ),
    )

    chosen = reranked[:top_k]
    final = [
        RetrievedChunk(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            page=c.page,
            text=c.text,
            char_start=c.char_start,
            char_end=c.char_end,
            para_idx=c.para_idx,
            section=c.section,
            kind=c.kind,
            score=round(
                0.75 * reranker.score(query, c, numeric_query=numeric)
                + 0.25 * min(1.0, rrf_scores.get(c.chunk_id, 0.0) * RRF_K),
                6,
            ),
            source=_source_label(c.chunk_id, lexical_ids, vector_ids),
        )
        for c in chosen
    ]

    diagnostics.returned = len(final)
    return RetrievalResult(chunks=final, diagnostics=diagnostics)


def _source_label(chunk_id: str, lexical: Sequence[str], vector: Sequence[str]) -> str:
    in_lex = chunk_id in lexical
    in_vec = chunk_id in vector
    if in_lex and in_vec:
        return "both"
    if in_lex:
        return "lexical"
    if in_vec:
        return "vector"
    return "unknown"
