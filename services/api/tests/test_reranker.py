"""The reranker, and the defect that made it weaker than it looked.

Coverage was unweighted: every query term counted the same. So for "what was
the operating margin of the West region", a chunk matching *the*, *of* and *in*
scored as well as one matching *operating margin West* — and the shortlist put
a decoy above the answer while looking like it was doing something.

The corpus's own document frequencies already exist in the BM25 index, so the
weights are measured rather than guessed.

**Why not a cross-encoder.** It is the better instrument and it also means
pulling roughly half a gigabyte of torch into a service with no ML runtime, a
model download before the first query, and seconds of CPU per rerank without a
GPU. That is a real trade, and not one worth making while the embedding
provider is what actually limits answer quality.
"""

from __future__ import annotations

from typing import Any, ClassVar

from app.rag.retrieve import LexicalReranker
from app.store.local import RetrievedChunk


def chunk(text: str, kind: str = "prose") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c",
        doc_id="d",
        page=1,
        text=text,
        char_start=0,
        char_end=len(text),
        para_idx=None,
        section=None,
        kind=kind,
    )


class FakeIndex:
    """A corpus where some words are everywhere and some are rare."""

    frequencies: ClassVar[dict[str, float]] = {
        "revenue": 0.08,
        "region": 0.20,
        "report": 0.10,
        "sanand": 3.50,
        "utilisation": 3.20,
        "margin": 1.10,
    }

    def _idf(self, term: str) -> float:
        return self.frequencies.get(term, 1.0)


def test_a_rare_term_match_beats_a_common_term_match() -> None:
    """The defect, stated as a test.

    One chunk matches the two rare terms and is the answer. The other repeats
    the commonest term three times and is a decoy. Unweighted coverage ranked
    them far too close together.
    """
    reranker = LexicalReranker(index=FakeIndex())
    query = "Sanand plant utilisation and revenue"

    answer = reranker.score(
        query,
        chunk("Sanand utilisation ran at 61 percent against the average."),
        numeric_query=False,
    )
    decoy = reranker.score(
        query,
        chunk("Revenue revenue revenue grew across every region in the report."),
        numeric_query=False,
    )
    assert answer > decoy
    # Not merely ahead — clearly ahead, which is what decides a shortlist.
    assert answer > decoy * 2


def test_weighting_widens_the_gap_rather_than_just_reordering() -> None:
    """Both rank correctly here; the point is how confidently."""
    query = "Sanand plant utilisation and revenue"
    answer = chunk("Sanand utilisation ran at 61 percent.")
    decoy = chunk("Revenue revenue revenue grew across every region.")

    flat = LexicalReranker()
    weighted = LexicalReranker(index=FakeIndex())

    flat_gap = flat.score(query, answer, numeric_query=False) - flat.score(
        query, decoy, numeric_query=False
    )
    weighted_gap = weighted.score(query, answer, numeric_query=False) - weighted.score(
        query, decoy, numeric_query=False
    )
    assert weighted_gap > flat_gap


def test_it_still_works_without_an_index() -> None:
    """Standalone use falls back to uniform weights — the old behaviour, which
    is an honest fallback rather than a crash."""
    reranker = LexicalReranker()
    assert reranker.score("revenue growth", chunk("revenue grew"), numeric_query=False) > 0


def test_a_query_of_only_common_words_does_not_divide_by_zero() -> None:
    """Every weight would be zero, leaving coverage undefined. Falling back to
    uniform beats returning NaN into a sort key."""

    class AllCommon:
        def _idf(self, _term: str) -> float:
            return 0.0

    reranker = LexicalReranker(index=AllCommon())
    score = reranker.score("the of in", chunk("the of in a report"), numeric_query=False)
    assert score == score  # not NaN
    assert score >= 0.0


def test_terms_near_each_other_beat_terms_scattered_apart() -> None:
    """A 900-word page mentioning "margin" at the top and "West" at the bottom
    is not about the West region's margin, and bag-of-words coverage cannot
    tell the difference."""
    reranker = LexicalReranker(index=FakeIndex())
    query = "West margin"

    together = reranker.score(query, chunk("The West margin fell."), numeric_query=False)
    apart = reranker.score(
        query,
        chunk("West " + " ".join(["filler"] * 60) + " margin"),
        numeric_query=False,
    )
    assert together > apart


def test_an_exact_figure_still_dominates() -> None:
    """A chunk containing the queried number is almost always the right chunk,
    and the new weights must not have diluted that."""
    reranker = LexicalReranker(index=FakeIndex())
    query = "what line shows 1,765,000"

    with_number = reranker.score(
        query, chunk("Total 1,765,000 for the year."), numeric_query=True
    )
    without = reranker.score(query, chunk("Total figures for the year."), numeric_query=True)
    assert with_number > without


def test_a_numeric_question_still_prefers_a_table() -> None:
    reranker = LexicalReranker(index=FakeIndex())
    query = "how much revenue by region"

    table = reranker.score(query, chunk("region | revenue", kind="table"), numeric_query=True)
    prose = reranker.score(query, chunk("region revenue"), numeric_query=True)
    assert table > prose


def test_an_empty_query_scores_nothing_rather_than_raising(
    monkeypatch: Any,
) -> None:
    reranker = LexicalReranker(index=FakeIndex())
    assert reranker.score("", chunk("anything at all"), numeric_query=False) == 0.0
