"""Okapi BM25 lexical index.

The lexical half of hybrid retrieval, and for this product it is not the junior
partner. Pure vector search fails on precisely what financial analysis asks
about most: an exact figure ("48,920"), a rare identifier ("note 22"), a
specific covenant ratio. Embeddings smooth those into their neighbourhoods;
BM25 finds them.

Implemented directly rather than pulled in as a dependency — it is forty lines
of well-specified arithmetic, and owning it means the tokenizer can be tuned
for financial text.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

# Numbers are kept intact, including thousands separators and decimals, because
# "48,920" must be findable as a unit. A tokenizer that splits on the comma
# turns the single most useful query term in a filing into "48" and "920".
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d[\d,]*\.?\d*|%")

# Deliberately short. Aggressive stop-word removal hurts here: "other income"
# and "net of tax" are meaningful phrases in a financial statement.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "and",
        "or",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "by",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
    ]
)


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def normalise_number(token: str) -> str:
    """Strip thousands separators so "48,920" and "48920" match."""
    if token and token[0].isdigit():
        return token.replace(",", "")
    return token


@dataclass(slots=True)
class BM25Index:
    """A BM25 index over a fixed corpus.

    `k1` controls term-frequency saturation and `b` controls length
    normalisation; 1.5 / 0.75 are the standard defaults and behave well on
    mixed prose-and-table text.
    """

    k1: float = 1.5
    b: float = 0.75

    doc_ids: list[str] = field(default_factory=list)
    doc_lengths: list[int] = field(default_factory=list)
    term_frequencies: list[Counter[str]] = field(default_factory=list)
    document_frequency: Counter[str] = field(default_factory=Counter)
    average_length: float = 0.0

    def add(self, doc_id: str, text: str) -> None:
        tokens = [normalise_number(t) for t in tokenize(text)]
        counts = Counter(tokens)

        self.doc_ids.append(doc_id)
        self.doc_lengths.append(len(tokens))
        self.term_frequencies.append(counts)
        for term in counts:
            self.document_frequency[term] += 1

    def finalise(self) -> None:
        total = sum(self.doc_lengths)
        self.average_length = total / len(self.doc_lengths) if self.doc_lengths else 0.0

    @property
    def size(self) -> int:
        return len(self.doc_ids)

    def _idf(self, term: str) -> float:
        """Robertson-Sparck-Jones IDF, floored at zero.

        The unfloored form goes negative for terms appearing in more than half
        the corpus, which would let a common word actively push a document
        *down* the ranking rather than merely not helping it.
        """
        n = self.size
        df = self.document_frequency.get(term, 0)
        if df == 0:
            return 0.0
        return max(0.0, math.log(1.0 + (n - df + 0.5) / (df + 0.5)))

    def search(self, query: str, *, limit: int = 50) -> list[tuple[str, float]]:
        if not self.doc_ids:
            return []
        if self.average_length == 0.0:
            self.finalise()

        query_terms = [normalise_number(t) for t in tokenize(query)]
        if not query_terms:
            return []

        idf_cache = {term: self._idf(term) for term in set(query_terms)}
        scored: list[tuple[str, float]] = []

        for i, doc_id in enumerate(self.doc_ids):
            counts = self.term_frequencies[i]
            length = self.doc_lengths[i]
            score = 0.0
            for term in query_terms:
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                idf = idf_cache[term]
                if idf == 0.0:
                    continue
                denom = tf + self.k1 * (1.0 - self.b + self.b * (length / self.average_length))
                score += idf * (tf * (self.k1 + 1.0)) / denom
            if score > 0.0:
                scored.append((doc_id, score))

        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:limit]


def build_index(
    documents: list[tuple[str, str]], *, k1: float = 1.5, b: float = 0.75
) -> BM25Index:
    index = BM25Index(k1=k1, b=b)
    for doc_id, text in documents:
        index.add(doc_id, text)
    index.finalise()
    return index
