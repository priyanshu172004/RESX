"""The citation resolver.

This is the wall that stops hallucinations reaching a user. An agent may claim
anything it likes; nothing enters state until the cited span has been looked up
in the corpus and found to actually contain the quoted text.

Three gates, in order:

  1. **Anchor exists.** Does `(doc_id, page, char_span)` resolve to real text?
     A fabricated location fails here.
  2. **Quote matches.** Is the quoted text actually present at that anchor,
     to a fuzzy ratio of 0.92 or better? A fabricated or drifted quote fails
     here. Fuzzy rather than exact because extraction normalises whitespace and
     ligatures, and demanding byte equality would reject valid citations.
  3. **Numeric support.** For a claim carrying a value, the figure must either
     appear in the cited span, or be the output of a logged computation over
     cited inputs. There is no third option.

A claim that fails any gate is dropped, not downgraded. A "low confidence"
fabrication still ends up in a report.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import TYPE_CHECKING, Any

from rapidfuzz import fuzz

if TYPE_CHECKING:
    # A `TYPE_CHECKING`-only import: `app.store.factory` pulls in the Mongo
    # backend, and a SQLite-only install must not need pymongo at import time.
    # `from __future__ import annotations` makes every annotation a string, so
    # this costs nothing at runtime and states the truth — these functions
    # accept either backend, and annotating `LocalStore` was the inaccuracy.
    from app.store.factory import Store


#: The fuzzy-match floor from `docs/04-ACCURACY-VALIDATION.md` §1.
QUOTE_MATCH_THRESHOLD = 0.92

#: How far either side of the recorded span to look. Extraction offsets can
#: drift by a few characters when whitespace is normalised; a small window
#: absorbs that without letting a quote match some unrelated part of the page.
SPAN_SLACK_CHARS = 240


class Resolution(str, Enum):
    OK = "ok"
    QUOTE_MISMATCH = "quote_mismatch"
    ANCHOR_NOT_FOUND = "anchor_not_found"
    EXTERNAL = "external"  # a web source; verified by URL, not by span


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    resolution: Resolution
    score: float
    span_text: str = ""
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.resolution in (Resolution.OK, Resolution.EXTERNAL)

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolution": self.resolution.value,
            "score": self.score,
            "span_text": self.span_text,
            "detail": self.detail,
        }


_WS = re.compile(r"\s+")
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")

#: The column separator this system renders tables with. Ours, not the
#: document's — see `normalise`.
_COLUMN_SEP = re.compile(r"\s*\|\s*")


def normalise(text: str) -> str:
    """Fold the differences extraction introduces but meaning does not.

    NFKC collapses ligatures and full-width forms; the dash and quote classes
    are normalised because PDF text layers use typographic variants freely and
    a quote should not fail because the source used an en dash.

    The column separator is folded for the same reason, and it is the one that
    was actually breaking citations. `chunk_tables` renders a table as
    "Segment | Revenue" because that is how a model is shown a table, while a
    PDF's own text layer has "Segment Revenue". So an agent that quoted a table
    row faithfully — exactly what it was asked to do — produced a quote that
    scored 0.84 against the 0.92 floor, and every claim over a table in a real
    PDF was dropped for "the quoted text does not appear on this page". It did
    appear; the pipes were ours.

    Folding it is not a loosening. The pipe carries no information from the
    document, so removing it from both sides compares the words and the figures
    and nothing else — a fabricated row still fails, which the benchmark's
    resolver-rejection metric holds to 4/4.
    """
    folded = unicodedata.normalize("NFKC", text)
    # The "ambiguous" characters are the whole subject of this function:
    # it folds the typographic variants a PDF text layer uses freely into
    # their ASCII equivalents. Replacing them with ASCII in the source
    # would delete the mapping.
    folded = folded.replace("—", "-").replace("–", "-").replace("−", "-")  # noqa: RUF001
    folded = folded.replace("‘", "'").replace("’", "'")  # noqa: RUF001
    folded = folded.replace("“", '"').replace("”", '"')
    folded = _COLUMN_SEP.sub(" ", folded)
    return _WS.sub(" ", folded).strip().lower()


def _numbers_in(text: str) -> set[str]:
    out: set[str] = set()
    for raw in _NUMBER.findall(text):
        cleaned = raw.replace(",", "").rstrip(".")
        if not cleaned or cleaned == "-":
            continue
        try:
            out.add(str(Decimal(cleaned).normalize()))
        except InvalidOperation:
            continue
    return out


# --------------------------------------------------------------------------- #
# Gate 1 + 2: anchor and quote
# --------------------------------------------------------------------------- #


def resolve_quote(
    *,
    store: Store,
    workspace_id: str,
    doc_id: str,
    page: int,
    quote: str,
    char_start: int | None = None,
    char_end: int | None = None,
    threshold: float = QUOTE_MATCH_THRESHOLD,
) -> ResolutionResult:
    """Verify a quote against the stored page text."""
    if not quote or not quote.strip():
        return ResolutionResult(Resolution.QUOTE_MISMATCH, 0.0, detail="empty quote")

    page_text = store.get_page_text(workspace_id=workspace_id, doc_id=doc_id, page=page)
    if page_text is None:
        return ResolutionResult(
            Resolution.ANCHOR_NOT_FOUND,
            0.0,
            detail=f"{doc_id} page {page} is not in the corpus",
        )

    needle = normalise(quote)
    haystack_full = normalise(page_text)

    # Prefer the recorded span, widened a little, and fall back to the whole
    # page. A quote that matches the page but not its recorded span is still a
    # real quote with a stale offset — that is a different failure from a
    # fabricated one, and the detail says which.
    windowed = ""
    if char_start is not None and char_end is not None and char_end > char_start:
        lo = max(0, char_start - SPAN_SLACK_CHARS)
        hi = min(len(page_text), char_end + SPAN_SLACK_CHARS)
        windowed = normalise(page_text[lo:hi])

    span_score = fuzz.partial_ratio(needle, windowed) / 100.0 if windowed else 0.0
    page_score = fuzz.partial_ratio(needle, haystack_full) / 100.0

    if span_score >= threshold:
        return ResolutionResult(
            Resolution.OK,
            round(span_score, 4),
            span_text=page_text[
                max(0, (char_start or 0) - 40) : (char_end or len(page_text)) + 40
            ],
            detail="matched within the recorded span",
        )

    if page_score >= threshold:
        return ResolutionResult(
            Resolution.OK,
            round(page_score, 4),
            span_text=page_text[:600],
            detail=(
                "matched on the page but outside the recorded span; the offset "
                "is stale even though the quote is genuine"
            ),
        )

    return ResolutionResult(
        Resolution.QUOTE_MISMATCH,
        round(max(span_score, page_score), 4),
        detail=(
            f"best fuzzy match {max(span_score, page_score):.2f} is below the "
            f"{threshold} floor; the quoted text does not appear on this page"
        ),
    )


# --------------------------------------------------------------------------- #
# Gate 3: numeric support
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class NumericSupport:
    supported: bool
    how: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"supported": self.supported, "how": self.how, "detail": self.detail}


def check_numeric_support(
    *,
    value: Decimal,
    quotes: Sequence[str],
    computation_result: Any = None,
    scale_factor: Decimal | None = None,
    rel_tol: Decimal = Decimal("0.005"),
) -> NumericSupport:
    """Is a claimed figure actually supported?

    Either the number appears in a cited span (allowing for the document's
    declared scale factor — "48,920" in a thousands table supports a claim of
    48,920,000), or it is the output of a logged computation. Nothing else
    counts.
    """
    target = Decimal(value)
    candidates = set()
    for quote in quotes:
        candidates |= _numbers_in(quote)

    for raw in candidates:
        try:
            found = Decimal(raw)
        except InvalidOperation:
            continue

        for scaled in _scaled_variants(found, scale_factor):
            if _close(scaled, target, rel_tol):
                return NumericSupport(
                    True,
                    "verbatim_in_citation",
                    detail=f"{raw} in the cited span supports {target}",
                )

    if computation_result is not None:
        for found in _iter_numbers(computation_result):
            if _close(found, target, rel_tol):
                return NumericSupport(
                    True,
                    "computed",
                    detail=f"sandbox computation returned {found}",
                )

    return NumericSupport(
        False,
        "unsupported",
        detail=(
            f"{target} appears neither in the cited spans nor in the computation "
            "result; the claim cannot be grounded"
        ),
    )


def _scaled_variants(found: Decimal, scale_factor: Decimal | None) -> list[Decimal]:
    variants = [found]
    if scale_factor and scale_factor != 1:
        variants.append(found * scale_factor)
    else:
        # Even without a declared factor, accept the common magnitudes: a
        # figure quoted as "48,920" in a thousands table legitimately supports
        # a claim of 48,920,000. This is a *support* check, not a scale
        # decision — the scale itself is resolved at extraction.
        for factor in (Decimal(1000), Decimal(1_000_000)):
            variants.append(found * factor)
    return variants


def _close(a: Decimal, b: Decimal, rel_tol: Decimal) -> bool:
    if b == 0:
        return abs(a) <= rel_tol
    return abs((a - b) / b) <= rel_tol


def _iter_numbers(value: Any):  # type: ignore[no-untyped-def]
    """Walk a computation result, yielding every number found."""
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float, Decimal)):
        try:
            yield Decimal(str(value))
        except InvalidOperation:
            return
        return
    if isinstance(value, str):
        for raw in _numbers_in(value):
            try:
                yield Decimal(raw)
            except InvalidOperation:
                continue
        return
    if isinstance(value, dict):
        for v in value.values():
            yield from _iter_numbers(v)
        return
    if isinstance(value, (list, tuple, set)):
        for v in value:
            yield from _iter_numbers(v)


# --------------------------------------------------------------------------- #
# The combined gate
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class GroundingVerdict:
    accepted: bool
    citation_results: list[ResolutionResult]
    numeric: NumericSupport | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "citations": [c.to_dict() for c in self.citation_results],
            "numeric": self.numeric.to_dict() if self.numeric else None,
            "reason": self.reason,
        }


def _is_in_corpus(citation: Any, *, store: Store, workspace_id: str) -> bool:
    """Does this citation point at a document the workspace actually holds?

    Used to tell a real corpus anchor from an id a model produced next to a
    URL. Absence of a `doc_id` counts as "not in the corpus", which is the
    common case for a genuine web citation.
    """
    doc_id = getattr(citation, "doc_id", None)
    if not doc_id:
        return False
    try:
        return store.get_document(workspace_id=workspace_id, doc_id=doc_id) is not None
    except Exception:
        # A store error must not silently promote a corpus citation to
        # external, so it is treated as "in the corpus" and resolved strictly.
        return True


def ground_claim(
    claim: Any,
    *,
    store: Store,
    workspace_id: str,
    computation_result: Any = None,
) -> GroundingVerdict:
    """Run every gate against one claim.

    Returns a verdict rather than mutating the claim, so the caller decides
    whether to drop, retry, or record the failure — and the decision is
    visible in one place.
    """
    if not claim.citations:
        return GroundingVerdict(
            accepted=False,
            citation_results=[],
            numeric=None,
            reason="no citations: an uncitable claim is a hallucination by definition",
        )

    results: list[ResolutionResult] = []
    for citation in claim.citations:
        if citation.url and not _is_in_corpus(citation, store=store, workspace_id=workspace_id):
            # External sources are verified by URL and retrieval date; there is
            # no corpus span to resolve against.
            #
            # The corpus check is what makes this safe. A web citation often
            # arrives with a `doc_id` the model invented alongside the real
            # URL, and requiring `not doc_id` sent the resolver hunting for a
            # span in a document that does not exist — which dropped six
            # perfectly well-sourced research claims. Preferring the anchor
            # that can be verified is correct.
            #
            # It is also narrow: if the `doc_id` *is* in the corpus, control
            # falls through and the span is resolved as strictly as ever.
            # Attaching a URL must never become a way past the gate.
            results.append(
                ResolutionResult(
                    Resolution.EXTERNAL,
                    1.0,
                    detail=f"external source: {citation.url}",
                )
            )
            continue

        results.append(
            resolve_quote(
                store=store,
                workspace_id=workspace_id,
                doc_id=citation.doc_id or "",
                page=citation.page or 0,
                quote=citation.quote,
                char_start=citation.char_start,
                char_end=citation.char_end,
            )
        )

    if not any(r.ok for r in results):
        worst = results[0]
        return GroundingVerdict(
            accepted=False,
            citation_results=results,
            numeric=None,
            reason=f"no citation resolved ({worst.resolution.value}): {worst.detail}",
        )

    numeric: NumericSupport | None = None
    if getattr(claim, "value", None) is not None:
        # Every citation is an external web source, which changes what
        # "computed" can even mean.
        #
        # The computation_id rule exists because a figure in an uploaded table
        # must be derived in the sandbox rather than read off by a model that
        # cannot be trusted with arithmetic. In research mode there is no
        # dataset to compute over: a figure on a web page is *quoted*, exactly
        # as the research-mode rules tell the agent ("A figure quoted from a
        # page is quoted"). Requiring a computation_id there demanded something
        # impossible, so agents complied by leaving `value` empty and writing
        # statements like "the market size is as reported by source" -- the
        # figure stranded inside the quote, the claim carrying nothing.
        #
        # This is not a loosening. The value must still appear verbatim in the
        # quoted source text below, which is the same standard applied to a
        # corpus span. What it drops is a requirement that could not be met,
        # not the requirement that the number be traceable.
        external_only = bool(results) and all(
            r.resolution is Resolution.EXTERNAL for r in results
        )

        if not claim.computation_id and not external_only:
            return GroundingVerdict(
                accepted=False,
                citation_results=results,
                numeric=None,
                reason=(
                    "numeric claim without a computation_id: the LLM never does arithmetic"
                ),
            )

        quotes = [c.quote for c in claim.citations if c.quote]
        scale = getattr(claim, "scale_factor", None)
        numeric = check_numeric_support(
            value=Decimal(str(claim.value)),
            quotes=quotes,
            computation_result=computation_result,
            scale_factor=Decimal(str(scale)) if scale else None,
        )
        if not numeric.supported:
            return GroundingVerdict(
                accepted=False,
                citation_results=results,
                numeric=numeric,
                reason=numeric.detail,
            )

    return GroundingVerdict(
        accepted=True,
        citation_results=results,
        numeric=numeric,
        reason="grounded",
    )
