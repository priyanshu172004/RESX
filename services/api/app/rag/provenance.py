"""Citations derived from a computation's own inputs.

The problem this solves, found by running the system on a four-row CSV.

A claim like "total revenue was 1,765,000" is *derived*: the figure appears
nowhere in the source, so there is nothing to quote. The rules therefore ask
the agent to cite the input rows instead. That works in principle and fails in
practice, because transcribing a table row exactly is a task language models
are unreliable at — the model was shown

    North | 1200 | 450000 | 260000

and quoted

    North,1200,450000,260000

reconstructing the CSV it imagined rather than copying the text it was given.
The resolver correctly scored that 0.67 against a 0.92 floor, the claim was
dropped, and a *correct* answer with a *correct* computation was thrown away
over punctuation.

The realisation: for a computation-backed claim, the system already knows the
provenance exactly. The computation declares which datasets it read; a dataset
records the document, page and table it was extracted from; and the chunk for
that table has a character span. Asking the model to also transcribe a quote is
asking it to prove something already known — and then discarding good work when
its handwriting is poor.

So the quote is **taken from the stored source text** rather than from the
model's transcription. This is stricter, not looser:

  * A model-supplied quote is a claim about the source that has to be checked.
  * A derived quote *is* the source, sliced at the recorded span.

A fabricated computation cannot produce one, because there is no dataset to
walk back from — so this closes no hole. What it removes is a failure mode
where the arithmetic was right, the provenance was recorded, and the citation
was rejected for a comma.

Derived citations are marked `derived_from_computation` so a reader can tell
the two apart, and `benchmarks/score.py` can score them separately.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("resx.provenance")

#: Characters of source text to quote. Enough to identify the rows that fed the
#: computation without pasting a whole page into the report.
QUOTE_CHARS = 400


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """A location in a document, with the text that is actually there."""

    doc_id: str
    page: int
    chunk_id: str
    char_start: int
    char_end: int
    text: str
    para_idx: int | None = None
    dataset_id: str = ""


def spans_for_datasets(
    dataset_ids: list[str], *, store: Any, workspace_id: str
) -> list[SourceSpan]:
    """Walk `dataset -> document/page -> chunk` for each dataset read.

    Returns the spans whose text can be quoted verbatim. A dataset with no
    recoverable chunk is skipped rather than guessed at: a citation pointing at
    a span we cannot read is worse than no citation.
    """
    if not dataset_ids:
        return []

    by_id = {d["dataset_id"]: d for d in store.list_datasets(workspace_id=workspace_id)}

    spans: list[SourceSpan] = []
    for dataset_id in dataset_ids:
        dataset = by_id.get(dataset_id)
        if dataset is None:
            continue

        doc_id = dataset.get("doc_id")
        if not doc_id:
            continue

        chunks = store.iter_chunks(workspace_id=workspace_id, doc_ids=[doc_id])
        if not chunks:
            continue

        page = dataset.get("source_page")
        name = dataset.get("name") or ""

        # Most specific match first. A document can hold many tables, and
        # citing the wrong one would be a real defect — worse than citing
        # nothing, because it looks supported.
        candidates = [
            c for c in chunks if c.kind == "table" and (page is None or c.page == page)
        ]
        if not candidates and page is not None:
            candidates = [c for c in chunks if c.page == page]
        if not candidates:
            candidates = [c for c in chunks if c.kind == "table"]
        if not candidates:
            candidates = chunks

        # Prefer a chunk whose recorded table name matches the dataset's.
        named = [c for c in candidates if name and c.section == name]
        chunk = (named or candidates)[0]

        # The quote is sliced from the PAGE text at the chunk's span, not from
        # the chunk's own text. That distinction is load-bearing.
        #
        # `resolve_quote` matches against page text, and a table chunk's text
        # is *not* a substring of its page: the table renderer inserts a
        # "-------" separator row that exists nowhere in the source. Quoting
        # the chunk therefore scores ~0.89 against the page and is rejected —
        # a derived citation failing its own verification, which would be
        # embarrassing and undetectable.
        #
        # Taking the text from the page guarantees the quote resolves, because
        # it is literally the string being searched.
        page_text = store.get_page_text(
            workspace_id=workspace_id, doc_id=chunk.doc_id, page=chunk.page
        )
        if page_text:
            lo = max(0, chunk.char_start)
            quote = page_text[lo : lo + QUOTE_CHARS]
        else:
            # No stored page (a backend that dropped it). Fall back to the
            # chunk and accept that a table quote may not resolve, rather than
            # asserting a span we cannot read.
            quote = chunk.text[:QUOTE_CHARS]

        if not quote.strip():
            continue

        spans.append(
            SourceSpan(
                doc_id=chunk.doc_id,
                page=chunk.page,
                chunk_id=chunk.chunk_id,
                char_start=chunk.char_start,
                char_end=min(chunk.char_end, chunk.char_start + len(quote)),
                text=quote,
                para_idx=chunk.para_idx,
                dataset_id=dataset_id,
            )
        )

    return spans


def spans_for_computation(
    computation_id: str, *, store: Any, workspace_id: str
) -> list[SourceSpan]:
    """The source spans behind one computation, via the datasets it read."""
    if not computation_id:
        return []

    record = store.get_computation(workspace_id=workspace_id, computation_id=computation_id)
    if record is None:
        # No computation, no provenance. A claim citing a computation that does
        # not exist must not acquire a citation from this path.
        return []

    inputs = record.get("inputs") or []
    dataset_ids = [str(i) for i in inputs if str(i).startswith("ds_")]
    if not dataset_ids:
        # A computation over no dataset — a constant, or a figure the agent
        # typed into the code. Nothing to cite, and inventing one would assert
        # provenance that does not exist.
        return []

    return spans_for_datasets(dataset_ids, store=store, workspace_id=workspace_id)


def derive_citations(
    claim: Any, *, store: Any, workspace_id: str, citation_factory: Any
) -> list[Any]:
    """Build citations for a claim from its computation's inputs.

    `citation_factory` constructs the project's `Citation` model; it is passed
    in so this module does not depend on the agent schemas, which import the
    store — the cycle is why this lives on its own.
    """
    spans = spans_for_computation(
        getattr(claim, "computation_id", "") or "",
        store=store,
        workspace_id=workspace_id,
    )
    if not spans:
        return []

    citations = []
    for span in spans:
        citations.append(
            citation_factory(
                doc_id=span.doc_id,
                page=span.page,
                chunk_id=span.chunk_id,
                para_idx=span.para_idx,
                char_start=span.char_start,
                char_end=span.char_end,
                quote=span.text,
                # Not "ok": a reader must be able to tell a quote the agent
                # produced from one the system derived. Both are verifiable;
                # only one was the model's work.
                resolution="derived_from_computation",
                match_score=1.0,
            )
        )
    return citations
