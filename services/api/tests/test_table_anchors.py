"""Table chunks must be quotable and anchored.

This is the bug that made document analysis produce no findings, and the
agents were not at fault.

`chunk_tables` rendered a line of dashes between the header and the body to
make the table readable. That rule appears nowhere in the document, so an agent
that quoted what it was shown -- header, rule, rows, which is the natural thing
to copy out of a table -- produced a quote that could not resolve. Measured on
the real path:

    a quote of actual rows                     -> 1.00  ok
    the same quote with the rule included      -> 0.77  quote_mismatch

Against a 0.92 floor, every trend claim over a table was dropped with "the
quoted text does not appear on this page". Which was true, and our doing.

The anchors were wrong in the same place: `char_start`/`char_end` were set to
`(0, len(text))` -- offsets into the chunk's own rendering, which for a 237-char
rendering of a 186-char page pointed past the end of the page. `verify_anchors`
skipped table chunks entirely, so nothing ever compared the two.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from app.ingest.pipeline import ingest_file
from app.rag.citations import resolve_quote
from app.store.local import LocalStore

WORKSPACE = "ws_anchors"

TABLE = (
    "Quarter,Revenue,Cost of sales,Gross margin %\n"
    "Q1 2024,1200000,780000,35.0\n"
    "Q2 2024,1350000,850000,37.0\n"
    "Q3 2024,1180000,810000,31.4\n"
    "Q4 2024,1520000,900000,40.8\n"
)


@pytest.fixture
def ingested(tmp_path: Path) -> Iterator[tuple[LocalStore, str]]:
    source = tmp_path / "quarterly.csv"
    source.write_text(TABLE, encoding="utf-8")
    store = LocalStore(tmp_path / "resx.db")
    report = ingest_file(
        source,
        workspace_id=WORKSPACE,
        store=store,
        dataset_dir=tmp_path / "datasets",
    )
    yield store, report.doc_id
    store.close()


def _table_chunk(store: LocalStore):
    return next(c for c in store.iter_chunks(workspace_id=WORKSPACE) if c.kind == "table")


def test_a_table_chunk_carries_no_invented_separator(
    ingested: tuple[LocalStore, str],
) -> None:
    """The rule was decoration, and it was the whole bug."""
    store, _ = ingested
    assert "---" not in _table_chunk(store).text


def test_a_table_chunks_body_appears_verbatim_on_its_page(
    ingested: tuple[LocalStore, str],
) -> None:
    """The property that makes a quote resolvable at all. Notes we add are
    bracketed and excluded; everything else has to be the document's text."""
    store, doc_id = ingested
    chunk = _table_chunk(store)
    page = store.get_page_text(workspace_id=WORKSPACE, doc_id=doc_id, page=chunk.page)
    body = "\n".join(
        line for line in chunk.text.splitlines() if not line.lstrip().startswith("[")
    ).strip()
    assert body in (page or "")


def test_a_table_chunk_anchor_addresses_the_page_not_the_chunk(
    ingested: tuple[LocalStore, str],
) -> None:
    """`(0, len(text))` pointed past the end of the page for any table whose
    rendering was longer than its source, which is every table with a rule."""
    store, doc_id = ingested
    chunk = _table_chunk(store)
    page = store.get_page_text(workspace_id=WORKSPACE, doc_id=doc_id, page=chunk.page) or ""
    assert chunk.char_end <= len(page)
    assert chunk.char_start < chunk.char_end


def test_the_quote_an_agent_would_actually_write_resolves(
    ingested: tuple[LocalStore, str],
) -> None:
    """The regression, end to end: quote the first lines of the chunk exactly
    as presented, and it must resolve."""
    store, doc_id = ingested
    chunk = _table_chunk(store)
    body_lines = [line for line in chunk.text.splitlines() if not line.lstrip().startswith("[")]
    quote = "\n".join(body_lines[:3])
    verdict = resolve_quote(
        store=store,
        workspace_id=WORKSPACE,
        doc_id=doc_id,
        page=chunk.page,
        quote=quote,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
    )
    assert verdict.ok, f"{verdict.resolution.value} at {verdict.score}: {verdict.detail}"
    assert verdict.score >= 0.92


def test_a_single_row_still_resolves(ingested: tuple[LocalStore, str]) -> None:
    store, doc_id = ingested
    verdict = resolve_quote(
        store=store,
        workspace_id=WORKSPACE,
        doc_id=doc_id,
        page=1,
        quote="Q3 2024 | 1180000 | 810000 | 31.4",
    )
    assert verdict.ok


def test_a_fabricated_row_is_still_rejected(
    ingested: tuple[LocalStore, str],
) -> None:
    """Making real quotes resolve must not make invented ones resolve."""
    store, doc_id = ingested
    verdict = resolve_quote(
        store=store,
        workspace_id=WORKSPACE,
        doc_id=doc_id,
        page=1,
        quote="Q5 2024 | 9900000 | 100000 | 88.8",
    )
    assert not verdict.ok


def test_ingestion_reports_no_anchor_problems(
    ingested: tuple[LocalStore, str],
) -> None:
    """`verify_anchors` now checks table chunks, so a regression here surfaces
    at ingestion rather than when a user clicks a citation."""
    store, doc_id = ingested
    document = store.get_document(workspace_id=WORKSPACE, doc_id=doc_id)
    warnings = document.get("warnings") or []
    assert not [w for w in warnings if "span" in str(w)], warnings


def test_verify_anchors_catches_a_span_past_the_end_of_the_page() -> None:
    """The check that was absent. Without it the broken table anchors were
    invisible for as long as they existed."""
    from app.ingest.chunk import Chunk, verify_anchors
    from app.ingest.extract import ExtractionResult, PageText

    result = ExtractionResult(
        doc_id="doc_1",
        source_name="doc.csv",
        kind="csv",
        sha256="0" * 64,
        pages=[PageText(page=1, text="Region | Revenue\nNorth | 100")],
        tables=[],
    )
    bad = Chunk(
        chunk_id="doc_1:t1",
        doc_id="doc_1",
        page=1,
        text="Region | Revenue\nNorth | 100",
        char_start=0,
        char_end=9_999,
        token_count=8,
        para_idx=None,
        section="table:t1",
        bbox=None,
        kind="table",
        table_name="t1",
    )
    problems = verify_anchors([bad], result)
    assert problems
    assert "past the end" in problems[0]


# --------------------------------------------------------------------------- #
# The column separator is ours, not the document's
# --------------------------------------------------------------------------- #


def test_a_pipe_delimited_quote_resolves_against_a_space_separated_page() -> None:
    """The third appearance of one bug, and the general fix.

    `chunk_tables` renders a table as "Segment | Revenue" because that is how a
    model is shown a table; a PDF's own text layer has "Segment Revenue". So an
    agent quoting a table row faithfully scored 0.84 against the 0.92 floor,
    and every claim over a table in a real PDF was dropped for "the quoted text
    does not appear on this page" — when it did, and the pipes were ours.

    Measured on the gold corpus: anchor completeness went from five reported
    problems to 1.0 once `normalise` folded the separator.
    """
    from app.rag.citations import normalise

    assert normalise("Segment | Revenue") == normalise("Segment Revenue")
    assert normalise("Q1 2024 | 1,200,000") == normalise("Q1 2024 1,200,000")


def test_folding_the_separator_does_not_let_a_fabrication_through() -> None:
    """The check that makes the fold safe. The pipe carries no information from
    the document, so removing it compares the words and the figures and nothing
    else — the benchmark holds resolver rejection at 4/4."""
    from app.rag.citations import normalise

    assert normalise("Segment | Revenue") != normalise("Segment | Expenses")
    assert normalise("Q1 2024 | 1,200,000") != normalise("Q1 2024 | 9,900,000")
