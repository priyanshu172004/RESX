"""Chunking.

Chunks are the unit of retrieval, and every one carries a full citation anchor:
`(doc_id, page, para_idx, char_start, char_end, bbox, section)`. That anchor is
the reason `"Source: page 12, para 3"` can be verified rather than merely
asserted.

Two rules shape the splitter:

  * **Never cross a page boundary.** A chunk spanning pages 11 and 12 cannot
    cite either honestly.
  * **Never split a table row.** Half a row of figures is worse than no row,
    because it still looks like data.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from app.ingest.extract import ExtractionResult, PageText, TextBlock

DEFAULT_TARGET_TOKENS = 512
DEFAULT_OVERLAP_TOKENS = 64

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def estimate_tokens(text: str) -> int:
    """Approximate token count for sizing chunks only.

    This is a heuristic (~4 characters per token) and is deliberately *not*
    used for billing, budgets, or context-limit decisions — those must come
    from the provider's own `count_tokens` endpoint, because a wrong estimate
    there causes truncation rather than a slightly odd chunk size.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable span with everything needed to cite it."""

    chunk_id: str
    doc_id: str
    page: int
    text: str
    char_start: int
    char_end: int
    token_count: int
    para_idx: int | None = None
    section: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    kind: str = "prose"  # prose | table
    table_name: str | None = None

    @property
    def anchor(self) -> dict[str, Any]:
        """The citation anchor, as it is persisted and later resolved."""
        return {
            "doc_id": self.doc_id,
            "page": self.page,
            "para_idx": self.para_idx,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "section": self.section,
            "bbox": list(self.bbox) if self.bbox else None,
        }

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bbox"] = list(self.bbox) if self.bbox else None
        return data

    def citation_label(self) -> str:
        label = f"p.{self.page}"
        if self.para_idx is not None:
            label += f" ¶{self.para_idx}"
        return label


@dataclass(slots=True)
class ChunkingStats:
    chunks: int = 0
    prose_chunks: int = 0
    table_chunks: int = 0
    pages_covered: int = 0
    dropped_empty: int = 0
    anchors_complete: int = 0

    @property
    def anchor_completeness(self) -> float:
        """Share of chunks with a resolvable anchor. Anything below 1.0 is a bug.

        An unciteable chunk cannot back a claim, so this is tracked as a
        first-class ingestion metric rather than assumed.
        """
        return 1.0 if self.chunks == 0 else self.anchors_complete / self.chunks

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["anchor_completeness"] = self.anchor_completeness
        return d


def _split_long_block(block: TextBlock, target_tokens: int) -> list[tuple[str, int, int]]:
    """Split one oversized paragraph on sentence boundaries.

    Returns `(text, char_start, char_end)` triples with offsets relative to the
    page, so the anchor survives the split.
    """
    sentences = _SENTENCE_END.split(block.text)
    if len(sentences) == 1:
        return [(block.text, block.char_start, block.char_end)]

    pieces: list[tuple[str, int, int]] = []
    buffer: list[str] = []
    buffer_tokens = 0
    cursor = block.char_start

    for sentence in sentences:
        s_tokens = estimate_tokens(sentence)
        if buffer and buffer_tokens + s_tokens > target_tokens:
            joined = " ".join(buffer)
            pieces.append((joined, cursor, cursor + len(joined)))
            cursor += len(joined) + 1
            buffer, buffer_tokens = [], 0
        buffer.append(sentence)
        buffer_tokens += s_tokens

    if buffer:
        joined = " ".join(buffer)
        pieces.append((joined, cursor, min(cursor + len(joined), block.char_end)))
    return pieces


def chunk_page(
    page: PageText,
    doc_id: str,
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunk one page. Never emits a chunk that spans pages."""
    chunks: list[Chunk] = []
    if not page.blocks:
        return chunks

    current: list[
        tuple[str, int, int, int | None, tuple[float, float, float, float] | None]
    ] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        text = "\n\n".join(part[0] for part in current).strip()
        if not text:
            current, current_tokens = [], 0
            return
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}:p{page.page}:c{len(chunks)}",
                doc_id=doc_id,
                page=page.page,
                text=text,
                char_start=current[0][1],
                char_end=current[-1][2],
                token_count=estimate_tokens(text),
                para_idx=current[0][3],
                section=page.section,
                bbox=current[0][4],
                kind="prose",
            )
        )
        current, current_tokens = [], 0

    for block in page.blocks:
        block_tokens = estimate_tokens(block.text)

        # A single paragraph larger than the target is split on sentence
        # boundaries rather than mid-word.
        if block_tokens > target_tokens:
            flush()
            for text, start, end in _split_long_block(block, target_tokens):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}:p{page.page}:c{len(chunks)}",
                        doc_id=doc_id,
                        page=page.page,
                        text=text,
                        char_start=start,
                        char_end=end,
                        token_count=estimate_tokens(text),
                        para_idx=block.para_idx,
                        section=page.section,
                        bbox=block.bbox,
                        kind="prose",
                    )
                )
            continue

        if current and current_tokens + block_tokens > target_tokens:
            tail = current[-1] if overlap_tokens > 0 else None
            flush()
            # Carry the previous paragraph forward so a fact split across a
            # boundary is still retrievable from at least one chunk.
            if tail is not None and estimate_tokens(tail[0]) <= overlap_tokens * 2:
                current = [tail]
                current_tokens = estimate_tokens(tail[0])

        current.append(
            (block.text, block.char_start, block.char_end, block.para_idx, block.bbox)
        )
        current_tokens += block_tokens

    flush()
    return chunks


def chunk_tables(result: ExtractionResult) -> list[Chunk]:
    """One chunk per table, rows kept intact.

    Tables are chunked separately and marked `kind="table"` so retrieval can
    prefer them for numeric questions. Row integrity is absolute: a chunk
    boundary never falls inside a row.
    """
    chunks: list[Chunk] = []
    for table in result.tables:
        header = " | ".join(table.header)

        scale_note = (
            f"[scale: {table.scale_phrase} => x{table.scale_factor}]"
            if table.scale_phrase
            else ""
        )
        trust_note = (
            "[WARNING: table extractors disagreed; figures need confirmation]"
            if table.agreement is False
            else ""
        )

        # No separator rule between the header and the body.
        #
        # A line of dashes was rendered here to make the table readable, and it
        # was the single reason document analysis produced no findings. It
        # appears nowhere in the document, so an agent that quoted what it was
        # shown — the header, the rule, then the rows, which is the natural
        # thing to copy out of a table — produced a quote that could not
        # resolve against the page. Measured on the real path: a quote of any
        # actual rows scores 1.00, and the same quote with the rule included
        # scores 0.77-0.82 against a 0.92 floor. Every trend claim over a
        # table was dropped for "the quoted text does not appear on this page",
        # which was true, and our fault rather than the model's.
        #
        # Nothing is lost. The rule carried no information; the pipe-delimited
        # header already separates itself from the body.
        body_lines = [" | ".join(row) for row in table.rows]
        quotable = "\n".join(([header] if header else []) + body_lines).strip()

        notes = [n for n in (scale_note, trust_note) if n]
        text = "\n".join([*notes, quotable]).strip()

        # The anchor has to address the page, not the chunk. This was
        # `(0, len(text))` — offsets into the chunk's own string, which for a
        # 237-char rendering of a 186-char page pointed past the end of the
        # page entirely. `verify_anchors` skipped table chunks, so it never
        # caught it.
        page = result.page(table.page)
        char_start, char_end = 0, len(quotable)
        if page is not None:
            found = page.text.find(quotable)
            if found >= 0:
                char_start, char_end = found, found + len(quotable)
            else:
                # The body is not contiguous on the page (a table split across
                # columns, or a renderer that reflowed it). Address the whole
                # page rather than a span that is wrong: the resolver falls
                # back to a page-wide match and reports the offset as stale,
                # which is honest, where a bogus span reads as precision.
                char_start, char_end = 0, len(page.text)

        chunks.append(
            Chunk(
                chunk_id=f"{result.doc_id}:{table.name}",
                doc_id=result.doc_id,
                page=table.page,
                text=text,
                char_start=char_start,
                char_end=char_end,
                token_count=estimate_tokens(text),
                para_idx=None,
                section=f"table:{table.name}",
                bbox=table.bbox,
                kind="table",
                table_name=table.name,
            )
        )
    return chunks


def chunk_document(
    result: ExtractionResult,
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> tuple[list[Chunk], ChunkingStats]:
    """Chunk a whole document and report anchor completeness."""
    chunks: list[Chunk] = []
    stats = ChunkingStats()

    for page in result.pages:
        page_chunks = chunk_page(
            page,
            result.doc_id,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
        )
        if page_chunks:
            stats.pages_covered += 1
        chunks.extend(page_chunks)

    table_chunks = chunk_tables(result)
    chunks.extend(table_chunks)

    stats.chunks = len(chunks)
    stats.prose_chunks = sum(1 for c in chunks if c.kind == "prose")
    stats.table_chunks = len(table_chunks)
    stats.anchors_complete = sum(1 for c in chunks if _anchor_is_complete(c))
    return chunks, stats


def _anchor_is_complete(chunk: Chunk) -> bool:
    """Minimum viable anchor: a document, a page, and a non-empty span."""
    return (
        bool(chunk.doc_id)
        and chunk.page >= 1
        and chunk.char_end > chunk.char_start
        and bool(chunk.text.strip())
    )


#: A header this system generated because the document did not supply one.
#: Not document text, so it can never be found on the page.
_GENERATED_HEADER = re.compile(r"col_\d+(?:\s*\|\s*col_\d+)*")


def verify_anchors(chunks: Iterable[Chunk], result: ExtractionResult) -> list[str]:
    """Confirm every chunk's span actually resolves in the extracted page text.

    Run at ingestion time, so a broken anchor is caught while the document is
    being processed rather than at the moment a user clicks a citation.
    """
    # Imported here rather than at module scope: `app.rag.citations`
    # imports the store, and the store imports this module.
    from app.rag.citations import normalise

    problems: list[str] = []
    for chunk in chunks:
        page = result.page(chunk.page)
        if page is None:
            problems.append(f"{chunk.chunk_id}: page {chunk.page} not in extraction")
            continue

        # Table chunks used to be skipped here, which is exactly why their
        # anchors were wrong for so long: they were set to `(0, len(text))` —
        # offsets into the chunk's own rendering — and nothing ever compared
        # them to a page. The check is the same for both kinds now.
        if chunk.char_end > len(page.text):
            problems.append(
                f"{chunk.chunk_id}: span [{chunk.char_start}:{chunk.char_end}] runs "
                f"past the end of page {chunk.page} ({len(page.text)} chars)"
            )
            continue

        span = page.text[chunk.char_start : chunk.char_end]
        # Compared with the resolver's own `normalise`, not by byte equality.
        #
        # That is the point of using it here: the chunk joins paragraphs with a
        # blank line and renders table columns with " | ", neither of which is
        # in the document. Checking for byte equality reported a problem on
        # every table in every PDF — correctly, in the sense that the strings
        # differed, and uselessly, because the resolver does not compare them
        # that way either. Using the same function means this check now agrees
        # with the gate it is supposed to predict.
        #
        # Excluded from the comparison: our bracketed scale and trust notes,
        # and generated `col_0` headers for a table the document gave no header
        # to. Neither is document text, so neither can be found in the page.
        lines = [
            line
            for line in chunk.text.strip().splitlines()
            if line.strip()
            and not line.lstrip().startswith("[")
            and not _GENERATED_HEADER.fullmatch(line.strip())
        ]
        head = normalise(lines[0][:60]) if lines else ""
        if head and head not in normalise(span):
            problems.append(
                f"{chunk.chunk_id}: span [{chunk.char_start}:{chunk.char_end}] does not "
                f"contain the chunk's opening text"
            )
    return problems
