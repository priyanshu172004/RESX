"""Extraction: documents in, page text plus typed tables out.

The single most important output of this module is not the text — it is the
**anchor**. Every piece of extracted text carries `(page, char_span, bbox)` so
a claim built on it can be cited and the citation can later be resolved back to
the exact span. A chunk without a resolvable anchor is unusable, because it
cannot support a claim (see `docs/04-ACCURACY-VALIDATION.md` §1).

Tables take a second path. Financial figures are never read out of prose if a
table is available: they become typed rows that the sandbox can sum exactly.
That distinction is what separates a demo from a product.
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from app.core.money import detect_currency, detect_scale

log = logging.getLogger(__name__)

SourceKind = Literal["pdf", "csv", "tsv", "xlsx", "docx", "txt", "json"]

# Extensions are a hint for dispatch only. The security-relevant type decision
# is the magic-byte check at upload time (docs/05-SECURITY.md §4.7); this module
# runs after a file has already been accepted.
_EXTENSION_KIND: dict[str, SourceKind] = {
    ".pdf": "pdf",
    ".csv": "csv",
    ".tsv": "tsv",
    ".xlsx": "xlsx",
    ".xlsm": "xlsx",
    ".xls": "xlsx",
    ".docx": "docx",
    ".txt": "txt",
    ".md": "txt",
    ".json": "json",
}


class ExtractionError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Data carriers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TextBlock:
    """A paragraph-sized run of text with its position on the page."""

    text: str
    para_idx: int
    char_start: int
    char_end: int
    bbox: tuple[float, float, float, float] | None = None
    ocr_confidence: float | None = None

    @property
    def is_low_confidence_ocr(self) -> bool:
        # 0.80 is the threshold below which a block is flagged for review
        # rather than silently used as evidence.
        return self.ocr_confidence is not None and self.ocr_confidence < 0.80


@dataclass(frozen=True, slots=True)
class PageText:
    page: int
    text: str
    blocks: list[TextBlock] = field(default_factory=list)
    section: str | None = None


@dataclass(frozen=True, slots=True)
class ExtractedTable:
    """A table lifted out of a document, plus what we know about its trust."""

    name: str
    page: int
    header: list[str]
    rows: list[list[str]]
    extractor: str
    #: Did two independent extractors agree on this table? `None` means only
    #: one strategy produced it, so there is nothing to compare against.
    agreement: bool | None = None
    scale_factor: str = "1"
    scale_phrase: str | None = None
    currency: str | None = None
    bbox: tuple[float, float, float, float] | None = None

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return len(self.header) if self.header else (len(self.rows[0]) if self.rows else 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "page": self.page,
            "header": self.header,
            "rows": self.rows,
            "extractor": self.extractor,
            "agreement": self.agreement,
            "scale_factor": self.scale_factor,
            "scale_phrase": self.scale_phrase,
            "currency": self.currency,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
        }


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    doc_id: str
    source_name: str
    kind: SourceKind
    sha256: str
    pages: list[PageText] = field(default_factory=list)
    tables: list[ExtractedTable] = field(default_factory=list)
    low_confidence_pages: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)

    def page(self, number: int) -> PageText | None:
        for p in self.pages:
            if p.page == number:
                return p
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source_name": self.source_name,
            "kind": self.kind,
            "sha256": self.sha256,
            "page_count": self.page_count,
            "char_count": self.char_count,
            "tables": [t.to_dict() for t in self.tables],
            "low_confidence_pages": self.low_confidence_pages,
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_PARA_SPLIT = re.compile(r"\n\s*\n+")
_HEADING = re.compile(r"^(?:[A-Z][A-Z \-&/]{4,}|\d+(?:\.\d+)*\s+[A-Z].{3,60})$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def detect_kind(path: Path) -> SourceKind:
    kind = _EXTENSION_KIND.get(path.suffix.lower())
    if kind is None:
        raise ExtractionError(f"unsupported file type: {path.suffix!r}")
    return kind


def _blocks_from_text(text: str) -> list[TextBlock]:
    """Split page text into blocks, recording exact char offsets.

    Offsets are computed against the page text as stored, so a citation's
    `char_span` indexes straight into it with no re-derivation.

    Blank lines are the preferred separator, but many PDF text layers contain
    none at all — the whole page arrives as one run of single-newline lines. In
    that case blank-line splitting yields a single block covering the entire
    page, which destroys citation granularity: every claim on the page would
    resolve to the same anchor. So when that happens we fall back to
    line-level blocks, which is what actually makes "page 2, line 4" citable.
    """
    stripped = text.strip()
    if not stripped:
        return []

    paragraphs = [p for p in (raw.strip() for raw in _PARA_SPLIT.split(text)) if p]

    if len(paragraphs) < 2 and len(stripped.splitlines()) >= 3:
        units = [line.strip() for line in text.splitlines()]
    else:
        units = paragraphs

    blocks: list[TextBlock] = []
    cursor = 0
    for unit in units:
        if not unit:
            continue
        start = text.find(unit, cursor)
        if start == -1:
            start = cursor
        end = start + len(unit)
        cursor = end
        blocks.append(
            TextBlock(text=unit, para_idx=len(blocks), char_start=start, char_end=end)
        )
    return blocks


def _detect_section(blocks: list[TextBlock]) -> str | None:
    for block in blocks[:4]:
        line = block.text.strip().splitlines()[0] if block.text.strip() else ""
        if _HEADING.match(line):
            return line.title() if line.isupper() else line
    return None


def _normalise_cell(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _clean_table(raw_rows: list[list[Any]]) -> tuple[list[str], list[list[str]]]:
    rows = [[_normalise_cell(c) for c in row] for row in raw_rows if row is not None]
    rows = [r for r in rows if any(cell for cell in r)]
    if not rows:
        return [], []

    header, body = rows[0], rows[1:]
    # A header of mostly-empty cells is not a header; keep every row as data
    # rather than silently discarding the first record.
    if sum(1 for c in header if c) < max(2, len(header) // 2):
        return [f"col_{i}" for i in range(len(header))], rows
    return header, body


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #

#: Mean OCR confidence below which a page is flagged rather than trusted. The
#: same 0.80 that `TextBlock.is_low_confidence_ocr` has always compared
#: against — it just never had anything to compare.
OCR_LOW_CONFIDENCE = 0.80


def _ocr_table(
    page_result: Any, doc_id: str, page_no: int, page_context: str
) -> ExtractedTable | None:
    """A table reconstructed from OCR geometry, if the page really has one.

    Without this a scanned financial statement reaches the agents as a
    paragraph of numbers they are forbidden to do arithmetic on: no dataset, no
    computation, no chart. The document reads successfully and analyses to
    nothing, which is the same end state as not reading it.

    `agreement=None` is correct and important here: only one strategy produced
    this table, so there is nothing to compare it against, and the profile
    already treats that as "confirm before backing a financial claim". An OCR
    table is a lead, not a ledger.
    """
    from app.ingest.ocr import table_from_lines

    built = table_from_lines(page_result.lines, page_result.word_rows)
    if built is None:
        return None

    header, rows = built
    scale, phrase = detect_scale(page_context)
    return ExtractedTable(
        name=f"ocr_p{page_no}_1",
        page=page_no,
        header=header,
        rows=rows,
        extractor="ocr",
        agreement=None,
        scale_factor=str(scale),
        scale_phrase=phrase,
        currency=detect_currency(page_context),
    )


def _ocr_pdf_page(
    path: Path, page_no: int, width: float, height: float
) -> tuple[str, list[TextBlock], str, float | None, Any]:
    """Read a scanned page, returning text, blocks, a note, and confidence.

    Returns rather than raises on every failure. A page that cannot be OCR'd
    must not abort ingestion of a 200-page document: the page is recorded as
    unreadable, the reason is attached to the document as a warning, and the
    rest of the file still ingests.

    Boxes are converted from OCR pixels back to PDF points, so a citation from
    a scanned page addresses the same coordinate space as one from a text
    layer. Without that conversion the UI would point at the wrong part of the
    page, which is worse than pointing nowhere.
    """
    from app.ingest.ocr import OCR_DPI, OcrUnavailableError, ocr_pdf_page

    try:
        result = ocr_pdf_page(str(path), page_no - 1)
    except OcrUnavailableError as exc:
        # Actionable, and said once per document rather than swallowed.
        return "", [], str(exc), None, None
    except Exception as exc:
        log.warning("OCR failed on page %d of %s: %s", page_no, path.name, exc)
        return "", [], f"OCR failed ({type(exc).__name__}: {exc})", None, None

    if not result.lines:
        return (
            "",
            [],
            "no text found by OCR either; the page may be blank or an image "
            "with no legible text",
            None,
            None,
        )

    scale = 72.0 / OCR_DPI
    blocks: list[TextBlock] = []
    cursor = 0
    # Block text comes from the *rendered* page, not from `line.text`: the page
    # renders a tabular line as "a | b | c" and a block holding "a b c" would
    # carry char offsets that address the wrong characters.
    rendered = result.text.splitlines()
    for index, line in enumerate(result.lines):
        text_line = rendered[index] if index < len(rendered) else line.text
        blocks.append(
            TextBlock(
                text=text_line,
                para_idx=index,
                char_start=cursor,
                char_end=cursor + len(text_line),
                bbox=(
                    line.bbox[0] * scale,
                    line.bbox[1] * scale,
                    line.bbox[2] * scale,
                    line.bbox[3] * scale,
                ),
                # The whole point: the confidence travels with the block.
                ocr_confidence=line.confidence,
            )
        )
        cursor += len(text_line) + 1  # the newline joining the lines below

    text = result.text
    pct = f"{result.confidence * 100:.0f}%" if result.confidence is not None else "?"
    note = (
        f"read by OCR ({result.word_count} words, mean confidence {pct}); "
        "figures from this page should be confirmed against the original"
        if result.is_low_confidence
        else f"read by OCR ({result.word_count} words, mean confidence {pct})"
    )
    return text, blocks, note, result.confidence, result


def _extract_pdf(
    path: Path, doc_id: str
) -> tuple[list[PageText], list[ExtractedTable], list[str], list[int]]:
    import pdfplumber

    pages: list[PageText] = []
    tables: list[ExtractedTable] = []
    warnings: list[str] = []
    low_conf: list[int] = []

    with pdfplumber.open(str(path)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            blocks = _blocks_from_text(text)

            # Attach bounding boxes by matching each block's first line against
            # the page's extracted lines. The bbox is what lets the UI point at
            # the exact region of the original document.
            try:
                lines = page.extract_text_lines()
            except Exception:
                lines = []

            if lines:
                positioned: list[TextBlock] = []
                for block in blocks:
                    first = block.text.splitlines()[0].strip()
                    bbox = None
                    for line in lines:
                        if first and first[:40] in line.get("text", ""):
                            bbox = (
                                float(line["x0"]),
                                float(line["top"]),
                                float(line["x1"]),
                                float(line["bottom"]),
                            )
                            break
                    positioned.append(
                        TextBlock(
                            text=block.text,
                            para_idx=block.para_idx,
                            char_start=block.char_start,
                            char_end=block.char_end,
                            bbox=bbox,
                        )
                    )
                blocks = positioned

            if not text.strip():
                # No text layer: almost always a scan. Read it with OCR rather
                # than recording an empty page.
                #
                # This used to stop at the warning below, and the consequence
                # was worse than an error: the document ingested "successfully"
                # with zero characters on the scanned pages, so every claim
                # about them was correctly dropped as uncitable and the user
                # got a completed run with no findings and no explanation.
                text, blocks, note, confidence, ocr_result = _ocr_pdf_page(
                    path, page_no, page.width, page.height
                )
                if note:
                    warnings.append(f"page {page_no}: {note}")
                if not text.strip():
                    low_conf.append(page_no)
                elif confidence is not None and confidence < OCR_LOW_CONFIDENCE:
                    # Read, but not well. Flagged so a figure off a poor scan
                    # does not back a claim with the same standing as one from
                    # a text layer -- which is what `is_low_confidence_ocr`
                    # has always been for.
                    low_conf.append(page_no)

                # A scanned table has to be rebuilt from geometry or it stays
                # prose, and prose is exactly what the agents may not compute
                # over. See `_ocr_table`.
                if ocr_result is not None:
                    ocr_table = _ocr_table(ocr_result, doc_id, page_no, text[:600])
                    if ocr_table is not None:
                        tables.append(ocr_table)
                        warnings.append(
                            f"page {page_no}: a table was reconstructed from the "
                            f"scan ({ocr_table.n_rows} rows x "
                            f"{ocr_table.n_cols} cols); no second extractor "
                            f"could confirm it, so confirm its figures before "
                            f"acting on them"
                        )

            pages.append(
                PageText(
                    page=page_no,
                    text=text,
                    blocks=blocks,
                    section=_detect_section(blocks),
                )
            )

            # --- dual table extraction ---
            # Two strategies, then compared. A table that only one strategy
            # finds, or that they disagree on, is flagged — and a flagged table
            # must not silently back a financial claim.
            lattice = _safe_tables(
                page, {"vertical_strategy": "lines", "horizontal_strategy": "lines"}
            )
            stream = _safe_tables(
                page, {"vertical_strategy": "text", "horizontal_strategy": "text"}
            )

            page_context = f"{pages[-1].section or ''}\n{text[:600]}"
            scale, phrase = detect_scale(page_context)
            currency = detect_currency(page_context)

            chosen = lattice or stream
            for t_idx, raw in enumerate(chosen):
                header, body = _clean_table(raw)
                if not body:
                    continue
                counterpart = stream if chosen is lattice else lattice
                agreement: bool | None = None
                if counterpart:
                    agreement = _tables_agree(raw, counterpart)
                    if not agreement:
                        missing = sorted(
                            set(_numeric_signature(raw))
                            - {n for other in counterpart for n in _numeric_signature(other)}
                        )[:6]
                        warnings.append(
                            f"page {page_no} table {t_idx}: extractors disagree on "
                            f"figure(s) {missing} — this table requires confirmation "
                            "before it can back a financial claim"
                        )
                tables.append(
                    ExtractedTable(
                        name=f"p{page_no}_t{t_idx}",
                        page=page_no,
                        header=header,
                        rows=body,
                        extractor="pdfplumber:lines"
                        if chosen is lattice
                        else "pdfplumber:text",
                        agreement=agreement,
                        scale_factor=str(scale),
                        scale_phrase=phrase,
                        currency=currency,
                    )
                )

    return pages, tables, warnings, low_conf


def _safe_tables(page: Any, settings: dict[str, Any]) -> list[list[list[Any]]]:
    try:
        return page.extract_tables(table_settings=settings) or []
    except Exception:
        return []


def _tables_agree(candidate: list[list[Any]], others: list[list[list[Any]]]) -> bool:
    """Do the two strategies agree on the numeric content of this table?

    Agreement is **containment, not equality**. The two strategies routinely
    disagree about where a table starts and ends, how many columns it has, and
    whether a caption is part of it — none of which affects a figure. Demanding
    identical signatures flagged every single table, and a warning on
    everything is the same as no warning at all.

    What we actually care about is a *figure* appearing in one extraction and
    not the other. So: agreement means every number the chosen table reports is
    also seen by the counterpart strategy somewhere on the same page.
    """
    want = set(_numeric_signature(candidate))
    if not want:
        return True
    seen: set[str] = set()
    for other in others:
        seen.update(_numeric_signature(other))
    return want.issubset(seen)


_NUM_IN_CELL = re.compile(r"-?\d[\d,]*\.?\d*")


def _numeric_signature(rows: list[list[Any]]) -> tuple[str, ...]:
    found: list[str] = []
    for row in rows or []:
        for cell in row or []:
            for match in _NUM_IN_CELL.findall(_normalise_cell(cell)):
                found.append(match.replace(",", ""))
    return tuple(sorted(found))


# --------------------------------------------------------------------------- #
# Spreadsheets and delimited text
# --------------------------------------------------------------------------- #


def _extract_xlsx(path: Path) -> tuple[list[PageText], list[ExtractedTable]]:
    from openpyxl import load_workbook

    # data_only=True gives computed values; the formulas are read separately so
    # a claim can cite the formula that produced a figure.
    values_wb = load_workbook(str(path), data_only=True, read_only=True)
    pages: list[PageText] = []
    tables: list[ExtractedTable] = []

    for sheet_no, name in enumerate(values_wb.sheetnames, start=1):
        sheet = values_wb[name]
        raw = [list(row) for row in sheet.iter_rows(values_only=True)]
        header, body = _clean_table(raw)

        preview = "\n".join(" | ".join(_normalise_cell(c) for c in row) for row in raw[:40])
        text = f"{name}\n\n{preview}"
        blocks = _blocks_from_text(text)
        pages.append(PageText(page=sheet_no, text=text, blocks=blocks, section=name))

        if body:
            scale, phrase = detect_scale(text[:600])
            tables.append(
                ExtractedTable(
                    name=f"sheet_{_slug(name)}",
                    page=sheet_no,
                    header=header,
                    rows=body,
                    extractor="openpyxl",
                    agreement=None,
                    scale_factor=str(scale),
                    scale_phrase=phrase,
                    currency=detect_currency(text[:600]),
                )
            )

    values_wb.close()
    return pages, tables


def _extract_delimited(
    path: Path, delimiter: str | None
) -> tuple[list[PageText], list[ExtractedTable]]:
    raw_bytes = path.read_bytes()
    text = _decode(raw_bytes)

    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","

    rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    header, body = _clean_table([list(r) for r in rows])

    preview = "\n".join(" | ".join(r) for r in rows[:40])
    blocks = _blocks_from_text(preview)
    pages = [PageText(page=1, text=preview, blocks=blocks, section=path.stem)]

    tables: list[ExtractedTable] = []
    if body:
        scale, phrase = detect_scale(preview[:600])
        tables.append(
            ExtractedTable(
                name=f"csv_{_slug(path.stem)}",
                page=1,
                header=header,
                rows=body,
                extractor="csv",
                scale_factor=str(scale),
                scale_phrase=phrase,
                currency=detect_currency(preview[:600]),
            )
        )
    return pages, tables


def _extract_docx(path: Path) -> tuple[list[PageText], list[ExtractedTable]]:
    from docx import Document

    document = Document(str(path))
    paragraphs = [p.text.strip() for p in document.paragraphs]
    text = "\n\n".join(p for p in paragraphs if p)
    blocks = _blocks_from_text(text)

    # DOCX has no intrinsic page concept, so everything is page 1 and the
    # paragraph index carries the position. That keeps the anchor honest
    # instead of inventing page numbers.
    pages = [PageText(page=1, text=text, blocks=blocks, section=_detect_section(blocks))]

    tables: list[ExtractedTable] = []
    for t_idx, table in enumerate(document.tables):
        raw = [[cell.text for cell in row.cells] for row in table.rows]
        header, body = _clean_table(raw)
        if body:
            tables.append(
                ExtractedTable(
                    name=f"docx_t{t_idx}",
                    page=1,
                    header=header,
                    rows=body,
                    extractor="python-docx",
                )
            )
    return pages, tables


def _extract_txt(path: Path) -> list[PageText]:
    text = _decode(path.read_bytes())
    # Paginate long plain text so citations stay usefully specific rather than
    # pointing at "page 1 of a 400-page file".
    pages: list[PageText] = []
    chunk_chars = 3000
    if len(text) <= chunk_chars:
        blocks = _blocks_from_text(text)
        return [PageText(page=1, text=text, blocks=blocks, section=_detect_section(blocks))]

    for page_no, start in enumerate(range(0, len(text), chunk_chars), start=1):
        segment = text[start : start + chunk_chars]
        blocks = _blocks_from_text(segment)
        pages.append(
            PageText(page=page_no, text=segment, blocks=blocks, section=_detect_section(blocks))
        )
    return pages


def _decode(raw: bytes) -> str:
    try:
        from charset_normalizer import from_bytes

        best = from_bytes(raw).best()
        if best is not None:
            return str(best)
    except Exception as exc:
        # Logged rather than swallowed: a systematically failing detector shows
        # up as mojibake in every document, and silence here is why that kind
        # of problem gets diagnosed by reading extracted text instead of a log.
        log.debug("charset detection failed, falling back to utf-8: %s", exc)
    return raw.decode("utf-8", errors="replace")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "table"


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def extract(path: str | Path, *, doc_id: str | None = None) -> ExtractionResult:
    """Extract one document into pages, blocks, and typed tables."""
    p = Path(path)
    if not p.exists():
        raise ExtractionError(f"no such file: {p}")

    kind = detect_kind(p)
    digest = sha256_file(p)
    resolved_id = doc_id or f"doc_{digest[:12]}"

    warnings: list[str] = []
    low_conf: list[int] = []
    tables: list[ExtractedTable] = []

    if kind == "pdf":
        pages, tables, warnings, low_conf = _extract_pdf(p, resolved_id)
    elif kind == "xlsx":
        pages, tables = _extract_xlsx(p)
    elif kind in {"csv", "tsv"}:
        pages, tables = _extract_delimited(p, "\t" if kind == "tsv" else None)
    elif kind == "docx":
        pages, tables = _extract_docx(p)
    else:
        pages = _extract_txt(p)

    if not pages:
        warnings.append("no pages extracted")

    return ExtractionResult(
        doc_id=resolved_id,
        source_name=p.name,
        kind=kind,
        sha256=digest,
        pages=pages,
        tables=tables,
        low_confidence_pages=low_conf,
        warnings=warnings,
    )


def iter_blocks(result: ExtractionResult) -> Iterator[tuple[int, TextBlock]]:
    for page in result.pages:
        for block in page.blocks:
            yield page.page, block
