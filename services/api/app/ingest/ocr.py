"""OCR for pages that carry no text layer.

A scanned PDF is a stack of photographs. `pdfplumber` extracts nothing from it,
so before this existed such a page ingested as zero characters — the pipeline
recorded "likely a scan; needs OCR" and moved on, and every claim about that
page was correctly dropped as uncitable. The document appeared to ingest
successfully and then produced no analysis at all.

Three things here are deliberate.

**Confidence is recorded per block, not thrown away.** Tesseract reports a
per-word confidence and `TextBlock.is_low_confidence_ocr` has always existed to
act on it — it was simply never fed. A figure read off a bad scan at 43%
confidence must not back a financial claim with the same standing as one read
from a text layer, so the confidence travels with the block and the page is
flagged.

**Word boxes are kept.** They are what make a citation from a scanned page
point at a region of the image rather than at a character offset in text that
has no original. Without them an OCR'd page is quotable but not locatable.

**Absence of the binary is reported, never silently skipped.** `pytesseract` is
a thin wrapper around an executable that has to be installed separately, and
the failure mode of ignoring that is a scan that ingests as empty with no
explanation — which is the bug this module was written to fix, reappearing one
layer up.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

log = logging.getLogger(__name__)

#: Rendering resolution. Tesseract is trained around 300 DPI and degrades
#: noticeably below it; above it costs time and memory for little gain.
OCR_DPI = 300

#: Words below this confidence are dropped rather than guessed at. Tesseract
#: emits -1 for non-text regions and single digits for noise, and a page of
#: garbage words is worse than a page recorded as unreadable: the garbage looks
#: like evidence.
MIN_WORD_CONFIDENCE = 40.0

#: A page whose mean word confidence is under this is flagged for review. The
#: same 0.80 that `TextBlock.is_low_confidence_ocr` has always used.
LOW_CONFIDENCE_PAGE = 0.80

#: Vertical tolerance, in pixels at OCR_DPI, for treating two words as being on
#: the same line. Generous, because a scan is never quite straight.
LINE_TOLERANCE_PX = 12


class OcrUnavailableError(RuntimeError):
    """The Tesseract binary is not installed or not on PATH."""


@dataclass(frozen=True, slots=True)
class OcrWord:
    text: str
    confidence: float
    #: Pixel box at OCR_DPI: (x0, top, x1, bottom).
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class OcrLine:
    text: str
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class OcrPage:
    text: str
    lines: list[OcrLine]
    #: Mean word confidence, 0.0-1.0. `None` when nothing was read.
    confidence: float | None
    word_count: int
    #: The words of each line, kept because table reconstruction needs the
    #: per-word x positions and the joined line text has thrown them away.
    word_rows: list[list[OcrWord]] = field(default_factory=list)

    @property
    def is_low_confidence(self) -> bool:
        return self.confidence is not None and self.confidence < LOW_CONFIDENCE_PAGE


def tesseract_path() -> str | None:
    """Where the binary is, or None.

    Checked before use so the caller can report a specific, actionable problem
    instead of a `TesseractNotFoundError` traceback from inside a worker.
    """
    found = shutil.which("tesseract")
    if found:
        return found

    # The Windows installer does not add itself to PATH, which makes "install
    # Tesseract and it still does not work" the common experience. Look where
    # it actually lands.
    import os

    candidates = [
        # Upper-cased because `os.environ` on Windows normalises keys that
        # way; the variables themselves are `ProgramFiles` and
        # `ProgramFiles(x86)`.
        os.path.join(os.environ.get("PROGRAMFILES", ""), "Tesseract-OCR", "tesseract.exe"),
        os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "Tesseract-OCR", "tesseract.exe"),
        os.path.join(
            os.environ.get("LOCALAPPDATA", ""),
            "Programs",
            "Tesseract-OCR",
            "tesseract.exe",
        ),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def is_available() -> bool:
    return tesseract_path() is not None


def _require_tesseract() -> Any:
    try:
        import pytesseract
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise OcrUnavailableError(
            "the pytesseract package is not installed; run "
            "`pip install -e '.[dev]'` in services/api"
        ) from exc

    binary = tesseract_path()
    if binary is None:
        raise OcrUnavailableError(
            "Tesseract is not installed or not on PATH, so scanned pages "
            "cannot be read. Install it with "
            "`winget install UB-Mannheim.TesseractOCR` on Windows, "
            "`brew install tesseract` on macOS, or "
            "`apt install tesseract-ocr` on Debian/Ubuntu. "
            "If it is installed elsewhere, set TESSERACT_CMD to its full path."
        )
    pytesseract.pytesseract.tesseract_cmd = binary
    return pytesseract


def _group_into_lines(
    words: list[OcrWord],
) -> tuple[list[OcrLine], list[list[OcrWord]]]:
    """Reassemble words into reading order.

    Tesseract returns words with boxes, not lines. Sorting by y and grouping
    within a tolerance recovers the lines; the tolerance is generous because a
    scan is never perfectly square and a strict comparison splits one line into
    several.
    """
    if not words:
        return [], []

    ordered = sorted(words, key=lambda w: (w.bbox[1], w.bbox[0]))
    lines: list[list[OcrWord]] = [[ordered[0]]]
    for word in ordered[1:]:
        current = lines[-1]
        baseline = sum(w.bbox[1] for w in current) / len(current)
        if abs(word.bbox[1] - baseline) <= LINE_TOLERANCE_PX:
            current.append(word)
        else:
            lines.append([word])

    out: list[OcrLine] = []
    rows: list[list[OcrWord]] = []
    for group in lines:
        group.sort(key=lambda w: w.bbox[0])
        text = " ".join(w.text for w in group).strip()
        if not text:
            continue
        rows.append(group)
        out.append(
            OcrLine(
                text=text,
                confidence=sum(w.confidence for w in group) / len(group) / 100.0,
                bbox=(
                    min(w.bbox[0] for w in group),
                    min(w.bbox[1] for w in group),
                    max(w.bbox[2] for w in group),
                    max(w.bbox[3] for w in group),
                ),
            )
        )
    return out, rows


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def split_into_cells(words: list[OcrWord]) -> list[str]:
    """Break one line of words into table cells at the wide gaps.

    Gaps between words, not shared left edges, are what separate columns.

    Left-edge clustering was the first attempt and it fails on the commonest
    label in a financial table: "Q1 2024" is two words at two different x
    positions, so clustering made it two columns and shifted every figure on
    the row one place left. Gaps do not have that problem — the space inside
    "Q1 2024" is an ordinary word space, while the space before the next column
    is several times wider.

    The threshold is derived per row rather than fixed, because it has to work
    for a dense statement and a widely-spaced summary in the same document.
    """
    if not words:
        return []

    gaps = [max(0.0, nxt.bbox[0] - cur.bbox[2]) for cur, nxt in pairwise(words)]
    if not gaps:
        return [words[0].text]

    # The threshold comes from the text's own size, not from the gaps.
    #
    # Using a multiple of the median gap looks reasonable and cannot work: in a
    # table row *most* gaps are column gaps, so the median already is one and
    # nothing ever exceeds a multiple of it. Every row collapsed into a single
    # cell, and the reconstruction silently declined on every document.
    #
    # Typography gives a stable unit instead. A word space is roughly 0.3 em
    # and the glyph height is roughly 1 em, while columns in a table are set at
    # least one em apart — so a gap wider than ~1.4x the height is a column
    # break and anything narrower is a space inside a cell. That is what keeps
    # "Q1 2024" as one label.
    heights = [w.bbox[3] - w.bbox[1] for w in words]
    threshold = max(_median(heights) * 1.4, 18.0)

    cells: list[list[str]] = [[words[0].text]]
    # `gaps` has exactly one entry per adjacent pair, so it is one shorter
    # than `words`; strict= would be wrong here rather than safer.
    for gap, word in zip(gaps, words[1:], strict=True):
        if gap > threshold:
            cells.append([word.text])
        else:
            cells[-1].append(word.text)
    return [" ".join(cell).strip() for cell in cells]


def _numeric_fraction(cells: list[str]) -> float:
    from app.core.money import parse_amount

    filled = [c for c in cells if c]
    if not filled:
        return 0.0
    return sum(1 for c in filled if parse_amount(c) is not None) / len(filled)


def table_from_lines(
    lines: list[OcrLine], words_by_line: list[list[OcrWord]]
) -> tuple[list[str], list[list[str]]] | None:
    """Reconstruct a table from OCR geometry, or decline.

    Why this exists: OCR gives us the text of a scanned financial statement as
    prose, and this system analyses *tables*. A scanned page with no table
    reaches the agents as a paragraph of numbers they are forbidden to do
    arithmetic on — so it yields no computation, no dataset and no chart. The
    document reads successfully and analyses to nothing, which is the same end
    state as not reading it at all.

    Deliberately conservative, and it returns None on any doubt. A
    wrongly-shaped table is worse than no table: figures would land under the
    wrong heading and every computation over them would be confidently wrong,
    with a citation to prove it. The bar is two or more rows that are mostly
    numeric *and* agree on how many cells they have — prose does not do that,
    because prose split by wide gaps produces a different count every line.
    """
    if len(words_by_line) < 2:
        return None

    rows = [(index, split_into_cells(words)) for index, words in enumerate(words_by_line)]

    # A data row is mostly numbers. This is what excludes the surrounding
    # prose, which is the reason an earlier version inferred twenty-one
    # columns from a page holding four.
    numeric_rows = [
        (index, cells)
        for index, cells in rows
        if len(cells) >= 2 and _numeric_fraction(cells) >= 0.5
    ]
    if len(numeric_rows) < 2:
        return None

    # And they have to agree on their arity. A table is regular; a sentence
    # that happens to contain two figures is not.
    counts: dict[int, int] = {}
    for _, cells in numeric_rows:
        counts[len(cells)] = counts.get(len(cells), 0) + 1
    width, agreeing = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    if agreeing < 2:
        return None

    body_rows = [(i, c) for i, c in numeric_rows if len(c) == width]
    first_data_line = body_rows[0][0]

    # The header is the nearest line above the data that is not itself data.
    header: list[str] = []
    for _, cells in reversed(rows[:first_data_line]):
        if _numeric_fraction(cells) >= 0.5:
            continue
        if len(cells) == width:
            header = cells
            break
        # A title spanning the page is one cell and is not a header; keep
        # looking upward rather than accepting it.
        if len(cells) > 1:
            # Close enough to align by position: the header of a scanned table
            # is often set in a different size and splits a cell in two.
            if abs(len(cells) - width) <= 1:
                header = (cells + [""] * width)[:width]
            break

    if not header:
        header = [f"col_{i}" for i in range(width)]
    header = [h.strip() or f"col_{i}" for i, h in enumerate(header)]

    return header, [cells for _, cells in body_rows]


def ocr_image(image: Any, *, language: str = "eng") -> OcrPage:
    """Read one page image.

    Uses `image_to_data` rather than `image_to_string` because the former
    returns confidences and boxes. The plain string call throws both away, and
    they are the difference between "we read something" and "we read something
    and here is how sure we are and where it was".
    """
    pytesseract = _require_tesseract()
    from pytesseract import Output

    data = pytesseract.image_to_data(image, lang=language, output_type=Output.DICT)

    words: list[OcrWord] = []
    for i, raw in enumerate(data.get("text") or []):
        text = str(raw).strip()
        if not text:
            continue
        try:
            confidence = float(data["conf"][i])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        # -1 marks a region Tesseract found no text in.
        if confidence < MIN_WORD_CONFIDENCE:
            continue
        words.append(
            OcrWord(
                text=text,
                confidence=confidence,
                bbox=(
                    float(data["left"][i]),
                    float(data["top"][i]),
                    float(data["left"][i] + data["width"][i]),
                    float(data["top"][i] + data["height"][i]),
                ),
            )
        )

    lines, rows = _group_into_lines(words)
    mean = sum(w.confidence for w in words) / len(words) / 100.0 if words else None
    return OcrPage(
        text=render_page_text(rows),
        lines=lines,
        confidence=mean,
        word_count=len(words),
        word_rows=rows,
    )


def render_page_text(word_rows: list[list[OcrWord]]) -> str:
    """The page text, with tabular lines column-delimited.

    A line that splits into two or more cells is rendered with " | " between
    them, and a line that does not is left as prose.

    This matters because of what the resolver compares. The table chunk built
    from this page is pipe-delimited — that is how a model is shown a table —
    and citations resolve against the *page*. Rendering the page with plain
    spaces meant a quote copied faithfully out of the chunk scored 0.84 against
    the 0.92 floor and every claim over a scanned table was dropped: the same
    failure as the invented `-----` separator in `chunk_tables`, one layer
    down.

    There is no fidelity cost. The original here is pixels, not characters, so
    the page text is our rendering either way; choosing the one that matches
    the chunk is free, and it also makes a scanned table read like a table.
    """
    out: list[str] = []
    for row in word_rows:
        cells = split_into_cells(row)
        out.append(" | ".join(cells) if len(cells) > 1 else " ".join(w.text for w in row))
    return "\n".join(line for line in out if line.strip())


def render_pdf_page(pdf_path: str, page_index: int, *, dpi: int = OCR_DPI) -> Any:
    """Rasterise one PDF page to a PIL image.

    `pypdfium2` rather than `pdf2image`, deliberately: pdf2image shells out to
    poppler, which is another system binary to install and another way for this
    to fail on a fresh machine. pypdfium2 ships its own renderer as a wheel and
    is already a dependency.
    """
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(pdf_path)
    try:
        page = document[page_index]
        # pypdfium2 takes a scale relative to 72 DPI.
        bitmap = page.render(scale=dpi / 72.0)
        return bitmap.to_pil()
    finally:
        document.close()


def ocr_pdf_page(
    pdf_path: str, page_index: int, *, language: str = "eng", dpi: int = OCR_DPI
) -> OcrPage:
    """Render and read one page of a PDF."""
    image = render_pdf_page(pdf_path, page_index, dpi=dpi)
    return ocr_image(image, language=language)


def ocr_image_file(path: str, *, language: str = "eng") -> OcrPage:
    """Read a standalone image file — a photographed page, a screenshot."""
    from PIL import Image

    with Image.open(path) as opened:
        # Some scanners save 1-bit or CMYK; Tesseract wants RGB or L.
        # `convert` returns a new image rather than mutating, so it gets its
        # own name -- rebinding the context variable changes its type.
        image = opened.convert("RGB") if opened.mode not in {"L", "RGB"} else opened
        return ocr_image(image, language=language)
