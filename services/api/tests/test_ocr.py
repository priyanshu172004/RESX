"""OCR: reading pages that have no text layer.

A scanned PDF is a stack of photographs. Before this path existed such a page
ingested as zero characters — the pipeline noted "likely a scan; needs OCR" and
carried on, so the document appeared to ingest successfully and then analysed
to nothing, because every claim about an empty page is correctly dropped as
uncitable.

Three properties are locked down here, and two of them are regressions from
mistakes made building it:

* **The text is read, with its confidence and its boxes.** A figure off a poor
  scan must not back a claim with the same standing as one off a text layer.

* **A scanned table is reconstructed.** This system analyses tables; a scanned
  statement arriving as prose yields no dataset, no computation and no chart,
  which is the same end state as not reading it. The first attempt inferred
  columns from shared left edges and split "Q1 2024" into two columns, shifting
  every figure one place; the second used a multiple of the median inter-word
  gap, which cannot work because in a table row most gaps *are* column gaps.

* **A quote copied from the chunk resolves against the page.** The page text is
  rendered with the same " | " separators the table chunk uses. With plain
  spaces, a faithful quote scored 0.84 against the 0.92 floor and every claim
  over a scanned table was dropped — the same failure as the invented `-----`
  separator in `chunk_tables`, one layer down.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from app.ingest.extract import extract
from app.ingest.ocr import (
    OcrWord,
    is_available,
    render_page_text,
    split_into_cells,
    table_from_lines,
)
from app.ingest.pipeline import ingest_file
from app.rag.citations import resolve_quote
from app.store.local import LocalStore

WORKSPACE = "ws_ocr"

needs_tesseract = pytest.mark.skipif(
    not is_available(),
    reason="the Tesseract binary is not installed on this machine",
)


# --------------------------------------------------------------------------- #
# The geometry, which needs no binary
# --------------------------------------------------------------------------- #


def word(text: str, x0: float, x1: float, top: float = 100.0, height: float = 50.0):
    return OcrWord(text=text, confidence=95.0, bbox=(x0, top, x1, top + height))


def test_a_two_word_label_stays_one_cell() -> None:
    """ "Q1 2024" is the commonest label in a financial table, and splitting it
    shifts every figure on the row one column left."""
    row = [
        word("Q1", 212, 252),
        word("2024", 293, 378),
        word("1,200,000", 492, 700),
        word("780,000", 850, 1000),
    ]
    assert split_into_cells(row) == ["Q1 2024", "1,200,000", "780,000"]


def test_prose_is_a_single_cell() -> None:
    """Ordinary word spacing must never read as columns, or every sentence
    becomes a table row."""
    row = [
        word("Revenue", 212, 340),
        word("for", 355, 400),
        word("the", 415, 460),
        word("year", 475, 545),
        word("was", 560, 620),
    ]
    assert len(split_into_cells(row)) == 1


def test_the_threshold_does_not_come_from_the_median_gap() -> None:
    """The second attempt at this. In a table row most gaps are column gaps, so
    the median already is one and a multiple of it exceeds every gap — every
    row collapsed to a single cell and reconstruction silently never fired."""
    row = [
        word("Q1", 212, 252),
        word("2024", 293, 378),
        word("1,200,000", 492, 700),
        word("780,000", 850, 1000),
        word("35.0%", 1197, 1320),
    ]
    assert len(split_into_cells(row)) == 4


def test_a_table_is_reconstructed_from_aligned_numeric_rows() -> None:
    rows = [
        [word("ACME", 212, 400, top=60)],
        [
            word("Quarter", 212, 380, top=120),
            word("Revenue", 493, 660, top=120),
            word("Cost", 778, 880, top=120),
        ],
        [
            word("Q1", 212, 252, top=180),
            word("2024", 293, 378, top=180),
            word("1,200,000", 492, 700, top=180),
            word("780,000", 850, 1000, top=180),
        ],
        [
            word("Q2", 212, 252, top=240),
            word("2024", 293, 378, top=240),
            word("1,350,000", 492, 700, top=240),
            word("850,000", 850, 1000, top=240),
        ],
    ]
    built = table_from_lines([], rows)
    assert built is not None
    header, body = built
    assert header == ["Quarter", "Revenue", "Cost"]
    assert body == [
        ["Q1 2024", "1,200,000", "780,000"],
        ["Q2 2024", "1,350,000", "850,000"],
    ]


def test_prose_alone_produces_no_table() -> None:
    """A wrongly-shaped table is worse than none: figures land under the wrong
    heading and every computation over them is confidently wrong, with a
    citation to prove it."""
    rows = [
        [
            word("Revenue", 212, 340, top=t),
            word("rose", 355, 420, top=t),
            word("by", 435, 470, top=t),
            word("19", 485, 520, top=t),
            word("percent", 535, 660, top=t),
        ]
        for t in (100, 160, 220)
    ]
    assert table_from_lines([], rows) is None


def test_one_numeric_row_is_not_a_table() -> None:
    """A sentence that happens to contain two figures is not tabular."""
    rows = [
        [word("Total", 212, 320), word("5,250,000", 492, 700)],
    ]
    assert table_from_lines([], rows) is None


def test_the_page_text_renders_tabular_lines_with_the_table_separator() -> None:
    """So a quote copied out of the table chunk resolves against the page."""
    rows = [
        [word("Header", 212, 380, top=60)],
        [
            word("Q1", 212, 252, top=120),
            word("2024", 293, 378, top=120),
            word("1,200,000", 492, 700, top=120),
        ],
    ]
    text = render_page_text(rows)
    assert text.splitlines()[0] == "Header"
    assert text.splitlines()[1] == "Q1 2024 | 1,200,000"


# --------------------------------------------------------------------------- #
# End to end, against a real scan
# --------------------------------------------------------------------------- #


def make_scan(path: Path) -> Path:
    """A PDF that is genuinely a scan: pixels, with no text layer at all."""
    from PIL import Image, ImageDraw, ImageFont

    lines = [
        "ACME TRADING LIMITED",
        "Quarterly Results Summary",
        "",
        "Quarter        Revenue      Costs        Margin",
        "Q1 2024      1,200,000     780,000       35.0%",
        "Q2 2024      1,350,000     850,000       37.0%",
        "Q3 2024      1,180,000     810,000       31.4%",
        "Q4 2024      1,520,000     900,000       40.8%",
        "",
        "Revenue for the full year was 5,250,000 against a",
        "prior year of 4,410,000, an increase of 19 percent.",
    ]

    def font(size: int):
        for name in ("arial.ttf", "DejaVuSans.ttf", "calibri.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()

    image = Image.new("RGB", (1654, 2339), "white")
    draw = ImageDraw.Draw(image)
    y = 180
    for index, line in enumerate(lines):
        if not line:
            y += 30
            continue
        draw.text((140, y), line, fill="black", font=font(46 if index == 0 else 34))
        y += 62 if index == 0 else 52
    image.save(path, "PDF", resolution=200.0)
    return path


@pytest.fixture
def scanned(tmp_path: Path) -> Iterator[tuple[LocalStore, str, Path]]:
    pdf = make_scan(tmp_path / "scan.pdf")
    store = LocalStore(tmp_path / "resx.db")
    datasets = tmp_path / "datasets"
    report = ingest_file(pdf, workspace_id=WORKSPACE, store=store, dataset_dir=datasets)
    yield store, report.doc_id, datasets
    store.close()


def test_the_scan_really_has_no_text_layer(tmp_path: Path) -> None:
    """Guards the fixture: if this PDF ever grows a text layer, every test
    below stops testing OCR and starts passing for the wrong reason."""
    import pdfplumber

    pdf = make_scan(tmp_path / "scan.pdf")
    with pdfplumber.open(pdf) as document:
        assert (document.pages[0].extract_text() or "") == ""


@needs_tesseract
def test_a_scanned_page_is_read(tmp_path: Path) -> None:
    result = extract(make_scan(tmp_path / "scan.pdf"))
    assert result.char_count > 200
    assert "1,200,000" in (result.page(1).text or "")
    assert any("read by OCR" in w for w in result.warnings)


@needs_tesseract
def test_confidence_and_boxes_travel_with_the_block(tmp_path: Path) -> None:
    """`TextBlock.is_low_confidence_ocr` has always existed and never had
    anything to compare — this is what feeds it."""
    result = extract(make_scan(tmp_path / "scan.pdf"))
    blocks = result.page(1).blocks
    assert blocks
    assert all(b.ocr_confidence is not None for b in blocks)
    assert all(b.bbox is not None for b in blocks)
    # Boxes are in PDF points, so they must fall inside a page, not inside a
    # 1654x2339 pixel raster.
    assert max(b.bbox[2] for b in blocks) < 1000


@needs_tesseract
def test_a_scanned_table_becomes_a_dataset(
    scanned: tuple[LocalStore, str, Path],
) -> None:
    """Without this a scanned statement reaches the agents as prose they are
    forbidden to do arithmetic on."""
    store, _, _ = scanned
    datasets = store.list_datasets(workspace_id=WORKSPACE)
    assert datasets, "no dataset registered from the scanned table"
    profile = datasets[0]["profile"]
    roles = {c["name"]: c["candidate_role"] for c in profile["columns"]}
    assert any(role == "currency" for role in roles.values())


@needs_tesseract
def test_a_reconstructed_table_is_marked_unconfirmed(
    scanned: tuple[LocalStore, str, Path],
) -> None:
    """Only one strategy produced it, so there is nothing to compare against.
    An OCR table is a lead, not a ledger."""
    store, doc_id, _ = scanned
    assert store.list_datasets(workspace_id=WORKSPACE)[0]["agreement"] is None
    document = store.get_document(workspace_id=WORKSPACE, doc_id=doc_id)
    assert any("reconstructed from the scan" in str(w) for w in document.get("warnings") or [])


@needs_tesseract
def test_a_quote_from_a_scanned_page_resolves(
    scanned: tuple[LocalStore, str, Path],
) -> None:
    """The regression. The page used to be rendered with plain spaces while the
    chunk used " | ", so a faithful quote scored 0.84 against a 0.92 floor."""
    store, doc_id, _ = scanned
    chunk = next(c for c in store.iter_chunks(workspace_id=WORKSPACE) if c.kind == "table")
    body = [line for line in chunk.text.splitlines() if not line.lstrip().startswith("[")]
    verdict = resolve_quote(
        store=store,
        workspace_id=WORKSPACE,
        doc_id=doc_id,
        page=1,
        quote="\n".join(body[:2]),
        char_start=chunk.char_start,
        char_end=chunk.char_end,
    )
    assert verdict.ok, f"{verdict.resolution.value} at {verdict.score}"


@needs_tesseract
def test_a_scanned_page_has_no_anchor_problems(
    scanned: tuple[LocalStore, str, Path],
) -> None:
    store, doc_id, _ = scanned
    document = store.get_document(workspace_id=WORKSPACE, doc_id=doc_id)
    assert not [w for w in document.get("warnings") or [] if "span" in str(w)]


@needs_tesseract
def test_charts_are_derived_from_a_scanned_pdf(
    scanned: tuple[LocalStore, str, Path],
) -> None:
    """The end of the chain: pixels in, a plotted time series out."""
    from app.reporting.charts import build_charts

    store, doc_id, datasets = scanned
    charts = build_charts(
        store=store,
        workspace_id=WORKSPACE,
        corpus_ids=[doc_id],
        dataset_dir=datasets,
    )
    assert charts, "no chart derived from the scanned table"
    values = [p["y"] for c in charts for s in c.series for p in s.points]
    assert 1200000 in values or 1200000.0 in values


def test_missing_tesseract_is_reported_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure mode of ignoring it is a scan that ingests as empty with no
    explanation — the exact bug this module was written to fix."""
    from app.ingest import ocr

    monkeypatch.setattr(ocr, "tesseract_path", lambda: None)
    result = extract(make_scan(tmp_path / "scan.pdf"))
    assert result.char_count == 0
    assert any("Tesseract is not installed" in w for w in result.warnings)
    # And the page is flagged, so nothing downstream treats it as read.
    assert result.low_confidence_pages == [1]
