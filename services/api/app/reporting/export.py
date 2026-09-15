"""The report as a file someone can send to someone else.

Markdown was the only format, and Markdown is a developer's format. A finance
lead asked for "the report" and got a `.md` file their machine had no idea how
to open. The analysis was finished and the last ten metres were missing.

Four formats, each for a different reader:

* **PDF** — the one that gets emailed and printed. Fixed layout, figures drawn
  as images, nothing to install.
* **DOCX** — the one that gets edited. Real Word heading styles, so the
  navigation pane and an inserted table of contents work and the reader can
  paste a section into their own deck.
* **XLSX** — the one that gets checked. Every chart's numbers as a sheet, plus
  claims, verdicts and citations, so a reader can re-derive a figure instead of
  trusting it. This is the format that makes the analysis auditable.
* **MD** — kept, because it is the one a diff can read.

**The rule that binds them: every export carries what the screen carries.** A
figure that appears in the run page appears in the PDF as an image *and* as its
numbers; a section in the document is a section in the file. Export formats
that quietly hold less than the UI are how a reader ends up citing a figure
they never saw the caveat for, so `test_export.py` asserts parity rather than
trusting that these three writers were kept in step by hand.

Charts are rendered by `app.reporting.figures`, which copies the web palette
so a figure is the same figure in both places.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.reporting.figures import chart_table, render_chart_png

#: What a caller may ask for.
FORMATS: tuple[str, ...] = ("pdf", "docx", "xlsx", "md")

MEDIA_TYPES: dict[str, str] = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "md": "text/markdown; charset=utf-8",
}

#: Excel's own limit on a worksheet name, minus the room a disambiguating
#: suffix needs.
SHEET_NAME_LIMIT = 28

_ILLEGAL_SHEET = re.compile(r"[\[\]:*?/\\]")
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_MD_BULLET = re.compile(r"^\s*[-*]\s+(.*)$")
_MD_TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")


class ExportError(RuntimeError):
    """A format could not be produced. Names the format and the reason."""


@dataclass
class Export:
    """A finished file, ready to stream."""

    content: bytes
    media_type: str
    filename: str


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def safe_filename(question: str, suffix: str) -> str:
    """A filename from the question, safe on every filesystem.

    Falls back to "report" rather than producing an empty name: a question in a
    script the regex strips entirely would otherwise download as ".pdf", which
    Windows refuses to save.
    """
    stem = re.sub(r"[^\w\s-]", "", question or "").strip()
    stem = re.sub(r"\s+", "-", stem)[:60].strip("-")
    return f"{stem or 'report'}.{suffix}"


def _sections(document: dict[str, Any]) -> list[dict[str, Any]]:
    sections = list(document.get("sections") or [])
    return sorted(
        sections,
        key=lambda s: (int(s.get("order") or 100), str(s.get("title") or "")),
    )


def _charts(report: dict[str, Any]) -> list[Any]:
    """Charts as objects the figure renderer understands.

    They are stored as plain dicts, and the renderer reads attributes, so they
    are wrapped rather than rewritten — one adapter here beats a second dict
    code path in every drawing function.
    """
    return [_ChartView(c) for c in (report.get("charts") or [])]


class _SeriesView:
    def __init__(self, data: dict[str, Any]) -> None:
        self.key = str(data.get("key") or "")
        self.label = str(data.get("label") or self.key)
        self.points = list(data.get("points") or [])


class _ChartView:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
        self.chart_id = str(data.get("chart_id") or "")
        self.title = str(data.get("title") or "Figure")
        self.kind = str(data.get("kind") or "bar")
        self.x_label = str(data.get("x_label") or "")
        self.y_label = str(data.get("y_label") or "")
        self.unit = data.get("unit")
        self.caption = str(data.get("caption") or "")
        self.warnings = list(data.get("warnings") or [])
        self.palette = str(data.get("palette") or "categorical")
        self.dataset_id = str(data.get("dataset_id") or "")
        self.doc_id = str(data.get("doc_id") or "")
        self.source_name = str(data.get("source_name") or "")
        self.source_page = data.get("source_page")
        self.series = [_SeriesView(s) for s in (data.get("series") or [])]

    def provenance(self) -> str:
        if not self.source_name:
            return ""
        page = f", p.{self.source_page}" if self.source_page else ""
        return f"Source: {self.source_name}{page}"


def _parse_markdown(body: str) -> list[tuple[str, Any]]:
    """Section body as a flat list of (kind, payload) blocks.

    A deliberately small subset — headings, bullets, tables, paragraphs — is
    parsed, because that is all `app.reporting.document` emits. A general
    Markdown parser would be more code and would invite the writers here to
    drift apart from what the assembler actually produces.
    """
    blocks: list[tuple[str, Any]] = []
    paragraph: list[str] = []
    table: list[list[str]] = []

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(("p", " ".join(paragraph).strip()))
            paragraph.clear()

    def flush_table() -> None:
        if table:
            blocks.append(("table", [list(row) for row in table]))
            table.clear()

    for raw in (body or "").splitlines():
        line = raw.rstrip()
        stripped = line.strip()

        if stripped.startswith("|") and _MD_TABLE_SEP.match(stripped):
            continue  # the |---|---| rule carries no data
        if stripped.startswith("|"):
            flush_paragraph()
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            table.append(cells)
            continue
        flush_table()

        if not stripped:
            flush_paragraph()
            continue

        heading = _MD_HEADING.match(stripped)
        if heading:
            flush_paragraph()
            blocks.append((f"h{len(heading.group(1))}", heading.group(2).strip()))
            continue

        bullet = _MD_BULLET.match(line)
        if bullet:
            flush_paragraph()
            blocks.append(("li", bullet.group(1).strip()))
            continue

        paragraph.append(stripped)

    flush_paragraph()
    flush_table()
    return blocks


def _strip_inline(text: str) -> str:
    """Drop Markdown emphasis markers.

    The writers below set weight with real styles, so leaving `**` in place
    would print the asterisks literally next to already-bold text.
    """
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*", r"\1", text)
    return re.sub(r"`(.+?)`", r"\1", text)


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #


def to_pdf(report: dict[str, Any], *, run_id: str = "") -> bytes:
    """The report as a PDF, figures included."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Image,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
        )
    except ImportError as exc:  # pragma: no cover - reportlab is a dependency
        raise ExportError(f"PDF export needs reportlab: {exc}") from exc

    document = dict(report.get("document") or {})
    question = str(document.get("question") or report.get("question") or "Report")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title=question[:120],
        author="RESX",
    )

    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontSize=20, leading=25, alignment=TA_LEFT
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontSize=14,
            leading=18,
            spaceBefore=14,
            textColor=colors.HexColor("#1c1c1c"),
        ),
        "h3": ParagraphStyle("h3", parent=base["Heading3"], fontSize=11.5, leading=15),
        "body": ParagraphStyle(
            "body", parent=base["BodyText"], fontSize=9.5, leading=14, spaceAfter=6
        ),
        "bullet": ParagraphStyle(
            "bullet", parent=base["BodyText"], fontSize=9.5, leading=14, leftIndent=10
        ),
        "muted": ParagraphStyle(
            "muted",
            parent=base["BodyText"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#6b6b6b"),
        ),
        "cell": ParagraphStyle("cell", parent=base["BodyText"], fontSize=8, leading=10.5),
    }

    def paragraph(text: str, style: str = "body") -> Any:
        return Paragraph(_escape(_strip_inline(text)), styles[style])

    story: list[Any] = [
        paragraph(question, "title"),
        Spacer(1, 4 * mm),
        paragraph(
            f"Generated {datetime.now(timezone.utc).strftime('%d %B %Y %H:%M UTC')}"
            + (f" · run {run_id}" if run_id else ""),
            "muted",
        ),
        Spacer(1, 6 * mm),
    ]

    for section in _sections(document):
        story.append(paragraph(str(section.get("title") or ""), "h2"))
        for kind, payload in _parse_markdown(str(section.get("body") or "")):
            if kind == "table":
                story.append(_pdf_table(payload, styles))
            elif kind.startswith("h"):
                story.append(paragraph(str(payload), "h3"))
            elif kind == "li":
                story.append(paragraph(f"• {payload}", "bullet"))
            else:
                story.append(paragraph(str(payload)))

    charts = _charts(report)
    if charts:
        story.append(PageBreak())
        story.append(paragraph("Figures", "h2"))
        for chart in charts:
            story.append(paragraph(chart.title, "h3"))
            png = render_chart_png(chart)
            if png:
                # Scaled to the text column so a wide chart never runs off the
                # page; the aspect ratio is preserved so nothing is distorted.
                width = doc.width
                story.append(Image(io.BytesIO(png), width=width, height=width * 0.5))
                story.append(Spacer(1, 2 * mm))
            if chart.caption:
                story.append(paragraph(chart.caption, "muted"))
            for warning in chart.warnings:
                story.append(paragraph(f"Warning: {warning}", "muted"))
            header, rows = chart_table(chart)
            if header and rows:
                story.append(Spacer(1, 2 * mm))
                story.append(_pdf_table([header, *rows], styles))
            if chart.provenance():
                story.append(paragraph(chart.provenance(), "muted"))
            story.append(Spacer(1, 7 * mm))

    def stamp(canvas: Any, _doc: Any) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#6b6b6b"))
        canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, f"Page {canvas.getPageNumber()}")
        canvas.drawString(18 * mm, 12 * mm, "RESX")
        canvas.restoreState()

    doc.build(story, onFirstPage=stamp, onLaterPages=stamp)
    return buffer.getvalue()


def _escape(text: str) -> str:
    """Reportlab reads a paragraph as mini-HTML, so raw `&`/`<` would break it."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _pdf_table(rows: list[list[str]], styles: Any) -> Any:
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Table, TableStyle

    # Every cell is a Paragraph so long text wraps instead of overflowing the
    # column, which is what a plain string does in reportlab.
    body = [
        [Paragraph(_escape(_strip_inline(str(c))), styles["cell"]) for c in row] for row in rows
    ]
    table = Table(body, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f4f7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1c1c1c")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dcdcdc")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


# --------------------------------------------------------------------------- #
# DOCX
# --------------------------------------------------------------------------- #


def to_docx(report: dict[str, Any], *, run_id: str = "") -> bytes:
    """The report as a Word document.

    Real heading styles rather than bold paragraphs, so Word's navigation pane
    and an inserted table of contents both work — which is the reason someone
    asks for .docx instead of .pdf in the first place.
    """
    try:
        from docx import Document
        from docx.shared import Inches, Pt, RGBColor
    except ImportError as exc:  # pragma: no cover - python-docx is a dependency
        raise ExportError(f"DOCX export needs python-docx: {exc}") from exc

    document = dict(report.get("document") or {})
    question = str(document.get("question") or report.get("question") or "Report")

    docx = Document()
    docx.core_properties.title = question[:120]
    docx.core_properties.author = "RESX"

    docx.add_heading(question, level=0)
    stamp = docx.add_paragraph(
        f"Generated {datetime.now(timezone.utc).strftime('%d %B %Y %H:%M UTC')}"
        + (f" · run {run_id}" if run_id else "")
    )
    for run in stamp.runs:
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(0x6B, 0x6B, 0x6B)

    for section in _sections(document):
        docx.add_heading(str(section.get("title") or ""), level=1)
        for kind, payload in _parse_markdown(str(section.get("body") or "")):
            if kind == "table":
                _docx_table(docx, payload)
            elif kind.startswith("h"):
                docx.add_heading(_strip_inline(str(payload)), level=2)
            elif kind == "li":
                docx.add_paragraph(_strip_inline(str(payload)), style="List Bullet")
            else:
                docx.add_paragraph(_strip_inline(str(payload)))

    charts = _charts(report)
    if charts:
        docx.add_page_break()  # type: ignore[no-untyped-call]
        docx.add_heading("Figures", level=1)
        for chart in charts:
            docx.add_heading(chart.title, level=2)
            png = render_chart_png(chart)
            if png:
                docx.add_picture(io.BytesIO(png), width=Inches(6.2))
            if chart.caption:
                _muted(docx.add_paragraph(chart.caption))
            for warning in chart.warnings:
                _muted(docx.add_paragraph(f"Warning: {warning}"))
            header, rows = chart_table(chart)
            if header and rows:
                _docx_table(docx, [header, *rows])
            if chart.provenance():
                _muted(docx.add_paragraph(chart.provenance()))

    buffer = io.BytesIO()
    docx.save(buffer)
    return buffer.getvalue()


def _muted(paragraph: Any) -> None:
    """Caption weight: small and grey, so it reads as provenance rather than
    as another sentence of the report."""
    from docx.shared import Pt, RGBColor

    for run in paragraph.runs:
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(0x6B, 0x6B, 0x6B)


def _docx_table(docx: Any, rows: list[list[str]]) -> None:
    if not rows:
        return
    width = max(len(r) for r in rows)
    table = docx.add_table(rows=0, cols=width)
    table.style = "Light Grid Accent 1"
    for index, row in enumerate(rows):
        cells = table.add_row().cells
        for column in range(width):
            text = _strip_inline(str(row[column])) if column < len(row) else ""
            cells[column].text = text
            if index == 0:
                for paragraph in cells[column].paragraphs:
                    for run in paragraph.runs:
                        run.bold = True


# --------------------------------------------------------------------------- #
# XLSX
# --------------------------------------------------------------------------- #


def to_xlsx(report: dict[str, Any], *, run_id: str = "") -> bytes:
    """The report as a workbook — the format that makes it checkable.

    A PDF asks to be believed. A workbook hands over the numbers: every chart's
    series as a sheet, every claim with its confidence and reviewer verdict,
    every citation with its source and page. A reader who wants to re-derive a
    figure can, which is the difference between a report and an assertion.
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError as exc:  # pragma: no cover - openpyxl is a dependency
        raise ExportError(f"XLSX export needs openpyxl: {exc}") from exc

    document = dict(report.get("document") or {})
    question = str(document.get("question") or report.get("question") or "Report")

    book = Workbook()
    head_font = Font(bold=True, color="FFFFFF")
    head_fill = PatternFill("solid", fgColor="2A78D6")
    wrap = Alignment(vertical="top", wrap_text=True)

    def sheet(title: str, headers: list[str]) -> Any:
        name = _unique_sheet_name(book, title)
        ws = book.create_sheet(name)
        ws.append(headers)
        for cell in ws[1]:
            cell.font = head_font
            cell.fill = head_fill
        ws.freeze_panes = "A2"
        return ws

    def autosize(ws: Any, widths: list[int]) -> None:
        for index, width in enumerate(widths, start=1):
            ws.column_dimensions[get_column_letter(index)].width = width

    book.remove(book.active)

    # --- Summary -----------------------------------------------------------
    overview = sheet("Summary", ["Field", "Value"])
    for key, value in (
        ("Question", question),
        ("Run", run_id),
        ("Generated", datetime.now(timezone.utc).strftime("%d %B %Y %H:%M UTC")),
        ("Estimated pages", document.get("estimated_pages", "")),
        ("Executive summary", report.get("executive_summary", "")),
        ("Financial summary", report.get("financial_summary", "")),
    ):
        overview.append([key, str(value)])
    for row in overview.iter_rows(min_row=2, min_col=2, max_col=2):
        for cell in row:
            cell.alignment = wrap
    autosize(overview, [22, 110])

    # --- Insights and recommendations --------------------------------------
    insights = report.get("insights") or []
    if insights:
        ws = sheet("Insights", ["#", "Insight", "So what", "Confidence", "Claims"])
        for index, item in enumerate(insights, start=1):
            ws.append(
                [
                    index,
                    str(item.get("statement") or item.get("insight") or ""),
                    str(item.get("so_what") or item.get("implication") or ""),
                    item.get("confidence", ""),
                    ", ".join(str(c) for c in (item.get("claim_ids") or [])),
                ]
            )
        _wrap_all(ws, wrap)
        autosize(ws, [5, 70, 60, 12, 30])

    recommendations = report.get("recommendations") or []
    if recommendations:
        ws = sheet(
            "Recommendations",
            ["#", "Recommendation", "Rationale", "Owner", "Horizon", "Claims"],
        )
        for index, item in enumerate(recommendations, start=1):
            ws.append(
                [
                    index,
                    str(item.get("action") or item.get("statement") or ""),
                    str(item.get("rationale") or item.get("why") or ""),
                    str(item.get("owner") or ""),
                    str(item.get("horizon") or item.get("timeframe") or ""),
                    ", ".join(str(c) for c in (item.get("claim_ids") or [])),
                ]
            )
        _wrap_all(ws, wrap)
        autosize(ws, [5, 60, 60, 18, 16, 30])

    # --- Risks, limitations, SWOT -----------------------------------------
    risks = report.get("risk_register") or []
    limitations = report.get("limitations") or []
    if risks or limitations:
        ws = sheet("Risks and limits", ["Kind", "Statement"])
        for risk in risks:
            ws.append(["Risk", str(risk)])
        for limit in limitations:
            ws.append(["Limitation", str(limit)])
        _wrap_all(ws, wrap)
        autosize(ws, [14, 110])

    swot = report.get("swot") or {}
    if any(swot.get(k) for k in ("strengths", "weaknesses", "opportunities", "threats")):
        ws = sheet("SWOT", ["Quadrant", "Point"])
        for quadrant in ("strengths", "weaknesses", "opportunities", "threats"):
            for point in swot.get(quadrant) or []:
                ws.append([quadrant.title(), str(point)])
        _wrap_all(ws, wrap)
        autosize(ws, [16, 110])

    # --- One sheet per chart ----------------------------------------------
    for chart in _charts(report):
        header, rows = chart_table(chart)
        if not header or not rows:
            continue
        ws = sheet(chart.title, header)
        for row in rows:
            ws.append(row)
        ws.append([])
        ws.append(["Caption", chart.caption])
        for warning in chart.warnings:
            ws.append(["Warning", warning])
        if chart.provenance():
            ws.append(["Provenance", chart.provenance()])
        autosize(ws, [26] + [18] * max(len(header) - 1, 1))

    # --- The document itself ----------------------------------------------
    ws = sheet("Document", ["Section", "Body"])
    for section in _sections(document):
        ws.append([str(section.get("title") or ""), str(section.get("body") or "")])
    _wrap_all(ws, wrap)
    autosize(ws, [28, 140])

    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


def _wrap_all(ws: Any, wrap: Any) -> None:
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = wrap


def _unique_sheet_name(book: Any, title: str) -> str:
    """A worksheet name Excel will accept and that is not already taken.

    Excel caps names at 31 characters and forbids `[]:*?/\\`. Two charts from
    the same table can easily share the first 28 characters, and openpyxl
    raises rather than disambiguating, so the suffix is added here.
    """
    cleaned = _ILLEGAL_SHEET.sub("-", title).strip() or "Sheet"
    base = cleaned[:SHEET_NAME_LIMIT]
    name = base
    counter = 2
    existing = set(book.sheetnames)
    while name in existing:
        name = f"{base} {counter}"
        counter += 1
    return name


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def export_report(report: dict[str, Any], fmt: str, *, run_id: str = "") -> Export:
    """One report in one format.

    The format is validated here rather than at the route, so every caller —
    the API, a test, a future scheduled email — refuses an unknown format the
    same way.
    """
    chosen = (fmt or "").strip().lower()
    if chosen not in FORMATS:
        raise ExportError(f"unknown export format {fmt!r}; choose one of {', '.join(FORMATS)}")

    document = dict(report.get("document") or {})
    question = str(document.get("question") or report.get("question") or "report")

    if chosen == "md":
        markdown = str(document.get("markdown") or "")
        if not markdown:
            raise ExportError("this run has no assembled document to export")
        content = markdown.encode("utf-8")
    elif chosen == "pdf":
        content = to_pdf(report, run_id=run_id)
    elif chosen == "docx":
        content = to_docx(report, run_id=run_id)
    else:
        content = to_xlsx(report, run_id=run_id)

    return Export(
        content=content,
        media_type=MEDIA_TYPES[chosen],
        filename=safe_filename(question, chosen),
    )
