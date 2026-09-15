"""Exports carry what the screen carries.

The risk with four writers over one report is not that one crashes — that is
loud. It is that one quietly holds less than the others: a section the PDF has
and the Word file does not, a chart whose caveat survives on screen and is
dropped from the workbook. A reader then cites a figure without the warning
attached to it, and nothing anywhere reported a failure.

So parity is asserted rather than maintained by hand. Every section title and
every chart title present in the report has to appear in each format that
claims to be the whole report.
"""

from __future__ import annotations

import io
from typing import Any

import pytest

from app.reporting.export import (
    FORMATS,
    ExportError,
    export_report,
    safe_filename,
    to_docx,
    to_pdf,
    to_xlsx,
)


def chart(chart_id: str, title: str, kind: str = "bar", **extra: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "chart_id": chart_id,
        "title": title,
        "kind": kind,
        "x_label": "Quarter",
        "y_label": "Amount",
        "unit": "INR",
        "caption": f"caption for {title}",
        "warnings": [],
        "dataset_id": "ds_1",
        "doc_id": "doc_1",
        "source_name": "sales.csv",
        "source_page": 1,
        "palette": "categorical",
        "series": [
            {
                "key": "revenue",
                "label": "Revenue",
                "points": [
                    {"x": "Q1", "y": 182.4},
                    {"x": "Q2", "y": 196.8},
                    {"x": "Q3", "y": 214.5},
                    {"x": "Q4", "y": 221.3},
                ],
            }
        ],
    }
    base.update(extra)
    return base


@pytest.fixture
def report() -> dict[str, Any]:
    return {
        "question": "What drove margin compression?",
        "executive_summary": "Gross margin fell from 34.2% to 29.8% across the year.",
        "financial_summary": "Revenue INR 815.0 crore.",
        "insights": [
            {
                "statement": "Margin compressed every quarter",
                "so_what": "The trend is structural",
                "confidence": 0.92,
                "claim_ids": ["c1"],
            }
        ],
        "recommendations": [
            {
                "action": "Raise list prices by 6%",
                "rationale": "4% does not recover 440bps",
                "owner": "Commercial",
                "horizon": "Q1 FY2026",
                "claim_ids": ["c1"],
            }
        ],
        "swot": {"strengths": ["North at 33.8%"], "weaknesses": ["West at 61% utilisation"]},
        "risk_register": ["Top five accounts are 41% of revenue"],
        "limitations": ["Two orders are unconfirmed"],
        "charts": [
            chart("ch_bar", "Revenue by quarter"),
            chart("ch_line", "Margin trend", kind="line"),
            chart(
                "ch_donut",
                "Review outcome",
                kind="donut",
                palette="status",
                warnings=["3 findings were not reviewed"],
                series=[
                    {
                        "key": "outcome",
                        "label": "Findings",
                        "points": [
                            {"x": "Confirmed", "y": 7.0},
                            {"x": "Unreviewed", "y": 3.0},
                            {"x": "Refuted", "y": 1.0},
                        ],
                    }
                ],
            ),
        ],
        "document": {
            "question": "What drove margin compression?",
            "estimated_pages": 6.1,
            "markdown": "# What drove margin compression?\n\nBody.",
            "sections": [
                {
                    "slug": "summary",
                    "title": "Executive summary",
                    "order": 10,
                    "body": (
                        "Margin fell from **34.2%** to **29.8%**.\n\n"
                        "- Input costs rose 14%\n"
                        "- No price increase was taken\n\n"
                        "| Quarter | Margin |\n| --- | --- |\n| Q1 | 34.2% |\n| Q4 | 29.8% |"
                    ),
                },
                {
                    "slug": "risks",
                    "title": "Risks",
                    "order": 85,
                    "body": "### Concentration\n\nTop five accounts are 41% of revenue.",
                },
            ],
        },
    }


def pdf_text(data: bytes) -> str:
    import pypdfium2

    doc = pypdfium2.PdfDocument(io.BytesIO(data))
    return "".join(doc[i].get_textpage().get_text_range() for i in range(len(doc)))


def docx_text(data: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            parts.extend(cell.text for cell in row.cells)
    return "\n".join(parts)


def xlsx_text(data: bytes) -> str:
    from openpyxl import load_workbook

    book = load_workbook(io.BytesIO(data))
    parts = list(book.sheetnames)
    for sheet in book.worksheets:
        for row in sheet.iter_rows(values_only=True):
            parts.extend(str(v) for v in row if v is not None)
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Parity — the property that matters
# --------------------------------------------------------------------------- #


def test_every_section_reaches_every_document_format(report: dict[str, Any]) -> None:
    """A section the reader sees on screen is in the PDF and in the Word file.

    Silently dropping one is the failure this whole test module exists for: it
    looks like a complete report and is not.
    """
    titles = [s["title"] for s in report["document"]["sections"]]
    for name, text in (
        ("pdf", pdf_text(to_pdf(report))),
        ("docx", docx_text(to_docx(report))),
    ):
        for title in titles:
            assert title in text, f"{name} is missing the {title!r} section"


def test_every_chart_reaches_every_format(report: dict[str, Any]) -> None:
    titles = [c["title"] for c in report["charts"]]
    for name, text in (
        ("pdf", pdf_text(to_pdf(report))),
        ("docx", docx_text(to_docx(report))),
        ("xlsx", xlsx_text(to_xlsx(report))),
    ):
        for title in titles:
            # The workbook truncates sheet names to Excel's limit, so a prefix
            # match is the honest assertion there.
            assert title[:24] in text, f"{name} is missing the {title!r} chart"


def test_a_chart_warning_is_never_dropped(report: dict[str, Any]) -> None:
    """The caveat has to travel with the figure.

    A reader who takes "Review outcome" from the PDF without "3 findings were
    not reviewed" has been told the run was cleaner than it was — and that is
    the one error an export can make that is worse than failing outright.
    """
    warning = report["charts"][2]["warnings"][0]
    assert warning in pdf_text(to_pdf(report))
    assert warning in docx_text(to_docx(report))
    assert warning in xlsx_text(to_xlsx(report))


def test_chart_numbers_travel_with_the_picture(report: dict[str, Any]) -> None:
    """Not just an image. A figure whose values cannot be read is half a
    figure, and colour-blind and print readers have only the table."""
    for text in (
        pdf_text(to_pdf(report)),
        docx_text(to_docx(report)),
        xlsx_text(to_xlsx(report)),
    ):
        assert "182.4" in text or "182" in text
        assert "221.3" in text or "221" in text


def test_provenance_travels_with_the_figure(report: dict[str, Any]) -> None:
    """A chart without its source is an assertion rather than evidence."""
    assert "sales.csv" in pdf_text(to_pdf(report))
    assert "sales.csv" in docx_text(to_docx(report))


# --------------------------------------------------------------------------- #
# Each format is genuinely that format
# --------------------------------------------------------------------------- #


def test_the_pdf_is_a_pdf_with_figures(report: dict[str, Any]) -> None:
    import pypdfium2

    data = to_pdf(report, run_id="run_x")
    assert data.startswith(b"%PDF-")
    doc = pypdfium2.PdfDocument(io.BytesIO(data))
    assert len(doc) >= 2, "a report with three figures does not fit on one page"


def test_the_docx_uses_real_heading_styles(report: dict[str, Any]) -> None:
    """Bold paragraphs would look the same and break Word's navigation pane and
    any inserted table of contents, which is the reason to want .docx at all."""
    from docx import Document

    doc = Document(io.BytesIO(to_docx(report)))
    headings = [p.text for p in doc.paragraphs if p.style.name.startswith("Heading")]
    assert "Executive summary" in headings
    assert "Risks" in headings


def test_the_docx_embeds_the_chart_images(report: dict[str, Any]) -> None:
    from docx import Document

    doc = Document(io.BytesIO(to_docx(report)))
    images = [r for r in doc.part.rels.values() if "image" in r.reltype]
    assert len(images) == len(report["charts"])


def test_the_workbook_gives_each_chart_its_own_sheet(report: dict[str, Any]) -> None:
    from openpyxl import load_workbook

    book = load_workbook(io.BytesIO(to_xlsx(report)))
    assert "Summary" in book.sheetnames
    assert "Insights" in book.sheetnames
    assert "Recommendations" in book.sheetnames
    # One per chart, plus the fixed sheets.
    assert len(book.sheetnames) >= len(report["charts"]) + 3


def test_two_charts_with_the_same_long_title_do_not_collide(report: dict[str, Any]) -> None:
    """Excel caps a sheet name at 31 characters and openpyxl raises on a
    duplicate rather than disambiguating, so two charts from one table — which
    is the normal case, one per unit — would have failed the whole export."""
    long = "Regional breakdown of revenue and margin by quarter"
    report["charts"] = [chart("a", long), chart("b", long, kind="line")]
    from openpyxl import load_workbook

    book = load_workbook(io.BytesIO(to_xlsx(report)))
    assert len(book.sheetnames) == len(set(book.sheetnames))
    assert all(len(n) <= 31 for n in book.sheetnames)


# --------------------------------------------------------------------------- #
# Edges
# --------------------------------------------------------------------------- #


def test_a_report_with_no_charts_still_exports(report: dict[str, Any]) -> None:
    """A research run with no tables has no figures, and must not be
    undownloadable because of it."""
    report["charts"] = []
    for fmt in FORMATS:
        assert export_report(report, fmt).content


def test_an_unknown_format_is_refused_by_name(report: dict[str, Any]) -> None:
    with pytest.raises(ExportError, match="unknown export format"):
        export_report(report, "pptx")


def test_a_chart_that_cannot_be_drawn_does_not_kill_the_export(
    report: dict[str, Any],
) -> None:
    """The numbers still reach the reader through the table, which is the point
    of carrying both."""
    report["charts"] = [chart("empty", "Nothing to draw", series=[])]
    assert to_pdf(report)
    assert to_docx(report)


def test_the_filename_survives_a_question_made_only_of_punctuation() -> None:
    """An empty stem would download as ".pdf", which Windows refuses to save."""
    assert safe_filename("???", "pdf") == "report.pdf"
    assert safe_filename("", "xlsx") == "report.xlsx"
    assert safe_filename("Q1 revenue: what happened?", "docx").endswith(".docx")
    assert "/" not in safe_filename("a/b/c", "pdf")


def test_markdown_export_is_the_assembled_document(report: dict[str, Any]) -> None:
    export = export_report(report, "md")
    assert export.content.decode("utf-8") == report["document"]["markdown"]
    assert export.media_type.startswith("text/markdown")


def test_every_format_declares_a_media_type_a_browser_acts_on(
    report: dict[str, Any],
) -> None:
    """Served as text/plain, a PDF opens as gibberish in the browser instead of
    downloading."""
    expected = {
        "pdf": "application/pdf",
        "docx": "officedocument.wordprocessingml",
        "xlsx": "officedocument.spreadsheetml",
        "md": "text/markdown",
    }
    for fmt, fragment in expected.items():
        assert fragment in export_report(report, fmt).media_type


# --------------------------------------------------------------------------- #
# The route
# --------------------------------------------------------------------------- #


def seed_run(client: Any, report: dict[str, Any]) -> str:
    """A finished run in the store, without spending a model call to get one."""
    from app.api import deps

    store = deps.get_store()
    workspace = client.get("/api/v1/auth/me").json()["workspace"]["workspace_id"]
    run_id = "run_exporttest"
    store.create_run(
        workspace_id=workspace,
        run_id=run_id,
        question=report["question"],
        corpus_ids=[],
    )
    store.update_run(workspace_id=workspace, run_id=run_id, status="done", report=report)
    return run_id


def test_the_route_serves_each_format(client: Any, report: dict[str, Any]) -> None:
    run_id = seed_run(client, report)
    for fmt, magic in (
        ("pdf", b"%PDF-"),
        ("docx", b"PK"),  # OOXML is a zip
        ("xlsx", b"PK"),
    ):
        response = client.get(f"/api/v1/runs/{run_id}/export?format={fmt}")
        assert response.status_code == 200, response.text
        assert response.content.startswith(magic), f"{fmt} is not really a {fmt}"
        assert "attachment" in response.headers["content-disposition"]


def test_the_route_refuses_an_unknown_format(client: Any, report: dict[str, Any]) -> None:
    run_id = seed_run(client, report)
    response = client.get(f"/api/v1/runs/{run_id}/export?format=pptx")
    assert response.status_code == 400
    assert "unknown export format" in str(response.json())


def test_exporting_an_unfinished_run_says_so(client: Any) -> None:
    """A 409 naming the status, rather than a 500 from rendering `None`."""
    from app.api import deps

    store = deps.get_store()
    workspace = client.get("/api/v1/auth/me").json()["workspace"]["workspace_id"]
    store.create_run(workspace_id=workspace, run_id="run_pending", question="q", corpus_ids=[])
    response = client.get("/api/v1/runs/run_pending/export?format=pdf")
    assert response.status_code == 409
    assert "no report yet" in str(response.json())


def test_exporting_an_unknown_run_is_a_404(client: Any) -> None:
    assert client.get("/api/v1/runs/run_nope/export?format=pdf").status_code == 404


def test_another_workspace_cannot_export_this_report(
    client: Any, report: dict[str, Any]
) -> None:
    """Tenancy holds on the export route too. A report is the most concentrated
    form of a workspace's private data that exists in this system."""
    run_id = seed_run(client, report)
    other = client.post(
        "/api/v1/auth/register",
        json={
            "email": "exporter@example.com",
            "password": "correct-horse-battery-staple-9",
            "name": "Other",
        },
    )
    assert other.status_code == 201, other.text
    client.headers["Authorization"] = f"Bearer {other.json()['access_token']}"
    assert client.get(f"/api/v1/runs/{run_id}/export?format=pdf").status_code == 404
