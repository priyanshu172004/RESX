"""Charts derived from extracted tables, and the parser bug that blocked them.

Two things are locked down here.

**A chart is computed, never written.** The Synthesizer is asked for
`ExecutiveReportDraft`, which has no `charts` field at all, so a fabricated
series cannot even be expressed — and `Strict` turns an attempt into a
validation error rather than accepted data. A model asked for "revenue by
quarter" produces a plausible, smooth, entirely invented series, and a reader
who would interrogate a sentence will not interrogate a line.

**`parse_amount` answered any string containing a digit.** It searches, so
`parse_amount("Q1 2024")` returned 1 with the year silently discarded. A column
headed "Quarter" therefore scored as numeric, was classified `quantity`, and
`write_dataset` normalised its cells -- so the stored CSV read 1 / 2 / 3 and the
periods no longer existed. That corrupted data on the way in: a computation
summing the column added 1+2+3, a citation quoting the row could never match the
source page, and no time series was possible.
"""

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from app.core.money import parse_amount
from app.ingest.pipeline import ingest_file
from app.reporting.charts import build_charts, charts_for_dataset
from app.store.local import LocalStore

WORKSPACE = "ws_charts"

QUARTERLY = (
    "Quarter,Revenue,Cost,Margin %\n"
    "Q1 2024,1200000,780000,35.0\n"
    "Q2 2024,1350000,850000,37.0\n"
    "Q3 2024,1180000,810000,31.4\n"
    "Q4 2024,1520000,900000,40.8\n"
)

BY_REGION = "Region,Revenue\nNorth,4200000\nSouth,3100000\nEast,2400000\nWest,1800000\n"


@pytest.fixture
def corpus(tmp_path: Path) -> Iterator[tuple[LocalStore, Path, list[str]]]:
    store = LocalStore(tmp_path / "resx.db")
    datasets = tmp_path / "datasets"
    doc_ids = []
    for name, body in (
        ("quarterly_results.csv", QUARTERLY),
        ("revenue_by_region.csv", BY_REGION),
    ):
        source = tmp_path / name
        source.write_text(body, encoding="utf-8")
        report = ingest_file(source, workspace_id=WORKSPACE, store=store, dataset_dir=datasets)
        doc_ids.append(report.doc_id)
    yield store, datasets, doc_ids
    store.close()


# --------------------------------------------------------------------------- #
# The parser
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "cell",
    ["Q1 2024", "Q2 2024", "H1 2025", "FY2025", "Jan 2024", "2024 Q1", "3 of 12"],
)
def test_a_label_containing_a_number_is_not_a_figure(cell: str) -> None:
    """The regression. Each of these returned a number before."""
    assert parse_amount(cell) is None


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("48,920", Decimal("48920")),
        ("(48,920)", Decimal("-48920")),
        ("$1,234.50", Decimal("1234.50")),
        ("USD 1,234", Decimal("1234")),
        ("US$1,234", Decimal("1234")),
        ("-450", Decimal("-450")),
        ("35.0", Decimal("35.0")),
        ("2024", Decimal("2024")),
        ("1,234 *", Decimal("1234")),
    ],
)
def test_a_real_figure_still_parses(cell: str, expected: Decimal) -> None:
    """Declining must not become declining everything: accounting negatives,
    separators, symbols and bare ISO codes all still have to work."""
    assert parse_amount(cell) == expected


def test_period_labels_survive_ingestion(
    corpus: tuple[LocalStore, Path, list[str]],
) -> None:
    """The consequence, checked where it did the damage: on disk.

    The stored CSV held 1 / 2 / 3 instead of the quarters, so every consumer
    inherited corrupted data.
    """
    store, _, _ = corpus
    dataset = next(
        d for d in store.list_datasets(workspace_id=WORKSPACE) if "quarterly" in d["name"]
    )
    written = Path(dataset["path"]).read_text(encoding="utf-8")
    assert "Q1 2024" in written
    assert "Q4 2024" in written


def test_a_period_column_is_classified_as_a_period_not_a_quantity(
    corpus: tuple[LocalStore, Path, list[str]],
) -> None:
    """The profile is what the agents read, so a wrong role misleads them too,
    not only the charts."""
    store, _, _ = corpus
    dataset = next(
        d for d in store.list_datasets(workspace_id=WORKSPACE) if "quarterly" in d["name"]
    )
    roles = {c["name"]: c["candidate_role"] for c in dataset["profile"]["columns"]}
    assert roles["Quarter"] in {"date", "period"}
    assert roles["Revenue"] == "currency"
    assert roles["Margin %"] == "percentage"


def test_a_low_cardinality_unique_column_is_a_category(
    corpus: tuple[LocalStore, Path, list[str]],
) -> None:
    """The ratio test alone needs repetition, so a breakdown table -- one row
    per region, the commonest shape of business table there is -- scored 1.0
    and was classified "text"."""
    store, _, _ = corpus
    dataset = next(
        d for d in store.list_datasets(workspace_id=WORKSPACE) if "region" in d["name"]
    )
    roles = {c["name"]: c["candidate_role"] for c in dataset["profile"]["columns"]}
    assert roles["Region"] == "category"


# --------------------------------------------------------------------------- #
# Derivation
# --------------------------------------------------------------------------- #


def test_charts_are_derived_from_the_extracted_tables(
    corpus: tuple[LocalStore, Path, list[str]],
) -> None:
    store, datasets, doc_ids = corpus
    charts = build_charts(
        store=store,
        workspace_id=WORKSPACE,
        corpus_ids=doc_ids,
        dataset_dir=datasets,
    )
    assert charts, "no charts derived from two chartable tables"
    kinds = {c.kind for c in charts}
    assert "line" in kinds, "a period column should produce a time series"
    assert "bar" in kinds, "a category column should produce a comparison"


def test_a_percentage_never_shares_an_axis_with_an_amount(
    corpus: tuple[LocalStore, Path, list[str]],
) -> None:
    """The one-axis rule, and the reason this module groups by role. Revenue
    and margin% on one plot needs two y-scales, which is the single most
    misleading thing a chart can do."""
    store, datasets, doc_ids = corpus
    charts = build_charts(
        store=store, workspace_id=WORKSPACE, corpus_ids=doc_ids, dataset_dir=datasets
    )
    for chart in charts:
        units = {chart.unit}
        assert len(units) == 1
        if chart.unit == "%":
            assert all("margin" in s.key.lower() for s in chart.series)
        else:
            assert all("margin" not in s.key.lower() for s in chart.series)


def test_every_chart_carries_its_provenance(
    corpus: tuple[LocalStore, Path, list[str]],
) -> None:
    """A chart without a source is an assertion; with one it is evidence."""
    store, datasets, doc_ids = corpus
    for chart in build_charts(
        store=store, workspace_id=WORKSPACE, corpus_ids=doc_ids, dataset_dir=datasets
    ):
        assert chart.dataset_id
        assert chart.doc_id in doc_ids
        assert chart.source_name
        assert chart.caption


def test_plotted_values_match_the_source_table(
    corpus: tuple[LocalStore, Path, list[str]],
) -> None:
    """The whole claim of this feature: the points are the document's numbers."""
    store, datasets, doc_ids = corpus
    charts = build_charts(
        store=store, workspace_id=WORKSPACE, corpus_ids=doc_ids, dataset_dir=datasets
    )
    revenue_over_time = next(
        c for c in charts if c.kind == "line" and any(s.key == "revenue" for s in c.series)
    )
    series = next(s for s in revenue_over_time.series if s.key == "revenue")
    assert [p["x"] for p in series.points] == [
        "Q1 2024",
        "Q2 2024",
        "Q3 2024",
        "Q4 2024",
    ]
    assert [p["y"] for p in series.points] == [1200000, 1350000, 1180000, 1520000]


def test_a_run_scoped_to_one_document_does_not_chart_another(
    corpus: tuple[LocalStore, Path, list[str]],
) -> None:
    """Charts follow the run's corpus selection, like every other read."""
    store, datasets, doc_ids = corpus
    charts = build_charts(
        store=store,
        workspace_id=WORKSPACE,
        corpus_ids=[doc_ids[0]],
        dataset_dir=datasets,
    )
    assert charts
    assert {c.doc_id for c in charts} == {doc_ids[0]}


def test_a_table_with_nothing_to_label_produces_no_chart(tmp_path: Path) -> None:
    """Forcing a chart out of a table with no label column would mean
    inventing an axis. No chart is the honest outcome."""
    dataset = {
        "dataset_id": "ds_x",
        "doc_id": "doc_x",
        "name": "notes",
        "path": str(tmp_path / "notes.csv"),
        "source_page": 1,
        "profile": {
            "columns": [
                {"name": "Comment", "index": 0, "candidate_role": "text"},
                {"name": "Detail", "index": 1, "candidate_role": "text"},
            ]
        },
    }
    Path(dataset["path"]).write_text("Comment,Detail\na,b\n", encoding="utf-8")
    assert charts_for_dataset(dataset, dataset_dir=tmp_path, source_name="d") == []


def test_a_missing_cell_is_a_gap_not_a_zero(tmp_path: Path) -> None:
    """Plotting a missing cell as zero invents a data point and, on a line,
    invents a cliff."""
    path = tmp_path / "gappy.csv"
    path.write_text("Month,Revenue\nJan,100\nFeb,\nMar,300\n", encoding="utf-8")
    dataset = {
        "dataset_id": "ds_g",
        "doc_id": "doc_g",
        "name": "gappy",
        "path": str(path),
        "source_page": 2,
        "profile": {
            "columns": [
                {"name": "Month", "index": 0, "candidate_role": "date"},
                {"name": "Revenue", "index": 1, "candidate_role": "currency"},
            ]
        },
    }
    charts = charts_for_dataset(dataset, dataset_dir=tmp_path, source_name="d")
    points = charts[0].series[0].points
    assert [p["x"] for p in points] == ["Jan", "Mar"]
    assert all(p["y"] != 0 for p in points)


def test_an_extraction_disagreement_is_shown_on_the_chart(tmp_path: Path) -> None:
    """A footnote in another panel does not reach someone looking at a line
    going up."""
    path = tmp_path / "disputed.csv"
    path.write_text("Region,Revenue\nNorth,100\nSouth,200\n", encoding="utf-8")
    dataset = {
        "dataset_id": "ds_d",
        "doc_id": "doc_d",
        "name": "disputed",
        "path": str(path),
        "source_page": 3,
        "agreement": False,
        "profile": {
            "columns": [
                {"name": "Region", "index": 0, "candidate_role": "category"},
                {"name": "Revenue", "index": 1, "candidate_role": "currency"},
            ]
        },
    }
    charts = charts_for_dataset(dataset, dataset_dir=tmp_path, source_name="d")
    assert charts
    assert any("disagreed" in w for w in charts[0].warnings)


def test_a_donut_is_only_offered_for_a_genuine_composition(tmp_path: Path) -> None:
    """Mixed signs are not parts of a whole, and a donut of them is
    meaningless."""
    path = tmp_path / "mixed.csv"
    path.write_text("Line,Amount\nSales,500\nRefunds,-200\nOther,300\n", encoding="utf-8")
    dataset = {
        "dataset_id": "ds_m",
        "doc_id": "doc_m",
        "name": "mixed",
        "path": str(path),
        "source_page": 1,
        "profile": {
            "columns": [
                {"name": "Line", "index": 0, "candidate_role": "category"},
                {"name": "Amount", "index": 1, "candidate_role": "currency"},
            ]
        },
    }
    charts = charts_for_dataset(dataset, dataset_dir=tmp_path, source_name="d")
    assert charts
    assert all(c.kind != "donut" for c in charts)


def test_no_datasets_means_no_charts_rather_than_a_placeholder(
    tmp_path: Path,
) -> None:
    """A document of pure prose produces none, and that is honest rather than
    decorative."""

    class _Empty:
        @staticmethod
        def list_documents(**_kwargs: Any) -> list[dict[str, Any]]:
            return []

        @staticmethod
        def list_datasets(**_kwargs: Any) -> list[dict[str, Any]]:
            return []

    assert (
        build_charts(
            store=_Empty(),
            workspace_id=WORKSPACE,
            corpus_ids=["doc_1"],
            dataset_dir=tmp_path,
        )
        == []
    )


# --------------------------------------------------------------------------- #
# The model cannot supply chart data
# --------------------------------------------------------------------------- #


def test_the_model_is_never_offered_a_charts_field() -> None:
    """The strongest form of the guarantee: not "we overwrite what it sends"
    but "it has nowhere to send it".

    `ExecutiveReportDraft` is what the Synthesizer is asked for, and it has no
    `charts` field, so a fabricated series cannot even be expressed. `Strict`
    forbids extra fields, so an attempt is a validation error rather than
    silently accepted data.
    """
    from app.agents.schemas import ExecutiveReport, ExecutiveReportDraft

    assert "charts" not in ExecutiveReportDraft.model_fields
    assert "charts" not in ExecutiveReportDraft.model_json_schema()["properties"]
    # The final report does carry them -- filled from the extracted tables.
    assert "charts" in ExecutiveReport.model_fields


def test_charts_on_the_report_come_from_the_tables_not_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run whose corpus has no extractable table gets no charts, rather than
    whatever a model might have offered."""
    from app.agents.llm import LLMResult, LLMUsage
    from app.agents.schemas import (
        AgentName,
        Budget,
        Citation,
        Claim,
        ExecutiveReportDraft,
    )
    from app.graph import nodes

    fabricated = ExecutiveReportDraft(
        question="q",
        executive_summary=(
            "A summary long enough to clear the schema floor, which exists to "
            "rule out a one-line report rather than to force padding. The "
            "content is irrelevant to what this test checks."
        ),
        insights=[],
    )

    class _LLM:
        def structured(self, **_kwargs: Any) -> LLMResult:
            return LLMResult(
                text="{}",
                parsed=fabricated,
                usage=LLMUsage(model="stub", provider="stub"),
                stop_reason="stop",
            )

    class _Store:
        @staticmethod
        def list_documents(**_kwargs: Any) -> list[dict[str, Any]]:
            return []

        @staticmethod
        def list_datasets(**_kwargs: Any) -> list[dict[str, Any]]:
            return []

    class _Deps:
        llm = _LLM()
        store = _Store()
        embedder = None
        dataset_dir = "./storage/datasets"

        def event(self, kind: str, payload: dict[str, Any]) -> None:
            return None

    monkeypatch.setattr(nodes, "is_exhausted", lambda _state: False)
    state = {
        "run_id": "r",
        "workspace_id": WORKSPACE,
        "question": "q",
        "corpus_ids": ["doc_1"],
        "research_mode": False,
        "claims": [
            Claim(
                claim_id="clm_1",
                agent=AgentName.FINANCE,
                statement="Revenue was reported for the period.",
                confidence=0.95,
                citations=[Citation(url="https://example.com/a", quote="A" * 20)],
            )
        ],
        "verdicts": [],
        "dropped_claims": [],
        "computations": [],
        "degraded": [],
        "errors": [],
        "injection_attempts": [],
        "budget": Budget(),
        "spend": [],
    }

    result = nodes.synthesize_node(state, _Deps())
    assert result["report"].charts == [], "a chart appeared with no table behind it"


# --------------------------------------------------------------------------- #
# Which charts survive the cap
# --------------------------------------------------------------------------- #


def test_every_table_is_represented_before_any_table_repeats(tmp_path: Path) -> None:
    """The cap must not silently delete a whole source.

    Ranking every chart by size and keeping the top N sounds neutral. It is
    not: one table with many rows outranks several small ones, so a corpus of
    four uploaded files produced figures from one of them and the reader had no
    signal that the other three had any. Their charts were not judged worse —
    they were never compared.

    Here the wide table alone supports more charts than the cap allows, which
    is exactly the case that used to crowd everything else out.
    """
    wide = "quarter,revenue_inr,opex_inr,margin_%,units,headcount\n" + "\n".join(
        f"Q{i} FY2025,{100 + i * 9},{40 + i * 3},{34 - i},{1000 + i * 90},{400 + i * 30}"
        for i in range(1, 9)
    )
    narrow_a = "region,revenue_inr\nNorth,291\nSouth,218\nWest,172\n"
    narrow_b = "month,cost_inr\n2025-01,12\n2025-02,14\n2025-03,17\n"

    store = LocalStore(tmp_path / "resx.db")
    datasets = tmp_path / "datasets"
    doc_ids = []
    try:
        for name, body in (
            ("wide.csv", wide),
            ("narrow_a.csv", narrow_a),
            ("narrow_b.csv", narrow_b),
        ):
            source = tmp_path / name
            source.write_text(body, encoding="utf-8")
            doc_ids.append(
                ingest_file(
                    source, workspace_id=WORKSPACE, store=store, dataset_dir=datasets
                ).doc_id
            )

        charts = build_charts(
            store=store,
            workspace_id=WORKSPACE,
            corpus_ids=doc_ids,
            dataset_dir=datasets,
            max_charts=3,
        )
        assert len(charts) == 3
        sources = {c.source_name for c in charts}
        assert len(sources) == 3, (
            f"only {sorted(sources)} reached the report; a source with no figure "
            f"at all reads as a source with nothing in it"
        )
    finally:
        store.close()


def test_the_strongest_figure_is_still_first(tmp_path: Path) -> None:
    """Round-robin spreads the coverage; it must not bury the lede. The table
    with the most to show still leads."""
    wide = "quarter,revenue_inr\n" + "\n".join(
        f"Q{i} FY2025,{100 + i * 9}" for i in range(1, 13)
    )
    narrow = "region,cost_inr\nNorth,29\nSouth,21\n"

    store = LocalStore(tmp_path / "resx.db")
    datasets = tmp_path / "datasets"
    doc_ids = []
    try:
        for name, body in (("narrow.csv", narrow), ("wide.csv", wide)):
            source = tmp_path / name
            source.write_text(body, encoding="utf-8")
            doc_ids.append(
                ingest_file(
                    source, workspace_id=WORKSPACE, store=store, dataset_dir=datasets
                ).doc_id
            )

        charts = build_charts(
            store=store, workspace_id=WORKSPACE, corpus_ids=doc_ids, dataset_dir=datasets
        )
        assert charts[0].source_name == "wide.csv"
    finally:
        store.close()
