"""Citations derived from a computation's inputs.

The failure these tests lock down was found by running the system on a
four-row CSV. The Finance agent computed total revenue correctly in the
sandbox, then supported the claim by quoting

    North,1200,450000,260000

when the text it had been shown was

    North | 1200 | 450000 | 260000

It reconstructed the CSV it imagined instead of copying what it was given. The
resolver scored that 0.67 against a 0.92 floor, the claim was dropped, and a
correct answer with a correct computation was lost to punctuation. Every claim
in the run went the same way, so the report read "no grounded claims survived"
— indistinguishable, to a user, from a broken product.
"""

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path

import pytest

from app.agents.schemas import Citation
from app.ingest.pipeline import ingest_file
from app.rag.citations import ground_claim, resolve_quote
from app.rag.provenance import (
    derive_citations,
    spans_for_computation,
    spans_for_datasets,
)
from app.store.local import LocalStore

WORKSPACE = "ws_provenance"

CSV = (
    "Region,Units,Revenue,Cost\n"
    "North,1200,450000,260000\n"
    "South,980,367000,215000\n"
    "East,1450,538000,301000\n"
    "West,1100,410000,238000\n"
)

#: 450000 + 367000 + 538000 + 410000
TOTAL_REVENUE = Decimal("1765000")


class FakeRecord:
    """The shape `store.add_computation` expects."""

    def __init__(self, computation_id: str, inputs: list[str], result: object) -> None:
        self.computation_id = computation_id
        self.code = "result = df['Revenue'].sum()"
        self.inputs = inputs
        self.stdout = ""
        self.stderr = ""
        self.result = result
        self.duration_ms = 12
        self.image_digest = "test:sha256:probe"
        self.ok = True
        self.error = None


class FakeClaim:
    """A numeric claim, with whatever citations a test wants to give it."""

    def __init__(self, *, computation_id: str, citations: list[Citation]) -> None:
        self.claim_id = "clm_probe"
        self.agent = "finance"
        self.statement = "Total revenue across all regions was 1,765,000."
        self.value = TOTAL_REVENUE
        self.unit = "USD"
        self.currency = "USD"
        self.period = "FY2025"
        self.scale_factor = "1"
        self.computation_id = computation_id
        self.confidence = 0.99
        self.confidence_reason = None
        self.payload: dict = {}
        self.citations = citations

    @property
    def is_externally_sourced(self) -> bool:
        """Mirrors `Claim`, because the store reads it to set `source_kind`.

        A stand-in for a real model has to keep up with the real model's
        surface; the alternative is the store guessing, and a guess here is a
        claim stored under the wrong rule.
        """
        return bool(self.citations) and all(c.url and not c.doc_id for c in self.citations)


@pytest.fixture
def ingested(tmp_path: Path) -> Iterator[tuple[LocalStore, str, str]]:
    """A store with the CSV ingested and a matching computation recorded."""
    source = tmp_path / "sales_sample.csv"
    source.write_text(CSV, encoding="utf-8")

    store = LocalStore(tmp_path / "resx.db")
    report = ingest_file(
        source,
        workspace_id=WORKSPACE,
        store=store,
        dataset_dir=tmp_path / "datasets",
    )
    assert report.datasets, "the CSV must produce a dataset to walk back to"

    store.add_computation(
        workspace_id=WORKSPACE,
        run_id="run_probe",
        agent="finance",
        record=FakeRecord("cmp_probe", list(report.datasets), str(TOTAL_REVENUE)),
    )
    yield store, report.doc_id, report.datasets[0]
    store.close()


# --------------------------------------------------------------------------- #
# The failure, reproduced
# --------------------------------------------------------------------------- #


def test_the_agents_transcription_really_does_fail(
    ingested: tuple[LocalStore, str, str],
) -> None:
    """Baseline. If this ever passes, the rest of this file is unnecessary."""
    store, doc_id, _ = ingested
    result = resolve_quote(
        quote="North,1200,450000,260000",  # the comma form the model invented
        store=store,
        workspace_id=WORKSPACE,
        doc_id=doc_id,
        page=1,
    )
    assert not result.ok
    assert result.score < 0.92


def test_a_quote_taken_from_the_chunk_resolves_perfectly(
    ingested: tuple[LocalStore, str, str],
) -> None:
    """So the resolver was never the problem — the transcription was."""
    store, doc_id, _ = ingested
    chunk = store.iter_chunks(workspace_id=WORKSPACE, doc_ids=[doc_id])[0]
    result = resolve_quote(
        quote=chunk.text.splitlines()[-1],
        store=store,
        workspace_id=WORKSPACE,
        doc_id=doc_id,
        page=chunk.page,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
    )
    assert result.ok
    assert result.score == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Walking the provenance
# --------------------------------------------------------------------------- #


def test_a_dataset_resolves_to_a_readable_source_span(
    ingested: tuple[LocalStore, str, str],
) -> None:
    store, doc_id, dataset_id = ingested
    spans = spans_for_datasets([dataset_id], store=store, workspace_id=WORKSPACE)
    assert len(spans) == 1
    span = spans[0]
    assert span.doc_id == doc_id
    assert span.text, "a span with no readable text is worse than no citation"
    # Verbatim by construction: sliced from the stored chunk, not transcribed.
    assert (
        span.text in store.get_page_text(workspace_id=WORKSPACE, doc_id=doc_id, page=span.page)
        or "|" in span.text
    )


def test_a_computation_resolves_through_its_declared_inputs(
    ingested: tuple[LocalStore, str, str],
) -> None:
    store, doc_id, _ = ingested
    spans = spans_for_computation("cmp_probe", store=store, workspace_id=WORKSPACE)
    assert spans and spans[0].doc_id == doc_id


def test_a_fabricated_computation_yields_no_provenance(
    ingested: tuple[LocalStore, str, str],
) -> None:
    """This is why the change closes no hole.

    A claim citing a computation that was never run cannot acquire a citation
    from this path, because there is no record to read inputs from.
    """
    store, _, _ = ingested
    assert spans_for_computation("cmp_never_ran", store=store, workspace_id=WORKSPACE) == []


def test_a_computation_over_no_dataset_yields_no_provenance(
    ingested: tuple[LocalStore, str, str],
) -> None:
    """A constant typed into the code has no source, so it gets no citation."""
    store, _, _ = ingested
    store.add_computation(
        workspace_id=WORKSPACE,
        run_id="run_probe",
        agent="finance",
        record=FakeRecord("cmp_constant", [], "42"),
    )
    assert spans_for_computation("cmp_constant", store=store, workspace_id=WORKSPACE) == []


def test_provenance_is_tenant_scoped(ingested: tuple[LocalStore, str, str]) -> None:
    """Another workspace must not be able to walk into this one's documents."""
    store, _, _ = ingested
    assert spans_for_computation("cmp_probe", store=store, workspace_id="ws_other") == []


# --------------------------------------------------------------------------- #
# The fix, end to end
# --------------------------------------------------------------------------- #


def test_a_correct_claim_with_a_bad_quote_is_now_grounded(
    ingested: tuple[LocalStore, str, str],
) -> None:
    """The whole point: right arithmetic, wrong punctuation, claim survives."""
    store, doc_id, _ = ingested

    bad = Citation(
        doc_id=doc_id,
        page=1,
        quote="North,1200,450000,260000",  # what the model actually produced
    )
    claim = FakeClaim(computation_id="cmp_probe", citations=[bad])

    before = ground_claim(
        claim,
        store=store,
        workspace_id=WORKSPACE,
        computation_result=str(TOTAL_REVENUE),
    )
    assert not before.accepted, "the baseline failure must still be a failure"

    derived = derive_citations(
        claim, store=store, workspace_id=WORKSPACE, citation_factory=Citation
    )
    assert derived, "the computation declares a dataset, so provenance exists"
    claim.citations = derived

    after = ground_claim(
        claim,
        store=store,
        workspace_id=WORKSPACE,
        computation_result=str(TOTAL_REVENUE),
    )
    assert after.accepted, after.reason
    assert after.numeric is not None and after.numeric.supported


def test_the_derived_citation_is_labelled_as_derived(
    ingested: tuple[LocalStore, str, str],
) -> None:
    """A reader must be able to tell whose work a quote is."""
    store, doc_id, _ = ingested
    claim = FakeClaim(
        computation_id="cmp_probe",
        citations=[Citation(doc_id=doc_id, page=1, quote="a wrong transcription")],
    )
    derived = derive_citations(
        claim, store=store, workspace_id=WORKSPACE, citation_factory=Citation
    )
    assert all(c.resolution == "derived_from_computation" for c in derived)
    assert all(c.char_start is not None for c in derived), (
        "a derived citation still needs a span, or it cannot be re-checked"
    )


def test_a_derived_citation_is_persistable(
    ingested: tuple[LocalStore, str, str],
) -> None:
    """The new resolution has to pass the store's own constraint.

    A value the database rejects would make the whole path useless at the last
    step, which is the sort of thing that is only found in production.
    """
    store, doc_id, _ = ingested
    claim = FakeClaim(
        computation_id="cmp_probe",
        citations=derive_citations(
            FakeClaim(
                computation_id="cmp_probe",
                citations=[Citation(doc_id=doc_id, page=1, quote="a wrong transcription")],
            ),
            store=store,
            workspace_id=WORKSPACE,
            citation_factory=Citation,
        ),
    )
    store.add_claim(workspace_id=WORKSPACE, run_id="run_probe", claim=claim)

    stored = store.list_claims(workspace_id=WORKSPACE, run_id="run_probe")
    assert len(stored) == 1
    assert stored[0]["citations"][0]["resolution"] == "derived_from_computation"


def test_a_derived_citation_counts_towards_validity(
    ingested: tuple[LocalStore, str, str],
) -> None:
    """It is verifiable, so it must not read as a grounding failure."""
    store, doc_id, _ = ingested
    claim = FakeClaim(
        computation_id="cmp_probe",
        citations=derive_citations(
            FakeClaim(
                computation_id="cmp_probe",
                citations=[Citation(doc_id=doc_id, page=1, quote="a wrong transcription")],
            ),
            store=store,
            workspace_id=WORKSPACE,
            citation_factory=Citation,
        ),
    )
    store.add_claim(workspace_id=WORKSPACE, run_id="run_probe", claim=claim)

    health = store.citation_validity(workspace_id=WORKSPACE)
    assert health["total_citations"] == 1
    assert health["validity"] == 1.0
