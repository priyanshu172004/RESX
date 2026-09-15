"""Gold-dataset scoring harness.

Measures the metrics in `docs/04-ACCURACY-VALIDATION.md` §4 and applies the
gates. This is what turns "the system seems accurate" into a number that can
regress a build.

Metrics that do not require a model are measured here unconditionally:

  * `retrieval_recall_at_k`   — is the gold span in the retrieved set?
  * `citation_validity`       — does every citation resolve to a real span?
  * `resolver_rejection_rate` — are fabricated citations actually caught?
  * `identity_pass_rate`      — do the accounting identities hold on truth?
  * `scale_detection`         — is "in thousands" applied?
  * `numeric_exactness`       — do sandbox computations match ground truth?

The agent-dependent metrics (`hallucination_rate`, `critic_catch_rate`,
`insight_precision`) require a live model and are reported as `not_measured`
rather than defaulted to a passing value. A benchmark that reports a metric it
did not measure is worse than one that admits the gap.

Run:  python benchmarks/score.py [--k 10] [--json]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "api"))

from app.analysis import identities  # noqa: E402
from app.analysis.sandbox import LocalSubprocessSandbox  # noqa: E402
from app.ingest.pipeline import ingest_file  # noqa: E402
from app.rag.citations import (  # noqa: E402
    QUOTE_MATCH_THRESHOLD,
    Resolution,
    check_numeric_support,
    resolve_quote,
)
from app.rag.embeddings import HashingEmbedder, build_embedder  # noqa: E402
from app.rag.retrieve import retrieve  # noqa: E402
from app.store.local import LocalStore  # noqa: E402

GOLD = Path(__file__).parent / "gold"
WORKSPACE = "ws_benchmark"

# The gates from docs/04 §4.
GATES: dict[str, tuple[str, float]] = {
    "citation_validity": ("==", 1.00),
    "numeric_exactness": (">=", 0.98),
    "retrieval_recall_at_k": (">=", 0.95),
    "identity_pass_rate": (">=", 1.00),
    "resolver_rejection_rate": ("==", 1.00),
    "scale_detection": ("==", 1.00),
    "undeclared_scale_flagged": ("==", 1.00),
    "adversarial_segment_trap_caught": ("==", 1.00),
    "anchor_completeness": ("==", 1.00),
}


@dataclass
class Metric:
    name: str
    value: float | None
    detail: str = ""
    measured: bool = True
    numerator: int = 0
    denominator: int = 0

    def gate_status(self) -> str:
        if not self.measured or self.value is None:
            return "NOT MEASURED"
        gate = GATES.get(self.name)
        if gate is None:
            return "—"
        op, threshold = gate
        ok = self.value >= threshold if op == ">=" else abs(self.value - threshold) < 1e-9
        return "PASS" if ok else "FAIL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "measured": self.measured,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "gate": f"{GATES[self.name][0]} {GATES[self.name][1]}" if self.name in GATES else None,
            "status": self.gate_status(),
            "detail": self.detail,
        }


@dataclass
class Scorecard:
    metrics: list[Metric] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add(self, metric: Metric) -> None:
        self.metrics.append(metric)

    @property
    def failures(self) -> list[Metric]:
        return [m for m in self.metrics if m.gate_status() == "FAIL"]

    @property
    def unmeasured(self) -> list[Metric]:
        return [m for m in self.metrics if not m.measured]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": [m.to_dict() for m in self.metrics],
            "gates_failed": [m.name for m in self.failures],
            "not_measured": [m.name for m in self.unmeasured],
            "notes": self.notes,
        }

    def render(self) -> str:
        width = max(len(m.name) for m in self.metrics) + 2
        lines = [
            "",
            "RESX gold-dataset scorecard",
            "=" * 72,
            f"{'metric'.ljust(width)}{'value'.rjust(10)}{'gate'.rjust(12)}{'status'.rjust(14)}",
            "-" * 72,
        ]
        for m in self.metrics:
            gate = f"{GATES[m.name][0]} {GATES[m.name][1]}" if m.name in GATES else "—"
            value = "—" if m.value is None else f"{m.value:.4f}"
            frac = f"  ({m.numerator}/{m.denominator})" if m.denominator else ""
            lines.append(
                f"{m.name.ljust(width)}{value.rjust(10)}{gate.rjust(12)}"
                f"{m.gate_status().rjust(14)}{frac}"
            )
        lines.append("-" * 72)

        for m in self.metrics:
            if m.detail:
                lines.append(f"  {m.name}: {m.detail}")

        if self.notes:
            lines.append("")
            lines.append("Notes")
            for note in self.notes:
                lines.append(f"  - {note}")

        lines.append("")
        if self.failures:
            lines.append(f"RESULT: {len(self.failures)} gate(s) FAILED -> "
                         f"{', '.join(m.name for m in self.failures)}")
        else:
            lines.append("RESULT: all measured gates PASS")
        if self.unmeasured:
            lines.append(
                f"        {len(self.unmeasured)} metric(s) not measured "
                f"({', '.join(m.name for m in self.unmeasured)}) — these need a live model"
            )
        lines.append("")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def score(k: int = 10, *, use_real_embedder: bool = False) -> Scorecard:
    card = Scorecard()

    labels = json.loads((GOLD / "synthetic-pnl" / "labels.json").read_text(encoding="utf-8"))
    pdf = GOLD / "synthetic-pnl" / labels["document"]
    csv_doc = GOLD / "synthetic-pnl" / "monthly_revenue_fy2025.csv"

    if not pdf.exists():
        raise SystemExit(
            f"gold document missing: {pdf}\nRun: python benchmarks/gold/generate.py"
        )

    workdir = Path(tempfile.mkdtemp(prefix="resx-bench-"))
    try:
        store = LocalStore(workdir / "bench.db")

        if use_real_embedder:
            from app.core.config import get_settings

            embedder = build_embedder(get_settings())
        else:
            embedder = HashingEmbedder(dimensions=1024)

        if not embedder.is_semantic:
            card.notes.append(
                "Retrieval was measured with the NON-SEMANTIC hashing embedder. "
                "It captures lexical overlap only, so this recall figure is a "
                "floor, not a prediction of production recall. Re-run with "
                "VOYAGE_API_KEY set and --real-embedder for a production number."
            )

        reports = [
            ingest_file(p, workspace_id=WORKSPACE, store=store, embedder=embedder,
                        dataset_dir=workdir / "datasets")
            for p in (pdf, csv_doc)
        ]
        pdf_report = reports[0]

        # -- anchor completeness ------------------------------------------
        card.add(
            Metric(
                name="anchor_completeness",
                value=pdf_report.anchor_completeness,
                numerator=int(pdf_report.anchor_completeness * pdf_report.chunks),
                denominator=pdf_report.chunks,
                detail=(
                    f"{len(pdf_report.anchor_problems)} anchor problem(s)"
                    if pdf_report.anchor_problems
                    else "every chunk carries a resolvable anchor"
                ),
            )
        )

        # -- scale detection ----------------------------------------------
        datasets = store.list_datasets(workspace_id=WORKSPACE)
        expected_scale = labels["scale_factor"]
        monetary = [d for d in datasets if d["source_page"] in (2, 3, 4, 5)]
        correct_scale = sum(1 for d in monetary if d["scale_factor"] == expected_scale)
        card.add(
            Metric(
                name="scale_detection",
                value=(correct_scale / len(monetary)) if monetary else None,
                numerator=correct_scale,
                denominator=len(monetary),
                detail=(
                    f"'{labels['scale_phrase']}' -> x{expected_scale} applied to "
                    f"{correct_scale}/{len(monetary)} monetary tables"
                ),
            )
        )

        # -- retrieval recall@k -------------------------------------------
        questions = labels["questions"]
        hits = 0
        misses: list[str] = []
        for q in questions:
            result = retrieve(
                q["question"], store=store, embedder=embedder,
                workspace_id=WORKSPACE, top_k=k,
            )
            pages = {c.page for c in result.chunks}
            if q["gold_page"] in pages:
                hits += 1
            else:
                misses.append(f"{q['id']} (wanted p{q['gold_page']}, got {sorted(pages)})")

        card.add(
            Metric(
                name="retrieval_recall_at_k",
                value=hits / len(questions),
                numerator=hits,
                denominator=len(questions),
                detail=(f"k={k}; missed: {', '.join(misses)}" if misses
                        else f"k={k}; every gold page retrieved"),
            )
        )

        # -- citation validity and resolver rejection ---------------------
        # A genuine quote must resolve, and a fabricated one must not. Both
        # halves matter: a resolver that accepts everything scores 1.00 on
        # validity while providing no protection at all.
        genuine = [
            (2, "Revenue 48,920 43,485"),
            (5, "One customer accounted for 23.0% of total revenue"),
            (4, "Cash at end of period"),
            (3, "Total assets"),
        ]
        resolved = 0
        for page, quote in genuine:
            r = resolve_quote(
                store=store, workspace_id=WORKSPACE, doc_id=pdf_report.doc_id,
                page=page, quote=quote,
            )
            if r.ok:
                resolved += 1
        card.add(
            Metric(
                name="citation_validity",
                value=resolved / len(genuine),
                numerator=resolved,
                denominator=len(genuine),
                detail=f"{resolved}/{len(genuine)} genuine quotes resolved",
            )
        )

        fabricated = [
            (2, "Revenue was 99,999 thousand and all risks were eliminated"),
            (2, "The board approved a special dividend of 12,000 thousand"),
            (99, "Revenue 48,920 43,485"),  # page does not exist
            (5, "The largest customer represented 71.4% of revenue"),
        ]
        rejected = 0
        rejection_detail: list[str] = []
        for page, quote in fabricated:
            r = resolve_quote(
                store=store, workspace_id=WORKSPACE, doc_id=pdf_report.doc_id,
                page=page, quote=quote,
            )
            if not r.ok:
                rejected += 1
            else:
                rejection_detail.append(f"p{page} '{quote[:40]}' scored {r.score}")
        card.add(
            Metric(
                name="resolver_rejection_rate",
                value=rejected / len(fabricated),
                numerator=rejected,
                denominator=len(fabricated),
                detail=(
                    f"{rejected}/{len(fabricated)} fabricated quotes rejected at "
                    f"threshold {QUOTE_MATCH_THRESHOLD}"
                    + ("; LEAKED: " + "; ".join(rejection_detail) if rejection_detail else "")
                ),
            )
        )

        # -- accounting identities on ground truth ------------------------
        figures = labels["figures"]
        cy = figures["current_year"]
        report = identities.check_all(
            {
                "income_statement": {
                    "revenue": cy["revenue"],
                    "cogs": cy["cogs"],
                    "gross_profit": cy["gross_profit"],
                    "opex": cy["opex"],
                    "operating_income": cy["operating_income"],
                    "other_income": cy["other_income"],
                    "tax": cy["tax"],
                    "net_income": cy["net_income"],
                },
                "balance_sheet": figures["balance_sheet"],
                "cash_flow": figures["cash_flow"],
                "segments": {
                    "segment_values": list(figures["segments"].values()),
                    "total": cy["revenue"],
                },
            }
        )
        card.add(
            Metric(
                name="identity_pass_rate",
                value=report.pass_rate,
                numerator=sum(1 for c in report.checks if c.passed),
                denominator=len(report.checks),
                detail=report.summary(),
            )
        )

        # -- numeric exactness via the sandbox ----------------------------
        # Every figure is computed in the sandbox from the registered dataset
        # and compared to ground truth. This is the metric that would catch a
        # 1000x scale error end to end.
        sandbox = LocalSubprocessSandbox(wall_timeout_seconds=30)
        csv_dataset = next(
            d for d in store.list_datasets(workspace_id=WORKSPACE)
            if d["name"].startswith("csv_")
        )
        dataset_dir = Path(csv_dataset["path"]).parent

        csv_labels = json.loads(
            (GOLD / "synthetic-pnl" / "monthly_revenue_fy2025.labels.json").read_text(
                encoding="utf-8"
            )
        )
        # The CSV declares no scale, so face value is the correct reading.
        # Expecting the x1000 figure here would be asserting that the system
        # should *guess* a multiplier — the exact behaviour the design forbids.
        expected_total = Decimal(str(csv_labels["monthly_revenue_total_facevalue"]))

        code = (
            "from decimal import Decimal\n"
            f"df = resx.load('{csv_dataset['dataset_id']}')\n"
            "total = sum(Decimal(str(v)) for v in df['revenue'])\n"
            "gross = sum(Decimal(str(v)) for v in df['revenue']) - "
            "sum(Decimal(str(v)) for v in df['cogs'])\n"
            "result = {'revenue_total': total, 'gross_profit': gross, 'n': int(len(df))}"
        )
        record = sandbox.run(code, inputs=[csv_dataset["dataset_id"]], dataset_dir=dataset_dir)

        checks: list[tuple[str, bool, str]] = []
        if not record.ok:
            checks.append(("sandbox_execution", False, record.error or "failed"))
        else:
            got = Decimal(str(record.result["revenue_total"]))
            within = abs(got - expected_total) / expected_total <= Decimal("0.005")
            checks.append(
                (
                    "monthly_revenue_total",
                    bool(within),
                    f"computed {got} vs truth {expected_total}",
                )
            )
            checks.append(
                ("row_count", record.result["n"] == 12, f"n={record.result['n']}")
            )

            support = check_numeric_support(
                value=got,
                quotes=["Revenue 48,920 43,485"],
                computation_result=record.result,
                scale_factor=Decimal(labels["scale_factor"]),
            )
            checks.append(
                ("numeric_support_gate", support.supported, support.detail)
            )

            # An unsupported figure must be *rejected*. Testing only the happy
            # path would let a resolver that accepts everything score 1.00.
            fabricated_support = check_numeric_support(
                value=Decimal("99999000"),
                quotes=["Revenue 48,920 43,485"],
                computation_result=record.result,
                scale_factor=Decimal(labels["scale_factor"]),
            )
            checks.append(
                (
                    "numeric_support_rejects_unsupported",
                    not fabricated_support.supported,
                    "fabricated 99,999,000 correctly unsupported"
                    if not fabricated_support.supported
                    else "LEAKED: fabricated figure was accepted as supported",
                )
            )

        passed = sum(1 for _, ok, _ in checks if ok)
        card.add(
            Metric(
                name="numeric_exactness",
                value=passed / len(checks),
                numerator=passed,
                denominator=len(checks),
                detail="; ".join(f"{n}={'ok' if ok else 'FAIL'} ({d})" for n, ok, d in checks),
            )
        )

        # -- undeclared scale must be flagged, never inferred -------------
        # The complement of scale_detection. When a document *does* declare a
        # scale it must be applied; when it does not, the system must read face
        # value and say so. Silently inferring x1000 because the numbers "look
        # like thousands" is the 1000x error, arrived at politely.
        if csv_labels.get("expect_undeclared_scale_flag"):
            issues = csv_dataset["profile"].get("issues", [])
            flagged = any(i["kind"] == "undeclared_scale" for i in issues)
            face_value = csv_dataset["scale_factor"] == "1"
            card.add(
                Metric(
                    name="undeclared_scale_flagged",
                    value=1.0 if (flagged and face_value) else 0.0,
                    numerator=int(flagged and face_value),
                    denominator=1,
                    detail=(
                        "no scale declared: read at face value and flagged for "
                        "confirmation"
                        if (flagged and face_value)
                        else f"flagged={flagged}, scale_factor="
                        f"{csv_dataset['scale_factor']} (expected '1')"
                    ),
                )
            )

        # -- adversarial corpus: the segment trap must be caught ----------
        adv_labels_path = GOLD / "adversarial" / "labels.json"
        if adv_labels_path.exists():
            trap = identities.check_segments(
                segment_values=["22310000", "15845000"], total="48920000"
            )
            card.add(
                Metric(
                    name="adversarial_segment_trap_caught",
                    value=0.0 if trap.passed else 1.0,
                    numerator=0 if trap.passed else 1,
                    denominator=1,
                    detail=(
                        "planted non-summing segments were correctly rejected"
                        if not trap.passed
                        else "TRAP MISSED: non-summing segments were accepted"
                    ),
                )
            )

        # -- metrics that need a live model -------------------------------
        for name, why in (
            ("hallucination_rate", "requires agent output over the gold corpus"),
            ("critic_catch_rate", "requires the Critic to run against seeded errors"),
            ("insight_precision", "requires expert rating of the top-5 insights"),
            ("calibration_error", "requires stated-vs-observed confidence over many runs"),
        ):
            card.add(Metric(name=name, value=None, measured=False, detail=why))

        card.notes.append(
            f"embedder: {embedder.model_name} (semantic={embedder.is_semantic})"
        )
        card.notes.append(
            f"sandbox: {sandbox.image_digest} "
            f"(security_boundary={sandbox.is_security_boundary})"
        )
        return card
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score RESX against the gold dataset")
    parser.add_argument("--k", type=int, default=10, help="retrieval depth for recall@k")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument(
        "--real-embedder",
        action="store_true",
        help="use the configured production embedder instead of the offline fallback",
    )
    args = parser.parse_args()

    card = score(k=args.k, use_real_embedder=args.real_embedder)

    if args.json:
        print(json.dumps(card.to_dict(), indent=2))
    else:
        print(card.render())

    return 1 if card.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
