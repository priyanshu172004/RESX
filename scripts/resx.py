#!/usr/bin/env python
"""RESX command line.

The whole pipeline without a browser. Useful on its own and it is the honest
demonstration of what works with and without a model:

    ingest    documents -> anchors -> chunks -> embeddings -> datasets
    search    hybrid retrieval with citations
    compute   run Python in the sandbox against a registered dataset
    check     accounting identities on a dataset
    analyse   the full multi-agent run  (needs GROQ_API_KEY or ANTHROPIC_API_KEY)
    reset     delete a workspace's corpus data
    status    what is in the workspace
    bench     the gold-dataset scorecard

Everything except `analyse` runs with no API key at all.

Examples
--------
    python scripts/resx.py ingest benchmarks/gold/synthetic-pnl
    python scripts/resx.py status
    python scripts/resx.py search "what was total revenue in FY2025"
    python scripts/resx.py compute --dataset ds_xxx --sum revenue
    python scripts/resx.py analyse "Is this business profitable and risky?"
"""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "api"))

from app.analysis import identities  # noqa: E402
from app.analysis.sandbox import build_sandbox  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.ingest.pipeline import ingest_directory, ingest_file  # noqa: E402
from app.rag.embeddings import build_embedder  # noqa: E402
from app.rag.retrieve import retrieve  # noqa: E402
from app.store.factory import build_store  # noqa: E402

DEFAULT_WORKSPACE = "ws_dev"


def _open_store() -> tuple[object, object]:
    """Open the same store the API uses.

    Routed through the factory rather than constructing SQLite directly, so
    `resx.py status` cannot describe a different database than the dashboard —
    which it would the moment MONGODB_URL was set.
    """
    settings = get_settings()
    settings.storage_root.mkdir(parents=True, exist_ok=True)
    return build_store(settings), settings


def _rule(title: str) -> None:
    print(f"\n{title}\n" + "-" * max(24, len(title)))


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_ingest(args: argparse.Namespace) -> int:
    store, settings = _open_store()
    embedder = build_embedder(settings)
    dataset_dir = settings.dataset_dir

    target = Path(args.path)
    if not target.exists():
        print(f"no such path: {target}", file=sys.stderr)
        return 2

    reports = (
        ingest_directory(
            target,
            workspace_id=args.workspace,
            store=store,
            embedder=embedder,
            dataset_dir=dataset_dir,
        )
        if target.is_dir()
        else [
            ingest_file(
                target,
                workspace_id=args.workspace,
                store=store,
                embedder=embedder,
                dataset_dir=dataset_dir,
            )
        ]
    )

    if not reports:
        print("nothing ingested (no supported files found)")
        return 1

    _rule(f"Ingested {len(reports)} document(s)")
    for report in reports:
        print(f"  {report.summary()}")
        print(f"    doc_id: {report.doc_id}")
        for dataset_id in report.datasets:
            print(f"    dataset: {dataset_id}")
        for warning in report.warnings[:4]:
            print(f"    ! {warning}")
        if report.anchor_problems:
            print(f"    ! {len(report.anchor_problems)} anchor problem(s)")

    print(f"\nembedder: {embedder.model_name} (semantic={embedder.is_semantic})")
    if not embedder.is_semantic:
        print(
            "  note: this is the offline lexical fallback. Set VOYAGE_API_KEY for\n"
            "        semantic retrieval; recall measured here is a floor."
        )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    store, settings = _open_store()

    documents = store.list_documents(workspace_id=args.workspace)
    datasets = store.list_datasets(workspace_id=args.workspace)
    runs = store.list_runs(workspace_id=args.workspace, limit=10)
    grounding = store.citation_validity(workspace_id=args.workspace)

    _rule(f"Workspace {args.workspace}")
    print(f"  documents : {len(documents)}")
    print(f"  datasets  : {len(datasets)}")
    print(f"  runs      : {len(runs)}")
    print(
        f"  grounding : {grounding['validity']:.2%} "
        f"({grounding['resolved']}/{grounding['total_citations']} citations resolved)"
    )

    if documents:
        _rule("Documents")
        for d in documents:
            flag = " [NEEDS OCR]" if d["low_conf_pages"] else ""
            print(
                f"  {d['doc_id']}  {d['source_name'][:38]:<38} "
                f"{d['kind']:<5} {d['page_count']:>4}p  {d['status']}{flag}"
            )

    if datasets:
        _rule("Datasets")
        for ds in datasets:
            note = ""
            if ds["agreement"] is False:
                note = "  ! extractors disagreed"
            print(
                f"  {ds['dataset_id']:<38} {ds['n_rows']:>4}x{ds['n_cols']:<3} "
                f"scale=x{ds['scale_factor']:<8} {ds['currency'] or '?':<4}{note}"
            )

    if runs:
        _rule("Recent runs")
        for run in runs:
            print(
                f"  {run['run_id']}  {run['status']:<10} "
                f"${run['usd_used']:.4f}  {run['question'][:44]}"
            )

    print()
    print(f"  model provider: {settings.provider}")
    print(
        f"  model key configured: {settings.has_model_key}"
        f"  ({settings.model_key_env_var})"
    )
    print(f"  search key configured: {bool(settings.tavily_api_key)}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    store, settings = _open_store()
    embedder = build_embedder(settings)

    result = retrieve(
        args.query,
        store=store,
        embedder=embedder,
        workspace_id=args.workspace,
        top_k=args.top_k,
    )

    if not result.chunks:
        print("no matching content — is anything ingested? try: resx.py status")
        return 1

    diagnostics = result.diagnostics
    _rule(f"{len(result.chunks)} result(s) for {args.query!r}")
    if diagnostics:
        print(
            f"  lexical={diagnostics.lexical_hits} vector={diagnostics.vector_hits} "
            f"fused={diagnostics.fused_candidates} semantic={diagnostics.semantic}\n"
        )

    for chunk in result.chunks:
        head = " / ".join(chunk.text.splitlines())[:130]
        print(
            f"  [{chunk.score:.3f} {chunk.source:<7} {chunk.kind:<5}] "
            f"{chunk.doc_id} {chunk.citation_label()}"
        )
        print(f"      {head}")
    return 0


def cmd_compute(args: argparse.Namespace) -> int:
    """Run a computation in the sandbox — the only way a number is produced."""
    store, settings = _open_store()
    sandbox = build_sandbox(settings)

    datasets = {d["dataset_id"]: d for d in store.list_datasets(workspace_id=args.workspace)}
    if args.dataset not in datasets:
        print(f"no such dataset: {args.dataset}", file=sys.stderr)
        print("available:", ", ".join(sorted(datasets)) or "(none)", file=sys.stderr)
        return 2

    dataset_dir = Path(datasets[args.dataset]["path"]).parent

    if args.code:
        code = args.code
    elif args.sum:
        code = (
            "from decimal import Decimal\n"
            f"df = resx.load('{args.dataset}')\n"
            f"col = '{args.sum}'\n"
            "total = sum(Decimal(str(v)) for v in df[col] if str(v).strip() not in ('', 'nan'))\n"
            "result = {'column': col, 'total': total, 'n': int(len(df))}"
        )
    else:
        code = (
            f"df = resx.load('{args.dataset}')\n"
            "result = {'columns': list(df.columns), 'n': int(len(df)), "
            "'head': df.head(3).to_dict('records')}"
        )

    record = sandbox.run(code, inputs=[args.dataset], dataset_dir=dataset_dir)

    _rule("Computation")
    print(f"  computation_id : {record.computation_id}")
    print(f"  sandbox        : {record.image_digest}")
    print(f"  security bound : {sandbox.is_security_boundary}")
    print(f"  ok             : {record.ok}  ({record.duration_ms}ms)")
    if record.stdout.strip():
        print(f"  stdout         : {record.stdout.strip()[:400]}")
    if record.error:
        print(f"  error          : {record.error}")
    print(f"  result         : {record.result}")

    store.add_computation(
        workspace_id=args.workspace, run_id=None, agent="cli", record=record
    )
    return 0 if record.ok else 1


def cmd_check(args: argparse.Namespace) -> int:
    """Assert the accounting identities on named figures."""
    figures = {}
    for pair in args.figure or []:
        if "=" not in pair:
            print(f"expected name=value, got {pair!r}", file=sys.stderr)
            return 2
        name, raw = pair.split("=", 1)
        figures[name.strip().lower()] = Decimal(raw.strip())

    statements: dict[str, dict[str, object]] = {}
    if {"revenue", "cogs", "gross_profit"} <= figures.keys():
        income = {
            "revenue": figures["revenue"],
            "cogs": figures["cogs"],
            "gross_profit": figures["gross_profit"],
        }
        if {"opex", "operating_income"} <= figures.keys():
            income["opex"] = figures["opex"]
            income["operating_income"] = figures["operating_income"]
        statements["income_statement"] = income
    if {"assets", "liabilities", "equity"} <= figures.keys():
        statements["balance_sheet"] = {
            k: figures[k] for k in ("assets", "liabilities", "equity")
        }
    if {"opening_cash", "net_cash_flow", "closing_cash"} <= figures.keys():
        statements["cash_flow"] = {
            k: figures[k] for k in ("opening_cash", "net_cash_flow", "closing_cash")
        }

    if not statements:
        print(
            "not enough figures to check anything. Supply at least "
            "revenue, cogs and gross_profit.",
            file=sys.stderr,
        )
        return 2

    report = identities.check_all(statements)
    _rule("Accounting identities")
    for check in report.checks:
        mark = "PASS" if check.passed else "FAIL"
        print(f"  [{mark}] {check.name}: {check.message}")
    print()
    print(f"  {report.summary()}")
    return 0 if report.passed else 1


def cmd_analyse(args: argparse.Namespace) -> int:
    store, settings = _open_store()

    if not settings.has_model_key:
        print(
            f"{settings.model_key_env_var} is not set, so the agents cannot run.\n"
            "Everything else works without it — try:\n"
            "  python scripts/resx.py search \"total revenue\"\n"
            "  python scripts/resx.py compute --dataset <id> --sum revenue\n"
            "  python benchmarks/score.py",
            file=sys.stderr,
        )
        return 2

    from app.services.runs import RunService

    ready = [
        d["doc_id"]
        for d in store.list_documents(workspace_id=args.workspace)
        if d["status"].startswith("ready")
    ]
    if not ready:
        print("no ingested documents; run `resx.py ingest <path>` first", file=sys.stderr)
        return 2

    service = RunService(store=store, settings=settings)
    ctx = service.create(
        workspace_id=args.workspace, question=args.question, corpus_ids=ready
    )

    _rule(f"Run {ctx.run_id}")
    print(f"  question : {args.question}")
    print(f"  corpus   : {len(ready)} document(s)")
    print(f"  cap      : ${settings.run_max_usd:.2f}\n")

    # Executed inline so the console streams the run as it happens.
    outcome = service.execute(ctx)

    for event in store.read_events(workspace_id=args.workspace, run_id=ctx.run_id):
        payload = event["payload"]
        detail = payload.get("statement") or payload.get("node") or payload.get("tool") or ""
        print(f"  [{event['seq']:>3}] {event['kind']:<18} {str(detail)[:88]}")

    run = store.get_run(workspace_id=args.workspace, run_id=ctx.run_id)
    report = (run or {}).get("report")

    _rule("Outcome")
    print(json.dumps(outcome, indent=2, default=str))

    if report:
        _rule("Executive insights")
        for i, insight in enumerate(report.get("insights", []), start=1):
            print(f"  {i}. {insight['headline']}")
            print(f"     {insight['so_what'][:200]}")
            print(
                f"     confidence={insight['confidence']} owner={insight['suggested_owner']} "
                f"effort={insight['effort']}"
            )
        if report.get("limitations"):
            _rule("Limitations")
            for item in report["limitations"]:
                print(f"  - {item}")

    return 0 if outcome.get("status") == "done" else 1


def cmd_reset(args: argparse.Namespace) -> int:
    """Delete every corpus artefact in a workspace.

    Requires the literal confirmation rather than a bare flag: this removes
    documents, datasets, runs, claims and computations, and a mistyped command
    should not be enough to do it.
    """
    store, settings = _open_store()

    if args.confirm != "DELETE":
        print(
            "This deletes every document, dataset, run, claim and computation "
            f"in workspace {args.workspace!r}, and removes the extracted "
            "dataset files from disk. User accounts are kept.\n\n"
            "Re-run with --confirm DELETE to proceed.",
            file=sys.stderr,
        )
        return 2

    counts = store.wipe_workspace(workspace_id=args.workspace)

    # The dataset files too. A stale parquet left behind would still answer a
    # computation for a document that no longer exists.
    removed = 0
    dataset_dir = settings.dataset_dir
    if dataset_dir.is_dir():
        for path in dataset_dir.iterdir():
            if path.is_file():
                path.unlink()
                removed += 1

    _rule(f"Reset {args.workspace}")
    for name, count in sorted(counts.items()):
        if count:
            print(f"  {name:<14} {count:>6} deleted")
    print(f"  {'dataset files':<14} {removed:>6} deleted")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(ROOT))
    from benchmarks.score import score

    card = score(k=args.k)
    print(card.render())
    return 1 if card.failures else 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="resx",
        description="RESX command line — ingest, retrieve, compute, analyse.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--workspace", default=DEFAULT_WORKSPACE, help="workspace id (default: ws_dev)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="ingest a file or directory")
    p_ingest.add_argument("path")
    p_ingest.set_defaults(func=cmd_ingest)

    p_status = sub.add_parser("status", help="what is in the workspace")
    p_status.set_defaults(func=cmd_status)

    p_search = sub.add_parser("search", help="hybrid retrieval with citations")
    p_search.add_argument("query")
    p_search.add_argument("--top-k", type=int, default=8)
    p_search.set_defaults(func=cmd_search)

    p_compute = sub.add_parser("compute", help="run Python in the sandbox")
    p_compute.add_argument("--dataset", required=True)
    p_compute.add_argument("--sum", help="sum this column exactly, using Decimal")
    p_compute.add_argument("--code", help="arbitrary Python; assign to `result`")
    p_compute.set_defaults(func=cmd_compute)

    p_check = sub.add_parser("check", help="assert accounting identities")
    p_check.add_argument(
        "-f", "--figure", action="append", metavar="NAME=VALUE",
        help="e.g. -f revenue=48920000 -f cogs=28471000 -f gross_profit=20449000",
    )
    p_check.set_defaults(func=cmd_check)

    p_analyse = sub.add_parser("analyse", help="full multi-agent run (needs a model key)")
    p_analyse.add_argument("question")
    p_analyse.set_defaults(func=cmd_analyse)

    p_reset = sub.add_parser("reset", help="delete a workspace's corpus data")
    p_reset.add_argument("--confirm", default="", help="must be the string DELETE")
    p_reset.set_defaults(func=cmd_reset)

    p_bench = sub.add_parser("bench", help="gold-dataset scorecard")
    p_bench.add_argument("--k", type=int, default=10)
    p_bench.set_defaults(func=cmd_bench)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
