"use client";

import Link from "next/link";
import { BookOpen, Terminal } from "lucide-react";

import { SiteHeader } from "@/components/site-header";
import { PageHeader, Panel } from "@/components/page-shell";

const WORKFLOW = [
  {
    title: "1 · Upload a document",
    body: "Go to Documents and drop in a PDF, spreadsheet or CSV. Extraction, chunking, table typing and indexing all run with no API key. Add a question in the same step to start an analysis the moment ingestion finishes.",
    href: "/documents",
  },
  {
    title: "2 · Check what was extracted",
    body: "Datasets lists every table found, with the scale factor that was applied and whether the two extractors agreed. A disagreement is worth a human read before the numbers are trusted.",
    href: "/datasets",
  },
  {
    title: "3 · Ask without a model",
    body: "Chat returns the source passages that answer a question, with page and paragraph anchors. No model is involved, so nothing can be invented.",
    href: "/chat",
  },
  {
    title: "4 · Run a full analysis",
    body: "This needs a model key. The Manager plans, specialists run in parallel, the Critic re-derives every number, and the Synthesizer writes at most five ranked insights plus stated limitations.",
    href: "/runs/new",
  },
];

const FAQ = [
  {
    q: "Why does a claim show a computation id?",
    a: "Because the model never does arithmetic. Every number is produced by Python in a sandbox and stored as a replayable record; the id is how you check it. A claim carrying a value with no computation id cannot be written to the database at all — that rule is a storage constraint, not a convention.",
  },
  {
    q: "Why did an analysis produce fewer insights than I expected?",
    a: "Anything the corpus does not support is dropped by the grounding gate rather than softened. Fewer claims usually means a stricter run, not a broken one. The report's stated limitations say what could not be established.",
  },
  {
    q: "What does CONTESTED mean?",
    a: "The Critic and the original agent disagreed, and three rounds of debate did not resolve it. Both positions ship. Averaging two irreconcilable readings of a figure would produce a number neither agent believes.",
  },
  {
    q: "Why is retrieval described as a floor?",
    a: "Without a Voyage key the pipeline uses an offline hashing embedder, which matches lexical overlap only. It works, but it cannot match a paraphrase — so measured recall understates what semantic retrieval would achieve.",
  },
  {
    q: "Why do the News and Market agents return nothing?",
    a: "They need a web search key. Without TAVILY_API_KEY their branch is marked degraded and the report discloses it rather than quietly omitting those perspectives.",
  },
  {
    q: "Is my data shared between accounts?",
    a: "No. Each registration creates its own workspace, and every database query is filtered by workspace id — the store refuses to build a query without one. A request for another workspace's document returns 404, not a redacted result.",
  },
];

export default function HelpPage() {
  return (
    <>
      <SiteHeader crumbs={[{ label: "Help" }]} />

      <main className="flex min-w-0 flex-1 flex-col gap-4 p-4">
        <PageHeader
          title="Getting started"
          description="What to do first, and why the system behaves the way it does."
        />

        <div className="grid gap-3 sm:grid-cols-2">
          {WORKFLOW.map((step) => (
            <Link
              key={step.title}
              href={step.href}
              className="flex flex-col gap-1.5 rounded-lg border bg-card p-4 transition-colors hover:bg-accent/40"
            >
              <span className="text-sm font-medium">{step.title}</span>
              <span className="text-xs leading-relaxed text-muted-foreground">
                {step.body}
              </span>
            </Link>
          ))}
        </div>

        <Panel title="Common questions">
          <dl className="flex flex-col divide-y">
            {FAQ.map((entry) => (
              <div key={entry.q} className="flex flex-col gap-1 py-3 first:pt-0">
                <dt className="text-sm font-medium">{entry.q}</dt>
                <dd className="text-xs leading-relaxed text-muted-foreground">
                  {entry.a}
                </dd>
              </div>
            ))}
          </dl>
        </Panel>

        <Panel
          title="From a terminal"
          description="The whole pipeline is usable without the browser, which is also how the accuracy gates are measured."
        >
          <div className="flex flex-col gap-3">
            <pre className="overflow-x-auto rounded bg-muted p-3 font-mono text-2xs leading-relaxed">
              {[
                "# score the pipeline against a corpus with known answers",
                "python benchmarks/gold/generate.py",
                "python benchmarks/score.py --k 10",
                "",
                "# ingest and query your own documents",
                "python scripts/resx.py ingest ./my-documents",
                "python scripts/resx.py status",
                'python scripts/resx.py search "total revenue"',
                "python scripts/resx.py check",
              ].join("\n")}
            </pre>
            <p className="flex items-start gap-1.5 text-2xs leading-relaxed text-muted-foreground">
              <Terminal className="mt-px size-3 shrink-0" />
              Step-by-step instructions, expected output and troubleshooting are
              in <span className="font-mono">RUNBOOK.md</span>.
            </p>
            <p className="flex items-start gap-1.5 text-2xs leading-relaxed text-muted-foreground">
              <BookOpen className="mt-px size-3 shrink-0" />
              The design of record is <span className="font-mono">RESX.md</span>,
              with detail in <span className="font-mono">docs/01</span>–
              <span className="font-mono">09</span> — the accuracy contract is{" "}
              <span className="font-mono">docs/04</span> and the threat model is{" "}
              <span className="font-mono">docs/05</span>.
            </p>
          </div>
        </Panel>
      </main>
    </>
  );
}
