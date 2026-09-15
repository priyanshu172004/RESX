"use client";

/**
 * The dashboard, backed by `/api/v1/dashboard`.
 *
 * Every figure here comes from the API. Where the workspace has no data the
 * page says so and points at the next action, rather than rendering a chart of
 * zeroes — a chart with no data in it looks like a broken chart, and a chart of
 * invented data is worse than either.
 */

import Link from "next/link";
import {
  Activity,
  Database,
  FileStack,
  Layers,
  Lightbulb,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  SquareFunction,
  TriangleAlert,
  Upload,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { SiteHeader } from "@/components/site-header";
import {
  EmptyState,
  ErrorState,
  LoadingBlock,
  PageHeader,
  Panel,
} from "@/components/page-shell";
import { SpendArea } from "@/components/charts/spend-area";
import { AgentClaimsBars } from "@/components/charts/agent-claims-bars";
import { CorpusDonut } from "@/components/charts/corpus-donut";
import { CapabilityBanner } from "@/components/capability-banner";
import { useDashboard } from "@/hooks/use-workspace-counts";
import { formatCompactNumber, formatNumber } from "@/lib/format";
import type { Kpi } from "@/lib/api";

function formatKpi(kpi: Kpi): string {
  switch (kpi.format) {
    case "percent":
      return `${(kpi.value * 100).toFixed(kpi.value === 1 ? 0 : 1)}%`;
    case "usd":
      // Four decimals: a Groq run costs fractions of a cent, and rounding to
      // two would report every one of them as $0.00.
      return `$${kpi.value.toFixed(kpi.value > 0 && kpi.value < 0.01 ? 4 : 2)}`;
    case "ratio":
      return kpi.value.toFixed(2);
    default:
      return formatNumber(kpi.value);
  }
}

/** Below target is a failure to show, not a number to soften. */
function kpiTone(kpi: Kpi): "good" | "critical" | null {
  if (kpi.target == null) return null;
  return kpi.value >= kpi.target ? "good" : "critical";
}

export default function DashboardPage() {
  const { dashboard, loading, error, refresh } = useDashboard();

  return (
    <>
      <SiteHeader crumbs={[{ label: "Dashboard" }]} />

      <main className="flex min-w-0 flex-1 flex-col gap-4 p-4">
        <PageHeader
          title="Workspace overview"
          description="Counts and rates over what is actually stored. Financial figures live on individual claims, each carrying the computation that produced it."
          actions={
            <>
              <Button size="sm" variant="outline" onClick={() => void refresh()}>
                <RefreshCw className="size-3.5" />
                Refresh
              </Button>
              <Button size="sm" asChild>
                <Link href="/runs/new">
                  <Sparkles className="size-3.5" />
                  New analysis
                </Link>
              </Button>
            </>
          }
        />

        {error != null && <ErrorState error={error} onRetry={() => void refresh()} />}

        {loading && !dashboard && <LoadingBlock rows={4} />}

        {dashboard && (
          <>
            <CapabilityBanner capabilities={dashboard.capabilities} />

            {dashboard.empty_reason && (
              <EmptyState
                icon={Upload}
                title={
                  dashboard.summary.documents === 0
                    ? "No documents yet"
                    : "Nothing analysed yet"
                }
                description={
                  dashboard.summary.documents === 0
                    ? "Upload a PDF, spreadsheet or CSV. Extraction, chunking and retrieval all run without an API key."
                    : "Your corpus is indexed. Ask a question to produce a cited report."
                }
                action={
                  dashboard.summary.documents === 0
                    ? { label: "Upload a document", href: "/documents" }
                    : { label: "Ask a question", href: "/runs/new" }
                }
              />
            )}

            {/* -- corpus strip ------------------------------------------ */}
            <section
              aria-label="Corpus summary"
              className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border bg-border lg:grid-cols-4"
            >
              {[
                {
                  label: "Documents",
                  value: formatNumber(dashboard.summary.documents),
                  icon: FileStack,
                },
                {
                  label: "Pages",
                  value: formatNumber(dashboard.summary.pages),
                  icon: Layers,
                },
                {
                  label: "Chunks indexed",
                  value: formatCompactNumber(dashboard.summary.chunks),
                  icon: SquareFunction,
                },
                {
                  label: "Citation validity",
                  value: `${(dashboard.summary.citation_validity * 100).toFixed(0)}%`,
                  icon: ShieldCheck,
                },
              ].map((stat) => (
                <div
                  key={stat.label}
                  className="flex items-center gap-2.5 bg-card px-4 py-3"
                >
                  <stat.icon
                    className="size-4 shrink-0 text-muted-foreground"
                    aria-hidden
                  />
                  <div className="flex min-w-0 flex-col">
                    <span
                      data-slot="kpi-value"
                      className="text-base leading-none font-semibold"
                    >
                      {stat.value}
                    </span>
                    <span className="truncate text-2xs text-muted-foreground">
                      {stat.label}
                    </span>
                  </div>
                </div>
              ))}
            </section>

            {/* -- KPI row ----------------------------------------------- */}
            <section
              aria-label="Key indicators"
              className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6"
            >
              {dashboard.kpis.map((kpi) => {
                const tone = kpiTone(kpi);
                return (
                  <div
                    key={kpi.key}
                    className="flex flex-col gap-1 rounded-lg border bg-card p-3"
                  >
                    <span className="text-2xs uppercase tracking-wide text-muted-foreground">
                      {kpi.label}
                    </span>
                    <span
                      data-slot="kpi-value"
                      className="text-xl font-semibold leading-none tracking-tight"
                    >
                      {formatKpi(kpi)}
                    </span>
                    <span className="flex items-center gap-1.5 text-2xs text-muted-foreground">
                      {tone === "critical" && (
                        <TriangleAlert className="size-3 text-critical" />
                      )}
                      {tone === "good" && (
                        <ShieldCheck className="size-3 text-good" />
                      )}
                      <span className="truncate">{kpi.detail}</span>
                    </span>
                  </div>
                );
              })}
            </section>

            {/* -- charts ------------------------------------------------ */}
            <section
              aria-label="Activity"
              className="grid gap-4 xl:grid-cols-[1.4fr_1fr]"
            >
              <SpendArea points={dashboard.spend} />
              <CorpusDonut datasets={dashboard.datasets} />
            </section>

            {dashboard.agent_activity.length > 0 && (
              <section aria-label="Agent activity">
                <AgentClaimsBars activity={dashboard.agent_activity} />
              </section>
            )}

            {/* -- recent runs ------------------------------------------- */}
            <Panel
              title="Recent analyses"
              description="Each run's status and what it cost."
              actions={
                <Button size="sm" variant="ghost" asChild>
                  <Link href="/runs">View all</Link>
                </Button>
              }
            >
              {dashboard.runs.length === 0 ? (
                <p className="text-xs text-muted-foreground">
                  No runs yet.{" "}
                  <Link href="/runs/new" className="underline underline-offset-4">
                    Ask a question
                  </Link>{" "}
                  to produce one.
                </p>
              ) : (
                <ul className="flex flex-col divide-y">
                  {dashboard.runs.slice(0, 6).map((run) => (
                    <li key={run.run_id}>
                      <Link
                        href={`/runs/${run.run_id}`}
                        className="flex flex-wrap items-center gap-3 py-2.5 transition-colors hover:bg-accent/40"
                      >
                        <Activity className="size-3.5 shrink-0 text-muted-foreground" />
                        <span className="min-w-0 flex-1 truncate text-sm">
                          {run.question}
                        </span>
                        <Badge
                          variant={
                            run.status === "completed"
                              ? "secondary"
                              : run.status === "failed"
                                ? "destructive"
                                : "outline"
                          }
                          className="text-2xs"
                        >
                          {run.status}
                        </Badge>
                        <span className="tabular text-2xs text-muted-foreground">
                          ${run.usd_used.toFixed(4)}
                        </span>
                      </Link>
                    </li>
                  ))}
                </ul>
              )}
            </Panel>

            {/* -- documents -------------------------------------------- */}
            <Panel
              title="Corpus"
              description="Documents currently indexed in this workspace."
              actions={
                <Button size="sm" variant="ghost" asChild>
                  <Link href="/documents">Manage</Link>
                </Button>
              }
            >
              {dashboard.documents.length === 0 ? (
                <p className="text-xs text-muted-foreground">
                  Nothing ingested yet.
                </p>
              ) : (
                <ul className="flex flex-col divide-y">
                  {dashboard.documents.map((doc) => (
                    <li
                      key={doc.doc_id}
                      className="flex flex-wrap items-center gap-3 py-2.5"
                    >
                      <FileStack className="size-3.5 shrink-0 text-muted-foreground" />
                      <span className="min-w-0 flex-1 truncate text-sm">
                        {doc.source_name}
                      </span>
                      <span className="tabular text-2xs text-muted-foreground">
                        {doc.page_count} pages
                      </span>
                      {doc.warnings.length > 0 && (
                        <Badge variant="outline" className="gap-1 text-2xs">
                          <TriangleAlert className="size-2.5 text-warning" />
                          {doc.warnings.length}
                        </Badge>
                      )}
                      <Badge variant="secondary" className="text-2xs">
                        {doc.kind}
                      </Badge>
                    </li>
                  ))}
                </ul>
              )}
            </Panel>

            <section className="grid gap-4 sm:grid-cols-3">
              {[
                {
                  href: "/documents",
                  icon: Upload,
                  title: "Upload and analyse",
                  body: "Drop a document in and ask a question in the same step.",
                },
                {
                  href: "/insights",
                  icon: Lightbulb,
                  title: "Browse claims",
                  body: "Every finding with its citation and computation id.",
                },
                {
                  href: "/datasets",
                  icon: Database,
                  title: "Inspect datasets",
                  body: "Extracted tables, their scale factors and row counts.",
                },
              ].map((card) => (
                <Link
                  key={card.href}
                  href={card.href}
                  className="flex flex-col gap-1.5 rounded-lg border bg-card p-4 transition-colors hover:bg-accent/40"
                >
                  <card.icon className="size-4 text-muted-foreground" />
                  <span className="text-sm font-medium">{card.title}</span>
                  <span className="text-2xs leading-relaxed text-muted-foreground">
                    {card.body}
                  </span>
                </Link>
              ))}
            </section>
          </>
        )}
      </main>
    </>
  );
}
