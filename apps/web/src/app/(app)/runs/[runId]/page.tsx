"use client";

/**
 * A single run: the live event stream, then the report.
 *
 * The stream is the product here, not a progress bar. A reader needs to see
 * *which* node produced *which* claim to trust the result, so every node
 * transition is rendered rather than collapsed into a percentage.
 */

import { use, useEffect } from "react";
import Link from "next/link";
import {
  Activity,
  BadgeCheck,
  CircleCheck,
  CircleX,
  Loader2,
  RefreshCw,
  ShieldAlert,
  TriangleAlert,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { SiteHeader } from "@/components/site-header";
import {
  ErrorState,
  LoadingBlock,
  PageHeader,
  Panel,
} from "@/components/page-shell";
import { ClaimCard } from "@/components/claim-card";
import {
  ReportCharts,
  type ReportChart,
} from "@/components/charts/report-charts";
import {
  ReportDocumentView,
  type ReportDocumentPayload,
} from "@/components/report-document";
import { runs as runsApi, type ClaimRow } from "@/lib/api";
import { useRunStream } from "@/hooks/use-run-stream";
import { cn } from "@/lib/utils";
import { useApiResource } from "@/hooks/use-api-resource";

const ACTIVE = new Set(["queued", "running", "planning"]);

export default function RunPage({
  params,
}: {
  params: Promise<{ runId: string }>;
}) {
  const { runId } = use(params);

  const {
    data: run,
    error,
    refresh: load,
  } = useApiResource<Record<string, unknown>>(() => runsApi.get(runId), [runId]);

  const status = String(run?.status ?? "queued");
  const isActive = ACTIVE.has(status);

  const { events, status: streamStatus, error: streamError, finished } =
    useRunStream(runId, isActive || run === null);

  // Refetch once the stream reports a terminal event: the stream carries the
  // narrative, the record carries the report, and only the record is durable.
  useEffect(() => {
    if (finished) load();
  }, [finished, load]);

  const report = run?.report as Record<string, unknown> | null | undefined;
  const claims = (run?.claims as ClaimRow[] | undefined) ?? [];

  return (
    <>
      <SiteHeader
        crumbs={[{ label: "Runs", href: "/runs" }, { label: runId }]}
      />

      <main className="flex min-w-0 flex-1 flex-col gap-4 p-4">
        <PageHeader
          title={String(run?.question ?? "Analysis")}
          description={
            run
              ? `${status} · ${Number(run.tokens_used ?? 0).toLocaleString()} tokens · $${Number(
                  run.usd_used ?? 0,
                ).toFixed(4)} of $${Number(run.usd_cap ?? 0).toFixed(2)}`
              : undefined
          }
          actions={
            <Button size="sm" variant="outline" onClick={load}>
              <RefreshCw className="size-3.5" />
              Refresh
            </Button>
          }
        />

        {error != null && <ErrorState error={error} onRetry={load} />}

        {run === null && !error && <LoadingBlock rows={3} />}

        {run && (
          <>
            <div className="flex flex-wrap items-center gap-2">
              <Badge
                variant={
                  status === "completed"
                    ? "secondary"
                    : status === "failed"
                      ? "destructive"
                      : "outline"
                }
                className="gap-1.5 text-2xs"
              >
                {isActive && <Loader2 className="size-2.5 animate-spin" />}
                {status}
              </Badge>

              {isActive && (
                <span className="text-2xs text-muted-foreground">
                  stream: {streamStatus}
                </span>
              )}

              {Array.isArray(run.degraded) && run.degraded.length > 0 && (
                <Badge variant="outline" className="gap-1 text-2xs">
                  <TriangleAlert className="size-2.5 text-warning" />
                  {run.degraded.length} degraded{" "}
                  {run.degraded.length === 1 ? "branch" : "branches"}
                </Badge>
              )}
            </div>

            {typeof run.error === "string" && run.error && (
              <ErrorState error={new Error(run.error)} />
            )}

            {streamError && isActive && (
              <p className="text-2xs text-muted-foreground">
                Stream interrupted ({streamError}). Reconnecting from the last
                event received — nothing is lost.
              </p>
            )}

            <Panel
              title="Execution"
              description="Every node transition, in order. A resumed stream picks up from the last sequence number rather than replaying."
            >
              {events.length === 0 ? (
                <p className="text-xs text-muted-foreground">
                  {isActive
                    ? "Waiting for the first node to report…"
                    : "No events recorded for this run."}
                </p>
              ) : (
                <ScrollArea className="h-[320px] pr-3">
                  <ol className="flex flex-col gap-2">
                    {events.map((event) => (
                      <li
                        key={event.seq}
                        className="flex items-start gap-2.5 border-b pb-2 last:border-0"
                      >
                        <EventIcon kind={event.kind} />
                        <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                          <span className="flex items-center gap-2">
                            <span className="font-mono text-2xs text-muted-foreground">
                              {event.seq}
                            </span>
                            <span className="text-xs font-medium">
                              {stageLabel(event.kind)}
                            </span>
                          </span>
                          <EventBody kind={event.kind} payload={event.payload} />
                        </div>
                      </li>
                    ))}
                  </ol>
                </ScrollArea>
              )}
            </Panel>

            {report && <ReportView report={report} runId={runId} />}

            {claims.length > 0 && (
              <Panel
                title={`${claims.length} claims`}
                description="Each with its citation and the computation that produced its number."
              >
                <div className="flex flex-col gap-3">
                  {claims.map((claim) => (
                    <ClaimCard key={claim.claim_id} claim={claim} />
                  ))}
                </div>
              </Panel>
            )}

            {status === "completed" && !report && (
              <Panel title="No report">
                <p className="text-xs text-muted-foreground">
                  The run completed without producing a report. That happens
                  when every claim was dropped by the grounding gate — the
                  corpus did not support an answer to this question. Try a
                  narrower question, or check the Documents page for extraction
                  warnings.
                </p>
              </Panel>
            )}

            <p className="text-2xs text-muted-foreground">
              <Link href="/insights" className="underline underline-offset-4">
                Browse every claim in this workspace
              </Link>
            </p>
          </>
        )}
      </main>
    </>
  );
}

/** The event kind as a stage name, rather than the internal identifier. */
function stageLabel(kind: string): string {
  const labels: Record<string, string> = {
    run_start: "Started",
    node_start: "Stage",
    node_end: "Stage complete",
    tool_call: "Action",
    computation: "Calculation",
    claim: "Finding",
    claim_dropped: "Rejected",
    citation_derived: "Source traced",
    verdict: "Review",
    critic_abstained: "Review incomplete",
    debate_round: "Debate",
    budget: "Cost",
    report: "Report",
    node_error: "Problem",
    error: "Problem",
    done: "Finished",
  };
  return labels[kind] ?? "Progress";
}

function EventIcon({ kind }: { kind: string }) {
  const className = "mt-0.5 size-3.5 shrink-0";
  if (kind.includes("failed") || kind.includes("error")) {
    return <CircleX className={cn(className, "text-critical")} />;
  }
  if (kind.includes("completed") || kind.includes("done")) {
    return <CircleCheck className={cn(className, "text-good")} />;
  }
  if (kind.includes("critic") || kind.includes("debate")) {
    return <ShieldAlert className={cn(className, "text-warning")} />;
  }
  if (kind.includes("claim") || kind.includes("verdict")) {
    return <BadgeCheck className={cn(className, "text-muted-foreground")} />;
  }
  return <Activity className={cn(className, "text-muted-foreground")} />;
}

/**
 * One line of plain English per event.
 *
 * This used to fall back to `JSON.stringify(payload)` for anything it did not
 * recognise, on the reasoning that an unrecognised event is the one worth
 * seeing. That was wrong for a viewer: the raw payloads carried internal node
 * names, tool identifiers, dataset ids and generated Python — a map of the
 * system's internals rendered to anyone watching a run.
 *
 * Unrecognised events are now summarised rather than dumped. The full payload
 * remains in the run's stored event log for an operator, and a specific number
 * is still auditable through its computation id.
 */
function EventBody({
  kind,
  payload,
}: {
  kind: string;
  payload: Record<string, unknown>;
}) {
  const text = describe(kind, payload);
  if (!text) return null;
  return (
    <span className="text-2xs leading-relaxed text-muted-foreground">
      {text}
    </span>
  );
}

function describe(kind: string, p: Record<string, unknown>): string | null {
  const str = (key: string) =>
    typeof p[key] === "string" ? (p[key] as string) : undefined;
  const num = (key: string) =>
    typeof p[key] === "number" ? (p[key] as number) : undefined;

  switch (kind) {
    case "run_start": {
      const corpus = Array.isArray(p.corpus_ids) ? p.corpus_ids.length : 0;
      return corpus
        ? `Analysing ${corpus} document${corpus === 1 ? "" : "s"}`
        : "Researching public sources";
    }
    case "node_start":
      return str("node") ? `${friendly(str("node")!)} started` : null;
    case "node_end": {
      const node = friendly(str("node") ?? "");
      const kept = num("claims_kept");
      const dropped = num("claims_dropped");
      if (kept !== undefined) {
        return `${node} finished — ${kept} finding${kept === 1 ? "" : "s"} kept${
          dropped ? `, ${dropped} dropped for lack of evidence` : ""
        }`;
      }
      if (num("insights") !== undefined) {
        return `${node} wrote ${num("insights")} insight${num("insights") === 1 ? "" : "s"}`;
      }
      if (num("verdicts") !== undefined) {
        return `${node} reviewed ${num("verdicts")} finding${num("verdicts") === 1 ? "" : "s"}`;
      }
      return `${node} finished`;
    }
    case "tool_call":
      // The API now sends a plain-language action rather than a tool id.
      return str("action") ?? "Used a capability";
    case "computation":
      return `Calculated ${str("label") ?? "a figure"}${
        p.ok === false ? " — the calculation failed" : ""
      }`;
    case "claim":
      return str("statement") ?? "Recorded a finding";
    case "claim_dropped":
      return `Dropped: ${str("statement") ?? "a finding"} — it could not be traced to a source`;
    case "citation_derived":
      return "Traced a figure back to its source rows";
    case "verdict":
      return `Review: ${str("verdict") ?? "assessed"}`;
    case "critic_abstained":
      return "The reviewer could not reach a supported verdict on one finding";
    case "debate_round":
      return `Debate round ${num("round") ?? ""}`.trim();
    case "budget":
      return `Spent $${(num("usd_used") ?? 0).toFixed(4)} of $${(num("usd_cap") ?? 0).toFixed(2)}`;
    case "report":
      return "Report written";
    case "node_error":
    case "error":
      // The message is shown: a user needs to know what failed. It is written
      // for a reader rather than being a stack trace.
      return str("message") ?? "Something failed";
    case "done":
      return `Finished — ${str("status") ?? "done"}`;
    default:
      return null;
  }
}

/** Internal node names, said the way a person would say them. */
function friendly(node: string): string {
  const names: Record<string, string> = {
    manager: "Planning",
    finance: "Finance analysis",
    risk: "Risk analysis",
    news: "External corroboration",
    market: "Market context",
    workflow: "Process review",
    critic: "Adversarial review",
    debate: "Debate",
    synthesize: "Report",
    identity_check: "Accounting checks",
    reingest: "Re-reading the source",
  };
  return names[node] ?? node.replace(/_/g, " ");
}

/**
 * The four kinds of recommendation, and why each is styled the way it is.
 *
 * `kind` encodes a category, not a severity, so it gets neutral treatment with
 * one exception: `mitigate` answers a risk the analysis found, and reads wrong
 * without the warning colour. Every one carries its label as text, so the
 * distinction is never colour-alone.
 */
const RECOMMENDATION_KINDS: Record<
  string,
  { label: string; hint: string; className: string }
> = {
  mitigate: {
    label: "Mitigate",
    hint: "Reduce a risk the analysis found",
    className: "border-warning/40 bg-warning/5",
  },
  grow: {
    label: "Grow",
    hint: "Act on an opportunity the analysis found",
    className: "border-border bg-background",
  },
  monitor: {
    label: "Monitor",
    hint: "Not yet actionable, but worth watching",
    className: "border-border bg-background",
  },
  investigate: {
    label: "Investigate",
    hint: "Close an evidence gap before committing",
    className: "border-border bg-background",
  },
};

const HORIZONS: Record<string, string> = {
  now: "Now",
  quarter: "This quarter",
  year: "This year",
};

const EFFORTS: Record<string, string> = {
  S: "Small",
  M: "Medium",
  L: "Large",
};

function RecommendationList({
  recommendations,
}: {
  recommendations: Record<string, unknown>[];
}) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-col gap-0.5">
        <p className="text-xs font-medium">Recommended next steps</p>
        <p className="text-2xs text-muted-foreground">
          Each one cites the findings it follows from. Advice that cited a
          finding this run did not produce was dropped before you saw it.
        </p>
      </div>

      <ul className="flex flex-col gap-2">
        {recommendations.map((rec, index) => {
          const kind =
            RECOMMENDATION_KINDS[String(rec.kind)] ??
            RECOMMENDATION_KINDS.investigate;
          const claimIds = Array.isArray(rec.supporting_claim_ids)
            ? (rec.supporting_claim_ids as string[])
            : [];
          return (
            <li
              key={index}
              className={cn("rounded-lg border p-3", kind.className)}
            >
              <div className="flex flex-col gap-2">
                <div className="flex flex-wrap items-center gap-1.5">
                  <Badge variant="outline" className="text-2xs">
                    {kind.label}
                  </Badge>
                  {typeof rec.horizon === "string" && (
                    <Badge variant="secondary" className="text-2xs">
                      {HORIZONS[rec.horizon] ?? rec.horizon}
                    </Badge>
                  )}
                  {typeof rec.effort === "string" && (
                    <span className="text-2xs text-muted-foreground">
                      {EFFORTS[rec.effort] ?? rec.effort} effort
                    </span>
                  )}
                  {typeof rec.owner === "string" && (
                    <span className="text-2xs text-muted-foreground">
                      &middot; {rec.owner}
                    </span>
                  )}
                </div>

                <p className="text-sm font-medium leading-snug">
                  {String(rec.action ?? "")}
                </p>

                {typeof rec.rationale === "string" && (
                  <p className="text-xs leading-relaxed text-muted-foreground">
                    {rec.rationale}
                  </p>
                )}

                <dl className="flex flex-col gap-1 border-t pt-2 text-2xs">
                  {typeof rec.metric === "string" && (
                    <div className="flex gap-1.5">
                      <dt className="font-medium">Track</dt>
                      <dd className="text-muted-foreground">{rec.metric}</dd>
                    </div>
                  )}
                  {typeof rec.expected_effect === "string" &&
                    rec.expected_effect && (
                      <div className="flex gap-1.5">
                        <dt className="font-medium">Expected</dt>
                        <dd className="text-muted-foreground">
                          {rec.expected_effect}
                        </dd>
                      </div>
                    )}
                  {claimIds.length > 0 && (
                    <div className="flex gap-1.5">
                      <dt className="font-medium">From</dt>
                      <dd className="text-muted-foreground">
                        {claimIds.length === 1
                          ? "1 finding"
                          : `${claimIds.length} findings`}{" "}
                        above
                      </dd>
                    </div>
                  )}
                </dl>
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

const SWOT_QUADRANTS = [
  { key: "strengths", label: "Strengths" },
  { key: "weaknesses", label: "Weaknesses" },
  { key: "opportunities", label: "Opportunities" },
  { key: "threats", label: "Threats" },
] as const;

function SwotGrid({ swot }: { swot: Record<string, unknown> }) {
  const quadrants = SWOT_QUADRANTS.map((q) => ({
    ...q,
    items: (Array.isArray(swot[q.key])
      ? (swot[q.key] as Record<string, unknown>[])
      : []
    ).filter((item) => typeof item.text === "string"),
  }));

  // An empty quadrant is a valid, honest output, but four empty ones is not a
  // grid worth drawing.
  if (quadrants.every((q) => q.items.length === 0)) return null;

  return (
    <div className="flex flex-col gap-2">
      <p className="text-xs font-medium">Position</p>
      <div className="grid gap-2 sm:grid-cols-2">
        {quadrants.map((q) => (
          <div key={q.key} className="rounded-lg border bg-background p-3">
            <p className="text-2xs font-medium uppercase tracking-wide text-muted-foreground">
              {q.label}
            </p>
            {q.items.length === 0 ? (
              // Stated rather than hidden: "nothing found here" is a result.
              <p className="mt-1 text-2xs text-muted-foreground">
                Nothing the evidence supports.
              </p>
            ) : (
              <ul className="mt-1.5 flex flex-col gap-1">
                {q.items.map((item, i) => (
                  <li key={i} className="text-xs leading-relaxed">
                    {String(item.text)}
                  </li>
                ))}
              </ul>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function ReportView({
  report,
  runId,
}: {
  report: Record<string, unknown>;
  /** Passed down so the document panel can request a server-built export. */
  runId: string;
}) {
  const insights = (report.insights ?? report.top_insights) as
    | Record<string, unknown>[]
    | undefined;
  const recommendations = report.recommendations as
    | Record<string, unknown>[]
    | undefined;
  const swot = report.swot as Record<string, unknown> | undefined;
  const charts = report.charts as ReportChart[] | undefined;
  const document = report.document as ReportDocumentPayload | undefined;
  const riskRegister = report.risk_register as string[] | undefined;
  const financialSummary = report.financial_summary as string | undefined;
  const limitations = report.limitations as string[] | undefined;
  const summary = report.executive_summary ?? report.summary;

  return (
    <Panel
      title="Report"
      description="Ranked findings, the steps they support, and what the analysis could not establish."
    >
      <div className="flex flex-col gap-4">
        {typeof summary === "string" && summary && (
          <p className="text-sm leading-relaxed">{summary}</p>
        )}

        {Array.isArray(insights) && insights.length > 0 && (
          <ol className="flex flex-col gap-3">
            {insights.map((insight, index) => (
              <li
                key={index}
                className="flex gap-3 rounded-lg border bg-background p-3"
              >
                <span className="grid size-5 shrink-0 place-items-center rounded bg-foreground text-2xs font-semibold text-background">
                  {index + 1}
                </span>
                <div className="flex min-w-0 flex-col gap-1">
                  <p className="text-sm font-medium">
                    {String(insight.headline ?? insight.title ?? "")}
                  </p>
                  {typeof insight.detail === "string" && (
                    <p className="text-xs leading-relaxed text-muted-foreground">
                      {insight.detail}
                    </p>
                  )}
                  {typeof insight.so_what === "string" && (
                    <p className="text-xs leading-relaxed">
                      <span className="font-medium">So what: </span>
                      {insight.so_what}
                    </p>
                  )}
                </div>
              </li>
            ))}
          </ol>
        )}

        {Array.isArray(charts) && charts.length > 0 && (
          <ReportCharts charts={charts} />
        )}

        {Array.isArray(recommendations) && recommendations.length > 0 && (
          <RecommendationList recommendations={recommendations} />
        )}

        {typeof financialSummary === "string" && financialSummary && (
          <div className="flex flex-col gap-1 rounded-lg border bg-background p-3">
            <p className="text-xs font-medium">Financial summary</p>
            <p className="text-xs leading-relaxed text-muted-foreground">
              {financialSummary}
            </p>
          </div>
        )}

        {swot && <SwotGrid swot={swot} />}

        {Array.isArray(riskRegister) && riskRegister.length > 0 && (
          <div className="flex flex-col gap-1.5 rounded-lg border bg-background p-3">
            <p className="text-xs font-medium">Risk register</p>
            <ul className="flex flex-col gap-1">
              {riskRegister.map((risk) => (
                <li
                  key={risk}
                  className="flex items-start gap-1.5 text-2xs leading-relaxed text-muted-foreground"
                >
                  <ShieldAlert className="mt-px size-3 shrink-0 text-muted-foreground" />
                  {risk}
                </li>
              ))}
            </ul>
          </div>
        )}

        {document && document.sections?.length > 0 && (
          <ReportDocumentView document={document} runId={runId} />
        )}

        {Array.isArray(limitations) && limitations.length > 0 && (
          <div className="flex flex-col gap-1.5 rounded-lg border border-warning/40 bg-warning/5 p-3">
            <p className="text-xs font-medium">Stated limitations</p>
            <ul className="flex flex-col gap-1">
              {limitations.map((limitation) => (
                <li
                  key={limitation}
                  className="flex items-start gap-1.5 text-2xs leading-relaxed text-muted-foreground"
                >
                  <TriangleAlert className="mt-px size-3 shrink-0 text-warning" />
                  {limitation}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* A report with neither a summary nor insights produced nothing a
            reader can use. Said plainly, rather than dumping the raw object —
            which exposed internal field names and helped nobody. */}
        {!summary && (!insights || insights.length === 0) && (
          <p className="text-xs leading-relaxed text-muted-foreground">
            This run produced no insights. That happens when no finding could
            be traced to a source, which the limitations above explain.
          </p>
        )}
      </div>
    </Panel>
  );
}
