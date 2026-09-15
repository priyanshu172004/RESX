"use client";

import Link from "next/link";

import { Activity, RefreshCw, Sparkles } from "lucide-react";

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
import { runs as runsApi, type RunSummary } from "@/lib/api";
import { useApiResource } from "@/hooks/use-api-resource";

function statusVariant(status: string) {
  if (status === "completed") return "secondary" as const;
  if (status === "failed") return "destructive" as const;
  return "outline" as const;
}

export default function RunsPage() {
  const { data: rows, error, refresh: load } = useApiResource<RunSummary[]>(
    async () => {
      const response = await runsApi.list();
      // Accept either shape. The endpoint returns a bare array today and a
      // wrapped object would be a reasonable future change; crashing the page
      // on the one it is not would be a silly way to find out.
      return Array.isArray(response) ? response : (response.runs ?? []);
    },
  );

  return (
    <>
      <SiteHeader crumbs={[{ label: "Runs" }]} />

      <main className="flex min-w-0 flex-1 flex-col gap-4 p-4">
        <PageHeader
          title="Analyses"
          description="Each run is a full pass: plan, specialists in parallel, grounding gate, identity check, Critic, debate, synthesis."
          actions={
            <>
              <Button size="sm" variant="outline" onClick={load}>
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

        {error != null && <ErrorState error={error} onRetry={load} />}

        {rows === null ? (
          <LoadingBlock rows={3} />
        ) : rows.length === 0 ? (
          <EmptyState
            icon={Activity}
            title="No analyses yet"
            description="Ask a question about your corpus. The Manager picks the minimum set of specialists needed to answer it."
            action={{ label: "Ask a question", href: "/runs/new" }}
          />
        ) : (
          <Panel title={`${rows.length} ${rows.length === 1 ? "run" : "runs"}`}>
            <ul className="flex flex-col divide-y">
              {rows.map((run) => (
                <li key={run.run_id}>
                  <Link
                    href={`/runs/${run.run_id}`}
                    className="flex flex-wrap items-center gap-3 py-3 transition-colors hover:bg-accent/40"
                  >
                    <Activity className="size-4 shrink-0 text-muted-foreground" />
                    <div className="flex min-w-0 flex-1 flex-col">
                      <span className="truncate text-sm">{run.question}</span>
                      <span className="font-mono text-2xs text-muted-foreground">
                        {run.run_id}
                      </span>
                    </div>
                    <Badge variant={statusVariant(run.status)} className="text-2xs">
                      {run.status}
                    </Badge>
                    <span className="tabular text-2xs text-muted-foreground">
                      {run.tokens_used.toLocaleString()} tok
                    </span>
                    <span className="tabular text-2xs text-muted-foreground">
                      ${run.usd_used.toFixed(4)}
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          </Panel>
        )}
      </main>
    </>
  );
}
