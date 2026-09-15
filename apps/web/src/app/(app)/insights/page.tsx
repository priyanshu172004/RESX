"use client";

import { useMemo, useState } from "react";
import { Lightbulb, RefreshCw, ShieldAlert, TriangleAlert } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { SiteHeader } from "@/components/site-header";
import {
  EmptyState,
  ErrorState,
  LoadingBlock,
  PageHeader,
  Panel,
} from "@/components/page-shell";
import { ClaimCard } from "@/components/claim-card";
import { workspace } from "@/lib/api";
import { useApiResource } from "@/hooks/use-api-resource";

type Filter = "all" | "numeric" | "contested" | "qualitative";

export default function InsightsPage() {
  const [filter, setFilter] = useState<Filter>("all");
  const { data, error, refresh: load } = useApiResource(
    () => workspace.insights(100),
  );

  const claims = data?.claims ?? null;
  const stats = {
    total: data?.total ?? 0,
    contested: data?.contested ?? 0,
    ungrounded: data?.ungrounded_numeric ?? 0,
  };

  const shown = useMemo(() => {
    const rows = claims ?? [];
    switch (filter) {
      case "numeric":
        return rows.filter((claim) => claim.value !== null);
      case "qualitative":
        return rows.filter((claim) => claim.value === null);
      case "contested":
        return rows.filter((claim) => claim.verdict === "contested");
      default:
        return rows;
    }
  }, [claims, filter]);

  return (
    <>
      <SiteHeader crumbs={[{ label: "Insights" }]} />

      <main className="flex min-w-0 flex-1 flex-col gap-4 p-4">
        <PageHeader
          title="Claims"
          description="Every finding across this workspace, ranked by how well supported it is rather than by recency."
          actions={
            <Button size="sm" variant="outline" onClick={load}>
              <RefreshCw className="size-3.5" />
              Refresh
            </Button>
          }
        />

        {error != null && <ErrorState error={error} onRetry={load} />}

        {claims === null ? (
          <LoadingBlock rows={4} />
        ) : claims.length === 0 ? (
          <EmptyState
            icon={Lightbulb}
            title="No claims yet"
            description="Claims are produced by an analysis. Each one arrives with its citation and, if it carries a number, the computation that produced it."
            action={{ label: "Run an analysis", href: "/runs/new" }}
          />
        ) : (
          <>
            <section className="grid gap-3 sm:grid-cols-3">
              <div className="flex flex-col gap-1 rounded-lg border bg-card p-3">
                <span className="text-2xs uppercase tracking-wide text-muted-foreground">
                  Total claims
                </span>
                <span
                  data-slot="kpi-value"
                  className="text-xl font-semibold tracking-tight"
                >
                  {stats.total}
                </span>
              </div>

              <div className="flex flex-col gap-1 rounded-lg border bg-card p-3">
                <span className="text-2xs uppercase tracking-wide text-muted-foreground">
                  Contested
                </span>
                <span
                  data-slot="kpi-value"
                  className="flex items-center gap-1.5 text-xl font-semibold tracking-tight"
                >
                  {stats.contested > 0 && (
                    <ShieldAlert className="size-4 text-warning" />
                  )}
                  {stats.contested}
                </span>
                <span className="text-2xs text-muted-foreground">
                  Disagreement that survived three rounds of debate. Shipped as
                  contested, not averaged away.
                </span>
              </div>

              <div
                className={
                  stats.ungrounded > 0
                    ? "flex flex-col gap-1 rounded-lg border border-critical/50 bg-critical/5 p-3"
                    : "flex flex-col gap-1 rounded-lg border bg-card p-3"
                }
              >
                <span className="text-2xs uppercase tracking-wide text-muted-foreground">
                  Numbers without a computation
                </span>
                <span
                  data-slot="kpi-value"
                  className="flex items-center gap-1.5 text-xl font-semibold tracking-tight"
                >
                  {stats.ungrounded > 0 && (
                    <TriangleAlert className="size-4 text-critical" />
                  )}
                  {stats.ungrounded}
                </span>
                <span className="text-2xs text-muted-foreground">
                  {stats.ungrounded === 0
                    ? "As it must be — the database rejects such a claim outright."
                    : "This should be impossible. Report it."}
                </span>
              </div>
            </section>

            <Tabs value={filter} onValueChange={(v) => setFilter(v as Filter)}>
              <TabsList>
                <TabsTrigger value="all" className="text-xs">
                  All ({claims.length})
                </TabsTrigger>
                <TabsTrigger value="numeric" className="text-xs">
                  Numeric ({claims.filter((c) => c.value !== null).length})
                </TabsTrigger>
                <TabsTrigger value="qualitative" className="text-xs">
                  Qualitative ({claims.filter((c) => c.value === null).length})
                </TabsTrigger>
                <TabsTrigger value="contested" className="text-xs">
                  Contested ({stats.contested})
                </TabsTrigger>
              </TabsList>
            </Tabs>

            <Panel
              title={`${shown.length} shown`}
              description="A qualitative claim carries no number, so it needs a citation but no computation."
            >
              {shown.length === 0 ? (
                <p className="text-xs text-muted-foreground">
                  Nothing matches this filter.
                </p>
              ) : (
                <div className="flex flex-col gap-3">
                  {shown.map((claim) => (
                    <div key={claim.claim_id} className="flex flex-col gap-1">
                      <span className="flex items-center gap-2 text-2xs text-muted-foreground">
                        <Badge variant="outline" className="text-2xs">
                          {claim.run_id}
                        </Badge>
                        <span className="truncate">{claim.question}</span>
                      </span>
                      <ClaimCard claim={claim} />
                    </div>
                  ))}
                </div>
              )}
            </Panel>
          </>
        )}
      </main>
    </>
  );
}
