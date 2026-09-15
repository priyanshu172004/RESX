"use client";

import { use, useMemo } from "react";
import { notFound } from "next/navigation";
import Link from "next/link";
import { Globe, ShieldCheck, Wrench } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { SiteHeader } from "@/components/site-header";
import {
  EmptyState,
  ErrorState,
  LoadingBlock,
  PageHeader,
  Panel,
} from "@/components/page-shell";
import { ClaimCard } from "@/components/claim-card";
import { workspace, type AgentInfo, type ClaimRow } from "@/lib/api";
import { seriesColor } from "@/lib/series";
import { useApiResource } from "@/hooks/use-api-resource";

/** The eight roles. A URL outside this set is a 404, not an empty page. */
const KNOWN = new Set([
  "manager",
  "finance",
  "risk",
  "news",
  "workflow",
  "market",
  "critic",
  "synthesizer",
]);

export default function AgentPage({
  params,
}: {
  params: Promise<{ agent: string }>;
}) {
  const { agent: slug } = use(params);
  const agentName = slug.toLowerCase();

  const { data, error, refresh: load } = useApiResource<{
    agents: AgentInfo[];
    claims: ClaimRow[];
  }>(async () => {
    // Fetched together: the page is one unit, and two sequential requests
    // would render the header before the claims it describes.
    const [roster, insights] = await Promise.all([
      workspace.agents(),
      workspace.insights(100),
    ]);
    return { agents: roster.agents, claims: insights.claims };
  });

  const agents = data?.agents ?? null;
  const info = agents?.find((entry) => entry.agent === agentName) ?? null;
  const mine = useMemo(
    () => (data?.claims ?? []).filter((claim) => claim.agent === agentName),
    [data, agentName],
  );

  // After the hooks, never before: an early return above them would change the
  // hook order between renders.
  if (!KNOWN.has(agentName)) notFound();

  return (
    <>
      <SiteHeader
        crumbs={[
          { label: "Agents", href: "/agents" },
          { label: agentName },
        ]}
      />

      <main className="flex min-w-0 flex-1 flex-col gap-4 p-4">
        <PageHeader
          title={info?.title ?? agentName}
          description={info?.description || undefined}
        />

        {error != null && <ErrorState error={error} onRetry={load} />}

        {agents === null ? (
          <LoadingBlock rows={3} />
        ) : (
          <>
            {info && (
              <>
                <section className="grid gap-3 sm:grid-cols-3">
                  {[
                    ["Claims produced", String(info.claims)],
                    ["Mean confidence", info.mean_confidence.toFixed(2)],
                    ["Cited share", `${(info.cited_share * 100).toFixed(0)}%`],
                  ].map(([label, value]) => (
                    <div
                      key={label}
                      className="flex flex-col gap-1 rounded-lg border bg-card p-3"
                    >
                      <span className="text-2xs uppercase tracking-wide text-muted-foreground">
                        {label}
                      </span>
                      <span className="truncate text-sm font-semibold">
                        {value}
                      </span>
                    </div>
                  ))}
                </section>

                <Panel
                  title="What it can do"
                  description="Derived from the allowlist that enforces it, so this cannot drift from what the agent is actually able to do."
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <span
                      className="size-2.5 rounded-sm"
                      style={{ background: seriesColor(agentName) }}
                    />
                    {info.network_access ? (
                      <Badge variant="outline" className="gap-1 text-2xs">
                        <Globe className="size-2.5 text-warning" />
                        web access
                      </Badge>
                    ) : (
                      <Badge variant="secondary" className="gap-1 text-2xs">
                        <ShieldCheck className="size-2.5 text-good" />
                        sandboxed
                      </Badge>
                    )}
                  </div>

                  <ul className="mt-3 flex flex-col gap-1.5">
                    {info.capabilities.length === 0 ? (
                      <li className="flex items-start gap-1.5 text-xs text-muted-foreground">
                        <Wrench className="mt-0.5 size-3 shrink-0" />
                        No capabilities by design — it reasons over what the
                        specialists returned and can reach neither a document
                        nor the network itself.
                      </li>
                    ) : (
                      info.capabilities.map((capability) => (
                        <li
                          key={capability}
                          className="flex items-start gap-1.5 text-xs"
                        >
                          <Wrench className="mt-0.5 size-3 shrink-0 text-muted-foreground" />
                          {capability}
                        </li>
                      ))
                    )}
                  </ul>
                </Panel>
              </>
            )}

            <Panel
              title={`Claims by ${agentName}`}
              description="Across this workspace's recent runs."
            >
              {mine.length === 0 ? (
                <EmptyState
                  title="No claims from this agent yet"
                  description={
                    agentName === "news" || agentName === "market"
                      ? "These agents need a web search key. Without TAVILY_API_KEY their branch is marked degraded and produces nothing."
                      : "Run an analysis and this agent will contribute if the Manager selects it."
                  }
                  action={{ label: "New analysis", href: "/runs/new" }}
                />
              ) : (
                <div className="flex flex-col gap-3">
                  {mine.map((claim) => (
                    <ClaimCard key={claim.claim_id} claim={claim} />
                  ))}
                </div>
              )}
            </Panel>

            <p className="text-2xs text-muted-foreground">
              <Link href="/agents" className="underline underline-offset-4">
                Back to the full roster
              </Link>
            </p>
          </>
        )}
      </main>
    </>
  );
}
