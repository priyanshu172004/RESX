"use client";

/**
 * What this deployment can and cannot currently do.
 *
 * Shown because a degraded capability is silent otherwise. Without a model key
 * the "New analysis" button appears to work and then fails on submit; without
 * a Voyage key retrieval still returns results, but lexical ones, and a user
 * comparing recall would have no way to know. Naming the state is cheaper than
 * debugging its symptoms.
 */

import Link from "next/link";
import { Check, CircleAlert, Info } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { Capabilities } from "@/lib/api";

type Row = {
  label: string;
  ok: boolean;
  detail: string;
  /** True when absence blocks a feature outright, rather than degrading it. */
  blocking: boolean;
};

export function CapabilityBanner({
  capabilities,
}: {
  capabilities: Capabilities;
}) {
  const rows: Row[] = [
    {
      label: `Model · ${capabilities.model_provider}`,
      ok: capabilities.model_ready,
      detail: capabilities.model_ready
        ? `Agent runs available on ${capabilities.model_provider}.`
        : "No model key configured, so agent runs are unavailable. Ingestion, retrieval and the math engine all work without one.",
      blocking: true,
    },
    {
      label: "Semantic embeddings",
      ok: capabilities.semantic_embeddings,
      detail: capabilities.semantic_embeddings
        ? "Voyage embeddings in use."
        : "Running on the offline hashing embedder: retrieval matches lexical overlap only, so recall is a floor rather than a prediction.",
      blocking: false,
    },
    {
      label: "Web search",
      ok: capabilities.search_ready,
      detail: capabilities.search_ready
        ? "News and Market agents can reach the web."
        : "No search key, so the News and Market agents return nothing and their branch is marked degraded in the report.",
      blocking: false,
    },
    {
      label: `Store · ${capabilities.store}`,
      ok: capabilities.store === "mongo",
      detail:
        capabilities.store === "mongo"
          ? "MongoDB, with the claims validator installed."
          : "SQLite fallback. Fine for a single node; set MONGODB_URL for the production backend.",
      blocking: false,
    },
  ];

  const problems = rows.filter((row) => !row.ok);
  if (problems.length === 0) return null;

  const blocked = problems.some((row) => row.blocking);

  return (
    <div
      className={
        blocked
          ? "flex flex-wrap items-center gap-x-4 gap-y-2 rounded-lg border border-warning/40 bg-warning/5 px-4 py-2.5"
          : "flex flex-wrap items-center gap-x-4 gap-y-2 rounded-lg border bg-card px-4 py-2.5"
      }
    >
      <span className="flex items-center gap-1.5 text-xs font-medium">
        {blocked ? (
          <CircleAlert className="size-3.5 text-warning" />
        ) : (
          <Info className="size-3.5 text-muted-foreground" />
        )}
        {blocked ? "Limited capability" : "Running in reduced mode"}
      </span>

      <div className="flex flex-wrap items-center gap-1.5">
        {rows.map((row) => (
          <Tooltip key={row.label}>
            <TooltipTrigger asChild>
              <Badge
                variant={row.ok ? "secondary" : "outline"}
                className="gap-1 text-2xs font-normal"
              >
                {row.ok ? (
                  <Check className="size-2.5 text-good" />
                ) : (
                  <CircleAlert className="size-2.5 text-warning" />
                )}
                {row.label}
              </Badge>
            </TooltipTrigger>
            <TooltipContent side="bottom" className="max-w-72">
              {row.detail}
            </TooltipContent>
          </Tooltip>
        ))}
      </div>

      <Link
        href="/settings"
        className="ml-auto text-2xs text-muted-foreground underline-offset-4 hover:underline"
      >
        Configure
      </Link>
    </div>
  );
}
