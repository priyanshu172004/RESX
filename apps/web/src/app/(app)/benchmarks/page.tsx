"use client";

/**
 * Accuracy signals for the user's own corpus.
 *
 * Distinct from `benchmarks/score.py`, which measures the pipeline against a
 * gold corpus with known answers. There is no ground truth for a user's
 * documents, so only the checks that need none are reported here — whether
 * citations resolve, whether the two table extractors agreed, whether a scale
 * factor was applied.
 *
 * The four metrics that *do* need ground truth are listed as not measured
 * rather than omitted. Omitting them would leave a page of green ticks
 * implying a completeness this cannot claim.
 */


import { CircleCheck, CircleMinus, FlaskConical, RefreshCw, TriangleAlert } from "lucide-react";

import { Button } from "@/components/ui/button";
import { SiteHeader } from "@/components/site-header";
import {
  ErrorState,
  LoadingBlock,
  PageHeader,
  Panel,
} from "@/components/page-shell";
import { workspace, type BenchmarkMetric } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useApiResource } from "@/hooks/use-api-resource";

function passes(metric: BenchmarkMetric): boolean | null {
  if (metric.target == null || metric.comparator == null) return null;
  switch (metric.comparator) {
    case "==":
      return metric.value === metric.target;
    case ">=":
      return metric.value >= metric.target;
    case "<=":
      return metric.value <= metric.target;
    default:
      return null;
  }
}

function formatValue(metric: BenchmarkMetric): string {
  if (metric.format === "ratio") return metric.value.toFixed(4);
  return String(metric.value);
}

export default function BenchmarksPage() {
  const { data, error, refresh: load } = useApiResource(
    () => workspace.benchmarks(),
  );

  return (
    <>
      <SiteHeader crumbs={[{ label: "Corpus" }, { label: "Benchmarks" }]} />

      <main className="flex min-w-0 flex-1 flex-col gap-4 p-4">
        <PageHeader
          title="Accuracy"
          description="What can be checked without a known answer, measured on your corpus."
          actions={
            <Button size="sm" variant="outline" onClick={load}>
              <RefreshCw className="size-3.5" />
              Refresh
            </Button>
          }
        />

        {error != null && <ErrorState error={error} onRetry={load} />}

        {data === null ? (
          <LoadingBlock rows={4} />
        ) : (
          <>
            <Panel title="Measured">
              <ul className="flex flex-col divide-y">
                {data.metrics.map((metric) => {
                  const ok = passes(metric);
                  return (
                    <li
                      key={metric.key}
                      className="flex flex-wrap items-start gap-3 py-3"
                    >
                      {ok === null ? (
                        <CircleMinus className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
                      ) : ok ? (
                        <CircleCheck className="mt-0.5 size-4 shrink-0 text-good" />
                      ) : (
                        <TriangleAlert className="mt-0.5 size-4 shrink-0 text-critical" />
                      )}

                      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                        <span className="text-sm font-medium">
                          {metric.label}
                        </span>
                        <span className="text-2xs leading-relaxed text-muted-foreground">
                          {metric.note}
                        </span>
                      </div>

                      <div className="flex shrink-0 items-baseline gap-2">
                        <span
                          className={cn(
                            "tabular text-sm font-semibold",
                            ok === false && "text-critical",
                          )}
                        >
                          {formatValue(metric)}
                        </span>
                        {metric.target != null && (
                          <span className="font-mono text-2xs text-muted-foreground">
                            {metric.comparator} {metric.target}
                          </span>
                        )}
                      </div>
                    </li>
                  );
                })}
              </ul>
            </Panel>

            <Panel
              title="Not measured"
              description="Reported as unmeasured rather than defaulted to a passing value."
            >
              <ul className="flex flex-col gap-2">
                {data.not_measured.map((entry) => (
                  <li
                    key={entry.key}
                    className="flex flex-wrap items-baseline gap-2 text-xs"
                  >
                    <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-2xs">
                      {entry.key}
                    </code>
                    <span className="text-muted-foreground">
                      {entry.reason}
                    </span>
                  </li>
                ))}
              </ul>
            </Panel>

            <Panel
              title="Gated metrics"
              description="Measured against a synthetic corpus whose answers are known in advance."
            >
              <div className="flex flex-col gap-3">
                <p className="text-xs leading-relaxed text-muted-foreground">
                  {data.gold_corpus_note}
                </p>
                <pre className="overflow-x-auto rounded bg-muted p-3 font-mono text-2xs">
                  {[
                    "python benchmarks/gold/generate.py",
                    "python benchmarks/score.py --k 10",
                  ].join("\n")}
                </pre>
                <p className="flex items-start gap-1.5 text-2xs leading-relaxed text-muted-foreground">
                  <FlaskConical className="mt-px size-3 shrink-0" />
                  That corpus includes planted traps — an undeclared scale
                  factor, segments that do not sum, and fabricated quotes — so a
                  pass means the checks actually fired rather than that nothing
                  was tested.
                </p>
              </div>
            </Panel>
          </>
        )}
      </main>
    </>
  );
}
