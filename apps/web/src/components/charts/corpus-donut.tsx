"use client";

import { useMemo } from "react";
import { Cell, Label, Pie, PieChart } from "recharts";

import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import { ChartCard } from "@/components/chart-card";
import { seriesColor } from "@/lib/series";
import { formatNumber } from "@/lib/format";
import type { Dataset } from "@/lib/api";

const chartConfig = { rows: { label: "Rows" } } satisfies ChartConfig;

/**
 * Extracted table rows by source document.
 *
 * A donut is only defensible when the slices are parts of a meaningful whole
 * and there are few of them, so documents past the fifth are folded into
 * "Other" rather than generating a sixth, seventh and eighth hue — the palette
 * has eight validated slots and a long tail would exhaust them for no gain.
 */
export function CorpusDonut({ datasets }: { datasets: Dataset[] }) {
  const { slices, total } = useMemo(() => {
    const byDoc = new Map<string, number>();
    for (const dataset of datasets) {
      byDoc.set(
        dataset.doc_id,
        (byDoc.get(dataset.doc_id) ?? 0) + (dataset.n_rows || 0),
      );
    }

    const sorted = [...byDoc.entries()].sort((a, b) => b[1] - a[1]);
    const head = sorted.slice(0, 5);
    const tail = sorted.slice(5);

    const result = head.map(([docId, rows], index) => ({
      key: docId,
      // The id's tail, which is what appears in URLs and log lines, so a
      // reader can match a slice to a document without a lookup.
      label: docId.replace(/^doc_/, "").slice(0, 8),
      rows,
      fill: seriesColor(
        ["costGoods", "costOperating", "costSales", "costRnd", "costGa"][index],
      ),
    }));

    if (tail.length) {
      result.push({
        key: "other",
        label: `Other (${tail.length})`,
        rows: tail.reduce((sum, [, rows]) => sum + rows, 0),
        fill: "var(--muted-foreground)",
      });
    }

    return {
      slices: result,
      total: sorted.reduce((sum, [, rows]) => sum + rows, 0),
    };
  }, [datasets]);

  if (slices.length === 0) {
    return (
      <ChartCard
        title="Extracted rows"
        description="Populated once a document with tables is ingested"
      >
        <div className="grid h-[220px] place-items-center">
          <p className="text-xs text-muted-foreground">
            No tabular data extracted yet.
          </p>
        </div>
      </ChartCard>
    );
  }

  return (
    <ChartCard
      title="Extracted rows by document"
      description="Tables become typed datasets the sandbox can compute over"
      table={{
        head: ["Document", "Rows"],
        numericFrom: 1,
        rows: slices.map((s) => [s.label, s.rows]),
      }}
      footer={
        <div className="flex w-full flex-wrap items-center gap-x-3 gap-y-1">
          {slices.map((slice) => (
            <span
              key={slice.key}
              className="flex items-center gap-1.5 text-2xs text-muted-foreground"
            >
              <span
                className="size-2 rounded-sm"
                style={{ background: slice.fill }}
              />
              {slice.label}
            </span>
          ))}
        </div>
      }
    >
      <ChartContainer config={chartConfig} className="mx-auto h-[220px] w-full">
        <PieChart>
          <ChartTooltip
            cursor={false}
            content={<ChartTooltipContent nameKey="label" hideLabel />}
          />
          <Pie
            data={slices}
            dataKey="rows"
            nameKey="label"
            // A wide ring rather than a thin one: the centre has to fit the
            // total without colliding with the arc.
            innerRadius={58}
            outerRadius={88}
            strokeWidth={2}
            isAnimationActive={false}
          >
            {slices.map((slice) => (
              <Cell key={slice.key} fill={slice.fill} />
            ))}
            <Label
              content={({ viewBox }) => {
                if (!viewBox || !("cx" in viewBox)) return null;
                const { cx, cy } = viewBox as { cx: number; cy: number };
                return (
                  <text x={cx} y={cy} textAnchor="middle">
                    <tspan
                      x={cx}
                      y={cy - 4}
                      className="fill-foreground text-lg font-semibold tabular-nums"
                    >
                      {formatNumber(total)}
                    </tspan>
                    <tspan
                      x={cx}
                      y={cy + 14}
                      className="fill-muted-foreground text-2xs"
                    >
                      rows
                    </tspan>
                  </text>
                );
              }}
            />
          </Pie>
        </PieChart>
      </ChartContainer>
    </ChartCard>
  );
}
