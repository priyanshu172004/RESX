"use client";

import { useMemo } from "react";
import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from "recharts";

import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import { ChartCard } from "@/components/chart-card";
import { seriesColor } from "@/lib/series";
import type { SpendPoint } from "@/lib/api";

const chartConfig = {
  usd: { label: "Cost (USD)", color: seriesColor("revenue") },
} satisfies ChartConfig;

/**
 * Cost per analysis over time.
 *
 * Cost is on its own axis and tokens are not plotted alongside it. A dual axis
 * lets the author choose a scale that makes any two series look correlated, so
 * §6 rules them out; the token count is in the tooltip and the table instead.
 */
export function SpendArea({ points }: { points: SpendPoint[] }) {
  const data = useMemo(
    () =>
      points.map((point, index) => ({
        // The run's ordinal, not its timestamp: runs are irregular, and a time
        // axis with three points a week apart and two a minute apart is
        // unreadable.
        label: `#${index + 1}`,
        usd: point.usd,
        tokens: point.tokens,
        question: point.label,
      })),
    [points],
  );

  const table = {
    head: ["Run", "Question", "Cost (USD)", "Tokens"],
    numericFrom: 2,
    rows: data.map((d) => [d.label, d.question, d.usd.toFixed(4), d.tokens]),
  };

  if (points.length === 0) {
    return (
      <ChartCard
        title="Cost per analysis"
        description="Populated once an analysis has run"
      >
        <div className="grid h-[220px] place-items-center">
          <p className="text-xs text-muted-foreground">
            No runs yet, so there is nothing to chart.
          </p>
        </div>
      </ChartCard>
    );
  }

  return (
    <ChartCard
      title="Cost per analysis"
      description="Every run is capped; the graph halts rather than looping"
      table={table}
      footer={
        <span className="text-xs text-muted-foreground">
          Total{" "}
          <span className="tabular font-medium text-foreground">
            ${points.reduce((sum, p) => sum + p.usd, 0).toFixed(4)}
          </span>{" "}
          across {points.length} {points.length === 1 ? "run" : "runs"}
        </span>
      }
    >
      <ChartContainer config={chartConfig} className="h-[220px] w-full">
        <AreaChart data={data} margin={{ left: 4, right: 8, top: 8 }}>
          <defs>
            <linearGradient id="resx-spend-fill" x1="0" y1="0" x2="0" y2="1">
              <stop
                offset="0%"
                stopColor={seriesColor("revenue")}
                stopOpacity={0.35}
              />
              <stop
                offset="100%"
                stopColor={seriesColor("revenue")}
                stopOpacity={0.02}
              />
            </linearGradient>
          </defs>
          <CartesianGrid vertical={false} strokeDasharray="3 3" />
          <XAxis
            dataKey="label"
            tickLine={false}
            axisLine={false}
            tickMargin={8}
            className="text-2xs"
          />
          <YAxis
            width={60}
            tickLine={false}
            axisLine={false}
            tickMargin={6}
            className="text-2xs"
            tickFormatter={(value: number) => `$${value.toFixed(3)}`}
          />
          <ChartTooltip
            cursor={false}
            content={
              <ChartTooltipContent
                indicator="line"
                labelFormatter={(_, payload) =>
                  String(payload?.[0]?.payload?.question ?? "")
                }
                formatter={(value) => `$${Number(value).toFixed(4)}`}
              />
            }
          />
          <Area
            dataKey="usd"
            type="monotone"
            stroke={seriesColor("revenue")}
            strokeWidth={2}
            fill="url(#resx-spend-fill)"
            // Recharts latches a stale clip-path width when the container is
            // measured mid-animation, which paints a sliver of the series.
            isAnimationActive={false}
          />
        </AreaChart>
      </ChartContainer>
    </ChartCard>
  );
}
