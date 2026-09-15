"use client";

import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  XAxis,
  YAxis,
} from "recharts";

import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import { ChartCard } from "@/components/chart-card";
import { seriesColor } from "@/lib/series";
import type { AgentActivity } from "@/lib/api";

const chartConfig = {
  claims: { label: "Claims" },
} satisfies ChartConfig;

/**
 * Claims produced per agent, sorted by count.
 *
 * Horizontal because agent names are words rather than dates, and sorted
 * because an unsorted category axis makes the reader do the ranking.
 *
 * Colour is looked up by the agent's *key*, so filtering one agent out cannot
 * repaint the others — the whole reason `seriesColor` takes a key and not an
 * index.
 */
export function AgentClaimsBars({ activity }: { activity: AgentActivity[] }) {
  const data = useMemo(
    () =>
      [...activity]
        .sort((a, b) => b.claims - a.claims)
        .map((entry) => ({
          ...entry,
          fill: seriesColor(entry.agent),
        })),
    [activity],
  );

  const table = {
    head: ["Agent", "Claims", "Mean confidence", "Cited"],
    numericFrom: 1,
    rows: data.map((d) => [
      d.agent,
      d.claims,
      d.mean_confidence.toFixed(2),
      `${(d.cited_share * 100).toFixed(0)}%`,
    ]),
  };

  return (
    <ChartCard
      title="Claims by agent"
      description="How many findings each specialist produced, and how well cited they are"
      table={table}
      footer={
        <span className="text-xs text-muted-foreground">
          A low count is not a fault. An agent drops anything it cannot ground
          or compute, so fewer claims can mean a stricter run.
        </span>
      }
    >
      <ChartContainer config={chartConfig} className="h-[220px] w-full">
        <BarChart
          data={data}
          layout="vertical"
          margin={{ left: 4, right: 32, top: 4, bottom: 4 }}
        >
          <CartesianGrid horizontal={false} strokeDasharray="3 3" />
          <XAxis type="number" hide />
          <YAxis
            type="category"
            dataKey="agent"
            width={92}
            tickLine={false}
            axisLine={false}
            className="text-2xs"
          />
          <ChartTooltip
            cursor={false}
            content={
              <ChartTooltipContent
                indicator="line"
                formatter={(value, _name, item) => {
                  const row = item?.payload as AgentActivity | undefined;
                  return `${value} claims · mean confidence ${
                    row?.mean_confidence.toFixed(2) ?? "—"
                  }`;
                }}
              />
            }
          />
          <Bar dataKey="claims" radius={[0, 4, 4, 0]} isAnimationActive={false}>
            {data.map((entry) => (
              <Cell key={entry.agent} fill={entry.fill} />
            ))}
            <LabelList
              dataKey="claims"
              position="right"
              className="fill-muted-foreground text-2xs"
            />
          </Bar>
        </BarChart>
      </ChartContainer>
    </ChartCard>
  );
}
