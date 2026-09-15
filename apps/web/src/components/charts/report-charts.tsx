"use client";

/**
 * The charts on a run report: the user's own figures, plotted.
 *
 * Every point here came out of a table extracted from their document, derived
 * server-side in `app/reporting/charts.py`. Nothing on this page is generated
 * by a model, which is why each chart states the document and page it came
 * from — a chart is an assertion until it says where it got its numbers.
 *
 * Form is chosen server-side from the data's job, and the split matters: a
 * table holding revenue, cost and margin% arrives as *two* charts, because
 * putting a percentage and an amount on one plot needs two y-scales and that
 * is the most misleading thing a chart can do.
 */

import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  XAxis,
  YAxis,
} from "recharts";
import { TriangleAlert } from "lucide-react";

import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { assignChartSlots, statusColor, type StatusRole } from "@/lib/series";

/**
 * Outcome label to reserved status step.
 *
 * "Unreviewed" is deliberately `warning` rather than a neutral grey: a
 * finding nobody checked is a caveat, not a null result, and rendering it as
 * absence is what would let a reader mistake an unchecked run for a clean
 * one. Anything unrecognised falls back to `warning` for the same reason —
 * never to "good".
 */
const STATUS_BY_LABEL: Record<string, StatusRole> = {
  confirmed: "good",
  unreviewed: "warning",
  contested: "serious",
  refuted: "critical",
};

function statusFor(label: string): string {
  return statusColor(STATUS_BY_LABEL[label.trim().toLowerCase()] ?? "warning");
}

export type ChartPoint = { x: string; y: number };
export type ChartSeries = { key: string; label: string; points: ChartPoint[] };
/**
 * Which palette a chart's marks take.
 *
 * `status` is the reserved good/warning/serious/critical set and is never
 * reused for "series 4". A reader who has learnt that red means *refuted* must
 * not meet red meaning *West Region* on the next chart, so the server declares
 * this per chart rather than the renderer inferring it from a title.
 */
export type ChartPalette = "categorical" | "status";

export type ReportChart = {
  chart_id: string;
  title: string;
  kind: "line" | "bar" | "donut" | string;
  palette?: ChartPalette;
  x_label: string;
  y_label: string;
  unit: string | null;
  caption: string;
  warnings: string[];
  dataset_id: string;
  doc_id: string;
  source_name: string;
  source_page: number | null;
  series: ChartSeries[];
};

/**
 * Compact axis numbers. A y-axis reading 1200000 makes the reader count
 * digits; 1.2M does not. Kept to one decimal so the tick labels stay short
 * enough not to collide.
 */
function compact(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(1)}B`;
  if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (abs >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return String(Number(value.toFixed(2)));
}

function formatValue(value: number, unit: string | null): string {
  if (unit === "%") return `${Number(value.toFixed(2))}%`;
  const text = compact(value);
  return unit ? `${text} ${unit}` : text;
}

/** Wide-format rows: one row per x, one column per series, as Recharts wants. */
function toRows(series: ChartSeries[]): Record<string, string | number>[] {
  const byX = new Map<string, Record<string, string | number>>();
  const order: string[] = [];
  for (const s of series) {
    for (const point of s.points) {
      if (!byX.has(point.x)) {
        byX.set(point.x, { x: point.x });
        order.push(point.x);
      }
      byX.get(point.x)![s.key] = point.y;
    }
  }
  return order.map((x) => byX.get(x)!);
}

function ChartFooter({ chart }: { chart: ReportChart }) {
  return (
    <div className="flex flex-col gap-1.5">
      <p className="text-2xs leading-relaxed text-muted-foreground">
        {chart.caption}
      </p>
      <p className="text-2xs text-muted-foreground">
        Source: {chart.source_name}
        {chart.source_page ? `, page ${chart.source_page}` : ""} &middot;{" "}
        {/* The provenance line has to be true of *this* chart. A statistics
            chart is counted from the run's own claims and verdicts, not read
            out of a document, and claiming otherwise would be a false
            provenance on a page that exists to make provenance checkable. */}
        {chart.dataset_id
          ? "plotted from the extracted table, not generated"
          : "counted from this run's own record, not generated"}
      </p>
      {chart.warnings.map((warning) => (
        // On the chart, not in a footnote elsewhere. Someone looking at a line
        // going up will not go and find a caveat in another panel.
        <p
          key={warning}
          className="flex items-start gap-1.5 text-2xs leading-relaxed text-warning"
        >
          <TriangleAlert className="mt-px size-3 shrink-0" />
          {warning}
        </p>
      ))}
    </div>
  );
}

/**
 * The table view, always rendered beneath the plot.
 *
 * Not an optional affordance. Some palette slots fall below 3:1 against the
 * light surface, and the validator's contrast warning is a relief obligation —
 * the numbers have to be readable without relying on the colour. It also makes
 * the figures copyable, which is what anyone checking a chart actually wants.
 */
function ChartTable({ chart }: { chart: ReportChart }) {
  const rows = toRows(chart.series);
  return (
    <div className="max-h-56 overflow-auto rounded-md border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="text-2xs">{chart.x_label || "Label"}</TableHead>
            {chart.series.map((s) => (
              <TableHead key={s.key} className="text-right text-2xs">
                {s.label}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row) => (
            <TableRow key={String(row.x)}>
              <TableCell className="text-2xs font-medium">{row.x}</TableCell>
              {chart.series.map((s) => (
                <TableCell key={s.key} className="text-right font-mono text-2xs">
                  {typeof row[s.key] === "number"
                    ? formatValue(row[s.key] as number, chart.unit)
                    : /* A gap, shown as a gap. The server omits missing cells
                         rather than zero-filling them, and so does this. */
                      "—"}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

function SingleChart({ chart }: { chart: ReportChart }) {
  const colors = useMemo(
    () => assignChartSlots(chart.series.map((s) => s.key)),
    [chart.series],
  );
  const rows = useMemo(() => toRows(chart.series), [chart.series]);

  const config = useMemo(() => {
    const out: ChartConfig = {};
    for (const s of chart.series) {
      out[s.key] = { label: s.label, color: colors[s.key] };
    }
    return out;
  }, [chart.series, colors]);

  const showLegend = chart.series.length >= 2;
  // Declared by the server per chart. See `ChartPalette`.
  const isStatus = chart.palette === "status";

  return (
    <figure className="flex flex-col gap-2 rounded-lg border bg-background p-3">
      <figcaption className="flex flex-col gap-0.5">
        <h4 className="text-xs font-medium">{chart.title}</h4>
        <p className="text-2xs text-muted-foreground">{chart.y_label}</p>
      </figcaption>

      <ChartContainer config={config} className="h-[220px] w-full">
        {chart.kind === "donut" ? (
          <PieChart>
            <ChartTooltip
              content={
                <ChartTooltipContent
                  formatter={(value) =>
                    formatValue(Number(value), chart.unit)
                  }
                />
              }
            />
            <Pie
              data={chart.series[0].points.map((p) => ({
                name: p.x,
                value: p.y,
              }))}
              dataKey="value"
              nameKey="name"
              innerRadius={52}
              outerRadius={82}
              // A 2px gap of surface between slices, so adjacent fills read as
              // separate marks rather than one continuous band.
              paddingAngle={2}
              stroke="var(--background)"
              strokeWidth={2}
            >
              {chart.series[0].points.map((p) => (
                <Cell
                  key={p.x}
                  // Status steps when the slices are states (confirmed /
                  // contested / refuted), the categorical palette when they
                  // are identities. Slice colour follows the label either
                  // way, so it is stable whatever order the rows arrive in.
                  fill={
                    isStatus
                      ? statusFor(p.x)
                      : assignChartSlots(
                          chart.series[0].points.map((q) => q.x),
                        )[p.x]
                  }
                />
              ))}
            </Pie>
          </PieChart>
        ) : chart.kind === "bar" ? (
          <BarChart data={rows} margin={{ left: 4, right: 8, top: 8 }}>
            <CartesianGrid vertical={false} strokeDasharray="3 3" />
            <XAxis
              dataKey="x"
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              className="text-2xs"
            />
            <YAxis
              tickFormatter={compact}
              tickLine={false}
              axisLine={false}
              width={48}
              className="text-2xs"
            />
            <ChartTooltip
              content={
                <ChartTooltipContent
                  formatter={(value) => formatValue(Number(value), chart.unit)}
                />
              }
            />
            {showLegend && <ChartLegend content={<ChartLegendContent />} />}
            {chart.series.map((s) => (
              <Bar
                key={s.key}
                dataKey={s.key}
                fill={colors[s.key]}
                // 4px rounded data-end, anchored to the baseline.
                radius={[4, 4, 0, 0]}
                maxBarSize={44}
              >
                {/* A status bar's colour belongs to the bar, not the series:
                    each bar is a different state. */}
                {isStatus &&
                  rows.map((row) => (
                    <Cell key={String(row.x)} fill={statusFor(String(row.x))} />
                  ))}
              </Bar>
            ))}
          </BarChart>
        ) : (
          <LineChart data={rows} margin={{ left: 4, right: 8, top: 8 }}>
            <CartesianGrid vertical={false} strokeDasharray="3 3" />
            <XAxis
              dataKey="x"
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              className="text-2xs"
            />
            <YAxis
              tickFormatter={compact}
              tickLine={false}
              axisLine={false}
              width={48}
              className="text-2xs"
            />
            <ChartTooltip
              content={
                <ChartTooltipContent
                  formatter={(value) => formatValue(Number(value), chart.unit)}
                />
              }
            />
            {showLegend && <ChartLegend content={<ChartLegendContent />} />}
            {chart.series.map((s) => (
              <Line
                key={s.key}
                dataKey={s.key}
                stroke={colors[s.key]}
                strokeWidth={2}
                dot={{ r: 3 }}
                activeDot={{ r: 5 }}
                // Gaps stay gaps rather than being bridged into a straight
                // line that implies data nobody measured.
                connectNulls={false}
              />
            ))}
          </LineChart>
        )}
      </ChartContainer>

      <ChartTable chart={chart} />
      <ChartFooter chart={chart} />
    </figure>
  );
}

export function ReportCharts({ charts }: { charts: ReportChart[] }) {
  const usable = charts.filter((c) => c.series.some((s) => s.points.length > 0));
  if (usable.length === 0) return null;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-col gap-0.5">
        <p className="text-xs font-medium">Figures from the source</p>
        <p className="text-2xs text-muted-foreground">
          Plotted directly from the tables extracted from your documents. Every
          chart names the page it came from.
        </p>
      </div>
      <div className="grid gap-2 lg:grid-cols-2">
        {usable.map((chart) => (
          <SingleChart key={chart.chart_id} chart={chart} />
        ))}
      </div>
    </div>
  );
}
