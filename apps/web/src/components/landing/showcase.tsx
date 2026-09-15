"use client";

/**
 * The five chart archetypes, on illustrative data that changes each visit.
 *
 * All five draw from the project's validated eight-slot palette through
 * `seriesColor(key)` — never a hex, never an array index. Each chart's series
 * occupy consecutive slots from 1, because the palette is validated on the
 * *adjacent* pairlist: a group assigned slots 3, 5 and 6 would put a pair on
 * screen that no check ever covered.
 *
 * The severity chart is the exception and deliberately so. Its colour encodes
 * **state**, not identity, so it draws from the reserved status palette and
 * ships with labels — status hues are never reused as "series 4".
 *
 * Every card says the data is illustrative. This is a marketing page for a
 * product whose entire claim is that it will not show a figure it cannot cite;
 * presenting invented numbers as real output would be precisely the dishonesty
 * it exists to prevent.
 */

import { useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Label,
  Line,
  LineChart,
  Pie,
  PieChart,
  PolarAngleAxis,
  RadialBar,
  RadialBarChart,
  XAxis,
  YAxis,
} from "recharts";

import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { seriesColor, statusColor } from "@/lib/series";
import {
  formatMoney,
  generate,
  newSeed,
  windowed,
  type LandingData,
} from "@/lib/landing-data";

// --------------------------------------------------------------------------- //
// Shared card
// --------------------------------------------------------------------------- //

function Card({
  title,
  caption,
  action,
  legend,
  children,
}: {
  title: string;
  caption: string;
  action?: React.ReactNode;
  legend?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <figure className="flex flex-col overflow-hidden rounded-xl border border-border bg-card">
      <header className="flex flex-wrap items-start justify-between gap-2 border-b border-border px-4 py-3">
        <div className="flex min-w-0 flex-col gap-0.5">
          <h3 className="text-sm font-medium">{title}</h3>
          <p className="text-2xs text-muted-foreground">{caption}</p>
        </div>
        {action}
      </header>
      <div className="px-2 pb-2 pt-4 sm:px-4">{children}</div>
      {legend ? (
        <figcaption className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-border px-4 py-2.5">
          {legend}
        </figcaption>
      ) : null}
    </figure>
  );
}

function Swatch({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5 text-2xs text-muted-foreground">
      <span className="size-2 rounded-sm" style={{ background: color }} />
      {label}
    </span>
  );
}

// --------------------------------------------------------------------------- //
// 1. Interactive area, with a range selector
// --------------------------------------------------------------------------- //

const RANGES = [
  { value: "12", label: "12 months" },
  { value: "6", label: "6 months" },
  { value: "3", label: "3 months" },
];

function RevenueArea({ data }: { data: LandingData }) {
  const [range, setRange] = useState("12");
  const rows = useMemo(
    () => windowed(data.months, Number(range)),
    [data.months, range],
  );

  const config = {
    revenue: { label: "Revenue", color: seriesColor("demoRevenue") },
    cost: { label: "Cost", color: seriesColor("demoCost") },
  } satisfies ChartConfig;

  return (
    <Card
      title="Revenue and cost"
      caption="One axis — two measures on the same scale, so the gap is the margin"
      action={
        <Select value={range} onValueChange={setRange}>
          <SelectTrigger
            size="sm"
            className="h-7 w-[124px] text-xs"
            aria-label="Select a time range"
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {RANGES.map((r) => (
              <SelectItem key={r.value} value={r.value} className="text-xs">
                {r.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      }
      legend={
        <>
          <Swatch color={seriesColor("demoRevenue")} label="Revenue" />
          <Swatch color={seriesColor("demoCost")} label="Cost" />
        </>
      }
    >
      <ChartContainer config={config} className="h-[220px] w-full">
        <AreaChart data={rows} margin={{ left: 4, right: 8, top: 8 }}>
          <defs>
            {(["demoRevenue", "demoCost"] as const).map((key) => (
              <linearGradient
                key={key}
                id={`fill-${key}`}
                x1="0"
                y1="0"
                x2="0"
                y2="1"
              >
                <stop offset="0%" stopColor={seriesColor(key)} stopOpacity={0.35} />
                <stop offset="100%" stopColor={seriesColor(key)} stopOpacity={0.03} />
              </linearGradient>
            ))}
          </defs>
          <CartesianGrid vertical={false} strokeDasharray="3 3" />
          <XAxis
            dataKey="month"
            tickLine={false}
            axisLine={false}
            tickMargin={8}
            className="text-2xs"
          />
          <YAxis
            width={54}
            tickLine={false}
            axisLine={false}
            tickMargin={6}
            className="text-2xs"
            tickFormatter={(v: number) => formatMoney(v)}
          />
          <ChartTooltip
            cursor={false}
            content={
              <ChartTooltipContent
                indicator="line"
                formatter={(value, name) => (
                  <span className="flex w-full justify-between gap-3">
                    <span className="text-muted-foreground">
                      {name === "revenue" ? "Revenue" : "Cost"}
                    </span>
                    <span className="font-medium tabular-nums">
                      {formatMoney(Number(value))}
                    </span>
                  </span>
                )}
              />
            }
          />
          {/* Cost drawn first so the revenue band reads on top of it. */}
          <Area
            dataKey="cost"
            type="monotone"
            stroke={seriesColor("demoCost")}
            strokeWidth={2}
            fill="url(#fill-demoCost)"
            isAnimationActive={false}
          />
          <Area
            dataKey="revenue"
            type="monotone"
            stroke={seriesColor("demoRevenue")}
            strokeWidth={2}
            fill="url(#fill-demoRevenue)"
            isAnimationActive={false}
          />
        </AreaChart>
      </ChartContainer>
    </Card>
  );
}

// --------------------------------------------------------------------------- //
// 2. Multi-line
// --------------------------------------------------------------------------- //

function MarginLines({ data }: { data: LandingData }) {
  const config = {
    grossMargin: { label: "Gross", color: seriesColor("demoGross") },
    operatingMargin: { label: "Operating", color: seriesColor("demoOperating") },
    netMargin: { label: "Net", color: seriesColor("demoNet") },
  } satisfies ChartConfig;

  return (
    <Card
      title="Margin trend"
      caption="Three ratios, one shared percentage axis"
      legend={
        <>
          <Swatch color={seriesColor("demoGross")} label="Gross" />
          <Swatch color={seriesColor("demoOperating")} label="Operating" />
          <Swatch color={seriesColor("demoNet")} label="Net" />
        </>
      }
    >
      <ChartContainer config={config} className="h-[220px] w-full">
        <LineChart data={data.months} margin={{ left: 4, right: 12, top: 8 }}>
          <CartesianGrid vertical={false} strokeDasharray="3 3" />
          <XAxis
            dataKey="month"
            tickLine={false}
            axisLine={false}
            tickMargin={8}
            className="text-2xs"
          />
          <YAxis
            width={46}
            tickLine={false}
            axisLine={false}
            tickMargin={6}
            className="text-2xs"
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
          />
          <ChartTooltip
            cursor={false}
            content={
              <ChartTooltipContent
                indicator="line"
                formatter={(value, name) => (
                  <span className="flex w-full justify-between gap-3">
                    <span className="text-muted-foreground">
                      {String(name).replace("Margin", "")}
                    </span>
                    <span className="font-medium tabular-nums">
                      {(Number(value) * 100).toFixed(1)}%
                    </span>
                  </span>
                )}
              />
            }
          />
          {(
            [
              ["grossMargin", "demoGross"],
              ["operatingMargin", "demoOperating"],
              ["netMargin", "demoNet"],
            ] as const
          ).map(([field, key]) => (
            <Line
              key={field}
              dataKey={field}
              type="monotone"
              stroke={seriesColor(key)}
              strokeWidth={2}
              dot={false}
              // Recharts can latch a stale clip-path width when the container
              // is measured mid-animation, which paints a sliver of the line.
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ChartContainer>
    </Card>
  );
}

// --------------------------------------------------------------------------- //
// 3. Donut with a centre total
// --------------------------------------------------------------------------- //

function CostDonut({ data }: { data: LandingData }) {
  const total = data.costSplit.reduce((sum, s) => sum + s.value, 0);
  const config = { value: { label: "Cost" } } satisfies ChartConfig;

  return (
    <Card
      title="Where the money goes"
      caption="Five parts of one whole — the only case a donut earns"
      legend={
        <>
          {data.costSplit.map((slice) => (
            <Swatch
              key={slice.key}
              color={seriesColor(slice.key)}
              label={slice.label}
            />
          ))}
        </>
      }
    >
      <ChartContainer config={config} className="mx-auto h-[220px] w-full">
        <PieChart>
          <ChartTooltip
            cursor={false}
            content={
              <ChartTooltipContent
                nameKey="label"
                hideLabel
                formatter={(value) => (
                  <span className="font-medium tabular-nums">
                    {formatMoney(Number(value))}
                  </span>
                )}
              />
            }
          />
          <Pie
            data={data.costSplit}
            dataKey="value"
            nameKey="label"
            // A wide ring: the centre has to hold the total without colliding
            // with the arc.
            innerRadius={58}
            outerRadius={88}
            // A 2px surface gap between segments, per the mark spec.
            paddingAngle={1.5}
            strokeWidth={2}
            isAnimationActive={false}
          >
            {data.costSplit.map((slice) => (
              <Cell key={slice.key} fill={seriesColor(slice.key)} />
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
                      {formatMoney(total)}
                    </tspan>
                    <tspan
                      x={cx}
                      y={cy + 14}
                      className="fill-muted-foreground text-2xs"
                    >
                      total cost
                    </tspan>
                  </text>
                );
              }}
            />
          </Pie>
        </PieChart>
      </ChartContainer>
    </Card>
  );
}

// --------------------------------------------------------------------------- //
// 4. Radial, with direct labels
// --------------------------------------------------------------------------- //

function ConfidenceRadial({ data }: { data: LandingData }) {
  const rows = useMemo(
    () =>
      data.agents.map((agent) => ({
        ...agent,
        // The radial axis is a percentage, so confidence maps onto it directly.
        pct: Math.round(agent.confidence * 100),
        fill: seriesColor(agent.key),
      })),
    [data.agents],
  );
  const config = { pct: { label: "Confidence" } } satisfies ChartConfig;

  return (
    <Card
      title="Confidence by specialist"
      caption="Stated confidence, not a score the model was asked to inflate"
      legend={
        <>
          {rows.map((row) => (
            <Swatch
              key={row.key}
              color={row.fill}
              label={`${row.label} ${row.pct}%`}
            />
          ))}
        </>
      }
    >
      <ChartContainer config={config} className="mx-auto h-[220px] w-full">
        <RadialBarChart
          data={rows}
          innerRadius="28%"
          outerRadius="98%"
          startAngle={90}
          endAngle={-270}
        >
          {/* The domain is fixed to 0-100 so the arcs are comparable between
              reloads rather than rescaling to whatever the maximum happens to
              be. */}
          <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
          <ChartTooltip
            cursor={false}
            content={
              <ChartTooltipContent
                nameKey="label"
                hideLabel
                formatter={(value, _name, item) => {
                  const row = item?.payload as (typeof rows)[number] | undefined;
                  return (
                    <span className="flex w-full justify-between gap-3">
                      <span className="text-muted-foreground">
                        {row?.label}
                      </span>
                      <span className="font-medium tabular-nums">
                        {value}% · {row?.claims} findings
                      </span>
                    </span>
                  );
                }}
              />
            }
          />
          <RadialBar
            dataKey="pct"
            background
            cornerRadius={4}
            isAnimationActive={false}
          >
            {rows.map((row) => (
              <Cell key={row.key} fill={row.fill} />
            ))}
          </RadialBar>
        </RadialBarChart>
      </ChartContainer>
    </Card>
  );
}

// --------------------------------------------------------------------------- //
// 5. Stacked bar — colour encodes STATE, from the reserved status palette
// --------------------------------------------------------------------------- //

function SeverityBars({ data }: { data: LandingData }) {
  const config = {
    critical: { label: "Critical", color: statusColor("critical") },
    serious: { label: "Serious", color: statusColor("serious") },
    warning: { label: "Warning", color: statusColor("warning") },
  } satisfies ChartConfig;

  const rows = useMemo(
    () =>
      [...data.severity].sort(
        (a, b) =>
          b.critical * 100 + b.serious * 10 + b.warning -
          (a.critical * 100 + a.serious * 10 + a.warning),
      ),
    [data.severity],
  );

  return (
    <Card
      title="Risk findings by category"
      caption="Sorted by exposure — an unsorted category axis makes the reader do the ranking"
      legend={
        <>
          {/* Status colour never travels alone: each swatch is labelled, so
              severity is not encoded by hue only. */}
          <Swatch color={statusColor("critical")} label="Critical" />
          <Swatch color={statusColor("serious")} label="Serious" />
          <Swatch color={statusColor("warning")} label="Warning" />
        </>
      }
    >
      <ChartContainer config={config} className="h-[220px] w-full">
        <BarChart
          data={rows}
          layout="vertical"
          margin={{ left: 4, right: 16, top: 4, bottom: 4 }}
        >
          <CartesianGrid horizontal={false} strokeDasharray="3 3" />
          <XAxis type="number" hide />
          <YAxis
            type="category"
            dataKey="category"
            width={132}
            tickLine={false}
            axisLine={false}
            className="text-2xs"
          />
          <ChartTooltip
            cursor={false}
            content={<ChartTooltipContent indicator="line" />}
          />
          {(["critical", "serious", "warning"] as const).map((key, index) => (
            <Bar
              key={key}
              dataKey={key}
              stackId="severity"
              fill={statusColor(key)}
              // A 2px surface gap between stacked segments, and only the
              // outermost segment gets the rounded data-end.
              stroke="var(--card)"
              strokeWidth={2}
              radius={index === 2 ? [0, 4, 4, 0] : 0}
              isAnimationActive={false}
            />
          ))}
        </BarChart>
      </ChartContainer>
    </Card>
  );
}

// --------------------------------------------------------------------------- //
// The section
// --------------------------------------------------------------------------- //

export default function Showcase() {
  // Generated once per mount. The module is loaded with `ssr: false`, so there
  // is no server render to disagree with and no hydration mismatch — which is
  // why this can be random at all.
  const [data] = useState(() => generate(newSeed()));

  return (
    <section id="dashboard" className="border-t border-border px-6 py-24">
      <div className="mx-auto max-w-6xl">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h2 className="text-2xl font-medium tracking-tight md:text-3xl">
              Every figure, in a form you can interrogate
            </h2>
            <p className="mt-3 max-w-2xl text-sm text-muted-foreground">
              Hover any mark for the underlying value. Change the range. Each
              chart carries a legend and reads the same in light and dark.
            </p>
          </div>
          <p className="rounded-full border border-border bg-card px-3 py-1 text-2xs text-muted-foreground">
            Illustrative data · regenerated on every visit
          </p>
        </div>

        <div className="mt-10 grid gap-4 lg:grid-cols-2">
          <div className="lg:col-span-2">
            <RevenueArea data={data} />
          </div>
          <MarginLines data={data} />
          <CostDonut data={data} />
          <ConfidenceRadial data={data} />
          <SeverityBars data={data} />
        </div>

        <p className="mt-6 text-2xs leading-relaxed text-muted-foreground">
          These figures are invented and marked as such. In the product every
          number on screen carries the source it came from and the calculation
          that produced it — which is the whole reason we will not dress up a
          demo as a real analysis.
        </p>
      </div>
    </section>
  );
}
