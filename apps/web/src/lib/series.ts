/**
 * Series colour assignment.
 *
 * The rule this file exists to enforce: **colour follows the entity, never its
 * rank or its array index.** If a filter removes one series, the survivors must
 * keep the colours they had. Looking a colour up by `data[i]` breaks that the
 * first time a filter changes the series count, so nothing in this codebase
 * indexes the palette directly — it calls `seriesColor(key)`.
 *
 * The eight slots and their validation results are documented in
 * docs/06-DESIGN-SYSTEM.md §2. Do not add a ninth.
 */

export const SERIES_SLOT_COUNT = 8;

/**
 * The all-pairs cap. Scatter, bubble, and small-multiple forms put every pair
 * of series on screen simultaneously, and only the first three slots clear the
 * all-pairs CVD and normal-vision floors in both themes. Past three, fold into
 * "Other" or facet — never generate a new hue.
 */
export const ALL_PAIRS_SERIES_CAP = 3;

/** Slots that fall below 3:1 against the LIGHT surface. */
const LIGHT_RELIEF_SLOTS = new Set([3, 4, 5]);

/**
 * A stable registry of every series key the product renders, in fixed slot
 * order. Adding a series means adding it here — which is deliberate friction,
 * because it forces the eight-slot ceiling to be a conscious decision.
 */
export const SERIES_REGISTRY = [
  // Revenue vs. cost over time
  "revenue",
  "cogs",
  // Margin waterfall
  "grossProfit",
  "operatingIncome",
  "netIncome",
  // Cash
  "cash",
  "burn",
  // Cost composition
  "costGoods",
  "costOperating",
  "costSales",
  "costRnd",
  "costGa",
  // Agents
  "manager",
  "finance",
  "risk",
  "news",
  "workflow",
  "market",
  "critic",
  "synthesizer",
  // Generic comparison
  "actual",
  "target",
  "forecast",
  "prior",
] as const;

export type SeriesKey = (typeof SERIES_REGISTRY)[number] | string;

/**
 * Explicit slot assignments, grouped by CO-OCCURRENCE.
 *
 * Two rules govern this table:
 *
 * 1. Keys that appear in the same chart must take **distinct** slots.
 * 2. Keys that never co-occur may reuse a slot — which is how the product
 *    names more than eight series while staying inside eight validated hues.
 *
 * Within a group the slots are **consecutive from 1**, because the palette is
 * validated on the *adjacent* pairlist. Assigning a group non-consecutive
 * slots (say 3, 5, 6) puts a pair on screen that was never validated together:
 * slot 3 (aqua) beside slot 6 (green) reads as two greens and fails the
 * normal-vision floor. Consecutive-from-1 is the only assignment the
 * validation actually covers.
 */
const SLOT_BY_KEY: Record<string, number> = {
  // Revenue vs. cost over time — 2 series.
  revenue: 1,
  cogs: 2,

  // Margin waterfall — 3 series, co-occurring only with each other.
  grossProfit: 1,
  operatingIncome: 2,
  netIncome: 3,

  // Cash — 2 series.
  cash: 1,
  burn: 2,

  // Cost composition — 5 slices, so 5 consecutive slots.
  costGoods: 1,
  costOperating: 2,
  costSales: 3,
  costRnd: 4,
  costGa: 5,

  // Agents — all eight can appear together in the run console, which is
  // exactly the eight-slot ceiling. There is no ninth agent.
  finance: 1,
  risk: 2,
  news: 3,
  workflow: 4,
  market: 5,
  critic: 6,
  synthesizer: 7,
  manager: 8,

  // Generic comparison.
  actual: 1,
  target: 2,
  forecast: 3,
  prior: 4,

  // Landing-page illustration. Registered explicitly rather than left to the
  // hash fallback, because the fallback can hand a co-occurring group
  // non-consecutive slots -- and the palette is only validated on the
  // *adjacent* pairlist. Each group below is consecutive from 1 and appears
  // only within its own chart.
  demoRevenue: 1,
  demoCost: 2,

  demoGross: 1,
  demoOperating: 2,
  demoNet: 3,

  demoGoods: 1,
  demoPeople: 2,
  demoSales: 3,
  demoTech: 4,
  demoOther: 5,

  demoFinance: 1,
  demoRisk: 2,
  demoMarket: 3,
  demoCritic: 4,
};

/** Deterministic fallback for a key not in the registry — stable across renders. */
function fallbackSlot(key: string): number {
  let h = 0;
  for (let i = 0; i < key.length; i += 1) {
    h = (h * 31 + key.charCodeAt(i)) % 1_000_003;
  }
  return (h % SERIES_SLOT_COUNT) + 1;
}

export function seriesSlot(key: SeriesKey): number {
  return SLOT_BY_KEY[key] ?? fallbackSlot(String(key));
}

/**
 * The CSS variable for a series. Returns `var(--chart-N)` so the value
 * resolves per theme at paint time — a chart never hard-codes a hex, and the
 * light/dark swap needs no JavaScript.
 *
 * It must be the RAW token (`--chart-1`), not the Tailwind theme name
 * (`--color-chart-1`). The theme block is declared `@theme inline`, which
 * inlines values into generated utilities and deliberately does NOT emit the
 * `--color-*` custom properties — so `var(--color-chart-1)` resolves to
 * nothing at runtime and Recharts silently falls back to grey. The raw tokens
 * are the ones actually present in `:root` and `.dark`.
 */
export function seriesColor(key: SeriesKey): string {
  return `var(--chart-${seriesSlot(key)})`;
}

/**
 * Does this set of series oblige visible direct labels or a table view in
 * light mode? True when any series lands on a slot whose light step is below
 * 3:1 against the surface. The contrast WARN from the validator is a relief
 * obligation, not something to dismiss — so the chart card reads this and
 * turns the table-view affordance on rather than leaving it optional.
 */
export function needsLightModeRelief(keys: SeriesKey[]): boolean {
  return keys.some((k) => LIGHT_RELIEF_SLOTS.has(seriesSlot(k)));
}

/**
 * Guard for the all-pairs forms. Returns the series to render plus whatever
 * was cut, so the caller can fold the remainder into "Other" instead of
 * silently dropping it.
 */
export function capForAllPairs<T extends { key: SeriesKey }>(
  series: T[],
): { rendered: T[]; folded: T[] } {
  if (series.length <= ALL_PAIRS_SERIES_CAP) {
    return { rendered: series, folded: [] };
  }
  return {
    rendered: series.slice(0, ALL_PAIRS_SERIES_CAP),
    folded: series.slice(ALL_PAIRS_SERIES_CAP),
  };
}

/**
 * Slot assignment for a chart whose series keys cannot be registered ahead of
 * time — the report charts, whose keys are the column names out of whatever
 * document the user uploaded.
 *
 * `seriesColor` alone is wrong for these. Its hash fallback is stable, which
 * is what it promises, but it can hand a co-occurring group non-consecutive
 * slots (say 3, 5, 6) — and the palette is validated on the *adjacent*
 * pairlist only, so slot 3 beside slot 6 is a pair nothing ever checked and
 * reads as two greens. Consecutive-from-1 is the only assignment the
 * validation covers, and this returns that.
 *
 * A registered key keeps its registered slot when that slot is still free in
 * this chart, so a column actually called "revenue" gets the revenue colour.
 * Everything else takes the lowest free slot in order.
 *
 * This is not colour-by-index in the sense the rule forbids. That rule exists
 * so a filter removing one series cannot repaint the survivors; a report chart
 * has a fixed series set decided at derivation and no filter, so the mapping
 * is stable for the life of the chart. It is keyed by name, not position,
 * within that set.
 */
export function assignChartSlots(keys: SeriesKey[]): Record<string, string> {
  const unique = [...new Set(keys.map(String))].slice(0, SERIES_SLOT_COUNT);
  const taken = new Set<number>();
  const out: Record<string, string> = {};

  for (const key of unique) {
    const preferred = SLOT_BY_KEY[key];
    if (preferred !== undefined && preferred <= unique.length && !taken.has(preferred)) {
      taken.add(preferred);
      out[key] = `var(--chart-${preferred})`;
    }
  }
  let next = 1;
  for (const key of unique) {
    if (out[key]) continue;
    while (taken.has(next)) next += 1;
    taken.add(next);
    out[key] = `var(--chart-${next})`;
  }
  return out;
}

/** Status palette — reserved, and never returned by `seriesColor`. */
export type StatusRole = "good" | "warning" | "serious" | "critical";

export function statusColor(role: StatusRole): string {
  return `var(--${role})`;
}
