/**
 * Illustrative data for the landing page, regenerated on each visit.
 *
 * Two things about this file are deliberate.
 *
 * **It is seeded, not `Math.random()` scattered through the components.** One
 * seed produces the whole set, so the revenue series, the margins derived from
 * it, and the cost split all describe the same imaginary company. Independent
 * random draws per chart would show a business whose margins contradict its
 * revenue — on a page whose entire argument is that figures should reconcile.
 *
 * **It is labelled as illustrative wherever it renders.** This is a marketing
 * page for a product that refuses to publish a number it cannot cite; quietly
 * presenting invented figures as if they were real output would be the exact
 * dishonesty the product exists to prevent. Every chart carries the label.
 */

/** A small deterministic PRNG. Same seed, same company. */
function rng(seed: number) {
  let state = seed >>> 0 || 1;
  return () => {
    // xorshift32: cheap, no dependency, and good enough to look organic.
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return ((state >>> 0) % 100_000) / 100_000;
  };
}

export function newSeed(): number {
  return Math.floor(Math.random() * 2_147_483_647) + 1;
}

const MONTHS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

export interface MonthPoint {
  month: string;
  revenue: number;
  cost: number;
  grossMargin: number;
  operatingMargin: number;
  netMargin: number;
}

export interface Slice {
  key: string;
  label: string;
  value: number;
}

export interface AgentPoint {
  key: string;
  label: string;
  claims: number;
  confidence: number;
}

export interface SeverityPoint {
  category: string;
  critical: number;
  serious: number;
  warning: number;
}

export interface LandingData {
  months: MonthPoint[];
  costSplit: Slice[];
  agents: AgentPoint[];
  severity: SeverityPoint[];
  headline: {
    revenue: number;
    growth: number;
    grossMargin: number;
    citations: number;
  };
}

export function generate(seed: number): LandingData {
  const rand = rng(seed);

  // A base and a trend, so the series has a direction rather than being noise.
  const base = 2_400_000 + Math.round(rand() * 1_600_000);
  const trend = 0.012 + rand() * 0.03;

  const months: MonthPoint[] = MONTHS.map((month, index) => {
    const seasonal = 1 + Math.sin((index / 12) * Math.PI * 2) * 0.06;
    const revenue = Math.round(base * (1 + trend * index) * seasonal);

    // Costs are derived from revenue rather than drawn independently, so the
    // margins below cannot contradict the revenue line above them.
    const cogsRate = 0.56 - trend * index * 0.35 + (rand() - 0.5) * 0.02;
    const cost = Math.round(revenue * cogsRate);
    const opex = Math.round(revenue * (0.24 + (rand() - 0.5) * 0.015));
    const other = Math.round(revenue * (0.05 + (rand() - 0.5) * 0.01));

    const gross = revenue - cost;
    const operating = gross - opex;
    const net = operating - other;

    return {
      month,
      revenue,
      cost,
      grossMargin: round(gross / revenue, 4),
      operatingMargin: round(operating / revenue, 4),
      netMargin: round(net / revenue, 4),
    };
  });

  const last = months[months.length - 1];
  const first = months[0];

  const weights = [0.42, 0.24, 0.16, 0.11, 0.07].map(
    (w) => w * (0.85 + rand() * 0.3),
  );
  const total = weights.reduce((sum, w) => sum + w, 0);
  const annualCost = months.reduce((sum, m) => sum + m.cost, 0);
  const costSplit: Slice[] = [
    { key: "demoGoods", label: "Cost of goods" },
    { key: "demoPeople", label: "People" },
    { key: "demoSales", label: "Sales & marketing" },
    { key: "demoTech", label: "Technology" },
    { key: "demoOther", label: "Other" },
  ].map((slice, index) => ({
    ...slice,
    value: Math.round((weights[index] / total) * annualCost),
  }));

  const agents: AgentPoint[] = [
    { key: "demoFinance", label: "Finance" },
    { key: "demoRisk", label: "Risk" },
    { key: "demoMarket", label: "Market" },
    { key: "demoCritic", label: "Review" },
  ].map((agent) => ({
    ...agent,
    claims: 6 + Math.round(rand() * 14),
    confidence: round(0.86 + rand() * 0.13, 3),
  }));

  const severity: SeverityPoint[] = [
    "Customer concentration",
    "Liquidity",
    "Supplier dependency",
    "Compliance",
    "Currency",
  ].map((category) => ({
    category,
    critical: Math.round(rand() * 3),
    serious: 1 + Math.round(rand() * 4),
    warning: 1 + Math.round(rand() * 6),
  }));

  return {
    months,
    costSplit,
    agents,
    severity,
    headline: {
      revenue: months.reduce((sum, m) => sum + m.revenue, 0),
      growth: round(last.revenue / first.revenue - 1, 4),
      grossMargin: last.grossMargin,
      // Always 1.00. It is the product's gate, not a variable to randomise:
      // showing 97% here would advertise a failure.
      citations: 1,
    },
  };
}

function round(value: number, places: number): number {
  const factor = 10 ** places;
  return Math.round(value * factor) / factor;
}

export function formatMoney(value: number): string {
  if (Math.abs(value) >= 1_000_000) return `$${(value / 1_000_000).toFixed(1)}M`;
  if (Math.abs(value) >= 1_000) return `$${Math.round(value / 1_000)}K`;
  return `$${value}`;
}

/** Trailing `n` months, for the range selector. */
export function windowed(months: MonthPoint[], n: number): MonthPoint[] {
  return months.slice(Math.max(0, months.length - n));
}
