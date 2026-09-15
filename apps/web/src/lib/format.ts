/**
 * Presentation-layer formatting only.
 *
 * Nothing here computes. All arithmetic in RESX happens in the Python sandbox
 * and arrives as a computed value with a `computation_id` attached — see
 * docs/04-ACCURACY-VALIDATION.md §3. These helpers exist to render a number
 * that has already been decided, and rounding happens here and nowhere else.
 */

export type TrendDirection = "up" | "down" | "flat";

const currencyFormatters = new Map<string, Intl.NumberFormat>();

function currencyFormatter(currency: string, maximumFractionDigits: number) {
  const key = `${currency}:${maximumFractionDigits}`;
  let f = currencyFormatters.get(key);
  if (!f) {
    f = new Intl.NumberFormat("en-US", {
      style: "currency",
      currency,
      minimumFractionDigits: maximumFractionDigits,
      maximumFractionDigits,
    });
    currencyFormatters.set(key, f);
  }
  return f;
}

export function formatCurrency(
  value: number,
  currency = "USD",
  fractionDigits = 2,
) {
  return currencyFormatter(currency, fractionDigits).format(value);
}

/** Compact form for KPI tiles and axis ticks, where column width is scarce. */
export function formatCompactCurrency(value: number, currency = "USD") {
  const abs = Math.abs(value);
  const sign = value < 0 ? "-" : "";
  const symbol = currency === "USD" ? "$" : `${currency} `;

  if (abs >= 1_000_000_000) return `${sign}${symbol}${(abs / 1_000_000_000).toFixed(2)}B`;
  if (abs >= 1_000_000) return `${sign}${symbol}${(abs / 1_000_000).toFixed(2)}M`;
  if (abs >= 10_000) return `${sign}${symbol}${(abs / 1_000).toFixed(1)}K`;
  return formatCurrency(value, currency, abs % 1 === 0 ? 0 : 2);
}

export function formatNumber(value: number, fractionDigits = 0) {
  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  }).format(value);
}

export function formatCompactNumber(value: number) {
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

/** Always signed — a delta without its sign is ambiguous. */
export function formatPercent(value: number, fractionDigits = 1) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(fractionDigits)}%`;
}

export function formatMonths(value: number) {
  return `${value.toFixed(1)} mo`;
}

/**
 * Direction is derived from SIGNIFICANCE, not from the sign of the delta.
 *
 * This is a correctness rule dressed as a formatting helper: rendering a
 * +0.2% move on noisy data as a green up-arrow asserts a trend the data does
 * not support. Below the threshold the honest answer is "flat".
 *
 * `significant` comes from the backend (an OLS slope with p < 0.05, or a
 * Mann-Kendall test — see docs/01-ANALYST-WORKFLOW.md §2.2). When it is
 * absent we fall back to a magnitude threshold, which is a weaker test and is
 * only used for fixture data.
 */
export function trendDirection(
  deltaPct: number,
  significant?: boolean,
  threshold = 1,
): TrendDirection {
  const meaningful = significant ?? Math.abs(deltaPct) >= threshold;
  if (!meaningful) return "flat";
  return deltaPct > 0 ? "up" : "down";
}

/**
 * Whether a positive delta is good news. Cost and churn go the other way, so
 * the colour of a delta chip cannot be read off its sign alone.
 */
export function deltaSentiment(
  direction: TrendDirection,
  higherIsBetter = true,
): "good" | "critical" | "neutral" {
  if (direction === "flat") return "neutral";
  const favourable = direction === "up" ? higherIsBetter : !higherIsBetter;
  return favourable ? "good" : "critical";
}

export function formatDateShort(input: string | Date) {
  const date = typeof input === "string" ? new Date(input) : input;
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

export function formatDateLong(input: string | Date) {
  const date = typeof input === "string" ? new Date(input) : input;
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

/** Renders a correlation with its significance, because `r` alone is not a finding. */
export function formatCorrelation(r: number, p: number, n: number) {
  const pText = p < 0.001 ? "p < 0.001" : `p = ${p.toFixed(3)}`;
  return `r = ${r.toFixed(2)} · ${pText} · n = ${n}`;
}
