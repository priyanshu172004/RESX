# 06 — Design System

The target is the aesthetic of the reference dashboards: **near-black canvas, hairline
separators, dense tabular data, and colour used only where it carries meaning.** The
interface should read as a precision instrument, not a marketing page.

---

## 1. Principles

1. **Monochrome first.** Greyscale carries structure — surfaces, borders, text hierarchy.
   Colour is reserved for meaning: delta direction, severity, confidence, series identity.
   A chart with five decorative colours is a design failure.
2. **Typography is the design.** One family, tabular numerals on every figure, a tight
   scale. No decorative type, no more than two weights in a view.
3. **Density with air.** Analysts want data on screen. Density comes from disciplined
   spacing and 1px separators, never from type below 12px.
4. **Motion clarifies causality.** 150–250ms, ease-out, and only where it explains a state
   change. Nothing bounces; nothing spins for decoration. `prefers-reduced-motion` honoured.
5. **Every number is inspectable.** Hovering any figure reveals its citation and its
   computation. This is a design requirement because it is the product's core promise.

---

## 2. Tokens

Tailwind v4 uses CSS-first configuration, so tokens are declared as custom properties in
`app/globals.css` and consumed through `@theme`.

### Surfaces and ink

| Role | Light | Dark | Use |
|---|---|---|---|
| `--background` | `#fcfcfb` | `#08080a` | Page canvas |
| `--card` | `#ffffff` | `#0e0e11` | Card / panel surface (**the chart surface**) |
| `--muted` | `#f4f4f2` | `#16161a` | Inset rows, table header, hover |
| `--border` | `#e6e6e3` | `#1f1f24` | Hairline separators — 1px, never heavier |
| `--foreground` | `#0b0b0b` | `#fafafa` | Primary ink |
| `--muted-foreground` | `#52514e` | `#a1a1aa` | Labels, axis text, secondary ink |
| `--ring` | `#2a78d6` | `#3987e5` | Focus ring — always visible, never removed |

Dark is the **default** and is a designed mode, not an inverted light mode.

### Series palette (validated)

Both columns are the same eight hues, stepped per surface. **These values are validated —
do not substitute by eye.** Re-run
`node scripts/validate_palette.js "<hex,...>" --mode dark --surface "#0e0e11"` after any
change.

| Slot | Hue | Light (`#fcfcfb`) | Dark (`#0e0e11`) |
|---|---|---|---|
| `--chart-1` | blue | `#2a78d6` | `#3987e5` |
| `--chart-2` | orange | `#eb6834` | `#d95926` |
| `--chart-3` | aqua | `#1baf7a` | `#199e70` |
| `--chart-4` | yellow | `#eda100` | `#c98500` |
| `--chart-5` | magenta | `#e87ba4` | `#d55181` |
| `--chart-6` | green | `#008300` | `#008300` |
| `--chart-7` | violet | `#4a3aa7` | `#9085e9` |
| `--chart-8` | red | `#e34948` | `#e66767` |

Validator results, recorded so a future change can be compared against them:

- **Dark, adjacent pairs, 8 slots** — all six checks pass. Worst adjacent CVD ΔE 8.4
  (protan), worst normal-vision ΔE 19.3, all slots ≥ 3:1 contrast on `#0e0e11`.
- **Dark, all pairs, first 3 slots** — pass. Worst CVD ΔE 9.4, normal-vision ΔE 20.9.
- **Light, adjacent pairs, 8 slots** — pass, with a **contrast relief** on aqua (2.74),
  yellow (2.11), and magenta (2.62) against the light surface.

Two rules follow directly from those results and are binding:

- **Scatter, bubble, and small-multiple forms cap at three series** (slots 1–3), because
  those forms put every pair on screen at once and the full eight cannot clear the
  all-pairs floors. Beyond three, fold into "Other" or facet.
- **In light mode, any chart using aqua, yellow, or magenta must ship visible direct
  labels or the table view.** The contrast WARN is a relief obligation, not a dismissal.

Assignment is by **fixed slot order, never cycled**. Colour follows the entity, so a filter
that removes a series must not repaint the survivors — series colour is looked up by key,
not by array index.

### Status palette (fixed, never themed, never reused as a series)

| Role | Hex | Meaning in RESX |
|---|---|---|
| `good` | `#0ca30c` | Confirmed claim, passing identity, favourable delta |
| `warning` | `#fab219` | Low confidence, method-dependent outlier, stale source |
| `serious` | `#ec835a` | Contested claim, degraded agent branch |
| `critical` | `#d03b3b` | Refuted claim, failed identity, high severity × likelihood risk |

Always paired with an **icon and a text label**. Status never communicates through colour
alone, and on the light surface `warning` and `serious` are deliberately sub-3:1, which is
exactly why the icon-plus-label pairing is mandatory.

### Sequential and diverging

- **Sequential** (magnitude — heatmaps, confidence density): one hue, blue, light → dark,
  steps `#cde2fb → #0d366b`. For an ordinal ramp, start no lighter than `#86b6ef` on light
  and no darker than `#184f95` on dark.
- **Diverging** (polarity — variance vs. target, sentiment): blue ↔ red with a **neutral
  grey midpoint** (`#f0efec` light, `#383835` dark). Never a hue at the midpoint; never a
  rainbow.

### Type

`Geist Sans` with `Geist Mono` for figures and code; system fallbacks declared.

| Token | Size / line | Use |
|---|---|---|
| `text-2xs` | 11px / 16 | Table micro-labels, axis ticks |
| `text-xs` | 12px / 16 | Chips, badges, metadata |
| `text-sm` | 13px / 20 | Body, table cells, controls — the workhorse |
| `text-base` | 14px / 22 | Card titles |
| `text-lg` | 16px / 24 | Section headings |
| `text-3xl` | 30px / 36 | KPI values |
| `text-4xl` | 36px / 40 | Hero figure |

**`font-variant-numeric: tabular-nums` is applied globally to every numeric element.** In a
dashboard where figures stack in columns, proportional digits make them impossible to
compare down the column.

### Spacing, radius, elevation

4px base scale. Radius: `sm 6px`, `md 8px`, `lg 12px` (cards), `full` (chips).

**No shadows.** Depth is communicated by surface value and a 1px border, matching the
reference screenshots. A drop shadow on a near-black surface reads as smudge.

---

## 3. Component inventory

Built on shadcn/ui, extended where RESX needs something specific.

### Layout

| Component | Notes |
|---|---|
| `AppSidebar` | Collapsible, grouped nav (Workspace / Documents / Analysis), workspace switcher at top, Settings and Help pinned to the bottom — the reference layout |
| `SiteHeader` | Breadcrumb, global search (`⌘K`), theme toggle, primary action |
| `PageShell` | Max-width container, consistent gutters, section rhythm |

### Data display

| Component | Notes |
|---|---|
| `KpiCard` | label · value · delta chip · trend headline · subtext. `direction` derived from **significance**, not the sign of the delta |
| `DataTable` | TanStack Table: sorting, column visibility ("Customize Columns"), row selection, drag handles, tabbed sub-views with counts, sticky header, virtualized past 200 rows |
| `StatusBadge` | Icon + label + colour, from the status palette |
| `CitationPopover` | The signature component. Hover a figure → document, page, verbatim quote, and the computation that produced it |
| `ConfidenceMeter` | Bounded 0–1 bar, sequential ramp, numeric value always shown |
| `EmptyState` | Explains what to do next; never a bare "no data" |

### Analysis

| Component | Notes |
|---|---|
| `RunConsole` | Live SSE timeline of node starts, tool calls, claims, verdicts, and debate rounds. Framer Motion list with layout animation |
| `AgentCard` | Per-agent status, token and dollar spend, claim count, degraded flag |
| `DebateThread` | Claim → Critic rationale → defence → verdict, as an evidence-linked thread |
| `InsightCard` | Headline, so-what, evidence chips, confidence, owner, effort, impact |
| `SwotGrid` | Four quadrants, every item citation-linked; an empty quadrant renders as empty |

### Charts

One `ChartCard` wrapper (title, description, optional range `Select`, footer note) around
the shadcn `ChartContainer`, so every chart in the product shares padding, legend position,
tooltip style, and grid treatment.

---

## 4. Chart specification

### Form heuristic

| The data's job | Form | RESX use |
|---|---|---|
| Change over time, 2–4 series | **Line**, monotone, 2px, no dots | Revenue vs. cost trend |
| Cumulative or part-to-whole over time | **Stacked area** with a gradient fill | Visitor / volume composition |
| Compare categories | **Horizontal bar**, sorted by value | Risk findings by category |
| Composition, ≤ 5 parts | **Donut** with a centre total | Spend mix |
| Composition, > 5 parts | Bar — *not* a donut | — |
| Progress toward a bound | **Radial** | Analysis completion, confidence |
| Relationship between two measures | **Scatter** with a fitted line, `r` and `p` annotated | Marketing spend vs. revenue |
| A single headline figure | **Stat tile**, not a chart | KPI row |

### Mark specs (from the marks reference, binding)

- Lines: 2px, `type="monotone"`, dots hidden by default, active dot ≥ 8px on hover.
- Bars: 4px rounded on the **data end only**, square against the baseline. In a stacked
  bar, the top segment rounds its top and the bottom segment rounds its bottom, exactly as
  the reference `chart5` does.
- **A 2px surface-coloured gap between adjacent fills** — stacked segments and neighbouring
  bars alike — so boundaries read without a border.
- Overlapping marks (scatter, active dots) carry a 2px surface-coloured ring.
- Grid: horizontal only (`vertical={false}`), border-token colour, recessive. No axis lines,
  no tick lines; `tickMargin={8}`.
- Labels selectively, never a number on every point.

### Axes and scales

- **Never a dual y-axis.** Two measures of different magnitude become two charts, small
  multiples, or an indexed series on a common base. This is the single most common
  dashboard error and it is prohibited.
- Bar charts baseline at zero, always. Line charts may crop the domain, but the axis is
  labelled so the crop is visible.
- Time on x, ordered ascending, formatted by density (`Apr 3` for daily, `Apr` for monthly).

### Legend and identity

For **two or more series** a legend is always present, and where there are four or fewer the
series are **also direct-labelled**, so identity never rests on colour alone. A single-series
chart needs no legend — the card title names it.

### Interaction

Every chart ships a hover layer by default; the only exception is a bare stat tile with no
plot.

- Line and area: shared crosshair with a tooltip listing every series at that x.
- Bar, dot, cell: per-mark tooltip.
- `cursor={false}` on the Recharts tooltip — the default grey cursor band is heavy noise on
  a near-black surface.
- Hit targets larger than the mark.
- Filters in one row above the charts, never interleaved: date range, then dimension
  selects. Filter state is URL-serialized so any view is a shareable link.
- Every chart offers a **table view** toggle, which also discharges the light-mode contrast
  relief obligation.

### Accessibility

`accessibilityLayer` on every Recharts chart (keyboard-navigable series), the table view as
a non-visual equivalent, texture fill available at 45°/135° for the CVD, print, and
forced-colors cases, and a visible focus ring on every interactive element.

---

## 5. Motion

Framer Motion, used sparingly.

| Interaction | Spec |
|---|---|
| Page / section entry | Fade + 4px rise, 200ms ease-out, 40ms stagger across cards |
| KPI value change | Count-up over 400ms, `tabular-nums` so width never shifts |
| Run console events | `AnimatePresence` list insert, 150ms, layout animation on reflow |
| Sidebar collapse | Width transition 200ms ease-out |
| Tab / view switch | Crossfade 120ms; no slide, which implies a spatial relationship that is not there |
| Chart mount | Recharts native animation, 300ms, once — never on re-render, which makes filtering feel broken |
| Hover | 100ms background transition; no scale transforms on data marks |

```tsx
const reduce = useReducedMotion()
const transition = reduce ? { duration: 0 } : { duration: 0.2, ease: "easeOut" }
```

`prefers-reduced-motion` disables all of it. Chart animation is also disabled on filter
re-render, so a filter change reads as instantaneous rather than as a reload.

---

## 6. Anti-patterns (rejected in review)

- A dual-axis chart. Two charts instead.
- More than eight series, or a generated ninth hue. Fold into "Other" or facet.
- A donut with more than five slices.
- A status colour reused as a series colour.
- A series colour applied to text. Values and labels wear ink tokens; a coloured mark beside
  them carries identity.
- A green up-arrow on a statistically insignificant delta.
- A truncated y-axis on a bar chart.
- A drop shadow.
- A number without tabular numerals.
- A spinner where a skeleton or a live event stream would show the actual work.
- Colour as the sole carrier of any meaning, anywhere.
