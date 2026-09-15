# 01 — The Analyst Workflow, Modelled

The purpose of this document is to make the manual job legible enough to automate. Each
section states what a human does, then the exact RESX component that replaces it, then the
failure mode we are guarding against.

---

## Stage 1 — Data Acquisition & Cleaning

### 1.1 Extract

| Source | Library | Notes |
|---|---|---|
| PDF (text) | `pdfplumber` | Keeps character-level bounding boxes — this is what makes a citation anchor possible |
| PDF (scanned) | `pytesseract` + `pdf2image` | OCR confidence recorded per block; blocks below 0.80 flagged for review |
| PDF (tables) | `camelot` (lattice) → `pdfplumber` fallback | Two extractors, compared; mismatch flags the table |
| XLSX / XLS | `openpyxl` / `pandas` | Merged cells resolved, formulas read as values *and* as formulas |
| CSV / TSV | `pandas` with sniffed dialect | Encoding detected via `charset-normalizer` |
| DOCX | `python-docx` | Paragraph index preserved |
| SQL | SQLAlchemy, read-only role | Schema introspected first |
| REST API | `httpx` + a declared response schema | Pagination handled; responses cached with an ETag |

**Failure mode guarded:** a silently mis-parsed table. Mitigation is dual extraction with
comparison, and a hard rule that any table feeding a financial claim must have passed the
comparison or been human-confirmed.

### 1.2 Profile before touching anything

Every ingested table gets a profile written to `dataset_profiles` before any cleaning:

```python
{
  "n_rows": int, "n_cols": int,
  "columns": [{
      "name": str, "inferred_type": str, "null_pct": float,
      "unique_pct": float, "sample": [...],
      "candidate_role": "date" | "currency" | "quantity" | "category" | "id" | "text",
  }],
  "duplicate_row_pct": float,
  "issues": [ {"kind": ..., "column": ..., "severity": ..., "count": int} ],
}
```

The LLM reads this profile — not the raw 50,000 rows — and proposes a repair plan. This is
both a cost decision and an accuracy decision: a profile is a faithful summary, whereas a
truncated sample invites the model to generalize from the first 20 rows.

### 1.3 Scrub

The repair plan is a list of typed, parameterized operations. The LLM chooses *which* ops
and *with what parameters*; it never writes free-form mutation code against production data.

| Op | Parameters | Guard |
|---|---|---|
| `drop_duplicates` | `subset`, `keep` | Row count delta logged; > 20% drop requires confirmation |
| `fill_missing` | `column`, `strategy` (`mean`/`median`/`mode`/`ffill`/`constant`/`drop`) | Never mean-fill a column whose role is `id` or `category` |
| `coerce_type` | `column`, `target_type` | Failures are quarantined to a `_rejected` frame, not coerced to NaN silently |
| `strip_format` | `column`, `pattern` | e.g. currency symbols, thousands separators |
| `clip_outliers` | `column`, `method`, `bounds` | Off by default. Clipping is analysis, not cleaning, and must be explicit |
| `rename` | `mapping` | — |

Every applied op appends to an immutable `transform_log`, so any cleaned dataset can be
reproduced from the raw upload by replaying the log. A number in the final report can be
traced back through the transform log to the exact byte range of the original PDF.

### 1.4 Normalize

- **Dates** → ISO-8601. Ambiguous `03/04/2024` is resolved using document locale evidence
  (other unambiguous dates in the same document); if it stays ambiguous, the field is
  flagged rather than guessed.
- **Currency** → a `Money{amount: Decimal, currency: str}` value object. `Decimal`, never
  `float` — binary floating point cannot represent `0.10`, and financial identities must
  hold to the cent. FX conversion uses a dated rate table, and the rate used is recorded on
  the claim.
- **Units** → `pint` unit registry. Scale words ("in thousands", "$M") are detected from
  table headers and footnotes, which is one of the most common sources of 1000x errors.
- **Entities** → fuzzy resolution (`rapidfuzz`) against a workspace entity table, with a
  confirmation step above a similarity threshold rather than automatic merging.

---

## Stage 2 — Exploratory Data Analysis

All of the following execute as Python in the sandbox and return a `ComputationRecord`.
Nothing here is produced by an LLM.

### 2.1 Descriptive statistics

`mean, median, mode, variance, std, min, max, q1, q3, iqr, skew, kurtosis, n, n_missing`

Reported with `n` always visible. A mean over `n=3` and a mean over `n=30000` are different
kinds of statement, and the UI shows which one you are looking at.

### 2.2 Trend analysis

- OLS slope with confidence interval (`statsmodels`).
- Mann-Kendall test for a monotonic trend — non-parametric, so it does not assume
  linearity the way a naive slope does.
- CAGR for financial series, with the period stated.
- STL decomposition where at least two full cycles exist, to separate trend from seasonality.

**Guard:** "growing" is only asserted when the slope is significant at p < 0.05. Otherwise
the language is "flat within noise", which is a genuinely different finding.

### 2.3 Correlation

Pearson and Spearman, each with `r`, `p`, `n`, and a CI. Reported as association only. The
Critic agent is prompted to reject any causal phrasing that rests on a correlation, because
"marketing spend drives revenue" and "marketing spend correlates with revenue" have very
different decision consequences for a CEO.

### 2.4 Outliers

IQR fence, z-score (|z| > 3), and isolation forest. Points flagged by only one of the three
are surfaced as "method-dependent", because an outlier that only one test finds usually says
more about the test than the business.

---

## Stage 3 — Visualization

Charts are generated in two places for two purposes:

1. **Interactive dashboard** — Recharts in the browser, fed by JSON from the API. This is
   what the user explores.
2. **Report artifacts** — matplotlib in the sandbox, emitted as PNG/SVG, for the exported
   PDF report and for cases where the agent needs to *look* at a plot.

Both are driven from the same computed series so they can never disagree.

The KPI tile contract (matching the reference dashboards):

```
label · value · delta_pct vs. prior period · direction (up/down/flat)
      · headline ("Trending up this month") · subtext ("Visitors for the last 6 months")
      · citation + computation handle on hover
```

`direction` is derived from significance, not from the sign of the delta — a +0.2% move on
noisy data is `flat`, and rendering it as a green up-arrow would be a lie in a green chip.

---

## Stage 4 — Business Intelligence

### 4.1 Gap analysis

For each KPI with a target: `variance = actual - target`, `variance_pct`, and a projected
close date from the current trend. Where a target is absent, the gap is against a stated
benchmark (prior period, or a market figure that the Market agent has cited).

### 4.2 SWOT

Populated from evidence, not from vibes:

- **Strengths / Weaknesses** — internal documents only, each item citing a document span.
- **Opportunities / Threats** — external search results, each item citing a URL and a
  retrieval date.

An empty quadrant is a valid and honest output. Fabricating a fourth "threat" to balance the
grid is exactly the behaviour the Critic exists to stop.

### 4.3 Financial modelling

- P&L reconstruction with the identity assertions from `RESX.md` §6.3.
- Margin waterfall: revenue → gross → operating → net, each step attributed.
- Burn rate: `(opening_cash - closing_cash) / months`, with runway `= cash / burn`.
- Three-scenario forecast (bear / base / bull) with every assumption stated as an explicit,
  editable parameter. A forecast whose assumptions are hidden inside a prompt is unusable
  for a real decision.

### 4.4 Executive reporting

The Synthesizer reduces everything to **five** insights. Five is a deliberate constraint:
it forces prioritization, which is the actual value an analyst adds. Each insight carries:

```
headline · so_what · evidence[citations] · confidence · suggested_owner
         · effort (S/M/L) · expected_impact (quantified where possible)
         · contested? (if the Critic could not reach consensus)
```

Ranking is by `expected_impact × confidence ÷ effort`, computed in code so the ordering is
reproducible rather than a matter of which insight the model happened to write first.
