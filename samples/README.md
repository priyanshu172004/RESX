# Sample corpus — Aurora Components Pvt Ltd, FY2025

A synthetic but internally consistent business. Built to exercise every part of
the analysis pipeline rather than to look impressive: the numbers tie together,
the trends are real, and there are genuine problems for the agents to find.

## Files

| File | Shape | What it exercises |
|---|---|---|
| `quarterly_performance_fy2025.csv` | time series | currency + percentage + count on separate axes; trend claims |
| `regional_breakdown_fy2025.csv` | category breakdown | bar and donut charts; comparison claims |
| `customer_economics_fy2025.csv` | time series | multi-series currency chart (CAC vs LTV) |
| `operating_costs_fy2025.csv` | time series | 3-series currency chart; cost-growth claims |
| `aurora_components_fy2025_review.md` | narrative | qualitative context for insights, risks, SWOT, recommendations |

Total: 13 charts supported (line, bar, donut), 4 datasets, 5 sources.

## The story in the data

Every figure is consistent with every other, so a claim checked against a second
table will hold up rather than contradicting.

- Revenue grows 16.7% for the year, but quarterly growth **decelerates**:
  7.9% → 9.0% → 3.2%
- Gross margin **compresses every quarter**: 34.2% → 29.8%
- Volume grows faster than revenue, so realisation per unit falls 5.0%
- Churn nearly doubles (4.1% → 7.3%) while NPS falls 47 → 38
- LTV/CAC deteriorates from 11.6x to 6.6x
- Headcount grows 28.9% against revenue growth of 21.3%, so revenue per
  employee falls
- West region is the weak spot: lowest margin, only region losing customers,
  plant at 61% utilisation
- Logistics cost grows 62.2% — expedited freight covering an OTIF gap of 87.2%
  against a 95% commitment

That last chain is the interesting one: the narrative explains *why* churn is
rising (delivery reliability, not price), and the cost table shows what is being
spent to paper over it. An agent that connects those two has found something a
reader would not get from either file alone.

## Questions worth asking

Ordered from simple to demanding.

1. "What happened to gross margin in FY2025 and why?"
2. "Is the revenue growth sustainable? Support your answer with the unit economics."
3. "Which region should we invest in and which should we fix or exit?"
4. "What is driving customer churn, and what would it cost to fix versus the
   revenue it protects?"
5. "Give me a full FY2026 plan: the three biggest risks, what to do about each,
   and what to track to know it is working."

Question 5 is the one that exercises all five agents and produces the longest
report.

## Uploading

Upload all five files, then ask your question. Order does not matter. The
narrative `.md` matters most for insight and recommendation quality — the CSVs
alone give the agents numbers with no context about *why*.
