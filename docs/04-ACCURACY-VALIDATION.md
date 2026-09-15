# 04 — Accuracy & Validation

The premise of this document: **an AI analyst is only worth deploying if it is verifiably
correct.** Plausibility is not a product. Four mechanisms, layered, each catching what the
previous one misses.

---

## Layer 1 — Grounding (source attribution)

### The contract

No claim ships without a resolvable citation. This is enforced by a resolver that runs after
every agent node, before the claim enters state.

```python
def resolve(citation: Citation, corpus: Corpus) -> Resolution:
    chunk = corpus.chunk_at(citation.doc_id, citation.page, citation.char_span)
    if chunk is None:
        return Resolution.ANCHOR_NOT_FOUND          # fabricated location
    score = fuzzy_ratio(normalize(citation.quote), normalize(chunk.text))
    if score < 0.92:
        return Resolution.QUOTE_MISMATCH             # fabricated or drifted quote
    return Resolution.OK
```

For a **numeric** claim there is one additional gate: the figure must either appear verbatim
in the cited span, or be the output of a `ComputationRecord` whose inputs are themselves
cited. There is no third option.

### Consequences of failure

| Resolution | Action |
|---|---|
| `OK` | Claim enters state |
| `QUOTE_MISMATCH` | Claim dropped, node retried once with the mismatch shown to the agent |
| `ANCHOR_NOT_FOUND` | Claim dropped, logged as a hallucination event, retried once |
| Retry also fails | Claim discarded; run records a `grounding_failure` for that sub-question |

We drop rather than downgrade. A claim that survives as "low confidence" still ends up in a
report, and a fabricated citation is not a confidence problem — it is a correctness problem.

### What the user sees

Every figure in the UI is hover-inspectable: the quoted span, the document, the page, and
the computation that produced it. This is the product promise made visible, and it is also
the fastest way for a human to catch the one thing the pipeline missed.

---

## Layer 2 — Cross-verification (multi-agent debate)

Grounding proves a claim is *attributable*. It does not prove it is *right* — a correctly
cited figure can still be the wrong figure for the question, or contradicted three pages
later. That is the Critic's job.

### Why a Critic works, and when it does not

It works because falsification is an easier task than derivation, and because a second pass
with a different prompt and independent computation is genuinely decorrelated from the first.

It fails when the reviewer is prompted to "review" — models default to agreement. Three
design choices prevent that:

1. The Critic **re-derives independently** in the sandbox and never sees the original
   computation. A numeric disagreement is then mechanical, not rhetorical.
2. The prompt sets the incentive explicitly: measured on errors caught, not claims approved.
3. The Critic runs on the strongest model available. A weaker reviewer produces theatre.

### The debate protocol

```
verdict REFUTED / CONTESTED
   └─▶ round 1  originating agent: concede+revise | defend with NEW evidence | narrow the claim
       round 2  Critic responds to the defence
       round 3  final positions
       exit     consensus  → CONFIRMED at the agreed value
                no consensus → CONTESTED, both positions shipped with both citations
```

Bounded at three rounds. Disagreements that survive three evidence-backed exchanges are
almost always genuine ambiguity in the source, and further rounds produce rhetoric rather
than evidence. **Shipping that ambiguity visibly is the honest outcome** — a `CONTESTED`
badge in the report is more useful to a decision-maker than a confident average of two
incompatible readings.

---

## Layer 3 — Deterministic validation (the math check)

> LLMs are probabilistic. Python is not. Therefore the LLM writes Python.

### Mechanical enforcement

The rule is not a prompt instruction that a model can drift from. It is a schema validator:
a `Claim` with a non-null `value` and a null `computation_id` fails Pydantic validation and
never enters state. See [`03-AGENTS.md`](./03-AGENTS.md) §0.

A `ComputationRecord` is the audit trail:

```python
class ComputationRecord(BaseModel):
    computation_id: str
    code: str                 # exact source executed
    inputs: list[str]         # dataset ids / chunk ids consumed
    stdout: str
    result: JSONValue
    duration_ms: int
    sandbox_image_digest: str  # reproducibility
```

Given the record and the raw upload, any figure in the report is reproducible byte-for-byte.

### Numeric hygiene

- `Decimal` for money, everywhere. `float` cannot represent `0.10`, and a report where
  gross profit is off by `0.0000001` fails its own identity assertion for no good reason.
- Explicit rounding at presentation only, never mid-calculation.
- Scale factors (`in thousands`, `$M`) resolved at extraction and carried on the claim, since
  a 1000x error is the single most damaging failure mode in financial extraction.
- Unit registry (`pint`) so a ratio of mismatched units raises rather than returns nonsense.

### Identity assertions

Run as code after the Finance agent, before the Critic:

```python
assert_close(revenue - cogs, gross_profit, tol="0.01")
assert_close(gross_profit - opex, operating_income, tol="0.01")
assert_close(assets, liabilities + equity, tol="0.01")
assert_close(sum(segment_revenue), total_revenue, rel_tol=0.005)
assert_close(opening_cash + net_cash_flow, closing_cash, tol="0.01")
```

A failure is a **hard stop that routes backwards to ingestion.** The interpretation is
specific and important: a broken identity means we misread the document — wrong scale, wrong
column, a subtotal read as a total. The correct response is to re-extract, never to nudge a
number until the identity closes. Silently reconciling is how a plausible, wrong report gets
written.

---

## Layer 4 — Benchmarking against gold datasets

### Composition of `benchmarks/gold/`

| Set | Content | What it tests |
|---|---|---|
| `synthetic-pnl/` | Generated statements with exact known ground truth | Numeric exactness, identity handling |
| `filings/` | Public annual reports, figures hand-labelled | Real-world extraction, table parsing, scale detection |
| `adversarial/` | Restated periods, non-summing segments, footnote qualifications, mixed currencies, scanned pages, prompt-injection payloads embedded in documents | The failure modes we know about |
| `insight-rated/` | Reports with expert-rated top-5 insights | Insight precision, prioritization |

The `adversarial/` set is where the real value is. It is populated from every production
failure: any error found in the field becomes a permanent regression case.

### Scoring

```python
metrics = {
  "numeric_exactness":  hits(|pred - truth| / |truth| <= 0.005) / n_numeric,
  "citation_validity":  resolved_citations / total_citations,          # must be 1.00
  "hallucination_rate": unsupported_claims / total_claims,
  "retrieval_recall_10": gold_span_in_top10 / n_questions,
  "identity_pass_rate": identities_passed / identities_checked,
  "insight_precision":  expert_agreement_on_top5,
  "critic_catch_rate":  seeded_errors_caught / seeded_errors_injected,
}
```

`critic_catch_rate` deserves emphasis: we deliberately inject known errors into the
`adversarial/` corpus and measure whether the Critic finds them. It is the only direct
measurement of whether the debate loop earns its cost, as opposed to being an expensive
formality.

### Gates

| Metric | Gate | On breach |
|---|---|---|
| `citation_validity` | `== 1.00` | Block merge, no exceptions |
| `numeric_exactness` | `>= 0.98` | Block merge |
| `hallucination_rate` | `<= 0.01` | Block merge |
| `retrieval_recall_10` | `>= 0.95` | Block merge (tune chunking or fusion) |
| `critic_catch_rate` | `>= 0.90` | Block merge |
| `insight_precision` | `>= 0.80` | Warn, review required |

Every prompt change, chunking change, model change, and retrieval change runs the suite. A
prompt is a dependency with a version, and it regresses like any other.

---

## Calibration, not confidence

Agents emit a confidence in `[0,1]`. Confidence is only meaningful if it is calibrated, so we
measure it: bucket claims by stated confidence and compare against gold-set correctness.

```
stated 0.9-1.0 → observed accuracy should be 0.9-1.0
stated 0.7-0.9 → observed accuracy should be 0.7-0.9
```

A systematically overconfident agent gets a prompt correction, and the calibration curve is
tracked as a first-class metric. An uncalibrated confidence score is worse than none, because
users reasonably act on it.

---

## What we do not claim

Honest limits, stated in the product:

- RESX cannot verify facts absent from the corpus and the web. It reports the absence.
- OCR on poor scans introduces error; low-confidence OCR blocks are flagged, not silently used.
- Forecasts are scenario models with stated assumptions, not predictions.
- `CONTESTED` claims are genuinely unresolved and are labelled as such.
- Insight prioritization is a judgment. The ranking formula is shown so it can be argued with.

A system that admits its boundaries is trustworthy inside them. One that does not is
unusable anywhere.
