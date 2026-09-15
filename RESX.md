# RESX — Autonomous Business Research & Data Analytics Agent

> **Single source of truth.** This file is the design of record. Every architectural
> decision, agent contract, validation rule, and security control lives here or is
> linked from here. Detail documents live in [`docs/`](./docs/).

| | |
|---|---|
| **Codename** | RESX |
| **Category** | AI-native Business Intelligence / Autonomous Analyst |
| **Thesis** | Replace the 40-hour manual analyst pipeline (ETL → EDA → Viz → Insight) with a validated, citation-grounded multi-agent system. |
| **Non-goal** | A chatbot that "talks about" data. RESX *computes*, *cites*, and *defends* every number. |
| **Status** | Phase 1 complete — design system and dashboard shell on fixture data; Phase 2 (auth) in progress |

---

## Table of Contents

1. [The Problem We Are Actually Solving](#1-the-problem-we-are-actually-solving)
2. [What a Business/Data Analyst Actually Does](#2-what-a-businessdata-analyst-actually-does)
3. [System Architecture — The Layer Cake](#3-system-architecture--the-layer-cake)
4. [LangGraph is the Brain, MCP is the Hands](#4-langgraph-is-the-brain-mcp-is-the-hands)
5. [The Agent Roster](#5-the-agent-roster)
6. [The Accuracy Contract (The "Truth" Problem)](#6-the-accuracy-contract-the-truth-problem)
7. [Security Architecture](#7-security-architecture)
8. [Design System & UI Language](#8-design-system--ui-language)
9. [Repository Layout](#9-repository-layout)
10. [Build Phases](#10-build-phases)
11. [Definition of Done](#11-definition-of-done)

---

## 1. The Problem We Are Actually Solving

A mid-market business sits on thousands of pages of unstructured truth: annual reports,
board decks, bank statements, CRM exports, vendor contracts, support tickets. The value is
real and the extraction cost is brutal. A human analyst spends **~70% of their time on
grunt work** (acquisition, cleaning, normalization) and only ~30% on the thing that
actually creates profit: **judgment**.

RESX inverts that ratio. It automates the 70% deterministically (Python, not vibes) and
augments the 30% with five specialist agents that argue with each other until the numbers
survive scrutiny.

**The hard constraint that shapes every design decision:**

> An analyst who is confidently wrong is worse than no analyst at all.
> A number without a citation is a rumour. RESX must never emit a rumour.

That single sentence is why the architecture has a Critic agent, a deterministic math
engine, and mandatory source attribution. See [§6](#6-the-accuracy-contract-the-truth-problem).

---

## 2. What a Business/Data Analyst Actually Does

You cannot automate a workflow you have not modelled. RESX mirrors the real four-stage
analyst pipeline, and each stage maps to concrete system components.

### Stage 1 — Data Acquisition & Cleaning (the "grunt" work)

| Manual activity | RESX component | Determinism |
|---|---|---|
| **ETL** — pull from SQL, CSV, PDF, XLSX, APIs | `services/api/ingest/` + MCP filesystem/database servers | 100% code |
| **Data scrubbing** — nulls, duplicates, malformed rows | pandas profile → repair plan → applied transform log | 100% code |
| **Normalization** — dates, currencies, units, entity names | `normalizers/` (ISO-8601, FX table, unit registry, fuzzy entity resolution) | 100% code |
| **Schema inference** — what *is* this file? | LLM proposes schema → Pydantic validates → code executes | LLM proposes, code decides |

**Rule:** the LLM may *propose* a cleaning plan. It may never *perform* the cleaning. Every
mutation is a logged, replayable Python transform. Detail:
[`docs/01-ANALYST-WORKFLOW.md`](./docs/01-ANALYST-WORKFLOW.md).

### Stage 2 — Exploratory Data Analysis (EDA)

- **Descriptive statistics** — mean, median, mode, variance, std-dev, skew, kurtosis, IQR.
  Computed by numpy/pandas in the sandbox. Never by the model.
- **Trend analysis** — OLS slope plus a Mann-Kendall test for monotonic trend, STL seasonal
  decomposition for periodic series, CAGR for financial series.
- **Correlation** — Pearson (linear) and Spearman (rank), each reported with `p_value` and
  `n`. The Critic rejects causal language resting on correlational evidence.
- **Outlier detection** — IQR fence, z-score, and isolation forest. Where the three methods
  disagree, the disagreement is surfaced rather than hidden.

### Stage 3 — Visualization

- **Dashboarding** — live KPI tiles with delta-vs-prior-period and a trend footnote.
- **Comparative charting** — form follows the question (see the heuristic in
  [§8](#8-design-system--ui-language)): bar for categories, line/area for time, scatter for
  relationships and outliers, donut for composition, radial for bounded progress.
- **Slice and dice** — every chart binds to one shared filter context (date range, region,
  product, segment), URL-serialized so any view is a shareable link.

### Stage 4 — Business Intelligence & Strategic Insight

| Framework | Implementation |
|---|---|
| **Gap analysis** | Target vs. actual per KPI, variance decomposition, time-to-close projection |
| **SWOT** | Internal evidence (documents) → S/W; external evidence (web) → O/T. Every quadrant item cited. |
| **Financial modelling** | P&L reconstruction, margin waterfall, burn rate, runway months, three-scenario forecast (bear/base/bull) |
| **Executive reporting** | 100 pages → 5 actionable insights, each with owner, effort, expected impact, and confidence |

---

## 3. System Architecture — The Layer Cake

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  PRESENTATION            Next.js 16 · React 19 · Tailwind v4 · shadcn/ui     │
│                          Recharts · Framer Motion · streaming run console    │
├──────────────────────────────────────────────────────────────────────────────┤
│  EDGE / BFF              Next route handlers · session cookies · CSP         │
│                          rate limiting · input schema validation (Zod)       │
├──────────────────────────────────────────────────────────────────────────────┤
│  ORCHESTRATION           LangGraph — StateGraph, conditional edges,          │
│                          checkpointing, human-in-the-loop interrupts,        │
│                          the Critic debate loop, retry/repair routing        │
├──────────────────────────────────────────────────────────────────────────────┤
│  INTELLIGENCE            Claude (Opus 5 / Sonnet 5) — reasoning, tool        │
│                          selection, synthesis. Structured output enforced.   │
├──────────────────────────────────────────────────────────────────────────────┤
│  ANALYTICAL ENGINE       Code interpreter — Python in a locked sandbox.      │
│                          pandas / numpy / scipy / statsmodels / matplotlib.  │
│                          ALL arithmetic happens here. No exceptions.         │
├──────────────────────────────────────────────────────────────────────────────┤
│  CONNECTIVITY (MCP)      filesystem · mongodb · tavily-search · sandbox      │
│                          Standardized tool surface. Swap a backend without   │
│                          rewriting a single agent.                           │
├──────────────────────────────────────────────────────────────────────────────┤
│  DATA                    MongoDB (documents + vectors) · object store       │
│                          (raw uploads) · Redis (queue, rate limits, cache)   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Why this split

**LangGraph alone** → you hand-write a bespoke Python tool for every data format, and
swapping the database means rewriting every tool.

**MCP alone** → excellent hands, no brain. Five disconnected chatbots with no shared state,
no synthesis step, and no validation loop.

**Together** → LangGraph is the *manager* (who works, when, and what happens on failure);
MCP is the *infrastructure* (how they touch data). That is the decision.

Full detail: [`docs/02-ARCHITECTURE.md`](./docs/02-ARCHITECTURE.md).

### The RAG pipeline — why we cannot just paste 1,000 pages

A 1,000-page corpus is roughly 500k–800k tokens of raw text before you add five agents
worth of reasoning. Even at a 1M-token context window, stuffing it is wasteful, slow,
expensive, and — critically — it *destroys citation precision*. Retrieval is not a
context-window workaround; it is the mechanism that makes "Source: page 12, ¶3" possible
at all.

```
Upload → virus/type check → text + table extraction (pdfplumber / camelot / openpyxl)
       → layout-aware chunking (respects tables, headers, page bounds)
       → chunk carries {doc_id, page, bbox, section, para_idx}  ← the citation anchor
       → embed (voyage-3 / text-embedding-3-large)
       → Atlas Vector Search (HNSW), with a numpy fallback index
       → hybrid retrieval: BM25 (lexical) ∪ vector (semantic) → reciprocal-rank fusion
       → cross-encoder rerank → top-k to agent
```

Numeric tables take a **second path**: extracted into typed DataFrames, persisted as
Parquet, and registered in the sandbox. Agents query them with real SQL or pandas instead
of reading numbers out of prose. This is the difference between a demo and a product.

---

## 4. LangGraph is the Brain, MCP is the Hands

### The graph

```
                          ┌──────────────┐
        user request  ──▶ │   MANAGER    │  plans, decomposes, delegates
                          └──────┬───────┘
                                 │
              ┌──────────┬───────┼───────┬──────────────┐
              ▼          ▼       ▼       ▼              ▼
         ┌────────┐ ┌──────┐ ┌──────┐ ┌────────┐ ┌────────────┐
         │FINANCE │ │ RISK │ │ NEWS │ │WORKFLOW│ │   MARKET   │
         └───┬────┘ └──┬───┘ └──┬───┘ └───┬────┘ └─────┬──────┘
             └─────────┴────────┴─────────┴────────────┘
                                 │  (claims + citations → state)
                                 ▼
                          ┌──────────────┐
                          │    CRITIC    │  hunts contradictions
                          └──────┬───────┘
                        disagree │ agree
                    ┌────────────┘  └──────────┐
                    ▼                          ▼
             ┌─────────────┐            ┌──────────────┐
             │ DEBATE LOOP │            │  SYNTHESIZER │
             │ (max 3 rds) │            └──────┬───────┘
             └──────┬──────┘                   ▼
                    │                   ┌──────────────┐
                    └──────────────────▶│    REPORT    │
                                        └──────────────┘
```

A claim still unresolved after three rounds ships flagged `CONTESTED`, carrying both
positions and both citations. RESX surfaces disagreement; it does not paper over it.

### The trace, end to end

1. **Request (LangGraph).** The user uploads 1,000 pages and asks *"is this business
   profitable and risky?"* LangGraph builds `RunState` and routes to the Manager, which
   emits a plan: Finance and Risk in parallel, then a News cross-check, then the Critic.
2. **Action (MCP).** Finance needs the actual numbers. It does not get a bespoke
   `read_pdf()`; it calls `mcp.filesystem.read_chunk(doc_id, page=12)` and
   `mcp.sandbox.run_python(...)`. The MCP server returns text plus table handles.
3. **Analysis (LangGraph).** Finance writes `Claim{value, unit, period, citations[],
   method, confidence}` into state. LangGraph marks the node complete and fans out to Risk.
4. **Cross-check (MCP + LangGraph).** Risk queries `mcp.tavily.search` for live market data,
   then LangGraph routes both agents' claims into the Critic node, which attempts
   falsification before anything reaches the Synthesizer.

**State is append-only and checkpointed.** A run can be paused, inspected, resumed, or
replayed, and every claim is traceable to the exact tool call that produced it.

---

## 5. The Agent Roster

Each agent is a `(system_prompt, tool_allowlist, output_schema)` triple. Tools are scoped
per agent — the Finance agent cannot reach the web, and the News agent cannot touch the
filesystem. Least privilege applies to agents, not just to users.

| Agent | Owns | MCP tools | Emits |
|---|---|---|---|
| **Manager** | Decomposition, routing, budget enforcement | — | `Plan{tasks[], deps[]}` |
| **Finance** | Revenue, COGS, margins, P&L, burn, runway, FX | filesystem, sandbox, database | `FinancialClaim[]` |
| **Risk** | Liabilities, covenants, litigation, concentration, volatility | filesystem, sandbox, search | `RiskFinding[]` (severity × likelihood) |
| **News** | Real-time external corroboration of internal claims | search, fetch | `Corroboration[]` (supports / contradicts / silent) |
| **Workflow** | Process diagnosis via Lean / Six Sigma / Theory of Constraints | filesystem, sandbox | `ProcessGap[]` (waste, bottleneck, DPMO) |
| **Market** | TAM/SAM/SOM, competitor set, share, positioning | search, filesystem | `MarketInsight[]` |
| **Critic** | Adversarial falsification of every claim | sandbox, filesystem | `Verdict{CONFIRMED / REFUTED / CONTESTED}` |
| **Synthesizer** | Merge → rank → five executive insights | — | `ExecutiveReport` |

Full prompts, schemas, and tool grants: [`docs/03-AGENTS.md`](./docs/03-AGENTS.md).

---

## 6. The Accuracy Contract (The "Truth" Problem)

This is the hardest part of the system and the only thing separating RESX from a toy. Four
mechanisms, none of them optional.

### 6.1 Grounding — source attribution

Every claim carries at least one `Citation{doc_id, page, para_idx, quote, char_span}`. The
pipeline enforces this structurally rather than hoping the model complies:

```
claim emitted → citation resolver looks up the cited span in the corpus
              → the verbatim quote must exist at that anchor (exact or fuzzy ≥ 0.92)
              → numeric claims: the number must appear in the cited span, OR be the
                output of a logged sandbox computation over cited inputs
              → any check fails → claim is DROPPED, node retried once, then flagged
```

An uncitable claim is a hallucination by definition, and it does not reach the user.

### 6.2 Cross-verification — multi-agent debate

The Critic is not a rubber stamp. Its prompt is adversarial — *find the contradiction* — and
it holds the sandbox and the corpus so it can re-derive the number independently.
Disagreement opens a bounded debate of at most three rounds; consensus or `CONTESTED` is
the only exit.

### 6.3 Deterministic validation — the math check

> **The LLM never does arithmetic.** It writes Python; Python computes.

This is enforced mechanically rather than by convention: a numeric field in any agent output
must be accompanied by a `computation_id` referencing a sandbox execution record
(`{code, stdout, result, duration}`). Schema validation rejects the claim otherwise. On top
of that, accounting identities are asserted in code:

```
revenue - cogs           == gross_profit       (±0.01)
gross_profit - opex      == operating_income   (±0.01)
assets                   == liabilities + equity (±0.01)
Σ(segment_revenue)       == total_revenue       (±0.5%)
```

A failed identity is a **hard stop** that routes back to the ingestion node. It means we
misread the document, and the correct response is to re-read it — not to report a plausible
number.

### 6.4 Benchmarking — gold datasets

`benchmarks/gold/` holds reports whose answers are already known: public filings with
hand-labelled figures, and synthetic P&Ls with exact ground truth. CI runs the full pipeline
and scores it.

| Metric | Target | Meaning |
|---|---|---|
| **Numeric exactness** | ≥ 0.98 | extracted figure equals ground truth (±0.5%) |
| **Citation validity** | 1.00 | every citation resolves to a real, matching span |
| **Hallucination rate** | ≤ 0.01 | claims with no supporting evidence |
| **Retrieval recall@10** | ≥ 0.95 | the gold span is present in the retrieved set |
| **Insight precision** | ≥ 0.80 | human-rated actionability of the top five |

Regressions block merge. If RESX says profit is $1.0M and the truth is $1.2M, we tune
retrieval or prompts — we do not ship. Detail:
[`docs/04-ACCURACY-VALIDATION.md`](./docs/04-ACCURACY-VALIDATION.md).

---

## 7. Security Architecture

RESX ingests untrusted files, executes generated code, and holds a company's most sensitive
financials. It is a high-value target and is built accordingly: defence in depth across
seven layers. Full control list and threat model in
[`docs/05-SECURITY.md`](./docs/05-SECURITY.md).

### 7.1 Authentication and session

- Argon2id password hashing (`m=64MiB, t=3, p=4`) — memory-hard and GPU-resistant. Never
  bcrypt at a low cost factor, and never an SHA-family digest for passwords.
- Short-lived **access JWT** (15 minutes, RS256) plus a rotating **refresh token** (7 days,
  opaque, hashed at rest, single-use with reuse detection that revokes the whole family).
- Tokens delivered as `HttpOnly; Secure; SameSite=Strict` cookies — not `localStorage`,
  which is XSS-readable by definition.
- TOTP two-factor, hashed recovery codes, generic auth errors (no user enumeration), and
  constant-time comparison on every secret check.

### 7.2 Authorization

- RBAC (`owner` / `admin` / `analyst` / `viewer`) **plus** per-workspace row scoping.
- Every query is tenant-filtered at the repository layer, enforced by each store
  method's signature
  as the backstop so an application bug cannot leak across tenants.
- Route protection is server-side. Client-side guards are UX, never security.
- Signed, expiring, single-use URLs for artifact and report downloads.

### 7.3 Injection defence

| Vector | Control |
|---|---|
| **SQL injection** | Parameterized queries only; ORM-generated SQL; zero string interpolation into SQL, enforced by a CI grep and a lint rule; a read-only DB role for analytics |
| **XSS** | No `dangerouslySetInnerHTML` on user data; DOMPurify for rendered markdown; strict CSP with nonces and no `unsafe-inline` / `unsafe-eval`; output encoding by default |
| **CSRF** | `SameSite=Strict` plus a double-submit token and an `Origin` check on every state-changing verb |
| **Prompt injection** | Retrieved document text is wrapped in untrusted-content delimiters and never treated as instructions; per-agent tool allowlists mean a document cannot escalate privileges. Treated as a *first-class* injection class, equal to SQLi. |
| **Path traversal** | The filesystem MCP server is chrooted per workspace; paths are canonicalized and prefix-asserted |
| **SSRF** | Egress allowlist; RFC-1918, link-local, and cloud-metadata addresses blocked; redirects re-validated |
| **Malicious upload** | Magic-byte type check (not the extension), size cap, ClamAV scan, quarantine bucket, no execute permission, served from a separate origin |

### 7.4 DoS and DDoS resilience

- Layered rate limits — per IP, per user, per workspace, per endpoint — as Redis token buckets.
- Far stricter budgets on expensive routes (upload, run, embed) than on reads.
- Request body size caps, and upload streaming with hard byte ceilings.
- Query cost limits: statement timeout, row caps, mandatory pagination.
- Sandbox limits: CPU seconds, memory ceiling, wall-clock timeout, no network, no filesystem
  write outside `/tmp`, dropped capabilities, non-root user, seccomp profile, one-shot
  container per execution.
- LLM spend caps per run and per workspace — an unbounded agent loop is a financial DoS.
- An async job queue, so a burst of heavy runs degrades latency rather than availability.
- A CDN/WAF absorbs L3/L4 volume, while the application-layer controls above assume the WAF
  may fail.

### 7.5 Transport, headers, cryptography

- A Helmet-equivalent header set: HSTS with preload, strict CSP, `X-Content-Type-Options`,
  `Referrer-Policy: strict-origin-when-cross-origin`, `X-Frame-Options: DENY`,
  `Permissions-Policy`, and COOP/CORP.
- TLS 1.3 only. AES-256-GCM at rest, envelope encryption via KMS with key rotation, and
  per-field encryption on the most sensitive financial columns.
- HMAC-SHA256 for webhook and download signatures; SHA-256 for content addressing;
  **Argon2id for passwords**. The three use cases are never conflated.

### 7.6 Validation

Every boundary validates: Zod at the edge, Pydantic at the service, database constraints
last. Allowlist over denylist, reject by default, and a structured output schema on every
LLM call.

### 7.7 Observability and audit

An append-only audit log (who, what, when, which workspace, which IP) covering auth events,
data access, exports, and role changes. Structured logs with PII redaction. OpenTelemetry
traces spanning UI → API → graph → agent → tool. Secrets are never logged.

---

## 8. Design System & UI Language

The reference dashboards set the bar: **near-black canvas, ultra-restrained palette, dense
information, zero decoration.** The interface should feel like a precision instrument.

### Principles

1. **Monochrome first.** Greyscale carries structure; colour carries *meaning only* —
   positive or negative delta, severity, confidence. A chart with five arbitrary colours is
   a design failure.
2. **Typography is the design.** One family (Geist / Inter), tabular numerals on every
   figure, a tight scale, no decorative type.
3. **Density with air.** Analysts want data on screen. Get density from disciplined spacing
   and hairline separators, not from shrinking type below 12px.
4. **Motion clarifies causality.** Framer Motion for entry stagger, layout transitions, and
   the agent run console: 150–250ms, ease-out, `prefers-reduced-motion` respected. Nothing
   bounces, and nothing spins for decoration.
5. **Every number is inspectable.** Hover any figure to see its citation and its
   computation. This is a *design* requirement because it is the product's core promise.

### Stack

`Next.js 16 (App Router)` · `React 19` · `TypeScript strict` · `Tailwind CSS v4` ·
`shadcn/ui` · `Recharts` (through the shadcn `ChartContainer`) · `Framer Motion` ·
`lucide-react` · `next-themes`

### Chart form heuristic

| Question shape | Form |
|---|---|
| How did X move over time? | Line (multi-series) or area (cumulative, stacked) |
| How do categories compare? | Horizontal bar, sorted by value |
| What composes the whole? | Donut with a centre total — **five slices maximum**, otherwise a bar |
| Progress toward a bound? | Radial |
| Does A relate to B? | Scatter with a fitted line and an r / p annotation |
| Where are the anomalies? | Scatter or box plot with flagged points |

The dashboard vocabulary — a KPI tile with a delta chip and trend footnote, a sectioned data
table with status and reviewer columns, tabbed sub-views, column customization — is taken
directly from the reference screenshots. Full tokens, palette, and component inventory:
[`docs/06-DESIGN-SYSTEM.md`](./docs/06-DESIGN-SYSTEM.md).

---

## 9. Repository Layout

```
ResX/
├─ RESX.md                      ← this file, the design of record
├─ docs/
│  ├─ 01-ANALYST-WORKFLOW.md    the manual pipeline, modelled
│  ├─ 02-ARCHITECTURE.md        layers, RAG, state, deployment
│  ├─ 03-AGENTS.md              prompts, schemas, tool grants
│  ├─ 04-ACCURACY-VALIDATION.md grounding, debate, math, benchmarks
│  ├─ 05-SECURITY.md            threat model and full control list
│  ├─ 06-DESIGN-SYSTEM.md       tokens, components, motion, charts
│  ├─ 07-API-CONTRACT.md        REST and SSE surface
│  ├─ 08-DATA-MODEL.md          collections, validators, vector search, tenancy
│  └─ 09-ROADMAP.md             phased delivery plan
├─ apps/
│  └─ web/                      Next.js frontend
├─ services/
│  └─ api/                      FastAPI + LangGraph + MCP clients
├─ benchmarks/gold/             gold datasets and scoring harness
└─ scripts/                     dev tooling
```

---

## 10. Build Phases

| Phase | Deliverable | Exit criterion |
|---|---|---|
| **0** | Specification suite and monorepo scaffold | This document; both apps boot |
| **1** | Design system and dashboard shell | Sidebar, KPI row, all five chart archetypes, theme toggle, motion |
| **2** | Auth and security baseline | Argon2id, JWT + refresh, RBAC, Helmet/CSP, rate limits, Zod/Pydantic |
| **3** | Ingestion and RAG | Upload → extract → chunk with anchors → embed → hybrid retrieve, cited |
| **4** | Sandbox and analytical engine | Deterministic statistics and chart generation, computation records |
| **5** | LangGraph orchestrator and MCP | Five agents plus Manager, streaming run console |
| **6** | Critic, debate, synthesis | Verdicts, `CONTESTED` handling, executive report |
| **7** | Benchmarks and hardening | Gold-dataset gates in CI, security review, load test |

Detail: [`docs/09-ROADMAP.md`](./docs/09-ROADMAP.md).

---

## 11. Definition of Done

RESX ships when a user can upload 1,000 pages and receive an executive report where:

- [ ] Every number traces to a citation **and** a logged computation.
- [ ] Every claim has survived an adversarial Critic pass.
- [ ] Accounting identities hold to the cent.
- [ ] Gold-dataset numeric exactness is ≥ 0.98 and citation validity is 1.00.
- [ ] The chatbot can answer *"why did the Risk agent flag page 42?"* with the retrieved
      span, the agent's reasoning, and the Critic's verdict.
- [ ] A security review finds no high-severity issue across the
      [§7](#7-security-architecture) control list.
- [ ] The interface is something an analyst would choose to look at for eight hours.

Anything less is a demo. RESX is not a demo.
