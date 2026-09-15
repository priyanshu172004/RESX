# RESX - Teamwork

**An autonomous business research and data analytics agent.** Upload a corpus of
business documents, ask a question, and get an executive report where every number
traces to a citation *and* to a logged computation.

> An analyst who is confidently wrong is worse than no analyst at all. A number without
> a citation is a rumour — and RESX must never emit a rumour.

That constraint is why this system has a Critic agent, a deterministic math engine, and
mandatory source attribution rather than a single clever prompt.

---

## What it does

| Stage | RESX |
|---|---|
| **Acquisition & cleaning** | PDF/XLSX/CSV/DOCX/SQL extraction with citation anchors, dual table extraction with agreement checks, profiled and replayable cleaning transforms |
| **Exploratory analysis** | Descriptives, OLS + Mann-Kendall trend, CAGR, STL, Pearson/Spearman with `p` and `n`, three outlier methods — all computed in Python, never by the model |
| **Visualization** | KPI tiles, five chart archetypes on a CVD-validated palette, a table view for every chart |
| **Business intelligence** | Gap analysis, evidence-cited SWOT, P&L reconstruction with identity assertions, burn and runway, three-scenario forecasts, five ranked executive insights |

Five specialist agents (Finance, Risk, News, Workflow, Market) work under a Manager, and
an adversarial **Critic** tries to falsify every claim before a Synthesizer writes the
report. Disagreement that survives three rounds of debate ships as `CONTESTED` rather
than being averaged away.

## Documentation

Start with **[`RESX.md`](./RESX.md)** — the design of record. Detail lives in [`docs/`](./docs/):

| Document | Covers |
|---|---|
| [01 — Analyst workflow](./docs/01-ANALYST-WORKFLOW.md) | The manual job, modelled component by component |
| [02 — Architecture](./docs/02-ARCHITECTURE.md) | The layer cake, LangGraph state and routing, MCP servers, the RAG pipeline |
| [03 — Agents](./docs/03-AGENTS.md) | Every system prompt, output schema, and tool grant |
| [04 — Accuracy & validation](./docs/04-ACCURACY-VALIDATION.md) | Grounding, debate, the math check, gold-dataset gates |
| [05 — Security](./docs/05-SECURITY.md) | Threat model and the full control list |
| [06 — Design system](./docs/06-DESIGN-SYSTEM.md) | Tokens, the validated palette, components, motion, chart rules |
| [07 — API contract](./docs/07-API-CONTRACT.md) | REST and SSE surface |
| [08 — Data model](./docs/08-DATA-MODEL.md) | Collections, validators, vector search, tenancy |
| [09 — Roadmap](./docs/09-ROADMAP.md) | Phased delivery plan and sequencing rationale |

---

## Stack

| Layer | Technology |
|---|---|
| Presentation | Next.js 16 · React 19 · TypeScript (strict) · Tailwind CSS v4 · shadcn/ui · Recharts · Framer Motion |
| Orchestration | LangGraph — `StateGraph`, checkpointing, the Critic debate loop, backward routing on failure |
| Intelligence | Groq (`llama-3.3-70b-versatile` / `llama-3.1-8b-instant`) or Claude (`claude-opus-5` / `claude-sonnet-5`), behind one `LLM` protocol |
| Analytical engine | Python in a locked sandbox — pandas, numpy, scipy, statsmodels, matplotlib |
| Connectivity | MCP — filesystem, database, search, sandbox |
| Data | MongoDB 7 (replica set) · Redis · object storage |

---

## Getting started

**Prerequisites** — Node 20+, pnpm 11+, Python 3.10+, Docker.

```bash
# 1. Configuration
cp .env.example .env          # then fill in the placeholders

# 2. Signing keypair (RS256 access tokens)
#    Generated automatically in development if absent; required in production,
#    where minting one on boot would invalidate every session on every restart.
mkdir -p secrets
openssl genrsa -out secrets/jwt_private.pem 2048
openssl rsa -in secrets/jwt_private.pem -pubout -out secrets/jwt_public.pem

# 3. Dependencies (optional — SQLite is the default fallback)
docker compose up -d          # MongoDB replica set, Redis
pnpm install
pnpm api:install

# 4. Run
pnpm dev                      # web  → http://localhost:3000
pnpm api:dev                  # api  → http://localhost:8000
```

### Useful commands

| Command | Does |
|---|---|
| `pnpm dev` | Next.js dev server |
| `pnpm build` | Production build |
| `pnpm typecheck` | `tsc --noEmit` |
| `pnpm api:dev` | FastAPI with reload |
| `pnpm api:test` | Backend test suite |
| `pnpm check` | Typecheck + lint, both apps |
| `pnpm infra:up` / `infra:down` | Local MongoDB and Redis |

### The pipeline from a terminal

No frontend and no API keys needed. Full walkthrough in [`RUNBOOK.md`](./RUNBOOK.md).

```bash
python benchmarks/gold/generate.py                  # build a corpus with known answers
python benchmarks/score.py --k 10                   # score the pipeline against it
python scripts/resx.py ingest benchmarks/gold/synthetic-pnl
python scripts/resx.py status
python scripts/resx.py search "total revenue FY2025"
python scripts/resx.py compute --dataset <id> --sum revenue
python scripts/resx.py analyse "Is this business profitable and risky?"   # needs a key
```

---

## Project status

Built in phases so nothing rests on an unproven foundation — full plan and sequencing
rationale in [`docs/09-ROADMAP.md`](./docs/09-ROADMAP.md).

| Phase | Status |
|---|---|
| **0 — Specification & scaffold** | Complete |
| **1 — Design system & dashboard shell** | Complete — 17 routes, all live-data backed |
| **2 — Auth & security baseline** | Complete — Argon2id, RS256 + rotating refresh with reuse detection, RBAC, route guards, progressive lockout, append-only audit log |
| **3 — Ingestion & RAG** | Complete — extraction with citation anchors, layout-aware chunking, hybrid BM25 ∪ vector retrieval with RRF, the citation resolver |
| **4 — Sandbox & analytical engine** | Complete — isolated runner with computation records, descriptives/trend/correlation/outliers, `Decimal` money, accounting identities |
| **5 — Orchestrator & the five agents** | Complete — LangGraph graph, per-agent tool allowlists, two-phase compute cycle, SSE event stream |
| **6 — Critic, debate, synthesis** | Complete — independent re-derivation, bounded 3-round debate, `CONTESTED` outcomes, ranked report |
| **7 — Benchmarks & hardening** | Partial — gold corpus, adversarial traps, and 9 scored gates are in; the 4 model-dependent metrics are reported as NOT MEASURED |

**196 backend tests pass and all nine measured gates pass.** Run it yourself with
[`RUNBOOK.md`](./RUNBOOK.md) — everything except the agent run works with **no API keys**,
and the agent run itself needs only a free Groq key.

Three things to be clear about:

- **The dashboard reads the live API.** Every figure comes from
  `/api/v1/dashboard`, `/insights`, `/agents` and the SSE run stream. Where a
  workspace has no data the page says why rather than rendering a chart of zeroes —
  a chart with no data looks broken, and a chart of invented data is worse than
  either.
- **MongoDB is the store; SQLite is the offline fallback.** `STORE_BACKEND=auto`
  uses MongoDB when `MONGODB_URL` is set and reachable, and otherwise falls back to
  SQLite so a clean checkout still runs its tests and benchmarks with no server.
  MongoDB has no row-level security, so the application-layer tenant filter is the
  only boundary rather than one of two — which is why it is enforced by every store
  method's signature and asserted directly in
  [`tests/test_auth.py`](./services/api/tests/test_auth.py).
- **Two model providers, and they are not equivalent.** Groq is the default because it
  is free, but it has no prompt cache, no thinking-effort control, and no native schema
  parsing — so structured output goes through JSON mode plus Pydantic plus one bounded
  repair round. What does *not* change is the accuracy contract: the model never does
  arithmetic on either provider, and an ungroundable claim is dropped by the same gate.
  A weaker model yields fewer insights, not softer ones.

The full list of gaps is in [`RUNBOOK.md`](./RUNBOOK.md#known-gaps).

### On the chart palette

The eight series colours are **validated**, not chosen by eye: lightness band, chroma
floor, colour-vision-deficiency separation, normal-vision separation, and contrast against
each theme's surface. Results are recorded in
[`docs/06-DESIGN-SYSTEM.md`](./docs/06-DESIGN-SYSTEM.md) §2. Two rules follow from those
measurements and are enforced in `apps/web/src/lib/series.ts`:

- Scatter and small-multiple forms cap at **three** series, because those forms put every
  pair on screen at once and only the first three slots clear the all-pairs floors.
- Series colour is looked up **by entity key, never by array index**, so a filter that
  removes one series cannot repaint the survivors.

Re-run the validator after any palette change; do not substitute a hex by eye.

---

## Security

RESX accepts untrusted files, executes model-generated code, and stores a company's most
sensitive financials. The threat model and the complete control list are in
[`docs/05-SECURITY.md`](./docs/05-SECURITY.md). Two notes worth surfacing here:

- **Prompt injection is treated as a first-class injection class**, equal to SQLi. A
  document is an untrusted input channel that reaches the *reasoning* layer. The primary
  control is capability isolation — the Finance agent has no web tool *in its schema* — not
  an instruction telling the model to resist being fooled.
- **The LLM never does arithmetic.** This is enforced by a Pydantic validator *and* a
  MongoDB collection validator: a claim carrying a value with no `computation_id` cannot be
  persisted.

Never commit `.env` or anything under `secrets/`.
