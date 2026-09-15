# 09 — Roadmap

Sequenced so that each phase produces something demonstrable and nothing is built on an
unproven foundation. The ordering is deliberate: **security before ingestion, and the
deterministic engine before the agents.**

---

## Phase 0 — Specification & scaffold ✅

**Deliverable:** this documentation suite plus a monorepo where both applications boot.

- [x] `RESX.md` — the design of record
- [x] `docs/01`–`docs/09` — workflow, architecture, agents, accuracy, security, design, API, data model, roadmap
- [x] Validated chart palette (six checks, both modes)
- [x] `apps/web` — Next.js 16, TS strict, Tailwind v4, shadcn/ui
- [x] `services/api` — FastAPI, Pydantic v2 settings, security primitives (SQLAlchemy/Alembic wiring lands with Phase 2)
- [x] Docker Compose: MongoDB, Redis, ClamAV profile
- [x] Root scripts, `.env.example`, `.gitignore`
- [ ] CI skeleton (lands with Phase 2)

**Exit:** `pnpm dev` and `uvicorn` both start; `/health` returns green; CI runs lint and types.

---

## Phase 1 — Design system & dashboard shell ✅

**Deliverable:** the interface, with realistic fixture data. This comes first because it is
what makes the rest of the work reviewable.

- Tokens from [`06-DESIGN-SYSTEM.md`](./06-DESIGN-SYSTEM.md), light and dark, both designed
- `AppSidebar`, `SiteHeader`, `PageShell` — the reference layout
- `KpiCard` row with delta chips and significance-derived direction
- All five chart archetypes on the validated palette: interactive area, multi-line, donut
  with centre total, radial, stacked bar
- `ChartCard` wrapper: shared padding, legend, tooltip, grid, range select, table-view toggle
- `DataTable`: sorting, column visibility, tabbed sub-views with counts, drag handles
- Framer Motion entry stagger, count-up KPIs, reduced-motion honoured
- Theme toggle wired through `next-themes`

**Exit:** the dashboard is indistinguishable from the reference in density and restraint;
every chart has a hover layer and a table view; no anti-pattern from §6 present.

---

## Phase 2 — Auth & security baseline ◑ partial

**Deliverable:** nobody can reach anyone else's data. Built before ingestion, because
retrofitting tenancy onto a working pipeline is how leaks happen.

- Argon2id registration and login; email verification
- Access JWT (RS256, 15 min) + rotating refresh with **reuse detection**
- TOTP 2FA, hashed recovery codes, progressive lockout
- RBAC (`owner`/`admin`/`analyst`/`viewer`), server-side route guards, deny by default
- Repository-layer tenant filter **and** the store's signature-enforced tenant filter
- Helmet-equivalent headers, strict CSP with nonces, CSRF double-submit
- Redis token-bucket rate limits, layered per the security doc
- Zod at the edge, Pydantic `extra="forbid"` at the service, DB `CHECK`s last
- Append-only `audit_log`
- CI: `pip-audit`, `npm audit`, `gitleaks`, the f-string-SQL check, the
  `dangerouslySetInnerHTML` lint rule

**Exit:** a deliberate cross-tenant attempt returns 404; refresh reuse revokes the
whole token family; every header from §6 present in a production response.

MongoDB has no row-level security, so the application filter is the only boundary
rather than one of two. It is therefore enforced by every store method's signature
and asserted directly in `tests/test_auth.py` — see `docs/08-DATA-MODEL.md` §6.

---

## Phase 3 — Ingestion & RAG ✅

**Deliverable:** upload 1,000 pages, ask a question, get a **cited** extractive answer. No
agents yet — this phase proves grounding works in isolation.

- Upload: magic-byte check, size caps, ClamAV, quarantine, content-addressed storage
- Extraction: pdfplumber with bboxes, OCR fallback with per-block confidence, dual table
  extraction with agreement flag, xlsx/csv/docx
- Layout-aware chunking with the full citation anchor
- Embeddings in batches; Atlas Vector Search (HNSW), with a numpy fallback
- Hybrid retrieval: BM25 ∪ vector → RRF → cross-encoder rerank
- Numeric path: tables → typed DataFrames → Parquet → `datasets` + profile
- **Citation resolver** with the fuzzy-match gate
- `CitationPopover` wired to the real resolver
- Chat endpoint with grounded, cited answers

**Exit:** `retrieval_recall@10 ≥ 0.95` on the gold set; `citation_validity = 1.00`; a
fabricated citation is caught and dropped by the resolver in a test.

---

## Phase 4 — Sandbox & analytical engine ✅

**Deliverable:** every number the system produces is computed and reproducible.

- Sandbox runner: no network, non-root, seccomp, dropped caps, CPU/memory/wall limits,
  one-shot container, image digest recorded
- `resx-sandbox` MCP server: `run_python`, `read_artifact`
- `ComputationRecord` persistence and the `/computations/{id}` audit endpoint
- Deterministic statistics library: descriptives, OLS + Mann-Kendall, CAGR, STL, Pearson and
  Spearman with `p`/`n`/CI, IQR + z-score + isolation forest
- `Decimal` money type, `pint` units, dated FX table, scale-factor resolution
- Accounting identity assertions with backward routing on failure
- matplotlib artifact generation for report export

**Exit:** sandbox escape attempts (network, filesystem, privilege) all fail; a seeded scale
error trips an identity assertion and routes back to ingestion; identity pass rate 1.00 on
the synthetic set.

---

## Phase 5 — Orchestrator & the five agents ✅

**Deliverable:** the multi-agent analysis, streaming live.

- MCP servers: `resx-filesystem` (chrooted), `resx-mongodb` (read-only role + SELECT-only
  parser), `resx-search` (egress allowlist, SSRF guarded)
- Untrusted-content wrapping on every tool result
- LangGraph `StateGraph`: append-only evidence, MongoDB checkpointer, budget decrement,
  node retry, degraded-branch handling
- Manager with dynamic fan-out; Finance, Risk, News, Workflow, Market with per-agent tool
  allowlists and structured output
- SSE event stream; `RunConsole`, `AgentCard`
- Model routing and prompt caching on the shared rules block

**Exit:** a run over a real annual report completes, streams every node, produces cited
claims, and respects its budget; the Finance agent has no web tool in its schema; an
injection payload in a document does not change agent behaviour.

---

## Phase 6 — Critic, debate, synthesis ✅

**Deliverable:** the report — the actual product.

- Critic with independent re-derivation, citation resolution, contradiction hunting
- Debate protocol, bounded at three rounds, with `CONTESTED` as a first-class outcome
- `route_after_critic` including the backward `reingest` edge
- Synthesizer: exactly five insights, reproducible ranking, stated limitations, disclosed
  degraded branches
- `DebateThread`, `InsightCard`, `SwotGrid`
- Report export: JSON, PDF, XLSX — audit-logged
- Optional human-in-the-loop `interrupt_before=["synthesize"]`
- Run-scoped chat: *"why did the Risk agent flag page 42?"*

**Exit:** a seeded error is caught by the Critic and visible in the debate thread; a genuinely
ambiguous figure ships as `CONTESTED` with both positions; the report never contains a number
no agent produced.

---

## Phase 7 — Benchmarks & hardening ◑ partial

**Deliverable:** measured accuracy, and a system that survives contact with reality.

- `benchmarks/gold/`: `synthetic-pnl/`, `filings/`, `adversarial/`, `insight-rated/`
- Scoring harness for all seven metrics, including `critic_catch_rate` via seeded errors
- CI gates: citation validity `1.00`, numeric exactness `≥ 0.98`, hallucination `≤ 0.01`,
  recall@10 `≥ 0.95`, critic catch `≥ 0.90`
- Calibration curve tracked as a first-class metric
- Load test; k6 or Locust against the rate limits and the queue
- Full security suite: SSRF, upload (EICAR, zip bomb, polyglot, SVG-with-script), traversal,
  injection corpus, refresh reuse
- `/security-review` on the accumulated diff; the §10 pre-launch checklist
- OpenTelemetry traces end to end; alerting on grounding failures and budget exhaustion
- Restore-from-backup drill

**Exit:** every gate green in CI; no high-severity security finding; the pre-launch checklist
complete.

---

## Sequencing rationale

Three ordering decisions are worth stating, because the obvious alternative is worse in each
case:

1. **Design before backend (Phase 1 before 2–6).** The UI is the specification made visible.
   Building it early means every later phase has a real place to render into, and it is the
   cheapest point at which to discover that an information architecture does not work.
2. **Security before ingestion (Phase 2 before 3).** Tenancy and rate limits are
   architectural. Adding them after a working pipeline exists means auditing every query
   already written, and that audit is exactly what gets skipped under deadline.
3. **The deterministic engine before the agents (Phase 4 before 5).** The agents are only
   trustworthy because the sandbox exists. Building agents first invites a temporary
   "just let the model add these two numbers" shortcut, and that shortcut never gets removed.

---

## Out of scope for v1

Named explicitly so the boundary is a decision rather than an oversight:

- Real-time streaming data sources (v1 is batch corpus analysis)
- Write-back to source systems — RESX reads and advises, it does not act
- Custom model fine-tuning; prompt and retrieval quality come first and are cheaper
- Multi-language corpora beyond English
- Mobile-native apps; the web app is responsive but desktop-first, because analysts work on
  large screens
