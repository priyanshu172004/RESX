# 02 — Architecture

## 1. Runtime topology

```
                         ┌───────────────────────────┐
   browser ──── HTTPS ──▶│  Next.js 16 (apps/web)    │
                         │  RSC + route handlers/BFF │
                         └────────────┬──────────────┘
                                      │ internal HTTP + SSE
                                      ▼
                         ┌───────────────────────────┐
                         │  FastAPI (services/api)   │
                         │  auth · runs · rag · chat │
                         └────┬──────────┬───────────┘
                              │          │
                 ┌────────────▼──┐   ┌───▼──────────────┐
                 │  LangGraph    │   │  Worker pool     │
                 │  orchestrator │   │  (arq / Redis)   │
                 └───────┬───────┘   └───┬──────────────┘
                         │ MCP stdio/HTTP│
        ┌────────────────┼───────────┬───┴────────────┐
        ▼                ▼           ▼                ▼
  ┌───────────┐   ┌───────────┐ ┌──────────┐   ┌────────────┐
  │filesystem │   │  mongodb  │ │  search  │   │  sandbox   │
  │MCP server │   │MCP server │ │MCP server│   │ MCP server │
  └─────┬─────┘   └─────┬─────┘ └────┬─────┘   └─────┬──────┘
        ▼               ▼            ▼               ▼
   object store    MongoDB +     Tavily        Docker/Firecracker
   (uploads)       the vector index       (web)         (no network)
```

Why a separate FastAPI service rather than doing everything in Next.js route handlers: the
orchestration layer is Python (LangGraph, pandas, statsmodels), runs for minutes rather than
milliseconds, and needs to survive a frontend redeploy mid-run. The BFF in Next exists to
hold the session cookie, apply the edge rate limit and CSP, and avoid exposing the API
directly to the browser.

---

## 2. The orchestration layer (LangGraph)

### 2.1 State

State is the contract between nodes. It is append-only for evidence — nodes add claims, they
never edit another node's claims — which is what makes the debate auditable.

```python
class RunState(TypedDict):
    run_id: str
    workspace_id: str
    question: str
    corpus_ids: list[str]

    plan: Plan | None
    claims: Annotated[list[Claim], operator.add]          # append-only
    verdicts: Annotated[list[Verdict], operator.add]      # append-only
    computations: Annotated[list[ComputationRecord], operator.add]
    debate_rounds: int
    report: ExecutiveReport | None

    budget: Budget          # tokens + dollars + wall-clock, decremented per node
    errors: Annotated[list[NodeError], operator.add]
```

`Annotated[..., operator.add]` is what allows the five specialist agents to run as parallel
branches and have their outputs merged without a write conflict.

### 2.2 Nodes and edges

```python
g = StateGraph(RunState)

g.add_node("manager", manager_node)
for name in ("finance", "risk", "news", "workflow", "market"):
    g.add_node(name, make_specialist_node(name))
g.add_node("critic", critic_node)
g.add_node("debate", debate_node)
g.add_node("synthesize", synthesize_node)

g.set_entry_point("manager")
g.add_conditional_edges("manager", fan_out, [...])   # plan decides which specialists run
for name in (...):
    g.add_edge(name, "critic")
g.add_conditional_edges("critic", route_after_critic, {
    "debate": "debate",          # contradiction found, rounds remaining
    "synthesize": "synthesize",  # consensus, or rounds exhausted
    "reingest": "manager",       # an accounting identity failed — we misread the source
})
g.add_edge("debate", "critic")
g.add_edge("synthesize", END)

graph = g.compile(checkpointer=MongoDBSaver(...), interrupt_before=["synthesize"])
```

Three routing decisions carry the design:

- **`fan_out`** — the Manager's plan, not a static edge list, decides which specialists run.
  A question about process waste should not spend tokens on the Market agent.
- **`reingest`** — a failed accounting identity routes *backwards*. This is the "if the
  Finance agent fails, go back to the Data agent" loop, and it is the difference between a
  pipeline and a graph.
- **`interrupt_before=["synthesize"]`** — an optional human-in-the-loop checkpoint before the
  report is written, for regulated or high-stakes use.

### 2.3 Budget and failure

Each node decrements `budget`. Exhaustion terminates the run cleanly with a partial report
marked incomplete, rather than looping. Node errors are caught, recorded in `errors`, and
retried once with the error text fed back; a second failure marks that branch degraded and
the run continues without it. **A degraded branch is always disclosed in the report** — a
report that silently omits the Risk analysis because the Risk agent crashed is dangerous.

---

## 3. The connectivity layer (MCP)

Four servers, each a separate process with its own privilege boundary.

| Server | Tools | Privilege |
|---|---|---|
| `resx-filesystem` | `list_documents`, `read_chunk`, `read_page`, `get_table` | Read-only, chrooted to one workspace prefix |
| `resx-mongodb` | `list_schemas`, `describe_table`, `query` | A read-only DB role; `query` is parameterized-only and row-capped |
| `resx-search` | `web_search`, `fetch_url` | Egress allowlist, SSRF guarded |
| `resx-sandbox` | `run_python`, `read_artifact` | No network, resource-capped, one-shot container |

Two properties matter more than the tool list:

1. **Swappability.** Moving from MongoDB to another store means replacing one MCP server.
   No agent prompt changes, because agents call `query`, not `psycopg`.
2. **Per-agent scoping.** The tool allowlist is applied when the agent's tool set is built,
   so the grant is structural. The Finance agent has no web tool in its schema at all, which
   is a stronger guarantee than instructing it not to browse.

MCP tool results are **untrusted input.** Every result is wrapped before it reaches a prompt:

```
<untrusted_document_content doc_id="..." page="12">
...retrieved text...
</untrusted_document_content>
```

with a standing system instruction that content inside those tags is data to analyse and
never instruction to follow. See [`05-SECURITY.md`](./05-SECURITY.md) §prompt-injection.

---

## 4. The RAG pipeline

### 4.1 Ingestion

```
upload → magic-byte + size + AV check → quarantine → object store (content-addressed)
      → job queued
      → extract text (per page, with bboxes) AND tables (to typed DataFrames)
      → chunk: layout-aware, target 512 tokens, 64 overlap,
               never split a table row, never cross a page boundary silently
      → each chunk persisted with its anchor: {doc_id, page, para_idx, char_span, bbox}
      → embed in batches → the vector index
      → tables persisted as Parquet + registered in `datasets`
      → status → ready
```

The anchor is the entire point. A chunk without a resolvable anchor is unusable, because it
cannot be cited, and an uncitable chunk cannot support a claim.

### 4.2 Retrieval

Hybrid, because pure vector search fails on the exact thing financial analysis needs most —
specific figures and rare identifiers:

```
query → (a) BM25 lexical over chunk text        → top 50
      → (b) vector kNN (HNSW, cosine)           → top 50
      → reciprocal rank fusion (k=60)           → top 50
      → cross-encoder rerank                    → top 8
      → assemble context with anchors attached
```

Retrieval is additionally **filtered by workspace at the SQL level**, so tenant isolation
holds inside the vector search rather than being applied afterwards.

### 4.3 Numeric path

Anything that looks like a financial table does not stay as text. It becomes a registered
dataset that the sandbox can load:

```python
df = resx.load("ds_a91f")     # inside the sandbox
gross = df.revenue.sum() - df.cogs.sum()
```

This is why the arithmetic can be exact. The agent is not reading "1,250.00" out of a
sentence and adding it in its head; it is summing a `Decimal` column.

---

## 5. Streaming and the run console

A run takes minutes, so the UI must show the work rather than a spinner. The API exposes SSE:

```
GET /api/runs/{id}/events
  event: node_start     {node, ts}
  event: tool_call      {node, tool, args_digest}
  event: claim          {claim}
  event: computation    {computation_id, code_preview, result}
  event: verdict        {claim_id, verdict, rationale}
  event: debate_round   {round, positions[]}
  event: node_end       {node, ms, tokens, cost}
  event: report         {report}
  event: error          {node, message}
```

Showing tool calls and computations live is a trust feature, not a debug feature. A user who
watches the Finance agent open page 12, run a sum, and get challenged by the Critic has a
justified reason to believe the number.

---

## 6. Environments and deployment

| Concern | Local | Production |
|---|---|---|
| Web | `next dev` | Vercel or a container behind the CDN |
| API | `uvicorn --reload` | Containers, horizontally scaled, stateless |
| Orchestrator checkpoints | MongoDB | MongoDB (managed, PITR) |
| Queue | Redis | Managed Redis |
| Sandbox | Docker, no network | Firecracker/gVisor microVM, one-shot |
| Secrets | `.env.local`, git-ignored | KMS / secret manager, never in an image |
| Vectors | the vector index | the vector index (or a managed vector DB if scale demands) |

The sandbox is deliberately the least-trusted component and gets the strongest isolation: it
runs model-generated code, which must be treated as hostile by default even when the model
is well-behaved.
