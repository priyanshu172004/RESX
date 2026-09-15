# 08 — Data model (MongoDB)

The production store is **MongoDB**, implemented in
[`services/api/app/store/mongo.py`](../services/api/app/store/mongo.py). SQLite
([`store/local.py`](../services/api/app/store/local.py)) implements the same
interface and remains the zero-dependency development backend — it is what lets
`pytest` and `benchmarks/score.py` run on a clean checkout with no server.

`STORE_BACKEND` selects between them. `auto` (the default) uses MongoDB when
`MONGODB_URL` is set and reachable and otherwise falls back to SQLite with a
warning; an explicit `mongo` **refuses to start** rather than degrading, because
silently dropping a production deployment onto a single file is worse than
failing to boot.

---

## 1. Why a document store, and what it costs

The corpus is document-shaped. A page has a variable number of blocks; a chunk
has an optional bounding box; a table's profile has a different shape per table.
Modelling that relationally means either a wide sparse table or a join per
attribute. Here it is one document.

Three things a relational schema gave for free have to be built explicitly, and
each is called out where it appears below:

| Lost | Replaced by |
|---|---|
| `CHECK` constraints | Collection-level `$jsonSchema` validators with `validationAction: "error"` |
| `ON DELETE CASCADE` | Explicit cascade in `delete_document` / `wipe_workspace` |
| `SEQUENCE` for ordering | `find_one_and_update` on a counter document |

Row-level security has no MongoDB equivalent at all. That is the significant
loss and it is addressed in §6.

---

## 2. Collections

`workspace_id` is present on **every** document in every collection. Not for
convenience — it is the tenant boundary, and §6 explains why it is a required
keyword argument rather than an optional filter.

### `users`

```js
{
  user_id: "usr_a9476c8a903e",        // unique
  email: "analyst@example.com",       // unique, lowercased before write
  password_hash: "$argon2id$v=19$…",  // never leaves the store
  name: "Priya Sharma",
  workspace_id: "ws_2adfa934c861",
  role: "owner",                      // owner | admin | analyst | viewer
  verified: true,
  failed_logins: 0,
  locked_until: null,                 // epoch seconds; progressive lockout
  totp_secret: null,                  // encrypted at rest when set
  created_at: 1788417593.147,
  last_login_at: null
}
```

Each registration creates its own workspace. Joining an existing one is an
invitation flow — a separate, authorised action — because defaulting to a shared
workspace would make every document mutually visible on first sign-up.

### `refresh_tokens`

```js
{
  token_hash: BinData(…),   // SHA-256 of the token. The plaintext is never stored.
  user_id: "usr_…",
  family_id: "fam_…",       // the rotation chain
  parent_hash: BinData(…),  // the token this one replaced
  expires_at: 1789022393.0,
  used_at: null,            // set on rotation; a second use is replay
  revoked: false,
  created_at: 1788417593.147
}
```

Only the digest is persisted, so a database dump does not yield usable sessions.
`used_at` is what makes reuse detectable: a token presented twice means it was
captured, and the response is to revoke the entire `family_id` rather than the
single replayed token — leaving the attacker's copy of the *next* token working
would be no protection at all.

### `documents`

```js
{
  doc_id: "doc_0f223cfead34",
  workspace_id: "ws_…",
  source_name: "annual-report-fy2025.pdf",
  kind: "pdf",                    // pdf | xlsx | csv | docx | txt
  sha256: "9f2a…",                // unique per (workspace_id, sha256)
  page_count: 148,
  status: "ready",                // queued | extracting | ready | failed
  failure_reason: null,
  low_conf_pages: [42, 43],       // OCR confidence below the floor
  warnings: ["page 42 appears to be a scan"],
  created_at: 1788417593.147
}
```

Content-addressed: the same bytes are the same document. A re-upload is a 409,
not a duplicate — re-ingesting would double every chunk and skew retrieval.

### `pages`

```js
{ doc_id, workspace_id, page: 3, text: "…", section: "Segment results" }
```

The extracted text, kept whole. It is what the citation resolver matches a
quote against, so it has to survive independently of chunking.

### `chunks`

```js
{
  chunk_id: "chk_…",
  workspace_id: "ws_…",
  doc_id: "doc_…",
  page: 3,
  para_idx: 2,                  // the full citation anchor
  section: "Segment results",
  char_start: 1840,
  char_end: 2310,
  bbox: [72.0, 421.5, 523.0, 604.2],
  text: "Revenue for the year was …",
  token_count: 118,
  kind: "prose",                // prose | table | heading
  table_name: null,
  embedding: BinData(…)         // float32 buffer, not an array of doubles
}
```

**Why the embedding is bytes.** A 1024-dimension vector as a BSON array of
doubles is 8 KB and decodes to a Python list of a thousand floats on every read.
As a `Binary` of float32 it is 4 KB and `np.frombuffer` is a view. Over a
100,000-chunk corpus that is the difference between a loadable matrix and an
unusable one.

### `datasets`

```js
{
  dataset_id: "ds_…", workspace_id, doc_id,
  name: "income_statement", source_page: 3,
  path: "storage/datasets/ds_….parquet",
  n_rows: 12, n_cols: 4,
  profile: { revenue: { dtype: "decimal", nulls: 0, min: "…", max: "…" } },
  scale_factor: "1000",     // a STRING, deliberately
  currency: "USD",
  agreement: true,          // did both table extractors agree?
  created_at: 1788417593.147
}
```

`scale_factor` is a string because it is exact. A float `1000.0` would
reintroduce precisely the rounding this system exists to eliminate, and a table
headed "in thousands" read at face value is wrong by 1000x — the most damaging
error the pipeline can make.

`agreement: false` is a flag for a human, not a defect to average away. Two
extractors disagreeing means the table needs reading before its numbers are
trusted.

### `computations`

```js
{
  computation_id: "cmp_ab7be99afe87",
  workspace_id, run_id, agent: "finance",
  code: "import pandas as pd\n…",
  inputs: ["ds_…"],
  stdout: "", stderr: "",
  result: { "Q4": "398000" },
  duration_ms: 412,
  image_digest: "sha256:…",   // the exact sandbox that ran it
  ok: true, error: null,
  created_at: 1788417593.147
}
```

The audit trail for every number in the system. `image_digest` is what makes a
computation *reproducible* rather than merely recorded: the same code on a
different image is a different computation.

### `claims` — the validated collection

```js
{
  claim_id: "clm_…", workspace_id, run_id, agent: "finance",
  statement: "Gross profit for Q4 FY2025 was 760,000.",
  value: "760000",              // string: Decimal in, Decimal out
  unit: "USD", period: "Q4 FY2025", currency: "USD",
  scale_factor: "1",
  computation_id: "cmp_…",      // REQUIRED whenever value is not null
  confidence: 0.99,
  confidence_reason: null,
  payload: {},
  created_at: 1788417593.147
}
```

The validator installed on this collection is the storage-level expression of
*the LLM never does arithmetic*:

```js
{
  $jsonSchema: {
    bsonType: "object",
    required: ["claim_id", "workspace_id", "run_id", "agent", "statement", "confidence"],
    properties: { confidence: { bsonType: ["double","int"], minimum: 0, maximum: 1 } }
  },
  $or: [
    { value: { $in: [null] } },
    { value: { $exists: false } },
    { computation_id: { $type: "string", $ne: null } }
  ]
}
```

A claim may have no value (a qualitative finding), or a value **and** the id of
the computation that produced it. Never a value alone. This duplicates the
Pydantic validator on purpose: if a bug ever bypasses the model, the write still
fails.

`ensure_schema()` raises rather than continuing if the validator cannot be
installed. A collection created implicitly by a first insert has no validator at
all, and the rule would then hold only in Python — which is exactly the
situation the constraint exists to rule out.

### `citations`

```js
{
  citation_id: "cit_…", workspace_id, claim_id,
  chunk_id, doc_id, page: 3, para_idx: 2,
  char_start: 1840, char_end: 2310,
  quote: "Revenue for the year was $48.92 million",
  url: null, publisher: null, retrieved_at: 1788417593.147,
  resolution: "ok",       // ok | quote_mismatch | anchor_not_found | external
  match_score: 0.997
}
```

`resolution` is an enum in the validator. `quote_mismatch` records a *rejected*
citation rather than deleting it — the rejection is the evidence that the gate
fired, and `citation_validity` is computed from these rows.

### `verdicts`

```js
{
  verdict_id: "vd_…", workspace_id, claim_id,
  verdict: "confirmed",          // confirmed | refuted | contested
  independent_value: "760000",   // the Critic's own re-derivation
  rationale: "…",
  debate_round: 0,
  created_at: 1788417593.147
}
```

`contested` is a first-class outcome, not a failure. Disagreement that survives
three rounds ships with both positions: averaging two irreconcilable readings
produces a number neither agent believes.

### `run_events` and `event_counters`

```js
// run_events — unique on (run_id, seq)
{ run_id, workspace_id, seq: 12, kind: "claim", payload: {…}, ts: … }

// event_counters — one document per run
{ _id: "run_…", seq: 12 }
```

**Why a counter document.** Sequence numbers are allocated with
`find_one_and_update(..., $inc)`, which is atomic. `MAX(seq) + 1` is not:
specialists run in parallel and emit concurrently, so a read-then-write hands
two of them the same number — and the unique index then rejects one event
outright, silently losing it from the stream. The SSE endpoint resumes from
`Last-Event-ID`, so unique and gapless sequence numbers are what make a
reconnect neither replay nor skip.

### `audit_log`

```js
{ audit_id, workspace_id, actor, action: "auth.login", target, detail: {}, ip, ts }
```

Append-only. Nothing in the application updates or deletes a row here.

---

## 3. Indexes

Created idempotently by `ensure_schema()` on every boot.

| Collection | Index | Why |
|---|---|---|
| `documents` | `(workspace_id, sha256)` **unique** | content addressing, per tenant |
| `documents` | `doc_id` unique, `(workspace_id, created_at ↓)` | lookup and listing |
| `pages` | `(doc_id, page)` **unique** | one page, one row |
| `chunks` | `chunk_id` unique | hydration by id |
| `chunks` | `(workspace_id, doc_id, page)` | the tenant-scoped vector load |
| `datasets` | `dataset_id` unique, `(workspace_id, doc_id)` | |
| `runs` | `run_id` unique, `(workspace_id, created_at ↓)` | |
| `claims` | `claim_id` unique, `(workspace_id, run_id)` | the report query |
| `citations` | `citation_id` unique, `claim_id` | resolve-and-attach |
| `run_events` | `(run_id, seq)` **unique** | SSE resumption |
| `users` | `user_id` unique, `email` unique | |
| `refresh_tokens` | `token_hash` unique, `family_id`, `user_id` | reuse detection |

---

## 4. Vector search

`$vectorSearch` is an **Atlas-only** aggregation stage. It is used when
available; otherwise the workspace's embedding matrix is loaded and scored with
numpy, which is what the SQLite backend already does and what a local
`docker compose up mongo` gets.

The tenant filter is part of the query in both paths:

```python
query = {"workspace_id": workspace_id, "embedding": {"$ne": None}}
if doc_ids:
    query["doc_id"] = {"$in": list(doc_ids)}
```

That ordering matters. Filtering *after* a similarity search means the search
itself ranged over other tenants' vectors, and a scoring bug would then leak
across the boundary. Isolation has to hold *during* the search.

**Scaling note, stated honestly.** Loading the matrix is O(corpus) per query and
is fine to roughly 10⁵ chunks. Past that, Atlas Vector Search with an HNSW index
is the answer, and the `load_vectors` seam is where it goes.

---

## 5. Transactions

A multi-document transaction needs a replica set. `docker-compose.yml` starts
MongoDB as a single-node replica set (`--replSet rs0`) specifically so that
change streams and transactions are available rather than failing at runtime.

`MongoStore.tx()` nevertheless yields the database rather than opening a
session. The writes that must be atomic are expressed as *single documents*
instead — which is why a claim embeds nothing and citations carry their own
`claim_id`. Ordering is what carries the invariant:

```python
for record in final["computations"]:
    store.add_computation(...)     # first
for claim in final["claims"]:
    store.add_claim(...)           # then — its computation_id must exist
```

A claim written before its computation would reference a row that is not there
yet, and the validator would reject it.

---

## 6. Tenancy — the part with no database backstop

Postgres offered row-level security: a second, independent enforcement point
below the application. **MongoDB has no equivalent.** So in this design the
application filter is not one of two defences, it is the only one.

That is why it is enforced by *signature*:

```python
def _require_workspace(workspace_id: str) -> str:
    if not workspace_id or not workspace_id.strip():
        raise StoreError("workspace_id is required on every store operation")
    return workspace_id
```

Every read and write takes `workspace_id` as a **required keyword argument** and
calls this first. An empty value raises rather than silently widening the query
to every tenant. A method that forgets it does not compile-and-leak; it fails on
the first call, in a test.

The workspace itself comes from the verified JWT claim and nowhere else — never
from a request body, a query parameter, or a header. A client-supplied workspace
id is a cross-tenant read with extra steps.

Three practices follow, and all three are tested in
[`tests/test_auth.py`](../services/api/tests/test_auth.py):

1. A cross-tenant fetch returns **404, not 403**. Confirming that an id exists
   is itself a leak.
2. Deletion cascades explicitly. An orphaned chunk is worse than a missing
   document: it stays retrievable, so it would be cited against a source the
   user believes they deleted.
3. `wipe_workspace` deletes corpus artefacts and leaves `users` untouched.
   Clearing a corpus should not sign the operator out of their own workspace.

Compensating controls for the absent RLS layer, in order of value:

- **A least-privilege connection user.** The API's MongoDB user has
  `readWrite` on one database and nothing else. The read-only MCP server gets a
  separate user with `read` only — stronger than any query parsing the tool
  could do.
- **The signature-level filter above**, which is the primary control.
- **Explicit tenancy tests** that assert the boundary rather than inferring it
  from a route returning 200.

---

## 7. Retention

| Data | Retention |
|---|---|
| Documents, chunks, datasets | Until deleted by the workspace |
| `run_events` | 90 days (the SSE log, not the record) |
| `refresh_tokens` | Purged past `expires_at` by `purge_expired_refresh_tokens()` |
| `audit_log` | 1 year, append-only |
| `computations` | Kept with their claims — deleting one makes its claim unverifiable |

---

## 8. Migrating from the SQLite backend

There is no automatic migration and none is planned. The SQLite backend is for
development and for the benchmark harness; a corpus is re-ingestible from its
source documents, which is cheaper and more trustworthy than translating a
schema. To move: point `MONGODB_URL` at the cluster, set
`STORE_BACKEND=mongo`, and re-run `resx.py ingest`.

Users are the exception — they cannot be re-derived. Registering again in the
new store is the intended path for a development database; a production
migration would export `users` and `refresh_tokens` and insert them directly,
and the password hashes carry their own parameters so they transfer unchanged.
