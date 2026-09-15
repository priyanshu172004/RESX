# 07 — API Contract

Base: `/api/v1`. JSON in, JSON out. Auth by `HttpOnly` cookie. Every request carries a
correlation id in `X-Request-Id` (generated if absent) which appears in every log line and
in error responses.

---

## Conventions

- **IDs** are prefixed UUIDs (`ws_`, `doc_`, `run_`, `clm_`, `cmp_`) — opaque, unguessable,
  no enumeration.
- **Errors** are uniform and never leak internals:

  ```json
  { "error": { "code": "VALIDATION_FAILED", "message": "Human-readable summary",
               "details": [{"field": "email", "issue": "invalid"}],
               "request_id": "req_..." } }
  ```

- **Pagination** is mandatory on collections: `?limit=50&cursor=...` (max `limit` 100),
  returning `{ items: [], next_cursor: string | null }`.
- **Rate-limit headers** on every response; `Retry-After` on 429.
- All mutating verbs require the CSRF double-submit token and an `Origin` check.

---

## Auth

| Method | Path | Body | Notes |
|---|---|---|---|
| `POST` | `/auth/register` | `{email, password, name}` | Password policy checked server-side (length ≥ 12, breach-list check). Returns 202; verification email sent. 3/hour per IP |
| `POST` | `/auth/verify-email` | `{token}` | Single-use, 24h expiry |
| `POST` | `/auth/login` | `{email, password, totp?}` | Sets access + refresh cookies. Generic error on failure. 5/15min per IP **and** per account |
| `POST` | `/auth/refresh` | — | Rotates the refresh token. **Reuse of a spent token revokes the whole family** |
| `POST` | `/auth/logout` | — | Revokes the family, clears cookies |
| `GET` | `/auth/me` | — | `{user, memberships[], active_workspace}` |
| `POST` | `/auth/2fa/enroll` | — | Returns a provisioning URI + 10 recovery codes (shown once) |
| `POST` | `/auth/2fa/confirm` | `{totp}` | Activates 2FA |
| `POST` | `/auth/password` | `{current, next}` | Invalidates all sessions |

---

## Workspaces & members

| Method | Path | Role | Notes |
|---|---|---|---|
| `GET` | `/workspaces` | any | Memberships of the caller only |
| `POST` | `/workspaces` | any | Caller becomes `owner` |
| `GET` | `/workspaces/{id}` | viewer+ | |
| `PATCH` | `/workspaces/{id}` | admin+ | |
| `DELETE` | `/workspaces/{id}` | owner | Soft delete, 30-day purge |
| `GET` | `/workspaces/{id}/members` | viewer+ | |
| `POST` | `/workspaces/{id}/members` | admin+ | Invite by email |
| `PATCH` | `/workspaces/{id}/members/{uid}` | admin+ | Role change → sessions invalidated, audit-logged |
| `DELETE` | `/workspaces/{id}/members/{uid}` | admin+ | Cannot remove the last owner |

---

## Documents

| Method | Path | Role | Notes |
|---|---|---|---|
| `POST` | `/documents` | analyst+ | `multipart/form-data`. Magic-byte + size + AV checks. Returns `{doc_id, status: "queued"}`. 20/hour per workspace |
| `GET` | `/documents` | viewer+ | Filter by `status`, `kind` |
| `GET` | `/documents/{id}` | viewer+ | Includes page count, extraction stats, OCR-confidence warnings |
| `GET` | `/documents/{id}/pages/{n}` | viewer+ | Extracted text plus chunk anchors — powers the citation viewer |
| `GET` | `/documents/{id}/download` | viewer+ | 302 to a signed, expiring, single-use URL on the asset origin |
| `DELETE` | `/documents/{id}` | analyst+ | Cascades to chunks, embeddings, datasets |

`status`: `queued → extracting → chunking → embedding → ready`, or `failed` / `quarantined`
with a reason.

### Datasets (the numeric path)

| Method | Path | Notes |
|---|---|---|
| `GET` | `/datasets` | Tables extracted from documents |
| `GET` | `/datasets/{id}/profile` | The profile from `01-ANALYST-WORKFLOW.md` §1.2 |
| `GET` | `/datasets/{id}/rows` | Paginated, capped |
| `GET` | `/datasets/{id}/transforms` | The immutable transform log |

---

## Runs (analysis)

| Method | Path | Role | Notes |
|---|---|---|---|
| `POST` | `/runs` | analyst+ | `{question, corpus_ids[], agents?[], budget?}`. Returns `{run_id, status:"queued"}`. 10/hour per workspace |
| `GET` | `/runs` | viewer+ | |
| `GET` | `/runs/{id}` | viewer+ | Full state: plan, claims, verdicts, spend, report |
| `GET` | `/runs/{id}/events` | viewer+ | **SSE stream** (below) |
| `POST` | `/runs/{id}/cancel` | analyst+ | Cooperative cancel at the next node boundary |
| `POST` | `/runs/{id}/approve` | analyst+ | Releases the `interrupt_before=["synthesize"]` checkpoint |
| `GET` | `/runs/{id}/report` | viewer+ | `?format=json\|pdf\|xlsx` — export is audit-logged |

### SSE event stream

```
event: node_start    {node, ts}
event: tool_call     {node, tool, args_digest}
event: claim         {claim}
event: computation   {computation_id, code_preview, result}
event: verdict       {claim_id, verdict, rationale}
event: debate_round  {round, positions[]}
event: node_end      {node, ms, tokens, cost_usd}
event: budget        {tokens_left, usd_left}
event: report        {report}
event: error         {node, message, request_id}
event: done          {status}
```

Heartbeat comment every 15s to keep intermediaries from closing the connection. The stream
is resumable with `Last-Event-ID`, because a run outlives a browser tab.

---

## Claims, citations, computations

| Method | Path | Notes |
|---|---|---|
| `GET` | `/claims/{id}` | Claim with citations, verdict, and debate history |
| `GET` | `/citations/{id}/resolve` | The verbatim span with surrounding context — what `CitationPopover` renders |
| `GET` | `/computations/{id}` | `{code, inputs, stdout, result, duration_ms, image_digest}` — full reproducibility |

These three endpoints are the audit surface. They exist so that any figure in the product
can be traced to its source and its arithmetic by a user, not just by an engineer.

---

## Chat (RAG over the corpus)

| Method | Path | Notes |
|---|---|---|
| `POST` | `/chat` | `{thread_id?, message, scope: {corpus_ids?, run_id?}}` → SSE token stream plus a `citations` event. 60/hour per user |
| `GET` | `/chat/threads` | |
| `GET` | `/chat/threads/{id}` | |

Answers are grounded the same way agent claims are: retrieval, then citation resolution
before the tokens are sent. A chat answer that cannot cite says so. When `run_id` is in
scope, the chat can answer *"why did the Risk agent flag page 42?"* from the run state —
the retrieved span, the agent's rationale, and the Critic's verdict.

---

## Metrics & benchmarks (admin)

| Method | Path | Notes |
|---|---|---|
| `GET` | `/metrics/accuracy` | Rolling grounding-failure and hallucination rates |
| `GET` | `/metrics/calibration` | Stated confidence vs. observed accuracy, bucketed |
| `GET` | `/metrics/spend` | Token and dollar spend by workspace and agent |
| `POST` | `/benchmarks/run` | Triggers the gold-dataset suite (also runs in CI) |

---

## Status codes

| Code | Meaning |
|---|---|
| 200 / 201 / 202 | OK / created / accepted for async processing |
| 400 | Malformed request |
| 401 | No or invalid session |
| 403 | Authenticated but not authorized — also returned instead of 404 where existence itself is sensitive |
| 404 | Not found within the caller's tenant scope |
| 409 | Conflict (duplicate, or a state transition that is not legal) |
| 413 | Body or upload too large |
| 415 | Unsupported media type (magic-byte check failed) |
| 422 | Schema validation failed |
| 429 | Rate limited — `Retry-After` present |
| 500 | Internal error, correlation id returned, no detail |
| 503 | Dependency unavailable (queue, model provider) |
