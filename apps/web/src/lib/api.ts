/**
 * The API client.
 *
 * Three decisions here are load-bearing rather than stylistic:
 *
 * 1. **The access token lives in memory, never in `localStorage`.** The refresh
 *    token is an HttpOnly cookie the browser holds and script cannot read. Put
 *    the access token in `localStorage` and an XSS gets a token it can
 *    exfiltrate; keep it in a module variable and the same XSS has to stay
 *    resident to use it, and loses it on reload.
 *
 * 2. **A 401 triggers exactly one refresh, shared across callers.** Six widgets
 *    mounting at once against an expired token would otherwise fire six
 *    refreshes; the first rotates the token and the other five present a
 *    now-used token, which the server correctly treats as replay and revokes
 *    the whole family. So concurrent callers await one in-flight refresh.
 *
 * 3. **Errors carry the server's message.** The API returns a specific reason
 *    ("this file is already ingested as doc_…"), and replacing that with
 *    "Something went wrong" throws away the only useful part.
 */

const BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ??
  "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
    readonly code?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }

  /** True when the caller should send the user to sign in rather than retry. */
  get isAuth() {
    return this.status === 401 || this.status === 403;
  }
}

let accessToken: string | null = null;
const listeners = new Set<(token: string | null) => void>();

export function setAccessToken(token: string | null) {
  accessToken = token;
  listeners.forEach((fn) => fn(token));
}

export function getAccessToken() {
  return accessToken;
}

export function onTokenChange(fn: (token: string | null) => void) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/**
 * Timeouts, per kind of request rather than one number for everything.
 *
 * `fetch` has no default timeout and a request that never settles is worse
 * than one that fails — the session bootstrap awaits one, and a hung socket
 * left the whole app on "Restoring your session…" with no way out. So a clock
 * is needed.
 *
 * But a *single* clock was wrong, and shipping one broke uploads. 15 seconds
 * is generous for reading a list and absurd for ingesting a PDF: extraction,
 * OCR at 300 DPI, chunking and embedding are seconds per page, so a scanned
 * document was aborted mid-ingest and surfaced as "signal timed out". The
 * request was not stuck; it was working.
 *
 * The lesson is that the timeout belongs to the operation. An interactive read
 * that has not answered in half a minute is broken; an ingest that has not
 * answered in half a minute is normal.
 */

/** Interactive reads and mutations. Long enough for a cold database. */
const DEFAULT_TIMEOUT_MS = 30_000;

/**
 * Ingestion and anything else that does real work per page.
 *
 * Ten minutes is not a prediction, it is a ceiling: it exists so a genuinely
 * dead connection eventually fails rather than spinning forever, and it must
 * never be the reason a large document fails to upload.
 */
const LONG_TIMEOUT_MS = 600_000;

/**
 * The session bootstrap, which the UI blocks on.
 *
 * Deliberately the shortest of the three. Everything behind it is unusable
 * until it settles, so a fast, honest "can't reach the API" beats a spinner.
 */
const AUTH_TIMEOUT_MS = 15_000;

/**
 * The outcome of a refresh, with the two failures kept apart.
 *
 * `token: null, reachable: true` means the server answered and declined —
 * there is no session, and the login form is the right destination.
 * `reachable: false` means nothing answered, which is a different situation
 * entirely: the user is probably still signed in, and sending them to a login
 * form that posts to the same dead API just makes them think their password
 * is wrong.
 */
export type RefreshOutcome = { token: string | null; reachable: boolean };

let refreshDetailed: Promise<RefreshOutcome> | null = null;

/**
 * The CSRF token the server issued alongside the session cookie.
 *
 * Read from `document.cookie` rather than stored, because the server rotates
 * it with every refresh and a copy held in memory would go stale exactly when
 * the session renews — turning "your session expired" into a permanent 403.
 *
 * Deliberately readable by script: the token proves a request came from *this
 * origin*, which an attacker's page cannot read across origins. An attacker who
 * can already run script here does not need it.
 */
function csrfToken(): string {
  if (typeof document === "undefined") return "";
  const match = /(?:^|;\s*)resx_csrf=([^;]+)/.exec(document.cookie);
  return match ? decodeURIComponent(match[1]) : "";
}

/** Headers for a request that authenticates from the session cookie. */
function csrfHeaders(): Record<string, string> {
  const token = csrfToken();
  return token ? { "X-CSRF-Token": token } : {};
}

/** Ask the server to rotate the refresh cookie. At most one at a time. */
export async function refreshSessionDetailed(): Promise<RefreshOutcome> {
  if (refreshDetailed) return refreshDetailed;

  refreshDetailed = (async () => {
    try {
      const response = await fetch(`${BASE}/api/v1/auth/refresh`, {
        method: "POST",
        credentials: "include",
        // This route authenticates from a cookie, so it is the one request
        // shape a cross-site page could cause. The echoed token is what the
        // server checks; omitting it is a 403, not a silent failure.
        headers: csrfHeaders(),
        signal: AbortSignal.timeout(AUTH_TIMEOUT_MS),
      });
      if (!response.ok) {
        setAccessToken(null);
        // A 5xx is a broken server rather than an absent one, but it is not a
        // statement about this user's session either, so it is not "signed
        // out" and retrying is the right advice.
        return { token: null, reachable: response.status < 500 };
      }
      const body = (await response.json()) as { access_token: string };
      setAccessToken(body.access_token);
      return { token: body.access_token, reachable: true };
    } catch {
      // A network failure or the timeout above. Nothing answered.
      setAccessToken(null);
      return { token: null, reachable: false };
    } finally {
      // Cleared in `finally` so a rejected refresh does not wedge every
      // subsequent request behind a permanently pending promise.
      refreshDetailed = null;
    }
  })();

  return refreshDetailed;
}

/** The token alone, for callers that only need to retry a request. */
export async function refreshSession(): Promise<string | null> {
  return (await refreshSessionDetailed()).token;
}

async function readError(response: Response): Promise<ApiError> {
  let message = response.statusText || `request failed (${response.status})`;
  let code: string | undefined;
  try {
    const body = await response.json();
    // The API wraps errors as { error: { code, message } }; FastAPI's own
    // guards use { detail }. Both are handled because both reach the client.
    message = body?.error?.message ?? body?.detail ?? message;
    code = body?.error?.code;
    if (Array.isArray(body?.error?.details) && body.error.details.length) {
      const first = body.error.details[0];
      message = `${message} (${first.field}: ${first.issue})`;
    }
  } catch {
    /* a non-JSON body: keep the status text */
  }
  return new ApiError(response.status, message, code);
}

type RequestOptions = Omit<RequestInit, "body"> & {
  body?: unknown;
  /** Set by the retry path so a failed refresh cannot loop. */
  retried?: boolean;
  /** Skip the refresh-and-retry dance entirely (used by /auth routes). */
  anonymous?: boolean;
  /**
   * Override the timeout for one call. `null` disables it entirely, for a
   * stream that manages its own lifetime.
   */
  timeoutMs?: number | null;
};

export async function api<T>(
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const { body, retried, anonymous, headers, timeoutMs, ...rest } = options;

  const finalHeaders = new Headers(headers);
  const isFormData = body instanceof FormData;
  if (body !== undefined && !isFormData) {
    finalHeaders.set("Content-Type", "application/json");
  }
  if (accessToken && !anonymous) {
    finalHeaders.set("Authorization", `Bearer ${accessToken}`);
  }

  let response: Response;
  try {
    response = await fetchWithClock();
  } catch (error) {
    // `AbortSignal.timeout` rejects with a bare DOMException whose message is
    // "signal timed out" — which tells the user nothing about what timed out
    // or how long it waited. Replaced with something actionable.
    if (error instanceof DOMException && error.name === "TimeoutError") {
      const seconds = Math.round((timeoutMs ?? DEFAULT_TIMEOUT_MS) / 1000);
      throw new ApiError(
        408,
        `the server did not respond within ${seconds}s`,
        "timeout",
      );
    }
    throw error;
  }

  async function fetchWithClock(): Promise<Response> {
    return fetch(`${BASE}${path}`, {
      ...rest,
      headers: finalHeaders,
      // Always included: the refresh cookie is scoped to /api/v1/auth, so this
      // is what lets the refresh call see it at all.
      credentials: "include",
      // Precedence: an explicit signal, then an explicit timeout, then the
      // default. `timeoutMs: null` means no clock at all.
      signal:
        rest.signal ??
        (timeoutMs === null
          ? undefined
          : AbortSignal.timeout(timeoutMs ?? DEFAULT_TIMEOUT_MS)),
      body: isFormData
        ? body
        : body === undefined
          ? undefined
          : JSON.stringify(body),
    });
  }

  if (response.status === 401 && !retried && !anonymous) {
    const token = await refreshSession();
    if (token) {
      return api<T>(path, { ...options, retried: true });
    }
  }

  if (!response.ok) throw await readError(response);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

// --------------------------------------------------------------------------- //
// Types — mirroring the API contract, not guessed at
// --------------------------------------------------------------------------- //

export type Role = "owner" | "admin" | "analyst" | "viewer";

export interface User {
  user_id: string;
  email: string;
  name: string;
  workspace_id: string;
  role: Role;
  verified: boolean;
  created_at?: number;
  last_login_at?: number | null;
}

export interface SessionResponse {
  access_token: string;
  token_type: string;
  expires_at: number;
  user: User;
}

/** The formats `/runs/{id}/export` can produce. Mirrors `FORMATS` in the API. */
export type ExportFormat = "pdf" | "docx" | "xlsx" | "md";

export interface DocumentSummary {
  doc_id: string;
  source_name: string;
  kind: string;
  page_count: number;
  status: string;
  low_conf_pages: number[];
  warnings: string[];
}

export interface Dataset {
  dataset_id: string;
  doc_id: string;
  name: string;
  source_page: number | null;
  n_rows: number;
  n_cols: number;
  scale_factor: string;
  currency: string | null;
  agreement: boolean | null;
  profile: Record<string, unknown>;
}

export interface RunSummary {
  run_id: string;
  question: string;
  status: string;
  tokens_used: number;
  usd_used: number;
  created_at: number;
  finished_at: number | null;
}

export interface Kpi {
  key: string;
  label: string;
  value: number;
  format: "integer" | "percent" | "usd" | "ratio";
  detail: string;
  target?: number | null;
}

export interface Capabilities {
  model_provider: string;
  model_ready: boolean;
  search_ready: boolean;
  semantic_embeddings: boolean;
  store: string;
}

export interface AgentActivity {
  agent: string;
  claims: number;
  mean_confidence: number;
  cited_share: number;
}

export interface SpendPoint {
  run_id: string;
  label: string;
  usd: number;
  tokens: number;
  at: number;
}

export interface Citation {
  citation_id: string;
  doc_id: string | null;
  page: number | null;
  para_idx: number | null;
  quote: string;
  resolution: string;
  match_score: number | null;
  url?: string | null;
}

/**
 * One review of one claim, as the Critic returned it.
 *
 * `rationale` is the part that was stored and never shown: the run page
 * displayed a verdict word and nothing about *why*, which asks a reader to
 * accept "refuted" on the same faith the whole system exists to remove.
 */
export interface VerdictRow {
  verdict_id: string;
  claim_id: string;
  verdict: "confirmed" | "refuted" | "contested" | string;
  /** What the Critic computed for itself, when it re-derived the figure. */
  independent_value: string | null;
  rationale: string;
  /** 0 is the first review; later rounds are the author answering back. */
  debate_round: number;
  created_at: number;
}

export interface ClaimRow {
  claim_id: string;
  run_id: string;
  question: string;
  agent: string;
  statement: string;
  value: string | null;
  unit: string | null;
  currency: string | null;
  period: string | null;
  confidence: number;
  computation_id: string | null;
  citations: Citation[];
  verdict: string | null;
  /** Every round, newest first. Empty when nothing reviewed this claim. */
  verdicts?: VerdictRow[];
  created_at: number;
}

export interface Dashboard {
  workspace: { workspace_id: string; name: string };
  summary: {
    documents: number;
    ready: number;
    pages: number;
    chunks: number;
    datasets: number;
    runs: number;
    claims: number;
    computations: number;
    citation_validity: number;
    total_citations: number;
  };
  kpis: Kpi[];
  documents: DocumentSummary[];
  datasets: Dataset[];
  runs: RunSummary[];
  latest_report: Record<string, unknown> | null;
  spend: SpendPoint[];
  agent_activity: AgentActivity[];
  capabilities: Capabilities;
  empty_reason: string | null;
}

export interface AgentInfo {
  agent: string;
  title: string;
  role: string;
  description: string;
  /** Plain-language capability labels, not internal tool identifiers. */
  capabilities: string[];
  /** Derived from the enforcing allowlist, so it is a checkable claim. */
  network_access: boolean;
  claims: number;
  mean_confidence: number;
  cited_share: number;
}

export interface Passage {
  chunk_id: string;
  doc_id: string;
  page: number;
  para_idx: number | null;
  section: string | null;
  kind: string;
  char_start: number;
  char_end: number;
  text: string;
  score: number;
  source: string;
}

export interface IngestResponse {
  doc_id: string;
  source_name: string;
  kind: string;
  pages: number;
  chunks: number;
  datasets: string[];
  anchor_completeness: number;
  low_confidence_pages: number[];
  warnings: string[];
  embedded: boolean;
  embedder: string;
  duration_ms: number;
  status: string;
  analysis?: { started: boolean; run_id?: string; reason?: string } | null;
}

export interface BenchmarkMetric {
  key: string;
  label: string;
  value: number;
  target: number | null;
  comparator: string | null;
  format: string;
  note: string;
}

// --------------------------------------------------------------------------- //
// Endpoints
// --------------------------------------------------------------------------- //

export const auth = {
  register: (body: {
    email: string;
    password: string;
    name: string;
    workspace_name?: string;
  }) =>
    api<SessionResponse>("/api/v1/auth/register", {
      method: "POST",
      body,
      anonymous: true,
    }),

  login: (body: { email: string; password: string }) =>
    api<SessionResponse>("/api/v1/auth/login", {
      method: "POST",
      body,
      anonymous: true,
    }),

  logout: () =>
    api<void>("/api/v1/auth/logout", {
      method: "POST",
      headers: csrfHeaders(),
    }),

  me: () =>
    api<{ user: User; workspace: Record<string, unknown> }>("/api/v1/auth/me"),

  changePassword: (body: { current_password: string; new_password: string }) =>
    api<void>("/api/v1/auth/change-password", { method: "POST", body }),

  audit: () =>
    api<{ entries: Record<string, unknown>[] }>("/api/v1/auth/audit"),
};

export const workspace = {
  dashboard: () => api<Dashboard>("/api/v1/dashboard"),
  agents: () =>
    api<{ agents: AgentInfo[]; runs_considered: number }>("/api/v1/agents"),
  insights: (limit = 25) =>
    api<{
      claims: ClaimRow[];
      total: number;
      contested: number;
      ungrounded_numeric: number;
    }>(`/api/v1/insights?limit=${limit}`),
  benchmarks: () =>
    api<{
      metrics: BenchmarkMetric[];
      not_measured: { key: string; reason: string }[];
      gold_corpus_note: string;
    }>("/api/v1/benchmarks"),
  reset: () =>
    api<{ deleted: Record<string, number>; dataset_files_removed: number }>(
      "/api/v1/workspace/reset",
      { method: "POST", body: { confirm: "DELETE" } },
    ),
};

export const documents = {
  list: () => api<DocumentSummary[]>("/api/v1/documents"),

  get: (docId: string) =>
    api<DocumentSummary & { datasets: Dataset[] }>(
      `/api/v1/documents/${encodeURIComponent(docId)}`,
    ),

  page: (docId: string, page: number) =>
    api<{ doc_id: string; page: number; text: string }>(
      `/api/v1/documents/${encodeURIComponent(docId)}/pages/${page}`,
    ),

  remove: (docId: string) =>
    api<{ deleted: string; datasets_removed: number; files_removed: number }>(
      `/api/v1/documents/${encodeURIComponent(docId)}`,
      { method: "DELETE" },
    ),

  upload: (file: File, analyse?: string) => {
    const form = new FormData();
    form.append("file", file);
    const query = analyse?.trim()
      ? `?analyse=${encodeURIComponent(analyse.trim())}`
      : "";
    return api<IngestResponse>(`/api/v1/documents${query}`, {
      method: "POST",
      body: form,
      // The request does the ingestion, so it is long by nature: extraction,
      // OCR, chunking and embedding all happen before it answers. The default
      // clock aborted scanned PDFs mid-ingest.
      timeoutMs: LONG_TIMEOUT_MS,
    });
  },

  datasets: (docId?: string) =>
    api<Dataset[]>(
      `/api/v1/datasets${docId ? `?doc_id=${encodeURIComponent(docId)}` : ""}`,
    ),

  search: (q: string, topK = 8) =>
    api<{
      query: string;
      chunks: Passage[];
      diagnostics: Record<string, unknown> | null;
    }>(`/api/v1/search?q=${encodeURIComponent(q)}&top_k=${topK}`),
};

export const runs = {
  list: () => api<{ runs: RunSummary[] } | RunSummary[]>("/api/v1/runs"),

  create: (body: {
    question: string;
    corpus_ids?: string[];
    /** Answer from public web sources instead of an uploaded corpus. */
    research?: boolean;
  }) =>
    api<{ run_id: string; status: string }>("/api/v1/runs", {
      method: "POST",
      body,
    }),

  get: (runId: string) =>
    api<Record<string, unknown>>(`/api/v1/runs/${encodeURIComponent(runId)}`),

  report: (runId: string) =>
    api<Record<string, unknown>>(
      `/api/v1/runs/${encodeURIComponent(runId)}/report`,
    ),

  /** SSE URL. Built here so the token-in-query decision lives in one place. */
  eventsUrl: (runId: string) =>
    `${BASE}/api/v1/runs/${encodeURIComponent(runId)}/events`,

  /**
   * Download the report in one of the export formats.
   *
   * Fetched rather than linked. A plain `<a href>` cannot carry the bearer
   * token, so the browser would follow it anonymously and be handed a 401 —
   * as an HTML error page saved under the report's filename, which is the
   * worst of both outcomes because it looks like a successful download.
   *
   * Generating a PDF of a dozen figures is seconds of server CPU, so this gets
   * the long clock rather than the 30s default.
   */
  exportReport: async (runId: string, format: ExportFormat): Promise<void> => {
    const response = await fetch(
      `${BASE}/api/v1/runs/${encodeURIComponent(runId)}/export?format=${format}`,
      {
        headers: accessToken
          ? { Authorization: `Bearer ${accessToken}` }
          : undefined,
        credentials: "include",
        signal: AbortSignal.timeout(LONG_TIMEOUT_MS),
      },
    );
    if (!response.ok) throw await readError(response);

    const blob = await response.blob();
    // The server already picked a safe filename and put it in the header; it
    // is parsed rather than rebuilt so the two cannot disagree.
    const disposition = response.headers.get("Content-Disposition") ?? "";
    const match = /filename="([^"]+)"/.exec(disposition);
    const filename = match?.[1] ?? `report.${format}`;

    const url = URL.createObjectURL(blob);
    const anchor = window.document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    URL.revokeObjectURL(url);
  },
};

export const chat = {
  ask: (body: { question: string; run_id?: string; top_k?: number }) =>
    api<{
      question: string;
      passages: Passage[];
      claims: ClaimRow[];
      answer_mode: string;
      note: string;
    }>("/api/v1/chat", { method: "POST", body }),
};

export const health = () =>
  api<Record<string, unknown>>("/health", { anonymous: true });

export const API_BASE = BASE;
