"""Typed application settings.

Every value is validated at startup, so a misconfigured deployment fails loudly
on boot rather than at the first request that happens to need the value.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "staging", "production"]


def repo_root() -> Path:
    """Walk up for the marker that identifies the checkout.

    Relative paths in `.env` are written relative to the repository, not to
    whichever directory a process happens to start in. uvicorn runs from
    `services/api`, the CLI and the benchmarks from the root, and the sandbox
    from a temporary directory — so resolving against the cwd silently gives
    each of them a different database.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pnpm-workspace.yaml").exists() or (candidate / ".git").exists():
            return candidate
    # Installed rather than checked out: app/core/config.py -> services/api.
    return here.parents[3]


def resolve_path(raw: str) -> Path:
    """Absolute paths are honoured; relative ones anchor to the repository."""
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (repo_root() / path).resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../../.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Core ---
    resx_env: Environment = "development"
    #: Shown in an authenticator app beside the account. Named here so the
    #: label a user sees does not depend on which module built the URI.
    app_name: str = "RESX"

    resx_log_level: str = "info"
    resx_base_url: str = "http://localhost:3000"

    # --- Database: MongoDB ---
    # `auto` uses MongoDB when MONGODB_URL is set and reachable, and otherwise
    # falls back to the SQLite backend so a clean checkout still runs its tests
    # and benchmarks with no server at all. An explicit value is honoured:
    # `mongo` refuses to start rather than silently degrading, which is what you
    # want in production, and `sqlite` never tries to reach a server.
    store_backend: Literal["auto", "mongo", "sqlite"] = "auto"
    mongodb_url: str = ""
    mongodb_database: str = "resx"
    mongodb_timeout_ms: int = Field(default=5_000, ge=250)
    # A user with no write privileges, handed to the read-only MCP server.
    # Stronger than any query parsing we could do in the tool itself.
    mongodb_readonly_url: str = ""
    db_max_page_size: int = 100
    # The SQLite file used by the fallback backend and the CLI. Empty means
    # "derive it from STORAGE_LOCAL_ROOT", which keeps the database beside the
    # uploads it describes -- pointing one at a temp directory and the other at
    # the repo is how a test ends up reading a different database than the app.
    sqlite_path: str = ""

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Auth ---
    jwt_private_key_path: str = "./secrets/jwt_private.pem"
    jwt_public_key_path: str = "./secrets/jwt_public.pem"
    jwt_algorithm: Literal["RS256"] = "RS256"
    jwt_issuer: str = "resx"
    jwt_audience: str = "resx-web"
    access_token_ttl_seconds: int = 900
    refresh_token_ttl_seconds: int = 604_800

    # Argon2id. Memory-hardness is the entire point of choosing it, so the
    # floor is enforced rather than merely documented.
    argon2_time_cost: int = Field(default=3, ge=2)
    argon2_memory_cost: int = Field(default=65_536, ge=65_536)
    argon2_parallelism: int = Field(default=4, ge=1)

    field_encryption_key: str = "replace-with-32-byte-base64-key"

    # Progressive lockout. Counted per account, not per IP: an attacker rotates
    # addresses freely, and the thing being protected is the account.
    login_max_attempts: int = Field(default=8, ge=3)
    login_lockout_seconds: int = Field(default=900, ge=30)
    # Password floor. Length is the only requirement that reliably correlates
    # with strength, so it is the only one enforced -- composition rules push
    # users towards `Password1!` and a reused password is the real risk.
    password_min_length: int = Field(default=12, ge=8)
    # Whether a newly registered account may sign in before verifying its
    # email. True in development so the flow is usable without a mail server.
    allow_unverified_login: bool = True
    # Set on the refresh cookie. False in development because localhost is
    # plain HTTP; refused in production by the validator below.
    cookie_secure: bool = False
    cookie_domain: str = ""

    # The X-Resx-Workspace escape hatch used by the CLI and the benchmark
    # harness. Off by default and refused in production: a header that selects
    # a tenant is a cross-tenant read for anyone who guesses an id, so it must
    # be switched on deliberately rather than inherited from a default.
    allow_dev_workspace_header: bool = False
    # How many proxies sit in front of this service. Zero means X-Forwarded-For
    # is ignored entirely -- the header is client-supplied, and reading it
    # unconditionally lets anyone forge the address in the audit log.
    trusted_proxy_hops: int = Field(default=0, ge=0, le=8)

    # --- Model providers ---
    # Two providers are supported. "auto" resolves at boot from whichever key
    # is present, preferring Groq, so a developer with only a free Groq key
    # needs no further configuration. An explicit value is honoured even if the
    # matching key is missing, because silently running on a provider the
    # operator did not choose is worse than failing to start.
    llm_provider: Literal["auto", "groq", "anthropic"] = "auto"

    anthropic_api_key: str = ""
    # Highest-stakes reasoning: Finance, Risk, Critic, Synthesizer. Errors in
    # these nodes are the expensive ones.
    resx_model_reasoning: str = "claude-opus-5"
    # Search-and-classify and framework application: Manager, News, Market,
    # Workflow.
    resx_model_fast: str = "claude-sonnet-5"

    # Groq is OpenAI-compatible, so it is reached over plain HTTP with the
    # httpx client already in the dependency set rather than a second SDK.
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # Groq has no thinking-effort control, so the routing pair is chosen for
    # schema-following ability first: the 120B model for the nodes whose output
    # is validated hard, the 20B for search-and-classify.
    #
    # Groq withdraws model ids without notice — the Llama 3 pair that was the
    # obvious choice here is already gone — so `GroqLLM.preflight()` checks
    # these against the live catalog and names the replacements rather than
    # letting every node 404 in turn.
    groq_model_reasoning: str = "openai/gpt-oss-120b"
    groq_model_fast: str = "openai/gpt-oss-20b"
    # Groq's free tier rate-limits aggressively. Retries are spaced rather
    # than immediate, because an immediate retry against a 429 just burns the
    # next token in the bucket.
    # The free tier is 8,000 tokens per minute, and a Critic call over a dozen
    # claims is most of that. Five retries with a ceiling above the one-minute
    # window is what lets a run wait out the limit instead of failing inside
    # it -- Groq states the required wait in its 429 body and it is honoured.
    groq_max_retries: int = 5
    groq_retry_backoff_seconds: float = 2.0
    groq_max_wait_seconds: float = 75.0
    # Ceiling on the *input* of one request, in tokens. Groq screens a request
    # before running the model and rejects an oversized one with HTTP 413, so
    # an over-long prompt is trimmed client-side instead. This counts input
    # only -- verified against the live API, a 78-token prompt with a 60,000
    # completion cap is accepted -- so the completion budget is not subtracted
    # from it. Held under the free tier's 8,000/minute with margin: a request
    # at the full limit would leave nothing for its own response inside the
    # same window and stall on a 429 instead.
    groq_max_request_tokens: int = 6_000
    # The provider's tokens-per-minute quota, paced client-side.
    #
    # The graph runs the specialists as parallel branches, so without this
    # three of them send ~4,000 tokens each in the same instant against a
    # limit of 8,000. The provider 429s whichever arrives last, every
    # branch retries into the same exhausted window, and after the retries
    # are spent one specialist contributes nothing -- the report quietly
    # loses a whole section and the run still reports success.
    #
    # Set this to the tier's actual limit. It makes a run slower and
    # complete, instead of fast and missing an agent.
    groq_tokens_per_minute: int = 8_000

    # --- Retrieval ---
    # `gemini` is here for the rate limit rather than the quality: Voyage's
    # free tier is 3 requests and 10,000 tokens per minute, and the token
    # ceiling binds first, so a real PDF spends most of its ingest waiting out
    # 429s. Gemini's free tier is materially more generous.
    #
    # Switching provider invalidates every stored vector. An embedding is only
    # meaningful inside its own model's space, so a corpus embedded with one
    # provider and queried with another returns confident nonsense rather than
    # an error. `build_embedder` refuses a model name belonging to a different
    # provider, which catches the common half of that mistake; detecting a
    # corpus embedded by a *previous* provider needs the embedder recorded per
    # document, which the schema does not yet do. Re-ingest after switching.
    embedding_provider: Literal["voyage", "gemini", "hashing"] = "voyage"
    voyage_api_key: str = ""
    gemini_api_key: str = ""
    #: Left empty to take the provider's default (`voyage-3` /
    #: `gemini-embedding-001`), so changing provider does not also require
    #: remembering to change the model name.
    embedding_model: str = ""
    embedding_dimensions: int = 1024
    # Batch size and pacing for the embedding provider. Voyage's free tier is
    # 3 requests/min and 10k tokens/min; leaving RPM at 0 disables client-side
    # pacing and relies on the retry path instead, which works but costs a full
    # minute per rejection.
    embedding_batch_size: int = 128
    # Tokens per request. The free tier's 10,000/minute is the limit that
    # actually rejects a request, so batching by count alone guarantees a 429
    # however well the requests are paced.
    embedding_token_budget: int = 7_000
    embedding_requests_per_minute: int = 3
    retrieval_top_k: int = 8
    retrieval_candidates: int = 50
    # Ceiling on the retrieved text handed to one agent, in characters.
    #
    # This was 24,000 -- a sensible figure for a 200k-token Claude context, and
    # the wrong one for Groq's free tier, where the whole request must fit in
    # 8,000 tokens. 24,000 chars is ~7,500 tokens on its own; added to a
    # ~2,600-token system prompt it exceeded the limit and every specialist
    # died with HTTP 413 before the model ran.
    #
    # Derived, not picked: `groq_max_request_tokens` (6,000) x 3.2 chars/token
    # is a 19,200-char request, and the largest system prompt plus output
    # schema is ~8,300 of that. 7,000 leaves roughly 3,900 chars of headroom
    # for the sub-question, the computation results and the phase instructions.
    # `test_the_retrieval_budget_fits_the_request_ceiling` holds this true.
    #
    # Reduced from 7,000 when the Critic's instructions grew: the budget
    # test refused the change, correctly, because the total had crept to
    # 17,670 of 19,200 and left too little for the sub-question and the
    # computation results. Paying for a longer prompt out of retrieval is
    # the right trade here -- ~300 tokens less evidence per call against a
    # reviewer that no longer refutes everything it cannot verify -- and
    # paying for it by raising the request ceiling would not be: 7,000
    # tokens of input leaves nothing for the response inside the same
    # 8,000-per-minute window, so the run would stall on 429s instead.
    retrieval_context_chars: int = 6_000

    # --- Search ---
    tavily_api_key: str = ""
    # Empty means deny all. An allowlist is the SSRF and exfiltration control,
    # so it fails closed.
    search_egress_allowlist: str = ""

    # --- Storage ---
    storage_backend: Literal["local", "s3"] = "local"
    storage_local_root: str = "./storage"

    # --- Sandbox: the least-trusted component in the system ---
    sandbox_image: str = "resx/sandbox:0.1.0"
    sandbox_cpu_seconds: int = 2
    #: Address space, not resident memory: this becomes `RLIMIT_AS`, which
    #: counts every reservation a process makes. numpy and pandas reserve far
    #: more than they ever touch, so 512 MB — a sane-looking number for "how
    #: much memory should a computation get" — killed the interpreter while it
    #: was still importing them. It only ever worked because the machines it
    #: was tried on were Windows, where `resource` does not exist and no limit
    #: was applied at all. This is a runaway guard, not a working budget.
    sandbox_memory_mb: int = 2048
    sandbox_wall_timeout_seconds: int = 30
    sandbox_network: Literal["none"] = "none"

    # --- Budgets: an unbounded agent loop is a financial denial of service ---
    run_max_usd: float = 5.0
    run_max_tokens: int = 2_000_000
    run_max_nodes: int = 40
    run_max_debate_rounds: int = 3
    workspace_monthly_usd_cap: float = 50.0

    # --- Uploads ---
    upload_max_bytes: int = 104_857_600
    workspace_storage_max_bytes: int = 2_147_483_648
    upload_allowed_mime: tuple[str, ...] = (
        "application/pdf",
        "text/csv",
        "text/tab-separated-values",
        "text/plain",
        "application/json",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.apache.parquet",
    )

    # --- Rate limits (docs/05-SECURITY.md §5) ---
    rate_limit_ip_per_min: int = 300
    rate_limit_user_per_min: int = 600
    rate_limit_login_per_15min: int = 5
    rate_limit_upload_per_hour: int = 20
    rate_limit_run_per_hour: int = 10
    rate_limit_chat_per_hour: int = 60

    # --- Observability ---
    otel_exporter_otlp_endpoint: str = ""

    @property
    def is_production(self) -> bool:
        return self.resx_env == "production"

    @property
    def storage_root(self) -> Path:
        """The upload and dataset directory, as an absolute path."""
        return resolve_path(self.storage_local_root)

    @property
    def dataset_dir(self) -> Path:
        return self.storage_root / "datasets"

    @property
    def sqlite_file(self) -> str:
        """The resolved SQLite path. One definition, used by every caller.

        Absolute, always. Two processes disagreeing about where the database
        lives is the single most confusing failure this project has produced:
        each one works, and together they appear to lose data.
        """
        if self.sqlite_path.strip():
            return str(resolve_path(self.sqlite_path.strip()))
        return str(self.storage_root / "resx.db")

    @property
    def resolved_store(self) -> Literal["mongo", "sqlite"]:
        """Which store backend will actually be used."""
        if self.store_backend != "auto":
            return self.store_backend
        return "mongo" if self.mongodb_url else "sqlite"

    @property
    def provider(self) -> Literal["groq", "anthropic"]:
        """The provider that will actually be used.

        Resolved here rather than at each call site so the CLI, the route
        guard and the client itself cannot disagree about which provider is
        live.
        """
        if self.llm_provider != "auto":
            return self.llm_provider
        if self.groq_api_key:
            return "groq"
        return "anthropic"

    @property
    def model_api_key(self) -> str:
        """The key for the resolved provider, empty if it is not configured."""
        return self.groq_api_key if self.provider == "groq" else self.anthropic_api_key

    @property
    def has_model_key(self) -> bool:
        return bool(self.model_api_key)

    @property
    def model_key_env_var(self) -> str:
        """The variable a user has to set — named so error messages can say it."""
        return "GROQ_API_KEY" if self.provider == "groq" else "ANTHROPIC_API_KEY"

    @property
    def cors_origins(self) -> list[str]:
        return [self.resx_base_url]

    @property
    def egress_allowlist(self) -> frozenset[str]:
        hosts = (h.strip().lower() for h in self.search_egress_allowlist.split(","))
        return frozenset(h for h in hosts if h)

    @field_validator(
        "anthropic_api_key",
        "groq_api_key",
        "tavily_api_key",
        "voyage_api_key",
        mode="before",
    )
    @classmethod
    def _blank_placeholder_keys(cls, v: object) -> object:
        """Treat an unfilled placeholder as absent, not as a key.

        `.env.example` ships values like `gsk_replace-me` so the file documents
        itself. Copied to `.env` and left alone, a truthiness check reads them
        as configured: `status` claims a key is present, the route guard lets
        the request through, and the failure finally surfaces as a 401 from the
        provider several minutes into a run. Blanking them here means the
        honest error — "no key configured" — arrives before any work is done.
        """
        if not isinstance(v, str):
            return v
        text = v.strip()
        lowered = text.lower()
        if "replace" in lowered or lowered in {"none", "null", "changeme", "change-me"}:
            return ""
        return text

    @field_validator("cookie_secure")
    @classmethod
    def _cookies_must_be_secure_in_production(cls, v: bool, info) -> bool:  # type: ignore[no-untyped-def]
        # The refresh cookie is a bearer credential with a week-long life. Sent
        # over plain HTTP it is readable by anything on the path, so shipping
        # `secure=false` to production is a session-hijacking vulnerability
        # rather than a configuration preference.
        if info.data.get("resx_env") == "production" and not v:
            raise ValueError("COOKIE_SECURE must be true in production")
        return v

    @field_validator("mongodb_url", "mongodb_readonly_url")
    @classmethod
    def _reject_unfilled_connection_string(cls, v: str) -> str:
        """An Atlas placeholder is not a credential.

        `mongodb+srv://<db_username>:<db_password>@...` reaches the driver as
        literal text and comes back as an auth failure, which points at the
        password rather than at the brackets. Named here instead.
        """
        text = v.strip()
        if not text:
            return text
        if "<" in text and ">" in text:
            raise ValueError(
                "the MongoDB connection string still contains a placeholder in "
                "angle brackets. Replace <db_username> and <db_password> with "
                "the database user from Atlas > Database Access (not your "
                "Atlas login), percent-encoding any of : / ? # [ ] @ that "
                "appear in the password."
            )
        if not text.startswith(("mongodb://", "mongodb+srv://")):
            raise ValueError(
                f"a MongoDB URL must start with mongodb:// or mongodb+srv://, "
                f"got {text.split(':', 1)[0]!r}"
            )
        return text

    @field_validator("mongodb_url")
    @classmethod
    def _production_needs_a_real_database(cls, v: str, info) -> str:  # type: ignore[no-untyped-def]
        # SQLite has no concurrent writer story and no replication. Falling back
        # to it in production would be a silent downgrade of durability.
        env = info.data.get("resx_env")
        backend = info.data.get("store_backend")
        if env == "production" and backend != "sqlite" and not v.strip():
            raise ValueError(
                "MONGODB_URL must be set in production (or STORE_BACKEND=sqlite "
                "set explicitly to accept the single-file backend)"
            )
        return v.strip()

    @field_validator("field_encryption_key")
    @classmethod
    def _reject_placeholder_key_in_production(cls, v: str, info) -> str:  # type: ignore[no-untyped-def]
        # A placeholder secret reaching production is a silent, total failure of
        # encryption at rest, so it is a boot-time error instead.
        env = info.data.get("resx_env")
        if env == "production" and v.startswith("replace-with"):
            raise ValueError("FIELD_ENCRYPTION_KEY must be set in production")
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
