"""The model clients.

Two providers are supported behind one `LLM` protocol: Anthropic (Claude) and
Groq. The graph is written against the protocol, so switching provider is a
configuration change and not a code change.

`AnthropicLLM` wraps the Anthropic SDK with the four things every RESX agent
call needs:

  * **Structured output.** Agents return validated Pydantic models via
    `messages.parse(output_format=...)`, so a malformed response is a
    validation error rather than a parsing adventure.
  * **Prompt caching.** The shared-rules block is byte-identical across every
    specialist in a run, which makes it a perfect stable prefix. It is marked
    `cache_control: ephemeral` and the cache-read tokens are reported so a
    silent invalidation shows up as a cost, not a mystery.
  * **Adaptive thinking and effort.** `thinking={"type": "adaptive"}` with
    `output_config.effort`; the old fixed `budget_tokens` form is rejected by
    current models.
  * **Budget accounting.** Every call returns its real token counts and a
    dollar cost, because the graph decrements a budget and an unbounded agent
    loop is a financial denial of service.

`GroqLLM` reaches Groq's OpenAI-compatible endpoint over plain `httpx`, which
is already a dependency, rather than pulling in a second SDK. It is deliberately
*not* a pretence at parity — three capabilities genuinely do not exist there,
and each is handled explicitly rather than silently:

  * **No native Pydantic parsing.** JSON mode plus the schema in the prompt,
    then `model_validate_json`, then a bounded repair round that feeds the
    validation errors back. A malformed response is still a validation error at
    the end of it, never a half-parsed object.
  * **No prompt caching.** `cache_prefix` is accepted and ignored, and the
    reported cache figures stay zero. That is why the cost of a Groq run scales
    with the number of specialists rather than flattening after the first.
  * **Effort maps onto `reasoning_effort`.** The gpt-oss family does support
    it, and it matters more here than on Claude: the reasoning trace is charged
    against the same completion budget as the answer, so an unbounded effort on
    a large prompt spends the whole budget thinking and returns no JSON at all.

The accuracy contract is unchanged by the choice of provider: the model still
never does arithmetic, and a claim still cannot carry a value without a
`computation_id`. A weaker model produces *fewer* claims here, not looser ones.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from app.agents.schemas import AgentName

Effort = Literal["low", "medium", "high", "xhigh", "max"]
T = TypeVar("T", bound=BaseModel)

#: Per-million-token prices, used to convert usage into a dollar figure for the
#: run budget. Cache reads are an order of magnitude cheaper than fresh input,
#: which is why the shared-rules prefix is worth caching at all.
PRICING: dict[str, dict[str, float]] = {
    "claude-opus-5": {"input": 5.00, "output": 25.00, "cache_write": 6.25, "cache_read": 0.50},
    "claude-sonnet-5": {
        "input": 2.00,
        "output": 10.00,
        "cache_write": 2.50,
        "cache_read": 0.20,
    },
    "claude-haiku-4-5": {
        "input": 1.00,
        "output": 5.00,
        "cache_write": 1.25,
        "cache_read": 0.10,
    },
    # Groq. Priced even though the free tier bills nothing, because a run that
    # costs a literal zero would make the budget ceiling unenforceable and the
    # node cap the only thing standing between a bug and an infinite loop.
    #
    # Groq rotates its catalog: the Llama 3 ids that were the obvious defaults
    # here have since been withdrawn, and a withdrawn id is a 404 on every
    # call. `GroqLLM.available_models()` is what turns that into a legible
    # error, and these entries only have to cover what is priced today.
    "openai/gpt-oss-120b": {
        "input": 0.15,
        "output": 0.75,
        "cache_write": 0.15,
        "cache_read": 0.15,
    },
    "openai/gpt-oss-20b": {
        "input": 0.10,
        "output": 0.50,
        "cache_write": 0.10,
        "cache_read": 0.10,
    },
    "openai/gpt-oss-safeguard-20b": {
        "input": 0.10,
        "output": 0.50,
        "cache_write": 0.10,
        "cache_read": 0.10,
    },
    "qwen/qwen3.6-27b": {
        "input": 0.29,
        "output": 0.59,
        "cache_write": 0.29,
        "cache_read": 0.29,
    },
    "qwen/qwen3.8-27b": {
        "input": 0.29,
        "output": 0.59,
        "cache_write": 0.29,
        "cache_read": 0.29,
    },
    "groq/compound": {
        "input": 0.15,
        "output": 0.75,
        "cache_write": 0.15,
        "cache_read": 0.15,
    },
    "groq/compound-mini": {
        "input": 0.10,
        "output": 0.50,
        "cache_write": 0.10,
        "cache_read": 0.10,
    },
    # Retained so an existing .env pinned to one of these is still charged
    # something rather than running free against the budget.
    "llama-3.3-70b-versatile": {
        "input": 0.59,
        "output": 0.79,
        "cache_write": 0.59,
        "cache_read": 0.59,
    },
    "llama-3.1-8b-instant": {
        "input": 0.05,
        "output": 0.08,
        "cache_write": 0.05,
        "cache_read": 0.05,
    },
}

#: Per-model completion ceilings, from Groq's own `/models` response.
#: Requesting more than a model allows is a 400, so the caller's `max_tokens`
#: is clamped rather than forwarded — a shared default of 16k would otherwise
#: fail on every small model.
GROQ_MAX_COMPLETION: dict[str, int] = {
    "openai/gpt-oss-120b": 65_536,
    "openai/gpt-oss-20b": 65_536,
    "openai/gpt-oss-safeguard-20b": 65_536,
    "qwen/qwen3.6-27b": 16_384,
    "qwen/qwen3.8-27b": 16_384,
    "groq/compound": 8_192,
    "groq/compound-mini": 8_192,
    "llama-3.3-70b-versatile": 32_768,
    "llama-3.1-8b-instant": 8_192,
}
GROQ_DEFAULT_MAX_COMPLETION = 8_192

#: RESX's five effort levels mapped onto the three Groq accepts. Deliberately
#: conservative at the top: `high` reasoning on a large prompt is what emptied
#: the completion budget and produced no JSON.
GROQ_REASONING_EFFORT: dict[str, str] = {
    "low": "low",
    "medium": "low",
    "high": "medium",
    "xhigh": "high",
    "max": "high",
}

#: Model families that accept `reasoning_effort`. Sending it to a model that
#: does not is a 400, so it is opt-in by prefix rather than sent to everything.
GROQ_REASONING_MODELS = ("openai/gpt-oss", "qwen/qwen3")

#: Extra completion tokens allowed on top of the caller's `max_tokens`.
#: The caller is budgeting for the answer; the reasoning trace is invisible to
#: it and is charged against the same ceiling.
GROQ_REASONING_HEADROOM = 8_192

#: HTTP 400 codes that mean "the model produced the wrong shape", not "your
#: request was malformed". These are retried and repaired; every other 400 is
#: a bug in the request and is raised immediately.
GROQ_REPAIRABLE_CODES = frozenset({"json_validate_failed", "tool_use_failed"})

#: Characters per token, used to size a request before sending it. Deliberately
#: pessimistic: tables and URLs tokenize worse than prose, and an optimistic
#: estimate produces the 413 this is meant to prevent.
CHARS_PER_TOKEN = 3.2

#: The least evidence worth sending. Below this the request is all rules and
#: no facts, and the model answers from the prompt rather than the documents --
#: a confident answer with nothing behind it, which is worse than a refusal.
MIN_EVIDENCE_CHARS = 2_000

#: The least any one trimmable message keeps when the allowance is shared out.
#: Protects the short ones: a one-line task instruction beside a huge evidence
#: block gets a tiny proportional share, and truncating the instruction breaks
#: the request in a way truncating evidence does not.
MIN_MESSAGE_CHARS = 400

#: Models that cannot hold a conversation and must never be selected for an
#: agent: speech-to-text, text-to-speech, and the prompt-injection classifiers.
#: Filtered out of the "did you mean" list so a 404 does not suggest Whisper.
GROQ_NON_CHAT_PREFIXES = (
    "whisper",
    "canopylabs/",
    "meta-llama/llama-prompt-guard",
    "playai-tts",
)

#: Model routing from `docs/03-AGENTS.md` §10. The Critic runs on the strongest
#: model available on purpose: a reviewer weaker than the agent it reviews
#: produces theatre, not falsification.
REASONING_AGENTS: frozenset[AgentName] = frozenset(
    {AgentName.FINANCE, AgentName.RISK, AgentName.CRITIC, AgentName.SYNTHESIZER}
)


log = logging.getLogger("resx.llm")


class LLMError(RuntimeError):
    pass


class JSONGenerationFailedError(LLMError):
    """The provider rejected its own generation.

    Two Groq responses land here, both HTTP 400 and both *generation* failures
    rather than bad requests:

      * `json_validate_failed` — JSON mode produced something unparseable.
      * `tool_use_failed` — the model emitted a tool call instead of content.
        The gpt-oss family is trained with tool calling, so an agent prompt
        that asks it to write Python reliably provokes one.

    Either often succeeds on a retry, and both carry whatever the provider
    captured in `failed_generation` — which for a tool call is the arguments
    the model wanted to pass, and is frequently the answer itself.
    """

    def __init__(self, message: str, *, failed_generation: str = "", code: str = "") -> None:
        super().__init__(message)
        self.failed_generation = failed_generation
        self.code = code


class LLMRefusalError(LLMError):
    """The model declined the request on policy grounds."""


@dataclass(slots=True)
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    model: str = ""
    provider: str = "anthropic"

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_creation_tokens
        )

    @property
    def cost_usd(self) -> float:
        rates = PRICING.get(self.model)
        if rates is None:
            # An unknown model must not silently cost zero — that would let a
            # misconfigured model id run without touching the budget at all.
            # It must also not be priced 60x over: charging an unrecognised
            # Groq model at Opus rates would exhaust a run budget in two nodes
            # and look like a budget bug rather than a config typo.
            fallback = "llama-3.3-70b-versatile" if self.provider == "groq" else "claude-opus-5"
            rates = PRICING[fallback]
        return (
            self.input_tokens * rates["input"]
            + self.output_tokens * rates["output"]
            + self.cache_creation_tokens * rates["cache_write"]
            + self.cache_read_tokens * rates["cache_read"]
        ) / 1_000_000

    @property
    def cache_hit_rate(self) -> float:
        cached = self.cache_read_tokens
        fresh = self.input_tokens + self.cache_creation_tokens
        total = cached + fresh
        return 0.0 if total == 0 else cached / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "cache_hit_rate": round(self.cache_hit_rate, 4),
        }


@dataclass(slots=True)
class LLMResult:
    text: str = ""
    parsed: Any = None
    usage: LLMUsage = field(default_factory=LLMUsage)
    stop_reason: str | None = None
    refusal_category: str | None = None

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"


class LLM(Protocol):
    """The seam the graph depends on.

    Nodes are written against this rather than against the SDK, so the graph
    can be exercised deterministically in tests with an explicit, clearly
    labelled test double instead of being untestable without network access.
    """

    def structured(
        self,
        *,
        agent: AgentName,
        system: str,
        user: str,
        schema: type[T],
        cache_prefix: str | None = ...,
        effort: Effort = ...,
        max_tokens: int = ...,
    ) -> LLMResult: ...

    def text(
        self,
        *,
        agent: AgentName,
        system: str,
        user: str,
        cache_prefix: str | None = ...,
        effort: Effort = ...,
        max_tokens: int = ...,
    ) -> LLMResult: ...


class AnthropicLLM:
    """The Claude client."""

    provider = "anthropic"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        reasoning_model: str = "claude-opus-5",
        fast_model: str = "claude-sonnet-5",
        timeout: float = 600.0,
        max_retries: int = 2,
    ) -> None:
        import anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise LLMError(
                "no ANTHROPIC_API_KEY configured. The agents require a model; "
                "run the ingestion and retrieval stages, or the benchmark "
                "harness, if you want to work without one."
            )

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=key, timeout=timeout, max_retries=max_retries)
        self.reasoning_model = reasoning_model
        self.fast_model = fast_model

    def model_for(self, agent: AgentName) -> str:
        return self.reasoning_model if agent in REASONING_AGENTS else self.fast_model

    # -- request assembly -------------------------------------------------

    def _system_blocks(self, system: str, cache_prefix: str | None) -> Any:
        """Build the system field, caching the stable prefix.

        Order matters: the cached block must come first and be byte-identical
        across calls. Anything volatile after the breakpoint is free to change
        without invalidating the cache.
        """
        if not cache_prefix:
            return [{"type": "text", "text": system}]

        remainder = system[len(cache_prefix) :] if system.startswith(cache_prefix) else system
        blocks: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": cache_prefix,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        if remainder.strip():
            blocks.append({"type": "text", "text": remainder})
        return blocks

    @staticmethod
    def _usage(response: Any, model: str) -> LLMUsage:
        raw = getattr(response, "usage", None)
        return LLMUsage(
            input_tokens=int(getattr(raw, "input_tokens", 0) or 0),
            output_tokens=int(getattr(raw, "output_tokens", 0) or 0),
            cache_read_tokens=int(getattr(raw, "cache_read_input_tokens", 0) or 0),
            cache_creation_tokens=int(getattr(raw, "cache_creation_input_tokens", 0) or 0),
            model=model,
        )

    @staticmethod
    def _text_of(response: Any) -> str:
        parts: list[str] = []
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "\n".join(parts)

    def _check_refusal(self, response: Any, agent: AgentName) -> str | None:
        """A refusal arrives as HTTP 200 with `stop_reason == "refusal"`.

        Checked before reading content, because the content of a refused
        response is not an answer and treating it as one would put the model's
        decline text into a report as analysis.
        """
        if getattr(response, "stop_reason", None) != "refusal":
            return None
        details = getattr(response, "stop_details", None)
        return getattr(details, "category", None) or "unspecified"

    # -- calls ------------------------------------------------------------

    def structured(
        self,
        *,
        agent: AgentName,
        system: str,
        user: str,
        schema: type[T],
        cache_prefix: str | None = None,
        effort: Effort = "high",
        max_tokens: int = 16_000,
    ) -> LLMResult:
        model = self.model_for(agent)
        try:
            response = self.client.messages.parse(
                model=model,
                max_tokens=max_tokens,
                system=self._system_blocks(system, cache_prefix),
                messages=[{"role": "user", "content": user}],
                thinking={"type": "adaptive"},
                output_config={"effort": effort},
                output_format=schema,
            )
        except self._anthropic.NotFoundError as exc:
            raise LLMError(f"model {model!r} not available: {exc}") from exc
        except self._anthropic.RateLimitError as exc:
            raise LLMError(f"rate limited calling {model}: {exc}") from exc
        except self._anthropic.APIStatusError as exc:
            raise LLMError(f"API error {exc.status_code} calling {model}: {exc}") from exc
        except self._anthropic.APIConnectionError as exc:
            raise LLMError(f"could not reach the API: {exc}") from exc

        refusal = self._check_refusal(response, agent)
        usage = self._usage(response, model)
        if refusal:
            return LLMResult(
                text="",
                parsed=None,
                usage=usage,
                stop_reason="refusal",
                refusal_category=refusal,
            )

        return LLMResult(
            text=self._text_of(response),
            parsed=getattr(response, "parsed_output", None),
            usage=usage,
            stop_reason=getattr(response, "stop_reason", None),
        )

    def text(
        self,
        *,
        agent: AgentName,
        system: str,
        user: str,
        cache_prefix: str | None = None,
        effort: Effort = "high",
        max_tokens: int = 16_000,
    ) -> LLMResult:
        model = self.model_for(agent)
        try:
            # Streaming so a long answer cannot hit the request timeout; the
            # helper still hands back one complete message.
            with self.client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                system=self._system_blocks(system, cache_prefix),
                messages=[{"role": "user", "content": user}],
                thinking={"type": "adaptive"},
                output_config={"effort": effort},
            ) as stream:
                response = stream.get_final_message()
        except self._anthropic.APIStatusError as exc:
            raise LLMError(f"API error {exc.status_code} calling {model}: {exc}") from exc
        except self._anthropic.APIConnectionError as exc:
            raise LLMError(f"could not reach the API: {exc}") from exc

        refusal = self._check_refusal(response, agent)
        usage = self._usage(response, model)
        if refusal:
            return LLMResult(
                text="", usage=usage, stop_reason="refusal", refusal_category=refusal
            )
        return LLMResult(
            text=self._text_of(response),
            usage=usage,
            stop_reason=getattr(response, "stop_reason", None),
        )


# --------------------------------------------------------------------------- #
# Groq
# --------------------------------------------------------------------------- #

#: Appended to the system prompt in JSON mode. Groq's JSON mode guarantees
#: syntactically valid JSON and nothing at all about its shape, so the shape has
#: to be stated in the prompt -- and `model_validate_json` still has the final
#: say, because a prompt is a request and a validator is a guarantee.
JSON_INSTRUCTION = """
--- OUTPUT FORMAT (MANDATORY) ---
Reply with a single JSON object and nothing else. No prose before or after it,
no markdown code fences, no explanation. It must validate against this JSON
Schema exactly:

{schema}

Rules for the JSON:
  * Do NOT call a tool or a function. No tools exist on this request, and an
    attempted call is rejected outright. Anything you would have passed as a
    tool argument belongs in the JSON object instead -- including source code,
    which goes in as an ordinary JSON string.
  * Use the exact field names from the schema. Do not invent fields.
  * Every field listed in "required" must be present.
  * Where a field has an "enum", use one of those literal values verbatim.
  * Numbers must be bare JSON numbers, not strings, and must carry no currency
    symbols, thousands separators, or units.
  * If the evidence provided does not support a field, use the schema's null or
    empty value for it. Do not guess, and do not fill it with placeholder text.
"""

#: Groq states the wait in the message body, not the `Retry-After` header:
#: "Please try again in 10.6125s". Matched because guessing shorter than the
#: server's own figure guarantees the next attempt is throttled too.
_RETRY_HINT = re.compile(r"try again in\s+([0-9.]+)\s*(ms|s|m)\b", re.IGNORECASE)


def parse_retry_hint(body: str) -> float | None:
    """Seconds to wait, from Groq's own message. `None` if it did not say."""
    match = _RETRY_HINT.search(body or "")
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    unit = match.group(2).lower()
    if unit == "ms":
        return value / 1000.0
    if unit == "m":
        return value * 60.0
    return value


#: Matches a leading reasoning block. Some Groq models (the gpt-oss family)
#: emit one before the answer even in JSON mode.
class TokenRateLimiter:
    """A shared, rolling tokens-per-minute budget.

    This exists because of a structural conflict rather than a bug. LangGraph
    fans the specialists out as parallel branches, which is the right shape for
    the graph — they are independent and the run is much faster for it. But a
    tokens-per-minute quota is shared, and three agents each sending ~4,000
    tokens in the same instant is 12,000 against a limit of 8,000. The provider
    answers 429 to whichever arrives last, and after the retries are exhausted
    that agent contributes nothing: the report silently loses a whole
    specialist's section and the run still reports success.

    Retrying harder does not fix it. Every branch retries into the same
    exhausted window, so they collide again, and the collisions are what
    consume the window.

    So requests are admitted rather than repaired. A caller declares its size,
    waits until the last sixty seconds have room for it, and only then sends.
    That converts a run that loses agents into a run that takes longer, which
    is the trade a free tier is actually offering.

    Held as a module-level singleton keyed by nothing: the quota is per API
    key, and one process uses one key. Two processes against the same key will
    still collide — the retry path remains the backstop for that.
    """

    def __init__(self, tokens_per_minute: int, *, window_seconds: float = 60.0) -> None:
        self.tokens_per_minute = max(1, tokens_per_minute)
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        #: (spent_at, tokens) for the current window, oldest first.
        self._spent: deque[tuple[float, int]] = deque()

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._spent and self._spent[0][0] <= cutoff:
            self._spent.popleft()

    def _in_window(self, now: float) -> int:
        self._prune(now)
        return sum(tokens for _, tokens in self._spent)

    def acquire(self, tokens: int) -> float:
        """Block until `tokens` fit in the window, then record them.

        Returns how long it waited, for logging. The whole wait happens under
        the lock on purpose: releasing it would let a second caller measure the
        same free space and both would proceed, which is the collision this
        class exists to prevent.
        """
        # A request larger than the entire window can never fit. Admitting it
        # immediately is right: the provider will reject it and say so, and
        # blocking forever would be worse than a clear error.
        cost = min(max(1, tokens), self.tokens_per_minute)
        waited = 0.0

        with self._lock:
            while True:
                now = time.monotonic()
                used = self._in_window(now)
                if used + cost <= self.tokens_per_minute:
                    self._spent.append((now, cost))
                    return waited

                # Sleep exactly until the oldest entry leaves the window.
                # Polling on a fixed interval would either waste time or spin.
                oldest_at = self._spent[0][0]
                sleep_for = max(0.05, (oldest_at + self.window_seconds) - now)
                log.info(
                    "token budget full (%d/%d in the last %.0fs); waiting %.1fs",
                    used,
                    self.tokens_per_minute,
                    self.window_seconds,
                    sleep_for,
                )
                time.sleep(sleep_for)
                waited += sleep_for

    def record_actual(self, estimated: int, actual: int) -> None:
        """Correct the last reservation once the real usage is known.

        The estimate is made from characters before sending and cannot include
        the completion, so it is always low. Reconciling keeps the window
        honest instead of letting it drift optimistic and start colliding
        again.
        """
        if actual <= estimated:
            return
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            self._spent.append((now, min(actual - estimated, self.tokens_per_minute)))


#: One budget per process. The quota belongs to the API key, not to a client
#: instance, and the graph builds its client once per run.
_TOKEN_LIMITER: TokenRateLimiter | None = None
_LIMITER_LOCK = threading.Lock()


def token_limiter(tokens_per_minute: int) -> TokenRateLimiter:
    global _TOKEN_LIMITER
    with _LIMITER_LOCK:
        if _TOKEN_LIMITER is None or _TOKEN_LIMITER.tokens_per_minute != tokens_per_minute:
            _TOKEN_LIMITER = TokenRateLimiter(tokens_per_minute)
        return _TOKEN_LIMITER


_THINK_BLOCK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _groq_error(response: Any) -> dict[str, Any]:
    """Pull Groq's `{"error": {...}}` envelope out of a response body."""
    try:
        payload = response.json()
    except Exception:
        return {}
    error = payload.get("error") if isinstance(payload, dict) else None
    return error if isinstance(error, dict) else {}


def salvage_tool_call(failed_generation: str) -> str:
    """Recover the payload from a rejected tool call.

    Groq reports a rejected call as `{"name": "...", "arguments": {...}}`, and
    the arguments are usually exactly the object the schema wanted. Worth one
    parse attempt before spending another request: the model already did the
    work, it just addressed it to a tool that does not exist.

    The `arguments` field is *not* reliably valid JSON — a model emitting
    Python code there routinely produces an unquoted blob — so this only
    returns something when it actually parses.
    """
    candidate = extract_json_object(failed_generation)
    if not candidate:
        return ""
    try:
        parsed = json.loads(candidate)
    except ValueError:
        return ""
    if not isinstance(parsed, dict):
        return ""

    arguments = parsed.get("arguments")
    if isinstance(arguments, dict):
        return json.dumps(arguments)
    if isinstance(arguments, str):
        try:
            inner = json.loads(arguments)
        except ValueError:
            return ""
        if isinstance(inner, dict):
            return json.dumps(inner)
        return ""

    # Not a tool-call envelope at all: a plain object that happened to be
    # rejected. Hand it back for validation.
    if "name" not in parsed:
        return candidate
    return ""


def extract_json_object(text: str) -> str:
    """Recover the JSON object from a model response.

    JSON mode should make this a no-op. It exists because a reasoning preamble
    or a stray fence would otherwise turn a perfectly good answer into a parse
    failure, and the single repair round is better spent on real schema
    violations than on punctuation.
    """
    cleaned = _THINK_BLOCK.sub("", text or "").strip()
    cleaned = _FENCE.sub("", cleaned).strip()
    if not cleaned:
        return ""

    start = cleaned.find("{")
    if start == -1:
        return cleaned

    # Brace matching rather than a regex, because a regex cannot tell a brace
    # inside a quoted string from a structural one.
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : i + 1]

    # Unbalanced: hand back what we have, so the validation error names the real
    # problem (a truncated object) rather than an empty string.
    return cleaned[start:]


class GroqLLM:
    """The Groq client.

    Speaks the OpenAI-compatible chat-completions API over `httpx`, which is
    already in the dependency set. Retries are spaced, never immediate: the free
    tier limits by tokens per minute, so retrying a 429 straight away only
    consumes the next slot in the bucket.
    """

    provider = "groq"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        reasoning_model: str = "llama-3.3-70b-versatile",
        fast_model: str = "llama-3.1-8b-instant",
        base_url: str = "https://api.groq.com/openai/v1",
        timeout: float = 180.0,
        max_retries: int = 5,
        retry_backoff_seconds: float = 2.0,
        max_wait_seconds: float = 75.0,
        max_request_tokens: int = 6_000,
        tokens_per_minute: int = 8_000,
    ) -> None:
        import httpx

        key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise LLMError(
                "no GROQ_API_KEY configured. The agents require a model; run "
                "the ingestion and retrieval stages, or the benchmark harness, "
                "if you want to work without one. A free key comes from "
                "https://console.groq.com/keys"
            )

        self._httpx = httpx
        self.base_url = base_url.rstrip("/")
        self.reasoning_model = reasoning_model
        self.fast_model = fast_model
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        # Longer than the free tier's one-minute token window. Capping below it
        # means every retry lands inside the same exhausted window.
        self.max_wait_seconds = max(1.0, max_wait_seconds)
        # The input ceiling for one request. The free tier's 8,000/minute
        # covers request and response together over the window, so this is
        # held below it rather than at it -- see `_fit_request` for why the
        # completion budget is not subtracted from this number.
        self.max_request_tokens = max(1_000, max_request_tokens)
        # Shared across every client in this process, because the quota
        # belongs to the API key rather than to an instance.
        self._limiter = token_limiter(tokens_per_minute)
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        )

    def close(self) -> None:
        self.client.close()

    def model_for(self, agent: AgentName) -> str:
        return self.reasoning_model if agent in REASONING_AGENTS else self.fast_model

    # -- catalog ----------------------------------------------------------

    def available_models(self) -> list[dict[str, Any]]:
        """Chat models this key can reach, largest output ceiling first.

        Queried rather than hardcoded because Groq's catalog changes: the
        Llama 3 ids that were the obvious defaults for this project have since
        been withdrawn entirely.
        """
        try:
            response = self.client.get("/models")
            response.raise_for_status()
            entries = response.json().get("data", [])
        except Exception as exc:
            raise LLMError(f"could not list Groq models: {exc}") from exc

        chat = [
            entry
            for entry in entries
            if entry.get("active", True)
            and not str(entry.get("id", "")).startswith(GROQ_NON_CHAT_PREFIXES)
        ]
        return sorted(
            chat, key=lambda e: int(e.get("max_completion_tokens") or 0), reverse=True
        )

    def _suggestions(self) -> str:
        try:
            models = self.available_models()
        except LLMError:
            return "  (could not list models; see https://console.groq.com/docs/models)"
        if not models:
            return "  (this key has no active chat models)"
        return "\n".join(
            f"  {m['id']}  (max output {m.get('max_completion_tokens', '?')})"
            for m in models[:12]
        )

    def preflight(self) -> None:
        """Fail at startup if a configured model does not exist.

        Called before a run rather than discovered inside one: a 404 on the
        Manager node produces an empty report with a degraded branch, which
        looks like a corpus problem rather than a typo in `.env`.
        """
        available = {m["id"] for m in self.available_models()}
        missing = [
            model for model in (self.reasoning_model, self.fast_model) if model not in available
        ]
        if missing:
            raise LLMError(
                f"configured Groq model(s) not available: {', '.join(missing)}.\n"
                f"Set GROQ_MODEL_REASONING / GROQ_MODEL_FAST to one of:\n" + self._suggestions()
            )

    @staticmethod
    def _cap(model: str, max_tokens: int) -> int:
        """Clamp to the model's ceiling instead of forwarding the caller's ask.

        The graph's shared 16k default is above what the small models allow, and
        Groq answers an over-large `max_completion_tokens` with a 400.

        Headroom is added on top of the request for a reasoning model, because
        the reasoning trace is charged against this same ceiling and the caller
        only ever budgeted for the answer. Without it a long critique spends
        the whole allowance thinking and returns no JSON at all.
        """
        ceiling = GROQ_MAX_COMPLETION.get(model, GROQ_DEFAULT_MAX_COMPLETION)
        requested = max_tokens
        if model.startswith(GROQ_REASONING_MODELS):
            requested += GROQ_REASONING_HEADROOM
        return max(256, min(requested, ceiling))

    def _fit_request(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Trim the request so it fits the provider's per-request token limit.

        Groq screens a request *before* running the model and rejects an
        oversized one with HTTP 413, so a node that exceeds the limit spends
        its turn for nothing. This is the guard against that, and three things
        about the arithmetic are load-bearing.

        **The completion budget is not part of the total.** Verified against
        the live API: a 78-token prompt with `max_completion_tokens` of 60,000
        is accepted, so the pre-flight figure counts input tokens only. An
        earlier version subtracted a quarter of the completion budget from the
        ceiling, which left a 16k-token caller just 2,000 tokens for a system
        message that is itself over 2,300 -- an impossible budget that made the
        function give up and forward the request untrimmed.

        **System messages are never trimmed, and everything else shares one
        allowance.** The system message carries the rules and the output
        schema; removing part of either yields an unparseable answer rather
        than a smaller one. The remaining messages -- the task, the evidence,
        and on a repair round the model's own rejected output -- are trimmed
        together, proportionally, so the largest gives up the most. Trimming
        each one against the full allowance independently, as an earlier
        version did, let three messages be three times over budget.

        **A too-large system message is an error, not something to trim
        around.** If the fixed part alone fills the ceiling there is no
        allowance to distribute, and forwarding the request would produce the
        exact 413 this exists to prevent.
        """
        budget_chars = int(self.max_request_tokens * CHARS_PER_TOKEN)

        def size(message: dict[str, Any]) -> int:
            return len(str(message.get("content") or ""))

        total = sum(size(m) for m in messages)
        if total <= budget_chars:
            return messages

        fixed = sum(size(m) for m in messages if m.get("role") == "system")
        allowance = budget_chars - fixed
        if allowance < MIN_EVIDENCE_CHARS:
            raise LLMError(
                f"the fixed part of this request (system prompt and output "
                f"schema) is {fixed} chars, leaving {allowance} of the "
                f"{budget_chars}-char budget for the task and its evidence. "
                f"Raise GROQ_MAX_REQUEST_TOKENS (currently "
                f"{self.max_request_tokens}) if the tier's tokens-per-minute "
                f"limit allows it."
            )

        trimmable = [m for m in messages if m.get("role") != "system"]
        used = sum(size(m) for m in trimmable)
        log.info(
            "request is %d chars against a %d budget; trimming %d chars of "
            "evidence across %d message(s)",
            total,
            budget_chars,
            used - allowance,
            len(trimmable),
        )

        # Proportional shares, so a long evidence block absorbs the cut rather
        # than the one-line task instruction beside it.
        shares = {id(m): max(MIN_MESSAGE_CHARS, allowance * size(m) // used) for m in trimmable}
        # The floor can push the shares above the allowance; scale back the
        # messages that are above the floor until the total fits.
        overshoot = sum(shares.values()) - allowance
        if overshoot > 0:
            for message in sorted(trimmable, key=size, reverse=True):
                if overshoot <= 0:
                    break
                room = shares[id(message)] - MIN_MESSAGE_CHARS
                take = min(room, overshoot)
                shares[id(message)] -= take
                overshoot -= take

        out: list[dict[str, Any]] = []
        for message in messages:
            content = str(message.get("content") or "")
            if message.get("role") == "system":
                out.append(message)
                continue

            share = shares[id(message)]
            if len(content) <= share:
                out.append(message)
                continue

            # Keep the head (the task) and the tail (the most recent
            # evidence), and say so where the middle was -- a model that
            # cannot see it omitted evidence will report a confident finding
            # drawn from a partial view without flagging it.
            marker = (
                "\n\n[... {n} characters of evidence omitted to fit the "
                "model's request limit. Base your answer only on what "
                "remains, and say so if it is insufficient. ...]\n\n"
            )
            room = share - len(marker.format(n=len(content)))
            head = room // 2
            tail = room - head
            out.append(
                {
                    **message,
                    "content": (
                        content[:head]
                        + marker.format(n=len(content) - head - tail)
                        + content[-tail:]
                    ),
                }
            )
        return out

    @staticmethod
    def _reasoning_effort(model: str, effort: Effort) -> dict[str, Any]:
        """Groq's `reasoning_effort`, for the models that accept it.

        Sending it to a model that does not is a 400, so this returns an empty
        dict rather than a default value for anything else.
        """
        if not model.startswith(GROQ_REASONING_MODELS):
            return {}
        return {"reasoning_effort": GROQ_REASONING_EFFORT.get(effort, "low")}

    # -- transport --------------------------------------------------------

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """One chat completion, admitted by the token budget then retried.

        The budget comes first and the retries are the backstop, not the other
        way round. Retrying into an exhausted per-minute window is what loses a
        whole agent: parallel branches collide, the collisions consume the
        window, and after four attempts one specialist contributes nothing to
        the report. See `TokenRateLimiter`.
        """
        last: Exception | None = None

        # Estimated from the assembled request, so it is the input side only
        # and therefore low. `record_actual` reconciles it below.
        estimated = int(
            sum(len(str(m.get("content") or "")) for m in payload.get("messages") or [])
            / CHARS_PER_TOKEN
        )
        waited = self._limiter.acquire(estimated)
        if waited > 1.0:
            log.info(
                "waited %.1fs for the token budget before calling %s",
                waited,
                payload.get("model"),
            )

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.post("/chat/completions", json=payload)
            except self._httpx.TimeoutException as exc:
                last = exc
                if attempt < self.max_retries:
                    time.sleep(min(self.retry_backoff_seconds * (2**attempt), 60.0))
                continue
            except self._httpx.HTTPError as exc:
                raise LLMError(f"could not reach Groq: {exc}") from exc

            if response.status_code < 400:
                try:
                    body_json = dict(response.json())
                except ValueError as exc:
                    raise LLMError(f"Groq returned a non-JSON body: {exc}") from exc
                # Reconcile the reservation with what was actually charged.
                # The estimate counted input characters only, and the
                # completion is charged against the same per-minute window, so
                # without this the budget drifts optimistic and starts
                # colliding again a few calls in.
                spent = (body_json.get("usage") or {}).get("total_tokens")
                if isinstance(spent, int):
                    self._limiter.record_actual(estimated, spent)
                return body_json

            body = response.text[:600]

            if response.status_code == 400:
                detail = _groq_error(response)
                code = str(detail.get("code") or "")
                if code in GROQ_REPAIRABLE_CODES:
                    # Retried in place: the model failed to produce the
                    # requested shape, and a second attempt at temperature 0
                    # frequently succeeds. Raised only once the attempts are
                    # exhausted, carrying the partial output so the caller can
                    # try to salvage or repair it.
                    last = JSONGenerationFailedError(
                        f"Groq rejected its own generation ({code})",
                        failed_generation=str(detail.get("failed_generation") or ""),
                        code=code,
                    )
                    if attempt < self.max_retries:
                        time.sleep(min(self.retry_backoff_seconds * (attempt + 1), 10.0))
                        continue
                    raise last

            if response.status_code == 413:
                # Too large even after `_fit_request`. Reported with the
                # numbers, because the fix is a configuration change (a bigger
                # tier, or a lower GROQ_MAX_REQUEST_TOKENS) rather than a
                # retry that would fail identically.
                sent = sum(
                    len(str(m.get("content") or "")) for m in payload.get("messages") or []
                )
                raise LLMError(
                    f"request too large for {payload.get('model')}: {body}. "
                    f"The request carried {sent} chars (~"
                    f"{sent / CHARS_PER_TOKEN:.0f} tokens) against a "
                    f"GROQ_MAX_REQUEST_TOKENS of {self.max_request_tokens}. "
                    f"If those numbers disagree, the estimate is off; if they "
                    f"agree, lower GROQ_MAX_REQUEST_TOKENS or raise the "
                    f"tier's tokens-per-minute limit."
                )

            if response.status_code == 401:
                raise LLMError("Groq rejected the API key (401). Check GROQ_API_KEY.")
            if response.status_code == 404:
                # Groq withdraws model ids without notice, and a withdrawn id
                # fails on every single call. Listing what this key can
                # actually reach turns a dead run into a one-line fix.
                raise LLMError(
                    f"model {payload.get('model')!r} is not available on Groq "
                    f"(404). Models this API key can use right now:\n" + self._suggestions()
                )
            if response.status_code not in (408, 409, 429, 500, 502, 503, 504):
                raise LLMError(f"Groq API error {response.status_code}: {body}")

            last = LLMError(f"Groq {response.status_code}: {body}")

            # Honour whatever the server says, from either place it might say
            # it. It knows when the token bucket refills; the local backoff is
            # only guessing, and guessing short means the retry is throttled
            # too.
            wait = self.retry_backoff_seconds * (2**attempt)
            header = response.headers.get("retry-after")
            if header:
                with contextlib.suppress(ValueError):
                    wait = max(wait, float(header))
            hinted = parse_retry_hint(response.text)
            if hinted is not None:
                # Plus a margin: waiting exactly the stated interval lands on
                # the boundary and is throttled about half the time.
                wait = max(wait, hinted + 1.0)
            if response.status_code == 429 and header is None and hinted is None:
                # Rate-limited with no hint at all. The window is a minute, so
                # a four-second guess is not a wait, it is a second rejection.
                wait = max(wait, 20.0)

            if attempt < self.max_retries:
                log.info(
                    "groq %s on %s; waiting %.1fs before retry %d/%d",
                    response.status_code,
                    payload.get("model"),
                    wait,
                    attempt + 1,
                    self.max_retries,
                )
                time.sleep(min(wait, self.max_wait_seconds))

        raise LLMError(f"Groq call failed after {self.max_retries + 1} attempts: {last}")

    # -- response reading -------------------------------------------------

    @staticmethod
    def _usage(response: dict[str, Any], model: str) -> LLMUsage:
        raw = response.get("usage") or {}
        return LLMUsage(
            input_tokens=int(raw.get("prompt_tokens") or 0),
            output_tokens=int(raw.get("completion_tokens") or 0),
            # Groq has no prompt cache. Left at zero rather than faked, so the
            # cache-hit rate reported for a Groq run is honestly 0.0 and nobody
            # tunes a cache prefix that is not doing anything.
            cache_read_tokens=0,
            cache_creation_tokens=0,
            model=model,
            provider="groq",
        )

    @staticmethod
    def _first_choice(response: dict[str, Any]) -> tuple[str, str | None]:
        choices = response.get("choices") or []
        if not choices:
            raise LLMError("Groq returned no choices")
        choice = choices[0] or {}
        message = choice.get("message") or {}
        content = message.get("content")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            # Some gateways return content as a list of parts.
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return content, choice.get("finish_reason")

    @staticmethod
    def _add(a: LLMUsage, b: LLMUsage) -> LLMUsage:
        """Accumulate across the repair round, so a retry is charged for."""
        return LLMUsage(
            input_tokens=a.input_tokens + b.input_tokens,
            output_tokens=a.output_tokens + b.output_tokens,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            model=b.model or a.model,
            provider="groq",
        )

    # -- calls ------------------------------------------------------------

    def structured(
        self,
        *,
        agent: AgentName,
        system: str,
        user: str,
        schema: type[T],
        cache_prefix: str | None = None,
        effort: Effort = "high",
        max_tokens: int = 16_000,
    ) -> LLMResult:
        """A validated model instance, a raised error, or nothing -- never a guess.

        `cache_prefix` and `effort` are accepted and ignored: Groq has neither a
        prompt cache nor an effort control. They stay in the signature so the
        graph never has to know which provider it is talking to.
        """
        model = self.model_for(agent)
        # Compact, not indented. The schema is consumed by a model, not read
        # by a person, and `indent=2` spent 2,215 chars -- roughly 690 tokens,
        # a fifth of the whole request budget -- on whitespace every call.
        instruction = JSON_INSTRUCTION.format(
            schema=json.dumps(schema.model_json_schema(), separators=(",", ":"))
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": f"{system}\n{instruction}"},
            {"role": "user", "content": user},
        ]

        usage = LLMUsage(model=model, provider="groq")
        errors = ""

        # Exactly one repair round. Bounded on purpose: a model that cannot
        # satisfy the schema after being shown its own validation errors will
        # not satisfy it on the fifth attempt either, and every attempt is
        # charged against the run budget.
        for attempt in range(2):
            try:
                response = self._post(
                    {
                        "model": model,
                        "messages": self._fit_request(messages),
                        "max_completion_tokens": self._cap(model, max_tokens),
                        "temperature": 0,
                        "response_format": {"type": "json_object"},
                        "stream": False,
                        **self._reasoning_effort(model, effort),
                    }
                )
            except JSONGenerationFailedError as exc:
                # The provider itself judged the output unusable. Try to
                # salvage it before spending another request: a rejected tool
                # call usually carries the answer in its arguments.
                salvaged = salvage_tool_call(exc.failed_generation)
                if salvaged:
                    try:
                        parsed = schema.model_validate_json(salvaged)
                    except (ValidationError, ValueError):
                        pass
                    else:
                        return LLMResult(
                            text=salvaged,
                            parsed=parsed,
                            usage=usage,
                            stop_reason=f"salvaged:{exc.code}",
                        )

                errors = (
                    f"the provider rejected its own output ({exc.code or 'unknown'}): "
                    f"{exc.failed_generation[:1000] or '(nothing captured)'}"
                )
                if attempt == 1:
                    break
                messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was rejected: it was not a "
                            "single valid JSON object. Do NOT call a tool or "
                            "function. Reply with the JSON object only — no "
                            "preamble, no explanation, no code fences."
                        ),
                    },
                ]
                continue

            usage = self._add(usage, self._usage(response, model))
            raw, finish = self._first_choice(response)

            if finish == "length":
                # A truncated object cannot be repaired by re-asking: the limit
                # is structural, so say which knob moves it.
                raise LLMError(
                    f"{model} hit its completion ceiling before closing the JSON "
                    "object. Reduce the evidence passed to this node, or set "
                    "GROQ_MODEL_REASONING to a model with a larger output limit."
                )

            candidate = extract_json_object(raw)
            try:
                parsed = schema.model_validate_json(candidate)
            except (ValidationError, ValueError) as exc:
                errors = str(exc)[:2_000]
                if attempt == 1:
                    break
                messages = [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "That response failed validation. Fix it and reply "
                            "with the corrected JSON object only.\n\n"
                            f"Errors:\n{errors}"
                        ),
                    },
                ]
                continue

            return LLMResult(
                text=candidate,
                parsed=parsed,
                usage=usage,
                stop_reason=finish,
            )

        raise LLMError(
            f"{model} did not produce output matching {schema.__name__}, even after "
            f"a repair round. Last validation errors:\n{errors}"
        )

    def text(
        self,
        *,
        agent: AgentName,
        system: str,
        user: str,
        cache_prefix: str | None = None,
        effort: Effort = "high",
        max_tokens: int = 16_000,
    ) -> LLMResult:
        model = self.model_for(agent)
        response = self._post(
            {
                "model": model,
                "messages": self._fit_request(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ]
                ),
                "max_completion_tokens": self._cap(model, max_tokens),
                "temperature": 0,
                "stream": False,
                **self._reasoning_effort(model, effort),
            }
        )
        content, finish = self._first_choice(response)
        return LLMResult(
            text=_THINK_BLOCK.sub("", content).strip(),
            usage=self._usage(response, model),
            stop_reason=finish,
        )


def build_llm(settings: Any) -> LLM:
    """The single place a provider is chosen.

    Resolution itself lives on `Settings.provider`, so the route guard, the CLI
    status line and this factory cannot disagree about which provider is live --
    a disagreement there surfaces as a 503 on a system that is in fact
    configured correctly, which is a miserable thing to debug.
    """
    provider = getattr(settings, "provider", None) or "anthropic"

    if provider == "groq":
        return GroqLLM(
            api_key=getattr(settings, "groq_api_key", "") or None,
            reasoning_model=getattr(
                settings, "groq_model_reasoning", "llama-3.3-70b-versatile"
            ),
            fast_model=getattr(settings, "groq_model_fast", "llama-3.1-8b-instant"),
            base_url=getattr(settings, "groq_base_url", "https://api.groq.com/openai/v1"),
            max_retries=int(getattr(settings, "groq_max_retries", 3)),
            retry_backoff_seconds=float(getattr(settings, "groq_retry_backoff_seconds", 2.0)),
            max_wait_seconds=float(getattr(settings, "groq_max_wait_seconds", 75.0)),
            max_request_tokens=int(getattr(settings, "groq_max_request_tokens", 6_000)),
            tokens_per_minute=int(getattr(settings, "groq_tokens_per_minute", 8_000)),
        )

    return AnthropicLLM(
        api_key=getattr(settings, "anthropic_api_key", "") or None,
        reasoning_model=getattr(settings, "resx_model_reasoning", "claude-opus-5"),
        fast_model=getattr(settings, "resx_model_fast", "claude-sonnet-5"),
    )
