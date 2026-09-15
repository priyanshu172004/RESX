"""Tests for the Groq client.

Groq gives us syntactically valid JSON and no shape guarantee at all, so every
test here is about what happens when the shape is wrong. Nothing hits the
network: the transport is stubbed, because the point under test is our parsing,
clamping, accounting and repair logic rather than Groq's uptime.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel, Field

from app.agents.llm import (
    CHARS_PER_TOKEN,
    GROQ_DEFAULT_MAX_COMPLETION,
    GROQ_MAX_COMPLETION,
    GROQ_REPAIRABLE_CODES,
    JSON_INSTRUCTION,
    MIN_EVIDENCE_CHARS,
    PRICING,
    AnthropicLLM,
    GroqLLM,
    JSONGenerationFailedError,
    LLMError,
    LLMUsage,
    _groq_error,
    build_llm,
    extract_json_object,
    parse_retry_hint,
    salvage_tool_call,
)
from app.agents.schemas import AgentName
from app.core.config import Settings


class Shape(BaseModel):
    label: str
    count: int = Field(ge=0)


def _completion(content: str, *, finish: str = "stop", prompt: int = 100, out: int = 20):
    return {
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": out},
    }


def _client(**kwargs: Any) -> GroqLLM:
    return GroqLLM(api_key="gsk_test", **kwargs)


def _stub(llm: GroqLLM, responses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace the transport, recording every payload it was given."""
    sent: list[dict[str, Any]] = []
    queue = list(responses)

    def fake_post(payload: dict[str, Any]) -> dict[str, Any]:
        sent.append(payload)
        if not queue:
            raise AssertionError("more calls than stubbed responses")
        return queue.pop(0)

    llm._post = fake_post  # type: ignore[method-assign]
    return sent


# --------------------------------------------------------------------------- #
# JSON recovery
# --------------------------------------------------------------------------- #


def test_extracts_a_bare_object() -> None:
    assert extract_json_object('{"a": 1}') == '{"a": 1}'


def test_strips_markdown_fences() -> None:
    assert extract_json_object('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strips_a_reasoning_preamble() -> None:
    raw = '<think>The user wants a count.</think>\n{"a": 1}'
    assert extract_json_object(raw) == '{"a": 1}'


def test_ignores_prose_around_the_object() -> None:
    raw = 'Sure, here you go:\n{"a": 1}\nHope that helps.'
    assert extract_json_object(raw) == '{"a": 1}'


def test_braces_inside_strings_do_not_end_the_object() -> None:
    """A regex would stop at the brace inside the quoted value; matching does not."""
    raw = '{"note": "a } inside a string", "n": 1}'
    assert extract_json_object(raw) == raw


def test_escaped_quote_does_not_end_the_string() -> None:
    raw = '{"note": "he said \\"} \\" once", "n": 1}'
    assert extract_json_object(raw) == raw


def test_a_truncated_object_is_returned_for_the_validator_to_reject() -> None:
    """Better a named parse failure than a silently empty string."""
    assert extract_json_object('{"a": 1, "b":') == '{"a": 1, "b":'


def test_empty_input_stays_empty() -> None:
    assert extract_json_object("") == ""
    assert extract_json_object("   \n ") == ""


# --------------------------------------------------------------------------- #
# Structured output
# --------------------------------------------------------------------------- #


def test_structured_returns_a_validated_model() -> None:
    llm = _client()
    _stub(llm, [_completion('{"label": "revenue", "count": 3}')])

    result = llm.structured(agent=AgentName.FINANCE, system="s", user="u", schema=Shape)

    assert isinstance(result.parsed, Shape)
    assert result.parsed.label == "revenue"
    assert result.parsed.count == 3


def test_the_schema_is_put_in_the_prompt_and_json_mode_is_requested() -> None:
    """Both halves are needed: JSON mode gives syntax, the prompt gives shape."""
    llm = _client()
    sent = _stub(llm, [_completion('{"label": "x", "count": 0}')])

    llm.structured(agent=AgentName.FINANCE, system="RULES", user="u", schema=Shape)

    payload = sent[0]
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["temperature"] == 0
    system = payload["messages"][0]["content"]
    assert system.startswith("RULES")
    assert '"count"' in system, "the schema itself must reach the model"


def test_a_schema_violation_is_repaired_once() -> None:
    llm = _client()
    sent = _stub(
        llm,
        [
            _completion('{"label": "x", "count": -5}'),  # violates ge=0
            _completion('{"label": "x", "count": 5}'),
        ],
    )

    result = llm.structured(agent=AgentName.FINANCE, system="s", user="u", schema=Shape)

    assert result.parsed is not None
    assert result.parsed.count == 5
    assert len(sent) == 2, "exactly one repair round"
    repair = sent[1]["messages"][-1]["content"]
    assert "failed validation" in repair
    assert "greater than or equal to 0" in repair, "the real errors are fed back"


def test_the_repair_round_is_bounded() -> None:
    """Two bad answers is an error, not a third attempt."""
    llm = _client()
    sent = _stub(
        llm,
        [
            _completion('{"label": "x"}'),
            _completion('{"label": "x"}'),
        ],
    )

    with pytest.raises(LLMError, match="did not produce output matching Shape"):
        llm.structured(agent=AgentName.FINANCE, system="s", user="u", schema=Shape)

    assert len(sent) == 2


def test_unparseable_output_raises_rather_than_returning_a_guess() -> None:
    llm = _client()
    _stub(llm, [_completion("I would rather not."), _completion("Still no.")])

    with pytest.raises(LLMError):
        llm.structured(agent=AgentName.FINANCE, system="s", user="u", schema=Shape)


def test_the_repair_round_is_charged_to_the_budget() -> None:
    """An unbilled retry would make the run budget a lie."""
    llm = _client()
    _stub(
        llm,
        [
            _completion('{"label": "x", "count": -1}', prompt=100, out=20),
            _completion('{"label": "x", "count": 1}', prompt=140, out=25),
        ],
    )

    result = llm.structured(agent=AgentName.FINANCE, system="s", user="u", schema=Shape)

    assert result.usage.input_tokens == 240
    assert result.usage.output_tokens == 45
    assert result.usage.cost_usd > 0


def test_truncation_raises_instead_of_being_repaired() -> None:
    """Re-asking cannot fix a token ceiling, so the message names the knob."""
    llm = _client()
    sent = _stub(llm, [_completion('{"label": "x", "count":', finish="length")])

    with pytest.raises(LLMError, match="GROQ_MODEL_REASONING"):
        llm.structured(agent=AgentName.FINANCE, system="s", user="u", schema=Shape)

    assert len(sent) == 1, "no repair attempt on a structural limit"


def test_no_choices_is_an_error() -> None:
    llm = _client()
    _stub(llm, [{"choices": [], "usage": {}}])

    with pytest.raises(LLMError, match="no choices"):
        llm.structured(agent=AgentName.FINANCE, system="s", user="u", schema=Shape)


# --------------------------------------------------------------------------- #
# Model routing and limits
# --------------------------------------------------------------------------- #


def test_reasoning_agents_get_the_stronger_model() -> None:
    llm = _client(reasoning_model="strong", fast_model="fast")
    assert llm.model_for(AgentName.FINANCE) == "strong"
    assert llm.model_for(AgentName.CRITIC) == "strong"
    assert llm.model_for(AgentName.SYNTHESIZER) == "strong"
    assert llm.model_for(AgentName.NEWS) == "fast"
    assert llm.model_for(AgentName.MANAGER) == "fast"


def test_max_tokens_is_clamped_to_the_model_ceiling() -> None:
    """The graph's shared 16k default exceeds what the small models allow."""
    llm = _client()
    assert llm._cap("llama-3.1-8b-instant", 16_000) == 8_192
    assert llm._cap("llama-3.3-70b-versatile", 16_000) == 16_000
    assert llm._cap("some-unknown-model", 999_999) == GROQ_DEFAULT_MAX_COMPLETION
    assert llm._cap("llama-3.1-8b-instant", 10) == 256


def test_the_clamp_is_applied_to_the_request() -> None:
    llm = _client(fast_model="llama-3.1-8b-instant")
    sent = _stub(llm, [_completion('{"label": "x", "count": 0}')])

    llm.structured(agent=AgentName.NEWS, system="s", user="u", schema=Shape, max_tokens=16_000)

    assert sent[0]["max_completion_tokens"] == 8_192


def test_every_default_model_has_a_price_and_a_ceiling() -> None:
    """An unpriced model would make the run budget unenforceable."""
    for model in ("llama-3.3-70b-versatile", "llama-3.1-8b-instant"):
        assert model in PRICING
        assert model in GROQ_MAX_COMPLETION


def test_an_unknown_groq_model_is_not_priced_at_claude_rates() -> None:
    """A 60x overcharge would look like a budget bug, not a config typo."""
    groq = LLMUsage(input_tokens=1_000_000, model="typo", provider="groq")
    claude = LLMUsage(input_tokens=1_000_000, model="typo", provider="anthropic")
    assert groq.cost_usd < claude.cost_usd


# --------------------------------------------------------------------------- #
# Honesty about what Groq does not have
# --------------------------------------------------------------------------- #


def test_cache_figures_stay_zero_rather_than_being_faked() -> None:
    """Groq has no prompt cache; reporting a hit rate would be a fiction."""
    llm = _client()
    _stub(llm, [_completion('{"label": "x", "count": 0}')])

    result = llm.structured(
        agent=AgentName.FINANCE,
        system="s",
        user="u",
        schema=Shape,
        cache_prefix="s",
    )

    assert result.usage.cache_read_tokens == 0
    assert result.usage.cache_creation_tokens == 0
    assert result.usage.cache_hit_rate == 0.0
    assert result.usage.to_dict()["provider"] == "groq"


def test_a_missing_key_fails_at_construction_with_the_url_to_get_one() -> None:
    with pytest.raises(LLMError, match=r"console\.groq\.com"):
        GroqLLM(api_key=None, base_url="http://127.0.0.1:1")


# --------------------------------------------------------------------------- #
# Provider selection
# --------------------------------------------------------------------------- #


def test_a_groq_key_alone_is_enough_to_boot() -> None:
    settings = Settings(_env_file=None, groq_api_key="gsk_test")
    llm = build_llm(settings)
    assert isinstance(llm, GroqLLM)
    assert llm.reasoning_model == settings.groq_model_reasoning
    llm.close()


def test_an_explicit_provider_wins_over_the_key_that_is_present() -> None:
    """Silently using a provider the operator did not choose is worse than failing."""
    settings = Settings(_env_file=None, llm_provider="anthropic", groq_api_key="gsk_test")
    assert settings.provider == "anthropic"
    assert settings.has_model_key is False
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        build_llm(settings)


def test_anthropic_is_still_selectable() -> None:
    settings = Settings(
        _env_file=None, llm_provider="anthropic", anthropic_api_key="sk-ant-test"
    )
    assert isinstance(build_llm(settings), AnthropicLLM)


def test_the_guard_names_the_variable_for_the_resolved_provider() -> None:
    """A Groq deployment must not be told to go and set an Anthropic key."""
    assert Settings(_env_file=None, llm_provider="groq").model_key_env_var == ("GROQ_API_KEY")
    assert Settings(_env_file=None, llm_provider="anthropic").model_key_env_var == (
        "ANTHROPIC_API_KEY"
    )


# --------------------------------------------------------------------------- #
# Key hygiene
# --------------------------------------------------------------------------- #


def test_an_unfilled_placeholder_is_not_a_key() -> None:
    """Otherwise `status` claims a key is set and the run dies on a 401."""
    settings = Settings(
        _env_file=None,
        groq_api_key="gsk_replace-me",
        anthropic_api_key="sk-ant-replace-me",
    )
    assert settings.provider == "anthropic", "a placeholder must not select a provider"
    assert settings.has_model_key is False


def test_a_real_key_survives_the_placeholder_check() -> None:
    settings = Settings(_env_file=None, groq_api_key="  gsk_liveKey123  ")
    assert settings.groq_api_key == "gsk_liveKey123", "whitespace trimmed, value kept"
    assert settings.has_model_key is True


def test_placeholder_blanking_covers_every_provider_key() -> None:
    settings = Settings(
        _env_file=None,
        tavily_api_key="replace-me",
        voyage_api_key="replace-me",
    )
    assert settings.tavily_api_key == ""
    assert settings.voyage_api_key == ""


def test_the_embedder_settings_the_pipeline_reads_actually_exist() -> None:
    """`extra="ignore"` silently drops undeclared fields.

    `build_embedder` reads both of these off settings, so leaving them
    undeclared meant VOYAGE_API_KEY could never arrive and semantic embeddings
    could never switch on, however the .env was filled in.
    """
    settings = Settings(_env_file=None, voyage_api_key="pa-live")
    assert settings.voyage_api_key == "pa-live"
    assert settings.embedding_provider == "voyage"


# --------------------------------------------------------------------------- #
# Reasoning budget
#
# Found by a live run: the Critic failed with `json_validate_failed` and an
# empty generation because the reasoning trace consumed the whole completion
# budget. All three of these guard that specific failure.
# --------------------------------------------------------------------------- #


def test_reasoning_models_get_headroom_above_the_request() -> None:
    """The caller budgets for the answer; reasoning is charged to the same cap."""
    llm = _client()
    assert llm._cap("openai/gpt-oss-120b", 16_000) > 16_000
    assert llm._cap("openai/gpt-oss-20b", 16_000) > 16_000


def test_headroom_never_exceeds_the_model_ceiling() -> None:
    llm = _client()
    assert llm._cap("openai/gpt-oss-120b", 60_000) <= GROQ_MAX_COMPLETION["openai/gpt-oss-120b"]


def test_non_reasoning_models_get_no_headroom() -> None:
    """`groq/compound` allows 8k and would 400 on anything larger."""
    llm = _client()
    assert llm._cap("groq/compound", 16_000) == 8_192


def test_effort_maps_onto_reasoning_effort_only_where_supported() -> None:
    # Sending `reasoning_effort` to a model that does not accept it is a 400,
    # so it must be absent rather than defaulted.
    llm = _client()
    assert llm._reasoning_effort("openai/gpt-oss-120b", "high") == {
        "reasoning_effort": "medium"
    }
    assert llm._reasoning_effort("openai/gpt-oss-120b", "max") == {"reasoning_effort": "high"}
    assert llm._reasoning_effort("groq/compound", "high") == {}
    assert llm._reasoning_effort("llama-3.3-70b-versatile", "high") == {}


def test_the_reasoning_effort_reaches_the_request() -> None:
    llm = _client(reasoning_model="openai/gpt-oss-120b")
    sent = _stub(llm, [_completion('{"label": "x", "count": 0}')])

    llm.structured(agent=AgentName.FINANCE, system="s", user="u", schema=Shape, effort="max")

    assert sent[0]["reasoning_effort"] == "high"


# --------------------------------------------------------------------------- #
# A rejected generation is repairable, not fatal
# --------------------------------------------------------------------------- #


def test_a_rejected_generation_gets_a_repair_round() -> None:
    """Groq's own 400 is a generation failure, not a bad request."""
    llm = _client()
    calls: list[dict[str, Any]] = []
    queue: list[Any] = [
        JSONGenerationFailedError("rejected", failed_generation='{"label": "x"'),
        _completion('{"label": "x", "count": 2}'),
    ]

    def fake_post(payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(payload)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    llm._post = fake_post  # type: ignore[method-assign]

    result = llm.structured(agent=AgentName.FINANCE, system="s", user="u", schema=Shape)

    assert result.parsed is not None
    assert result.parsed.count == 2
    assert len(calls) == 2, "the rejection must be retried, not raised"


def test_two_rejections_in_a_row_is_an_error() -> None:
    llm = _client()

    def always_reject(payload: dict[str, Any]) -> dict[str, Any]:
        raise JSONGenerationFailedError("rejected", failed_generation="")

    llm._post = always_reject  # type: ignore[method-assign]

    with pytest.raises(LLMError, match="did not produce output matching Shape"):
        llm.structured(agent=AgentName.FINANCE, system="s", user="u", schema=Shape)


def test_the_error_reader_survives_a_non_json_body() -> None:
    class NotJson:
        def json(self):
            raise ValueError("no")

    assert _groq_error(NotJson()) == {}

    class Wrapped:
        def json(self):
            return {"error": {"code": "json_validate_failed", "failed_generation": "x"}}

    assert _groq_error(Wrapped())["code"] == "json_validate_failed"


# --------------------------------------------------------------------------- #
# A rejected tool call
#
# Found by a live run: the gpt-oss family is trained with tool calling, so the
# Finance agent's "write Python" prompt provokes a call to a `run_python` tool
# that does not exist on the request. Groq answers 400 `tool_use_failed`.
# --------------------------------------------------------------------------- #


def test_a_rejected_tool_call_is_salvaged_without_another_request() -> None:
    """The arguments the model wanted to pass are usually the answer."""
    llm = _client()
    calls: list[dict[str, Any]] = []

    def fake_post(payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(payload)
        raise JSONGenerationFailedError(
            "rejected",
            code="tool_use_failed",
            failed_generation=json.dumps(
                {"name": "run_python", "arguments": {"label": "revenue", "count": 7}}
            ),
        )

    llm._post = fake_post  # type: ignore[method-assign]

    result = llm.structured(agent=AgentName.FINANCE, system="s", user="u", schema=Shape)

    assert result.parsed is not None
    assert result.parsed.count == 7
    assert result.stop_reason == "salvaged:tool_use_failed"
    assert len(calls) == 1, "salvage must not cost a second request"


def test_salvage_handles_arguments_as_a_json_string() -> None:
    assert json.loads(
        salvage_tool_call(json.dumps({"name": "f", "arguments": json.dumps({"a": 1})}))
    ) == {"a": 1}


def test_salvage_refuses_unparseable_arguments() -> None:
    """A model writing Python into `arguments` produces an unquoted blob."""
    assert salvage_tool_call('{"name": "run_python", "arguments": import pandas') == ""


def test_salvage_passes_through_a_plain_object() -> None:
    # Not a tool-call envelope: a plain object that happened to be rejected.
    assert salvage_tool_call('{"label": "z", "count": 0}') == '{"label": "z", "count": 0}'


def test_salvage_returns_nothing_for_empty_input() -> None:
    assert salvage_tool_call("") == ""
    assert salvage_tool_call("not json at all") == ""


def test_an_unsalvageable_tool_call_falls_through_to_the_repair_round() -> None:
    llm = _client()
    queue: list[Any] = [
        JSONGenerationFailedError(
            "rejected",
            code="tool_use_failed",
            failed_generation='{"name": "run_python", "arguments": import pandas',
        ),
        _completion('{"label": "x", "count": 4}'),
    ]
    calls: list[dict[str, Any]] = []

    def fake_post(payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(payload)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    llm._post = fake_post  # type: ignore[method-assign]

    result = llm.structured(agent=AgentName.FINANCE, system="s", user="u", schema=Shape)
    assert result.parsed is not None
    assert result.parsed.count == 4
    assert len(calls) == 2
    assert "Do NOT call a tool" in calls[1]["messages"][-1]["content"]


def test_the_instruction_forbids_tool_calls() -> None:
    """Stated in the prompt, not just handled after the fact."""
    llm = _client()
    sent = _stub(llm, [_completion('{"label": "x", "count": 0}')])
    llm.structured(agent=AgentName.FINANCE, system="s", user="u", schema=Shape)
    assert "Do NOT call a tool" in sent[0]["messages"][0]["content"]


def test_only_generation_failures_are_retried() -> None:
    """A genuinely malformed request must fail fast, not be retried."""
    assert "json_validate_failed" in GROQ_REPAIRABLE_CODES
    assert "tool_use_failed" in GROQ_REPAIRABLE_CODES
    assert "invalid_api_key" not in GROQ_REPAIRABLE_CODES
    assert "model_not_found" not in GROQ_REPAIRABLE_CODES


# --------------------------------------------------------------------------- #
# Rate limits
#
# Groq's free tier is 8,000 tokens per minute and a Critic call over a dozen
# claims is most of that, so a 429 mid-run is the normal case. The 429 states
# the required wait in its *message body*, not the Retry-After header, and an
# exponential backoff that caps below it retries inside the same exhausted
# window and then gives up.
# --------------------------------------------------------------------------- #


def test_the_retry_hint_is_parsed_from_the_message_body() -> None:
    assert parse_retry_hint("Please try again in 10.6125s. Need more?") == 10.6125


def test_the_retry_hint_handles_every_unit_groq_uses() -> None:
    assert parse_retry_hint("try again in 512ms") == pytest.approx(0.512)
    assert parse_retry_hint("try again in 1.5m") == pytest.approx(90.0)
    assert parse_retry_hint("try again in 3s") == pytest.approx(3.0)


def test_no_hint_returns_none_rather_than_zero() -> None:
    """Zero would read as 'retry immediately', which is the opposite."""
    assert parse_retry_hint("rate limited") is None
    assert parse_retry_hint("") is None
    assert parse_retry_hint("try again in soon") is None


def test_the_wait_ceiling_outlasts_a_one_minute_window() -> None:
    llm = _client()
    assert llm.max_wait_seconds > 60.0, (
        "a ceiling below the token window means every retry lands inside the "
        "same exhausted minute"
    )


# --------------------------------------------------------------------------- #
# Request fitting
#
# Every test here is a regression. The first version of `_fit_request` shipped
# with a guard that disabled trimming whenever the system message was larger
# than the (mis-computed) budget -- which was always -- so the trimming never
# ran once in production, and the 413 it was written to prevent recurred on
# every call, identically, because a retry re-sent the same bytes.
# --------------------------------------------------------------------------- #


def test_a_request_over_the_ceiling_is_actually_trimmed() -> None:
    llm = _client(max_request_tokens=6_000)
    messages = [
        {"role": "system", "content": "S" * 9_600},
        {"role": "user", "content": "U" * 60_000},
    ]
    fitted = llm._fit_request(messages)
    total = sum(len(str(m["content"])) for m in fitted)
    assert total <= int(6_000 * CHARS_PER_TOKEN)


def test_a_large_system_message_does_not_disable_trimming() -> None:
    """The exact production failure.

    A system message of ~3,000 tokens against a budget the old arithmetic
    computed as 2,000 produced a negative allowance, and the function returned
    the request untouched. It must trim instead.
    """
    llm = _client(max_request_tokens=6_000)
    # The measured size of the market agent's phase-B system message.
    system = "S" * 9_619
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "U" * 30_000},
    ]
    fitted = llm._fit_request(messages)
    assert len(str(fitted[1]["content"])) < 30_000, "the user message was not trimmed"
    assert sum(len(str(m["content"])) for m in fitted) <= int(6_000 * CHARS_PER_TOKEN)


def test_the_system_message_is_never_trimmed() -> None:
    """It carries the rules and the output schema; a partial one is unparseable."""
    llm = _client(max_request_tokens=6_000)
    system = "S" * 16_000  # most of the budget, but not all of it
    fitted = llm._fit_request(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": "U" * 50_000},
        ]
    )
    assert fitted[0]["content"] == system
    assert len(str(fitted[1]["content"])) < 50_000


def test_several_messages_share_one_allowance() -> None:
    """Each was previously trimmed against the full allowance independently, so
    a repair round with three messages could be three times over budget."""
    llm = _client(max_request_tokens=6_000)
    fitted = llm._fit_request(
        [
            {"role": "system", "content": "S" * 4_000},
            {"role": "user", "content": "A" * 30_000},
            {"role": "assistant", "content": "B" * 30_000},
            {"role": "user", "content": "C" * 30_000},
        ]
    )
    total = sum(len(str(m["content"])) for m in fitted)
    assert total <= int(6_000 * CHARS_PER_TOKEN)


def test_the_models_own_rejected_output_is_trimmed_too() -> None:
    """The repair path appends the raw generation, which can be thousands of
    tokens. Only user messages were trimmed before."""
    llm = _client(max_request_tokens=3_000)
    fitted = llm._fit_request(
        [
            {"role": "system", "content": "S" * 2_000},
            {"role": "user", "content": "U" * 2_000},
            {"role": "assistant", "content": "R" * 40_000},
        ]
    )
    assert len(str(fitted[2]["content"])) < 40_000


def test_a_short_instruction_survives_a_huge_evidence_block() -> None:
    """Proportional shares alone would give a one-line instruction almost
    nothing, and truncating the instruction breaks the request in a way
    truncating evidence does not."""
    llm = _client(max_request_tokens=6_000)
    instruction = "Reply with the corrected JSON object only."
    fitted = llm._fit_request(
        [
            {"role": "system", "content": "S" * 4_000},
            {"role": "user", "content": "U" * 80_000},
            {"role": "user", "content": instruction},
        ]
    )
    assert fitted[2]["content"] == instruction


def test_a_request_under_the_ceiling_is_passed_through_untouched() -> None:
    llm = _client(max_request_tokens=6_000)
    messages = [
        {"role": "system", "content": "S" * 1_000},
        {"role": "user", "content": "U" * 1_000},
    ]
    assert llm._fit_request(messages) == messages


def test_trimming_leaves_a_marker_where_the_evidence_was() -> None:
    """A model that cannot see that evidence was dropped reports a confident
    finding from a partial view without flagging it."""
    llm = _client(max_request_tokens=6_000)
    fitted = llm._fit_request(
        [
            {"role": "system", "content": "S" * 4_000},
            {"role": "user", "content": "U" * 60_000},
        ]
    )
    assert "characters of evidence omitted" in str(fitted[1]["content"])


def test_a_system_message_that_fills_the_budget_is_an_error() -> None:
    """There is no allowance left to distribute, and forwarding the request
    would produce the 413 this function exists to prevent."""
    llm = _client(max_request_tokens=1_000)
    with pytest.raises(LLMError, match="fixed part of this request"):
        llm._fit_request(
            [
                {"role": "system", "content": "S" * 4_000},
                {"role": "user", "content": "U" * 4_000},
            ]
        )


def test_the_completion_budget_is_not_charged_against_the_request() -> None:
    """Verified against the live API: a 78-token prompt with a 60,000-token
    completion cap is accepted, so the pre-flight check counts input only.
    Subtracting the completion budget is what made the budget impossible.
    """
    llm = _client(max_request_tokens=6_000)
    fitted = llm._fit_request(
        [
            {"role": "system", "content": "S" * 9_619},
            {"role": "user", "content": "U" * 30_000},
        ]
    )
    # The whole ceiling is available to the input, so what survives is the
    # ceiling minus the system message, not minus a completion reserve too.
    kept = len(str(fitted[1]["content"]))
    assert kept > 8_000, f"only {kept} chars of evidence survived"


def test_the_real_specialist_prompt_fits_with_room_for_evidence() -> None:
    """The end-to-end arithmetic against the actual prompts and schema rather
    than placeholder strings -- the numbers that produced the 413."""
    from app.agents import prompts
    from app.graph.nodes import SpecialistOutput

    llm = _client(max_request_tokens=6_000)
    system = prompts.system_prompt(AgentName.MARKET, research=True) + JSON_INSTRUCTION.format(
        schema=json.dumps(SpecialistOutput.model_json_schema(), separators=(",", ":"))
    )
    fitted = llm._fit_request(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": "evidence. " * 6_000},
        ]
    )
    total = sum(len(str(m["content"])) for m in fitted)
    assert total <= int(6_000 * CHARS_PER_TOKEN)
    assert len(str(fitted[1]["content"])) >= MIN_EVIDENCE_CHARS


def test_every_agent_prompt_leaves_room_for_evidence() -> None:
    """The failure was agent-specific: it appeared on market and news first
    because their prompts are longest. Every agent must clear the floor."""
    from app.agents import prompts
    from app.graph.nodes import ComputePlan, SpecialistOutput

    budget = int(6_000 * CHARS_PER_TOKEN)
    for agent in AgentName:
        for schema in (ComputePlan, SpecialistOutput):
            system = prompts.system_prompt(agent, research=True) + JSON_INSTRUCTION.format(
                schema=json.dumps(schema.model_json_schema(), separators=(",", ":"))
            )
            room = budget - len(system)
            assert room >= MIN_EVIDENCE_CHARS, (
                f"{agent.value} with {schema.__name__} leaves only {room} chars for evidence"
            )


def test_the_schema_is_sent_compactly() -> None:
    """`indent=2` spent ~690 tokens of a 6,000-token budget on whitespace."""
    llm = _client()
    sent = _stub(llm, [_completion(json.dumps({"label": "a", "count": 1}))])
    llm.structured(agent=AgentName.MARKET, system="sys", user="usr", schema=Shape)
    system = sent[0]["messages"][0]["content"]
    assert '"label"' in system
    assert '\n    "' not in system, "the schema is being pretty-printed"


def test_the_retrieval_budget_fits_the_request_ceiling() -> None:
    """The three budgets have to agree, and nothing else makes them agree.

    `retrieval_context_chars` was effectively 24,000 while the request ceiling
    was 19,200: retrieval alone was over the limit, so every specialist died
    with HTTP 413 before the model ran. The numbers live in different modules
    -- the setting in `config`, the estimator in `llm`, the system prompt in
    `prompts` -- and nothing tied them together, so they drifted. This is the
    tie.

    Checked against the pairs that actually occur. An earlier version took the
    cross product of every agent and every schema and failed on a specialist
    prompt plus `ExecutiveReport`, a request no code path ever builds.
    """
    from app.agents import prompts
    from app.agents.schemas import ExecutiveReport
    from app.core.config import Settings
    from app.graph.nodes import ComputePlan, SpecialistOutput

    settings = Settings(groq_api_key="gsk_test")
    ceiling = int(settings.groq_max_request_tokens * CHARS_PER_TOKEN)

    def system_size(agent: AgentName, schema: type, research: bool) -> int:
        return len(prompts.system_prompt(agent, research=research)) + len(
            JSON_INSTRUCTION.format(
                schema=json.dumps(schema.model_json_schema(), separators=(",", ":"))
            )
        )

    # Specialists retrieve, so their request carries the retrieval budget too.
    retrieving = [a for a in AgentName if a not in (AgentName.MANAGER, AgentName.SYNTHESIZER)]
    worst = max(
        system_size(agent, schema, research)
        for agent in retrieving
        for schema in (ComputePlan, SpecialistOutput)
        for research in (True, False)
    )
    used = worst + settings.retrieval_context_chars
    assert used <= ceiling, (
        f"the largest specialist system message ({worst} chars) plus the "
        f"retrieval budget ({settings.retrieval_context_chars}) is {used} "
        f"chars, over the {ceiling}-char request ceiling. Lower "
        f"retrieval_context_chars or raise groq_max_request_tokens."
    )
    # And enough left for the sub-question, the computation results and the
    # phase instructions, which sit in the same request.
    assert ceiling - used >= 2_000, f"only {ceiling - used} chars of slack"

    # The Synthesizer retrieves nothing: its user message is the claim block.
    # Adding `recommendations` to ExecutiveReport grew this schema, so the
    # room left for claims is worth asserting rather than assuming.
    for research in (True, False):
        size = system_size(AgentName.SYNTHESIZER, ExecutiveReport, research)
        room = ceiling - size
        assert room >= 4_000, (
            f"the synthesizer's system message is {size} chars in "
            f"research={research} mode, leaving only {room} for the claims it "
            f"has to summarise"
        )


def test_retrieval_never_cuts_an_untrusted_wrapper_in_half() -> None:
    """Half a delimiter leaves the data/instruction boundary ambiguous, which
    is the one thing the wrapper exists to make unambiguous."""
    from app.rag.retrieve import RetrievalResult

    class _Chunk:
        def __init__(self, i: int, size: int) -> None:
            self.chunk_id, self.doc_id, self.page = f"c{i}", "d1", 1
            self.para_idx, self.kind, self.text = i, "paragraph", "x" * size

    result = RetrievalResult(chunks=[_Chunk(0, 20_000), _Chunk(1, 500)])  # type: ignore[arg-type]
    block = result.context_block(max_chars=7_000)

    assert block.count("<untrusted_document_content") == block.count(
        "</untrusted_document_content>"
    ), "an untrusted wrapper was left unclosed"
    # The oversized chunk is skipped, not truncated, and the smaller one that
    # follows it still arrives -- the old code stopped at the first overflow.
    assert "x" * 20_000 not in block
    assert "x" * 500 in block
    assert "did not fit this request" in block
