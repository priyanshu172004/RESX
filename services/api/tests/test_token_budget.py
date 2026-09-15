"""The client-side tokens-per-minute budget.

Why it exists: the graph runs the specialists as parallel branches, which is
the right shape for the graph and the wrong shape for a shared per-minute
quota. Three agents sending ~4,000 tokens each in the same instant is 12,000
against a limit of 8,000, so the provider 429s whichever arrives last. Every
branch then retries into the same exhausted window, the collisions are what
consume the window, and after the retries are spent one specialist contributes
nothing — the report quietly loses a whole section and the run still reports
success.

Observed on a real run: three specialists planned, two produced claims, the
third died on `Groq 429 ... rate limit reached` after four attempts.

So requests are admitted rather than repaired.
"""

from __future__ import annotations

import threading
import time

from app.agents.llm import TokenRateLimiter, token_limiter


def test_a_request_that_fits_is_not_delayed() -> None:
    limiter = TokenRateLimiter(8_000)
    assert limiter.acquire(4_000) == 0.0
    assert limiter.acquire(4_000) == 0.0


def test_a_request_that_would_exceed_the_window_waits() -> None:
    """The core behaviour. A short window keeps the test fast; the arithmetic
    is the same at sixty seconds."""
    limiter = TokenRateLimiter(1_000, window_seconds=0.4)
    limiter.acquire(800)
    started = time.monotonic()
    limiter.acquire(800)
    assert time.monotonic() - started >= 0.3


def test_the_window_rolls_rather_than_resetting() -> None:
    """Spend is forgotten as it ages out, so a steady sender is not
    progressively throttled."""
    limiter = TokenRateLimiter(1_000, window_seconds=0.3)
    limiter.acquire(1_000)
    time.sleep(0.35)
    assert limiter.acquire(1_000) == 0.0


def test_parallel_callers_do_not_both_see_the_same_free_space() -> None:
    """The whole point, and the reason the wait happens under the lock.

    If the lock were released while waiting, two branches would measure the
    same free space and both proceed — which is exactly the collision this
    replaces.
    """
    limiter = TokenRateLimiter(1_000, window_seconds=10.0)
    admitted: list[int] = []
    lock = threading.Lock()

    def send(size: int) -> None:
        # A short-lived thread that gives up rather than blocking the test.
        if limiter._in_window(time.monotonic()) + size > limiter.tokens_per_minute:
            return
        limiter.acquire(size)
        with lock:
            admitted.append(size)

    threads = [threading.Thread(target=send, args=(600,)) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    # 600 + 600 exceeds 1,000, so at most one may be admitted.
    assert sum(admitted) <= 1_000


def test_a_request_larger_than_the_whole_window_is_not_blocked_forever() -> None:
    """It cannot ever fit, so admitting it and letting the provider explain is
    better than hanging. The 413 path already reports that clearly."""
    limiter = TokenRateLimiter(1_000, window_seconds=60.0)
    started = time.monotonic()
    limiter.acquire(50_000)
    assert time.monotonic() - started < 1.0


def test_actual_usage_is_reconciled() -> None:
    """The estimate counts input characters only and the completion is charged
    against the same window, so without reconciling, the budget drifts
    optimistic and starts colliding a few calls in."""
    limiter = TokenRateLimiter(1_000, window_seconds=60.0)
    limiter.acquire(400)
    limiter.record_actual(400, 900)
    # 400 reserved + 500 correction = 900 spent, so only 100 is left.
    assert limiter._in_window(time.monotonic()) == 900


def test_reconciling_downwards_does_nothing() -> None:
    """An over-estimate is safe and must not be refunded into a credit that
    lets the next caller overshoot."""
    limiter = TokenRateLimiter(1_000, window_seconds=60.0)
    limiter.acquire(500)
    limiter.record_actual(500, 100)
    assert limiter._in_window(time.monotonic()) == 500


def test_the_budget_is_shared_across_clients() -> None:
    """The quota belongs to the API key, not to a client instance, and the
    graph builds more than one caller per run."""
    first = token_limiter(8_000)
    second = token_limiter(8_000)
    assert first is second


def test_changing_the_limit_replaces_the_budget() -> None:
    """So a settings change takes effect rather than being ignored for the
    life of the process."""
    first = token_limiter(8_000)
    second = token_limiter(30_000)
    assert first is not second
    assert second.tokens_per_minute == 30_000
    # Restore, so test order cannot affect anything else.
    token_limiter(8_000)


def test_the_client_paces_its_requests(monkeypatch) -> None:
    """End to end through `_post`: two large requests against a small budget,
    and the second has to wait."""
    from app.agents.llm import GroqLLM

    llm = GroqLLM(api_key="gsk_test", tokens_per_minute=1_000)
    # A fresh budget for this test rather than the process-wide one.
    llm._limiter = TokenRateLimiter(1_000, window_seconds=0.4)

    class _Response:
        status_code = 200
        headers: dict[str, str] = {}  # noqa: RUF012

        @staticmethod
        def json() -> dict[str, object]:
            return {
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

    monkeypatch.setattr(llm.client, "post", lambda *a, **k: _Response())

    payload = {"model": "m", "messages": [{"role": "user", "content": "x" * 2_600}]}
    llm._post(payload)
    started = time.monotonic()
    llm._post(payload)
    assert time.monotonic() - started >= 0.3
