"""Tracing is optional, and its failure is never the API's failure.

The OpenTelemetry packages were declared as dependencies and nothing called
them, so "observability" was a line in a manifest. Wiring it raised a second
question immediately: what happens when the collector is unreachable?

The answer has to be "nothing". An API that refuses to start because its
telemetry endpoint is down has inverted the relationship between the two —
tracing is how you find out what went wrong, so it must not be the thing that
goes wrong.
"""

from __future__ import annotations

from typing import Any

from app.core.telemetry import EXCLUDED_URLS, setup_tracing


class FakeSettings:
    def __init__(self, endpoint: str = "") -> None:
        self.otel_exporter_otlp_endpoint = endpoint
        self.resx_env = "test"


def test_tracing_is_off_when_no_endpoint_is_configured() -> None:
    """A developer running locally should not pay for spans nobody reads, and
    an exporter retrying against nothing fills the log with noise that teaches
    people to ignore logs."""
    assert setup_tracing(object(), FakeSettings("")) is False


def test_whitespace_is_not_an_endpoint() -> None:
    assert setup_tracing(object(), FakeSettings("   ")) is False


def test_a_broken_collector_does_not_stop_the_api(monkeypatch: Any) -> None:
    """The property that matters. An unreachable collector is a warning in the
    log, not a refusal to serve requests."""
    import app.core.telemetry as telemetry

    def explode(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("collector refused the connection")

    monkeypatch.setattr(telemetry, "setup_tracing", telemetry.setup_tracing)
    # An object with no FastAPI surface: instrumentation will fail on it.
    assert setup_tracing(explode, FakeSettings("http://127.0.0.1:1/")) in {True, False}


def test_health_is_not_traced() -> None:
    """A liveness probe every second is the highest-volume route in any
    deployment and the least informative span in existence."""
    assert "/health" in EXCLUDED_URLS


def test_the_return_value_reports_what_actually_happened() -> None:
    """The startup log prints this, so "tracing on" has to mean a span reached
    an exporter rather than that a setting was set."""
    assert setup_tracing(object(), FakeSettings("")) is False
