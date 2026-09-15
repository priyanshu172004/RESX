"""Tracing, wired only when somewhere has been named to send it.

The OpenTelemetry packages were declared as dependencies and nothing ever
called them, so the observability story was a line in a manifest.

Two decisions shape this module:

**Off by default.** With no `OTEL_EXPORTER_OTLP_ENDPOINT` nothing is
instrumented at all — not a no-op exporter, not a console one. A developer
running the API locally should not pay for spans nobody reads, and a background
exporter retrying against an endpoint that does not exist is a log full of
connection errors that teaches people to ignore logs.

**Failure here never takes the API down.** Tracing is how you find out what
went wrong; it must not itself be what goes wrong. Every failure path logs and
returns, because an API that refuses to start because its telemetry collector
is unreachable has inverted the relationship between the two.

What is deliberately *not* traced: request bodies, headers and query strings.
A span carrying an uploaded financial table, or an `Authorization` header, has
moved the most sensitive data in the system into a third-party service that was
never assessed for it. Route templates and status codes are enough to find a
slow endpoint, and that is what tracing is for here.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("resx.telemetry")

#: Never recorded on a span. Each would move data into the collector that the
#: collector was never meant to hold.
EXCLUDED_URLS = "/health,/metrics"


def setup_tracing(app: Any, settings: Any) -> bool:
    """Instrument the app, returning whether tracing was actually enabled.

    The return value is what the startup log reports, so "tracing on" in the
    logs means a span reached an exporter rather than that a setting was set.
    """
    endpoint = str(getattr(settings, "otel_exporter_otlp_endpoint", "") or "").strip()
    if not endpoint:
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        # The OTLP HTTP exporter ships separately from the SDK, so this is the
        # ordinary case of "endpoint configured, package not installed" rather
        # than a broken environment. Named, so the fix is obvious.
        log.warning(
            "OTEL_EXPORTER_OTLP_ENDPOINT is set but tracing packages are "
            "missing (%s). Install opentelemetry-exporter-otlp-proto-http, or "
            "unset the endpoint to run without tracing.",
            exc,
        )
        return False

    try:
        provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": "resx-api",
                    "deployment.environment": str(getattr(settings, "resx_env", "development")),
                }
            )
        )
        # Batched rather than simple: a span exported inline on the request
        # path adds the collector's latency to every response, and a slow
        # collector would then look like a slow API.
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces"))
        )
        trace.set_tracer_provider(provider)

        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls=EXCLUDED_URLS,
            # Route template and status only. See the module docstring: bodies
            # and headers would move uploaded financials and bearer tokens into
            # a service that was never assessed to hold them.
            http_capture_headers_server_request=None,
            http_capture_headers_server_response=None,
        )
    except Exception as exc:
        # Tracing is how you find out what went wrong. It must never be the
        # thing that goes wrong.
        log.warning("tracing could not be enabled (%s); continuing without it", exc)
        return False

    log.info("tracing enabled, exporting to %s", endpoint)
    return True
