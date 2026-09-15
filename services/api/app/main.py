"""RESX API entrypoint.

Middleware order matters and is not arbitrary. Starlette applies middleware in
reverse registration order, so the last one added is the outermost. The order
below gives, from outside in:

    RequestContext  ->  SecurityHeaders  ->  RateLimit  ->  BodySizeLimit

Request context is outermost so a correlation id and CSP nonce exist even for a
request that a later layer rejects. Security headers sit above the rate limiter
so that a 429 still carries the full header set. Body-size is innermost of the
four so it runs only for requests that already passed the limiter.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

from app.api import auth, documents, health, runs, workspace
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.middleware import (
    BodySizeLimitMiddleware,
    RateLimitMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)
from app.core.telemetry import setup_tracing


def _embedding_label(settings: Any) -> str:
    """What the startup line reports about embeddings.

    It used to read `"voyage" if voyage_api_key else "hashing(offline)"`, which
    was true when Voyage was the only real provider and became a lie the moment
    Gemini was added: a correctly configured Gemini setup was reported as the
    offline hashing fallback, so the one line that tells you semantic retrieval
    is working said it was not.
    """
    provider = str(getattr(settings, "embedding_provider", "") or "")
    key = str(getattr(settings, f"{provider}_api_key", "") or "")
    if provider == "hashing":
        return "hashing(offline)"
    if provider and key:
        model = str(getattr(settings, "embedding_model", "") or "default")
        return f"{provider}({model})"
    return f"hashing(offline, no {provider.upper() or 'EMBEDDING'}_API_KEY)"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Touch settings on boot so a misconfiguration fails here rather than on
    # the first request that happens to need the bad value.
    settings = get_settings()

    # Build the store on boot too. It creates indexes and installs the claims
    # validator, and doing that here means an unreachable database is a startup
    # failure rather than a 500 on the first upload.
    from app.api.deps import get_store

    store = get_store()
    logging.getLogger("resx").info(
        "ready: store=%s%s provider=%s model=%s key=%s embeddings=%s search=%s",
        store.backend,
        f"/{settings.mongodb_database}" if store.backend == "mongo" else "",
        settings.provider,
        settings.groq_model_reasoning
        if settings.provider == "groq"
        else settings.resx_model_reasoning,
        settings.has_model_key,
        _embedding_label(settings),
        bool(settings.tavily_api_key),
    )
    # Tracing last, and only when an endpoint is configured. A failure here
    # logs and continues: tracing is how you find out what went wrong, so it
    # must never be the thing that goes wrong.
    setup_tracing(app, settings)

    try:
        yield
    finally:
        store.close()


def create_app() -> FastAPI:
    settings = get_settings()
    # Before anything else logs. Without this the `resx` logger has no handler
    # and every application message -- including which store is live -- is
    # discarded, because uvicorn only configures its own loggers.
    configure_logging(settings.resx_log_level)

    app = FastAPI(
        title="RESX API",
        version="0.1.0",
        summary="Autonomous business research and data analytics agent.",
        lifespan=lifespan,
        # No interactive docs in production: the schema is a map of the attack
        # surface, and it is not needed by the browser client.
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None,
        openapi_url=None if settings.is_production else "/openapi.json",
    )

    # Innermost first — see the module docstring for why this order.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.upload_max_bytes)
    app.add_middleware(RateLimitMiddleware, settings=settings)
    app.add_middleware(SecurityHeadersMiddleware, settings=settings)
    app.add_middleware(RequestContextMiddleware)

    # An explicit origin list, never a wildcard: credentials travel on cookies,
    # and `allow_origins=["*"]` with credentials is both invalid and unsafe.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=[
            "Content-Type",
            "Authorization",
            "X-Request-Id",
            "X-CSRF-Token",
            "X-Resx-Workspace",
            "Last-Event-ID",
        ],
        max_age=600,
    )

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(documents.router)
    app.include_router(runs.router)
    app.include_router(workspace.router)

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Report which field failed and how, but never echo the submitted value
        # back — that is how a validation error becomes a reflection primitive.
        details = [
            {"field": ".".join(str(p) for p in err.get("loc", ())), "issue": err.get("msg", "")}
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_FAILED",
                    "message": "Request validation failed.",
                    "details": details,
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": "HTTP_ERROR",
                    "message": str(exc.detail),
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    @app.exception_handler(Exception)
    async def on_unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Generic message plus a correlation id. A stack trace must never reach
        # a client; the trace lives in the logs against this id.
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred.",
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, Any]:
        return {"service": "resx-api", "docs": "/docs", "health": "/health"}

    return app


app = create_app()
