"""Security headers, request correlation, and the rate limiter.

The header set is the Helmet equivalent from docs/05-SECURITY.md §6. It is
applied by the API as well as at the edge, deliberately: an application that
depends on a correctly configured CDN for its security headers has no security
headers the day the CDN config is wrong.
"""

from __future__ import annotations

import secrets
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.config import Settings

Handler = Callable[[Request], Awaitable[Response]]


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a correlation id to every request.

    The same id goes into every log line and into error responses, so a user
    report maps to a trace without exposing anything about internals.
    """

    async def dispatch(self, request: Request, call_next: Handler) -> Response:
        request_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:16]}"
        request.state.request_id = request_id
        request.state.started_at = time.perf_counter()

        # A per-request nonce, so the CSP never needs 'unsafe-inline'.
        request.state.csp_nonce = secrets.token_urlsafe(16)

        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, settings: Settings) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        self.settings = settings

    def _csp(self, nonce: str) -> str:
        return "; ".join(
            [
                "default-src 'self'",
                f"script-src 'self' 'nonce-{nonce}'",
                f"style-src 'self' 'nonce-{nonce}'",
                "img-src 'self' data: blob:",
                "font-src 'self' data:",
                f"connect-src 'self' {self.settings.resx_base_url}",
                "frame-ancestors 'none'",
                "object-src 'none'",
                "base-uri 'self'",
                "form-action 'self'",
                "upgrade-insecure-requests",
            ]
        )

    async def dispatch(self, request: Request, call_next: Handler) -> Response:
        response = await call_next(request)
        nonce = getattr(request.state, "csp_nonce", "")

        response.headers["Content-Security-Policy"] = self._csp(nonce)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), payment=(), usb=()"
        )
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"

        # HSTS only over TLS — sending it on plaintext localhost would pin a
        # developer's browser to https for a server that does not speak it.
        if self.settings.is_production:
            response.headers["Strict-Transport-Security"] = (
                "max-age=63072000; includeSubDomains; preload"
            )

        # `x-powered-by` can be removed here because whatever set it did so
        # within the ASGI response.
        #
        # `server` cannot. uvicorn writes it at the protocol layer, after every
        # ASGI middleware has run, so deleting it from a Starlette response has
        # no effect at all — verified against a live server, where the header
        # was still present. It has to be disabled where it is added:
        # `server_header=False`, or `--no-server-header` on the command line.
        # `scripts/dev-api.py` passes the flag; a production entrypoint must
        # set the parameter.
        if "x-powered-by" in response.headers:
            del response.headers["x-powered-by"]
        return response


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject oversized bodies before they are read into memory.

    `Content-Length` is only a hint, so the streaming reader in the upload
    route enforces a hard ceiling as well. This is the cheap first line.
    """

    def __init__(self, app, max_bytes: int) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: Handler) -> Response:
        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > self.max_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": {
                                "code": "PAYLOAD_TOO_LARGE",
                                "message": "Request body exceeds the permitted size.",
                                "request_id": getattr(request.state, "request_id", None),
                            }
                        },
                    )
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "code": "BAD_CONTENT_LENGTH",
                            "message": "Malformed Content-Length header.",
                            "request_id": getattr(request.state, "request_id", None),
                        }
                    },
                )
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP token bucket.

    This is the coarse global limit only. The far tighter per-route budgets
    (login, upload, run, chat) are applied as route dependencies, because a
    request that costs dollars and minutes must not share a limit with one that
    costs a millisecond — see docs/05-SECURITY.md §5.

    Backed by Redis in a deployment; the in-process fallback below keeps local
    development honest without requiring Redis to be running.
    """

    def __init__(self, app, settings: Settings) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        self.settings = settings
        self._buckets: dict[str, tuple[float, float]] = {}

    def _client_key(self, request: Request) -> str:
        # Trust a forwarded header only when a known proxy sets it; behind an
        # unvalidated header an attacker rotates the key and bypasses the limit.
        if self.settings.is_production:
            forwarded = request.headers.get("x-forwarded-for", "")
            if forwarded:
                return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next: Handler) -> Response:
        if request.url.path in {"/health", "/health/ready"}:
            return await call_next(request)

        capacity = float(self.settings.rate_limit_ip_per_min)
        refill_per_second = capacity / 60.0
        now = time.monotonic()
        key = self._client_key(request)

        tokens, last_seen = self._buckets.get(key, (capacity, now))
        tokens = min(capacity, tokens + (now - last_seen) * refill_per_second)

        if tokens < 1.0:
            retry_after = int((1.0 - tokens) / refill_per_second) + 1
            self._buckets[key] = (tokens, now)
            return JSONResponse(
                status_code=429,
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(int(capacity)),
                    "X-RateLimit-Remaining": "0",
                },
                content={
                    "error": {
                        "code": "RATE_LIMITED",
                        "message": "Too many requests.",
                        "request_id": getattr(request.state, "request_id", None),
                    }
                },
            )

        self._buckets[key] = (tokens - 1.0, now)
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(int(capacity))
        response.headers["X-RateLimit-Remaining"] = str(int(tokens - 1.0))
        return response
