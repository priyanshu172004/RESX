"""Store selection.

One place decides which backend is live, so the API, the CLI and the benchmark
harness cannot disagree about where the data is.

The fallback is deliberately *loud*. An `auto` configuration that cannot reach
MongoDB says so and continues on SQLite, because a developer with no server
running should still be able to work. An explicit `STORE_BACKEND=mongo` raises
instead: silently degrading a production deployment to a single file is worse
than refusing to boot.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from app.core.config import Settings
from app.store.local import LocalStore
from app.store.mongo import MongoStore

log = logging.getLogger("resx.store")

Store = LocalStore | MongoStore


@runtime_checkable
class SupportsIdentity(Protocol):
    """The identity surface both backends implement.

    Declared so a third backend cannot be added without it: auth would
    otherwise fail at request time rather than at construction.
    """

    def create_user(self, **kwargs: object) -> dict[str, Any]: ...
    def get_user_by_email(self, email: str) -> dict[str, Any] | None: ...
    def get_user(self, user_id: str) -> dict[str, Any] | None: ...
    def store_refresh_token(self, **kwargs: object) -> None: ...
    def get_refresh_token(self, token_hash: bytes) -> dict[str, Any] | None: ...
    def revoke_token_family(self, family_id: str) -> int: ...


class StoreUnavailableError(RuntimeError):
    pass


def _how_to_fix(url: str, exc: Exception) -> str:
    """The advice that actually applies to *this* URL.

    This said "Start it with `docker compose up -d mongo`" for every failure,
    including an Atlas cluster refusing the TLS handshake — which sent a reader
    to start a local container that had nothing to do with the problem. A
    cloud URL and a local one fail for entirely different reasons and the
    remedy is never the same one.

    The Atlas case is worth naming precisely. It rejects a connection from an
    unknown IP *during the TLS handshake*, before any credential is sent, so
    the error says "SSL handshake failed" and never mentions the access list —
    and a reader reasonably concludes their certificates or password are
    broken. Residential IP addresses rotate, so this happens to a working
    setup that nobody touched.
    """
    detail = str(exc).lower()
    if url.startswith("mongodb+srv://") or ".mongodb.net" in url:
        hint = (
            "This is a MongoDB Atlas URL, so nothing local will fix it. Check "
            "Atlas -> Network Access: a connection from an IP that is not on "
            "the access list is refused during the TLS handshake, before any "
            "password is sent, which is why the error mentions SSL rather than "
            "permissions. Add your current IP. Also check the cluster is not "
            "paused."
        )
        if "ssl" in detail or "tls" in detail or "handshake" in detail:
            return hint
        if "auth" in detail or "credential" in detail:
            return (
                "Atlas accepted the connection and rejected the credentials. "
                "Check the user and password in MONGODB_URL against Atlas -> "
                "Database Access, and percent-encode any of : / ? # [ ] @ in "
                "the password."
            )
        return hint
    return (
        "Start it with `docker compose up -d mongo`, or set "
        "STORE_BACKEND=sqlite to accept the single-file backend."
    )


def build_store(settings: Settings) -> Store:
    """Construct the configured store."""
    resolved = settings.resolved_store

    if resolved == "sqlite":
        return _sqlite(settings)

    url = settings.mongodb_url or "mongodb://localhost:27017"
    try:
        store = MongoStore(
            url,
            database=settings.mongodb_database,
            server_selection_timeout_ms=settings.mongodb_timeout_ms,
        )
        # `MongoClient` connects lazily, so constructing one proves nothing.
        # The ping is what turns "misconfigured" into an error here instead of
        # a timeout on the first upload.
        if not store.ping():
            raise StoreUnavailableError(f"no response from MongoDB at {_redact(url)}")
    except Exception as exc:
        if settings.store_backend == "mongo":
            raise StoreUnavailableError(
                f"STORE_BACKEND=mongo but MongoDB is unreachable at "
                f"{_redact(url)}: {exc}. {_how_to_fix(url, exc)}"
            ) from exc
        # `StoreUnavailableError` is raised above for an explicit mongo request; an
        # `auto` configuration falls through to SQLite below.
        log.warning(
            "MongoDB unreachable at %s (%s); falling back to SQLite at %s. "
            "Set STORE_BACKEND=mongo to make this an error instead.",
            _redact(url),
            exc,
            settings.sqlite_file,
        )
        return _sqlite(settings)

    # The database is logged separately rather than appended to the URL: an
    # Atlas string already carries one in its path, and appending produced
    # ".../ResXDB?appName=Cluster0/ResXDB".
    log.info("store: mongodb %s (database %s)", _redact(url), settings.mongodb_database)
    return store


def _sqlite(settings: Settings) -> LocalStore:
    path = Path(settings.sqlite_file)
    log.info("store: sqlite %s", path)
    return LocalStore(path)


def _redact(url: str) -> str:
    """Strip credentials before a connection string reaches a log line."""
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.rpartition("@")
    return f"{scheme}://***@{host}" if scheme else f"***@{host}"
