"""Shared API dependencies.

Two things live here and both are deliberate.

**Process-wide singletons.** The store, embedder and sandbox are expensive to
build and safe to share, so they are constructed once. `lru_cache` gives that
without a global.

**The workspace.** Every route takes its workspace from a verified JWT claim,
and the store refuses any query without one. There is exactly one resolver, so
tenancy cannot be forgotten on a route: a route that wants data must declare
`WorkspaceDep`, and the only thing that can produce one is a validated token.

A development header is still accepted, but only when
`RESX_ALLOW_DEV_WORKSPACE_HEADER` is explicitly on *and* the environment is not
production. It is a convenience for the CLI and the benchmark harness, not an
authentication mechanism, and it is off by default so it cannot be left on by
accident.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Annotated, Any

import jwt
from fastapi import Depends, Header, HTTPException, Request, status

from app.analysis.sandbox import Sandbox, build_sandbox
from app.core import security
from app.core.config import Settings, get_settings
from app.rag.embeddings import Embedder, build_embedder
from app.services.auth import AuthService, role_allows
from app.services.runs import RunService
from app.store.factory import Store, build_store

#: The workspace used by the development header and by the CLI.
DEV_WORKSPACE = "ws_dev"

#: The pseudo-user the development header resolves to. Marked so that anything
#: reading `role` cannot mistake it for a real account.
DEV_USER: dict[str, Any] = {
    "user_id": "usr_dev",
    "email": "dev@localhost",
    "name": "Local Developer",
    "workspace_id": DEV_WORKSPACE,
    "role": "owner",
    "verified": True,
    "is_dev_principal": True,
}


@lru_cache
def get_store() -> Store:
    return build_store(get_settings())


@lru_cache
def get_embedder() -> Embedder:
    return build_embedder(get_settings())


@lru_cache
def get_sandbox() -> Sandbox:
    return build_sandbox(get_settings())


@lru_cache
def get_run_service() -> RunService:
    return RunService(
        store=get_store(),
        settings=get_settings(),
        embedder=get_embedder(),
        sandbox=get_sandbox(),
    )


def get_auth_service() -> AuthService:
    return AuthService(store=get_store(), settings=get_settings())


def client_ip(request: Request) -> str | None:
    """The caller's address, for the audit log.

    `X-Forwarded-For` is trusted only when a proxy count is configured, because
    the header is client-supplied and an unconditional read lets anyone forge
    the address that lands in the audit trail.
    """
    settings = get_settings()
    if settings.trusted_proxy_hops > 0:
        forwarded = request.headers.get("x-forwarded-for", "")
        parts = [p.strip() for p in forwarded.split(",") if p.strip()]
        if len(parts) >= settings.trusted_proxy_hops:
            return parts[-settings.trusted_proxy_hops]
    return request.client.host if request.client else None


StoreDep = Annotated[Store, Depends(get_store)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
SandboxDep = Annotated[Sandbox, Depends(get_sandbox)]
RunServiceDep = Annotated[RunService, Depends(get_run_service)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]


# --------------------------------------------------------------------------- #
# The session
# --------------------------------------------------------------------------- #


def _bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def get_current_user(
    request: Request,
    store: StoreDep,
    settings: SettingsDep,
    authorization: Annotated[str | None, Header()] = None,
    x_resx_workspace: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    """Resolve the caller from a verified access token.

    The token is the only source of identity and of the workspace. A
    `workspace_id` taken from the request body or a header would be a
    cross-tenant read with extra steps, which is why the claim is read from the
    signature-verified payload and the header below cannot override it.
    """
    token = _bearer(authorization)

    if token is None:
        # No credential. The development header is the only other way in, and
        # only when explicitly enabled outside production.
        if (
            settings.allow_dev_workspace_header
            and not settings.is_production
            and x_resx_workspace is not None
        ):
            workspace = x_resx_workspace.strip() or DEV_WORKSPACE
            return {**DEV_USER, "workspace_id": workspace}
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="sign in to continue",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        claims = security.decode_access_token(token, settings)
    except jwt.ExpiredSignatureError as exc:
        # Distinguished from a bad token so the client knows to refresh rather
        # than to send the user back to the login screen.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="access token expired",
            headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
        ) from exc
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid access token",
            headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
        ) from exc

    if claims.get("typ") != "access":
        # A refresh token presented as an access token must not work: it has a
        # week-long life and is meant to be usable only at /auth/refresh.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="that token cannot be used to authorise a request",
        )

    user = store.get_user(str(claims.get("sub", "")))
    if user is None:
        # A valid signature over a deleted account. The token is genuine and
        # the account is gone, so the session must not survive.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="that account no longer exists",
        )

    if user["workspace_id"] != claims.get("workspace_id"):
        # The token's workspace no longer matches the account's. Refuse rather
        # than picking one: either could be the stale value.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="workspace changed; sign in again",
        )

    request.state.user_id = user["user_id"]
    request.state.workspace_id = user["workspace_id"]
    return user


CurrentUser = Annotated[dict[str, Any], Depends(get_current_user)]


def get_workspace_id(user: CurrentUser) -> str:
    """The caller's workspace, from the verified session and nowhere else."""
    return str(user["workspace_id"])


WorkspaceDep = Annotated[str, Depends(get_workspace_id)]


def require_role(minimum: str) -> Callable[..., dict[str, Any]]:
    """Deny by default, at the route.

    Written as a factory so a route declares its own floor
    (`Depends(require_role("analyst"))`) and the check cannot be forgotten in a
    handler body where a missing `if` is invisible in review.
    """

    def guard(user: CurrentUser) -> dict[str, Any]:
        if not role_allows(str(user.get("role", "viewer")), minimum):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"this action requires the {minimum} role or higher",
            )
        return user

    return guard


#: Roles that may spend money or mutate the corpus. `viewer` is read-only, so
#: uploads and runs sit behind this.
RequireAnalyst = Annotated[dict[str, Any], Depends(require_role("analyst"))]
RequireAdmin = Annotated[dict[str, Any], Depends(require_role("admin"))]


def require_model(settings: Settings) -> None:
    """Fail early and specifically when a route needs a model and none is set.

    The message names the variable for the *resolved* provider, so a Groq
    deployment is not told to go and set an Anthropic key.
    """
    if not settings.has_model_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"no {settings.model_key_env_var} is configured, so agent runs "
                "are unavailable. Ingestion, retrieval, chat grounding and the "
                "benchmark harness all work without one."
            ),
        )
