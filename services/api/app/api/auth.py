"""Authentication routes.

The refresh token is delivered as an **HttpOnly cookie**, never in the response
body, and the access token is returned in the body for the client to hold in
memory. That split is the point: JavaScript cannot read the long-lived
credential, so an XSS that gets script execution still cannot exfiltrate a
week-long session. The short-lived access token it *can* reach expires in
fifteen minutes.

`SameSite=Strict` on the cookie is the CSRF control for the refresh endpoint.
The other mutating routes are protected by the `Authorization` header, which a
cross-site form post cannot set.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from app.api.deps import (
    AuthServiceDep,
    CurrentUser,
    RequireAdmin,
    SettingsDep,
    StoreDep,
    WorkspaceDep,
    client_ip,
)
from app.core import csrf
from app.services.auth import AuthError, public_user

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

#: The cookie the browser sends back on /refresh and /logout.
REFRESH_COOKIE = "resx_refresh"


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=1024)
    name: str = Field(min_length=1, max_length=120)
    workspace_name: str | None = Field(default=None, max_length=120)


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=1024)
    #: Six digits from the authenticator, or a recovery code. Optional because
    #: most accounts have no second factor; required by the service for the
    #: ones that do, which answers with 401 and a message naming this field.
    totp_code: str | None = Field(default=None, max_length=64)


class TotpConfirmRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=6, max_length=10)


class TotpDisableRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    password: str = Field(min_length=1, max_length=1024)


class ChangePasswordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_password: str = Field(min_length=1, max_length=1024)
    new_password: str = Field(min_length=1, max_length=1024)


def _set_refresh_cookie(response: Response, token: str, settings: Any) -> None:
    response.set_cookie(
        REFRESH_COOKIE,
        token,
        max_age=settings.refresh_token_ttl_seconds,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="strict",
        path="/api/v1/auth",
        domain=settings.cookie_domain or None,
    )


def _issue_session_cookies(response: Response, token: str, settings: Any) -> None:
    """Set the refresh cookie and a matching CSRF token.

    Together, because they are only meaningful together: the CSRF token exists
    to prove that a request carrying the refresh cookie came from this site, so
    a session without one would fail every subsequent refresh.
    """
    _set_refresh_cookie(response, token, settings)
    csrf.issue(response, settings)


def _clear_refresh_cookie(response: Response, settings: Any) -> None:
    # The attributes must match the ones used to set it, or the browser keeps
    # the original cookie and "sign out" silently does nothing.
    response.delete_cookie(
        REFRESH_COOKIE,
        path="/api/v1/auth",
        domain=settings.cookie_domain or None,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="strict",
    )


def _as_http(exc: AuthError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.message)


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest,
    request: Request,
    response: Response,
    auth: AuthServiceDep,
    settings: SettingsDep,
) -> dict[str, Any]:
    try:
        session = auth.register(
            email=body.email,
            password=body.password,
            name=body.name,
            workspace_name=body.workspace_name,
            ip=client_ip(request),
        )
    except AuthError as exc:
        raise _as_http(exc) from exc

    _issue_session_cookies(response, session.refresh_token, settings)
    return session.to_dict()


@router.post("/login")
async def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    auth: AuthServiceDep,
    settings: SettingsDep,
) -> dict[str, Any]:
    try:
        session = auth.login(
            email=body.email,
            password=body.password,
            ip=client_ip(request),
            totp_code=body.totp_code,
        )
    except AuthError as exc:
        raise _as_http(exc) from exc

    _issue_session_cookies(response, session.refresh_token, settings)
    return session.to_dict()


@router.post("/refresh")
async def refresh(
    request: Request,
    response: Response,
    auth: AuthServiceDep,
    settings: SettingsDep,
) -> dict[str, Any]:
    # Before anything else: this route authenticates from a cookie, which a
    # browser attaches automatically, so it is the one shape of request an
    # attacker's page could cause. No-op when no session cookie is present.
    csrf.verify(request, cookie_name=REFRESH_COOKIE)

    token = request.cookies.get(REFRESH_COOKIE)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="no session cookie"
        )
    try:
        session = auth.refresh(refresh_token=token, ip=client_ip(request))
    except AuthError as exc:
        # Clear the cookie on any failure. Leaving a dead token in the browser
        # makes the client retry forever against a session that cannot recover.
        _clear_refresh_cookie(response, settings)
        csrf.clear(response, settings)
        raise _as_http(exc) from exc

    _issue_session_cookies(response, session.refresh_token, settings)
    return session.to_dict()


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    request: Request,
    response: Response,
    auth: AuthServiceDep,
    settings: SettingsDep,
) -> Response:
    csrf.verify(request, cookie_name=REFRESH_COOKIE)
    auth.logout(refresh_token=request.cookies.get(REFRESH_COOKIE), ip=client_ip(request))
    _clear_refresh_cookie(response, settings)
    csrf.clear(response, settings)
    return Response(status_code=status.HTTP_204_NO_CONTENT, headers=dict(response.headers))


@router.post("/totp/enrol")
async def begin_totp(user: CurrentUser, auth: AuthServiceDep) -> dict[str, str]:
    """Start two-factor enrolment.

    Returns the secret and an `otpauth://` URI for the QR code. Neither is
    active yet: nothing changes about this account until the user proves they
    can generate a code from it. Activating on generation would lock out anyone
    whose app failed to scan, with no way back in.
    """
    try:
        return auth.begin_totp_enrolment(user_id=user["user_id"])
    except AuthError as exc:
        raise _as_http(exc) from exc


@router.post("/totp/confirm")
async def confirm_totp(
    body: TotpConfirmRequest, user: CurrentUser, auth: AuthServiceDep
) -> dict[str, Any]:
    """Finish enrolment by proving the authenticator works.

    The recovery codes come back once and are never retrievable again — they
    are stored hashed, like passwords, because they grant a session exactly as
    a password does.
    """
    try:
        codes = auth.confirm_totp_enrolment(user_id=user["user_id"], code=body.code)
    except AuthError as exc:
        raise _as_http(exc) from exc
    return {
        "enabled": True,
        "recovery_codes": codes,
        "note": (
            "Save these now. Each works once, and they are stored hashed — "
            "nobody can show them to you again."
        ),
    }


@router.post("/totp/disable", status_code=status.HTTP_204_NO_CONTENT)
async def disable_totp(
    body: TotpDisableRequest, user: CurrentUser, auth: AuthServiceDep
) -> Response:
    """Turn two-factor off. The password is re-checked first, because an
    unlocked laptop is the ordinary way someone else reaches this."""
    try:
        auth.disable_totp(user_id=user["user_id"], password=body.password)
    except AuthError as exc:
        raise _as_http(exc) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


class InviteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=3, max_length=320)
    #: Never "owner". Ownership transfers as its own authorised action.
    role: str = Field(default="analyst", max_length=16)


class AcceptInviteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token: str = Field(min_length=16, max_length=256)
    password: str = Field(min_length=1, max_length=1024)
    name: str = Field(min_length=1, max_length=200)


@router.post("/invitations", status_code=status.HTTP_201_CREATED)
async def create_invitation(
    body: InviteRequest,
    request: Request,
    auth: AuthServiceDep,
    settings: SettingsDep,
    workspace_id: WorkspaceDep,
    actor: RequireAdmin,
) -> dict[str, Any]:
    """Invite someone into this workspace.

    Returns the link **once**. The token is stored only as a SHA-256 digest,
    so a database dump yields no working invitations — and it also means nobody
    can show it again, which the response says out loud rather than letting the
    inviter discover it by coming back later.

    No email is sent: there is no mail provider configured, and pretending to
    send one would produce an invitation that silently goes nowhere. The
    inviter passes the link on however they already talk to the person.
    """
    try:
        invitation = auth.invite(
            workspace_id=workspace_id,
            inviter_id=actor["user_id"],
            email=body.email,
            role=body.role,
        )
    except AuthError as exc:
        raise _as_http(exc) from exc

    token = invitation.pop("token")
    base = str(getattr(settings, "resx_base_url", "") or "").rstrip("/")
    return {
        **invitation,
        "invite_url": f"{base}/join?token={token}" if base else None,
        "token": token,
        "note": (
            "Send this link to them yourself — no email was sent. It is shown "
            "once and cannot be retrieved again; revoke and re-issue if it is "
            "lost."
        ),
    }


@router.get("/invitations")
async def list_invitations(
    store: StoreDep, workspace_id: WorkspaceDep, actor: RequireAdmin
) -> list[dict[str, Any]]:
    """Outstanding and accepted invitations. Never the tokens."""
    return store.list_invitations(workspace_id=workspace_id)


@router.delete("/invitations/{invite_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_invitation(
    invite_id: str,
    store: StoreDep,
    workspace_id: WorkspaceDep,
    actor: RequireAdmin,
) -> Response:
    """Withdraw an invitation that has not been accepted."""
    if not store.revoke_invitation(workspace_id=workspace_id, invite_id=invite_id):
        raise HTTPException(
            status_code=404,
            detail="no such open invitation; an accepted one cannot be revoked",
        )
    store.audit(
        workspace_id=workspace_id,
        actor=actor.get("user_id"),
        action="workspace.invite_revoked",
        target=invite_id,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/invitations/accept", status_code=status.HTTP_201_CREATED)
async def accept_invitation(
    body: AcceptInviteRequest,
    request: Request,
    response: Response,
    auth: AuthServiceDep,
    settings: SettingsDep,
) -> dict[str, Any]:
    """Join a workspace with an invitation link.

    Anonymous by design — the person accepting has no account yet, which is
    the whole point. The email comes from the invitation rather than from this
    request, so a link issued to one address cannot be redeemed under another.
    """
    try:
        session = auth.accept_invitation(
            token=body.token,
            password=body.password,
            name=body.name,
            ip=client_ip(request),
        )
    except AuthError as exc:
        raise _as_http(exc) from exc

    _issue_session_cookies(response, session.refresh_token, settings)
    return session.to_dict()


@router.get("/me")
async def me(user: CurrentUser, store: StoreDep, workspace_id: WorkspaceDep) -> dict[str, Any]:
    """The signed-in user plus their workspace's headline counts.

    Bundled because the dashboard shell needs both on first paint, and two
    round trips would show an empty header for the length of one of them.
    """
    return {
        "user": public_user(user),
        "workspace": {
            "workspace_id": workspace_id,
            **store.corpus_summary(workspace_id=workspace_id),
        },
    }


@router.post("/change-password", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(
    body: ChangePasswordRequest,
    request: Request,
    user: CurrentUser,
    auth: AuthServiceDep,
) -> Response:
    try:
        auth.change_password(
            user_id=user["user_id"],
            current=body.current_password,
            replacement=body.new_password,
            ip=client_ip(request),
        )
    except AuthError as exc:
        raise _as_http(exc) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/audit")
async def audit_trail(
    user: CurrentUser,
    store: StoreDep,
    workspace_id: WorkspaceDep,
    limit: Annotated[int, Field(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    """The workspace's audit trail. Admin and above.

    Enforced here rather than in the UI: hiding a menu item is not access
    control, and this endpoint reveals who signed in and when.
    """
    from app.services.auth import role_allows

    if not role_allows(user.get("role", "viewer"), "admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="the audit trail requires the admin role",
        )
    return {"entries": store.read_audit(workspace_id=workspace_id, limit=limit)}


__all__ = ["REFRESH_COOKIE", "router"]
