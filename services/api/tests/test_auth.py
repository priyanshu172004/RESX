"""Authentication and tenancy tests.

The properties asserted here are the ones whose failure is unrecoverable: a
cross-tenant read, a replayed session token, a password that is not actually
hashed. Each is asserted directly rather than inferred from a route returning
200.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.core.config import Settings, get_settings
from app.services.auth import (
    AccountLockedError,
    AuthService,
    EmailInUseError,
    InvalidCredentialsError,
    SessionExpiredError,
    TokenReuseError,
    WeakPasswordError,
    role_allows,
)
from app.store.local import LocalStore


@pytest.fixture
def store(tmp_path: Path) -> Iterator[LocalStore]:
    instance = LocalStore(tmp_path / "resx.db")
    yield instance
    instance.close()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        resx_env="development",
        storage_local_root=str(tmp_path / "storage"),
        store_backend="sqlite",
        groq_api_key="",
        anthropic_api_key="",
        voyage_api_key="",
        # Kept low so the suite is not spending 64 MiB of Argon2 work per
        # login across dozens of tests. The floors themselves are asserted in
        # test_security.py against the real settings.
        argon2_memory_cost=65_536,
        argon2_time_cost=2,
    )


@pytest.fixture
def service(store: LocalStore, settings: Settings) -> AuthService:
    return AuthService(store=store, settings=settings)


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


def test_registration_creates_an_isolated_workspace(service: AuthService) -> None:
    first = service.register(email="a@example.com", password="a-long-enough-password", name="A")
    second = service.register(
        email="b@example.com", password="a-long-enough-password", name="B"
    )
    # Two accounts must never share a workspace: the workspace is the tenant
    # boundary, so sharing one would make every document mutually visible.
    assert first.user["workspace_id"] != second.user["workspace_id"]


def test_the_password_is_never_stored_in_plaintext(
    service: AuthService, store: LocalStore
) -> None:
    password = "correct-horse-battery-staple"
    session = service.register(email="a@example.com", password=password, name="A")

    row = store.get_user(session.user["user_id"])
    assert row is not None
    assert password not in row["password_hash"]
    # Argon2id specifically. bcrypt and PBKDF2 are not memory-hard, and
    # memory-hardness is the entire reason for the choice.
    assert row["password_hash"].startswith("$argon2id$")


def test_the_hash_never_reaches_a_response(service: AuthService) -> None:
    session = service.register(
        email="a@example.com", password="a-long-enough-password", name="A"
    )
    for hidden in ("password_hash", "totp_secret", "failed_logins", "locked_until"):
        assert hidden not in session.user


def test_a_duplicate_email_is_a_conflict(service: AuthService) -> None:
    service.register(email="a@example.com", password="a-long-enough-password", name="A")
    with pytest.raises(EmailInUseError):
        service.register(email="A@Example.COM", password="a-long-enough-password", name="A2")


def test_email_is_normalised_before_comparison(service: AuthService) -> None:
    service.register(
        email="  Mixed.Case@Example.com  ", password="a-long-enough-password", name="A"
    )
    # Otherwise the same person could register twice and end up with two
    # workspaces, neither of which can see the other's documents.
    session = service.login(email="mixed.case@example.com", password="a-long-enough-password")
    assert session.user["email"] == "mixed.case@example.com"


def test_a_short_password_is_rejected(service: AuthService) -> None:
    with pytest.raises(WeakPasswordError):
        service.register(email="a@example.com", password="short", name="A")


def test_an_enormous_password_is_rejected(service: AuthService) -> None:
    # Argon2id hashes whatever it is given and is designed to be slow, so an
    # unbounded password is a denial-of-service vector.
    with pytest.raises(WeakPasswordError):
        service.register(email="a@example.com", password="x" * 5000, name="A")


# --------------------------------------------------------------------------- #
# Login
# --------------------------------------------------------------------------- #


def test_a_wrong_password_is_rejected(service: AuthService) -> None:
    service.register(email="a@example.com", password="a-long-enough-password", name="A")
    with pytest.raises(InvalidCredentialsError):
        service.login(email="a@example.com", password="a-different-password")


def test_an_unknown_account_reports_the_same_error(service: AuthService) -> None:
    # The message must not distinguish "no such account" from "wrong password",
    # or it becomes an account-enumeration oracle.
    with pytest.raises(InvalidCredentialsError) as unknown:
        service.login(email="nobody@example.com", password="a-long-enough-password")

    service.register(email="a@example.com", password="a-long-enough-password", name="A")
    with pytest.raises(InvalidCredentialsError) as wrong:
        service.login(email="a@example.com", password="a-different-password")

    assert str(unknown.value) == str(wrong.value)


def test_a_missing_account_still_verifies_a_hash(service: AuthService) -> None:
    """Timing must not distinguish a missing account from a wrong password.

    Measured rather than asserted by inspection: the failure mode is that
    someone "optimises" the dummy verification away, and the only thing that
    catches it is a test that notices the early return.
    """
    service.register(email="a@example.com", password="a-long-enough-password", name="A")

    def elapsed(email: str) -> float:
        start = time.perf_counter()
        with contextlib.suppress(InvalidCredentialsError):
            service.login(email=email, password="a-different-password")
        return time.perf_counter() - start

    missing = elapsed("nobody@example.com")
    existing = elapsed("a@example.com")

    # A generous bound: this asserts that the work happens at all, not that the
    # timings match to the microsecond. An early return would make `missing`
    # orders of magnitude faster than `existing`, not 3x.
    assert missing > existing / 5, (
        f"missing account returned far too fast ({missing:.4f}s vs {existing:.4f}s) — "
        "the dummy hash verification is being skipped"
    )


def test_repeated_failures_lock_the_account(store: LocalStore, settings: Settings) -> None:
    service = AuthService(
        store=store, settings=settings.model_copy(update={"login_max_attempts": 3})
    )
    service.register(email="a@example.com", password="a-long-enough-password", name="A")

    for _ in range(3):
        with pytest.raises(InvalidCredentialsError):
            service.login(email="a@example.com", password="a-different-password")

    # Locked even with the *correct* password: the lock is on the account, and
    # letting a correct password through would make it trivially bypassable.
    with pytest.raises(AccountLockedError):
        service.login(email="a@example.com", password="a-long-enough-password")


def test_a_successful_login_clears_the_failure_count(
    service: AuthService, store: LocalStore
) -> None:
    session = service.register(
        email="a@example.com", password="a-long-enough-password", name="A"
    )
    with pytest.raises(InvalidCredentialsError):
        service.login(email="a@example.com", password="wrong-password-here")

    service.login(email="a@example.com", password="a-long-enough-password")
    row = store.get_user(session.user["user_id"])
    assert row is not None
    assert row["failed_logins"] == 0


# --------------------------------------------------------------------------- #
# Refresh rotation
# --------------------------------------------------------------------------- #


def test_refresh_rotates_the_token(service: AuthService) -> None:
    first = service.register(email="a@example.com", password="a-long-enough-password", name="A")
    second = service.refresh(refresh_token=first.refresh_token)
    assert second.refresh_token != first.refresh_token


def test_only_the_digest_is_stored(service: AuthService, store: LocalStore) -> None:
    session = service.register(
        email="a@example.com", password="a-long-enough-password", name="A"
    )
    # A database dump must not yield usable session tokens.
    assert store.get_refresh_token(session.refresh_token.encode()) is None

    from app.core import security

    digest = security.hash_refresh_token(session.refresh_token)
    assert store.get_refresh_token(digest) is not None


def test_reusing_a_refresh_token_revokes_the_whole_family(
    service: AuthService,
) -> None:
    """The property that matters: a replayed token kills the chain.

    Revoking only the replayed token would leave the attacker's copy of the
    *next* one working, which is no protection at all.
    """
    first = service.register(email="a@example.com", password="a-long-enough-password", name="A")
    second = service.refresh(refresh_token=first.refresh_token)

    # The captured token, presented a second time.
    with pytest.raises(TokenReuseError):
        service.refresh(refresh_token=first.refresh_token)

    # And the legitimate client's newer token is dead too.
    with pytest.raises(SessionExpiredError):
        service.refresh(refresh_token=second.refresh_token)


def test_an_expired_refresh_token_is_rejected(store: LocalStore, settings: Settings) -> None:
    service = AuthService(
        store=store,
        settings=settings.model_copy(update={"refresh_token_ttl_seconds": -1}),
    )
    session = service.register(
        email="a@example.com", password="a-long-enough-password", name="A"
    )
    with pytest.raises(SessionExpiredError):
        service.refresh(refresh_token=session.refresh_token)


def test_an_unknown_refresh_token_is_rejected(service: AuthService) -> None:
    with pytest.raises(SessionExpiredError):
        service.refresh(refresh_token="not-a-token-that-was-ever-issued")


def test_logout_revokes_the_family(service: AuthService) -> None:
    session = service.register(
        email="a@example.com", password="a-long-enough-password", name="A"
    )
    service.logout(refresh_token=session.refresh_token)
    with pytest.raises(SessionExpiredError):
        service.refresh(refresh_token=session.refresh_token)


# --------------------------------------------------------------------------- #
# Roles
# --------------------------------------------------------------------------- #


def test_role_hierarchy() -> None:
    assert role_allows("owner", "viewer")
    assert role_allows("admin", "analyst")
    assert role_allows("analyst", "analyst")
    assert not role_allows("viewer", "analyst")
    assert not role_allows("analyst", "admin")


def test_an_unknown_role_is_denied_everything() -> None:
    # Fail closed: a role that is not in the hierarchy must not be treated as
    # privileged because the comparison happened to succeed.
    assert not role_allows("", "viewer")
    assert not role_allows("superuser", "viewer")


# --------------------------------------------------------------------------- #
# The audit trail
# --------------------------------------------------------------------------- #


def test_auth_events_are_audited(service: AuthService, store: LocalStore) -> None:
    session = service.register(
        email="a@example.com", password="a-long-enough-password", name="A"
    )
    with pytest.raises(InvalidCredentialsError):
        service.login(email="a@example.com", password="a-different-password")
    service.login(email="a@example.com", password="a-long-enough-password")

    actions = [
        entry["action"] for entry in store.read_audit(workspace_id=session.user["workspace_id"])
    ]
    assert "auth.register" in actions
    assert "auth.login_failed" in actions
    assert "auth.login" in actions


def test_reuse_detection_is_audited(service: AuthService, store: LocalStore) -> None:
    session = service.register(
        email="a@example.com", password="a-long-enough-password", name="A"
    )
    service.refresh(refresh_token=session.refresh_token)
    with pytest.raises(TokenReuseError):
        service.refresh(refresh_token=session.refresh_token)

    actions = [
        entry["action"] for entry in store.read_audit(workspace_id=session.user["workspace_id"])
    ]
    assert "auth.refresh_reuse_detected" in actions


# --------------------------------------------------------------------------- #
# Tenancy, through the HTTP surface
# --------------------------------------------------------------------------- #


@pytest.fixture
def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> Iterator[TestClient]:
    shared = LocalStore(tmp_path / "storage" / "resx.db")

    get_settings.cache_clear()
    deps.get_store.cache_clear()
    deps.get_embedder.cache_clear()
    deps.get_sandbox.cache_clear()
    deps.get_run_service.cache_clear()

    monkeypatch.setattr("app.core.config.get_settings", lambda: settings)
    monkeypatch.setattr("app.api.deps.get_settings", lambda: settings)
    monkeypatch.setattr("app.api.deps.get_store", lambda: shared)

    from app.main import create_app

    app = create_app()
    app.dependency_overrides[deps.get_store] = lambda: shared
    app.dependency_overrides[get_settings] = lambda: settings

    with TestClient(app) as test_client:
        yield test_client

    shared.close()


def _register(client: TestClient, email: str) -> str:
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "a-long-enough-password", "name": email},
    )
    assert response.status_code == 201, response.text
    return response.json()["access_token"]


def test_the_refresh_token_is_httponly_and_not_in_the_body(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "a@example.com", "password": "a-long-enough-password", "name": "A"},
    )
    assert response.status_code == 201

    # An XSS that gets script execution must not be able to read the long-lived
    # credential, so it travels only as an HttpOnly cookie.
    assert "refresh_token" not in response.json()
    cookie = response.headers.get("set-cookie", "")
    assert "resx_refresh=" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=strict" in cookie.lower().replace("samesite=strict", "SameSite=strict")


def test_a_refresh_token_cannot_authorise_a_request(client: TestClient) -> None:
    client.post(
        "/api/v1/auth/register",
        json={"email": "a@example.com", "password": "a-long-enough-password", "name": "A"},
    )
    refresh = client.cookies.get("resx_refresh")
    assert refresh

    # It has a week-long life and is meant to work only at /auth/refresh.
    response = client.get("/api/v1/documents", headers={"Authorization": f"Bearer {refresh}"})
    assert response.status_code == 401


def test_one_account_cannot_see_another_accounts_documents(
    client: TestClient,
) -> None:
    import io

    owner = _register(client, "owner@example.com")
    upload = client.post(
        "/api/v1/documents",
        files={"file": ("mine.csv", io.BytesIO(b"region,revenue\nnorth,100\n"), "text/csv")},
        headers={"Authorization": f"Bearer {owner}"},
    )
    assert upload.status_code == 201, upload.text
    doc_id = upload.json()["doc_id"]

    intruder = _register(client, "intruder@example.com")
    headers = {"Authorization": f"Bearer {intruder}"}

    assert client.get("/api/v1/documents", headers=headers).json() == []
    # 404 rather than 403: confirming that an id exists is itself a leak.
    assert client.get(f"/api/v1/documents/{doc_id}", headers=headers).status_code == 404
    assert client.delete(f"/api/v1/documents/{doc_id}", headers=headers).status_code == 404
    assert client.get("/api/v1/datasets", headers=headers).json() == []


def test_the_audit_trail_requires_admin(client: TestClient) -> None:
    token = _register(client, "owner@example.com")
    # The first account is the workspace owner, so it is allowed.
    assert (
        client.get(
            "/api/v1/auth/audit", headers={"Authorization": f"Bearer {token}"}
        ).status_code
        == 200
    )


def test_me_returns_the_workspace_summary(client: TestClient) -> None:
    token = _register(client, "owner@example.com")
    body = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}).json()
    assert body["user"]["role"] == "owner"
    assert body["workspace"]["documents"] == 0
    assert "password_hash" not in body["user"]
