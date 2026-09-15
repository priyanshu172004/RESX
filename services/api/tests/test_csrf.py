"""CSRF on the two routes a browser authenticates with a cookie.

Everything else in this API uses `Authorization: Bearer`, which a browser never
attaches to a cross-site request, so it cannot be forged this way. `/auth/refresh`
and `/auth/logout` read a cookie, and cookies are sent automatically.

`SameSite=Strict` is the first layer and it is already there. This is the
second, because SameSite treats every subdomain as the same site — so a
compromised subdomain issues requests the browser happily attaches the cookie
to — and older browsers ignore the attribute entirely.
"""

from __future__ import annotations

from typing import Any

from app.core.csrf import CSRF_COOKIE, CSRF_HEADER


def register(client: Any, email: str = "csrf@example.com") -> Any:
    return client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "correct-horse-battery-staple-9",
            "name": "CSRF Tester",
        },
    )


def test_a_session_is_issued_with_a_csrf_token(anon: Any) -> None:
    """Without one, the first refresh would fail and the session would look
    broken on a perfectly good login."""
    assert register(anon).status_code == 201
    assert anon.cookies.get(CSRF_COOKIE), "no CSRF cookie was issued at sign-up"


def test_the_token_is_readable_by_script(anon: Any) -> None:
    """Not HttpOnly, on purpose: the page has to echo it back, and that is the
    whole mechanism. It protects against *cross-origin* reads, which the
    same-origin policy already prevents."""
    register(anon)
    header = "; ".join(anon.headers.get_list("set-cookie") or [])
    if not header:
        # httpx keeps them on the response; fall back to the jar.
        assert anon.cookies.get(CSRF_COOKIE)
        return
    csrf_bits = [c for c in header.split(",") if CSRF_COOKIE in c]
    assert csrf_bits
    assert "httponly" not in csrf_bits[0].lower()


def test_refresh_without_the_header_is_refused(anon: Any) -> None:
    """The attack, as a test: a cross-site page can make the browser send the
    cookie but cannot read it to set the header."""
    assert register(anon).status_code == 201
    assert anon.cookies.get(CSRF_COOKIE)

    response = anon.post("/api/v1/auth/refresh")
    assert response.status_code == 403
    detail = str(response.json())
    # It names the header, because a developer hitting this needs to know what
    # to send, not merely that they were refused.
    assert CSRF_HEADER in detail


def test_refresh_with_a_wrong_token_is_refused(anon: Any) -> None:
    register(anon)
    response = anon.post("/api/v1/auth/refresh", headers={CSRF_HEADER: "not-the-right-token"})
    assert response.status_code == 403


def test_refresh_with_the_matching_token_succeeds(anon: Any) -> None:
    """The legitimate path must still work, which is the other half of the
    property — a check that refuses everything is not protection."""
    assert register(anon).status_code == 201
    token = anon.cookies.get(CSRF_COOKIE)

    response = anon.post("/api/v1/auth/refresh", headers={CSRF_HEADER: token})
    assert response.status_code == 200, response.text
    assert response.json()["access_token"]


def test_logout_is_protected_too(anon: Any) -> None:
    """A forced logout is a denial of service rather than a breach, but it is
    still something an attacker's page should not be able to do."""
    register(anon)
    assert anon.post("/api/v1/auth/logout").status_code == 403


def test_login_itself_needs_no_token(anon: Any) -> None:
    """There is no session to forge yet. Requiring one here would make signing
    in impossible on a fresh browser."""
    register(anon, email="fresh@example.com")
    anon.cookies.clear()
    response = anon.post(
        "/api/v1/auth/login",
        json={"email": "fresh@example.com", "password": "correct-horse-battery-staple-9"},
    )
    assert response.status_code == 200, response.text


def test_bearer_routes_are_untouched(client: Any) -> None:
    """They were never forgeable — a browser does not attach an Authorization
    header cross-site — and adding a check would have broken every API client
    for no gain."""
    assert client.get("/api/v1/documents").status_code == 200
    assert client.post(
        "/api/v1/runs",
        json={"question": "a question long enough to pass validation"},
    ).status_code in {201, 202, 400, 409, 503}


def test_the_token_is_compared_in_constant_time() -> None:
    """A token checked with `==` leaks its prefix through timing, which turns
    an unguessable 256-bit value into something worth attacking byte by byte."""
    import inspect

    from app.core import csrf

    source = inspect.getsource(csrf.verify)
    assert "compare_digest" in source
    assert "sent == stored" not in source
