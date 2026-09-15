"""CSRF protection for the two routes that authenticate from a cookie.

Most of this API cannot be attacked this way. Every data route authenticates
from an `Authorization: Bearer` header, and a browser never attaches those to a
cross-site request — the attacker's page would have to read the token first,
which is the same-origin policy's job to prevent. Only `/auth/refresh` and
`/auth/logout` authenticate from a cookie, and a cookie *is* sent automatically.

Those two already carry the modern defence: the refresh cookie is
`SameSite=Strict`, so a browser that honours it never sends the cookie on a
cross-site request at all, plus `HttpOnly` and a path scope of `/api/v1/auth`.

This is the second layer, and it exists because `SameSite` has two gaps that
are not theoretical:

  * **Same-site is not same-origin.** `SameSite` treats every subdomain of the
    registrable domain as the same site, so a compromised or attacker-held
    subdomain can issue requests the browser considers same-site and attaches
    the cookie to.
  * Older browsers ignore the attribute entirely and send the cookie anyway.

The mechanism is double-submit: a random token in a cookie the page *can* read,
echoed back in a request header. An attacker's page can cause the cookie to be
sent but cannot read it to set the header, because reading it across origins is
what the same-origin policy forbids.

The comparison is constant-time. A token checked with `==` leaks its prefix
through timing, which turns an unguessable 256-bit value into something worth
attacking one byte at a time.
"""

from __future__ import annotations

import hmac
import secrets
from typing import Any

from fastapi import HTTPException, Request, status

#: Readable by JavaScript on purpose — the page has to echo it into a header.
#: That is not a weakness: the value protects against *cross-origin* requests,
#: and an attacker who can already run script on this origin has no need of it.
CSRF_COOKIE = "resx_csrf"

#: Where the page sends it back.
CSRF_HEADER = "X-CSRF-Token"

#: 32 bytes of randomness, URL-safe. Long enough that guessing is not a
#: strategy, short enough not to bloat every request.
TOKEN_BYTES = 32

#: Methods that cannot change anything, so they need no token. A CSRF check on
#: GET would break every ordinary page load for no gain.
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})


def new_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


def issue(response: Any, settings: Any, token: str | None = None) -> str:
    """Attach a fresh CSRF cookie to a response, returning the token.

    Not `HttpOnly`: the page must read it to echo it back, which is the entire
    mechanism. `SameSite=Strict` and `Secure` still apply, and the value is
    useless to anyone who cannot also cause the refresh cookie to be sent.
    """
    value = token or new_token()
    response.set_cookie(
        CSRF_COOKIE,
        value,
        max_age=settings.refresh_token_ttl_seconds,
        httponly=False,
        secure=settings.cookie_secure,
        samesite="strict",
        path="/",
        domain=settings.cookie_domain or None,
    )
    return value


def clear(response: Any, settings: Any) -> None:
    """Remove the cookie, matching the attributes it was set with.

    A mismatch here leaves the original cookie in place and sign-out silently
    does nothing — the same trap the refresh cookie documents.
    """
    response.delete_cookie(
        CSRF_COOKIE,
        path="/",
        domain=settings.cookie_domain or None,
        secure=settings.cookie_secure,
        samesite="strict",
    )


def verify(request: Request, *, cookie_name: str) -> None:
    """Refuse a state-changing request whose header does not match its cookie.

    `cookie_name` is the *session* cookie. A request that carries no session
    cookie is not authenticated by a cookie, so there is nothing to forge and
    the check does not apply — this is what keeps the bearer-token routes and
    the very first login working.
    """
    if request.method.upper() in SAFE_METHODS:
        return
    if not request.cookies.get(cookie_name):
        return

    sent = request.headers.get(CSRF_HEADER, "")
    stored = request.cookies.get(CSRF_COOKIE, "")

    # Both must be present. Treating an absent pair as a match would make the
    # whole check opt-out by simply omitting it.
    if not sent or not stored or not hmac.compare_digest(sent, stored):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"CSRF check failed: send the {CSRF_COOKIE} cookie's value in "
                f"the {CSRF_HEADER} header. This request carried a session "
                f"cookie, so it has to prove it came from this site."
            ),
        )
