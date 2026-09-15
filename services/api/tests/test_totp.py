"""Two-factor enrolment.

The schema already had a `totp_secret` column and nothing ever wrote to it, so
the feature existed as a place to put a secret and no way to obtain one.

Two decisions here are load-bearing, and both are about not locking people out:

  * **Enrolment is two steps.** The secret is generated and held pending; it
    becomes real only once the user proves they can produce a code from it.
    Activating on generation would lock out anyone whose app failed to scan —
    an account demanding a code nothing can produce, with no way back in.
  * **Recovery codes.** A lost phone is otherwise a lost account, which is the
    reason people decline 2FA in the first place.
"""

from __future__ import annotations

from typing import Any

import pyotp


def enrol(client: Any) -> tuple[str, list[str]]:
    """Complete an enrolment, returning the secret and its recovery codes."""
    started = client.post("/api/v1/auth/totp/enrol")
    assert started.status_code == 200, started.text
    secret = started.json()["secret"]

    confirmed = client.post(
        "/api/v1/auth/totp/confirm", json={"code": pyotp.TOTP(secret).now()}
    )
    assert confirmed.status_code == 200, confirmed.text
    return secret, confirmed.json()["recovery_codes"]


def test_enrolment_returns_a_scannable_uri(client: Any) -> None:
    body = client.post("/api/v1/auth/totp/enrol").json()
    assert body["secret"]
    # The URI is what the QR code encodes; without it the user has to type a
    # base32 secret by hand.
    assert body["otpauth_uri"].startswith("otpauth://totp/")
    assert "issuer=" in body["otpauth_uri"]


def test_generating_a_secret_does_not_enable_anything(client: Any) -> None:
    """The decision that prevents a lockout. Someone whose app fails to scan
    must still be able to sign in."""
    client.post("/api/v1/auth/totp/enrol")
    assert client.get("/api/v1/auth/me").json()["user"]["totp_enabled"] is False


def test_a_wrong_code_does_not_complete_enrolment(client: Any) -> None:
    client.post("/api/v1/auth/totp/enrol")
    response = client.post("/api/v1/auth/totp/confirm", json={"code": "000000"})
    assert response.status_code == 400
    assert client.get("/api/v1/auth/me").json()["user"]["totp_enabled"] is False


def test_confirming_enables_it_and_hands_over_recovery_codes(client: Any) -> None:
    _secret, codes = enrol(client)
    assert len(codes) == 10
    assert len(set(codes)) == 10, "duplicate recovery codes halve the protection"
    assert client.get("/api/v1/auth/me").json()["user"]["totp_enabled"] is True


def test_confirming_without_starting_is_refused(client: Any) -> None:
    response = client.post("/api/v1/auth/totp/confirm", json={"code": "123456"})
    assert response.status_code == 409


def test_the_secret_never_appears_in_a_response(client: Any) -> None:
    """The one thing that must not leak. Anyone holding it can generate codes
    forever."""
    secret, codes = enrol(client)
    body = str(client.get("/api/v1/auth/me").json())
    assert secret not in body
    assert "totp_secret" not in body
    # Nor the recovery hashes, nor how many are left — that last one tells an
    # attacker how close the account is to being locked out.
    for code in codes:
        assert code not in body
    assert "recovery_codes" not in body


# --------------------------------------------------------------------------- #
# Login
# --------------------------------------------------------------------------- #


CREDENTIALS = {
    "email": "totp@example.com",
    "password": "correct-horse-battery-staple-9",
    "name": "TOTP Tester",
}


def registered(anon: Any) -> Any:
    response = anon.post("/api/v1/auth/register", json=CREDENTIALS)
    assert response.status_code == 201, response.text
    anon.headers["Authorization"] = f"Bearer {response.json()['access_token']}"
    return anon


def test_login_demands_a_code_once_enrolled(anon: Any) -> None:
    client = registered(anon)
    enrol(client)
    client.headers.pop("Authorization", None)

    response = client.post(
        "/api/v1/auth/login",
        json={"email": CREDENTIALS["email"], "password": CREDENTIALS["password"]},
    )
    assert response.status_code == 401
    # Distinguishable from a wrong password, or a working login looks broken
    # and the user goes off to reset a password that is fine.
    assert "totp_code" in str(response.json())


def test_login_succeeds_with_the_right_code(anon: Any) -> None:
    client = registered(anon)
    secret, _codes = enrol(client)
    client.headers.pop("Authorization", None)

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": CREDENTIALS["email"],
            "password": CREDENTIALS["password"],
            "totp_code": pyotp.TOTP(secret).now(),
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["access_token"]


def test_a_recovery_code_works_in_place_of_the_authenticator(anon: Any) -> None:
    """The lost-phone path, which is the whole reason recovery codes exist."""
    client = registered(anon)
    _secret, codes = enrol(client)
    client.headers.pop("Authorization", None)

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": CREDENTIALS["email"],
            "password": CREDENTIALS["password"],
            "totp_code": codes[0],
        },
    )
    assert response.status_code == 200, response.text


def test_a_recovery_code_works_only_once(anon: Any) -> None:
    """A one-time code that survives use is a permanent credential sitting in
    a list the user believes is shrinking."""
    client = registered(anon)
    _secret, codes = enrol(client)
    client.headers.pop("Authorization", None)

    payload = {
        "email": CREDENTIALS["email"],
        "password": CREDENTIALS["password"],
        "totp_code": codes[0],
    }
    assert client.post("/api/v1/auth/login", json=payload).status_code == 200
    assert client.post("/api/v1/auth/login", json=payload).status_code == 401


def test_the_password_is_still_required(anon: Any) -> None:
    """A valid code is a second factor, not a replacement for the first."""
    client = registered(anon)
    secret, _codes = enrol(client)
    client.headers.pop("Authorization", None)

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": CREDENTIALS["email"],
            "password": "not-the-password",
            "totp_code": pyotp.TOTP(secret).now(),
        },
    )
    assert response.status_code == 401


def test_an_account_without_2fa_logs_in_unchanged(anon: Any) -> None:
    """Most accounts have no second factor and must not pay for this."""
    registered(anon)
    anon.headers.pop("Authorization", None)
    response = anon.post(
        "/api/v1/auth/login",
        json={"email": CREDENTIALS["email"], "password": CREDENTIALS["password"]},
    )
    assert response.status_code == 200, response.text


# --------------------------------------------------------------------------- #
# Turning it off
# --------------------------------------------------------------------------- #


def test_disabling_requires_the_password(client: Any) -> None:
    """An unlocked laptop is the ordinary way someone else reaches this
    endpoint, and removing 2FA is exactly what they would want to do."""
    enrol(client)
    refused = client.post("/api/v1/auth/totp/disable", json={"password": "wrong-password"})
    assert refused.status_code == 401
    assert client.get("/api/v1/auth/me").json()["user"]["totp_enabled"] is True


#: What `conftest.py` registers the shared `client` with. Named here so a
#: change there fails loudly rather than as a puzzling 401.
CLIENT_PASSWORD = "a-long-enough-password"


def test_disabling_with_the_password_turns_it_off(client: Any) -> None:
    enrol(client)
    response = client.post(
        "/api/v1/auth/totp/disable",
        json={"password": CLIENT_PASSWORD},
    )
    assert response.status_code == 204
    assert client.get("/api/v1/auth/me").json()["user"]["totp_enabled"] is False


def test_enrolling_twice_is_refused(client: Any) -> None:
    """Silently replacing the secret would invalidate the authenticator the
    user is still relying on."""
    enrol(client)
    assert client.post("/api/v1/auth/totp/enrol").status_code == 409
