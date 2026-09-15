"""Tests for the auth primitives.

These assert the properties docs/05-SECURITY.md actually promises, rather than
just that the functions run: that the password hash is Argon2id with a
memory-hard cost, that a refresh token is never stored in the clear, and that
token verification rejects a wrong audience, a wrong issuer, and an expired
token.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from app.core import security
from app.core.config import Settings

# --------------------------------------------------------------------------- #
# Passwords
# --------------------------------------------------------------------------- #


def test_password_hash_is_argon2id_and_memory_hard() -> None:
    digest = security.hash_password("correct horse battery staple")

    # Argon2id specifically — not argon2i, not argon2d.
    assert digest.startswith("$argon2id$")

    # 64 MiB floor. A lowered cost here is the single easiest way to silently
    # destroy the value of choosing Argon2 at all.
    params = dict(pair.split("=", 1) for pair in digest.split("$")[3].split(",") if "=" in pair)
    assert int(params["m"]) >= 65_536
    assert int(params["t"]) >= 2


def test_password_verifies_and_rejects() -> None:
    digest = security.hash_password("s3cret-passphrase")
    assert security.verify_password("s3cret-passphrase", digest) is True
    assert security.verify_password("s3cret-passphrase ", digest) is False
    assert security.verify_password("wrong", digest) is False


def test_password_hash_is_salted() -> None:
    # Identical passwords must not produce identical hashes, or the database
    # reveals which users share a password.
    a = security.hash_password("same-password")
    b = security.hash_password("same-password")
    assert a != b


def test_malformed_hash_returns_false_rather_than_raising() -> None:
    assert security.verify_password("anything", "not-a-hash") is False


# --------------------------------------------------------------------------- #
# Refresh tokens
# --------------------------------------------------------------------------- #


def test_refresh_token_is_never_stored_in_the_clear() -> None:
    plaintext, digest = security.new_refresh_token()

    assert len(plaintext) >= 32
    assert isinstance(digest, bytes)
    assert len(digest) == 32  # SHA-256
    assert plaintext.encode() not in digest
    assert security.hash_refresh_token(plaintext) == digest


def test_refresh_tokens_are_unique() -> None:
    tokens = {security.new_refresh_token()[0] for _ in range(50)}
    assert len(tokens) == 50


def test_constant_time_equals() -> None:
    assert security.constant_time_equals("abc", "abc") is True
    assert security.constant_time_equals("abc", "abd") is False
    assert security.constant_time_equals(b"abc", "abc") is True


# --------------------------------------------------------------------------- #
# Access tokens
# --------------------------------------------------------------------------- #


@pytest.fixture
def rsa_settings(tmp_path, monkeypatch) -> Settings:  # type: ignore[no-untyped-def]
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    private_path = tmp_path / "jwt_private.pem"
    public_path = tmp_path / "jwt_public.pem"
    private_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )

    settings = Settings(
        _env_file=None,
        jwt_private_key_path=str(private_path),
        jwt_public_key_path=str(public_path),
    )
    monkeypatch.setattr(security, "_private_key", lambda: private_path.read_bytes())
    monkeypatch.setattr(security, "_public_key", lambda: public_path.read_bytes())
    return settings


def test_access_token_roundtrip(rsa_settings: Settings) -> None:
    user_id, workspace_id = uuid.uuid4(), uuid.uuid4()

    token, expires = security.issue_access_token(
        user_id=user_id,
        workspace_id=workspace_id,
        role="analyst",
        settings=rsa_settings,
    )
    claims = security.decode_access_token(token, settings=rsa_settings)

    assert claims["sub"] == str(user_id)
    assert claims["workspace_id"] == str(workspace_id)
    assert claims["role"] == "analyst"
    assert claims["typ"] == "access"
    assert claims["iss"] == rsa_settings.jwt_issuer
    assert claims["aud"] == rsa_settings.jwt_audience

    # 15 minutes, so a stolen access token has a short useful life.
    assert timedelta(seconds=0) < expires - datetime.now(timezone.utc) <= timedelta(minutes=15)


def test_access_token_is_asymmetric(rsa_settings: Settings) -> None:
    token, _ = security.issue_access_token(
        user_id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        role="viewer",
        settings=rsa_settings,
    )
    header = jwt.get_unverified_header(token)
    assert header["alg"] == "RS256"


def test_wrong_audience_is_rejected(rsa_settings: Settings) -> None:
    token, _ = security.issue_access_token(
        user_id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        role="viewer",
        settings=rsa_settings,
    )
    other = rsa_settings.model_copy(update={"jwt_audience": "some-other-service"})
    with pytest.raises(jwt.InvalidAudienceError):
        security.decode_access_token(token, settings=other)


def test_wrong_issuer_is_rejected(rsa_settings: Settings) -> None:
    token, _ = security.issue_access_token(
        user_id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        role="viewer",
        settings=rsa_settings,
    )
    other = rsa_settings.model_copy(update={"jwt_issuer": "impostor"})
    with pytest.raises(jwt.InvalidIssuerError):
        security.decode_access_token(token, settings=other)


def test_expired_token_is_rejected(rsa_settings: Settings) -> None:
    expired = rsa_settings.model_copy(update={"access_token_ttl_seconds": -1})
    token, _ = security.issue_access_token(
        user_id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        role="viewer",
        settings=expired,
    )
    with pytest.raises(jwt.ExpiredSignatureError):
        security.decode_access_token(token, settings=rsa_settings)


def test_tampered_token_is_rejected(rsa_settings: Settings) -> None:
    token, _ = security.issue_access_token(
        user_id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        role="viewer",
        settings=rsa_settings,
    )
    head, payload, sig = token.split(".")
    # Flip a character in the signature.
    tampered = f"{head}.{payload}.{'A' if sig[0] != 'A' else 'B'}{sig[1:]}"
    with pytest.raises(jwt.InvalidSignatureError):
        security.decode_access_token(tampered, settings=rsa_settings)


# --------------------------------------------------------------------------- #
# Signed URLs
# --------------------------------------------------------------------------- #


def test_signature_roundtrip_and_rejection() -> None:
    payload, key = "doc_123:1767225600", "signing-key"
    signature = security.sign_payload(payload, key)

    assert security.verify_signature(payload, signature, key) is True
    assert security.verify_signature("doc_124:1767225600", signature, key) is False
    assert security.verify_signature(payload, signature, "other-key") is False
