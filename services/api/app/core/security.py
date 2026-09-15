"""Password hashing, token issuance, and constant-time secret comparison.

The three primitives here are deliberately different from one another, and the
distinction is the most commonly botched decision in this area:

  * Passwords are low-entropy and human-chosen, so they need a **slow,
    memory-hard** hash: Argon2id.
  * Refresh tokens are 256 bits of CSPRNG output. They are not brute-forceable,
    so they need only integrity at rest: SHA-256 is correct and Argon2 would be
    pointless overhead on every refresh.
  * Access tokens need asymmetric signing so a compromised verifier cannot mint
    tokens: RS256.

See docs/05-SECURITY.md §2 and §6.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

from app.core.config import Settings, get_settings

TokenType = Literal["access"]


# --------------------------------------------------------------------------- #
# Passwords
# --------------------------------------------------------------------------- #


@lru_cache
def _hasher() -> PasswordHasher:
    s = get_settings()
    return PasswordHasher(
        time_cost=s.argon2_time_cost,
        memory_cost=s.argon2_memory_cost,
        parallelism=s.argon2_parallelism,
        hash_len=32,
        salt_len=16,
    )


#: Length of a two-factor recovery code, in random bytes before encoding.
#: Ten bytes is 80 bits — far beyond guessing, and short enough that a person
#: can copy one off a screen onto paper without a mistake, which is the point
#: of a recovery code.
RECOVERY_CODE_BYTES = 10


def new_recovery_code() -> str:
    """One single-use code for getting back in without the authenticator.

    Formatted in two groups so it can be read aloud and typed without losing
    the place. Stored hashed like a password, because that is what it is: a
    credential that grants a session.
    """
    raw = secrets.token_hex(RECOVERY_CODE_BYTES)
    return f"{raw[:10]}-{raw[10:]}"


def hash_password(password: str) -> str:
    """Argon2id. Parameters are embedded in the returned hash string, which is
    what makes a future cost increase a rehash-on-next-login rather than a
    migration."""
    return _hasher().hash(password)


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        return _hasher().verify(stored_hash, password)
    except (VerifyMismatchError, InvalidHashError, ValueError):
        # Never distinguish "wrong password" from "malformed hash" to a caller
        # that might surface the difference to a user.
        return False


def needs_rehash(stored_hash: str) -> bool:
    """True when the hash was made with weaker parameters than current policy."""
    try:
        return _hasher().check_needs_rehash(stored_hash)
    except (InvalidHashError, ValueError):
        return True


# --------------------------------------------------------------------------- #
# Access tokens (RS256)
# --------------------------------------------------------------------------- #


@lru_cache
def repo_root() -> Path:
    """Walk up for the marker that identifies the checkout.

    The configured key paths are relative (`./secrets/...`), and the service is
    started from at least three different working directories: `services/api`
    for uvicorn, the repository root for the CLI and the benchmarks, and a
    temporary directory inside the sandbox. Resolving against the cwd therefore
    finds the keys sometimes and not others, which presents as a
    `FileNotFoundError` on the first login rather than as a configuration
    problem.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pnpm-workspace.yaml").exists() or (candidate / ".git").exists():
            return candidate
    return here.parents[3]


def resolve_secret_path(raw: str) -> Path:
    """Locate a secret file whether the path is absolute, cwd- or root-relative."""
    path = Path(raw)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (repo_root() / path).resolve()


def ensure_signing_keypair(settings: Settings | None = None) -> tuple[Path, Path]:
    """Return the keypair's paths, generating one in development if missing.

    Generated only outside production, and loudly. In production a missing key
    is a deployment error: minting a fresh keypair on boot would silently
    invalidate every existing session on every restart, and would differ per
    replica behind a load balancer, so half of all requests would reject a
    perfectly good token.
    """
    s = settings or get_settings()
    private = resolve_secret_path(s.jwt_private_key_path)
    public = resolve_secret_path(s.jwt_public_key_path)
    if private.exists() and public.exists():
        return private, public

    if s.is_production:
        raise FileNotFoundError(
            f"RS256 signing keypair not found at {private} and {public}. "
            "Generate it before deploying:\n"
            "  openssl genrsa -out secrets/jwt_private.pem 2048\n"
            "  openssl rsa -in secrets/jwt_private.pem -pubout "
            "-out secrets/jwt_public.pem"
        )

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    logging.getLogger("resx.security").warning(
        "no RS256 keypair at %s; generating a development keypair. Existing "
        "sessions will not verify against it.",
        private,
    )
    private.parent.mkdir(parents=True, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public.write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private, public


@lru_cache
def _private_key() -> bytes:
    return ensure_signing_keypair()[0].read_bytes()


@lru_cache
def _public_key() -> bytes:
    return ensure_signing_keypair()[1].read_bytes()


def issue_access_token(
    *,
    user_id: str | uuid.UUID,
    workspace_id: str | uuid.UUID,
    role: str,
    settings: Settings | None = None,
) -> tuple[str, datetime]:
    """Mint a short-lived access token.

    Ids are accepted as strings as well as UUIDs: RESX uses readable prefixed
    ids (`usr_`, `ws_`) so that a token, a log line and an audit row can be
    matched up by eye during an incident.
    """
    s = settings or get_settings()
    now = datetime.now(timezone.utc)
    expires = now + timedelta(seconds=s.access_token_ttl_seconds)

    claims: dict[str, Any] = {
        "sub": str(user_id),
        "workspace_id": str(workspace_id),
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int(expires.timestamp()),
        "jti": secrets.token_urlsafe(16),
        "iss": s.jwt_issuer,
        "aud": s.jwt_audience,
        "typ": "access",
    }
    token = jwt.encode(claims, _private_key(), algorithm=s.jwt_algorithm)
    return token, expires


def decode_access_token(token: str, settings: Settings | None = None) -> dict[str, Any]:
    """Verify and decode. Raises `jwt.PyJWTError` on any failure.

    `algorithms` is pinned to a single value to prevent algorithm-confusion
    attacks, and both `aud` and `iss` are verified — an unverified audience is
    how a token minted for one service gets replayed against another.
    """
    s = settings or get_settings()
    return jwt.decode(
        token,
        _public_key(),
        algorithms=[s.jwt_algorithm],
        audience=s.jwt_audience,
        issuer=s.jwt_issuer,
        options={"require": ["exp", "iat", "sub", "aud", "iss"]},
    )


# --------------------------------------------------------------------------- #
# Refresh tokens
# --------------------------------------------------------------------------- #


def new_refresh_token() -> tuple[str, bytes]:
    """Return `(plaintext, sha256_digest)`.

    Only the digest is persisted, so a database dump does not yield usable
    tokens. 256 bits of CSPRNG output means a fast hash is the right choice
    here — see the module docstring.
    """
    plaintext = secrets.token_urlsafe(32)
    return plaintext, hash_refresh_token(plaintext)


def hash_refresh_token(plaintext: str) -> bytes:
    return hashlib.sha256(plaintext.encode("utf-8")).digest()


def constant_time_equals(a: str | bytes, b: str | bytes) -> bool:
    """Compare secrets without leaking their contents through timing."""
    if isinstance(a, str):
        a = a.encode("utf-8")
    if isinstance(b, str):
        b = b.encode("utf-8")
    return hmac.compare_digest(a, b)


# --------------------------------------------------------------------------- #
# Signed URLs (report and artifact downloads)
# --------------------------------------------------------------------------- #


def sign_payload(payload: str, key: str) -> str:
    return hmac.new(key.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_signature(payload: str, signature: str, key: str) -> bool:
    return constant_time_equals(sign_payload(payload, key), signature)
