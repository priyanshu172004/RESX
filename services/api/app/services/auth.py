"""Authentication and session management.

The design decisions worth stating, because in each case the obvious
alternative is a real vulnerability rather than a style preference:

  * **Argon2id, not bcrypt or PBKDF2.** Memory-hardness is the point. The cost
    parameters are enforced as floors in `Settings`, not merely documented.
  * **Login is constant-work whether the account exists or not.** A missing
    account still runs a password verification against a dummy hash, so
    response timing does not enumerate registered emails.
  * **Refresh tokens rotate, and reuse revokes the family.** A refresh token
    presented twice means the token was captured. Revoking only the replayed
    token leaves the attacker's copy of the *next* one working, so the whole
    chain dies instead.
  * **The workspace comes from the verified token, never the request.** A
    client-supplied workspace id is a cross-tenant read with extra steps.
  * **Lockout counts per account, not per IP.** An attacker rotates addresses
    for free; the thing being protected is the account.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

from app.core import security
from app.core.config import Settings

#: Deliberately permissive. Email validity is decided by delivery, not by a
#: regex, and an over-strict pattern rejects real addresses (plus-addressing,
#: long TLDs, unicode domains). This rejects only what is certainly not an
#: address.
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+\.[^@\s]+$")

#: A valid Argon2id hash of a random string, used to keep the failure path's
#: work equal to the success path's. Generated once at import.
_DUMMY_HASH: str | None = None

ROLES = ("owner", "admin", "analyst", "viewer")

#: Ordered by privilege. `require_role` compares positions rather than
#: membership, so a new role slots into the hierarchy in one place.
ROLE_RANK = {role: i for i, role in enumerate(reversed(ROLES))}


class AuthError(Exception):
    """Base for anything that should become a 4xx rather than a 500."""

    status_code = 400

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code


class InvalidCredentialsError(AuthError):
    status_code = 401


class AccountLockedError(AuthError):
    status_code = 423


class EmailInUseError(AuthError):
    status_code = 409


class WeakPasswordError(AuthError):
    status_code = 422


class SessionExpiredError(AuthError):
    status_code = 401


class TokenReuseError(AuthError):
    """A refresh token was presented twice. Treated as compromise."""

    status_code = 401


@dataclass(frozen=True, slots=True)
class Session:
    access_token: str
    expires_at: float
    refresh_token: str
    user: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "access_token": self.access_token,
            "token_type": "Bearer",
            "expires_at": self.expires_at,
            "user": self.user,
        }


def _dummy_hash() -> str:
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = security.hash_password(uuid.uuid4().hex)
    return _DUMMY_HASH


class TotpRequiredError(AuthError):
    """The password was right and a code is still needed.

    A distinct type because the client has to tell these apart: "wrong
    password" sends the user to reset it, while this one needs a six-digit
    prompt. Collapsing them into one 401 is how a working login looks broken.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=401)


def public_user(user: dict[str, Any]) -> dict[str, Any]:
    """Strip everything that must never reach a response body."""
    # `recovery_codes` holds password hashes and leaks how many remain, which
    # tells an attacker how close an account is to being locked out.
    hidden = {
        "password_hash",
        "totp_secret",
        "pending_totp_secret",
        "recovery_codes",
        "failed_logins",
        "locked_until",
    }
    public = {k: v for k, v in user.items() if k not in hidden}
    # A boolean, not the secret. The UI has to be able to show whether
    # two-factor is enabled without ever being handed the means to generate a
    # code.
    public["totp_enabled"] = bool(user.get("totp_secret"))
    return public


#: How many recovery codes an enrolment hands out. Ten is enough that losing a
#: phone is an inconvenience rather than a lockout, and few enough that a user
#: will actually keep them somewhere.
RECOVERY_CODE_COUNT = 10

#: Roles an invitation may grant. `owner` is absent deliberately: ownership
#: transfers as its own authorised action, not by someone sending a link.
INVITABLE_ROLES = frozenset({"admin", "analyst", "viewer"})

#: How long an invitation stays valid. Seven days is long enough to survive a
#: holiday and short enough that a link forwarded into a group chat two months
#: ago is no longer a way in.
INVITE_TTL_SECONDS = 7 * 24 * 60 * 60


class AuthService:
    def __init__(self, *, store: Any, settings: Settings) -> None:
        self.store = store
        self.settings = settings

    # -- registration -----------------------------------------------------

    def validate_password(self, password: str) -> None:
        """Length only, on purpose.

        Composition rules ("one uppercase, one symbol") measurably push users
        towards predictable patterns like `Password1!`, and the real risk is a
        password reused from a breached site. Length is the requirement that
        actually correlates with strength.
        """
        if len(password) < self.settings.password_min_length:
            raise WeakPasswordError(
                f"password must be at least {self.settings.password_min_length} characters"
            )
        if len(password) > 1024:
            # An unbounded password is a denial-of-service vector: Argon2id
            # hashes whatever it is given, and it is designed to be slow.
            raise WeakPasswordError("password must be at most 1024 characters")

    def register(
        self,
        *,
        email: str,
        password: str,
        name: str,
        workspace_name: str | None = None,
        ip: str | None = None,
    ) -> Session:
        email = email.strip().lower()
        if not EMAIL_RE.match(email):
            raise AuthError("that does not look like an email address", status_code=422)
        self.validate_password(password)
        if not name.strip():
            raise AuthError("a name is required", status_code=422)

        if self.store.get_user_by_email(email) is not None:
            raise EmailInUseError("that email is already registered")

        # Each registration creates its own workspace. Joining an existing one
        # is an invitation flow, which is a separate, authorised action.
        workspace_id = f"ws_{uuid.uuid4().hex[:12]}"
        try:
            user = self.store.create_user(
                email=email,
                password_hash=security.hash_password(password),
                name=name,
                workspace_id=workspace_id,
                role="owner",
                verified=self.settings.allow_unverified_login,
            )
        except Exception as exc:
            # A unique-index violation here means a concurrent registration won
            # the race. Report it as the conflict it is.
            if "already registered" in str(exc):
                raise EmailInUseError("that email is already registered") from exc
            raise

        self.store.audit(
            workspace_id=workspace_id,
            actor=user["user_id"],
            action="auth.register",
            detail={"email": email, "workspace_name": workspace_name or name},
            ip=ip,
        )
        return self._issue(user, ip=ip)

    # -- login ------------------------------------------------------------

    def login(
        self,
        *,
        email: str,
        password: str,
        ip: str | None = None,
        totp_code: str | None = None,
    ) -> Session:
        email = email.strip().lower()
        user = self.store.get_user_by_email(email)

        # Verify against a dummy hash when the account does not exist, so the
        # work done is the same either way and timing cannot enumerate users.
        stored = user["password_hash"] if user else _dummy_hash()

        locked_until = (user or {}).get("locked_until")
        if user and locked_until and float(locked_until) > time.time():
            remaining = int(float(locked_until) - time.time())
            raise AccountLockedError(
                f"too many failed attempts; try again in {remaining} seconds"
            )

        ok = security.verify_password(password, stored)
        if not ok or user is None:
            if user is not None:
                self._record_failure(user, ip=ip)
            raise InvalidCredentialsError("email or password is incorrect")

        if not user.get("verified") and not self.settings.allow_unverified_login:
            raise AuthError("verify your email address first", status_code=403)

        if user.get("totp_secret"):
            # Checked *after* the password, so an attacker cannot use this
            # endpoint to discover which accounts have two-factor enabled
            # without the password in hand.
            if not totp_code:
                raise TotpRequiredError("this account has two-factor enabled; send totp_code")
            if not self.verify_totp(user=user, code=totp_code):
                # Counted as a failed attempt. Otherwise the lockout protects
                # the password and leaves the second factor open to unlimited
                # guessing, which is the weaker of the two at six digits.
                self._record_failure(user, ip=ip)
                raise InvalidCredentialsError("that code is not valid")

        # Rehash transparently when the cost parameters have been raised, so
        # existing accounts benefit from a hardening change on next login.
        if security.needs_rehash(stored):
            self.store.update_user(
                user_id=user["user_id"],
                password_hash=security.hash_password(password),
            )

        self.store.update_user(
            user_id=user["user_id"],
            failed_logins=0,
            locked_until=None,
            last_login_at=time.time(),
        )
        self.store.audit(
            workspace_id=user["workspace_id"],
            actor=user["user_id"],
            action="auth.login",
            ip=ip,
        )
        return self._issue(user, ip=ip)

    def _record_failure(self, user: dict[str, Any], *, ip: str | None) -> None:
        attempts = int(user.get("failed_logins") or 0) + 1
        fields: dict[str, Any] = {"failed_logins": attempts}
        if attempts >= self.settings.login_max_attempts:
            fields["locked_until"] = time.time() + self.settings.login_lockout_seconds
            fields["failed_logins"] = 0
        self.store.update_user(user_id=user["user_id"], **fields)
        self.store.audit(
            workspace_id=user["workspace_id"],
            actor=user["user_id"],
            action="auth.login_failed",
            detail={"attempts": attempts, "locked": "locked_until" in fields},
            ip=ip,
        )

    # -- sessions ---------------------------------------------------------

    def _issue(
        self,
        user: dict[str, Any],
        *,
        family_id: str | None = None,
        parent_hash: bytes | None = None,
        ip: str | None = None,
    ) -> Session:
        access, expires = security.issue_access_token(
            user_id=user["user_id"],
            workspace_id=user["workspace_id"],
            role=user.get("role", "viewer"),
            settings=self.settings,
        )
        plaintext, digest = security.new_refresh_token()
        self.store.store_refresh_token(
            token_hash=digest,
            user_id=user["user_id"],
            family_id=family_id or f"fam_{uuid.uuid4().hex[:12]}",
            parent_hash=parent_hash,
            expires_at=time.time() + self.settings.refresh_token_ttl_seconds,
        )
        return Session(
            access_token=access,
            expires_at=expires.timestamp(),
            refresh_token=plaintext,
            user=public_user(user),
        )

    def refresh(self, *, refresh_token: str, ip: str | None = None) -> Session:
        """Rotate a refresh token, revoking the family if it was replayed."""
        digest = security.hash_refresh_token(refresh_token)
        record = self.store.get_refresh_token(digest)
        if record is None:
            raise SessionExpiredError("that session is not recognised; sign in again")

        if record.get("revoked"):
            raise SessionExpiredError("that session has been revoked; sign in again")

        if record.get("used_at"):
            # Reuse. The legitimate client rotated this token already, so
            # whoever presented it again has a captured copy. Kill the chain.
            revoked = self.store.revoke_token_family(record["family_id"])
            user = self.store.get_user(record["user_id"])
            if user:
                self.store.audit(
                    workspace_id=user["workspace_id"],
                    actor=user["user_id"],
                    action="auth.refresh_reuse_detected",
                    detail={"family_id": record["family_id"], "revoked": revoked},
                    ip=ip,
                )
            raise TokenReuseError(
                "this session token was already used. For your security every "
                "session in this chain has been signed out."
            )

        if float(record["expires_at"]) < time.time():
            raise SessionExpiredError("that session has expired; sign in again")

        user = self.store.get_user(record["user_id"])
        if user is None:
            raise SessionExpiredError("that account no longer exists")

        self.store.mark_refresh_token_used(digest)
        return self._issue(user, family_id=record["family_id"], parent_hash=digest, ip=ip)

    def logout(self, *, refresh_token: str | None, ip: str | None = None) -> None:
        """Revoke the presented session's whole family.

        Family-wide rather than token-only: a user clicking "sign out" means
        this device's chain should stop working, and the chain is the family.
        """
        if not refresh_token:
            return
        record = self.store.get_refresh_token(security.hash_refresh_token(refresh_token))
        if record is None:
            return
        self.store.revoke_token_family(record["family_id"])
        user = self.store.get_user(record["user_id"])
        if user:
            self.store.audit(
                workspace_id=user["workspace_id"],
                actor=user["user_id"],
                action="auth.logout",
                ip=ip,
            )

    # -- password change --------------------------------------------------

    # -- invitations ------------------------------------------------------

    def invite(
        self, *, workspace_id: str, inviter_id: str, email: str, role: str
    ) -> dict[str, Any]:
        """Create an invitation and return the link, once.

        The plaintext token is returned here and never stored — only its
        SHA-256 digest is — so a database dump does not yield working invite
        links. That also means it cannot be shown again, which is why the
        response says so.

        No email is sent. There is no mail provider configured, and inventing
        one would mean an invitation that silently goes nowhere. The inviter
        gets the link and passes it on however they already talk to the person,
        which is honest about what the system can actually do.
        """
        email = email.strip().lower()
        if not EMAIL_RE.match(email):
            raise AuthError("that does not look like an email address", status_code=422)
        if role not in INVITABLE_ROLES:
            raise AuthError(
                f"role must be one of {', '.join(sorted(INVITABLE_ROLES))}; "
                f"ownership transfers deliberately, not by invitation",
                status_code=422,
            )

        existing = self.store.get_user_by_email(email)
        if existing is not None:
            if existing.get("workspace_id") == workspace_id:
                raise AuthError("that person is already in this workspace", status_code=409)
            raise AuthError(
                "that email already has an account in another workspace; a "
                "user belongs to one workspace",
                status_code=409,
            )

        plaintext, digest = security.new_refresh_token()
        invite_id = f"inv_{uuid.uuid4().hex[:12]}"
        expires_at = time.time() + INVITE_TTL_SECONDS
        self.store.create_invitation(
            invite_id=invite_id,
            workspace_id=workspace_id,
            email=email,
            role=role,
            token_hash=digest,
            invited_by=inviter_id,
            expires_at=expires_at,
        )
        self.store.audit(
            workspace_id=workspace_id,
            actor=inviter_id,
            action="workspace.invite",
            target=email,
            detail={"role": role},
        )
        return {
            "invite_id": invite_id,
            "email": email,
            "role": role,
            "token": plaintext,
            "expires_at": expires_at,
        }

    def accept_invitation(
        self, *, token: str, password: str, name: str, ip: str | None = None
    ) -> Session:
        """Join a workspace using an invitation link.

        The email is taken from the invitation, not from the request. Letting
        the caller supply it would mean an invitation addressed to one person
        could be redeemed by another — the link is the credential, and the
        address it was issued for is the whole point of it.
        """
        invitation = self.store.get_invitation(security.hash_refresh_token(token))
        # One message for every rejection: "expired" and "already used" and
        # "no such invitation" together tell someone probing which tokens once
        # existed.
        refusal = AuthError(
            "that invitation is not valid — it may have expired, already been "
            "used, or been revoked",
            status_code=400,
        )
        if invitation is None:
            raise refusal
        if invitation.get("accepted_at"):
            raise refusal
        if float(invitation.get("expires_at") or 0) < time.time():
            raise refusal

        self.validate_password(password)
        if not name.strip():
            raise AuthError("a name is required", status_code=422)

        email = str(invitation["email"])
        if self.store.get_user_by_email(email) is not None:
            raise EmailInUseError("that email is already registered")

        user = self.store.create_user(
            email=email,
            password_hash=security.hash_password(password),
            name=name.strip(),
            # The existing workspace, which is the entire point.
            workspace_id=str(invitation["workspace_id"]),
            role=str(invitation["role"]),
            verified=True,
        )
        # Marked immediately, and the row is the only record of the token, so a
        # second use finds it already accepted and is refused.
        self.store.mark_invitation_accepted(invite_id=str(invitation["invite_id"]))
        self.store.audit(
            workspace_id=str(invitation["workspace_id"]),
            actor=user["user_id"],
            action="workspace.invite_accepted",
            ip=ip,
        )
        return self._issue(user, ip=ip)

    # -- two-factor -------------------------------------------------------

    def begin_totp_enrolment(self, *, user_id: str) -> dict[str, str]:
        """Generate a secret and the URI an authenticator app scans.

        The secret is returned but **not** stored as active. Enrolment that
        activated on generation would lock out anyone whose app failed to scan
        the code — they would hold an account demanding a code nothing could
        produce, and no way back in. It becomes real only when they prove they
        can generate a code from it, in `confirm_totp_enrolment`.
        """
        import pyotp

        user = self.store.get_user(user_id=user_id)
        if user is None:
            raise AuthError("no such user", status_code=404)
        if user.get("totp_secret"):
            raise AuthError(
                "two-factor is already enabled; disable it before enrolling again",
                status_code=409,
            )

        secret = pyotp.random_base32()
        # Stored in a column of its own, apart from the active secret.
        #
        # An in-process dictionary was the first attempt and it does not work:
        # `AuthService` is built per request, so the secret generated by the
        # enrol call had already been discarded by the time the confirm call
        # arrived, and every enrolment failed with "no enrolment in progress".
        #
        # Keeping it out of `totp_secret` is what makes it pending: login reads
        # only the active column, so an abandoned enrolment can never start
        # demanding codes.
        self.store.update_user(user_id=user_id, pending_totp_secret=secret)

        uri = pyotp.TOTP(secret).provisioning_uri(
            name=str(user["email"]),
            issuer_name=self.settings.app_name,
        )
        return {"secret": secret, "otpauth_uri": uri}

    def confirm_totp_enrolment(self, *, user_id: str, code: str) -> list[str]:
        """Activate two-factor once the user proves they can generate a code.

        Returns single-use recovery codes. They are the answer to the failure
        mode that makes people avoid 2FA at all: a lost phone is otherwise a
        lost account. Shown once and stored hashed, for the same reason
        passwords are.
        """
        import pyotp

        user = self.store.get_user(user_id=user_id)
        if user is None:
            raise AuthError("no such user", status_code=404)
        secret = user.get("pending_totp_secret")
        if not secret:
            raise AuthError(
                "no enrolment in progress; request a new secret first",
                status_code=409,
            )
        if not pyotp.TOTP(str(secret)).verify(code, valid_window=1):
            # One step of tolerance either side, because phone clocks drift and
            # a user typing a code as it rolls over is not an attacker.
            raise AuthError("that code is not valid", status_code=400)

        codes = [security.new_recovery_code() for _ in range(RECOVERY_CODE_COUNT)]
        self.store.update_user(
            user_id=user_id,
            totp_secret=secret,
            pending_totp_secret=None,
            recovery_codes=json.dumps([security.hash_password(c) for c in codes]),
        )

        self.store.audit(
            workspace_id=user.get("workspace_id", ""),
            actor=user_id,
            action="auth.totp_enabled",
        )
        return codes

    def disable_totp(self, *, user_id: str, password: str) -> None:
        """Turn two-factor off, re-proving the password first.

        The password is required because an unlocked laptop is the ordinary way
        someone else reaches this endpoint, and removing 2FA is exactly what
        they would want to do with it.
        """
        user = self.store.get_user(user_id=user_id)
        if user is None:
            raise AuthError("no such user", status_code=404)
        if not security.verify_password(password, user["password_hash"]):
            raise InvalidCredentialsError("password is incorrect")

        self.store.update_user(
            user_id=user_id,
            totp_secret=None,
            pending_totp_secret=None,
            recovery_codes=None,
        )
        self.store.audit(
            workspace_id=user["workspace_id"],
            actor=user_id,
            action="auth.totp_disabled",
        )

    def verify_totp(self, *, user: dict[str, Any], code: str) -> bool:
        """Check a code, accepting a recovery code in its place.

        A used recovery code is removed, not merely marked — a one-time code
        that still exists after use is a permanent credential in a list the
        user believes is shrinking.
        """
        import pyotp

        secret = user.get("totp_secret")
        if not secret:
            return True  # not enrolled; nothing to check

        if pyotp.TOTP(str(secret)).verify(code, valid_window=1):
            return True

        stored = user.get("recovery_codes")
        if not stored:
            return False
        remaining = list(json.loads(stored))
        for hashed in remaining:
            if security.verify_password(code, hashed):
                remaining.remove(hashed)
                self.store.update_user(
                    user_id=user["user_id"], recovery_codes=json.dumps(remaining)
                )
                return True
        return False

    def change_password(
        self, *, user_id: str, current: str, replacement: str, ip: str | None = None
    ) -> None:
        user = self.store.get_user(user_id)
        if user is None:
            raise InvalidCredentialsError("account not found")
        if not security.verify_password(current, user["password_hash"]):
            raise InvalidCredentialsError("current password is incorrect")
        self.validate_password(replacement)
        self.store.update_user(
            user_id=user_id, password_hash=security.hash_password(replacement)
        )
        self.store.audit(
            workspace_id=user["workspace_id"],
            actor=user_id,
            action="auth.password_changed",
            ip=ip,
        )


def role_allows(actual: str, required: str) -> bool:
    """Whether `actual` is at least as privileged as `required`."""
    return ROLE_RANK.get(actual, -1) >= ROLE_RANK.get(required, 99)
