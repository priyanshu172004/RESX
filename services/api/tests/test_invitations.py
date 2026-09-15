"""Joining an existing workspace.

Registration creates a workspace, and that was the only way to get an account —
so a second person could never reach the first person's corpus at all. The
whole tenancy model existed to let people share a workspace and nothing could
put two people in one.

Three decisions are load-bearing:

  * **The token is stored hashed.** Like a refresh token, this row grants
    membership of a workspace, so a database dump must not yield working
    invite links.
  * **The email comes from the invitation, not the request.** Otherwise a link
    issued to one person is redeemable by anyone who obtains it under whatever
    address they like — the address it was issued for is the point of it.
  * **Every rejection reads the same.** Distinguishing "expired" from "already
    used" from "no such invitation" tells someone probing which tokens once
    existed.

No email is sent. There is no mail provider configured, and inventing one would
produce invitations that silently go nowhere.
"""

from __future__ import annotations

from typing import Any

import pytest

ACCEPTOR = {
    "token": "",
    "password": "a-long-enough-password-for-them",
    "name": "Second Person",
}


def invite(client: Any, email: str = "colleague@example.com", role: str = "analyst") -> Any:
    return client.post("/api/v1/auth/invitations", json={"email": email, "role": role})


def test_an_invitation_returns_a_link_once(client: Any) -> None:
    response = invite(client)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["token"]
    assert body["invite_url"] and "token=" in body["invite_url"]
    # Said out loud, because discovering it by coming back later is the wrong
    # time to find out.
    assert "once" in body["note"].lower()


def test_the_token_is_never_listed_afterwards(client: Any) -> None:
    """The listing is for managing invitations, not for recovering them. A
    token retrievable from a list is a token a database dump yields."""
    token = invite(client).json()["token"]
    listed = client.get("/api/v1/auth/invitations").json()
    assert listed
    assert token not in str(listed)
    assert "token_hash" not in str(listed)


def test_accepting_puts_them_in_the_same_workspace(client: Any) -> None:
    """The entire point. Two accounts, one corpus."""
    mine = client.get("/api/v1/auth/me").json()["workspace"]["workspace_id"]
    token = invite(client).json()["token"]

    accepted = client.post("/api/v1/auth/invitations/accept", json={**ACCEPTOR, "token": token})
    assert accepted.status_code == 201, accepted.text

    client.headers["Authorization"] = f"Bearer {accepted.json()['access_token']}"
    theirs = client.get("/api/v1/auth/me").json()["workspace"]["workspace_id"]
    assert theirs == mine


def test_they_can_read_the_shared_corpus(client: Any) -> None:
    import io

    client.post(
        "/api/v1/documents",
        files={"file": ("shared.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")},
    )
    token = invite(client).json()["token"]
    accepted = client.post("/api/v1/auth/invitations/accept", json={**ACCEPTOR, "token": token})
    client.headers["Authorization"] = f"Bearer {accepted.json()['access_token']}"

    assert len(client.get("/api/v1/documents").json()) == 1


def test_the_invited_role_is_applied(client: Any) -> None:
    token = invite(client, role="viewer").json()["token"]
    accepted = client.post("/api/v1/auth/invitations/accept", json={**ACCEPTOR, "token": token})
    assert accepted.json()["user"]["role"] == "viewer"


def test_owner_cannot_be_invited(client: Any) -> None:
    """Ownership transfers as its own authorised action, not by someone
    sending a link."""
    response = invite(client, role="owner")
    assert response.status_code == 422
    assert "ownership transfers" in str(response.json())


def test_the_email_comes_from_the_invitation_not_the_request(client: Any) -> None:
    """A link issued to one address must not be redeemable under another —
    otherwise the address it was issued for means nothing."""
    token = invite(client, email="intended@example.com").json()["token"]
    accepted = client.post(
        "/api/v1/auth/invitations/accept",
        json={**ACCEPTOR, "token": token},
    )
    assert accepted.status_code == 201
    assert accepted.json()["user"]["email"] == "intended@example.com"


def test_an_invitation_works_only_once(client: Any) -> None:
    """A link that still works after it was used is a permanent back door."""
    token = invite(client).json()["token"]
    first = client.post("/api/v1/auth/invitations/accept", json={**ACCEPTOR, "token": token})
    assert first.status_code == 201

    again = client.post(
        "/api/v1/auth/invitations/accept",
        json={
            "token": token,
            "password": "another-long-enough-password",
            "name": "Third Person",
        },
    )
    assert again.status_code == 400


def test_a_made_up_token_is_refused(client: Any) -> None:
    response = client.post(
        "/api/v1/auth/invitations/accept",
        json={**ACCEPTOR, "token": "x" * 40},
    )
    assert response.status_code == 400


def test_every_rejection_reads_the_same(client: Any) -> None:
    """Distinguishing "expired" from "already used" from "never existed" tells
    someone probing which tokens once existed."""
    token = invite(client).json()["token"]
    client.post("/api/v1/auth/invitations/accept", json={**ACCEPTOR, "token": token})

    used = client.post(
        "/api/v1/auth/invitations/accept",
        json={"token": token, "password": "another-long-password", "name": "X"},
    )
    unknown = client.post(
        "/api/v1/auth/invitations/accept",
        json={"token": "y" * 40, "password": "another-long-password", "name": "X"},
    )
    assert used.json()["error"]["message"] == unknown.json()["error"]["message"]


def test_an_expired_invitation_is_refused(client: Any, monkeypatch: Any) -> None:
    from app.services import auth as auth_service

    monkeypatch.setattr(auth_service, "INVITE_TTL_SECONDS", -1)
    token = invite(client).json()["token"]
    response = client.post("/api/v1/auth/invitations/accept", json={**ACCEPTOR, "token": token})
    assert response.status_code == 400


def test_revoking_stops_the_link_working(client: Any) -> None:
    body = invite(client).json()
    assert client.delete(f"/api/v1/auth/invitations/{body['invite_id']}").status_code == 204
    response = client.post(
        "/api/v1/auth/invitations/accept", json={**ACCEPTOR, "token": body["token"]}
    )
    assert response.status_code == 400


def test_an_accepted_invitation_cannot_be_revoked(client: Any) -> None:
    """Deleting the record of who joined, and how, is not a revocation — it is
    losing the audit trail."""
    body = invite(client).json()
    client.post("/api/v1/auth/invitations/accept", json={**ACCEPTOR, "token": body["token"]})
    assert client.delete(f"/api/v1/auth/invitations/{body['invite_id']}").status_code == 404


def test_inviting_someone_who_already_has_an_account_is_refused(client: Any) -> None:
    token = invite(client).json()["token"]
    client.post("/api/v1/auth/invitations/accept", json={**ACCEPTOR, "token": token})

    response = invite(client, email="colleague@example.com")
    assert response.status_code == 409


@pytest.mark.parametrize("email", ["not-an-email", "@example.com", "a b@example.com"])
def test_a_malformed_address_is_refused(client: Any, email: str) -> None:
    assert invite(client, email=email).status_code == 422


def test_another_workspace_cannot_see_these_invitations(client: Any) -> None:
    invite(client)
    other = client.post(
        "/api/v1/auth/register",
        json={
            "email": "outsider@example.com",
            "password": "a-long-enough-password",
            "name": "Outsider",
        },
    )
    client.headers["Authorization"] = f"Bearer {other.json()['access_token']}"
    assert client.get("/api/v1/auth/invitations").json() == []
