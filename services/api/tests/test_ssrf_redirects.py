"""A redirect is a second request, and it gets a second check.

`assert_url_is_safe` resolves the hostname and rejects private, loopback and
link-local addresses — 169.254.169.254, the cloud metadata endpoint, being the
one that turns an SSRF into stolen credentials. That check runs on the URL the
caller supplied.

Which is exactly why following redirects automatically would defeat it. An
attacker controls a public HTTPS host that passes every check, and answers 302
to `http://169.254.169.254/latest/meta-data/`. httpx would follow it happily:
the guard already ran, on the URL that was safe.

So redirects are not followed. The hop is surfaced as an error naming the
target, and reaching it means passing it through the guard again as a fresh
fetch. That is the property asserted here — previously the behaviour was
correct and untested, which is one edit away from not being the behaviour.
"""

from __future__ import annotations

import pytest

from app.agents.tools import ToolError, assert_url_is_safe


class FakeResponse:
    def __init__(self, status_code: int, location: str = "", text: str = "") -> None:
        self.status_code = status_code
        self.headers = {"location": location} if location else {}
        self.text = text


@pytest.fixture
def belt():
    from app.agents.schemas import AgentName
    from app.agents.tools import Toolbelt
    from app.rag.embeddings import HashingEmbedder

    return Toolbelt(
        AgentName.NEWS,
        store=None,
        workspace_id="ws_x",
        embedder=HashingEmbedder(dimensions=32),
    )


@pytest.mark.parametrize("status_code", [301, 302, 303, 307, 308])
def test_no_redirect_status_is_followed(belt, monkeypatch, status_code: int) -> None:
    """Every redirect status, not just 302. A guard that catches four of five
    is not a guard."""
    import httpx

    monkeypatch.setattr(
        httpx,
        "get",
        lambda *a, **k: FakeResponse(status_code, "https://evil.example/next"),
    )
    with pytest.raises(ToolError) as caught:
        belt._fetch_url("https://example.com/start")
    assert "redirect" in str(caught.value).lower()


def test_the_client_is_told_not_to_follow_redirects(belt, monkeypatch) -> None:
    """The status check above is the second line. The first is httpx itself
    never following one — without it the guard would run on a URL that had
    already been left behind."""
    import httpx

    seen: dict[str, object] = {}

    def capture(*args, **kwargs):
        seen.update(kwargs)
        return FakeResponse(200, text="fine")

    monkeypatch.setattr(httpx, "get", capture)
    belt._fetch_url("https://example.com/page")
    assert seen.get("follow_redirects") is False


def test_the_redirect_target_is_named(belt, monkeypatch) -> None:
    """A caller that cannot see where it was being sent cannot decide whether
    to follow it deliberately."""
    import httpx

    monkeypatch.setattr(
        httpx,
        "get",
        lambda *a, **k: FakeResponse(302, "https://elsewhere.example/doc"),
    )
    with pytest.raises(ToolError) as caught:
        belt._fetch_url("https://example.com/start")
    assert "elsewhere.example" in str(caught.value)


def test_a_redirect_to_metadata_is_not_reachable_by_re_validating(belt) -> None:
    """The whole point. Following the hop means passing it through the guard
    again, and the guard refuses it — so the attack has nowhere to go."""
    with pytest.raises(ToolError):
        assert_url_is_safe("https://169.254.169.254/latest/meta-data/", frozenset())


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1/admin",
        "https://localhost/internal",
        "https://10.0.0.5/secrets",
        "https://192.168.1.1/",
        "https://[::1]/",
    ],
)
def test_a_redirect_target_inside_the_network_is_refused(url: str) -> None:
    with pytest.raises(ToolError):
        assert_url_is_safe(url, frozenset())


def test_a_non_https_redirect_target_is_refused() -> None:
    """Downgrading to http is how a redirect reaches a service that never
    speaks TLS — which is most internal ones."""
    with pytest.raises(ToolError, match="https"):
        assert_url_is_safe("http://example.com/plain", frozenset())


def test_the_allowlist_still_applies_to_a_redirect_target() -> None:
    """A hop to a host that is public and perfectly safe is still egress to a
    place this deployment did not agree to talk to."""
    with pytest.raises(ToolError, match="allowlist"):
        assert_url_is_safe("https://elsewhere.example/x", frozenset({"api.tavily.com"}))


def test_an_ordinary_response_is_returned_with_its_provenance(belt, monkeypatch) -> None:
    """The non-redirect path has to keep working, and the body has to arrive
    wrapped as untrusted with its source attached."""
    import httpx

    monkeypatch.setattr(
        httpx, "get", lambda *a, **k: FakeResponse(200, text="quarterly revenue rose")
    )
    out = belt._fetch_url("https://example.com/report")
    assert "quarterly revenue rose" in out
    assert "example.com" in out
