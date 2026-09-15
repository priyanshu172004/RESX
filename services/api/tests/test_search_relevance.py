"""Web search relevance: grounding proves provenance, not relevance.

The bug these lock down: the provider scores every result and the tool ignored
the score, so research answers arrived as one good finding plus several
unrelated ones.

On "What was Apple's revenue in Q1 2025?" the top three results scored 0.93-0.96
and answered the question. Two more scored 0.25 and their entire content was
store navigation -- "Shop iPad / iPad Accessories / Apple Trade In" -- handed to
the agent with exactly the same standing as the earnings release. The same page
was also returned twice.

The grounding gate cannot catch this. "Shop iPad" *is* verbatim in the source,
so a claim built on it is perfectly traceable and utterly useless. Provenance
and relevance are different properties and only one of them was being checked.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.agents.schemas import AgentName
from app.agents.tools import Toolbelt


def belt_with(results: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch) -> Toolbelt:
    """A Toolbelt whose search transport returns exactly `results`."""
    import httpx

    class _Response:
        status_code = 200

        @staticmethod
        def json() -> dict[str, Any]:
            return {"results": results}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Response())
    return Toolbelt(
        AgentName.NEWS,
        store=None,  # type: ignore[arg-type]
        workspace_id="ws",
        embedder=None,  # type: ignore[arg-type]
        tavily_api_key="tvly_test",
    )


def result(
    *, score: float, url: str, content: str = "Revenue was $143.8 billion.", title: str = "T"
) -> dict[str, Any]:
    return {"score": score, "url": url, "content": content, "title": title}


# The two clusters actually observed, reproduced.
APPLE_RESULTS = [
    result(
        score=0.96, url="https://sixcolors.com/q1", content="Apple reported record revenue."
    ),
    result(score=0.95, url="https://cnbc.com/q1", content="Services revenue rose to $26.3bn."),
    result(
        score=0.93, url="https://stockanalysis.com/aapl", content="Annual revenue was $416.16B."
    ),
    result(
        score=0.25,
        url="https://apple.com/store",
        content="Shop iPad iPad Accessories Apple Trade In",
    ),
    result(
        score=0.25,
        url="https://apple.com/store",
        content="Shop iPad iPad Accessories Apple Trade In",
    ),
]


def test_a_low_scoring_result_never_reaches_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact regression: store navigation presented as evidence."""
    belt = belt_with(APPLE_RESULTS, monkeypatch)
    out = belt._web_search("What was Apple's revenue in Q1 2025?")
    assert "Shop iPad" not in out


def test_the_same_url_is_not_presented_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two copies of one source read to a model as two independent sources
    agreeing, which is the corroboration the citation rules exist to
    establish."""
    duplicated = [
        result(score=0.9, url="https://example.com/a"),
        result(score=0.9, url="https://example.com/a"),
        result(score=0.8, url="https://example.com/b"),
    ]
    belt = belt_with(duplicated, monkeypatch)
    out = belt._web_search("q")
    assert out.count("https://example.com/a") == 1
    assert "1 duplicate result(s) discarded" in out


def test_the_relevant_results_are_all_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Filtering must not become "return almost nothing"."""
    belt = belt_with(APPLE_RESULTS, monkeypatch)
    out = belt._web_search("q")
    assert out.count("<untrusted_document_content") == 3


def test_the_agent_is_told_results_were_discarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Silence reads as "this is everything the web has on the subject"."""
    belt = belt_with(APPLE_RESULTS, monkeypatch)
    out = belt._web_search("q")
    assert "not relevant to the query" in out


def test_results_arrive_most_relevant_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """So that if the size ceiling forces anything out, it is the weakest
    result rather than whichever the provider listed last."""
    shuffled = [
        result(score=0.6, url="https://example.com/weak", content="Weak but relevant."),
        result(score=0.99, url="https://example.com/strong", content="Directly answers it."),
        result(score=0.8, url="https://example.com/mid", content="Partly relevant."),
    ]
    belt = belt_with(shuffled, monkeypatch)
    out = belt._web_search("q")
    assert out.index("strong") < out.index("mid") < out.index("weak")


def test_all_results_below_the_floor_is_reported_as_no_relevant_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinct from "no results": the useful response differs, and the agents
    are instructed that "I could not find reliable evidence" is a complete
    answer. Inventing a finding from weak sources is the alternative."""
    junk = [
        result(score=0.2, url="https://example.com/nav1", content="Home About Contact"),
        result(score=0.1, url="https://example.com/nav2", content="Cookie preferences"),
    ]
    belt = belt_with(junk, monkeypatch)
    out = belt._web_search("an obscure question")
    assert "no sufficiently relevant results" in out
    assert "2 result(s)" in out
    assert "<untrusted_document_content" not in out


def test_an_empty_result_set_is_still_reported_honestly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    belt = belt_with([], monkeypatch)
    assert belt._web_search("q") == "no results"


def test_a_missing_score_is_treated_as_irrelevant_not_as_perfect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider that stops returning scores must fail closed. Defaulting a
    missing score to 1.0 would silently restore the original bug."""
    belt = belt_with([result(score=0.0, url="https://example.com/x")], monkeypatch)
    out = belt._web_search("q")
    assert "<untrusted_document_content" not in out


def test_the_floor_sits_between_the_two_observed_clusters() -> None:
    """0.5 is chosen, not arbitrary: the good cluster scored 0.76-0.96 and the
    junk cluster 0.25. If this is ever tuned, it should stay in the gap."""
    assert 0.25 < Toolbelt.SEARCH_MIN_SCORE < 0.76
