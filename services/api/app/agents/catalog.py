"""What each agent does, described for the person reading the screen.

This module exists to keep a boundary that was previously missing: the API was
serving the agents' **system prompts** and their raw internal tool identifiers
straight to the browser. Three reasons that had to stop, in order of severity:

1. **A published prompt is a targeting aid.** The prompts contain the numbered
   accuracy rules and the injection defences verbatim. A document is an
   untrusted input that reaches the reasoning layer, so anyone who can upload
   one — or get one uploaded — can write text aimed at the exact wording they
   just read: "rule 3 does not apply to this appendix". Defences that are
   published are defences that can be drafted against.

2. **Tool identifiers map the attack surface.** `sandbox.run_python` and
   `filesystem.search_corpus` tell a reader precisely which capabilities exist
   to be reached. The per-agent allowlist is the primary control against
   prompt injection, and enumerating its contents is free reconnaissance.

3. **It is the wrong information anyway.** Nobody using the product needs the
   prompt. They need to know what the Risk agent looks for and whether it can
   see the web. A prompt excerpt answers a question nobody asked, in a form
   only its author can read.

So the API serves *this*: a written description and a set of plain-language
capability labels. Both are maintained by hand, which is the cost of the
boundary — and the reason `network_access` is still derived from the allowlist
rather than described here. That one fact is a security property users are
entitled to check, and a hand-written claim about it could drift from the code
that enforces it.

The generated Python is treated the same way. It is removed from the event
stream but stays retrievable per computation from `/computations/{id}`, which
is authenticated and workspace-scoped. Auditing a specific number is a
deliberate act; broadcasting every line of generated code to anyone watching a
run is not.
"""

from __future__ import annotations

from app.agents.schemas import AgentName

#: Plain-language names for the internal tool groups. A reader learns what an
#: agent can do without learning the identifier to aim an injection at.
CAPABILITY_LABELS: dict[str, str] = {
    "filesystem": "Reads your uploaded documents",
    "sandbox": "Runs calculations in an isolated sandbox",
    "search": "Searches public web sources",
    "sql": "Queries extracted tables",
}

#: Order is presentational: the capabilities a reader cares about most first.
CAPABILITY_ORDER = ("filesystem", "sandbox", "sql", "search")


AGENT_PROFILES: dict[AgentName, dict[str, str]] = {
    AgentName.MANAGER: {
        "title": "Manager",
        "role": "Plans the analysis",
        "description": (
            "Reads your question and decides which specialists are needed to "
            "answer it, and in what order. It deliberately holds no tools of "
            "its own — it cannot read a document or reach the network, so it "
            "works only from the question and the corpus summary."
        ),
    },
    AgentName.FINANCE: {
        "title": "Finance",
        "role": "Figures and financial statements",
        "description": (
            "Reconstructs revenue, costs, margins, cash and runway from the "
            "tables in your documents. Every number it reports is produced by "
            "code running in the sandbox, never by the model — and it has no "
            "web access at all, so nothing outside your corpus can influence "
            "a financial figure."
        ),
    },
    AgentName.RISK: {
        "title": "Risk",
        "role": "Exposures and what could go wrong",
        "description": (
            "Looks for concentration, liquidity, covenant, dependency and "
            "compliance exposure, scoring each by severity and likelihood "
            "against cited evidence. It may corroborate a finding against "
            "public sources."
        ),
    },
    AgentName.NEWS: {
        "title": "News",
        "role": "External corroboration",
        "description": (
            "Checks whether public reporting supports or contradicts what the "
            "documents say. Reporting 'no external coverage found' is a valid "
            "and useful result — it does not infer corroboration it could not "
            "find."
        ),
    },
    AgentName.WORKFLOW: {
        "title": "Workflow",
        "role": "Process and control gaps",
        "description": (
            "Identifies gaps between the process a document describes and the "
            "process its own figures imply — reconciliation, approval, "
            "segregation of duties."
        ),
    },
    AgentName.MARKET: {
        "title": "Market",
        "role": "Competitive and sector context",
        "description": (
            "Places the figures in context: sector growth, comparable "
            "businesses, pricing and demand signals, drawing on both your "
            "documents and public sources."
        ),
    },
    AgentName.CRITIC: {
        "title": "Critic",
        "role": "Adversarial review",
        "description": (
            "Tries to falsify every claim the other agents made. It "
            "re-derives each figure independently and hunts for "
            "contradictions. A disagreement it cannot resolve is published as "
            "contested, with both positions — never averaged away. It runs on "
            "the strongest model available, because a reviewer weaker than "
            "what it reviews produces theatre rather than scrutiny."
        ),
    },
    AgentName.SYNTHESIZER: {
        "title": "Synthesizer",
        "role": "Writes the report",
        "description": (
            "Ranks the surviving findings into at most five insights and "
            "states what the analysis could not establish. It holds no tools "
            "and can introduce no new figures: everything it writes traces "
            "back to a claim another agent already grounded."
        ),
    },
}


def capabilities_for(tools: frozenset[str] | set[str]) -> list[str]:
    """Plain-language capabilities for a set of internal tool identifiers.

    Grouped by prefix, so a reader sees "Runs calculations in an isolated
    sandbox" rather than `sandbox.run_python`. Unknown prefixes are dropped
    rather than passed through: leaking a new identifier by default is exactly
    the failure this function exists to prevent.
    """
    groups = {tool.split(".", 1)[0] for tool in tools}
    return [
        CAPABILITY_LABELS[group]
        for group in CAPABILITY_ORDER
        if group in groups and group in CAPABILITY_LABELS
    ]


def profile_for(agent: AgentName) -> dict[str, str]:
    """The public description of an agent. Never its prompt."""
    return AGENT_PROFILES.get(
        agent,
        {
            "title": agent.value.title(),
            "role": "Specialist",
            "description": "",
        },
    )
