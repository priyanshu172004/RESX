"""The MCP-shaped tool layer.

Agents never touch a database driver, a file path, or an HTTP client. They call
named tools, exactly as they would across an MCP boundary, which buys the two
properties `docs/02-ARCHITECTURE.md` §3 is about: a backend can be swapped
without changing a prompt, and **every tool is granted per agent**.

That second property is the strongest defence RESX has against prompt
injection. The Finance agent does not have a web tool it is instructed not to
use — it has no web tool at all, so a document telling it to exfiltrate a
balance sheet is asking for a capability that does not exist in that context.
Instructing a model not to be fooled is not a security control; removing the
capability is.

Every tool result is also wrapped in `<untrusted_document_content>` before it
can reach a prompt, so retrieved text arrives as data with a visible boundary
rather than as free-floating instructions.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import socket
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from app.agents.schemas import AgentName
from app.analysis.sandbox import ComputationRecord, Sandbox
from app.rag.embeddings import Embedder
from app.rag.retrieve import retrieve

if TYPE_CHECKING:
    # Type-only: importing the factory would pull in pymongo, which a
    # SQLite-only install does not have.
    from app.store.factory import Store


class ToolError(RuntimeError):
    pass


class ToolNotGrantedError(ToolError):
    """The agent asked for a capability it does not have.

    Distinct from "the tool failed": this is the isolation boundary doing its
    job, and it is logged as a security-relevant event rather than a bug.
    """


# --------------------------------------------------------------------------- #
# Grants
# --------------------------------------------------------------------------- #

FILESYSTEM_TOOLS = frozenset(
    {
        "filesystem.list_documents",
        "filesystem.read_page",
        "filesystem.search_corpus",
        "filesystem.list_datasets",
        "filesystem.get_dataset_profile",
    }
)
SANDBOX_TOOLS = frozenset({"sandbox.run_python"})
SEARCH_TOOLS = frozenset({"search.web_search", "search.fetch_url"})
SQL_TOOLS = frozenset({"postgres.query"})

#: Per-agent grants, straight from `docs/03-AGENTS.md`.
#:
#: Note what is *absent* in each row — that is the actual control. Finance has
#: no search; News has no filesystem (it receives claims through state, which
#: also stops it drifting into the Finance agent's job); Manager and
#: Synthesizer have nothing at all, so neither can invent a figure by going and
#: looking for one.
ALLOWLIST: dict[AgentName, frozenset[str]] = {
    AgentName.MANAGER: frozenset(),
    AgentName.FINANCE: FILESYSTEM_TOOLS | SANDBOX_TOOLS | SQL_TOOLS,
    AgentName.RISK: FILESYSTEM_TOOLS | SANDBOX_TOOLS | SEARCH_TOOLS,
    AgentName.NEWS: SEARCH_TOOLS,
    AgentName.WORKFLOW: FILESYSTEM_TOOLS | SANDBOX_TOOLS,
    AgentName.MARKET: SEARCH_TOOLS | FILESYSTEM_TOOLS,
    # The Critic gets search because corroboration is its job, and without
    # it the reviewer cannot check a web-sourced claim at all. That was not
    # a gap in capability, it was a source of wrong verdicts: in research
    # mode it searched an empty corpus, found nothing, and returned
    # `refuted` for every claim -- which the Verdict schema then rejected
    # for having no contradicting source, so every finding in the report
    # came back "not reviewed".
    AgentName.CRITIC: SANDBOX_TOOLS | FILESYSTEM_TOOLS | SEARCH_TOOLS,
    AgentName.SYNTHESIZER: frozenset(),
}


#: Plain-language names for the event stream. The identifier itself stays
#: internal: it names a capability an injection would try to reach, and the
#: per-agent allowlist is the primary control against that. A tool with no
#: entry here reads as "Used a tool" rather than leaking its name by default.
PUBLIC_TOOL_NAMES: dict[str, str] = {
    "filesystem.list_documents": "Listed your documents",
    "filesystem.list_datasets": "Listed extracted tables",
    "filesystem.search_corpus": "Searched your documents",
    "filesystem.read_page": "Read a page",
    "sandbox.run_python": "Ran a calculation",
    "sandbox.read_artifact": "Read a computation result",
    "search.web_search": "Searched the web",
    "search.fetch_url": "Fetched a web page",
    "sql.query": "Queried a table",
}


@dataclass(slots=True)
class ToolInvocation:
    """A record of one tool call, for the run event stream."""

    tool: str
    agent: AgentName
    args_digest: str
    ok: bool
    duration_ms: int
    detail: str = ""

    def public_name(self) -> str:
        return PUBLIC_TOOL_NAMES.get(self.tool, "Used a tool")

    def to_dict(self) -> dict[str, Any]:
        """The public form, for the run event stream.

        `tool` and `args_digest` are omitted: the identifier enumerates the
        capabilities available to reach, and the digest is a fingerprint of
        arguments that could include a query or a path.
        """
        return {
            "action": self.public_name(),
            "agent": self.agent.value,
            "ok": self.ok,
            "duration_ms": self.duration_ms,
            "detail": self.detail,
        }

    def to_audit_dict(self) -> dict[str, Any]:
        """The full form, for the audit log. Never sent to a browser."""
        return {
            "tool": self.tool,
            "agent": self.agent.value,
            "args_digest": self.args_digest,
            "ok": self.ok,
            "duration_ms": self.duration_ms,
            "detail": self.detail,
        }


def wrap_untrusted(text: str, **attrs: Any) -> str:
    """Delimit content that came from a document or the web.

    The delimiter is what the standing system rule refers to. Attribute values
    are sanitised so a document cannot inject a fake closing tag and escape the
    boundary — which would defeat the entire mechanism.
    """
    safe = " ".join(
        f'{k}="{re.sub(chr(34) + "|<|>", "", str(v))}"'
        for k, v in attrs.items()
        if v is not None
    )
    body = text.replace("</untrusted_document_content>", "[/untrusted]")
    return f"<untrusted_document_content {safe}>\n{body}\n</untrusted_document_content>"


# --------------------------------------------------------------------------- #
# SSRF guard
# --------------------------------------------------------------------------- #

_BLOCKED_SCHEMES = frozenset({"file", "gopher", "ftp", "data", "dict"})


def assert_url_is_safe(url: str, allowlist: frozenset[str]) -> str:
    """Reject anything that could reach an internal service.

    Cloud metadata endpoints (169.254.169.254) are the highest-value SSRF
    target in a hosted deployment, so link-local is blocked explicitly along
    with every private range. DNS is resolved here rather than trusting the
    hostname, because a public name can legitimately resolve to a private
    address.
    """
    parsed = urlparse(url)

    if parsed.scheme in _BLOCKED_SCHEMES or parsed.scheme != "https":
        raise ToolError(f"only https URLs may be fetched (got {parsed.scheme!r})")

    host = parsed.hostname
    if not host:
        raise ToolError("URL has no host")

    if allowlist and host.lower() not in allowlist:
        raise ToolError(
            f"host {host!r} is not on the egress allowlist; the allowlist fails "
            "closed by design"
        )

    try:
        infos = socket.getaddrinfo(host, parsed.port or 443, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise ToolError(f"could not resolve {host!r}: {exc}") from exc

    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            raise ToolError(
                f"{host!r} resolves to the non-public address {address}; refusing "
                "to fetch (SSRF guard)"
            )
    return url


# --------------------------------------------------------------------------- #
# Toolbelt
# --------------------------------------------------------------------------- #


class Toolbelt:
    """The tools one agent is permitted to use, for one run."""

    def __init__(
        self,
        agent: AgentName,
        *,
        store: Store,
        workspace_id: str,
        embedder: Embedder,
        sandbox: Sandbox | None = None,
        dataset_dir: str | Path = "./storage/datasets",
        tavily_api_key: str = "",
        egress_allowlist: frozenset[str] = frozenset(),
        corpus_ids: Sequence[str] | None = None,
        top_k: int = 8,
        context_chars: int = 7_000,
    ) -> None:
        self.agent = agent
        self.store = store
        self.workspace_id = workspace_id
        self.embedder = embedder
        self.sandbox = sandbox
        self.dataset_dir = Path(dataset_dir)
        self.tavily_api_key = tavily_api_key
        self.egress_allowlist = egress_allowlist
        self.corpus_ids = list(corpus_ids) if corpus_ids else None
        self.top_k = top_k
        # The retrieved text handed to the model, bounded by what the model's
        # request limit can actually carry. Unbounded retrieval is what
        # produced HTTP 413 on every specialist -- the retrieved spans, not the
        # web results, were the large part of the request.
        self.context_chars = context_chars

        self.invocations: list[ToolInvocation] = []
        self.computations: list[ComputationRecord] = []
        self.retrieved_chunks: dict[str, Any] = {}

    # -- introspection ----------------------------------------------------

    @property
    def granted(self) -> frozenset[str]:
        return ALLOWLIST[self.agent]

    def schema(self) -> list[dict[str, Any]]:
        """The tool definitions this agent may see.

        Only granted tools are described. An agent is never told about a tool
        it cannot call, so a document cannot talk it into using one.
        """
        return [spec for name, spec in _TOOL_SPECS.items() if name in self.granted]

    def describe_granted(self) -> str:
        if not self.granted:
            return "You have no tools. Work from the information given to you."
        lines = ["You may call exactly these tools:"]
        for name in sorted(self.granted):
            spec = _TOOL_SPECS.get(name, {})
            lines.append(f"  - {name}: {spec.get('description', '')}")
        return "\n".join(lines)

    # -- dispatch ---------------------------------------------------------

    def call(self, tool: str, **kwargs: Any) -> Any:
        if tool not in self.granted:
            raise ToolNotGrantedError(
                f"agent '{self.agent.value}' is not granted '{tool}'. "
                f"Granted: {sorted(self.granted) or 'none'}"
            )

        handler = self._handlers().get(tool)
        if handler is None:
            raise ToolError(f"unknown tool: {tool}")

        digest = hashlib.sha256(
            json.dumps(kwargs, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]
        started = time.perf_counter()
        try:
            result = handler(**kwargs)
        except Exception as exc:
            self.invocations.append(
                ToolInvocation(
                    tool=tool,
                    agent=self.agent,
                    args_digest=digest,
                    ok=False,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            raise
        self.invocations.append(
            ToolInvocation(
                tool=tool,
                agent=self.agent,
                args_digest=digest,
                ok=True,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
        )
        return result

    def _handlers(self) -> dict[str, Callable[..., Any]]:
        return {
            "filesystem.list_documents": self._list_documents,
            "filesystem.read_page": self._read_page,
            "filesystem.search_corpus": self._search_corpus,
            "filesystem.list_datasets": self._list_datasets,
            "filesystem.get_dataset_profile": self._dataset_profile,
            "sandbox.run_python": self._run_python,
            "postgres.query": self._sql_query,
            "search.web_search": self._web_search,
            "search.fetch_url": self._fetch_url,
        }

    # -- filesystem -------------------------------------------------------

    def _list_documents(self) -> str:
        docs = self.store.list_documents(workspace_id=self.workspace_id)
        if self.corpus_ids:
            docs = [d for d in docs if d["doc_id"] in self.corpus_ids]
        lines = [
            f"{d['doc_id']} | {d['source_name']} | {d['kind']} | "
            f"{d['page_count']} page(s) | status={d['status']}"
            + (f" | NEEDS OCR: pages {d['low_conf_pages']}" if d["low_conf_pages"] else "")
            for d in docs
        ]
        return "\n".join(lines) if lines else "no documents in this workspace"

    def _read_page(self, doc_id: str, page: int) -> str:
        # Documents are addressed by opaque id, never by a caller-supplied
        # path, so there is no traversal input to sanitise. The store is
        # workspace-scoped, so one tenant cannot read another's page.
        text = self.store.get_page_text(
            workspace_id=self.workspace_id, doc_id=doc_id, page=int(page)
        )
        if text is None:
            return f"no such page: {doc_id} page {page}"
        return wrap_untrusted(text, doc_id=doc_id, page=page)

    def _search_corpus(self, query: str, top_k: int | None = None) -> str:
        result = retrieve(
            query,
            store=self.store,
            embedder=self.embedder,
            workspace_id=self.workspace_id,
            doc_ids=self.corpus_ids,
            top_k=int(top_k or self.top_k),
        )
        for chunk in result.chunks:
            self.retrieved_chunks[chunk.chunk_id] = chunk
        if not result.chunks:
            return "no matching content in the corpus"
        return result.context_block(max_chars=self.context_chars)

    def _list_datasets(self) -> str:
        datasets = self.store.list_datasets(workspace_id=self.workspace_id)
        if self.corpus_ids:
            datasets = [d for d in datasets if d["doc_id"] in self.corpus_ids]
        lines = []
        for d in datasets:
            note = ""
            if d["agreement"] is False:
                note = " | WARNING: extractors disagreed, confirm before use"
            if d["scale_factor"] == "1":
                issues = d.get("profile", {}).get("issues", [])
                if any(i.get("kind") == "undeclared_scale" for i in issues):
                    note += " | scale UNDECLARED, figures are face value"
            lines.append(
                f"{d['dataset_id']} | {d['name']} | p{d['source_page']} | "
                f"{d['n_rows']}x{d['n_cols']} | scale=x{d['scale_factor']} | "
                f"currency={d['currency'] or 'unknown'}{note}"
            )
        return "\n".join(lines) if lines else "no datasets in this workspace"

    def _dataset_profile(self, dataset_id: str) -> str:
        datasets = self.store.list_datasets(workspace_id=self.workspace_id)
        for d in datasets:
            if d["dataset_id"] == dataset_id:
                return json.dumps(d["profile"], indent=2, default=str)
        return f"no such dataset: {dataset_id}"

    # -- sandbox ----------------------------------------------------------

    def _run_python(self, code: str, inputs: list[str] | None = None) -> str:
        if self.sandbox is None:
            raise ToolError("no sandbox is configured for this run")

        record = self.sandbox.run(code, inputs=inputs or [], dataset_dir=self.dataset_dir)
        self.computations.append(record)

        if not record.ok:
            # Errors are returned to the agent, not raised, so it can fix its
            # own code — a traceback is the most useful thing it can receive.
            return json.dumps(
                {
                    "computation_id": record.computation_id,
                    "ok": False,
                    "error": record.error,
                    "stdout": record.stdout[:2000],
                },
                default=str,
            )

        return json.dumps(
            {
                "computation_id": record.computation_id,
                "ok": True,
                "result": record.result,
                "stdout": record.stdout[:4000],
                "duration_ms": record.duration_ms,
            },
            default=str,
        )

    # -- sql --------------------------------------------------------------

    _SELECT_ONLY = re.compile(r"^\s*select\b", re.I)
    _FORBIDDEN = re.compile(
        r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|"
        r"attach|pragma|vacuum)\b",
        re.I,
    )

    def _sql_query(self, sql: str, limit: int = 100) -> str:
        """Read-only SQL over registered datasets.

        Two independent controls, deliberately: the statement is parsed to
        accept only a single `SELECT`, **and** in the production deployment the
        connection holds a role that is physically incapable of writing. The
        parser can have gaps; a role without write grants cannot.
        """
        if not self._SELECT_ONLY.match(sql):
            raise ToolError("only SELECT statements are permitted")
        if ";" in sql.rstrip().rstrip(";"):
            raise ToolError("multi-statement queries are not permitted")
        if self._FORBIDDEN.search(sql):
            raise ToolError("statement contains a forbidden keyword")

        return (
            "postgres.query is not wired to a live warehouse in this deployment. "
            "Use filesystem.list_datasets and sandbox.run_python with "
            "resx.load(dataset_id) to work with the extracted tables."
        )

    # -- search -----------------------------------------------------------

    #: Characters kept from each search result. A result is evidence to quote
    #: from, not a document to read: two or three paragraphs carry the claim
    #: and its numbers, and the rest only consumes the request budget.
    SEARCH_CHARS_PER_RESULT = 900

    #: Ceiling on the whole result set. Sized well under the free tier's 8,000
    #: tokens-per-minute so the request cannot be rejected before the model
    #: sees it -- at roughly 3.2 characters per token this is about 2,800
    #: tokens, leaving room for the system prompt and the answer.
    SEARCH_CHARS_TOTAL = 9_000

    #: Minimum relevance score a result must carry to reach the model.
    #:
    #: The provider scores every result and we were ignoring it, which is why
    #: research answers arrived with one good finding and several unrelated
    #: ones. On "What was Apple's revenue in Q1 2025?" the top three results
    #: scored 0.93-0.96 and answered it; two more scored 0.25 and their entire
    #: content was store navigation -- "Shop iPad / iPad Accessories / Apple
    #: Trade In" -- passed to the agent with exactly the same standing as the
    #: earnings release.
    #:
    #: The grounding gate cannot catch that: "Shop iPad" *is* verbatim in the
    #: source, so a claim built on it is perfectly traceable and completely
    #: irrelevant. Grounding proves provenance, not relevance, and relevance
    #: has to be filtered where the score exists.
    #:
    #: 0.5 sits in the empty band between the two clusters observed. If every
    #: result falls below it the tool says it found nothing, which is the
    #: correct answer and one the agents are instructed to report.
    SEARCH_MIN_SCORE = 0.5

    def _web_search(self, query: str, max_results: int = 4) -> str:
        if not self.tavily_api_key:
            # An honest unavailability beats a fabricated result set. The News
            # agent is instructed that "silent" is a valid finding, so this
            # degrades correctly rather than inventing corroboration.
            return (
                "web search is unavailable: no TAVILY_API_KEY is configured. "
                "Report external corroboration as 'silent' rather than "
                "inferring it."
            )

        import httpx

        try:
            response = httpx.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.tavily_api_key,
                    "query": query,
                    "max_results": min(int(max_results), 6),
                    # "basic" rather than "advanced": the deeper extraction
                    # returned whole page bodies, which is what overflowed the
                    # request budget and produced HTTP 413.
                    "search_depth": "basic",
                },
                timeout=30.0,
            )
        except httpx.HTTPError as exc:
            raise ToolError(f"search request failed: {exc}") from exc

        if response.status_code != 200:
            raise ToolError(f"search failed ({response.status_code})")

        payload = response.json()
        blocks: list[str] = []
        used = 0
        dropped = 0
        irrelevant = 0
        duplicates = 0
        seen_urls: set[str] = set()

        # Highest-scoring first, so if the size ceiling forces anything out it
        # is the least relevant result rather than whichever the provider
        # happened to list last.
        results = sorted(
            payload.get("results", []),
            key=lambda item: float(item.get("score") or 0.0),
            reverse=True,
        )

        for item in results:
            content = str(item.get("content") or "").strip()
            if not content:
                continue

            score = float(item.get("score") or 0.0)
            if score < self.SEARCH_MIN_SCORE:
                # Below the floor the content is usually not prose at all --
                # navigation menus, cookie banners, footer links. Passing it
                # on yields claims that are traceable and irrelevant.
                irrelevant += 1
                continue

            # The provider returned the same page twice in the observed
            # sample, with identical scores. Two copies of one source read to
            # a model as two independent sources agreeing, which is precisely
            # the corroboration the citation rules exist to establish.
            url = str(item.get("url") or "").strip()
            if url and url in seen_urls:
                duplicates += 1
                continue
            if url:
                seen_urls.add(url)

            if len(content) > self.SEARCH_CHARS_PER_RESULT:
                # Cut on a sentence boundary where one is close by, so the
                # agent is not asked to quote from a fragment that stops
                # mid-clause.
                cut = content[: self.SEARCH_CHARS_PER_RESULT]
                boundary = max(cut.rfind(". "), cut.rfind("\n"))
                if boundary > self.SEARCH_CHARS_PER_RESULT // 2:
                    cut = cut[: boundary + 1]
                content = cut.rstrip() + " […]"

            block = wrap_untrusted(
                content,
                url=item.get("url"),
                publisher=item.get("title"),
                retrieved_at=time.strftime("%Y-%m-%d"),
            )

            if used + len(block) > self.SEARCH_CHARS_TOTAL and blocks:
                # Stop rather than truncate mid-block: a half-closed untrusted
                # wrapper would leave the boundary between data and
                # instruction ambiguous, which is the one thing that wrapper
                # exists to make unambiguous.
                dropped += 1
                continue

            blocks.append(block)
            used += len(block)

        if not blocks:
            if irrelevant:
                # Distinguished from "nothing came back", because the useful
                # response differs: a narrower query may help here, whereas
                # no results at all means the evidence is not on the web.
                return (
                    f"no sufficiently relevant results: {irrelevant} result(s) "
                    f"were returned but all scored below the relevance floor. "
                    f"Report that you could not find reliable evidence rather "
                    f"than using weak sources."
                )
            return "no results"

        notes: list[str] = []
        if dropped:
            notes.append(
                f"{dropped} further relevant result(s) omitted to stay within "
                f"the request size limit"
            )
        if irrelevant:
            notes.append(f"{irrelevant} result(s) discarded as not relevant to the query")
        if duplicates:
            notes.append(f"{duplicates} duplicate result(s) discarded")

        if notes:
            # Said out loud: the agent should know what its view excludes
            # rather than concluding from silence. The relevance and duplicate
            # counts are here so it does not treat a short result set as the
            # web being empty on the subject.
            blocks.append(f"[{'; '.join(notes)}.]")
        return "\n\n".join(blocks)

    def _fetch_url(self, url: str) -> str:
        import httpx

        safe = assert_url_is_safe(url, self.egress_allowlist)
        try:
            response = httpx.get(
                safe,
                timeout=20.0,
                follow_redirects=False,  # each hop is re-validated by the caller
                headers={"User-Agent": "RESX/0.1 (+analysis)"},
            )
        except httpx.HTTPError as exc:
            raise ToolError(f"fetch failed: {exc}") from exc

        if response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("location", "")
            raise ToolError(
                f"redirect to {location!r} not followed automatically; "
                "re-validate and fetch explicitly"
            )

        # Cap the body so a large page cannot blow up the context or the bill.
        return wrap_untrusted(
            response.text[:40_000],
            url=safe,
            retrieved_at=time.strftime("%Y-%m-%d"),
        )


# --------------------------------------------------------------------------- #
# Tool descriptions shown to the model
# --------------------------------------------------------------------------- #

_TOOL_SPECS: dict[str, dict[str, Any]] = {
    "filesystem.list_documents": {
        "name": "filesystem.list_documents",
        "description": "List the documents in this workspace with page counts and status.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "filesystem.read_page": {
        "name": "filesystem.read_page",
        "description": "Read the full extracted text of one page of one document.",
        "input_schema": {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string"},
                "page": {"type": "integer", "minimum": 1},
            },
            "required": ["doc_id", "page"],
            "additionalProperties": False,
        },
    },
    "filesystem.search_corpus": {
        "name": "filesystem.search_corpus",
        "description": (
            "Hybrid search over the corpus. Returns the most relevant spans with "
            "their doc_id, page and para_idx so they can be cited."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    "filesystem.list_datasets": {
        "name": "filesystem.list_datasets",
        "description": (
            "List extracted tables registered as datasets, with their scale factor, "
            "currency and any extraction warnings. Load these in the sandbox to sum "
            "columns exactly."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "filesystem.get_dataset_profile": {
        "name": "filesystem.get_dataset_profile",
        "description": (
            "Get the column profile of a dataset: inferred roles, null rates, "
            "sample values and data-quality issues."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"dataset_id": {"type": "string"}},
            "required": ["dataset_id"],
            "additionalProperties": False,
        },
    },
    "sandbox.run_python": {
        "name": "sandbox.run_python",
        "description": (
            "Run Python in an isolated sandbox and return its computation_id and "
            "result. Use resx.load(dataset_id) for a DataFrame and Decimal for all "
            "money. Assign your answer to `result`. No network access. This is the "
            "ONLY way to obtain a number."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "inputs": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
    "postgres.query": {
        "name": "postgres.query",
        "description": "Run a single read-only SELECT against the analytics warehouse.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
            "required": ["sql"],
            "additionalProperties": False,
        },
    },
    "search.web_search": {
        "name": "search.web_search",
        "description": "Search the public web for external corroboration.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    "search.fetch_url": {
        "name": "search.fetch_url",
        "description": "Fetch one https URL already present in the conversation.",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
            "additionalProperties": False,
        },
    },
}
