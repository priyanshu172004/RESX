"""Agent output schemas.

These are the contract between the models and the rest of the system. Two of
them do real work rather than just describing shapes:

  * `Claim` refuses to exist with a value and no `computation_id`. That is the
    "LLM never does arithmetic" rule expressed as a validator — it is a wall,
    not advice, and the same rule is repeated as a `CHECK` constraint in the
    store so a bug that bypasses Pydantic still cannot persist a bare number.

  * `Citation` requires a non-empty quote. An agent cannot cite a location
    without also committing to what it says is there, which is what makes the
    resolver able to catch a fabrication.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AgentName(str, Enum):
    MANAGER = "manager"
    FINANCE = "finance"
    RISK = "risk"
    NEWS = "news"
    WORKFLOW = "workflow"
    MARKET = "market"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"


SPECIALISTS: tuple[AgentName, ...] = (
    AgentName.FINANCE,
    AgentName.RISK,
    AgentName.NEWS,
    AgentName.WORKFLOW,
    AgentName.MARKET,
)


class VerdictKind(str, Enum):
    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    CONTESTED = "contested"


class Stance(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    SILENT = "silent"


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Strict(BaseModel):
    """Base config: reject unexpected fields rather than ignoring them.

    `extra="forbid"` is mass-assignment prevention and also a correctness
    signal: a model inventing a field name usually means it misunderstood the
    schema, and failing loudly surfaces that instead of dropping data.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #


class Citation(Strict):
    citation_id: str = Field(default_factory=lambda: _new_id("cit"))

    # Internal corpus source.
    doc_id: str | None = None
    page: int | None = Field(default=None, ge=1)
    para_idx: int | None = Field(default=None, ge=0)
    chunk_id: str | None = None
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)

    #: Verbatim text from the source. The resolver matches on this, so an agent
    #: that cannot produce it cannot cite.
    quote: str = Field(min_length=8, max_length=2000)

    # External source (News, Market).
    url: str | None = None
    publisher: str | None = None
    published_at: str | None = None

    #: `derived_from_computation` means the quote was sliced from the stored
    #: source text by walking the computation's inputs back to their table,
    #: rather than transcribed by the agent. Kept distinct from "ok" so a
    #: reader — and the scorecard — can tell whose work each citation is.
    resolution: Literal[
        "ok",
        "derived_from_computation",
        "quote_mismatch",
        "anchor_not_found",
        "external",
    ] = "anchor_not_found"
    match_score: float | None = None

    @model_validator(mode="after")
    def _must_identify_a_source(self) -> Citation:
        if not self.doc_id and not self.url:
            raise ValueError("a citation must name either a corpus document or an external URL")
        if self.doc_id and self.page is None:
            raise ValueError("a corpus citation must include a page number")
        if (
            self.char_start is not None
            and self.char_end is not None
            and self.char_end <= self.char_start
        ):
            raise ValueError("char_end must be greater than char_start")
        return self

    @property
    def label(self) -> str:
        if self.url:
            return self.publisher or self.url
        label = f"p.{self.page}"
        if self.para_idx is not None:
            label += f" ¶{self.para_idx}"
        return label


class Claim(Strict):
    claim_id: str = Field(default_factory=lambda: _new_id("clm"))
    agent: AgentName
    statement: str = Field(min_length=8, max_length=2000)

    value: Decimal | None = None
    unit: str | None = None
    period: str | None = None
    currency: str | None = Field(default=None, min_length=3, max_length=3)
    scale_factor: Decimal = Decimal(1)

    #: Required whenever `value` is set. See the validator below.
    computation_id: str | None = None

    citations: list[Citation] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_reason: str | None = None

    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("value", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Any:
        # Accept a string or an int from the model, but never store a float:
        # binary floating point cannot represent 0.10 and the identities must
        # hold to the cent.
        if v is None or isinstance(v, Decimal):
            return v
        return Decimal(str(v))

    @property
    def is_externally_sourced(self) -> bool:
        """Every citation points at a web page rather than an uploaded document.

        Checked structurally — a `url` and no `doc_id` — because a schema has no
        store to ask. `ground_claim` re-checks it against the corpus, and that
        check is the authoritative one: a `doc_id` that really is in the corpus
        is resolved as strictly as ever.
        """
        return bool(self.citations) and all(c.url and not c.doc_id for c in self.citations)

    @model_validator(mode="after")
    def _numeric_claims_need_a_computation(self) -> Claim:
        if (
            self.value is not None
            and not self.computation_id
            and not self.is_externally_sourced
        ):
            # The exemption is for web sources only, and it is not a
            # concession: with no uploaded corpus there is no dataset to
            # compute over, so a figure on a page is *quoted*, not derived.
            # Demanding a computation_id there demanded the impossible, and
            # agents complied by omitting `value` and writing claims like "the
            # market size is as reported by source" — a sentence that names a
            # subject and asserts nothing, with the figure left inside the
            # quote where nothing downstream could use it.
            #
            # The number still has to appear verbatim in the cited quote;
            # `ground_claim` enforces that and rejects the claim otherwise. A
            # figure read off an uploaded table by the model remains banned,
            # which is the case this rule was written for.
            raise ValueError(
                "numeric claim without a computation_id: every figure must come "
                "from a logged sandbox computation, never from the model"
            )
        if self.confidence < 0.9 and not self.confidence_reason:
            raise ValueError(
                "confidence below 0.9 requires confidence_reason — calibration "
                "matters more than confidence"
            )
        return self

    @property
    def is_numeric(self) -> bool:
        return self.value is not None


# --------------------------------------------------------------------------- #
# Specialist outputs
# --------------------------------------------------------------------------- #


class FinancialClaim(Claim):
    statement_type: Literal[
        "income_statement", "balance_sheet", "cash_flow", "ratio", "other"
    ] = "other"
    fiscal_period: str | None = None
    fx_rate_used: Decimal | None = None


class RiskFinding(Claim):
    category: Literal[
        "liability",
        "covenant",
        "litigation",
        "concentration",
        "volatility",
        "regulatory",
        "going_concern",
        "other",
    ] = "other"
    severity: int = Field(ge=1, le=5)
    likelihood: int = Field(ge=1, le=5)
    mitigation_stated: str | None = None
    exposure_amount: Decimal | None = None

    @property
    def risk_score(self) -> int:
        return self.severity * self.likelihood


class Corroboration(Strict):
    claim_id: str
    stance: Stance
    sources: list[Citation] = Field(default_factory=list)
    source_tier: Literal["primary", "secondary", "none"] = "none"
    recency_days: int | None = None
    note: str = ""

    @model_validator(mode="after")
    def _non_silent_needs_a_source(self) -> Corroboration:
        # "Silent" is a real and common result. Anything else must point at
        # something, or it is manufactured corroboration.
        if self.stance is not Stance.SILENT and not self.sources:
            raise ValueError(f"stance '{self.stance.value}' requires at least one source")
        return self


class ProcessGap(Claim):
    framework: Literal["lean", "six_sigma", "theory_of_constraints", "cycle_time"]
    gap_type: str
    quantified_cost: Decimal | None = None
    cost_unit: Literal["usd_per_year", "hours_per_week", "defects_per_million"] | None = None
    bottleneck: str | None = None
    recommended_action: str


class MarketInsight(Claim):
    tam: Decimal | None = None
    sam: Decimal | None = None
    som: Decimal | None = None
    method: Literal["top_down", "bottom_up", "both", "qualitative"] = "qualitative"
    competitors: list[str] = Field(default_factory=list)
    share_estimate_pct: float | None = None
    as_of: str | None = None


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #


class Task(Strict):
    agent: AgentName
    sub_question: str = Field(min_length=8, max_length=1000)
    corpus_hint: str | None = None
    depends_on: list[AgentName] = Field(default_factory=list)


class Plan(Strict):
    tasks: list[Task] = Field(min_length=1, max_length=8)
    rationale: str = ""

    @property
    def agents(self) -> list[AgentName]:
        seen: list[AgentName] = []
        for task in self.tasks:
            if task.agent not in seen:
                seen.append(task.agent)
        return seen


# --------------------------------------------------------------------------- #
# Criticism
# --------------------------------------------------------------------------- #


class Verdict(Strict):
    claim_id: str
    verdict: VerdictKind
    independent_value: Decimal | None = None
    rationale: str = Field(min_length=8, max_length=4000)
    contradicting_citations: list[Citation] = Field(default_factory=list)
    debate_round: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _adverse_verdicts_need_evidence(self) -> Verdict:
        # A bare "this looks wrong" is not a finding. Refuting or contesting a
        # claim requires either a contradicting source or an independently
        # derived number.
        if (
            self.verdict in (VerdictKind.REFUTED, VerdictKind.CONTESTED)
            and not self.contradicting_citations
            and self.independent_value is None
        ):
            raise ValueError(
                f"a {self.verdict.value} verdict must supply contradicting "
                "citations or an independently derived value"
            )
        return self


class DebatePosition(Strict):
    party: Literal["author", "critic"]
    round: int = Field(ge=1)
    position: Literal["concede", "defend", "narrow", "maintain"]
    argument: str = Field(min_length=8, max_length=4000)
    new_citations: list[Citation] = Field(default_factory=list)
    revised_value: Decimal | None = None


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


class Insight(Strict):
    headline: str = Field(min_length=8, max_length=140)
    #: The implication, with the figures in it. The floor was 20 characters,
    #: which "Revenue is down." satisfies -- and that is a headline repeated,
    #: not an implication. 90 forces the sentence that says what it means.
    so_what: str = Field(min_length=90, max_length=1200)
    evidence: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    suggested_owner: str
    effort: Literal["S", "M", "L"]
    expected_impact: Decimal | None = None
    impact_unit: str | None = None
    contested: bool = False
    supporting_claim_ids: list[str] = Field(default_factory=list)

    @property
    def priority_score(self) -> float:
        """impact x confidence / effort, computed in code.

        The ranking is deliberately mechanical so the ordering is reproducible
        and arguable, rather than being whichever insight the model happened to
        write first.
        """
        weights = {"S": 1.0, "M": 2.0, "L": 4.0}
        impact = float(self.expected_impact) if self.expected_impact else 0.0
        # A qualitative insight still needs a place in the ranking, so it is
        # scored on confidence alone rather than dropped to zero.
        base = impact if impact > 0 else 1.0
        return (base * self.confidence) / weights[self.effort]


class SwotItem(Strict):
    text: str = Field(min_length=8, max_length=600)
    #: Filled in code from `supporting_claim_ids`, never transcribed by the
    #: model. See `ExecutiveReportDraft`.
    citations: list[Citation] = Field(default_factory=list)
    supporting_claim_ids: list[str] = Field(default_factory=list)


class Swot(Strict):
    strengths: list[SwotItem] = Field(default_factory=list)
    weaknesses: list[SwotItem] = Field(default_factory=list)
    opportunities: list[SwotItem] = Field(default_factory=list)
    threats: list[SwotItem] = Field(default_factory=list)


class Recommendation(Strict):
    """A step someone can actually take, tied to the evidence that motivates it.

    An insight says what is true; this says what to do about it. They are kept
    separate because they fail differently: an insight is wrong when it
    misreads the evidence, and a recommendation is wrong when it does not
    follow from evidence that is perfectly correct. Merging them hides the
    second kind.

    `supporting_claim_ids` is required and is checked against the run's
    accepted claims, so advice is grounded on exactly the terms a figure is.
    "Diversify into adjacent markets" with nothing behind it is the kind of
    sentence a model will produce indefinitely and no one can act on.
    """

    #: An imperative. "Renegotiate the top-three supplier contracts before
    #: renewal in Q3", not "supplier concentration should be considered".
    action: str = Field(min_length=12, max_length=400)

    #: What kind of move this is, which is what makes the list sortable into
    #: something a reader can act on rather than a flat wall of advice.
    kind: Literal["mitigate", "grow", "monitor", "investigate"]

    #: Why this follows from the findings. Not a restatement of the action.
    rationale: str = Field(min_length=20, max_length=800)

    #: The observable that says whether it worked. A recommendation with no
    #: metric cannot be reviewed later, which makes it advice rather than a
    #: plan — and "track this" is itself one of the kinds above.
    metric: str = Field(min_length=4, max_length=200)

    horizon: Literal["now", "quarter", "year"]
    owner: str = Field(min_length=2, max_length=80)
    effort: Literal["S", "M", "L"]

    #: Free text on purpose. A quantified effect would be a new number, and
    #: the Synthesizer is forbidden from introducing those.
    expected_effect: str | None = Field(default=None, max_length=300)

    #: At least one. The gate below drops any recommendation whose claims are
    #: not in the run, which is what stops invented justifications.
    supporting_claim_ids: list[str] = Field(min_length=1)

    @property
    def urgency_rank(self) -> tuple[int, int]:
        """Sort key: soonest first, then cheapest.

        Mechanical, like `Insight.priority_score`, so the ordering is
        reproducible and arguable instead of being whichever recommendation the
        model wrote first.
        """
        horizons = {"now": 0, "quarter": 1, "year": 2}
        efforts = {"S": 0, "M": 1, "L": 2}
        return (horizons[self.horizon], efforts[self.effort])


class ExecutiveReport(Strict):
    question: str

    #: The narrative answer, in prose, carrying the figures.
    #:
    #: This existed in the UI before it existed here: the run page rendered
    #: `report.executive_summary` and the schema had no such field, so the
    #: paragraph slot at the top of every report was permanently empty and the
    #: whole report read as a bare list of headlines.
    #:
    #: The floor is low and the ceiling is high on purpose. A narrow question
    #: ("what was revenue in Q1") is answered honestly in two sentences, and
    #: padding it out would contradict the instruction not to pad; a broad one
    #: deserves several paragraphs. Length should follow the evidence, so the
    #: prompt asks for substance and the schema only rules out a one-liner.
    executive_summary: str = Field(min_length=120, max_length=4000)

    insights: list[Insight] = Field(max_length=5)
    #: What to do about the insights. Capped: a list of twenty next steps is
    #: not a plan, it is a way of avoiding choosing.
    recommendations: list[Recommendation] = Field(default_factory=list, max_length=8)
    swot: Swot = Field(default_factory=Swot)
    financial_summary: str = ""
    risk_register: list[str] = Field(default_factory=list)
    #: Charts of the extracted tables, derived in code by
    #: `app.reporting.charts` and filled in after the model returns.
    #:
    #: The model never sees this field: it is asked for
    #: `ExecutiveReportDraft`, which does not have it, so a fabricated series
    #: cannot be expressed rather than merely being discarded. A chart is a
    #: series of numbers wearing a visual claim to precision, and a model
    #: asked for "revenue by quarter" will produce a plausible, smooth,
    #: entirely invented one that a reader who would interrogate a sentence
    #: will not interrogate. Every point plotted comes from the extracted CSV.
    charts: list[dict[str, Any]] = Field(default_factory=list)
    #: The full report as a document: one section per specialist, every figure
    #: with its source and the reviewer's verdict. Assembled in code from the
    #: run's own claims and verdicts -- see `app.reporting.document` for why it
    #: is not written by a model.
    document: dict[str, Any] = Field(default_factory=dict)
    #: Anything the report cannot stand behind. A report that hides its own
    #: gaps is more dangerous than one that has none.
    limitations: list[str] = Field(default_factory=list)
    contested_claim_ids: list[str] = Field(default_factory=list)
    degraded_agents: list[AgentName] = Field(default_factory=list)

    def ranked_insights(self) -> list[Insight]:
        return sorted(self.insights, key=lambda i: -i.priority_score)

    def ranked_recommendations(self) -> list[Recommendation]:
        return sorted(self.recommendations, key=lambda r: r.urgency_rank)


# --------------------------------------------------------------------------- #
# What the Synthesizer is actually asked for
# --------------------------------------------------------------------------- #


class InsightDraft(Strict):
    """An insight as the model writes it: no citation objects.

    The model names the claims it is synthesising and the code attaches their
    citations. Asking it to re-transcribe a citation instead was a live bug:
    it copied a claim's quote and doc_id but dropped the page, and `Citation`
    requires a page for a corpus source — correctly, since a corpus citation
    without one cannot be resolved. The whole report then failed validation
    twice and the run died with everything else about it correct.

    Re-transcription was never worth attempting. The citation already exists,
    verified, attached to the claim; copying it through a language model can
    only lose or corrupt fields. This way the evidence on an insight is
    guaranteed to be the evidence that was actually checked.
    """

    headline: str = Field(min_length=8, max_length=140)
    so_what: str = Field(min_length=90, max_length=1200)
    confidence: float = Field(ge=0.0, le=1.0)
    suggested_owner: str
    effort: Literal["S", "M", "L"]
    expected_impact: Decimal | None = None
    impact_unit: str | None = None
    contested: bool = False
    supporting_claim_ids: list[str] = Field(default_factory=list)


class SwotItemDraft(Strict):
    text: str = Field(min_length=8, max_length=600)
    supporting_claim_ids: list[str] = Field(default_factory=list)


class SwotDraft(Strict):
    strengths: list[SwotItemDraft] = Field(default_factory=list)
    weaknesses: list[SwotItemDraft] = Field(default_factory=list)
    opportunities: list[SwotItemDraft] = Field(default_factory=list)
    threats: list[SwotItemDraft] = Field(default_factory=list)


class ExecutiveReportDraft(Strict):
    """The report as the model writes it.

    Differs from `ExecutiveReport` in exactly two ways, both deliberate:

    * No citation objects anywhere — claim ids instead, resolved in code.
    * No `charts` — those are derived from the extracted tables, and a field
      the model cannot see is a field it cannot invent. It also keeps roughly
      1,400 characters of chart schema out of every request, which matters on
      a tokens-per-minute budget.
    """

    question: str
    executive_summary: str = Field(min_length=120, max_length=4000)
    insights: list[InsightDraft] = Field(max_length=5)
    recommendations: list[Recommendation] = Field(default_factory=list, max_length=8)
    swot: SwotDraft = Field(default_factory=SwotDraft)
    financial_summary: str = ""
    risk_register: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    contested_claim_ids: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #


class Budget(Strict):
    """Spend ceilings. An unbounded agent loop is a financial denial of service."""

    usd_cap: float = 5.0
    token_cap: int = 2_000_000
    node_cap: int = 40
    debate_round_cap: int = 3

    usd_used: float = 0.0
    tokens_used: int = 0
    nodes_run: int = 0

    @property
    def usd_remaining(self) -> float:
        return max(0.0, self.usd_cap - self.usd_used)

    @property
    def exhausted(self) -> bool:
        return (
            self.usd_used >= self.usd_cap
            or self.tokens_used >= self.token_cap
            or self.nodes_run >= self.node_cap
        )

    def charge(self, *, usd: float = 0.0, tokens: int = 0, nodes: int = 0) -> None:
        self.usd_used += usd
        self.tokens_used += tokens
        self.nodes_run += nodes

    def reason(self) -> str:
        if self.usd_used >= self.usd_cap:
            return f"spend cap reached (${self.usd_used:.2f} of ${self.usd_cap:.2f})"
        if self.tokens_used >= self.token_cap:
            return f"token cap reached ({self.tokens_used:,} of {self.token_cap:,})"
        if self.nodes_run >= self.node_cap:
            return f"node cap reached ({self.nodes_run} of {self.node_cap})"
        return "within budget"
