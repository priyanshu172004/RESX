"""The report as a document.

The `ExecutiveReport` is a summary: five insights, a few recommendations, a
SWOT. That is the right thing for someone with four minutes, and it is not what
someone asking for "the full analysis" wants — they want each specialist's
answer, the figures each one established, the sources behind them, and the
Critic's verdict on each. None of that was reachable: it existed in the store as
claims and verdicts and was never assembled into anything a reader could read.

**Assembled in code, not written by a model.** Every section here is built from
stored claims, verdicts, computations and citations. Three reasons, in order of
weight:

1. **Completeness is guaranteed.** A model asked to "write a section per agent"
   omits one when the request gets long, and the omission is invisible — the
   report simply has four sections instead of five and looks complete. Here, an
   agent that produced claims has a section because the loop cannot skip it.

2. **Nothing new can be introduced.** The figures, quotes, sources and verdicts
   are copied from what passed the grounding gate. A generated section could
   restate a figure slightly wrong and it would carry a citation, which is the
   worst failure this system has.

3. **It costs nothing.** On a free tier, a run already spends most of its time
   waiting out rate limits. Asking for six more pages of prose would double the
   run and risk the whole report failing on the last call.

What the model does write is the narrative: the executive summary, the
insights, the recommendations. Those need judgement. Assembly does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from app.agents.catalog import AGENT_PROFILES
from app.agents.schemas import AgentName, VerdictKind

#: Roughly how many characters of Markdown make a printed page at the size the
#: report renders. Used only to report the length honestly, never to pad.
CHARS_PER_PAGE = 2_400


@dataclass
class Section:
    """One part of the document, in Markdown."""

    slug: str
    title: str
    body: str
    #: Ordering hint for the renderer. Lower comes first.
    order: int = 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "title": self.title,
            "body": self.body,
            "order": self.order,
        }


@dataclass
class ReportDocument:
    question: str
    sections: list[Section] = field(default_factory=list)

    @property
    def markdown(self) -> str:
        parts = [f"# {self.question}\n"]
        for section in sorted(self.sections, key=lambda s: (s.order, s.title)):
            parts.append(f"\n## {section.title}\n\n{section.body.strip()}\n")
        return "\n".join(parts)

    @property
    def estimated_pages(self) -> float:
        return round(len(self.markdown) / CHARS_PER_PAGE, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "markdown": self.markdown,
            "estimated_pages": self.estimated_pages,
            "sections": [s.to_dict() for s in sorted(self.sections, key=lambda s: s.order)],
        }


def _fmt_value(value: Any, unit: str | None) -> str:
    if value is None:
        return ""
    try:
        number = Decimal(str(value))
    except Exception:
        return f"{value} {unit or ''}".strip()
    # Thousands separators, and no trailing zeros on a whole number: a report
    # that prints 1200000.00 makes the reader count digits.
    text = f"{number:,.2f}".rstrip("0").rstrip(".")
    return f"{text} {unit or ''}".strip()


def _source_line(citation: Any) -> str:
    if getattr(citation, "url", None):
        publisher = getattr(citation, "publisher", None) or citation.url
        return f"[{publisher}]({citation.url})"
    doc = getattr(citation, "doc_id", None) or "document"
    page = getattr(citation, "page", None)
    return f"`{doc}`, page {page}" if page else f"`{doc}`"


def _verdict_note(verdict: Any) -> str:
    if verdict is None:
        return "not reviewed"
    kind = verdict.verdict
    if kind is VerdictKind.CONFIRMED:
        return "confirmed by the reviewer"
    if kind is VerdictKind.REFUTED:
        return "**refuted by the reviewer**"
    return "**contested by the reviewer**"


def agent_section(
    agent: AgentName,
    claims: list[Any],
    verdicts: dict[str, Any],
    *,
    order: int,
) -> Section:
    """One specialist's findings, in full.

    Every claim it established, with the figure, the source, the quote and the
    Critic's verdict. Long by design — this is the part of the report that
    answers "what did each agent actually find", and summarising it here would
    just reproduce the executive summary at greater length.
    """
    # `AGENT_PROFILES` holds plain dicts — the abstraction boundary that keeps
    # prompts and tool identifiers out of anything user-facing.
    profile = AGENT_PROFILES.get(agent) or {}
    title = str(profile.get("title") or agent.value.title())
    lines: list[str] = []
    if profile.get("role"):
        lines.append(f"*{profile['role']}*\n")

    numeric = [c for c in claims if c.value is not None]
    if numeric:
        lines.append("### Figures established\n")
        lines.append("| Figure | Value | Period | Basis |")
        lines.append("| --- | --- | --- | --- |")
        for claim in numeric:
            basis = (
                f"computation `{claim.computation_id}`"
                if claim.computation_id
                else "quoted from source"
            )
            label = str(claim.payload.get("figure") or claim.statement)[:70]
            lines.append(
                f"| {label} | {_fmt_value(claim.value, claim.unit)} "
                f"| {claim.period or '—'} | {basis} |"
            )
        lines.append("")

    lines.append("### Findings\n")
    for index, claim in enumerate(claims, start=1):
        verdict = verdicts.get(claim.claim_id)
        lines.append(f"**{index}. {claim.statement}**\n")
        details = [f"Confidence {claim.confidence:.0%}", _verdict_note(verdict)]
        if claim.value is not None:
            details.insert(0, _fmt_value(claim.value, claim.unit))
        lines.append(f"- {' · '.join(details)}")
        for citation in claim.citations[:3]:
            quote = " ".join(str(citation.quote).split())[:280]
            lines.append(f"- Source: {_source_line(citation)}")
            lines.append(f"  > {quote}")
        if verdict is not None and verdict.rationale:
            reasoning = " ".join(str(verdict.rationale).split())[:400]
            lines.append(f"- Reviewer: {reasoning}")
        lines.append("")

    return Section(slug=agent.value, title=title, body="\n".join(lines), order=order)


def build_document(
    *,
    question: str,
    report: Any,
    claims: list[Any],
    verdicts: list[Any],
    computations: list[dict[str, Any]] | None = None,
) -> ReportDocument:
    """Assemble the full document from the run's own record.

    Section order follows how a report is read rather than how it was
    produced: the answer first, then what to do, then the evidence per
    specialist, then the workings, then the caveats.
    """
    document = ReportDocument(question=question)
    latest: dict[str, Any] = {}
    for verdict in verdicts:
        # Keep the last round's verdict per claim: a debate that ran three
        # rounds has three, and the final one is the conclusion.
        current = latest.get(verdict.claim_id)
        if current is None or verdict.debate_round >= current.debate_round:
            latest[verdict.claim_id] = verdict

    if getattr(report, "executive_summary", ""):
        document.sections.append(
            Section(
                slug="summary",
                title="Executive summary",
                body=report.executive_summary,
                order=10,
            )
        )

    insights = list(getattr(report, "insights", []) or [])
    if insights:
        lines = []
        for index, insight in enumerate(
            sorted(insights, key=lambda i: -i.priority_score), start=1
        ):
            lines.append(f"### {index}. {insight.headline}\n")
            lines.append(insight.so_what + "\n")
            meta = [
                f"Confidence {insight.confidence:.0%}",
                f"Owner: {insight.suggested_owner}",
                f"Effort: {insight.effort}",
            ]
            if insight.contested:
                meta.append("**contested**")
            lines.append(f"- {' · '.join(meta)}")
            for citation in insight.evidence[:3]:
                lines.append(f"- Source: {_source_line(citation)}")
            lines.append("")
        document.sections.append(
            Section(slug="insights", title="Key findings", body="\n".join(lines), order=20)
        )

    recommendations = list(getattr(report, "recommendations", []) or [])
    if recommendations:
        lines = ["| Action | Kind | When | Owner | Track |", "| --- | --- | --- | --- | --- |"]
        for rec in sorted(recommendations, key=lambda r: r.urgency_rank):
            lines.append(
                f"| {rec.action} | {rec.kind} | {rec.horizon} | {rec.owner} | {rec.metric} |"
            )
        lines.append("")
        for rec in sorted(recommendations, key=lambda r: r.urgency_rank):
            lines.append(f"**{rec.action}**\n")
            lines.append(f"{rec.rationale}\n")
            lines.append(f"- Measure by: {rec.metric}")
            if rec.expected_effect:
                lines.append(f"- Expected: {rec.expected_effect}")
            lines.append("")
        document.sections.append(
            Section(
                slug="recommendations",
                title="Recommended next steps",
                body="\n".join(lines),
                order=30,
            )
        )

    if getattr(report, "financial_summary", ""):
        document.sections.append(
            Section(
                slug="financials",
                title="Financial position",
                body=report.financial_summary,
                order=40,
            )
        )

    # One section per specialist that actually produced something. Ordered by
    # the fixed agent order rather than by claim count, so the same run always
    # reads the same way.
    by_agent: dict[AgentName, list[Any]] = {}
    for claim in claims:
        by_agent.setdefault(claim.agent, []).append(claim)

    for offset, agent in enumerate(AgentName):
        found = by_agent.get(agent)
        if not found:
            continue
        document.sections.append(agent_section(agent, found, latest, order=50 + offset))

    swot = getattr(report, "swot", None)
    if swot is not None:
        quadrants = [
            ("Strengths", swot.strengths),
            ("Weaknesses", swot.weaknesses),
            ("Opportunities", swot.opportunities),
            ("Threats", swot.threats),
        ]
        if any(items for _, items in quadrants):
            lines = []
            for name, items in quadrants:
                lines.append(f"### {name}\n")
                if not items:
                    # Said rather than hidden: an empty quadrant is a result.
                    lines.append("Nothing the evidence supports.\n")
                for item in items:
                    lines.append(f"- {item.text}")
                lines.append("")
            document.sections.append(
                Section(slug="swot", title="Position", body="\n".join(lines), order=80)
            )

    register = list(getattr(report, "risk_register", []) or [])
    if register:
        document.sections.append(
            Section(
                slug="risks",
                title="Risk register",
                body="\n".join(f"- {item}" for item in register),
                order=85,
            )
        )

    # The charts, as tables.
    #
    # Markdown cannot hold a plot, and a downloaded report that silently drops
    # every figure is a different document from the one on screen. So each
    # chart's own numbers go in, with its caption and its provenance — which is
    # also the table view the chart carries on screen, for the same reason:
    # the figures have to be readable without relying on the colour.
    charts = list(getattr(report, "charts", []) or [])
    if charts:
        # A distinct name: `lines` is already bound above and reusing it makes
        # the two blocks look coupled when they are not.
        figures: list[str] = []
        for chart in charts:
            series = chart.get("series") or []
            if not series or not series[0].get("points"):
                continue
            figures.append(f"### {chart.get('title')}\n")
            if chart.get("caption"):
                figures.append(f"{chart['caption']}\n")
            header = [chart.get("x_label") or "Label"] + [
                str(s.get("label") or s.get("key")) for s in series
            ]
            figures.append("| " + " | ".join(header) + " |")
            figures.append("| " + " | ".join("---" for _ in header) + " |")
            # One row per x, taking the value from each series if it has one.
            labels: list[str] = []
            for s in series:
                for point in s.get("points") or []:
                    if str(point.get("x")) not in labels:
                        labels.append(str(point.get("x")))
            for label in labels:
                cells = [label]
                for s in series:
                    match = next(
                        (p for p in s.get("points") or [] if str(p.get("x")) == label),
                        None,
                    )
                    # A gap stays a gap: the chart omits missing points rather
                    # than zero-filling them, and so does this.
                    cells.append(
                        _fmt_value(match.get("y"), chart.get("unit")) if match else "—"
                    )
                figures.append("| " + " | ".join(cells) + " |")
            figures.append("")
            for warning in chart.get("warnings") or []:
                figures.append(f"> {warning}\n")
            source = chart.get("source_name") or ""
            page = chart.get("source_page")
            figures.append(f"*Source: {source}{f', page {page}' if page else ''}.*\n")
        if figures:
            document.sections.append(
                Section(
                    slug="figures",
                    title="Figures and metrics",
                    body="\n".join(figures),
                    order=45,
                )
            )

    records = [c for c in (computations or []) if c.get("ok")]
    if records:
        lines = [
            "Every figure above that was computed rather than quoted came from "
            "one of these, and each is auditable in full.\n",
            "| Computation | Agent | Result |",
            "| --- | --- | --- |",
        ]
        for record in records:
            result = record.get("result")
            shown = str(result)[:80] if result is not None else "—"
            lines.append(
                f"| `{record.get('computation_id')}` | {record.get('agent', '')} | {shown} |"
            )
        document.sections.append(
            Section(slug="workings", title="Workings", body="\n".join(lines), order=90)
        )

    limitations = list(getattr(report, "limitations", []) or [])
    if limitations:
        document.sections.append(
            Section(
                slug="limitations",
                title="What this report does not establish",
                body="\n".join(f"- {item}" for item in limitations),
                order=95,
            )
        )

    return document
