/**
 * The argument behind a verdict.
 *
 * The failure this guards against is not a crash. It is a claim that shows
 * "confirmed" with an empty thread and looks reviewed, when in fact nobody
 * checked it — which is exactly what a rate-limited Critic produces.
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { DebateThread } from "./debate-thread";
import type { VerdictRow } from "@/lib/api";

function verdict(overrides: Partial<VerdictRow> = {}): VerdictRow {
  return {
    verdict_id: "vd_1",
    claim_id: "clm_1",
    verdict: "confirmed",
    independent_value: null,
    rationale: "Re-derived from the source table and it matches.",
    debate_round: 0,
    created_at: 0,
    ...overrides,
  };
}

describe("DebateThread", () => {
  it("says plainly when nothing reviewed the claim", () => {
    // An empty thread and an unreviewed claim look identical otherwise, and
    // only one of them means "nobody checked this".
    render(<DebateThread verdicts={[]} />);
    expect(screen.getByText(/Nothing reviewed this claim/i)).toBeDefined();
    expect(screen.getByText(/unverified rather than as passed/i)).toBeDefined();
  });

  it("shows the rationale, not just the verdict word", () => {
    render(<DebateThread verdicts={[verdict()]} />);
    expect(
      screen.getByText("Re-derived from the source table and it matches."),
    ).toBeDefined();
  });

  it("reads forwards through the rounds", () => {
    // The store returns newest first. An argument only makes sense read the
    // other way: the review, then the answer, then the response.
    const rounds = [
      verdict({ verdict_id: "v2", debate_round: 1, rationale: "Second word." }),
      verdict({ verdict_id: "v1", debate_round: 0, rationale: "First word." }),
    ];
    render(<DebateThread verdicts={rounds} />);

    const text = document.body.textContent ?? "";
    expect(text.indexOf("First word.")).toBeLessThan(text.indexOf("Second word."));
  });

  it("labels the first review rather than calling it round 0", () => {
    render(<DebateThread verdicts={[verdict()]} />);
    expect(screen.getByText("first review")).toBeDefined();
  });

  it("puts the two figures side by side when they disagree", () => {
    // "Refuted" is an assertion. "Refuted, I computed 182.4 against 196.8
    // claimed" is evidence the reader can settle themselves.
    render(
      <DebateThread
        verdicts={[
          verdict({
            verdict: "refuted",
            independent_value: "182.4",
            rationale: "The total does not reconcile.",
          }),
        ]}
        claimValue="196.8"
      />,
    );
    expect(screen.getByText("182.4")).toBeDefined();
    expect(screen.getByText("196.8")).toBeDefined();
  });

  it("does not claim a disagreement when the figures agree", () => {
    render(
      <DebateThread
        verdicts={[verdict({ independent_value: "196.8" })]}
        claimValue="196.8"
      />,
    );
    expect(screen.queryByText(/claimed/)).toBeNull();
  });

  it("never styles an unrecognised verdict as good news", () => {
    // Guessing optimistically about a state we failed to parse is the one
    // error that misleads rather than merely confusing.
    const { container } = render(
      <DebateThread verdicts={[verdict({ verdict: "something-new" })]} />,
    );
    expect(container.querySelector(".text-good")).toBeNull();
  });
});
