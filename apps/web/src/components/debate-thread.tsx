"use client";

/**
 * Why the Critic said what it said.
 *
 * The verdict word was shown and the argument behind it was not. A claim
 * displaying "refuted" and nothing else asks the reader to accept a judgement
 * on faith — which is the exact thing this system exists to remove. The
 * rationale, the independently derived figure and the round it came from were
 * all stored and none of them were reachable.
 *
 * Rounds run oldest first here, against the store's newest-first ordering,
 * because an argument only makes sense read forwards: the first review, then
 * the author's answer, then the reviewer's response to that.
 *
 * The independent value is the most useful line on the page when it is
 * present. "Refuted" is an assertion; "refuted, I computed 182.4 and the claim
 * says 196.8" is evidence, and the reader can settle it themselves.
 */

import { BadgeCheck, CircleAlert, Scale, ShieldAlert } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import type { VerdictRow } from "@/lib/api";
import { cn } from "@/lib/utils";

function tone(verdict: string) {
  switch (verdict) {
    case "confirmed":
      return { Icon: BadgeCheck, className: "text-good" };
    case "refuted":
      return { Icon: ShieldAlert, className: "text-critical" };
    case "contested":
      return { Icon: CircleAlert, className: "text-serious" };
    default:
      // Never "good" for something we failed to recognise: guessing
      // optimistically about a state we could not parse is the one error that
      // misleads rather than merely confuses.
      return { Icon: CircleAlert, className: "text-warning" };
  }
}

export function DebateThread({
  verdicts,
  claimValue,
}: {
  verdicts: VerdictRow[];
  /** The figure under review, so a disagreement can be shown side by side. */
  claimValue?: string | null;
}) {
  if (!verdicts.length) {
    // Stated, not omitted. An absent thread and an unreviewed claim look
    // identical otherwise, and only one of them means "nobody checked this".
    return (
      <p className="flex items-start gap-1.5 text-2xs leading-relaxed text-muted-foreground">
        <Scale className="mt-px size-3 shrink-0" />
        Nothing reviewed this claim. It is grounded in a source, but no
        independent check was made — treat it as unverified rather than as
        passed.
      </p>
    );
  }

  const ordered = [...verdicts].sort((a, b) => a.debate_round - b.debate_round);

  return (
    <ol className="flex flex-col gap-2">
      {ordered.map((verdict) => {
        const { Icon, className } = tone(verdict.verdict);
        const disagrees =
          verdict.independent_value != null &&
          claimValue != null &&
          verdict.independent_value !== claimValue;

        return (
          <li
            key={verdict.verdict_id}
            className="flex flex-col gap-1 rounded-md border bg-background p-2"
          >
            <span className="flex flex-wrap items-center gap-1.5">
              <Icon className={cn("size-3", className)} />
              <span className={cn("text-2xs font-medium", className)}>
                {verdict.verdict}
              </span>
              <Badge variant="secondary" className="text-2xs">
                {verdict.debate_round === 0
                  ? "first review"
                  : `round ${verdict.debate_round}`}
              </Badge>
            </span>

            <p className="text-2xs leading-relaxed text-muted-foreground">
              {verdict.rationale}
            </p>

            {verdict.independent_value != null && (
              <p className="flex flex-wrap items-center gap-1.5 text-2xs">
                <span className="text-muted-foreground">
                  Re-derived independently:
                </span>
                <span className="tabular font-medium">
                  {verdict.independent_value}
                </span>
                {disagrees && (
                  // The number that matters. Two figures side by side let the
                  // reader settle it instead of choosing whom to believe.
                  <span className="text-muted-foreground">
                    against{" "}
                    <span className="tabular font-medium text-foreground">
                      {claimValue}
                    </span>{" "}
                    claimed
                  </span>
                )}
              </p>
            )}
          </li>
        );
      })}
    </ol>
  );
}
