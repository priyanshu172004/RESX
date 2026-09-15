"use client";

/**
 * One claim, with everything needed to check it.
 *
 * The `computation_id` is displayed, not hidden behind a details toggle. It is
 * the difference between a number a reader can verify and a number they have
 * to take on faith, and this whole system exists to make that difference
 * visible.
 *
 * A claim carrying a value with no computation should be impossible — the
 * database rejects it. If one ever appears, it is rendered as a defect rather
 * than styled to look normal.
 */

import { useState } from "react";
import {
  BadgeCheck,
  BookOpen,
  Calculator,
  CircleAlert,
  Quote,
  ShieldAlert,
} from "lucide-react";

import { DebateThread } from "@/components/debate-thread";
import { SourceViewer } from "@/components/source-viewer";
import { Badge } from "@/components/ui/badge";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import type { Citation, ClaimRow } from "@/lib/api";
import { cn } from "@/lib/utils";

function verdictTone(verdict: string | null) {
  switch (verdict) {
    case "confirmed":
      return { icon: BadgeCheck, className: "text-good", label: "confirmed" };
    case "refuted":
      return {
        icon: CircleAlert,
        className: "text-critical",
        label: "refuted",
      };
    case "contested":
      return {
        icon: ShieldAlert,
        className: "text-warning",
        label: "contested",
      };
    default:
      return null;
  }
}

export function ClaimCard({ claim }: { claim: ClaimRow }) {
  const [openCitation, setOpenCitation] = useState<Citation | null>(null);
  const tone = verdictTone(claim.verdict);
  const ungrounded = claim.value !== null && !claim.computation_id;
  const resolved = claim.citations.filter(
    (citation) =>
      citation.resolution === "ok" || citation.resolution === "external",
  );

  return (
    <article
      className={cn(
        "flex flex-col gap-2 rounded-lg border p-3",
        ungrounded ? "border-critical/50 bg-critical/5" : "bg-background",
      )}
    >
      <header className="flex flex-wrap items-center gap-2">
        <Badge variant="secondary" className="text-2xs">
          {claim.agent}
        </Badge>

        {tone && (
          // The verdict word opens the argument behind it. Shown alone it asks
          // the reader to accept a judgement on faith, which is the one thing
          // this system exists to remove.
          <Popover>
            <PopoverTrigger asChild>
              <button
                type="button"
                className={cn(
                  "flex items-center gap-1 text-2xs underline underline-offset-2",
                  tone.className,
                )}
              >
                <tone.icon className="size-3" />
                {tone.label}
              </button>
            </PopoverTrigger>
            <PopoverContent align="start" className="w-96">
              <p className="mb-2 text-2xs font-medium">Review</p>
              <DebateThread
                verdicts={claim.verdicts ?? []}
                claimValue={claim.value}
              />
            </PopoverContent>
          </Popover>
        )}

        <span className="ml-auto flex items-center gap-2">
          <span
            className="tabular text-2xs text-muted-foreground"
            title="The agent's stated confidence in this claim"
          >
            {(claim.confidence * 100).toFixed(0)}% confident
          </span>
        </span>
      </header>

      <p className="text-sm leading-relaxed">{claim.statement}</p>

      {claim.value !== null && (
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <span
            data-slot="kpi-value"
            className="text-lg font-semibold tracking-tight"
          >
            {claim.currency ? `${claim.currency} ` : ""}
            {claim.value}
            {claim.unit && claim.unit !== claim.currency
              ? ` ${claim.unit}`
              : ""}
          </span>
          {claim.period && (
            <span className="text-2xs text-muted-foreground">
              {claim.period}
            </span>
          )}
        </div>
      )}

      <footer className="flex flex-wrap items-center gap-2">
        {claim.computation_id ? (
          <span
            className="flex items-center gap-1 font-mono text-2xs text-muted-foreground"
            title="The sandboxed computation that produced this number"
          >
            <Calculator className="size-3" />
            {claim.computation_id}
          </span>
        ) : ungrounded ? (
          <span className="flex items-center gap-1 text-2xs text-critical">
            <CircleAlert className="size-3" />
            numeric claim with no computation — this should be impossible
          </span>
        ) : null}

        {claim.citations.length > 0 && (
          <Popover>
            <PopoverTrigger asChild>
              <button
                type="button"
                className="flex items-center gap-1 rounded border px-1.5 py-0.5 text-2xs text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
              >
                <Quote className="size-2.5" />
                {resolved.length}/{claim.citations.length} cited
              </button>
            </PopoverTrigger>
            <PopoverContent align="start" className="w-96">
              <ul className="flex flex-col gap-3">
                {claim.citations.map((citation) => (
                  <li
                    key={citation.citation_id}
                    className="flex flex-col gap-1"
                  >
                    <span className="flex items-center gap-2">
                      <span className="font-mono text-2xs text-muted-foreground">
                        {citation.doc_id?.replace(/^doc_/, "").slice(0, 8) ??
                          "external"}
                        {citation.page != null && ` p.${citation.page}`}
                        {citation.para_idx != null && ` ¶${citation.para_idx}`}
                      </span>
                      <Badge
                        variant={
                          citation.resolution === "ok" ||
                          citation.resolution === "external"
                            ? "secondary"
                            : "destructive"
                        }
                        className="text-2xs"
                      >
                        {citation.resolution}
                      </Badge>
                      {citation.match_score != null && (
                        <span className="tabular text-2xs text-muted-foreground">
                          {citation.match_score.toFixed(2)}
                        </span>
                      )}
                    </span>
                    <blockquote className="border-l-2 pl-2 text-xs leading-relaxed text-muted-foreground">
                      {citation.quote}
                    </blockquote>
                    {citation.doc_id && citation.page != null && (
                      // The citation stops being a dead end here. Checking a
                      // quote used to mean opening the original PDF by hand
                      // and scrolling to the page, which is a verification
                      // step nobody performs.
                      <button
                        type="button"
                        onClick={() => setOpenCitation(citation)}
                        className="flex w-fit items-center gap-1 text-2xs text-muted-foreground underline underline-offset-2 hover:text-foreground"
                      >
                        <BookOpen className="size-2.5" />
                        Open page {citation.page}
                      </button>
                    )}
                  </li>
                ))}
              </ul>
            </PopoverContent>
          </Popover>
        )}
      </footer>

      <SourceViewer
        citation={openCitation}
        open={openCitation !== null}
        onOpenChange={(next) => {
          if (!next) setOpenCitation(null);
        }}
      />
    </article>
  );
}
