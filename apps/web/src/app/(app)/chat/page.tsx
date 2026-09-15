"use client";

/**
 * Grounded chat over the corpus.
 *
 * This returns the passages that answer a question, not prose about them.
 * That is a deliberate limit rather than an unfinished feature: with no model
 * in the loop, writing a summary would mean generating text that is not
 * traceable to a source — which is the exact failure the system exists to
 * prevent. The passages, with their page anchors, are the honest answer.
 */

import Link from "next/link";
import { useRef, useState } from "react";
import { Loader2, MessageSquare, Quote, Send } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { SiteHeader } from "@/components/site-header";
import {
  EmptyState,
  ErrorState,
  PageHeader,
  Panel,
} from "@/components/page-shell";
import { ClaimCard } from "@/components/claim-card";
import { chat, type ClaimRow, type Passage } from "@/lib/api";

interface Turn {
  question: string;
  passages: Passage[];
  claims: ClaimRow[];
  note: string;
}

export default function ChatPage() {
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const endRef = useRef<HTMLDivElement>(null);

  async function ask(event: React.FormEvent) {
    event.preventDefault();
    const asked = question.trim();
    if (asked.length < 2) return;

    setBusy(true);
    setError(null);
    try {
      const response = await chat.ask({ question: asked });
      setTurns((previous) => [
        ...previous,
        {
          question: asked,
          passages: response.passages,
          claims: response.claims,
          note: response.note,
        },
      ]);
      setQuestion("");
      // Scrolled after the state commit so the new turn is already laid out.
      requestAnimationFrame(() =>
        endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }),
      );
    } catch (caught) {
      setError(caught);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <SiteHeader crumbs={[{ label: "Chat" }]} />

      <main className="flex min-w-0 flex-1 flex-col gap-4 p-4">
        <PageHeader
          title="Ask the corpus"
          description="Extractive and cited. This returns the source passages that answer your question rather than a summary of them — a summary would be text no anchor supports."
        />

        {error != null && <ErrorState error={error} />}

        {turns.length === 0 && !busy && (
          <EmptyState
            icon={MessageSquare}
            title="Ask anything about your documents"
            description="Retrieval is hybrid — lexical and semantic candidates fused by reciprocal rank. Every passage comes back with its page and character span."
          />
        )}

        <div className="flex flex-col gap-5">
          {turns.map((turn, index) => (
            <section key={index} className="flex flex-col gap-3">
              <p className="flex items-start gap-2 text-sm font-medium">
                <Quote className="mt-0.5 size-3.5 shrink-0 text-muted-foreground" />
                {turn.question}
              </p>

              {turn.passages.length === 0 ? (
                <p className="text-xs text-muted-foreground">
                  Nothing in the corpus matched. Either the documents do not
                  cover this, or retrieval is running on the offline embedder
                  and missed a semantic match — check the banner on the{" "}
                  <Link
                    href="/dashboard"
                    className="underline underline-offset-4"
                  >
                    dashboard
                  </Link>
                  .
                </p>
              ) : (
                <ul className="flex flex-col gap-2">
                  {turn.passages.map((passage) => (
                    <li
                      key={passage.chunk_id}
                      className="flex flex-col gap-1.5 rounded-lg border bg-card p-3"
                    >
                      <header className="flex flex-wrap items-center gap-2">
                        <span className="font-mono text-2xs text-muted-foreground">
                          {passage.doc_id.replace(/^doc_/, "").slice(0, 8)} p.
                          {passage.page}
                          {passage.para_idx != null && ` ¶${passage.para_idx}`}
                        </span>
                        {passage.section && (
                          <span className="text-2xs text-muted-foreground">
                            {passage.section}
                          </span>
                        )}
                        <Badge variant="secondary" className="text-2xs">
                          {passage.kind}
                        </Badge>
                        <span className="ml-auto flex items-center gap-2">
                          <span className="text-2xs text-muted-foreground">
                            {passage.source}
                          </span>
                          <span className="tabular text-2xs text-muted-foreground">
                            {passage.score.toFixed(3)}
                          </span>
                        </span>
                      </header>
                      <p className="text-xs leading-relaxed">{passage.text}</p>
                    </li>
                  ))}
                </ul>
              )}

              {turn.claims.length > 0 && (
                <Panel
                  title="Related claims"
                  description="Findings from previous analyses that touch this question."
                >
                  <div className="flex flex-col gap-3">
                    {turn.claims.map((claim) => (
                      <ClaimCard key={claim.claim_id} claim={claim} />
                    ))}
                  </div>
                </Panel>
              )}
            </section>
          ))}
          <div ref={endRef} />
        </div>

        <form
          onSubmit={ask}
          className="sticky bottom-0 flex items-center gap-2 border-t bg-background/90 py-3 backdrop-blur-sm"
        >
          <Input
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="What was total revenue in the most recent year?"
            disabled={busy}
          />
          <Button type="submit" disabled={busy || question.trim().length < 2}>
            {busy ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <Send className="size-4" />
            )}
            <span className="sr-only">Ask</span>
          </Button>
        </form>

        <p className="text-2xs text-muted-foreground">
          For a synthesised report with ranked insights,{" "}
          <Link href="/runs/new" className="underline underline-offset-4">
            run a full analysis
          </Link>{" "}
          instead.
        </p>
      </main>
    </>
  );
}
