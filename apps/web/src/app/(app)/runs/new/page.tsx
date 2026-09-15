"use client";

import { Suspense, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { FileStack, Globe, Loader2, Sparkles } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { SiteHeader } from "@/components/site-header";
import {
  EmptyState,
  ErrorState,
  LoadingBlock,
  PageHeader,
  Panel,
} from "@/components/page-shell";
import { documents, runs as runsApi, type DocumentSummary } from "@/lib/api";
import { useDashboard } from "@/hooks/use-workspace-counts";
import { cn } from "@/lib/utils";
import { useApiResource } from "@/hooks/use-api-resource";

const RESEARCH_EXAMPLES = [
  "What are the main risks facing the Indian IT services sector in 2026?",
  "How has the EV battery supply chain changed over the last two years?",
  "Who are the largest players in commercial drone delivery, and how do they compare?",
  "What regulatory changes affect fintech lending in India this year?",
];

const EXAMPLES = [
  "Is this business profitable, and what are the main risks?",
  "Reconstruct the P&L and check that it reconciles.",
  "What is the revenue trend, and is it statistically significant?",
  "Where is the customer concentration risk?",
];

function NewRunForm() {
  const router = useRouter();
  const params = useSearchParams();
  const { dashboard } = useDashboard();

  // `null` means "no explicit choice yet, use the default". Storing the
  // default in state instead would mean writing derived state from an effect,
  // and it would also silently discard the user's choice on every refresh.
  const [chosen, setChosen] = useState<Set<string> | null>(null);
  const [question, setQuestion] = useState("");
  // "corpus" analyses uploaded documents; "research" answers from the web.
  // An explicit choice rather than one inferred from an empty corpus: the two
  // produce very different answers and guessing would silently pick one.
  const [mode, setMode] = useState<"corpus" | "research">("corpus");
  const [submitError, setSubmitError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);

  const preselect = params.get("doc");

  const {
    data: docs,
    error: loadError,
  } = useApiResource<DocumentSummary[]>(() => documents.list());

  const error = submitError ?? loadError;

  // The whole ready corpus by default. Analysing one document when the user
  // has five gives a silently narrower answer than they asked for; a `?doc=`
  // in the URL is an explicit request for just that one.
  const defaultSelection = useMemo(() => {
    const ready = (docs ?? []).filter((d) => d.status.startsWith("ready"));
    return new Set(
      preselect && ready.some((d) => d.doc_id === preselect)
        ? [preselect]
        : ready.map((d) => d.doc_id),
    );
  }, [docs, preselect]);

  const selected = chosen ?? defaultSelection;

  const modelReady = dashboard?.capabilities.model_ready ?? true;
  const searchReady = dashboard?.capabilities.search_ready ?? false;
  const hasCorpus = (docs ?? []).some((d) => d.status.startsWith("ready"));

  // With an empty workspace, corpus mode has nothing to offer, so research is
  // the useful default. Computed rather than stored so it cannot fight a
  // choice the user has already made.
  //
  // Derived *above* the submit handler, and that placement is the fix rather
  // than a tidy-up. Everything the page renders read `effectiveMode` while the
  // handler alone read the raw `mode` state, so on an empty workspace the UI
  // showed research mode, the user asked a web question, and the request went
  // out as `corpus_ids: []` — which the API correctly refused with "no
  // ingested documents in this workspace". The screen and the request
  // disagreed, and only the request counted.
  const effectiveMode = !hasCorpus && searchReady ? "research" : mode;
  const researching = effectiveMode === "research";

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setSubmitError(null);
    try {
      const { run_id } = await runsApi.create({
        question: question.trim(),
        ...(researching ? { research: true } : { corpus_ids: [...selected] }),
      });
      router.push(`/runs/${run_id}`);
    } catch (caught) {
      setSubmitError(caught);
      setBusy(false);
    }
  }

  const canSubmit =
    modelReady &&
    question.trim().length >= 8 &&
    (researching ? searchReady : selected.size > 0);

  return (
    <>
      <SiteHeader crumbs={[{ label: "Runs", href: "/runs" }, { label: "New" }]} />

      <main className="flex min-w-0 flex-1 flex-col gap-4 p-4">
        <PageHeader
          title="New analysis"
          description={
            researching
              ? "No upload needed. The News, Market and Risk agents gather public sources, the Critic re-derives every figure, and anything that cannot be sourced is dropped."
              : "The Manager plans the minimum set of specialists, then each writes claims against real computations. Nothing that cannot be cited survives the grounding gate."
          }
        />

        {!modelReady && (
          <div className="rounded-lg border border-warning/40 bg-warning/5 px-4 py-3 text-xs">
            No model key is configured, so a run cannot start. Set{" "}
            <span className="font-mono">GROQ_API_KEY</span> in{" "}
            <span className="font-mono">.env</span> and restart the API — see{" "}
            <Link href="/settings" className="underline underline-offset-4">
              Settings
            </Link>
            .
          </div>
        )}

        {error != null && <ErrorState error={error} />}

        {docs === null ? (
          <LoadingBlock rows={2} />
        ) : docs.length === 0 && !searchReady ? (
          <EmptyState
            title="Nothing to analyse yet"
            description="Upload a document, or set TAVILY_API_KEY to research a question from public web sources instead."
            action={{ label: "Upload a document", href: "/documents" }}
          />
        ) : (
          <form onSubmit={submit} className="flex flex-col gap-4">
            <Panel
              title="Where should the answer come from?"
              description="Both modes cite every claim. They differ in what they are allowed to cite."
            >
              <div className="grid gap-3 sm:grid-cols-2">
                {(
                  [
                    {
                      value: "corpus" as const,
                      icon: FileStack,
                      title: "My documents",
                      body: "Cites page and paragraph anchors in what you uploaded. Figures are computed in the sandbox from extracted tables.",
                      disabled: !hasCorpus,
                      disabledNote: "No documents ingested yet",
                    },
                    {
                      value: "research" as const,
                      icon: Globe,
                      title: "Research the web",
                      body: "No upload needed. Cites the URL, publisher and retrieval date of each source, and the report states that web evidence is weaker.",
                      disabled: !searchReady,
                      disabledNote: "Needs TAVILY_API_KEY",
                    },
                  ]
                ).map((option) => {
                  const active = effectiveMode === option.value;
                  return (
                    <button
                      key={option.value}
                      type="button"
                      disabled={option.disabled}
                      onClick={() => setMode(option.value)}
                      className={cn(
                        "flex flex-col items-start gap-1.5 rounded-lg border p-3 text-left transition-colors",
                        active
                          ? "border-primary bg-accent/50"
                          : "border-border hover:bg-accent/30",
                        option.disabled && "cursor-not-allowed opacity-50",
                      )}
                    >
                      <span className="flex items-center gap-2 text-sm font-medium">
                        <option.icon className="size-4" />
                        {option.title}
                      </span>
                      <span className="text-2xs leading-relaxed text-muted-foreground">
                        {option.disabled ? option.disabledNote : option.body}
                      </span>
                    </button>
                  );
                })}
              </div>
            </Panel>

            <Panel title="Question">
              <div className="flex flex-col gap-3">
                <div className="flex flex-col gap-1.5">
                  <Label htmlFor="question">What do you want to know?</Label>
                  <Input
                    id="question"
                    value={question}
                    onChange={(event) => setQuestion(event.target.value)}
                    placeholder="Is this business profitable, and what are the main risks?"
                    required
                    minLength={8}
                    disabled={busy}
                  />
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {(researching ? RESEARCH_EXAMPLES : EXAMPLES).map((example) => (
                    <button
                      key={example}
                      type="button"
                      onClick={() => setQuestion(example)}
                      className="rounded-full border px-2.5 py-1 text-2xs text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
                    >
                      {example}
                    </button>
                  ))}
                </div>
              </div>
            </Panel>

            {researching ? (
              <Panel
                title="Sources"
                description="Gathered at run time by the News, Market and Risk agents."
              >
                <div className="flex flex-col gap-2 text-xs leading-relaxed text-muted-foreground">
                  <p>
                    Every claim still carries a source — the URL, the publisher
                    and the date it was retrieved. The accuracy rules do not
                    change: the model still performs no arithmetic, and a
                    finding it cannot source is dropped rather than softened.
                  </p>
                  <p>
                    What does change is how much the sourcing is worth. A public
                    page can be out of date, wrong, or written to rank rather
                    than to inform, and you cannot audit a corpus that does not
                    exist. Every research report says so in its limitations.
                  </p>
                </div>
              </Panel>
            ) : (
            <Panel
              title="Corpus"
              description="Only these documents may be cited. Narrowing the corpus narrows the answer."
            >
              <ul className="flex flex-col gap-2">
                {docs.map((doc) => {
                  const ready = doc.status.startsWith("ready");
                  return (
                    <li key={doc.doc_id} className="flex items-center gap-2.5">
                      <Checkbox
                        id={doc.doc_id}
                        checked={selected.has(doc.doc_id)}
                        disabled={!ready}
                        onCheckedChange={(checked) =>
                          setChosen((previous) => {
                            // Seeded from the default on the first toggle, so
                            // unticking one document does not clear the rest.
                            const next = new Set(previous ?? defaultSelection);
                            if (checked) next.add(doc.doc_id);
                            else next.delete(doc.doc_id);
                            return next;
                          })
                        }
                      />
                      <Label
                        htmlFor={doc.doc_id}
                        className="flex min-w-0 flex-1 items-center gap-2 text-sm font-normal"
                      >
                        <span className="truncate">{doc.source_name}</span>
                        <span className="shrink-0 text-2xs text-muted-foreground">
                          {doc.page_count} pages
                        </span>
                        {!ready && (
                          <span className="shrink-0 text-2xs text-warning">
                            {doc.status}
                          </span>
                        )}
                      </Label>
                    </li>
                  );
                })}
              </ul>
            </Panel>
            )}

            <div className="flex items-center gap-3">
              <Button type="submit" disabled={busy || !canSubmit}>
                {busy ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <>
                    <Sparkles className="size-4" />
                    Start analysis
                  </>
                )}
              </Button>
              <span className="text-2xs text-muted-foreground">
                {researching
                  ? "Researching public sources — this takes a few minutes"
                  : `${selected.size} of ${docs.length} documents selected`}
              </span>
            </div>
          </form>
        )}
      </main>
    </>
  );
}

export default function NewRunPage() {
  // `useSearchParams` needs a Suspense boundary during prerender.
  return (
    <Suspense fallback={<LoadingBlock rows={3} className="p-4" />}>
      <NewRunForm />
    </Suspense>
  );
}
