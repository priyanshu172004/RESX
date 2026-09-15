"use client";

/**
 * The cited page, with the quote found in it.
 *
 * A citation was a dead end. It showed a document id, a page number and the
 * quoted span, and there it stopped — checking whether the quote really said
 * that, in that context, meant opening the original PDF by hand and scrolling
 * to the page. So nobody did, which quietly makes every citation decorative.
 *
 * The whole premise of this system is that a reader can verify a figure, and a
 * verification step that costs three minutes is one that does not happen.
 *
 * What it does *not* do is render the PDF. The claim was made against the
 * extracted text — that is what the agent read and what the anchor points at —
 * so showing the extraction is showing the evidence. A page image would look
 * more authoritative while being one step further from what was actually
 * quoted, and would hide extraction faults instead of exposing them.
 */

import { useMemo } from "react";
import { ExternalLink, FileText, Loader2, TriangleAlert } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { useApiResource } from "@/hooks/use-api-resource";
import { documents, type Citation } from "@/lib/api";

/**
 * Collapse whitespace so a quote extracted from a table still matches the page.
 *
 * Table chunks are pipe-joined and page text is space-separated, so a faithful
 * quote frequently fails a literal search against the page it came from. The
 * separators are folded to a single space on both sides before comparing —
 * the same normalisation the anchor verifier applies server-side, for the same
 * reason.
 */
function normalise(text: string): string {
  return text
    .replace(/[|\s]+/g, " ")
    .trim()
    .toLowerCase();
}

/** Where the quote sits in the page text, or null when it cannot be located. */
function locate(
  page: string,
  quote: string,
): { start: number; end: number } | null {
  const trimmed = quote.trim();
  if (!trimmed) return null;

  const direct = page.indexOf(trimmed);
  if (direct >= 0) return { start: direct, end: direct + trimmed.length };

  // Fall back to a normalised search, mapping the hit back to real offsets by
  // walking both strings together. Needed because the literal form differs even
  // when the content is identical.
  const flatPage = normalise(page);
  const flatQuote = normalise(trimmed);
  const at = flatPage.indexOf(flatQuote);
  if (at < 0) return null;

  let seen = 0;
  let start = -1;
  let end = page.length;
  let previousWasSpace = false;

  for (let i = 0; i < page.length; i += 1) {
    const isSeparator = /[|\s]/.test(page[i]);
    if (isSeparator) {
      if (previousWasSpace) continue;
      previousWasSpace = true;
    } else {
      previousWasSpace = false;
    }
    if (seen === at && start < 0) start = i;
    if (seen === at + flatQuote.length) {
      end = i;
      break;
    }
    seen += 1;
  }
  return start < 0 ? null : { start, end };
}

export function SourceViewer({
  citation,
  open,
  onOpenChange,
}: {
  citation: Citation | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        className="flex w-full flex-col gap-3 sm:max-w-2xl"
      >
        {/*
          Mounted only while open, and keyed by the citation.

          That is what keeps the fetch out of an "if (open)" effect: the panel
          starts loading because it came into existence, not because a flag
          changed. Keying by citation also means opening a second citation
          cannot briefly show the first one's page — the component is replaced
          rather than updated.
        */}
        {open && citation ? (
          <SourcePanel
            key={`${citation.doc_id}:${citation.page}:${citation.citation_id}`}
            citation={citation}
          />
        ) : null}
      </SheetContent>
    </Sheet>
  );
}

function SourcePanel({ citation }: { citation: Citation }) {
  const docId = citation.doc_id;
  const page = citation.page;

  // The shared hook rather than a hand-rolled effect: it already owns the
  // out-of-order-response guard, which matters here because a reader clicking
  // through several citations quickly can otherwise be shown the page they
  // asked for first.
  const { data, error, loading } = useApiResource(
    () =>
      docId && page != null
        ? documents.page(docId, page)
        : Promise.resolve(null),
    [docId, page],
  );
  const text = data?.text ?? null;

  const parts = useMemo(() => {
    if (!text || !citation) return null;
    const found = locate(text, citation.quote);
    if (!found) return null;
    return {
      before: text.slice(0, found.start),
      match: text.slice(found.start, found.end),
      after: text.slice(found.end),
    };
  }, [text, citation]);

  return (
    <>
      <SheetHeader className="gap-1">
        <SheetTitle className="flex items-center gap-2 text-sm">
          <FileText className="size-4" />
          {citation?.doc_id?.replace(/^doc_/, "") ?? "Source"}
          {page != null && (
            <Badge variant="secondary" className="text-2xs">
              page {page}
            </Badge>
          )}
        </SheetTitle>
        <SheetDescription className="text-2xs">
          The extracted text the claim was made against — not a picture of the
          page, because the extraction is what the agent actually read.
        </SheetDescription>
      </SheetHeader>

      {citation?.url && !citation.doc_id ? (
        <div className="flex flex-col gap-2 px-4">
          <p className="text-xs text-muted-foreground">
            This citation points at a public page rather than an uploaded
            document, so there is no extracted text to show.
          </p>
          <Button asChild size="sm" variant="outline">
            <a href={citation.url} target="_blank" rel="noopener noreferrer">
              <ExternalLink className="size-3.5" />
              Open the source
            </a>
          </Button>
        </div>
      ) : null}

      {loading && (
        <p className="flex items-center gap-2 px-4 text-xs text-muted-foreground">
          <Loader2 className="size-3.5 animate-spin" />
          Loading page {page}…
        </p>
      )}

      {error ? (
        <p role="alert" className="px-4 text-xs text-critical">
          {error instanceof Error
            ? error.message
            : "the page could not be loaded"}
        </p>
      ) : null}

      {text && !parts && (
        // Said plainly rather than hidden. A quote that cannot be found on
        // the page it cites is the single most important thing this panel can
        // tell a reader, and showing the page without comment would imply the
        // quote was verified.
        <p className="mx-4 flex items-start gap-2 rounded-md border border-warning/40 bg-warning/5 p-2 text-2xs leading-relaxed">
          <TriangleAlert className="mt-px size-3 shrink-0" />
          The quoted text was not found on this page. The page is shown below
          unchanged so you can judge it yourself.
        </p>
      )}

      {text && (
        <div className="mx-4 mb-4 flex-1 overflow-y-auto rounded-md border bg-background p-3">
          <pre className="whitespace-pre-wrap break-words font-sans text-xs leading-relaxed">
            {parts ? (
              <>
                {parts.before}
                <mark className="rounded bg-warning/30 px-0.5 text-foreground">
                  {parts.match}
                </mark>
                {parts.after}
              </>
            ) : (
              text
            )}
          </pre>
        </div>
      )}
    </>
  );
}
