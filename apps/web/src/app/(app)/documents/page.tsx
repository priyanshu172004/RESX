"use client";

/**
 * The corpus page: upload, inspect, delete.
 *
 * Upload and analyse are one action rather than two. Asking the user to upload,
 * find the document in a list, and then start a run is three chances to lose
 * the intent — and the API accepts the question alongside the file precisely so
 * that it cannot be dropped between the two calls.
 */

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useRef, useState } from "react";
import {
  CircleCheck,
  FileStack,
  Loader2,
  Sparkles,
  Trash2,
  TriangleAlert,
  Upload,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
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
import { documents, type DocumentSummary, type IngestResponse } from "@/lib/api";
import { useDashboard } from "@/hooks/use-workspace-counts";
import { cn } from "@/lib/utils";
import { useApiResource } from "@/hooks/use-api-resource";

const ACCEPTED = ".pdf,.csv,.tsv,.txt,.md,.json,.xlsx,.xlsm,.docx";

export default function DocumentsPage() {
  const router = useRouter();
  const { refresh: refreshCounts } = useDashboard();

  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [result, setResult] = useState<IngestResponse | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [actionError, setActionError] = useState<unknown>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const {
    data: docs,
    error: loadError,
    refresh: load,
  } = useApiResource<DocumentSummary[]>(() => documents.list());

  // Upload and delete failures are shown next to the control that caused them,
  // so they are tracked separately from the list's own load failure.
  const error = actionError ?? loadError;

  const upload = useCallback(
    async (file: File) => {
      setBusy(true);
      setActionError(null);
      setResult(null);
      try {
        const response = await documents.upload(file, question);
        setResult(response);
        setQuestion("");
        load();
        await refreshCounts();
        if (response.analysis?.started && response.analysis.run_id) {
          // Straight to the live stream. The run is already going; leaving the
          // user on this page would hide the thing they asked for.
          router.push(`/runs/${response.analysis.run_id}`);
        }
      } catch (caught) {
        setActionError(caught);
      } finally {
        setBusy(false);
        if (inputRef.current) inputRef.current.value = "";
      }
    },
    [question, load, refreshCounts, router],
  );

  async function remove(docId: string) {
    setDeleting(docId);
    setActionError(null);
    try {
      await documents.remove(docId);
      load();
      await refreshCounts();
    } catch (caught) {
      setActionError(caught);
    } finally {
      setDeleting(null);
    }
  }

  return (
    <>
      <SiteHeader crumbs={[{ label: "Corpus" }, { label: "Documents" }]} />

      <main className="flex min-w-0 flex-1 flex-col gap-4 p-4">
        <PageHeader
          title="Documents"
          description="PDF, XLSX, CSV, TSV, DOCX and plain text. The type is decided by magic bytes, not the file extension, so a renamed executable is rejected."
        />

        {/* -- upload ------------------------------------------------- */}
        <Panel
          title="Upload and analyse"
          description="Add a question to start an analysis the moment ingestion finishes."
        >
          <div className="flex flex-col gap-3">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="question">Question (optional)</Label>
              <Input
                id="question"
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Is this business profitable, and what are the main risks?"
                disabled={busy}
              />
              <p className="text-2xs text-muted-foreground">
                Leave this blank to ingest only. You can ask questions later
                from the Runs page.
              </p>
            </div>

            <div
              onDragOver={(event) => {
                event.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => {
                event.preventDefault();
                setDragging(false);
                const file = event.dataTransfer.files?.[0];
                if (file) void upload(file);
              }}
              className={cn(
                "flex flex-col items-center justify-center rounded-lg border border-dashed px-6 py-10 text-center transition-colors",
                dragging ? "border-primary bg-accent/40" : "border-border",
                busy && "opacity-60",
              )}
            >
              {busy ? (
                <>
                  <Loader2 className="size-5 animate-spin text-muted-foreground" />
                  <p className="mt-3 text-sm">Extracting and indexing…</p>
                  <p className="mt-1 text-2xs text-muted-foreground">
                    Tables are extracted twice and compared. Large PDFs take a
                    moment.
                  </p>
                </>
              ) : (
                <>
                  <Upload className="size-5 text-muted-foreground" />
                  <p className="mt-3 text-sm">
                    Drop a file here, or{" "}
                    <button
                      type="button"
                      className="font-medium underline underline-offset-4"
                      onClick={() => inputRef.current?.click()}
                    >
                      choose one
                    </button>
                  </p>
                  <p className="mt-1 text-2xs text-muted-foreground">
                    Up to 100 MB. Nothing leaves your workspace.
                  </p>
                </>
              )}
              <input
                ref={inputRef}
                type="file"
                accept={ACCEPTED}
                className="hidden"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) void upload(file);
                }}
              />
            </div>

            {error != null && <ErrorState error={error} />}

            {result && <IngestReport report={result} />}
          </div>
        </Panel>

        {/* -- list --------------------------------------------------- */}
        {docs === null ? (
          <LoadingBlock rows={3} />
        ) : docs.length === 0 ? (
          <EmptyState
            icon={FileStack}
            title="No documents in this workspace"
            description="Upload a report, a spreadsheet or a CSV. Extraction, chunking and retrieval all run without an API key."
          />
        ) : (
          <Panel
            title={`${docs.length} ${docs.length === 1 ? "document" : "documents"}`}
            description="Deleting a document removes its chunks and datasets too — an orphaned chunk stays retrievable and would be cited against a source you thought was gone."
          >
            <ul className="flex flex-col divide-y">
              {docs.map((doc) => (
                <li
                  key={doc.doc_id}
                  className="flex flex-wrap items-center gap-3 py-3"
                >
                  <FileStack className="size-4 shrink-0 text-muted-foreground" />
                  <div className="flex min-w-0 flex-1 flex-col">
                    <span className="truncate text-sm font-medium">
                      {doc.source_name}
                    </span>
                    <span className="font-mono text-2xs text-muted-foreground">
                      {doc.doc_id}
                    </span>
                  </div>

                  <Badge variant="secondary" className="text-2xs">
                    {doc.kind}
                  </Badge>
                  <span className="tabular text-2xs text-muted-foreground">
                    {doc.page_count} {doc.page_count === 1 ? "page" : "pages"}
                  </span>

                  {doc.warnings.length > 0 && (
                    <Badge
                      variant="outline"
                      className="gap-1 text-2xs"
                      title={doc.warnings.join("; ")}
                    >
                      <TriangleAlert className="size-2.5 text-warning" />
                      {doc.warnings.length}{" "}
                      {doc.warnings.length === 1 ? "warning" : "warnings"}
                    </Badge>
                  )}

                  <Button size="sm" variant="ghost" asChild>
                    <Link href={`/runs/new?doc=${encodeURIComponent(doc.doc_id)}`}>
                      <Sparkles className="size-3.5" />
                      Analyse
                    </Link>
                  </Button>

                  <Button
                    size="sm"
                    variant="ghost"
                    disabled={deleting === doc.doc_id}
                    onClick={() => void remove(doc.doc_id)}
                    aria-label={`Delete ${doc.source_name}`}
                  >
                    {deleting === doc.doc_id ? (
                      <Loader2 className="size-3.5 animate-spin" />
                    ) : (
                      <Trash2 className="size-3.5 text-critical" />
                    )}
                  </Button>
                </li>
              ))}
            </ul>
          </Panel>
        )}
      </main>
    </>
  );
}

function IngestReport({ report }: { report: IngestResponse }) {
  return (
    <div className="flex flex-col gap-2 rounded-lg border border-good/40 bg-good/5 px-4 py-3">
      <p className="flex items-center gap-2 text-sm font-medium">
        <CircleCheck className="size-4 text-good" />
        Ingested {report.source_name}
      </p>

      <dl className="grid grid-cols-2 gap-x-6 gap-y-1 text-2xs sm:grid-cols-4">
        {[
          ["Pages", report.pages],
          ["Chunks", report.chunks],
          ["Datasets", report.datasets.length],
          [
            "Anchor completeness",
            `${(report.anchor_completeness * 100).toFixed(1)}%`,
          ],
        ].map(([label, value]) => (
          <div key={String(label)} className="flex flex-col">
            <dt className="text-muted-foreground">{label}</dt>
            <dd className="tabular font-medium">{value}</dd>
          </div>
        ))}
      </dl>

      <p className="text-2xs text-muted-foreground">
        Embedded with{" "}
        <span className="font-mono">{report.embedder}</span> in{" "}
        {report.duration_ms} ms.
        {!report.embedded &&
          " Embeddings were skipped, so retrieval will be lexical only."}
      </p>

      {report.warnings.length > 0 && (
        <ul className="flex flex-col gap-1">
          {report.warnings.map((warning) => (
            <li
              key={warning}
              className="flex items-start gap-1.5 text-2xs text-muted-foreground"
            >
              <TriangleAlert className="mt-px size-3 shrink-0 text-warning" />
              {warning}
            </li>
          ))}
        </ul>
      )}

      {report.analysis && !report.analysis.started && (
        <p className="flex items-start gap-1.5 text-2xs text-muted-foreground">
          <TriangleAlert className="mt-px size-3 shrink-0 text-warning" />
          {report.analysis.reason}
        </p>
      )}
    </div>
  );
}
