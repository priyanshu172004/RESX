"use client";

/**
 * The full report, section by section.
 *
 * The panels above this one are the four-minute read: five insights, the
 * recommendations, the charts. This is the rest — one section per specialist,
 * every figure it established with the source behind it and the reviewer's
 * verdict on each. It exists because that material was already in the run and
 * had nowhere to be read: the claims and verdicts were stored, shown one at a
 * time as cards, and never assembled into a document.
 *
 * Assembled server-side in `app/reporting/document.py`, in code rather than by
 * a model, so a section cannot be silently omitted and no figure can be
 * restated slightly wrong. This component only renders it.
 */

import { useMemo, useState } from "react";
import {
  ChevronRight,
  Download,
  FileSpreadsheet,
  FileText,
  FileType,
  Loader2,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { runs, type ExportFormat } from "@/lib/api";
import { cn } from "@/lib/utils";

/**
 * What each format is for, said in the menu.
 *
 * Four download buttons with only file extensions on them asks the reader to
 * know what a .docx gets them that a .pdf does not. The distinction that
 * matters is not the format, it is the job — send it, edit it, or check it.
 */
const EXPORTS: {
  format: ExportFormat;
  label: string;
  hint: string;
  Icon: typeof FileText;
}[] = [
  {
    format: "pdf",
    label: "PDF",
    hint: "Send or print — figures included",
    Icon: FileType,
  },
  {
    format: "docx",
    label: "Word",
    hint: "Edit — real headings, figures embedded",
    Icon: FileText,
  },
  {
    format: "xlsx",
    label: "Excel",
    hint: "Check — every figure's numbers, one sheet each",
    Icon: FileSpreadsheet,
  },
  {
    format: "md",
    label: "Markdown",
    hint: "Plain text",
    Icon: FileText,
  },
];

export type DocumentSection = {
  slug: string;
  title: string;
  body: string;
  order: number;
};

export type ReportDocumentPayload = {
  question: string;
  markdown: string;
  estimated_pages: number;
  sections: DocumentSection[];
};

/**
 * The small subset of Markdown the document actually emits.
 *
 * Deliberately not a Markdown library. The generator is ours and emits exactly
 * these constructs — headings, bold, tables, blockquotes, bullets, links — so
 * a renderer for those is a few lines, and pulling in a parser would also pull
 * in its HTML sanitisation question. Nothing here builds HTML from the input:
 * every branch returns React elements, so a `<script>` in a quoted source is
 * text and cannot execute.
 */
function inline(text: string, keyBase: string): React.ReactNode[] {
  const out: React.ReactNode[] = [];
  // Bold, inline code, and links, in one pass so nesting order is stable.
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/g;
  let last = 0;
  let match: RegExpExecArray | null;
  let index = 0;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > last) out.push(text.slice(last, match.index));
    const token = match[0];
    const key = `${keyBase}-${index++}`;
    if (token.startsWith("**")) {
      out.push(<strong key={key}>{token.slice(2, -2)}</strong>);
    } else if (token.startsWith("`")) {
      out.push(
        <code key={key} className="rounded bg-muted px-1 py-0.5 font-mono text-2xs">
          {token.slice(1, -1)}
        </code>,
      );
    } else {
      const label = token.slice(1, token.indexOf("]"));
      const href = token.slice(token.indexOf("(") + 1, -1);
      out.push(
        <a
          key={key}
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className="underline underline-offset-2"
        >
          {label}
        </a>,
      );
    }
    last = pattern.lastIndex;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

function Markdown({ body }: { body: string }) {
  const blocks = useMemo(() => {
    const lines = body.split("\n");
    const out: React.ReactNode[] = [];
    let i = 0;

    while (i < lines.length) {
      const line = lines[i];

      if (!line.trim()) {
        i += 1;
        continue;
      }

      if (line.startsWith("### ")) {
        out.push(
          <h4 key={i} className="mt-4 text-xs font-semibold">
            {inline(line.slice(4), `h${i}`)}
          </h4>,
        );
        i += 1;
        continue;
      }

      // A table: a header row, a separator, then body rows.
      if (line.trim().startsWith("|") && lines[i + 1]?.includes("---")) {
        const cells = (row: string) =>
          row
            .trim()
            .replace(/^\||\|$/g, "")
            .split("|")
            .map((c) => c.trim());
        const head = cells(line);
        const rows: string[][] = [];
        i += 2;
        while (i < lines.length && lines[i].trim().startsWith("|")) {
          rows.push(cells(lines[i]));
          i += 1;
        }
        out.push(
          // Its own scroll container: a wide table must never make the page
          // scroll sideways.
          <div key={`t${i}`} className="my-3 overflow-x-auto rounded-md border">
            <table className="w-full text-2xs">
              <thead className="bg-muted/50">
                <tr>
                  {head.map((cell, c) => (
                    <th key={c} className="px-2 py-1.5 text-left font-medium">
                      {inline(cell, `th${i}-${c}`)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, r) => (
                  <tr key={r} className="border-t">
                    {row.map((cell, c) => (
                      <td key={c} className="px-2 py-1.5 align-top">
                        {inline(cell, `td${i}-${r}-${c}`)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>,
        );
        continue;
      }

      if (line.trimStart().startsWith("> ")) {
        const quote: string[] = [];
        while (i < lines.length && lines[i].trimStart().startsWith("> ")) {
          quote.push(lines[i].trimStart().slice(2));
          i += 1;
        }
        out.push(
          <blockquote
            key={`q${i}`}
            className="my-2 border-l-2 pl-3 text-2xs italic leading-relaxed text-muted-foreground"
          >
            {inline(quote.join(" "), `bq${i}`)}
          </blockquote>,
        );
        continue;
      }

      if (line.trimStart().startsWith("- ")) {
        const items: string[] = [];
        while (i < lines.length && lines[i].trimStart().startsWith("- ")) {
          items.push(lines[i].trimStart().slice(2));
          i += 1;
        }
        out.push(
          <ul key={`u${i}`} className="my-2 flex flex-col gap-1">
            {items.map((item, n) => (
              <li key={n} className="flex gap-1.5 text-2xs leading-relaxed">
                <span className="text-muted-foreground">·</span>
                <span>{inline(item, `li${i}-${n}`)}</span>
              </li>
            ))}
          </ul>,
        );
        continue;
      }

      if (line.startsWith("*") && line.endsWith("*") && !line.startsWith("**")) {
        out.push(
          <p key={i} className="text-2xs italic text-muted-foreground">
            {line.slice(1, -1)}
          </p>,
        );
        i += 1;
        continue;
      }

      const paragraph: string[] = [];
      while (
        i < lines.length &&
        lines[i].trim() &&
        !lines[i].startsWith("### ") &&
        !lines[i].trimStart().startsWith("- ") &&
        !lines[i].trimStart().startsWith("> ") &&
        !lines[i].trim().startsWith("|")
      ) {
        paragraph.push(lines[i]);
        i += 1;
      }
      out.push(
        <p key={`p${i}`} className="my-2 text-xs leading-relaxed">
          {inline(paragraph.join(" "), `pp${i}`)}
        </p>,
      );
    }
    return out;
  }, [body]);

  return <div>{blocks}</div>;
}

export function ReportDocumentView({
  document: doc,
  runId,
}: {
  document: ReportDocumentPayload;
  /**
   * Needed because the exports are built server-side. The Markdown could be
   * assembled in the browser from what is already loaded, but a PDF cannot —
   * the figures have to be drawn, and drawing them twice in two languages is
   * how the two would drift apart.
   */
  runId: string;
}) {
  // Open by default. A report the reader has to click nine times to read is a
  // report they will not read, and the whole complaint this answers was that
  // the analysis produced nothing they could see.
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());

  const [downloading, setDownloading] = useState<ExportFormat | null>(null);
  const [failed, setFailed] = useState<string | null>(null);

  const download = async (format: ExportFormat) => {
    setDownloading(format);
    setFailed(null);
    try {
      await runs.exportReport(runId, format);
    } catch (error) {
      // Shown, not swallowed. A download that silently does nothing is read as
      // a broken button, and the reader retries it rather than telling anyone.
      setFailed(
        error instanceof Error
          ? error.message
          : `the ${format.toUpperCase()} could not be generated`,
      );
    } finally {
      setDownloading(null);
    }
  };

  const toggle = (slug: string) =>
    setCollapsed((current) => {
      const next = new Set(current);
      if (next.has(slug)) next.delete(slug);
      else next.add(slug);
      return next;
    });

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-col gap-0.5">
          <p className="flex items-center gap-1.5 text-xs font-medium">
            <FileText className="size-3.5" />
            Full report
          </p>
          <p className="text-2xs text-muted-foreground">
            {doc.sections.length} sections · about {doc.estimated_pages} pages ·
            every figure with its source and the reviewer&apos;s verdict
          </p>
        </div>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button size="sm" variant="outline" disabled={downloading !== null}>
              {downloading ? (
                <Loader2 className="size-3.5 animate-spin" />
              ) : (
                <Download className="size-3.5" />
              )}
              {downloading ? `Building ${downloading.toUpperCase()}…` : "Download"}
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-64">
            {EXPORTS.map(({ format, label, hint, Icon }) => (
              <DropdownMenuItem
                key={format}
                onSelect={() => void download(format)}
                className="flex items-start gap-2"
              >
                <Icon className="mt-0.5 size-3.5 shrink-0" />
                <span className="flex flex-col">
                  <span className="text-xs font-medium">{label}</span>
                  <span className="text-2xs text-muted-foreground">{hint}</span>
                </span>
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {failed ? (
        <p role="alert" className="text-2xs text-critical">
          {failed}
        </p>
      ) : null}

      <div className="flex flex-col gap-2">
        {doc.sections.map((section) => {
          const isCollapsed = collapsed.has(section.slug);
          return (
            <section
              key={section.slug}
              className="rounded-lg border bg-background"
            >
              <button
                type="button"
                onClick={() => toggle(section.slug)}
                className="flex w-full items-center gap-2 px-3 py-2 text-left"
                aria-expanded={!isCollapsed}
              >
                <ChevronRight
                  className={cn(
                    "size-3.5 shrink-0 text-muted-foreground transition-transform",
                    !isCollapsed && "rotate-90",
                  )}
                />
                <span className="flex-1 text-xs font-medium">{section.title}</span>
                <span className="text-2xs text-muted-foreground">
                  {section.body.length.toLocaleString()} chars
                </span>
              </button>
              {!isCollapsed && (
                <div className="border-t px-3 py-2">
                  <Markdown body={section.body} />
                </div>
              )}
            </section>
          );
        })}
      </div>
    </div>
  );
}
