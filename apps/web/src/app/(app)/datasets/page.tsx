"use client";

/**
 * Extracted datasets.
 *
 * The scale factor and the extractor-agreement flag are the two columns that
 * matter and they are the two that get dropped in most table UIs. A table read
 * at face value when its header said "in thousands" is off by 1000x, and that
 * error is invisible in the numbers themselves — so the applied factor is
 * shown per row rather than assumed.
 */


import { Database, TriangleAlert } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { SiteHeader } from "@/components/site-header";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  EmptyState,
  ErrorState,
  LoadingBlock,
  PageHeader,
  Panel,
} from "@/components/page-shell";
import { documents } from "@/lib/api";
import { useApiResource } from "@/hooks/use-api-resource";

export default function DatasetsPage() {
  const { data: rows, error, refresh: load } = useApiResource(
    () => documents.datasets(),
  );

  const disagreed = rows?.filter((r) => r.agreement === false).length ?? 0;
  const scaled = rows?.filter((r) => r.scale_factor !== "1").length ?? 0;

  return (
    <>
      <SiteHeader crumbs={[{ label: "Corpus" }, { label: "Datasets" }]} />

      <main className="flex min-w-0 flex-1 flex-col gap-4 p-4">
        <PageHeader
          title="Datasets"
          description="Every table found in the corpus, typed and stored so the sandbox can compute over it. These are what a computation reads — no figure is ever taken from prose."
        />

        {error != null && <ErrorState error={error} onRetry={load} />}

        {rows === null ? (
          <LoadingBlock rows={3} />
        ) : rows.length === 0 ? (
          <EmptyState
            icon={Database}
            title="No datasets yet"
            description="Datasets are created automatically from tables in an uploaded document. A text-only PDF produces none."
            action={{ label: "Upload a document", href: "/documents" }}
          />
        ) : (
          <Panel
            title={`${rows.length} ${rows.length === 1 ? "dataset" : "datasets"}`}
            description={
              disagreed > 0
                ? `${disagreed} flagged: the two extractors returned different values, so the table needs a human read before it is trusted.`
                : `${scaled} of ${rows.length} had a declared scale factor applied.`
            }
          >
            <div className="overflow-x-auto rounded-md border">
              <Table>
                <TableHeader className="bg-muted">
                  <TableRow>
                    <TableHead className="h-8 text-2xs">Dataset</TableHead>
                    <TableHead className="h-8 text-2xs">Source</TableHead>
                    <TableHead className="h-8 text-right text-2xs">Rows</TableHead>
                    <TableHead className="h-8 text-right text-2xs">Cols</TableHead>
                    <TableHead className="h-8 text-right text-2xs">Scale</TableHead>
                    <TableHead className="h-8 text-2xs">Currency</TableHead>
                    <TableHead className="h-8 text-2xs">Extractors</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {rows.map((row) => (
                    <TableRow key={row.dataset_id}>
                      <TableCell className="py-2 font-mono text-2xs whitespace-nowrap">
                        {row.dataset_id}
                      </TableCell>
                      <TableCell className="py-2 text-xs whitespace-nowrap">
                        {row.name}
                        {row.source_page != null && (
                          <span className="ml-1.5 text-muted-foreground">
                            p.{row.source_page}
                          </span>
                        )}
                      </TableCell>
                      <TableCell className="py-2 text-right text-xs tabular-nums">
                        {row.n_rows}
                      </TableCell>
                      <TableCell className="py-2 text-right text-xs tabular-nums">
                        {row.n_cols}
                      </TableCell>
                      <TableCell className="py-2 text-right text-xs tabular-nums">
                        {row.scale_factor === "1" ? (
                          <span className="text-muted-foreground">—</span>
                        ) : (
                          <span className="font-medium">×{row.scale_factor}</span>
                        )}
                      </TableCell>
                      <TableCell className="py-2 text-xs">
                        {row.currency ?? (
                          <span className="text-muted-foreground">unknown</span>
                        )}
                      </TableCell>
                      <TableCell className="py-2">
                        {row.agreement === false ? (
                          <Badge variant="outline" className="gap-1 text-2xs">
                            <TriangleAlert className="size-2.5 text-warning" />
                            disagree
                          </Badge>
                        ) : row.agreement === true ? (
                          <Badge variant="secondary" className="text-2xs">
                            agree
                          </Badge>
                        ) : (
                          <span className="text-2xs text-muted-foreground">
                            single
                          </span>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </Panel>
        )}
      </main>
    </>
  );
}
