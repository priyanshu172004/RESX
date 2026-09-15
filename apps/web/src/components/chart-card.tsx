"use client";

import { useState } from "react";
import { BarChart3, Table2 } from "lucide-react";

import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

export type RangeOption = { value: string; label: string; short: string };

export type ChartTable = {
  head: string[];
  rows: (string | number)[][];
  /** Column indices to right-align — numeric columns, always. */
  numericFrom?: number;
};

/**
 * The single wrapper every chart in RESX sits inside, so padding, legend
 * position, range control, and the table view are identical product-wide.
 *
 * The table view is not optional decoration. It is the non-visual equivalent
 * required by the accessibility pass, and it is also the documented relief for
 * the light-mode contrast WARN on slots 3, 4 and 5 — so when `reliefRequired`
 * is set the affordance is labelled rather than quietly available.
 */
export function ChartCard({
  title,
  description,
  ranges,
  range,
  onRangeChange,
  table,
  reliefRequired = false,
  footer,
  className,
  children,
}: {
  title: string;
  description?: string;
  ranges?: RangeOption[];
  range?: string;
  onRangeChange?: (value: string) => void;
  table?: ChartTable;
  reliefRequired?: boolean;
  footer?: React.ReactNode;
  className?: string;
  children: React.ReactNode;
}) {
  const [view, setView] = useState<"chart" | "table">("chart");

  return (
    <Card className={cn("gap-0 overflow-hidden py-0 shadow-none", className)}>
      <CardHeader className="flex-row flex-wrap items-start gap-3 border-b px-4 py-3.5 [.border-b]:pb-3.5">
        <div className="grid flex-1 gap-1">
          <CardTitle className="text-base leading-none font-semibold">
            {title}
          </CardTitle>
          {description ? (
            <CardDescription className="text-xs">{description}</CardDescription>
          ) : null}
        </div>

        <CardAction className="flex items-center gap-1.5 self-center">
          {/* Filters sit in one row above the plot, never interleaved with it. */}
          {ranges && range && onRangeChange ? (
            <>
              <ToggleGroup
                type="single"
                value={range}
                onValueChange={(v) => v && onRangeChange(v)}
                variant="outline"
                size="sm"
                className="hidden h-7 lg:flex"
              >
                {ranges.map((r) => (
                  <ToggleGroupItem
                    key={r.value}
                    value={r.value}
                    className="px-2.5 text-xs"
                  >
                    {r.label}
                  </ToggleGroupItem>
                ))}
              </ToggleGroup>

              <Select value={range} onValueChange={onRangeChange}>
                <SelectTrigger
                  size="sm"
                  className="h-7 w-[132px] text-xs lg:hidden"
                  aria-label="Select a time range"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {ranges.map((r) => (
                    <SelectItem key={r.value} value={r.value} className="text-xs">
                      {r.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : null}

          {table ? (
            <Tooltip>
              <TooltipTrigger asChild>
                <ToggleGroup
                  type="single"
                  value={view}
                  onValueChange={(v) => v && setView(v as "chart" | "table")}
                  variant="outline"
                  size="sm"
                  className="h-7"
                >
                  <ToggleGroupItem
                    value="chart"
                    aria-label="Chart view"
                    className="px-2"
                  >
                    <BarChart3 className="size-3.5" />
                  </ToggleGroupItem>
                  <ToggleGroupItem
                    value="table"
                    aria-label="Table view"
                    className="px-2"
                  >
                    <Table2 className="size-3.5" />
                  </ToggleGroupItem>
                </ToggleGroup>
              </TooltipTrigger>
              <TooltipContent side="bottom" className="max-w-56">
                {reliefRequired
                  ? "Table view — the non-visual equivalent, and the documented relief for this palette in light mode."
                  : "Switch between the chart and its underlying values."}
              </TooltipContent>
            </Tooltip>
          ) : null}
        </CardAction>
      </CardHeader>

      <CardContent className="px-2 pt-4 pb-2 sm:px-4 sm:pt-5">
        {view === "chart" || !table ? (
          children
        ) : (
          /* Wide content scrolls inside its own container so the page body
             never scrolls horizontally. */
          <div className="max-h-[280px] overflow-auto rounded-md border">
            <Table>
              <TableHeader className="sticky top-0 bg-muted">
                <TableRow>
                  {table.head.map((h, i) => (
                    <TableHead
                      key={h}
                      className={cn(
                        "h-8 text-2xs whitespace-nowrap",
                        table.numericFrom !== undefined &&
                          i >= table.numericFrom &&
                          "text-right",
                      )}
                    >
                      {h}
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {table.rows.map((row, ri) => (
                  <TableRow key={ri}>
                    {row.map((cell, ci) => (
                      <TableCell
                        key={ci}
                        className={cn(
                          "py-1.5 text-xs whitespace-nowrap",
                          table.numericFrom !== undefined &&
                            ci >= table.numericFrom &&
                            "text-right",
                        )}
                      >
                        {cell}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </CardContent>

      {footer ? (
        <CardFooter className="flex-col items-start gap-1 border-t px-4 py-3 text-sm">
          {footer}
        </CardFooter>
      ) : null}
    </Card>
  );
}
