"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useSyncExternalStore } from "react";
import { Search, ShieldCheck } from "lucide-react";

import { SidebarTrigger } from "@/components/ui/sidebar";
import { Separator } from "@/components/ui/separator";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/theme-toggle";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import {
  useDashboardOptional,
  useWorkspaceCounts,
} from "@/hooks/use-workspace-counts";

/** Never fires: the platform does not change for the life of the document. */
function subscribeNever() {
  return () => {};
}

function isMacClient() {
  return /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
}

/** The server cannot know, and Ctrl is the safer default to render first. */
function isMacServer() {
  return false;
}

export function SiteHeader({
  crumbs,
}: {
  crumbs: { label: string; href?: string }[];
}) {
  const router = useRouter();
  const counts = useWorkspaceCounts();
  const dashboard = useDashboardOptional();
  // Read through `useSyncExternalStore` rather than in an effect. The value
  // never changes, `navigator` does not exist during SSR, and the server
  // snapshot below is what keeps hydration consistent — an effect would set
  // state on mount and trigger a second render for a constant.
  const mac = useSyncExternalStore(subscribeNever, isMacClient, isMacServer);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key.toLowerCase() === "k" && (event.metaKey || event.ctrlKey)) {
        event.preventDefault();
        router.push("/chat");
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [router]);

  const validity = dashboard?.summary.citation_validity;
  const citations = dashboard?.summary.total_citations ?? 0;

  return (
    <header className="sticky top-0 z-20 flex h-12 shrink-0 items-center gap-2 border-b bg-background/80 px-3 backdrop-blur-sm">
      <SidebarTrigger className="-ml-1 size-7 text-muted-foreground hover:text-foreground" />
      <Separator orientation="vertical" className="mr-1 !h-4" />

      <Breadcrumb>
        <BreadcrumbList className="text-sm">
          {crumbs.map((crumb, i) => {
            const last = i === crumbs.length - 1;
            return (
              <span key={`${crumb.label}-${i}`} className="contents">
                <BreadcrumbItem>
                  {last || !crumb.href ? (
                    <BreadcrumbPage className="max-w-64 truncate font-medium">
                      {crumb.label}
                    </BreadcrumbPage>
                  ) : (
                    <BreadcrumbLink asChild>
                      <Link href={crumb.href}>{crumb.label}</Link>
                    </BreadcrumbLink>
                  )}
                </BreadcrumbItem>
                {!last ? <BreadcrumbSeparator /> : null}
              </span>
            );
          })}
        </BreadcrumbList>
      </Breadcrumb>

      <div className="ml-auto flex items-center gap-1.5">
        {/* Citation validity is the product's core promise, so its live value
            sits in the header rather than on a metrics page. Hidden entirely
            when there is nothing cited yet: "100% of zero" reads as a
            guarantee that has not actually been tested. */}
        {validity != null && citations > 0 && (
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="mr-1 hidden items-center gap-1.5 rounded-md border px-2 py-1 text-2xs text-muted-foreground sm:flex">
                <ShieldCheck
                  className={
                    validity >= 1 ? "size-3.5 text-good" : "size-3.5 text-critical"
                  }
                />
                <span className="tabular">
                  {(validity * 100).toFixed(validity >= 1 ? 0 : 1)}% cited
                </span>
              </div>
            </TooltipTrigger>
            <TooltipContent side="bottom" className="max-w-64">
              {citations} citations checked against their source anchors across{" "}
              {counts.documents ?? 0} documents. The gate is exactly 1.00, so
              anything lower is a failure rather than a score.
            </TooltipContent>
          </Tooltip>
        )}

        <Button
          variant="outline"
          size="sm"
          className="h-7 gap-2 px-2 text-xs text-muted-foreground"
          onClick={() => router.push("/chat")}
        >
          <Search className="size-3.5" />
          <span className="hidden sm:inline">Search corpus</span>
          <kbd className="hidden rounded border bg-muted px-1 font-mono text-2xs sm:inline">
            {mac ? "⌘K" : "Ctrl K"}
          </kbd>
        </Button>

        <ThemeToggle />
      </div>
    </header>
  );
}
