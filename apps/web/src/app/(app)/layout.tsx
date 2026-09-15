"use client";

/**
 * The authenticated shell.
 *
 * The guard lives here rather than on each page, so a new page under `(app)/`
 * is protected by existing rather than by remembering to add a check. That
 * matters more than it sounds: the failure mode of the per-page approach is a
 * page that silently serves one workspace's data to anyone with the URL.
 *
 * This is a *client-side* guard, and it is not the security boundary — the API
 * refuses every request without a verified token, which is. This only decides
 * what to render while that is true.
 */

import { useRouter, usePathname } from "next/navigation";
import { useEffect } from "react";
import { CloudOff, Loader2, RefreshCw } from "lucide-react";

import { AppSidebar } from "@/components/app-sidebar";
import { Button } from "@/components/ui/button";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";
import { useAuth } from "@/lib/auth-context";
import { WorkspaceCountsProvider } from "@/hooks/use-workspace-counts";

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const { status, retry } = useAuth();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (status === "anonymous") {
      // Carry the intended destination so the user lands where they were
      // going, not on a generic dashboard.
      router.replace(`/login?next=${encodeURIComponent(pathname)}`);
    }
  }, [status, router, pathname]);

  if (status === "unreachable") {
    // Said plainly, with a way forward. Redirecting to /login here would be
    // actively misleading: the form posts to the same API and would fail the
    // same way, so the user would conclude their password was wrong.
    return (
      <div className="grid min-h-screen place-items-center bg-background p-6">
        <div className="flex max-w-sm flex-col items-center gap-3 text-center">
          <CloudOff className="size-5 text-muted-foreground" />
          <p className="text-sm font-medium">Can&apos;t reach the API</p>
          <p className="text-xs leading-relaxed text-muted-foreground">
            Your session could not be checked because the server did not
            answer. You are most likely still signed in. Start the API and try
            again.
          </p>
          <Button size="sm" variant="outline" onClick={() => void retry()}>
            <RefreshCw className="size-3.5" />
            Try again
          </Button>
        </div>
      </div>
    );
  }

  if (status !== "authenticated") {
    // "checking" and "anonymous" both render this. Showing the shell during
    // the check would paint an empty dashboard and then swap it out; showing
    // the login form would flash it on every reload before the silent refresh
    // completes.
    //
    // This state is now guaranteed to end: every path out of the bootstrap
    // sets a status, and the requests underneath it carry a timeout. It used
    // to be reachable forever — see `lib/auth-context`.
    return (
      <div className="grid min-h-screen place-items-center bg-background">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="size-4 animate-spin" />
          {status === "checking" ? "Restoring your session…" : "Redirecting…"}
        </div>
      </div>
    );
  }

  return (
    <WorkspaceCountsProvider>
      <SidebarProvider>
        <AppSidebar />
        <SidebarInset className="min-w-0">{children}</SidebarInset>
      </SidebarProvider>
    </WorkspaceCountsProvider>
  );
}
