"use client";

/**
 * Workspace counts, fetched once and shared.
 *
 * The sidebar badges and several page headers all want the same numbers. Each
 * calling its own effect would mean five `/dashboard` requests on every
 * navigation, so the fetch is hoisted into a provider and the hook reads from
 * it.
 *
 * `refresh()` is exposed because these numbers change as a result of user
 * actions — uploading a document, deleting one, starting a run. A cache with no
 * invalidation would leave the sidebar claiming three documents after the third
 * was deleted.
 */

import { createContext, useCallback, useContext, useMemo } from "react";

import { workspace, type Dashboard } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";
import { useApiResource } from "@/hooks/use-api-resource";

type Counts = {
  documents?: number;
  datasets?: number;
  runs?: number;
  claims?: number;
};

interface CountsState extends Counts {
  dashboard: Dashboard | null;
  loading: boolean;
  error: unknown;
  refresh: () => Promise<void>;
}

const CountsContext = createContext<CountsState | null>(null);

export function WorkspaceCountsProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const { status } = useAuth();

  const {
    data: dashboard,
    error,
    loading,
    refresh: reload,
  } = useApiResource<Dashboard | null>(
    // Guarded on the session: firing this while the silent refresh is still in
    // flight produces a 401 that would clear the very token being restored.
    () => (status === "authenticated" ? workspace.dashboard() : Promise.resolve(null)),
    [status],
  );

  // Kept promise-returning so callers can `await refresh()` after a mutation
  // and know the numbers they render next are the new ones.
  const refresh = useCallback(async () => {
    reload();
  }, [reload]);

  const value = useMemo<CountsState>(
    () => ({
      documents: dashboard?.summary.documents,
      datasets: dashboard?.summary.datasets,
      runs: dashboard?.summary.runs,
      claims: dashboard?.summary.claims,
      dashboard: dashboard ?? null,
      loading,
      error,
      refresh,
    }),
    [dashboard, loading, error, refresh],
  );

  return (
    <CountsContext.Provider value={value}>{children}</CountsContext.Provider>
  );
}

/**
 * Badge counts. Returns an empty object outside the provider rather than
 * throwing, so the sidebar still renders (without badges) if it is ever
 * mounted on its own — a missing number is not worth a crashed navigation.
 */
export function useWorkspaceCounts(): Counts {
  return useContext(CountsContext) ?? {};
}

/**
 * The dashboard payload, or `null` outside the provider.
 *
 * The forgiving counterpart to `useDashboard` below. The site header wants
 * these numbers but is also rendered on pages that may sit outside the
 * authenticated shell, and a header that crashes the page because a badge has
 * no data is a worse trade than a header with no badge.
 */
export function useDashboardOptional(): Dashboard | null {
  return useContext(CountsContext)?.dashboard ?? null;
}

/** The full dashboard payload plus its loading state. */
export function useDashboard(): CountsState {
  const context = useContext(CountsContext);
  if (!context) {
    throw new Error("useDashboard must be used inside <WorkspaceCountsProvider>");
  }
  return context;
}
