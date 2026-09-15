"use client";

/**
 * One client-side fetch, done correctly.
 *
 * Written once and shared because the naive version has three bugs that are
 * easy to repeat in every page:
 *
 * 1. **`setState` synchronously inside the effect body.** That triggers a
 *    cascading render, and the React Compiler lint rule rejects it. Here the
 *    effect body only *starts* the promise; every state update happens in a
 *    settled callback, which is the sanctioned pattern.
 *
 * 2. **No cancellation.** Navigating away mid-request left the response
 *    landing on an unmounted component. A `cancelled` flag in the cleanup
 *    means a late response is dropped instead.
 *
 * 3. **A stale response overwriting a fresh one.** Two overlapping refreshes
 *    can settle out of order, and the slower-but-older one would win. Each
 *    request carries a sequence number and only the newest is allowed to
 *    write.
 */

import { useCallback, useEffect, useRef, useState } from "react";

export interface Resource<T> {
  data: T | null;
  error: unknown;
  loading: boolean;
  refresh: () => void;
  /** Replace the data locally, for an optimistic update after a mutation. */
  set: (value: T) => void;
}

export function useApiResource<T>(
  fetcher: () => Promise<T>,
  deps: readonly unknown[] = [],
): Resource<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  // Bumped to force a re-run of the effect on `refresh()`.
  const [nonce, setNonce] = useState(0);

  const latest = useRef(0);
  // The fetcher is a new closure on every render, so it is held in a ref and
  // deliberately kept out of the dependency array — including it would make
  // the effect re-run on every render and fetch in a loop.
  //
  // The ref is updated in an effect, not during render. Writing a ref while
  // rendering is not allowed: React may render a component without committing
  // it, so a render-phase write can leave the ref describing work that was
  // thrown away. `useRef(fetcher)` seeds it for the first pass, and this
  // effect is declared *before* the fetching one so it has already run by the
  // time a re-render triggers a refetch.
  const fetcherRef = useRef(fetcher);
  useEffect(() => {
    fetcherRef.current = fetcher;
  });

  useEffect(() => {
    let cancelled = false;
    const sequence = ++latest.current;

    fetcherRef.current().then(
      (value) => {
        if (cancelled || sequence !== latest.current) return;
        setData(value);
        setError(null);
        setLoading(false);
      },
      (caught: unknown) => {
        if (cancelled || sequence !== latest.current) return;
        setError(caught);
        setLoading(false);
      },
    );

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nonce, ...deps]);

  const refresh = useCallback(() => setNonce((value) => value + 1), []);
  const set = useCallback((value: T) => setData(value), []);

  return { data, error, loading, refresh, set };
}
