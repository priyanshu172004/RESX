import * as React from "react";

const MOBILE_BREAKPOINT = 768;
const QUERY = `(max-width: ${MOBILE_BREAKPOINT - 1}px)`;

/**
 * Tracks the mobile breakpoint via `useSyncExternalStore`.
 *
 * A media query is an external store, which is exactly what this hook is for.
 * The shadcn default read the width inside an effect and called `setState`
 * synchronously in the effect body — that produces a cascading render on every
 * mount and React Compiler rejects it. Subscribing instead gives the correct
 * value on the first client render, with no extra pass.
 *
 * The third argument is the server snapshot: desktop-first, matching the
 * product's stated bias toward large screens.
 */
function subscribe(onChange: () => void): () => void {
  const mql = window.matchMedia(QUERY);
  mql.addEventListener("change", onChange);
  return () => mql.removeEventListener("change", onChange);
}

export function useIsMobile(): boolean {
  return React.useSyncExternalStore(
    subscribe,
    () => window.matchMedia(QUERY).matches,
    () => false,
  );
}
