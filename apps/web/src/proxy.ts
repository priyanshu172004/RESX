import { NextResponse, type NextRequest } from "next/server";

/**
 * The Content-Security-Policy, issued per request with a fresh nonce.
 *
 * This has to live here, in the request-interception layer, rather than in
 * `next.config.ts`, and the reason is worth recording because the first
 * attempt shipped a broken app.
 *
 * The file is `proxy.ts` because Next.js 16 renamed the convention: the
 * `middleware.ts` filename and its `middleware` export are deprecated and warn
 * on every build. Same runtime, same matcher semantics, different name.
 *
 * Setting `script-src 'self'` as a static header looks correct and is fatal:
 * Next.js emits inline `<script>` tags carrying the hydration payload, and a
 * static policy has no way to permit them. Verified in a real browser, the
 * production build failed with
 *
 *     Executing inline script violates ... 'script-src 'self''
 *     Minified React error #412
 *
 * React never hydrated. Every client component — the whole authenticated app —
 * was dead, while the page still rendered its server-side HTML and therefore
 * looked fine at a glance. A security header had silently disabled the product.
 *
 * The two ways out are `'unsafe-inline'`, which discards most of what a CSP is
 * for, or a nonce, which requires a per-request value. This is the only layer
 * where a per-request value can exist, and Next.js reads the nonce back out of
 * the request header set below and stamps it onto the scripts it generates.
 *
 * `'strict-dynamic'` comes with it by necessity: a nonced bootstrap script
 * then loads the route chunks, and without it each of those would need its own
 * nonce. It also makes host allowlists redundant, which is a genuine
 * improvement — trust follows the nonce rather than the origin.
 */

/** Paths with no HTML to protect, where a per-request nonce buys nothing. */
export const config = {
  matcher: [
    "/((?!api|_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp|ico|woff|woff2)$).*)",
  ],
};

const API_ORIGIN = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const isProduction = process.env.NODE_ENV === "production";

export function proxy(request: NextRequest) {
  // `crypto.randomUUID` is available in the edge runtime; `btoa` keeps it
  // compact and header-safe.
  const nonce = btoa(crypto.randomUUID());

  const csp = [
    "default-src 'self'",
    // The nonce covers Next's inline bootstrap; 'strict-dynamic' lets that
    // bootstrap pull in the route chunks it needs.
    //
    // `'unsafe-eval'` is development-only: React Fast Refresh and the
    // source-map machinery require it. Gated on NODE_ENV so a production
    // build cannot inherit it, because 'unsafe-eval' plus any injection is
    // arbitrary code execution.
    isProduction
      ? `script-src 'self' 'nonce-${nonce}' 'strict-dynamic'`
      : `script-src 'self' 'nonce-${nonce}' 'strict-dynamic' 'unsafe-eval'`,
    // A narrow, deliberate concession. Tailwind's runtime and the chart
    // library set inline `style` *attributes*, which a nonce cannot cover —
    // a nonce applies to elements, not attributes. Far weaker than
    // `script-src 'unsafe-inline'`, which is not granted.
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob:",
    "font-src 'self' data:",
    // The API and its event stream, named exactly. A wildcard here would let
    // an injected script exfiltrate to any origin it chose.
    `connect-src 'self' ${API_ORIGIN}`,
    "frame-ancestors 'none'",
    "form-action 'self'",
    "base-uri 'self'",
    "object-src 'none'",
    ...(isProduction ? ["upgrade-insecure-requests"] : []),
  ].join("; ");

  // Next.js looks for this header on the inbound request and applies the
  // nonce to the scripts it renders.
  const requestHeaders = new Headers(request.headers);
  requestHeaders.set("x-nonce", nonce);

  const response = NextResponse.next({ request: { headers: requestHeaders } });
  response.headers.set("Content-Security-Policy", csp);
  return response;
}
