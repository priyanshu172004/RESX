import type { NextConfig } from "next";

/**
 * The API layer was fully hardened and this one was not.
 *
 * `services/api` sets a strict CSP, HSTS, frame-ancestors, COOP/CORP and the
 * rest on every response — but those headers only protect *API* responses. The
 * document your browser actually loads, parses and executes comes from Next.js
 * on port 3000, and it was being served with no security headers at all.
 *
 * That gap is the one that matters most: clickjacking, MIME sniffing and
 * referrer leakage are attacks against the *page*, not against a JSON
 * endpoint. A hardened API behind an unhardened page is a hardened back door
 * behind an open front one.
 */

const isProduction = process.env.NODE_ENV === "production";

/**
 * The static headers.
 *
 * `Content-Security-Policy` is deliberately NOT here — it needs a fresh nonce
 * per request so that Next's inline hydration scripts can be permitted without
 * `'unsafe-inline'`, and only the request-interception layer can produce one.
 * See `src/proxy.ts`; setting it statically here shipped a build where React
 * never hydrated.
 */
const securityHeaders = [
  // Redundant with `frame-ancestors` for modern browsers, kept for older ones.
  { key: "X-Frame-Options", value: "DENY" },
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  {
    key: "Permissions-Policy",
    value: "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
  },
  { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
  { key: "X-Permitted-Cross-Domain-Policies", value: "none" },
  // Financial documents must never be cached by an intermediary.
  { key: "Cache-Control", value: "no-store, max-age=0" },
];

const nextConfig: NextConfig = {
  // `next dev` writes AGENTS.md and CLAUDE.md into this directory on every
  // start and re-creates them if deleted. This repo does not carry AI-tooling
  // files, so the generator is turned off at the source rather than the files
  // being deleted in a loop.
  agentRules: false,

  // The floating dev-tools button. Removed because it sits over the sidebar's
  // bottom-left controls and is not part of the product.
  devIndicators: false,

  // Never advertise the framework or its version: it turns "find a Next.js
  // app" into "find a Next.js app on a version with a known advisory".
  poweredByHeader: false,

  async headers() {
    return [
      {
        // Every route, including the static assets — a missing `nosniff` on a
        // served file is as exploitable as one on a document.
        source: "/:path*",
        headers: isProduction
          ? [
              ...securityHeaders,
              {
                key: "Strict-Transport-Security",
                // Production only. Sent over plaintext localhost it would pin
                // a developer's browser to HTTPS for a server that does not
                // speak it, and the pin outlives the mistake.
                value: "max-age=63072000; includeSubDomains; preload",
              },
            ]
          : securityHeaders,
      },
    ];
  },
};

export default nextConfig;
