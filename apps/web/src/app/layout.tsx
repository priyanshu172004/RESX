import { headers } from "next/headers";
import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";

import { ThemeProvider } from "@/components/theme-provider";
import { AuthProvider } from "@/lib/auth-context";
import { TooltipProvider } from "@/components/ui/tooltip";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
  display: "swap",
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
  display: "swap",
});

/**
 * Rendered per request, so the CSP nonce can be stamped onto the scripts.
 *
 * A statically prerendered page has its HTML generated at build time, long
 * before any nonce exists — so Next cannot mark its script tags, and a
 * nonce-based CSP with 'strict-dynamic' blocks every one of them. Verified in
 * a browser: React never hydrated.
 *
 * The cost is real but small here. Every page under `(app)` is a client
 * component behind authentication and had nothing meaningful to prerender; the
 * public pages are cheap to render. Paying that to keep `script-src` free of
 * 'unsafe-inline' is the right side of the trade for an application that
 * ingests untrusted documents.
 */
export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: {
    default: "RESX — Autonomous Business Research & Analytics",
    template: "%s · RESX",
  },
  description:
    "An autonomous business research and data analytics agent. Every number cited, computed, and defended.",
  robots: { index: false, follow: false },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#fcfcfb" },
    { media: "(prefers-color-scheme: dark)", color: "#08080a" },
  ],
};

export default async function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // `next-themes` writes an inline script that sets the theme class before
  // first paint, which is what prevents a flash of the wrong theme. Under a
  // nonce CSP that script is blocked unless it carries the nonce, so the value
  // `src/proxy.ts` generated is read back here and handed to the provider.
  const nonce = (await headers()).get("x-nonce") ?? undefined;
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${geistSans.variable} ${geistMono.variable} h-full`}
    >
      <body className="min-h-full antialiased">
        <ThemeProvider nonce={nonce}>
          <AuthProvider>
            <TooltipProvider delayDuration={200}>{children}</TooltipProvider>
          </AuthProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
