"use client";

/**
 * The landing page.
 *
 * Rewritten to remove the architecture. The previous version named LangGraph,
 * MCP, Pydantic, the sandbox, the per-agent allowlist and the 0.92 resolver
 * threshold, and drew the internal layer cake as a diagram. Two problems with
 * that:
 *
 *   * **It was a targeting aid.** Naming the defences and their thresholds
 *     tells anyone who can get a document into a workspace exactly what to
 *     write against. A public threshold is a threshold to tune under.
 *
 *   * **It answered the wrong question.** A visitor wants to know what they
 *     get and whether they can trust it. "LangGraph is the brain" tells them
 *     neither, and reads as engineering talking to itself.
 *
 * So the page now argues from the reader's position: what you put in, what you
 * get back, and why you can check it. The one implementation fact that stays
 * is that figures are computed rather than generated — because that is a
 * promise the reader can verify in the product, not an internal detail.
 */

import dynamic from "next/dynamic";
import Link from "next/link";
import { useState } from "react";
import {
  ArrowRight,
  FileSearch,
  Menu,
  Quote,
  ScanSearch,
  ShieldCheck,
  Sparkles,
  X,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/theme-toggle";

/**
 * Charts render client-only.
 *
 * The illustrative data is randomised per visit, so a server render would
 * disagree with the client and React would report a hydration mismatch.
 * `ssr: false` removes the server render rather than the randomness.
 */
const Showcase = dynamic(() => import("@/components/landing/showcase"), {
  ssr: false,
  loading: () => (
    <div className="border-t border-border px-6 py-24">
      <div className="mx-auto h-[420px] max-w-6xl animate-pulse rounded-xl border border-border bg-card/50" />
    </div>
  ),
});

const NAV_LINKS = [
  { href: "#how", label: "How it works" },
  { href: "#dashboard", label: "Dashboard" },
  { href: "#trust", label: "Why trust it" },
];

function Navigation() {
  const [open, setOpen] = useState(false);

  return (
    <header className="fixed top-0 z-50 w-full border-b border-border/60 bg-background/70 backdrop-blur-xl">
      <nav className="mx-auto max-w-6xl px-6">
        <div className="flex h-16 items-center justify-between">
          <Link href="/" className="flex items-center gap-2">
            <span className="grid size-7 place-items-center rounded-md bg-foreground text-background">
              <Quote className="size-3.5" strokeWidth={2.5} />
            </span>
            <span className="text-base font-semibold tracking-tight">RESX</span>
          </Link>

          <div className="absolute left-1/2 hidden -translate-x-1/2 items-center gap-8 md:flex">
            {NAV_LINKS.map((link) => (
              <a
                key={link.href}
                href={link.href}
                className="text-sm text-muted-foreground transition-colors hover:text-foreground"
              >
                {link.label}
              </a>
            ))}
          </div>

          <div className="hidden items-center gap-2 md:flex">
            <ThemeToggle />
            <Button asChild variant="ghost" size="sm">
              <Link href="/login">Sign in</Link>
            </Button>
            <Button asChild size="sm">
              <Link href="/register">Start free</Link>
            </Button>
          </div>

          <button
            type="button"
            className="text-foreground md:hidden"
            onClick={() => setOpen((v) => !v)}
            aria-label={open ? "Close menu" : "Open menu"}
            aria-expanded={open}
          >
            {open ? <X className="size-5" /> : <Menu className="size-5" />}
          </button>
        </div>
      </nav>

      {open && (
        <div className="border-t border-border/60 bg-background/95 backdrop-blur-xl md:hidden">
          <div className="flex flex-col gap-1 px-6 py-4">
            {NAV_LINKS.map((link) => (
              <a
                key={link.href}
                href={link.href}
                className="py-2 text-sm text-muted-foreground transition-colors hover:text-foreground"
                onClick={() => setOpen(false)}
              >
                {link.label}
              </a>
            ))}
            <div className="mt-3 flex flex-col gap-2 border-t border-border/60 pt-4">
              <Button asChild variant="outline" size="sm">
                <Link href="/login">Sign in</Link>
              </Button>
              <Button asChild size="sm">
                <Link href="/register">Start free</Link>
              </Button>
            </div>
          </div>
        </div>
      )}
    </header>
  );
}

function Hero() {
  return (
    <section className="relative overflow-hidden px-6 pb-20 pt-36 md:pt-44">
      {/* A single soft wash, tinted from the chart palette so the hero and the
          charts below belong to one system. Tokens rather than hexes, so it
          follows the theme instead of fighting it. */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 -top-24 -z-10 h-[560px]"
        style={{
          background:
            "radial-gradient(ellipse 55% 45% at 50% 0%, color-mix(in oklch, var(--chart-1) 18%, transparent), transparent 72%)",
        }}
      />
      {/* A faint grid, masked to fade outward. Structure without noise. */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 -z-10 opacity-[0.035]"
        style={{
          backgroundImage:
            "linear-gradient(to right, var(--foreground) 1px, transparent 1px), linear-gradient(to bottom, var(--foreground) 1px, transparent 1px)",
          backgroundSize: "56px 56px",
          maskImage:
            "radial-gradient(ellipse 70% 55% at 50% 25%, black, transparent)",
          WebkitMaskImage:
            "radial-gradient(ellipse 70% 55% at 50% 25%, black, transparent)",
        }}
      />

      <div className="mx-auto flex max-w-4xl flex-col items-center">
        <Link
          href="/register"
          className="group mb-8 inline-flex items-center gap-2 rounded-full border border-border bg-card/60 py-1 pl-1 pr-3 backdrop-blur-sm transition-colors hover:bg-card"
        >
          <span className="rounded-full bg-foreground px-2 py-0.5 text-2xs font-medium text-background">
            Free
          </span>
          <span className="text-xs text-muted-foreground">
            Analyse your first report in minutes
          </span>
          <ArrowRight className="size-3 text-muted-foreground transition-transform group-hover:translate-x-0.5" />
        </Link>

        <h1 className="max-w-3xl text-center text-[2.5rem] font-medium leading-[1.05] tracking-[-0.045em] text-balance md:text-6xl">
          The analyst that shows
          <br />
          <span className="text-muted-foreground">its working.</span>
        </h1>

        <p className="mt-7 max-w-xl text-center text-base leading-relaxed text-muted-foreground text-pretty">
          Upload a report. Ask a question. Get five ranked findings where every
          figure links back to the page it came from — and the calculation that
          produced it.
        </p>

        <div className="mt-9 flex flex-wrap items-center justify-center gap-3">
          <Button asChild size="lg" className="h-11 px-6">
            <Link href="/register">
              Analyse a document
              <ArrowRight className="size-4" />
            </Link>
          </Button>
          <Button asChild size="lg" variant="outline" className="h-11 px-6">
            <Link href="/login">Sign in</Link>
          </Button>
        </div>

        <p className="mt-5 text-2xs text-muted-foreground">
          No credit card. Works with a free API key, or with no key at all for
          search and extraction.
        </p>

        {/* Three claims, each verifiable in the product rather than asserted. */}
        <dl className="mt-16 grid w-full max-w-3xl grid-cols-1 gap-px overflow-hidden rounded-xl border border-border bg-border sm:grid-cols-3">
          {[
            {
              value: "Every figure",
              label: "traceable to a page and a calculation",
            },
            { value: "Five", label: "ranked findings, not fifty bullet points" },
            {
              value: "Says so",
              label: "when the evidence will not support an answer",
            },
          ].map((stat) => (
            <div key={stat.value} className="bg-card px-5 py-4 text-center">
              <dt className="text-base font-medium tracking-tight">
                {stat.value}
              </dt>
              <dd className="mt-1 text-2xs leading-relaxed text-muted-foreground">
                {stat.label}
              </dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  );
}

const STEPS = [
  {
    icon: FileSearch,
    step: "01",
    title: "Add what you have",
    body: "A PDF annual report, a spreadsheet, a folder of CSVs. Tables are read as tables, and a figure written 'in thousands' is treated as thousands.",
  },
  {
    icon: Sparkles,
    step: "02",
    title: "Ask in plain language",
    body: "“Is this business profitable, and what could go wrong?” Specialists in finance, risk, market and process work the question in parallel.",
  },
  {
    icon: ScanSearch,
    step: "03",
    title: "A reviewer argues back",
    body: "Before you see anything, a second reader re-derives every figure independently and hunts for contradictions. Disagreements it cannot settle are shown as disagreements.",
  },
  {
    icon: ShieldCheck,
    step: "04",
    title: "Check any number yourself",
    body: "Click a figure to see the page it came from and the calculation behind it. Nothing in the report is something you have to take on trust.",
  },
];

function HowItWorks() {
  return (
    <section id="how" className="border-t border-border px-6 py-24">
      <div className="mx-auto max-w-6xl">
        <h2 className="max-w-2xl text-2xl font-medium tracking-tight md:text-3xl">
          Four steps, and you can audit every one of them
        </h2>

        <div className="mt-12 grid gap-x-8 gap-y-10 sm:grid-cols-2 lg:grid-cols-4">
          {STEPS.map((step) => (
            <div key={step.step} className="flex flex-col">
              <div className="flex items-center gap-2.5">
                <span className="grid size-8 place-items-center rounded-lg border border-border bg-card">
                  <step.icon className="size-4 text-muted-foreground" />
                </span>
                <span className="font-mono text-2xs text-muted-foreground">
                  {step.step}
                </span>
              </div>
              <h3 className="mt-4 text-sm font-medium">{step.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                {step.body}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function Trust() {
  return (
    <section id="trust" className="border-t border-border px-6 py-24">
      <div className="mx-auto grid max-w-6xl gap-12 lg:grid-cols-[1.05fr_1fr]">
        <div>
          <h2 className="max-w-md text-2xl font-medium tracking-tight md:text-3xl">
            An analyst who is confidently wrong is worse than no analyst at all
          </h2>
          <p className="mt-5 text-sm leading-relaxed text-muted-foreground">
            That is the whole design brief. Most tools in this space are
            fluent and unfalsifiable — they produce a confident paragraph you
            have no way to check. Three commitments follow from refusing that:
          </p>

          <dl className="mt-8 space-y-6">
            {[
              {
                title: "Numbers are calculated, not written",
                body: "Language models are unreliable at arithmetic and always sound certain about it. Every figure here is produced by code, and the calculation is kept so you can re-run it.",
              },
              {
                title: "A quote is checked against the page",
                body: "A claim's supporting quote is matched back against the document it cites. If it does not actually appear there, the claim is removed — not softened.",
              },
              {
                title: "Silence is a valid answer",
                body: "When the evidence will not carry a conclusion, the report says which question it could not answer instead of filling the space.",
              },
            ].map((item) => (
              <div key={item.title}>
                <dt className="flex items-start gap-2 text-sm font-medium">
                  <ShieldCheck className="mt-0.5 size-4 shrink-0 text-good" />
                  {item.title}
                </dt>
                <dd className="mt-1.5 pl-6 text-sm leading-relaxed text-muted-foreground">
                  {item.body}
                </dd>
              </div>
            ))}
          </dl>
        </div>

        {/* A sample finding — the actual product surface, which argues the
            case better than a description of it. */}
        <div className="lg:pt-2">
          <p className="text-2xs uppercase tracking-wide text-muted-foreground">
            What a finding looks like
          </p>

          <figure className="mt-4 rounded-xl border border-border bg-card p-4">
            <figcaption className="flex flex-wrap items-center gap-2">
              <span className="rounded bg-muted px-1.5 py-0.5 text-2xs">
                Finance
              </span>
              <span className="flex items-center gap-1 text-2xs text-good">
                <ShieldCheck className="size-3" />
                confirmed on review
              </span>
              <span className="ml-auto text-2xs text-muted-foreground">
                99% confident
              </span>
            </figcaption>

            <p className="mt-3 text-sm leading-relaxed">
              Gross margin improved to 44.97% in Q4, up 2.6 points across the
              year.
            </p>

            <p
              className="mt-3 text-2xl font-semibold tracking-tight"
              data-slot="kpi-value"
            >
              44.97%
            </p>

            <div className="mt-4 space-y-2 border-t border-border pt-3">
              <p className="text-2xs text-muted-foreground">
                <span className="font-medium text-foreground">Source</span> —
                page 3, quarterly results table
              </p>
              <blockquote className="border-l-2 border-border pl-2 text-2xs leading-relaxed text-muted-foreground">
                Q4 FY2025 | 1,690,000 | 930,000 | 362,000
              </blockquote>
              <p className="text-2xs text-muted-foreground">
                <span className="font-medium text-foreground">Calculated</span>{" "}
                — (revenue − cost) ÷ revenue, in exact decimal arithmetic
              </p>
            </div>
          </figure>

          <p className="mt-4 text-2xs leading-relaxed text-muted-foreground">
            The quote is the text as it appears in your document. The
            calculation is kept and re-runnable. Neither is something we ask
            you to believe.
          </p>
        </div>
      </div>
    </section>
  );
}

function Cta() {
  return (
    <section className="border-t border-border px-6 py-24">
      <div className="mx-auto max-w-2xl text-center">
        <h2 className="text-2xl font-medium tracking-tight md:text-3xl">
          Bring a report you already know well
        </h2>
        <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
          It is the fastest way to judge this: ask something you can already
          answer, and see whether the sources hold up.
        </p>
        <div className="mt-8 flex flex-wrap justify-center gap-3">
          <Button asChild size="lg" className="h-11 px-6">
            <Link href="/register">
              Create a workspace
              <ArrowRight className="size-4" />
            </Link>
          </Button>
          <Button asChild size="lg" variant="outline" className="h-11 px-6">
            <Link href="/login">I already have an account</Link>
          </Button>
        </div>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="border-t border-border px-6 py-10">
      <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-4 sm:flex-row">
        <p className="flex items-center gap-2 text-2xs text-muted-foreground">
          <span className="grid size-4 place-items-center rounded bg-foreground text-background">
            <Quote className="size-2" strokeWidth={3} />
          </span>
          RESX — business research you can check.
        </p>
        <div className="flex items-center gap-5">
          <Link
            href="/login"
            className="text-2xs text-muted-foreground transition-colors hover:text-foreground"
          >
            Sign in
          </Link>
          <Link
            href="/register"
            className="text-2xs text-muted-foreground transition-colors hover:text-foreground"
          >
            Start free
          </Link>
        </div>
      </div>
    </footer>
  );
}

export default function SaasLanding() {
  return (
    <main className="min-h-screen bg-background text-foreground">
      <Navigation />
      <Hero />
      <HowItWorks />
      <Showcase />
      <Trust />
      <Cta />
      <Footer />
    </main>
  );
}
