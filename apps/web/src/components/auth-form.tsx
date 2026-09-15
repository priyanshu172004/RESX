"use client";

/**
 * The sign-in and sign-up form.
 *
 * One component for both because they differ by two fields and a verb, and
 * two near-identical forms drift: the password rule gets updated in one and
 * not the other, and then registration accepts what login cannot.
 *
 * Client-side validation here mirrors the server's rules but does not replace
 * them. The server is the authority — this exists only so the user is not made
 * to wait for a round trip to be told their password is eight characters short.
 */

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { ArrowRight, Loader2, Quote, TriangleAlert } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ThemeToggle } from "@/components/theme-toggle";
import { useAuth } from "@/lib/auth-context";
import { ApiError } from "@/lib/api";

/** Mirrors `Settings.password_min_length`. */
const MIN_PASSWORD = 12;

export function AuthForm({ mode }: { mode: "login" | "register" }) {
  const router = useRouter();
  const params = useSearchParams();
  const { signIn, signUp, status } = useAuth();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const isRegister = mode === "register";
  // Preserved through the redirect so a user who deep-linked to /runs/abc and
  // was bounced to sign in lands back on /runs/abc rather than the dashboard.
  const next = params.get("next") || "/dashboard";

  useEffect(() => {
    if (status === "authenticated") router.replace(next);
  }, [status, router, next]);

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);

    if (isRegister && password.length < MIN_PASSWORD) {
      setError(`Password must be at least ${MIN_PASSWORD} characters.`);
      return;
    }
    if (isRegister && !name.trim()) {
      setError("Please enter your name.");
      return;
    }

    setBusy(true);
    try {
      if (isRegister) {
        await signUp({ email, password, name });
      } else {
        await signIn(email, password);
      }
      router.replace(next);
    } catch (caught) {
      // The API's message is specific ("too many failed attempts; try again in
      // 840 seconds"). Replacing it with a generic string would discard the
      // only actionable part.
      setError(
        caught instanceof ApiError
          ? caught.message
          : "Could not reach the API. Is it running on port 8000?",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <header className="flex items-center justify-between px-6 py-5">
        <Link href="/" className="flex items-center gap-2">
          <span className="grid size-7 place-items-center rounded-md bg-primary text-primary-foreground">
            <Quote className="size-3.5" strokeWidth={2.5} />
          </span>
          <span className="text-base font-semibold tracking-tight">RESX</span>
        </Link>
        <ThemeToggle />
      </header>

      <div className="flex flex-1 items-center justify-center px-6 pb-20">
        <div className="w-full max-w-sm">
          <h1 className="text-xl font-medium tracking-tight">
            {isRegister ? "Create your workspace" : "Sign in to RESX"}
          </h1>
          <p className="mt-1.5 text-sm text-muted-foreground">
            {isRegister
              ? "Each account gets its own workspace. Documents are never shared between them."
              : "Your corpus, runs and reports are scoped to your workspace."}
          </p>

          <form onSubmit={onSubmit} className="mt-8 flex flex-col gap-4">
            {isRegister && (
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="name">Name</Label>
                <Input
                  id="name"
                  name="name"
                  autoComplete="name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Priya Sharma"
                  required
                />
              </div>
            )}

            <div className="flex flex-col gap-1.5">
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                name="email"
                type="email"
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@company.com"
                required
              />
            </div>

            <div className="flex flex-col gap-1.5">
              <div className="flex items-baseline justify-between">
                <Label htmlFor="password">Password</Label>
                {isRegister && (
                  <span className="text-2xs text-muted-foreground">
                    {MIN_PASSWORD}+ characters
                  </span>
                )}
              </div>
              <Input
                id="password"
                name="password"
                type="password"
                autoComplete={isRegister ? "new-password" : "current-password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••••••"
                required
              />
              {isRegister && (
                <p className="text-2xs leading-relaxed text-muted-foreground">
                  Length is the only rule. Composition requirements push people
                  towards predictable passwords; a long passphrase is stronger.
                </p>
              )}
            </div>

            {error != null && (
              <p
                role="alert"
                className="flex items-start gap-2 rounded-md border border-critical/40 bg-critical/10 px-3 py-2 text-xs text-foreground"
              >
                <TriangleAlert className="mt-px size-3.5 shrink-0 text-critical" />
                <span>{error}</span>
              </p>
            )}

            <Button type="submit" disabled={busy} className="mt-1">
              {busy ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <>
                  {isRegister ? "Create workspace" : "Sign in"}
                  <ArrowRight className="size-4" />
                </>
              )}
            </Button>
          </form>

          <p className="mt-6 text-sm text-muted-foreground">
            {isRegister ? (
              <>
                Already have an account?{" "}
                <Link
                  href="/login"
                  className="font-medium text-foreground underline-offset-4 hover:underline"
                >
                  Sign in
                </Link>
              </>
            ) : (
              <>
                No account yet?{" "}
                <Link
                  href="/register"
                  className="font-medium text-foreground underline-offset-4 hover:underline"
                >
                  Create a workspace
                </Link>
              </>
            )}
          </p>
        </div>
      </div>
    </div>
  );
}
