"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { Check, Loader2, ShieldAlert, TriangleAlert } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { SiteHeader } from "@/components/site-header";
import { ErrorState, PageHeader, Panel } from "@/components/page-shell";
import { auth, workspace } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";
import { useDashboard } from "@/hooks/use-workspace-counts";

export default function SettingsPage() {
  const router = useRouter();
  const { user, signOut } = useAuth();
  const { dashboard, refresh } = useDashboard();

  return (
    <>
      <SiteHeader crumbs={[{ label: "Settings" }]} />

      <main className="flex min-w-0 flex-1 flex-col gap-4 p-4">
        <PageHeader
          title="Settings"
          description="Your account, this deployment's capabilities, and destructive actions."
        />

        <Panel title="Account">
          <dl className="grid gap-3 sm:grid-cols-2">
            {[
              ["Name", user?.name],
              ["Email", user?.email],
              ["Role", user?.role],
              ["Workspace", user?.workspace_id],
            ].map(([label, value]) => (
              <div key={String(label)} className="flex flex-col gap-0.5">
                <dt className="text-2xs uppercase tracking-wide text-muted-foreground">
                  {label}
                </dt>
                <dd className="truncate font-mono text-xs">{value ?? "—"}</dd>
              </div>
            ))}
          </dl>
        </Panel>

        <ChangePassword />

        {dashboard && (
          <Panel
            title="Deployment"
            description="Read from the running API, so this reflects the process actually serving you."
          >
            <ul className="flex flex-col divide-y">
              {[
                {
                  label: "Model provider",
                  value: dashboard.capabilities.model_provider,
                  ok: dashboard.capabilities.model_ready,
                  fix: "Set GROQ_API_KEY (free at console.groq.com/keys) or ANTHROPIC_API_KEY in .env, then restart the API.",
                },
                {
                  label: "Semantic embeddings",
                  value: dashboard.capabilities.semantic_embeddings
                    ? "voyage"
                    : "hashing (offline)",
                  ok: dashboard.capabilities.semantic_embeddings,
                  fix: "Set VOYAGE_API_KEY to enable semantic retrieval. Without it, matching is lexical only and measured recall is a floor.",
                },
                {
                  label: "Web search",
                  value: dashboard.capabilities.search_ready
                    ? "enabled"
                    : "disabled",
                  ok: dashboard.capabilities.search_ready,
                  fix: "Set TAVILY_API_KEY to let the News and Market agents run. Without it their branch is reported as degraded.",
                },
                {
                  label: "Store",
                  value: dashboard.capabilities.store,
                  ok: dashboard.capabilities.store === "mongo",
                  fix: "Set MONGODB_URL for the production backend. SQLite works but has no concurrent-writer story.",
                },
              ].map((row) => (
                <li key={row.label} className="flex flex-wrap items-start gap-3 py-3">
                  {row.ok ? (
                    <Check className="mt-0.5 size-4 shrink-0 text-good" />
                  ) : (
                    <TriangleAlert className="mt-0.5 size-4 shrink-0 text-warning" />
                  )}
                  <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                    <span className="text-sm font-medium">{row.label}</span>
                    {!row.ok && (
                      <span className="text-2xs leading-relaxed text-muted-foreground">
                        {row.fix}
                      </span>
                    )}
                  </div>
                  <Badge
                    variant={row.ok ? "secondary" : "outline"}
                    className="shrink-0 font-mono text-2xs"
                  >
                    {row.value}
                  </Badge>
                </li>
              ))}
            </ul>
            <p className="mt-3 text-2xs leading-relaxed text-muted-foreground">
              Keys are read from <span className="font-mono">.env</span> by the
              API process and are never sent to the browser. Changing one needs
              an API restart — these values are resolved at boot so that a
              misconfiguration fails loudly there rather than mid-run.
            </p>
          </Panel>
        )}

        <DangerZone
          onReset={async () => {
            await workspace.reset();
            await refresh();
          }}
          onSignOut={async () => {
            await signOut();
            router.replace("/login");
          }}
        />
      </main>
    </>
  );
}

function ChangePassword() {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(false);
  const [error, setError] = useState<unknown>(null);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setDone(false);
    try {
      await auth.changePassword({
        current_password: current,
        new_password: next,
      });
      setCurrent("");
      setNext("");
      setDone(true);
    } catch (caught) {
      setError(caught);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Panel
      title="Change password"
      description="Requires your current password — a session alone is not enough to rotate a credential."
    >
      <form onSubmit={submit} className="flex flex-col gap-3 sm:max-w-sm">
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="current">Current password</Label>
          <Input
            id="current"
            type="password"
            autoComplete="current-password"
            value={current}
            onChange={(event) => setCurrent(event.target.value)}
            required
          />
        </div>
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="next">New password</Label>
          <Input
            id="next"
            type="password"
            autoComplete="new-password"
            value={next}
            onChange={(event) => setNext(event.target.value)}
            minLength={12}
            required
          />
        </div>

        {error != null && <ErrorState error={error} />}
        {done && (
          <p className="flex items-center gap-1.5 text-xs text-good">
            <Check className="size-3.5" />
            Password updated.
          </p>
        )}

        <Button type="submit" disabled={busy} className="self-start">
          {busy ? <Loader2 className="size-4 animate-spin" /> : "Update password"}
        </Button>
      </form>
    </Panel>
  );
}

function DangerZone({
  onReset,
  onSignOut,
}: {
  onReset: () => Promise<void>;
  onSignOut: () => Promise<void>;
}) {
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [result, setResult] = useState<string | null>(null);

  async function reset() {
    setBusy(true);
    setError(null);
    try {
      await onReset();
      setResult("Workspace cleared.");
      setConfirm("");
    } catch (caught) {
      setError(caught);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="flex flex-col rounded-lg border border-critical/40 bg-critical/5">
      <header className="flex items-center gap-2 border-b border-critical/30 px-4 py-3">
        <ShieldAlert className="size-4 text-critical" />
        <h2 className="text-sm font-medium">Danger zone</h2>
      </header>

      <div className="flex flex-col gap-4 p-4">
        <div className="flex flex-col gap-2">
          <p className="text-sm font-medium">Delete all corpus data</p>
          <p className="text-xs leading-relaxed text-muted-foreground">
            Removes every document, dataset, run, claim and computation in this
            workspace, and deletes the extracted dataset files from disk. Your
            account survives. This cannot be undone.
          </p>
          <div className="flex flex-wrap items-center gap-2">
            <Input
              value={confirm}
              onChange={(event) => setConfirm(event.target.value)}
              placeholder="Type DELETE to confirm"
              className="max-w-56"
            />
            <Button
              variant="destructive"
              disabled={busy || confirm !== "DELETE"}
              onClick={() => void reset()}
            >
              {busy ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                "Delete everything"
              )}
            </Button>
          </div>
          {error != null && <ErrorState error={error} />}
          {result && <p className="text-xs text-good">{result}</p>}
        </div>

        <div className="flex flex-col gap-2 border-t border-critical/30 pt-4">
          <p className="text-sm font-medium">Sign out</p>
          <p className="text-xs text-muted-foreground">
            Revokes this device&apos;s whole session chain, not just the current
            token.
          </p>
          <Button
            variant="outline"
            className="self-start"
            onClick={() => void onSignOut()}
          >
            Sign out
          </Button>
        </div>
      </div>
    </section>
  );
}
