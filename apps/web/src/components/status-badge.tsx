"use client";

import {
  CircleCheck,
  CircleDashed,
  CircleSlash,
  Loader,
  ShieldAlert,
  TriangleAlert,
} from "lucide-react";

import { cn } from "@/lib/utils";
import type { StatusRole } from "@/lib/series";

/**
 * Status is never carried by colour alone — every badge ships an icon and a
 * text label (docs/06-DESIGN-SYSTEM.md §2). On the light surface `warning`
 * and `serious` are deliberately below 3:1, which is exactly why the pairing
 * is mandatory rather than a nicety.
 */

const ROLE_CLASS: Record<StatusRole, string> = {
  good: "text-good border-good/30 bg-good/10",
  warning: "text-warning border-warning/30 bg-warning/10",
  serious: "text-serious border-serious/30 bg-serious/10",
  critical: "text-critical border-critical/30 bg-critical/10",
};

const ROLE_ICON: Record<StatusRole, React.ComponentType<{ className?: string }>> = {
  good: CircleCheck,
  warning: TriangleAlert,
  serious: ShieldAlert,
  critical: CircleSlash,
};

export function StatusBadge({
  role,
  label,
  className,
}: {
  role: StatusRole;
  label: string;
  className?: string;
}) {
  const Icon = ROLE_ICON[role];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-2xs font-medium whitespace-nowrap",
        ROLE_CLASS[role],
        className,
      )}
    >
      <Icon className="size-3 shrink-0" aria-hidden />
      {label}
    </span>
  );
}

/* -- Verdicts -------------------------------------------------------------- */

const VERDICT_MAP = {
  confirmed: { role: "good" as StatusRole, label: "Confirmed" },
  contested: { role: "serious" as StatusRole, label: "Contested" },
  refuted: { role: "critical" as StatusRole, label: "Refuted" },
} as const;

export function VerdictBadge({
  verdict,
  className,
}: {
  verdict: keyof typeof VERDICT_MAP | null;
  className?: string;
}) {
  if (!verdict) {
    return (
      <span
        className={cn(
          "inline-flex items-center gap-1.5 rounded-full border border-border px-2 py-0.5 text-2xs text-muted-foreground",
          className,
        )}
      >
        <CircleDashed className="size-3 shrink-0" aria-hidden />
        Not reviewed
      </span>
    );
  }
  const { role, label } = VERDICT_MAP[verdict];
  return <StatusBadge role={role} label={label} className={className} />;
}

/* -- Run / section progress ------------------------------------------------ */

export function ProgressBadge({
  status,
  className,
}: {
  status: "done" | "in_process" | "running" | "queued" | "blocked" | "degraded";
  className?: string;
}) {
  if (status === "done") {
    return <StatusBadge role="good" label="Done" className={className} />;
  }
  if (status === "degraded") {
    return <StatusBadge role="critical" label="Degraded" className={className} />;
  }
  if (status === "blocked") {
    return <StatusBadge role="warning" label="Blocked" className={className} />;
  }

  const label = status === "queued" ? "Queued" : "In process";
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border border-border px-2 py-0.5 text-2xs text-muted-foreground whitespace-nowrap",
        className,
      )}
    >
      <Loader
        className={cn(
          "size-3 shrink-0",
          status !== "queued" && "animate-spin [animation-duration:2.4s]",
        )}
        aria-hidden
      />
      {label}
    </span>
  );
}
