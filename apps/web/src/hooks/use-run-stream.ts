"use client";

/**
 * Live run events over SSE.
 *
 * `EventSource` is not used, for one disqualifying reason: it cannot set an
 * `Authorization` header. The alternatives are putting the access token in the
 * query string — where it lands in access logs, proxy logs and the browser's
 * history — or reading the stream with `fetch`. This reads it with `fetch`.
 *
 * Resumption is handled by `Last-Event-ID`, which the API honours: on a dropped
 * connection the reconnect asks for everything after the last sequence number
 * seen, so a reconnect neither replays the log nor silently skips a node.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { API_BASE, getAccessToken, refreshSession } from "@/lib/api";

export interface RunEvent {
  seq: number;
  kind: string;
  payload: Record<string, unknown>;
  ts?: number;
}

type StreamStatus = "idle" | "connecting" | "open" | "closed" | "error";

/** Kinds that mean the run is over and the stream should not be reopened. */
const TERMINAL = new Set(["completed", "failed", "done", "run.completed", "run.failed"]);

export function useRunStream(runId: string | null, enabled = true) {
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [status, setStatus] = useState<StreamStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  // Mirrored into state as well as the ref: the ref is what the reconnect loop
  // reads, but reading `.current` during render is not allowed — it is not a
  // render input, so React cannot know to re-render when it changes.
  const [isFinished, setIsFinished] = useState(false);

  // Held in a ref rather than read from `events`: the reconnect needs the
  // latest sequence number, and closing over the state array would capture
  // whatever it was when the effect ran.
  const lastSeq = useRef(0);
  const finished = useRef(false);
  const abort = useRef<AbortController | null>(null);

  const reset = useCallback(() => {
    lastSeq.current = 0;
    finished.current = false;
    setEvents([]);
    setError(null);
    setIsFinished(false);
  }, []);

  useEffect(() => {
    if (!runId || !enabled) return;

    let cancelled = false;
    let attempt = 0;

    async function connect() {
      if (cancelled || finished.current) return;

      const controller = new AbortController();
      abort.current = controller;
      setStatus("connecting");

      try {
        let token = getAccessToken();
        if (!token) token = await refreshSession();

        const response = await fetch(`${API_BASE}/api/v1/runs/${runId}/events`, {
          headers: {
            Accept: "text/event-stream",
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
            ...(lastSeq.current
              ? { "Last-Event-ID": String(lastSeq.current) }
              : {}),
          },
          credentials: "include",
          signal: controller.signal,
        });

        if (response.status === 401) {
          // One refresh, then one retry. Looping on 401 would hammer the
          // refresh endpoint and trip its own rate limit.
          const refreshed = await refreshSession();
          if (refreshed && !cancelled) {
            attempt += 1;
            if (attempt < 3) return connect();
          }
          throw new Error("not authorised to watch this run");
        }
        if (!response.ok || !response.body) {
          throw new Error(`stream failed (${response.status})`);
        }

        setStatus("open");
        attempt = 0;

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (!cancelled) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          // SSE frames are separated by a blank line. Anything after the last
          // separator is a partial frame and must stay in the buffer.
          const frames = buffer.split("\n\n");
          buffer = frames.pop() ?? "";

          for (const frame of frames) {
            const parsed = parseFrame(frame);
            if (!parsed) continue;

            if (parsed.seq) lastSeq.current = parsed.seq;
            setEvents((previous) =>
              // Deduplicated by sequence number: a resumed stream can legally
              // resend the boundary event.
              previous.some((e) => e.seq === parsed.seq)
                ? previous
                : [...previous, parsed],
            );
            if (TERMINAL.has(parsed.kind)) {
              finished.current = true;
              setIsFinished(true);
            }
          }
        }

        setStatus(finished.current ? "closed" : "connecting");
        if (!cancelled && !finished.current) {
          // The server closed a stream that is not finished. Back off and
          // resume from the last sequence rather than starting over.
          attempt += 1;
          const delay = Math.min(1000 * 2 ** attempt, 15_000);
          setTimeout(() => void connect(), delay);
        }
      } catch (caught) {
        if (cancelled || (caught as Error)?.name === "AbortError") return;
        setError((caught as Error).message);
        setStatus("error");
        attempt += 1;
        if (attempt < 5) {
          setTimeout(() => void connect(), Math.min(1000 * 2 ** attempt, 15_000));
        }
      }
    }

    void connect();

    return () => {
      cancelled = true;
      abort.current?.abort();
    };
  }, [runId, enabled]);

  return { events, status, error, reset, finished: isFinished };
}

function parseFrame(frame: string): RunEvent | null {
  let id = 0;
  let kind = "message";
  const dataLines: string[] = [];

  for (const line of frame.split("\n")) {
    if (line.startsWith(":")) continue; // a keep-alive comment
    if (line.startsWith("id:")) id = Number(line.slice(3).trim()) || 0;
    else if (line.startsWith("event:")) kind = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }

  if (dataLines.length === 0) return null;

  const raw = dataLines.join("\n");
  try {
    const payload = JSON.parse(raw) as Record<string, unknown>;
    return {
      seq: id || Number(payload.seq) || 0,
      kind: String(payload.kind ?? kind),
      payload: (payload.payload as Record<string, unknown>) ?? payload,
      ts: typeof payload.ts === "number" ? payload.ts : undefined,
    };
  } catch {
    // Not JSON. Kept rather than dropped, so a malformed frame is visible in
    // the console instead of vanishing.
    return { seq: id, kind, payload: { message: raw } };
  }
}
