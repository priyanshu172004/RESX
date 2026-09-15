"""A load test that measures the things that actually break.

Not a throughput contest. This API's slow paths are known and they are not the
read routes — ingestion is CPU-bound with OCR at 300 DPI, and a run is minutes
of rate-limited model calls. Hammering `/health` with a thousand requests
proves nothing anyone was worried about.

What it measures instead is **starvation**: whether the read routes stay
responsive while an upload is being processed. That is not hypothetical. It
happened, and the symptom was the dashboard timing out for no visible reason —
`ingest_file` is synchronous and was being called directly from an `async def`
handler, so it blocked the event loop for the whole extraction and every other
request queued behind it. The upload was not slow because the dashboard was
busy; the dashboard was starved by the upload.

So the shape of the test is: start an upload, then measure the dashboard while
it runs, and fail if the p95 moves.

    python scripts/loadtest.py --base http://localhost:8000 \\
        --email you@example.com --password ... [--json]

Exits non-zero when a threshold is breached, so CI can run it.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

#: A read route must answer in this long even while an ingest is running. Set
#: from what a person perceives as responsive rather than from what the server
#: is capable of: past about a second a dashboard reads as broken.
READ_P95_BUDGET_MS = 1_000.0

#: How much slower reads may get while an upload is in flight. A little is
#: honest — the box is doing real work. Three times is starvation.
STARVATION_FACTOR = 3.0

#: Rows in the generated upload. Enough that extraction takes measurable time,
#: small enough that the test finishes.
UPLOAD_ROWS = 4_000


@dataclass
class Samples:
    name: str
    durations_ms: list[float] = field(default_factory=list)
    failures: int = 0

    def record(self, started: float, ok: bool) -> None:
        self.durations_ms.append((time.perf_counter() - started) * 1000.0)
        if not ok:
            self.failures += 1

    def percentile(self, p: float) -> float:
        if not self.durations_ms:
            return 0.0
        ordered = sorted(self.durations_ms)
        index = min(len(ordered) - 1, round(p / 100.0 * (len(ordered) - 1)))
        return ordered[index]

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "requests": len(self.durations_ms),
            "failures": self.failures,
            "median_ms": round(statistics.median(self.durations_ms), 1)
            if self.durations_ms
            else 0.0,
            "p95_ms": round(self.percentile(95), 1),
            "max_ms": round(max(self.durations_ms), 1) if self.durations_ms else 0.0,
        }


def sign_in(client: httpx.Client, base: str, email: str, password: str) -> str:
    response = client.post(
        f"{base}/api/v1/auth/login", json={"email": email, "password": password}
    )
    if response.status_code != 200:
        # Registering here rather than failing: a fresh environment has no
        # account, and asking the operator to create one by hand before they
        # can measure anything is friction for no benefit.
        response = client.post(
            f"{base}/api/v1/auth/register",
            json={"email": email, "password": password, "name": "Load test"},
        )
    response.raise_for_status()
    return str(response.json()["access_token"])


def sample_reads(
    base: str, token: str, samples: Samples, stop: threading.Event, pause: float = 0.2
) -> None:
    """Poll a read route until told to stop, recording every latency."""
    headers = {"Authorization": f"Bearer {token}"}
    with httpx.Client(timeout=30.0) as client:
        while not stop.is_set():
            started = time.perf_counter()
            try:
                response = client.get(f"{base}/api/v1/documents", headers=headers)
                samples.record(started, response.status_code == 200)
            except httpx.HTTPError:
                samples.record(started, False)
            time.sleep(pause)


def build_csv(rows: int) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["quarter", "region", "revenue_inr", "margin_%", "units"])
    for i in range(rows):
        writer.writerow(
            [f"Q{i % 4 + 1} FY2025", f"R{i % 12}", 1000 + i, 30 + (i % 9), 500 + i * 3]
        )
    return buffer.getvalue().encode("utf-8")


def run(base: str, email: str, password: str) -> dict[str, Any]:
    base = base.rstrip("/")
    with httpx.Client(timeout=30.0) as client:
        token = sign_in(client, base, email, password)

    headers = {"Authorization": f"Bearer {token}"}

    # --- baseline: reads with nothing else happening --------------------
    baseline = Samples("reads (idle)")
    stop = threading.Event()
    reader = threading.Thread(target=sample_reads, args=(base, token, baseline, stop))
    reader.start()
    time.sleep(6)
    stop.set()
    reader.join()

    # --- under load: the same reads while an upload is processed --------
    under_load = Samples("reads (during ingest)")
    stop = threading.Event()
    reader = threading.Thread(target=sample_reads, args=(base, token, under_load, stop))
    reader.start()

    upload = Samples("upload")
    started = time.perf_counter()
    try:
        with httpx.Client(timeout=600.0) as client:
            response = client.post(
                f"{base}/api/v1/documents",
                headers=headers,
                files={
                    "file": (
                        f"loadtest-{int(time.time())}.csv",
                        build_csv(UPLOAD_ROWS),
                        "text/csv",
                    )
                },
            )
        upload.record(started, response.status_code in {200, 201, 202})
    except httpx.HTTPError:
        upload.record(started, False)

    stop.set()
    reader.join()

    baseline_p95 = baseline.percentile(95)
    loaded_p95 = under_load.percentile(95)
    # Guarded against a zero baseline, which happens on a very fast machine and
    # would otherwise divide by zero and report infinite starvation.
    ratio = (loaded_p95 / baseline_p95) if baseline_p95 > 1.0 else 1.0

    breaches: list[str] = []
    if loaded_p95 > READ_P95_BUDGET_MS:
        breaches.append(
            f"reads during ingest p95 {loaded_p95:.0f}ms exceeds the "
            f"{READ_P95_BUDGET_MS:.0f}ms budget"
        )
    if ratio > STARVATION_FACTOR:
        breaches.append(
            f"reads got {ratio:.1f}x slower during ingest (limit "
            f"{STARVATION_FACTOR:.1f}x) — the event loop is being blocked"
        )
    if under_load.failures:
        breaches.append(f"{under_load.failures} read(s) failed during ingest")

    return {
        "baseline": baseline.summary(),
        "under_load": under_load.summary(),
        "upload": upload.summary(),
        "starvation_ratio": round(ratio, 2),
        "breaches": breaches,
        "passed": not breaches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://localhost:8000")
    parser.add_argument("--email", default="loadtest@example.com")
    parser.add_argument("--password", default="a-long-enough-load-test-password")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    try:
        report = run(args.base, args.email, args.password)
    except httpx.HTTPError as exc:
        print(f"could not reach {args.base}: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for key in ("baseline", "under_load", "upload"):
            row = report[key]
            print(
                f"  {row['name']:<24} n={row['requests']:<4} "
                f"median={row['median_ms']:>8.1f}ms  p95={row['p95_ms']:>8.1f}ms  "
                f"failures={row['failures']}"
            )
        print(f"\n  starvation ratio: {report['starvation_ratio']}x")
        for breach in report["breaches"]:
            print(f"  FAIL {breach}")
        print("\n  PASSED" if report["passed"] else "\n  FAILED")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
