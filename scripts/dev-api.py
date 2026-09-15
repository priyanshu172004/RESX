#!/usr/bin/env python
"""Start the API, freeing its port first.

Exists because of one recurring, badly-reported failure. A previous uvicorn
still holding the port makes Windows answer with:

    ERROR: [WinError 10013] An attempt was made to access a socket in a way
    forbidden by its access permissions

That reads as a permissions problem, so it sends you looking at firewalls,
antivirus and admin rights. It is almost always just a stale server — and
`--reload` makes it more likely, because an interrupted reload can leave the
child process holding the socket after the parent is gone.

So: identify what holds the port, say what it is, offer to stop it, then start.
Only processes that are actually listening on the target port are touched;
nothing is killed by name.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "services" / "api"


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    """Can we actually bind it? The only question that matters."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        # No SO_REUSEADDR: we want to know whether uvicorn will succeed, and
        # uvicorn does not set it either. Setting it here would report a busy
        # port as free.
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def holders(port: int) -> list[tuple[int, str]]:
    """`(pid, name)` for every process listening on `port`."""
    found: dict[int, str] = {}

    try:
        import psutil  # noqa: F401
    except ImportError:
        pass
    else:
        import psutil

        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr and conn.laddr.port == port and conn.pid:
                try:
                    found[conn.pid] = psutil.Process(conn.pid).name()
                except Exception:
                    found[conn.pid] = "unknown"
        return sorted(found.items())

    # No psutil: parse netstat. Available on every Windows install, and the
    # -o flag is what gives the owning pid.
    if os.name == "nt":
        try:
            out = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True, text=True, timeout=20,
            ).stdout
        except Exception:
            return []
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 5 or parts[3] != "LISTENING":
                continue
            local = parts[1]
            if local.rsplit(":", 1)[-1] != str(port):
                continue
            try:
                found[int(parts[4])] = _name_for(int(parts[4]))
            except ValueError:
                continue
    else:
        lsof = shutil.which("lsof")
        if lsof:
            try:
                out = subprocess.run(
                    [lsof, "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                    capture_output=True, text=True, timeout=20,
                ).stdout
                for token in out.split():
                    found[int(token)] = _name_for(int(token))
            except Exception:
                return []
    return sorted(found.items())


def _name_for(pid: int) -> str:
    if os.name == "nt":
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=15,
            ).stdout.strip()
            if out and "," in out:
                return out.split(",")[0].strip('"')
        except Exception:
            pass
        return "unknown"
    try:
        return subprocess.run(
            ["ps", "-p", str(pid), "-o", "comm="],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def is_serving(port: int, host: str = "127.0.0.1", timeout: float = 1.5) -> bool:
    """Does anything accept a connection on this port?

    Asked instead of `does this pid exist`, because the pid is unreliable.
    Windows attributed a socket to an exited pid on a port that was serving
    `/health` without trouble, so the launcher waited for a teardown that was
    never coming. A successful connect is unambiguous: something is there.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(timeout)
        try:
            probe.connect((host, port))
        except OSError:
            return False
    return True


def wait_for_port(port: int, host: str, seconds: float) -> bool:
    """Poll until the port can be bound, or the budget runs out."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if port_is_free(port, host):
            return True
        time.sleep(0.5)
    return port_is_free(port, host)


def stop(pid: int) -> bool:
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/F"],
                capture_output=True, timeout=20, check=False,
            )
        else:
            os.kill(pid, 15)
            time.sleep(1)
    except Exception:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--no-reload", action="store_true", help="disable the autoreloader"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="stop whatever holds the port without asking",
    )
    args = parser.parse_args()

    if not port_is_free(args.port, args.host):
        busy = holders(args.port)
        if not busy:
            # Bindable-but-not-listening: on Windows this is usually a
            # reserved range, which killing a process cannot fix.
            print(
                f"port {args.port} cannot be bound and nothing is listening on it.\n"
                f"On Windows this is usually a reserved range — check with:\n"
                f"  netsh int ipv4 show excludedportrange protocol=tcp\n"
                f"Then start on a free port:  python scripts/dev-api.py --port 8001",
                file=sys.stderr,
            )
            return 1

        # A connect attempt, not a pid lookup: the pid Windows reports for a
        # socket does not reliably correspond to a process you can find.
        serving = is_serving(args.port, args.host)
        live = busy if serving else []
        stale = [pid for pid, _ in busy] if not serving else []

        if not serving:
            # Nothing accepts a connection, yet the port will not bind. The
            # socket is being torn down and this clears on its own.
            print(
                f"port {args.port} will not bind but nothing answers on it "
                f"(socket attributed to pid "
                f"{', '.join(str(p) for p in stale) or 'unknown'}). Waiting for "
                f"the kernel to release it…"
            )
            if not wait_for_port(args.port, args.host, 45.0):
                print(
                    f"port {args.port} is still not bindable after 45s. Either "
                    f"wait a little longer or start on another port:\n"
                    f"  python scripts/dev-api.py --port 8001",
                    file=sys.stderr,
                )
                return 1
            print(f"port {args.port} released")
        else:
            print(f"port {args.port} is already in use by:")
            for pid, name in live:
                print(f"  PID {pid}  {name}")

            if not args.force:
                # Asked rather than assumed: the holder might be something the
                # developer cares about, and silently killing by port is the
                # kind of convenience that eventually kills the wrong process.
                answer = input("stop it and continue? [y/N] ").strip().lower()
                if answer not in {"y", "yes"}:
                    print("left alone. Use --port to pick another.", file=sys.stderr)
                    return 1

            for pid, name in live:
                print(f"stopping PID {pid} ({name})…")
                stop(pid)

            # Generous, because a killed server's socket lingers exactly like
            # the case above -- the previous 5s budget was the reason this
            # reported failure on a port that was about to be free.
            if not wait_for_port(args.port, args.host, 45.0):
                print(
                    f"port {args.port} is still held after 45s. Try:\n"
                    f"  python scripts/dev-api.py --port 8001",
                    file=sys.stderr,
                )
                return 1
            print(f"port {args.port} freed")

    command = [
        sys.executable, "-m", "uvicorn", "app.main:app",
        "--host", args.host, "--port", str(args.port),
        # uvicorn writes `Server:` and `Date:` at the protocol layer, below any
        # ASGI middleware, so they cannot be stripped from a response. Turning
        # the server banner off here is the only place it works, and naming the
        # framework and version is free reconnaissance for anyone matching
        # against known advisories.
        "--no-server-header",
    ]
    if not args.no_reload:
        command.append("--reload")

    print(f"$ {' '.join(command)}  (cwd {API_DIR})")
    try:
        return subprocess.call(command, cwd=str(API_DIR))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
