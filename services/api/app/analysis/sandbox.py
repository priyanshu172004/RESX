"""The code interpreter.

An LLM writes Python; this module runs it and records exactly what happened.
The record is what makes a figure reproducible: given a `ComputationRecord` and
the raw upload, any number in a RESX report can be re-derived byte for byte.

## A blunt statement about isolation

There are two backends, and they are **not** equivalent:

  * `DockerSandbox` — the production backend. No network, non-root, dropped
    capabilities, read-only root filesystem, memory and CPU ceilings, one-shot
    container. This is a real security boundary.

  * `LocalSubprocessSandbox` — **development only, and NOT a security
    boundary.** It blocks sockets and applies resource limits where the OS
    supports it, but a determined payload can escape a same-user subprocess. On
    Windows there is no `resource` module at all, so only the wall-clock
    timeout applies. It exists so the pipeline is runnable on a laptop without
    Docker, and `is_security_boundary` is `False` so callers cannot mistake it
    for the real thing.

Selecting the local backend in production is refused rather than warned about.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ComputationRecord:
    """The audit trail for one computation. See docs/04 §3."""

    computation_id: str
    code: str
    inputs: list[str]
    stdout: str
    stderr: str
    result: Any
    duration_ms: int
    image_digest: str
    ok: bool
    error: str | None = None
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "computation_id": self.computation_id,
            "code": self.code,
            "inputs": self.inputs,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "result": self.result,
            "duration_ms": self.duration_ms,
            "image_digest": self.image_digest,
            "ok": self.ok,
            "error": self.error,
            "artifacts": self.artifacts,
        }

    @property
    def code_preview(self) -> str:
        lines = self.code.strip().splitlines()
        head = lines[:3]
        suffix = "" if len(lines) <= 3 else f"\n… (+{len(lines) - 3} lines)"
        return "\n".join(head) + suffix


class SandboxError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# The in-sandbox bootstrap
# --------------------------------------------------------------------------- #

# Runs *inside* the sandboxed process, before any model-written code. It
# installs the guards, exposes the `resx` data-access helper, executes the
# payload, and emits a single JSON line on a sentinel so stdout written by the
# payload cannot be confused with the result envelope.
_BOOTSTRAP = textwrap.dedent(
    """
    import builtins, io, json, os, sys, types
    from contextlib import redirect_stdout, redirect_stderr
    from decimal import Decimal

    SENTINEL = "__RESX_RESULT__"
    CODE_PATH = sys.argv[1]
    DATA_DIR = sys.argv[2]
    OUT_DIR = sys.argv[3]
    CPU_SECONDS = int(sys.argv[4])
    MEMORY_MB = int(sys.argv[5])

    # One thread per BLAS backend, set before numpy is imported.
    #
    # Two reasons, and the first is why CI failed. RLIMIT_AS below caps
    # *address space*, not resident memory, and every OpenBLAS/OpenMP worker
    # reserves its own arena on import -- so a machine with many cores
    # reserves hundreds of megabytes before a single array exists, and the
    # process is killed importing pandas. The developer machines where this
    # passed were Windows, where `resource` does not exist and no limit was
    # ever applied.
    #
    # The second reason stands on its own: a computation that has to be
    # reproducible should not depend on how many cores the box has.
    for _var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[_var] = "1"

    # Import paths are handed over explicitly by the parent rather than being
    # rediscovered from the environment. Relying on env vars is fragile: the
    # minimal env below has no APPDATA, and on Windows that alone makes the
    # user site directory unresolvable, which silently breaks pandas via its
    # dateutil dependency. Passing the parent's resolved sys.path makes the
    # child's imports deterministic on every platform.
    if len(sys.argv) > 6 and sys.argv[6]:
        sys.path[:0] = [p for p in json.loads(sys.argv[6]) if p not in sys.path]

    # --- the application package is off limits, by name -----------------
    # Filtering sys.path is not sufficient on its own. An editable install
    # (`pip install -e .`) drops a .pth into site-packages that registers a
    # meta-path finder for `app`, and site-packages is processed even under
    # `-I`. So the package stays importable no matter what sys.path says, and
    # sandboxed code could reach RESX internals and step around these guards.
    # Blocking the name is the control that actually holds: it does not care
    # how the package got onto the path.
    _BLOCKED_ROOTS = ("app", "resx_api", "__editable__")

    class _DenyApplicationImports:
        def find_module(self, name, path=None):
            self.find_spec(name, path)
            return None

        def find_spec(self, name, path=None, target=None):
            root = name.split(".", 1)[0]
            if root in _BLOCKED_ROOTS:
                raise ImportError("No module named '" + root + "'", name=name)
            return None

    sys.meta_path.insert(0, _DenyApplicationImports())
    for _mod in [m for m in sys.modules if m.split(".", 1)[0] in _BLOCKED_ROOTS]:
        del sys.modules[_mod]

    # --- resource ceilings (POSIX only) ---------------------------------
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
        mem = MEMORY_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        # Windows has no `resource`. The wall-clock timeout in the parent is
        # then the only limit, which is exactly why this backend is dev-only.
        pass

    # --- no network -----------------------------------------------------
    import socket
    def _blocked(*a, **k):
        raise OSError("network access is disabled inside the RESX sandbox")
    socket.socket = _blocked
    socket.create_connection = _blocked
    socket.getaddrinfo = _blocked

    # --- no subprocesses ------------------------------------------------
    os.system = _blocked
    os.popen = _blocked
    for _name in ("fork", "forkpty", "spawnl", "spawnv", "execv", "execve"):
        if hasattr(os, _name):
            setattr(os, _name, _blocked)
    try:
        import subprocess as _sp
        _sp.Popen = _blocked
        _sp.run = _blocked
        _sp.call = _blocked
        _sp.check_output = _blocked
    except Exception:
        pass

    # --- data access ----------------------------------------------------
    # Datasets are addressed by opaque id, never by a caller-supplied path, so
    # there is no traversal input to sanitise. pandas is imported lazily: a
    # computation that does no data loading must not fail because the
    # dataframe stack is missing.
    def _load(dataset_id):
        import pandas as pd
        base = os.path.join(DATA_DIR, str(dataset_id))
        for ext, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
            path = base + ext
            if os.path.exists(path):
                return reader(path)
        raise FileNotFoundError("no such dataset: " + str(dataset_id))

    def _list():
        if not os.path.isdir(DATA_DIR):
            return []
        return sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(DATA_DIR)
            if f.endswith((".parquet", ".csv"))
        )

    def _artifact(name):
        return os.path.join(OUT_DIR, os.path.basename(str(name)))

    resx = types.ModuleType("resx")
    resx.load = _load
    resx.list_datasets = _list
    resx.artifact = _artifact
    resx.Decimal = Decimal
    sys.modules["resx"] = resx

    with open(CODE_PATH, "r", encoding="utf-8") as fh:
        source = fh.read()

    env = {
        "__name__": "__resx_sandbox__",
        "__builtins__": builtins,
        "resx": resx,
        "Decimal": Decimal,
        "result": None,
    }

    out, err = io.StringIO(), io.StringIO()
    payload = {"ok": False, "result": None, "error": None}
    try:
        with redirect_stdout(out), redirect_stderr(err):
            exec(compile(source, "<resx-computation>", "exec"), env)
        payload["ok"] = True
        payload["result"] = env.get("result")
    except BaseException as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"

    def _coerce(value):
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            return value if value == value and abs(value) != float("inf") else str(value)
        if isinstance(value, Decimal):
            return {"__decimal__": str(value)}
        if isinstance(value, dict):
            return {str(k): _coerce(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_coerce(v) for v in value]
        for attr in ("to_dict", "tolist", "item"):
            if hasattr(value, attr):
                try:
                    return _coerce(getattr(value, attr)())
                except Exception:
                    pass
        return repr(value)

    payload["result"] = _coerce(payload["result"])
    payload["stdout"] = out.getvalue()
    payload["stderr"] = err.getvalue()
    payload["artifacts"] = sorted(os.listdir(OUT_DIR)) if os.path.isdir(OUT_DIR) else []

    sys.__stdout__.write("\\n" + SENTINEL + json.dumps(payload) + "\\n")
    sys.__stdout__.flush()
    """
).strip()

_SENTINEL = "__RESX_RESULT__"


def _revive(value: Any) -> Any:
    """Turn the JSON envelope back into Python, restoring Decimals."""
    if isinstance(value, dict):
        if set(value) == {"__decimal__"}:
            from decimal import Decimal

            return Decimal(value["__decimal__"])
        return {k: _revive(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_revive(v) for v in value]
    return value


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


class Sandbox(ABC):
    is_security_boundary: bool = False
    image_digest: str = "unknown"

    @abstractmethod
    def run(
        self,
        code: str,
        *,
        inputs: list[str] | None = None,
        dataset_dir: Path | None = None,
        timeout_seconds: int | None = None,
    ) -> ComputationRecord: ...


def _describe_exit(returncode: int) -> str:
    """Turn a process exit into something a reader can act on.

    A negative return code is a signal, and "SandboxError: the process
    produced no result envelope (exit -9)" is true and useless — it does not
    say the CPU ceiling fired, which is the single most likely reason a
    computation of this kind dies.

    It also hid a platform difference for a long time. `RLIMIT_CPU` is POSIX
    only, so on Windows an infinite loop runs until the wall clock stops it
    and the caller sees a timeout; on Linux the CPU limit fires first and the
    caller saw an unexplained signal. Same cause, two different stories.
    """
    if returncode >= 0:
        return f"SandboxError: the process produced no result envelope (exit {returncode})"

    import signal as _signal

    fired = -returncode
    # Built from the signals this platform actually has: SIGKILL and SIGXCPU do
    # not exist on Windows, and a dict keyed on a missing one would collapse
    # every entry onto `None`.
    named = {
        getattr(_signal, "SIGXCPU", None): (
            "TimeoutError: exceeded the CPU-time ceiling for this sandbox"
        ),
        getattr(_signal, "SIGKILL", None): (
            "TimeoutError: the sandbox was killed, which for this backend means "
            "it exceeded its CPU-time or memory ceiling"
        ),
        getattr(_signal, "SIGSEGV", None): (
            "SandboxError: the computation crashed the interpreter (segmentation fault)"
        ),
    }
    described = {k: v for k, v in named.items() if k is not None}.get(fired)
    if described:
        return described
    try:
        name = _signal.Signals(fired).name
    except (ValueError, AttributeError):
        name = str(fired)
    return f"SandboxError: the sandbox was terminated by signal {name}"


class LocalSubprocessSandbox(Sandbox):
    """Development backend. **Not** a security boundary — see the module docstring."""

    is_security_boundary = False

    def __init__(
        self,
        *,
        cpu_seconds: int = 2,
        # Address space, not resident memory -- this becomes `RLIMIT_AS`, which
        # counts reservations. numpy and pandas reserve far more than they ever
        # touch, so 512 killed the interpreter mid-import. The Docker backend
        # below keeps 512 because `--memory` limits RSS, which is the number
        # that reads as "how much memory may this use".
        memory_mb: int = 2048,
        wall_timeout_seconds: int = 30,
    ) -> None:
        self.cpu_seconds = cpu_seconds
        self.memory_mb = memory_mb
        self.wall_timeout_seconds = wall_timeout_seconds
        # Content-address the interpreter + bootstrap so a record identifies
        # the exact execution environment, as an image digest would.
        stamp = f"{sys.version}|{_BOOTSTRAP}".encode()
        self.image_digest = "local-subprocess:sha256:" + hashlib.sha256(stamp).hexdigest()[:32]

    @staticmethod
    def _import_paths() -> str:
        """The parent's real import paths, as JSON, for the child to adopt.

        Any directory that *contains* the RESX application package is dropped,
        not merely the current working directory: when the service is started
        from elsewhere, or installed editable, the package's own directory is
        on `sys.path` under some other name and a cwd check would sail past it.

        This is defence in depth rather than the control. The control is the
        name-based denial in the bootstrap, because an editable install can
        make `app` importable with no matching `sys.path` entry at all.
        """
        cwd = os.path.abspath(os.getcwd())
        paths: list[str] = []
        for entry in sys.path:
            if not entry or not os.path.isdir(entry):
                continue
            resolved = os.path.abspath(entry)
            if resolved == cwd:
                continue
            if os.path.isfile(os.path.join(resolved, "app", "__init__.py")):
                continue
            paths.append(entry)
        return json.dumps(paths)

    def run(
        self,
        code: str,
        *,
        inputs: list[str] | None = None,
        dataset_dir: Path | None = None,
        timeout_seconds: int | None = None,
    ) -> ComputationRecord:
        computation_id = f"cmp_{uuid.uuid4().hex[:12]}"
        timeout = timeout_seconds or self.wall_timeout_seconds
        started = time.perf_counter()

        with tempfile.TemporaryDirectory(prefix="resx-sbx-") as tmp:
            tmpdir = Path(tmp)
            code_path = tmpdir / "computation.py"
            boot_path = tmpdir / "_bootstrap.py"
            out_dir = tmpdir / "out"
            out_dir.mkdir()

            code_path.write_text(code, encoding="utf-8")
            boot_path.write_text(_BOOTSTRAP, encoding="utf-8")

            # Resolved to absolute: the child runs with the temp directory as
            # its cwd, so a relative dataset path would silently resolve to
            # nothing and every load would fail with "no such dataset".
            data_dir = str(Path(dataset_dir).resolve()) if dataset_dir else str(tmpdir / "data")
            Path(data_dir).mkdir(exist_ok=True, parents=True)

            # A minimal environment: no inherited API keys, no proxy settings.
            env = {
                "PATH": os.environ.get("PATH", ""),
                "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
                "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
                "MPLBACKEND": "Agg",
                "HOME": str(tmpdir),
                "TMPDIR": str(tmpdir),
            }

            try:
                proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
                    [
                        sys.executable,
                        # -I is safe now that import paths are passed
                        # explicitly: it ignores env vars and the user site
                        # dir, and the bootstrap re-adds exactly the paths we
                        # chose rather than whatever the environment implies.
                        "-I",
                        str(boot_path),
                        str(code_path),
                        data_dir,
                        str(out_dir),
                        str(self.cpu_seconds),
                        str(self.memory_mb),
                        self._import_paths(),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=str(tmpdir),
                    env=env,
                    shell=False,
                )
            except subprocess.TimeoutExpired:
                duration = int((time.perf_counter() - started) * 1000)
                return ComputationRecord(
                    computation_id=computation_id,
                    code=code,
                    inputs=inputs or [],
                    stdout="",
                    stderr="",
                    result=None,
                    duration_ms=duration,
                    image_digest=self.image_digest,
                    ok=False,
                    error=f"TimeoutError: exceeded {timeout}s wall clock",
                )

            duration = int((time.perf_counter() - started) * 1000)
            raw = proc.stdout or ""

            marker = raw.rfind(_SENTINEL)
            if marker == -1:
                return ComputationRecord(
                    computation_id=computation_id,
                    code=code,
                    inputs=inputs or [],
                    stdout=raw,
                    stderr=proc.stderr or "",
                    result=None,
                    duration_ms=duration,
                    image_digest=self.image_digest,
                    ok=False,
                    error=_describe_exit(proc.returncode),
                )

            envelope = json.loads(raw[marker + len(_SENTINEL) :].splitlines()[0])
            artifacts = [str(out_dir / name) for name in envelope.get("artifacts", [])]
            # Artifacts live in the temp dir, so copy anything produced to a
            # durable location before it is cleaned up.
            kept: list[str] = []
            if artifacts:
                keep_dir = Path(tempfile.mkdtemp(prefix="resx-artifacts-"))
                for path in artifacts:
                    src = Path(path)
                    if src.exists():
                        dest = keep_dir / src.name
                        dest.write_bytes(src.read_bytes())
                        kept.append(str(dest))

            return ComputationRecord(
                computation_id=computation_id,
                code=code,
                inputs=inputs or [],
                stdout=envelope.get("stdout", ""),
                stderr=envelope.get("stderr", "") or (proc.stderr or ""),
                result=_revive(envelope.get("result")),
                duration_ms=duration,
                image_digest=self.image_digest,
                ok=bool(envelope.get("ok")),
                error=envelope.get("error"),
                artifacts=kept,
            )


class DockerSandbox(Sandbox):
    """Production backend. A real isolation boundary.

    Every flag below is load-bearing:
      --network none            no egress, so exfiltration has no channel
      --user 65534:65534        non-root
      --cap-drop ALL            no capabilities
      --security-opt no-new-privileges
      --read-only               immutable root filesystem
      --tmpfs /tmp:size=64m     the only writable location, size-capped
      --memory / --cpus         resource ceilings
      --pids-limit              no fork bombs
      --rm                      one-shot; the container never serves two runs
    """

    is_security_boundary = True

    def __init__(
        self,
        *,
        image: str,
        cpu_seconds: int = 2,
        memory_mb: int = 512,
        wall_timeout_seconds: int = 30,
        docker_bin: str = "docker",
    ) -> None:
        self.image = image
        self.cpu_seconds = cpu_seconds
        self.memory_mb = memory_mb
        self.wall_timeout_seconds = wall_timeout_seconds
        self.docker_bin = docker_bin
        self.image_digest = self._resolve_digest()

    @staticmethod
    def available(docker_bin: str = "docker") -> bool:
        """Is a working Docker daemon actually reachable?

        Constructing `DockerSandbox` is not a probe: the digest lookup swallows
        failures so that an unresolvable digest is recorded honestly rather
        than crashing. That meant a machine with no Docker still got a Docker
        backend, which then failed on the first computation. So availability is
        checked explicitly, before the backend is chosen.
        """
        try:
            proc = subprocess.run(  # noqa: S603
                [docker_bin, "version", "--format", "{{.Server.Version}}"],
                capture_output=True,
                text=True,
                timeout=15,
                shell=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return proc.returncode == 0 and bool(proc.stdout.strip())

    def _resolve_digest(self) -> str:
        try:
            proc = subprocess.run(  # noqa: S603
                [self.docker_bin, "image", "inspect", self.image, "--format", "{{.Id}}"],
                capture_output=True,
                text=True,
                timeout=20,
                shell=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
        # An unresolvable digest is recorded honestly; it means the record
        # cannot claim exact reproducibility.
        return f"{self.image}:digest-unavailable"

    def run(
        self,
        code: str,
        *,
        inputs: list[str] | None = None,
        dataset_dir: Path | None = None,
        timeout_seconds: int | None = None,
    ) -> ComputationRecord:
        computation_id = f"cmp_{uuid.uuid4().hex[:12]}"
        timeout = timeout_seconds or self.wall_timeout_seconds
        started = time.perf_counter()

        with tempfile.TemporaryDirectory(prefix="resx-sbx-") as tmp:
            tmpdir = Path(tmp)
            (tmpdir / "computation.py").write_text(code, encoding="utf-8")
            (tmpdir / "_bootstrap.py").write_text(_BOOTSTRAP, encoding="utf-8")
            out_dir = tmpdir / "out"
            out_dir.mkdir()

            mounts = [
                "-v",
                f"{tmpdir}:/work:ro",
                "-v",
                f"{out_dir}:/out:rw",
            ]
            if dataset_dir:
                mounts += ["-v", f"{dataset_dir}:/data:ro"]

            argv = [
                self.docker_bin,
                "run",
                "--rm",
                "--network",
                "none",
                "--user",
                "65534:65534",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--read-only",
                "--tmpfs",
                # Paths inside the container, not on the host. A 64 MB
                # noexec tmpfs over /tmp on an otherwise read-only
                # filesystem is the hardening, not a weakness.
                "/tmp:size=64m,noexec",  # noqa: S108
                "--memory",
                f"{self.memory_mb}m",
                "--memory-swap",
                f"{self.memory_mb}m",
                "--cpus",
                "1",
                "--pids-limit",
                "64",
                "--workdir",
                "/tmp",  # noqa: S108 - inside the container
                *mounts,
                self.image,
                "python",
                "-I",
                "/work/_bootstrap.py",
                "/work/computation.py",
                "/data" if dataset_dir else "/tmp/data",  # noqa: S108
                "/out",
                str(self.cpu_seconds),
                str(self.memory_mb),
                "",  # the image owns its import paths; nothing to inject
            ]

            try:
                proc = subprocess.run(  # noqa: S603
                    argv, capture_output=True, text=True, timeout=timeout, shell=False
                )
            except subprocess.TimeoutExpired:
                return ComputationRecord(
                    computation_id=computation_id,
                    code=code,
                    inputs=inputs or [],
                    stdout="",
                    stderr="",
                    result=None,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    image_digest=self.image_digest,
                    ok=False,
                    error=f"TimeoutError: exceeded {timeout}s wall clock",
                )
            except OSError as exc:
                raise SandboxError(f"docker is not available: {exc}") from exc

            duration = int((time.perf_counter() - started) * 1000)
            raw = proc.stdout or ""
            marker = raw.rfind(_SENTINEL)
            if marker == -1:
                return ComputationRecord(
                    computation_id=computation_id,
                    code=code,
                    inputs=inputs or [],
                    stdout=raw,
                    stderr=proc.stderr or "",
                    result=None,
                    duration_ms=duration,
                    image_digest=self.image_digest,
                    ok=False,
                    error=f"SandboxError: no result envelope (exit {proc.returncode})",
                )

            envelope = json.loads(raw[marker + len(_SENTINEL) :].splitlines()[0])
            kept: list[str] = []
            for name in envelope.get("artifacts", []):
                src = out_dir / name
                if src.exists():
                    keep = Path(tempfile.mkdtemp(prefix="resx-artifacts-")) / name
                    keep.write_bytes(src.read_bytes())
                    kept.append(str(keep))

            return ComputationRecord(
                computation_id=computation_id,
                code=code,
                inputs=inputs or [],
                stdout=envelope.get("stdout", ""),
                stderr=envelope.get("stderr", ""),
                result=_revive(envelope.get("result")),
                duration_ms=duration,
                image_digest=self.image_digest,
                ok=bool(envelope.get("ok")),
                error=envelope.get("error"),
                artifacts=kept,
            )


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def build_sandbox(settings: Any) -> Sandbox:
    """Choose a backend.

    Production refuses the local backend outright. A dev-only sandbox running
    model-written code in production is not a misconfiguration to warn about —
    it is a vulnerability, so it fails on boot.
    """
    is_production = getattr(settings, "is_production", False)
    cpu = getattr(settings, "sandbox_cpu_seconds", 2)
    mem = getattr(settings, "sandbox_memory_mb", 512)
    wall = getattr(settings, "sandbox_wall_timeout_seconds", 30)
    image = getattr(settings, "sandbox_image", "resx/sandbox:0.1.0")

    if is_production:
        if not DockerSandbox.available():
            # Refused, not warned about. Running model-written code without an
            # isolation boundary in production is a vulnerability, so it fails
            # on boot rather than at the first computation.
            raise SandboxError(
                "production requires the Docker sandbox, but no Docker daemon is "
                "reachable. Refusing to fall back to the development backend, "
                "which is not a security boundary."
            )
        return DockerSandbox(
            image=image, cpu_seconds=cpu, memory_mb=mem, wall_timeout_seconds=wall
        )

    if DockerSandbox.available():
        return DockerSandbox(
            image=image, cpu_seconds=cpu, memory_mb=mem, wall_timeout_seconds=wall
        )

    return LocalSubprocessSandbox(cpu_seconds=cpu, memory_mb=mem, wall_timeout_seconds=wall)
