"""Shared test setup.

The gold corpus is **generated**, never committed. It contains a synthetic
annual report, a monthly CSV and an adversarial PDF with deliberately planted
traps, and nine tests plus the whole scoring harness measure against its known
answers.

It is generated here rather than left to a manual step because the alternative
is what the suite did before: skip those nine tests when the files are absent.
A green run that quietly skipped every accuracy gate is worse than a red one —
it reports success for checks that never executed.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.core.config import Settings, get_settings
from app.store.local import LocalStore

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLD_DIR = REPO_ROOT / "benchmarks" / "gold"
GENERATOR = GOLD_DIR / "generate.py"

#: One file from each corpus the generator produces. Their presence is what
#: "already generated" means.
EXPECTED = (
    GOLD_DIR / "synthetic-pnl" / "labels.json",
    GOLD_DIR / "adversarial" / "labels.json",
)


@pytest.fixture(scope="session", autouse=True)
def hermetic_settings() -> Iterator[None]:
    """Disable `.env` loading for the whole session.

    Passing `_env_file=None` to a `Settings()` in a fixture only covers that
    one instance. Application code calls `get_settings()` on its own — the
    RS256 key loader and the Argon2 hasher both do — and those reached the
    developer's real `.env`, which meant the suite was partly a report on local
    configuration. A valid `.env` made tests pass that had nothing to do with
    the code, and an invalid one failed tests that were fine.
    """
    from app.core.config import Settings, get_settings

    original = dict(Settings.model_config)
    Settings.model_config["env_file"] = None
    get_settings.cache_clear()
    try:
        yield
    finally:
        Settings.model_config.clear()
        Settings.model_config.update(original)
        get_settings.cache_clear()


@pytest.fixture(scope="session", autouse=True)
def gold_corpus() -> None:
    """Generate the gold corpus once per session if it is missing."""
    if all(path.exists() for path in EXPECTED):
        return

    if not GENERATOR.exists():
        pytest.skip(f"gold generator not found at {GENERATOR}", allow_module_level=True)

    result = subprocess.run(
        [sys.executable, str(GENERATOR)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        # Fail loudly. A silent skip here would disable the accuracy gates and
        # still report a passing suite.
        raise RuntimeError(
            "could not generate the gold corpus, so the accuracy gates cannot "
            f"run:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )


# --------------------------------------------------------------------------- #
# The API client
#
# Shared rather than living in one test module. Three separate test files have
# now been written against a `client` fixture that was private to
# `test_api.py`, each failing at collection with "fixture 'client' not found" —
# so the fixture belongs here.
# --------------------------------------------------------------------------- #


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A client with an isolated store, so tests never share state."""
    settings = Settings(
        _env_file=None,
        resx_env="development",
        storage_local_root=str(tmp_path / "storage"),
        # Agent runs unavailable, on purpose. Both provider keys are cleared
        # explicitly: leaving either to the environment would make the guard
        # tests pass or fail depending on the developer's own .env.
        llm_provider="groq",
        groq_api_key="",
        anthropic_api_key="",
    )

    store = LocalStore(tmp_path / "storage" / "resx.db")

    get_settings.cache_clear()
    deps.get_store.cache_clear()
    deps.get_embedder.cache_clear()
    deps.get_sandbox.cache_clear()
    deps.get_run_service.cache_clear()

    monkeypatch.setattr("app.core.config.get_settings", lambda: settings)
    monkeypatch.setattr("app.api.deps.get_settings", lambda: settings)
    monkeypatch.setattr("app.api.deps.get_store", lambda: store)

    from app.main import create_app

    app = create_app()
    app.dependency_overrides[deps.get_store] = lambda: store
    app.dependency_overrides[get_settings] = lambda: settings

    with TestClient(app) as test_client:
        # Register, then attach the access token to every subsequent request.
        # The suite therefore runs through the real session path; a bypass here
        # would leave the guard itself untested.
        registered = test_client.post(
            "/api/v1/auth/register",
            json={
                "email": "analyst@example.com",
                "password": "a-long-enough-password",
                "name": "Test Analyst",
            },
        )
        assert registered.status_code == 201, registered.text
        token = registered.json()["access_token"]
        test_client.headers.update({"Authorization": f"Bearer {token}"})
        yield test_client

    # Caches are cleared on setup, not teardown: monkeypatch has replaced these
    # names with plain lambdas, which have no cache_clear, and its own undo
    # runs after this block.
    store.close()


@pytest.fixture
def anon(client: TestClient) -> Iterator[TestClient]:
    """The same app with no credentials attached.

    Derived from `client` so the app and store are identical -- the only
    difference is the missing Authorization header, which is exactly the
    variable under test.
    """
    saved = client.headers.pop("Authorization", None)
    yield client
    if saved:
        client.headers["Authorization"] = saved
