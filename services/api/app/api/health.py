"""Liveness and readiness.

Split deliberately: liveness answers "is this process alive" and must never
depend on a downstream service, or a Redis blip restarts every API container.
Readiness answers "can this process serve traffic" and does check dependencies.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response

from app.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, Any]:
    s = get_settings()
    return {
        "status": "ok",
        "service": "resx-api",
        "version": "0.1.0",
        "environment": s.resx_env,
    }


@router.get("/health/ready")
async def ready(response: Response) -> dict[str, Any]:
    """Report each dependency separately so a failure names itself.

    Phase 0 reports `not_configured` rather than faking a green check — a
    readiness probe that lies is worse than one that is honest about what has
    not been wired up yet.
    """
    s = get_settings()
    checks: dict[str, str] = {
        "config": "ok",
        "database": "not_configured",
        "redis": "not_configured",
        "sandbox": "not_configured",
    }

    degraded = [name for name, state in checks.items() if state not in {"ok", "not_configured"}]
    if degraded:
        response.status_code = 503

    return {
        "status": "degraded" if degraded else "ok",
        "checks": checks,
        "budgets": {
            "run_max_usd": s.run_max_usd,
            "run_max_debate_rounds": s.run_max_debate_rounds,
        },
    }
