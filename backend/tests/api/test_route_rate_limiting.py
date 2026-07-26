"""Route-level distributed rate-limit contracts for AgentFlow APIs."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies.security import get_current_principal
from app.core.config import get_settings
from app.core.security import AuthenticatedPrincipal
from app.main import app


@pytest.fixture(autouse=True)
def enable_rate_limiting(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Enable deterministic Redis rate limiting for every route test."""
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv(
        "RATE_LIMIT_KEY_HMAC_SECRET",
        "test-route-rate-limit-secret",
    )
    get_settings.cache_clear()

    async def override_get_current_principal() -> AuthenticatedPrincipal:
        """Return a trusted principal for protected route tests."""
        return AuthenticatedPrincipal(
            subject="route-test-user",
            roles=frozenset({"release_manager"}),
            scopes=frozenset({"*"}),
        )

    app.dependency_overrides[
        get_current_principal
    ] = override_get_current_principal

    yield

    app.dependency_overrides.clear()
    app.state.redis_client = None
    get_settings.cache_clear()


def _denying_redis_client(
    *,
    retry_after_ms: int = 5_000,
) -> Mock:
    """Return a Redis-compatible test double that denies one request."""
    client = Mock()
    client.eval = AsyncMock(
        return_value=[0, "0.0", retry_after_ms]
    )
    return client


@pytest.mark.parametrize(
    ("path", "payload", "expected_capacity", "expected_refill_per_ms"),
    [
        (
            "/api/v1/agent/query-plan",
            {"query": "What are the biggest release risks this week?"},
            60,
            0.001,
        ),
        (
            "/api/v1/agent/query-dynamic",
            {"query": "What is the current workflow status?"},
            5,
            0.0001,
        ),
        (
            "/api/v1/agent/query",
            {"query": "What are the biggest release risks this week?"},
            5,
            0.0001,
        ),
    ],
)
def test_agent_routes_enforce_expected_rate_limit_policy(
    path: str,
    payload: dict[str, Any],
    expected_capacity: int,
    expected_refill_per_ms: float,
) -> None:
    """Agent routes should reject before planning, persistence, or execution."""
    redis_client = _denying_redis_client()
    app.state.redis_client = redis_client

    response = TestClient(
        app,
        raise_server_exceptions=False,
    ).post(path, json=payload)

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "5"
    assert response.json()["error"]["code"] == "RATE_LIMIT_EXCEEDED"

    arguments = redis_client.eval.await_args.args

    assert arguments[3] == expected_capacity
    assert arguments[4] == expected_refill_per_ms


def test_health_endpoint_does_not_depend_on_redis_rate_limiting() -> None:
    """Container health must remain available during a Redis outage."""
    app.state.redis_client = None

    response = TestClient(
        app,
        raise_server_exceptions=False,
    ).get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"

@pytest.mark.parametrize(
    ("path", "payload", "expected_capacity", "expected_refill_per_ms"),
    [
        (
            "/api/v1/release-runs",
            {"query": "Assess this week's release risks."},
            60,
            0.001,
        ),
        (
            "/api/v1/engineering-documents/retrieve",
            {"query": "How do I roll back the payment service?"},
            60,
            0.001,
        ),
    ],
)
def test_standard_business_routes_use_standard_policy(
    path: str,
    payload: dict[str, Any],
    expected_capacity: int,
    expected_refill_per_ms: float,
) -> None:
    """Standard API work should use the higher-capacity shared policy."""
    redis_client = _denying_redis_client()
    app.state.redis_client = redis_client

    response = TestClient(
        app,
        raise_server_exceptions=False,
    ).post(path, json=payload)

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "RATE_LIMIT_EXCEEDED"

    arguments = redis_client.eval.await_args.args

    assert arguments[3] == expected_capacity
    assert arguments[4] == expected_refill_per_ms


@pytest.mark.parametrize(
    "path",
    [
        f"/api/v1/release-runs/{uuid4()}/risks",
        f"/api/v1/release-runs/{uuid4()}/github-risks",
    ],
)
def test_release_risk_routes_use_expensive_policy(path: str) -> None:
    """Release collection should be limited before external collectors run."""
    redis_client = _denying_redis_client()
    app.state.redis_client = redis_client

    response = TestClient(
        app,
        raise_server_exceptions=False,
    ).post(path)

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "RATE_LIMIT_EXCEEDED"

    arguments = redis_client.eval.await_args.args

    assert arguments[3] == 5
    assert arguments[4] == 0.0001


def test_expensive_route_fails_closed_when_redis_is_unavailable() -> None:
    """An expensive workflow must not run without distributed enforcement."""
    app.state.redis_client = None

    response = TestClient(
        app,
        raise_server_exceptions=False,
    ).post(
        "/api/v1/agent/query",
        json={"query": "What are the biggest release risks this week?"},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == (
        "RATE_LIMIT_SERVICE_UNAVAILABLE"
    )
