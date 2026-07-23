"""Route-level authentication contracts for AgentFlow APIs."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies.security import get_current_principal
from app.core.config import get_settings
from app.core.security import AuthenticatedPrincipal
from app.main import app


@pytest.fixture(autouse=True)
def enable_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Require bearer authentication for every route-contract test."""
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv(
        "DATABASE_URL",
        "sqlite+aiosqlite:///:memory:",
    )
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "AUTH_JWT_ISSUER",
        "https://identity.example.com/",
    )
    monkeypatch.setenv("AUTH_JWT_AUDIENCE", "agentflow-api")
    monkeypatch.setenv(
        "AUTH_JWT_PUBLIC_KEY",
        "test-public-key-not-used-without-token",
    )
    get_settings.cache_clear()

    yield

    get_settings.cache_clear()


def test_health_endpoint_remains_public() -> None:
    """Infrastructure health checks must not require user credentials."""
    response = TestClient(app).get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        (
            "POST",
            "/api/v1/release-runs",
            {"query": "What are the biggest release risks this week?"},
        ),
        ("GET", "/api/v1/release-runs/approvals/pending", None),
        ("GET", f"/api/v1/release-runs/{uuid4()}", None),
        ("GET", f"/api/v1/release-runs/{uuid4()}/events", None),
        ("GET", f"/api/v1/release-runs/{uuid4()}/approvals", None),
        (
            "POST",
            (
                f"/api/v1/release-runs/{uuid4()}"
                f"/approvals/{uuid4()}/decision"
            ),
            {"approval_status": "approved"},
        ),
        (
            "POST",
            f"/api/v1/release-runs/{uuid4()}/slack-alert",
            None,
        ),
        ("POST", f"/api/v1/release-runs/{uuid4()}/risks", None),
        (
            "POST",
            f"/api/v1/release-runs/{uuid4()}/github-risks",
            None,
        ),
        (
            "POST",
            "/api/v1/engineering-documents/ingest",
            {
                "title": "Payment Service Runbook",
                "source_type": "runbook",
                "source_uri": "docs/payment-service-runbook.md",
                "raw_content": "Rollback the payment service safely.",
            },
        ),
        (
            "POST",
            "/api/v1/engineering-documents/retrieve",
            {"query": "How do I rollback payment service?"},
        ),
        (
            "POST",
            "/api/v1/agent/query-plan",
            {"query": "What are the biggest release risks this week?"},
        ),
        (
            "POST",
            "/api/v1/agent/query-dynamic",
            {"query": "What is the current workflow status?"},
        ),
        (
            "POST",
            "/api/v1/agent/query",
            {"query": "What are the biggest release risks this week?"},
        ),
    ],
)
def test_business_endpoints_require_authentication(
    method: str,
    path: str,
    payload: dict[str, Any] | None,
) -> None:
    """Every business endpoint must reject missing bearer credentials."""
    response = TestClient(app).request(
        method=method,
        url=path,
        json=payload,
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "AUTHENTICATION_FAILED"


@pytest.mark.parametrize(
    ("method", "path", "payload", "granted_scope"),
    [
        (
            "GET",
            "/api/v1/release-runs/approvals/pending",
            None,
            "release:write",
        ),
        (
            "POST",
            "/api/v1/release-runs",
            {"query": "What are the biggest release risks this week?"},
            "release:read",
        ),
        (
            "POST",
            (
                f"/api/v1/release-runs/{uuid4()}"
                f"/approvals/{uuid4()}/decision"
            ),
            {"approval_status": "approved"},
            "release:read",
        ),
        (
            "POST",
            f"/api/v1/release-runs/{uuid4()}/slack-alert",
            None,
            "release:read",
        ),
        (
            "POST",
            "/api/v1/engineering-documents/ingest",
            {
                "title": "Payment Service Runbook",
                "source_type": "runbook",
                "source_uri": "docs/payment-service-runbook.md",
                "raw_content": "Rollback the payment service safely.",
            },
            "knowledge:read",
        ),
        (
            "POST",
            "/api/v1/engineering-documents/retrieve",
            {"query": "How do I rollback payment service?"},
            "knowledge:write",
        ),
        (
            "POST",
            "/api/v1/agent/query-plan",
            {"query": "What are the biggest release risks this week?"},
            "release:read",
        ),
    ],
)
def test_business_endpoints_reject_insufficient_scope(
    method: str,
    path: str,
    payload: dict[str, Any] | None,
    granted_scope: str,
) -> None:
    """Authenticated callers must still hold the route's required scope."""

    async def override_get_current_principal() -> AuthenticatedPrincipal:
        """Return an authenticated principal with an unrelated scope."""
        return AuthenticatedPrincipal(
            subject="limited-user-123",
            email="limited.user@example.com",
            roles=frozenset({"viewer"}),
            scopes=frozenset({granted_scope}),
        )

    app.dependency_overrides[
        get_current_principal
    ] = override_get_current_principal

    try:
        response = TestClient(app).request(
            method=method,
            url=path,
            json=payload,
        )
    finally:
        app.dependency_overrides.pop(get_current_principal, None)

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "AUTHORIZATION_FAILED"


def test_agent_slack_action_requires_release_notify_scope() -> None:
    """Query permission alone must not authorize a Slack side effect."""

    async def override_get_current_principal() -> AuthenticatedPrincipal:
        """Return a caller allowed to query but not send notifications."""
        return AuthenticatedPrincipal(
            subject="analyst-123",
            email="analyst@example.com",
            roles=frozenset({"release_analyst"}),
            scopes=frozenset({"agent:query"}),
        )

    app.dependency_overrides[
        get_current_principal
    ] = override_get_current_principal

    try:
        response = TestClient(app).post(
            "/api/v1/agent/query",
            json={
                "query": "Can you send this to Slack?",
                "release_run_id": str(uuid4()),
            },
        )
    finally:
        app.dependency_overrides.pop(get_current_principal, None)

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "AUTHORIZATION_FAILED"
