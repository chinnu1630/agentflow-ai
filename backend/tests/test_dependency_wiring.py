"""Smoke tests for production dependency construction."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import cast

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.api.routes.release_runs import (
    get_jira_risk_collector,
    get_risk_collector,
)
from app.core.config import get_settings
from app.services.github_risk_collector import RiskCollector
from app.services.jira_risk_collector import JiraRiskCollector


def _request() -> Request:
    """Build a minimal HTTP request with application request context."""
    request = Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/",
            "raw_path": b"/",
            "query_string": b"",
            "headers": [],
            "client": ("test", 123),
            "server": ("test", 80),
            "root_path": "",
        }
    )
    request.state.request_id = "test-request-id"
    return request


@pytest.mark.anyio
async def test_github_dependency_builds_real_collector(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configured GitHub settings should build the real collector graph."""
    monkeypatch.setenv("GITHUB_REPOSITORY_OWNER", "acme")
    monkeypatch.setenv("GITHUB_REPOSITORY_NAME", "backend")
    get_settings.cache_clear()

    dependency = cast(
        AsyncGenerator[RiskCollector, None],
        get_risk_collector(_request()),
    )
    collector = await anext(dependency)

    assert isinstance(collector, RiskCollector)

    await dependency.aclose()
    get_settings.cache_clear()


@pytest.mark.anyio
async def test_jira_dependency_builds_real_collector(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configured Jira settings should build the real client and collector graph."""
    monkeypatch.setenv("JIRA_BASE_URL", "https://jira.example.com")
    monkeypatch.setenv("JIRA_EMAIL", "manager@example.com")
    monkeypatch.setenv("JIRA_API_TOKEN", "test-token")
    monkeypatch.setenv("JIRA_PROJECT_KEY", "PAY")
    get_settings.cache_clear()

    dependency = cast(
        AsyncGenerator[JiraRiskCollector, None],
        get_jira_risk_collector(),
    )
    collector = await anext(dependency)

    assert isinstance(collector, JiraRiskCollector)

    await dependency.aclose()
    get_settings.cache_clear()


@pytest.mark.anyio
async def test_jira_dependency_rejects_missing_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing Jira settings should return a controlled service-unavailable error."""
    for variable in ("JIRA_BASE_URL", "JIRA_EMAIL", "JIRA_API_TOKEN", "JIRA_PROJECT_KEY"):
        monkeypatch.delenv(variable, raising=False)
    get_settings.cache_clear()

    dependency = get_jira_risk_collector()

    with pytest.raises(HTTPException) as exc_info:
        await anext(dependency)

    assert exc_info.value.status_code == 503
    get_settings.cache_clear()


@pytest.mark.anyio
async def test_anthropic_dependency_returns_none_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabled Claude synthesis should preserve the deterministic workflow."""
    from app.api.routes.release_runs import get_risk_synthesis_service

    monkeypatch.setenv("ANTHROPIC_ENABLED", "false")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()

    dependency = cast(
        AsyncGenerator[object, None],
        get_risk_synthesis_service(_request()),
    )
    synthesis_service = await anext(dependency)

    assert synthesis_service is None

    await dependency.aclose()
    get_settings.cache_clear()


@pytest.mark.anyio
async def test_anthropic_dependency_rejects_missing_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled Claude synthesis must require an API key from the environment."""
    from app.api.routes.release_runs import get_risk_synthesis_service

    monkeypatch.setenv("ANTHROPIC_ENABLED", "true")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()

    dependency = cast(
        AsyncGenerator[object, None],
        get_risk_synthesis_service(_request()),
    )

    with pytest.raises(HTTPException) as exc_info:
        await anext(dependency)

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == (
        "Claude risk synthesis is enabled but not configured. "
        "Set ANTHROPIC_API_KEY."
    )

    await dependency.aclose()
    get_settings.cache_clear()


@pytest.mark.anyio
async def test_anthropic_dependency_builds_synthesis_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured Claude settings should build the real synthesis client."""
    from app.api.routes.release_runs import get_risk_synthesis_service
    from app.integrations.anthropic_client import AnthropicRiskSynthesisClient

    monkeypatch.setenv("ANTHROPIC_ENABLED", "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "test-claude-model")
    get_settings.cache_clear()

    dependency = cast(
        AsyncGenerator[object, None],
        get_risk_synthesis_service(_request()),
    )
    synthesis_service = await anext(dependency)

    assert isinstance(synthesis_service, AnthropicRiskSynthesisClient)

    await dependency.aclose()
    get_settings.cache_clear()
