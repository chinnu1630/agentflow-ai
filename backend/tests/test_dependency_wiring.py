"""Smoke tests for production dependency construction."""

from __future__ import annotations

from collections.abc import AsyncIterator

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

    dependency: AsyncIterator[RiskCollector] = get_risk_collector(_request())
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

    dependency: AsyncIterator[JiraRiskCollector] = get_jira_risk_collector()
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
