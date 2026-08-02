"""Tests for AgentFlow query collector dependency selection."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import TypeVar
from uuid import uuid4

import pytest
from pydantic import SecretStr
from starlette.requests import Request

from app.api.routes import agent_queries
from app.schemas.agent_query import (
    AgentIntent,
    AgentQueryPlan,
    AgentQueryRequest,
    ResponseDepth,
)

T = TypeVar("T")


async def consume_dependency(
    dependency: AsyncIterator[T],
) -> T:
    """Read one yielded dependency value and close its async generator."""

    try:
        return await anext(dependency)
    finally:
        await dependency.aclose()


def build_plan(
    *,
    intent: AgentIntent,
    release_run_id: object | None = None,
) -> AgentQueryPlan:
    """Build a validated query plan for collector dependency tests."""

    return AgentQueryPlan(
        intent=intent,
        response_depth=ResponseDepth.STANDARD,
        confidence=1.0,
        release_run_id=release_run_id,
        requires_current_snapshot=True,
        routing_reason_code="test_collector_dependency_policy",
    )


@pytest.fixture
def configured_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provide deterministic GitHub and Jira configuration."""

    settings = SimpleNamespace(
        github_repository_owner="agentflow-test",
        github_repository_name="release-service",
        github_default_branch="main",
        github_token=SecretStr("test-github-token"),
        jira_base_url="https://jira.example.test",
        jira_email="manager@example.test",
        jira_api_token=SecretStr("test-jira-token"),
        jira_project_key="REL",
    )

    monkeypatch.setattr(
        agent_queries,
        "get_settings",
        lambda: settings,
    )


@pytest.mark.anyio
async def test_fresh_analytical_query_creates_collectors(
    configured_settings: None,
) -> None:
    """Fresh analytical intents must receive real collector dependencies."""

    payload = AgentQueryRequest(query="Why is the risk score high?")
    plan = build_plan(intent=AgentIntent.EXPLAIN_RISK_SCORE)
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/agent/query",
            "headers": [],
        }
    )
    request.state.request_id = "collector-dependency-test"

    github_collector = await consume_dependency(
        agent_queries.get_agent_github_risk_collector(
            request=request,
            payload=payload,
            plan=plan,
        )
    )
    jira_collector = await consume_dependency(
        agent_queries.get_agent_jira_risk_collector(
            payload=payload,
            plan=plan,
        )
    )

    assert github_collector is not None
    assert jira_collector is not None


@pytest.mark.anyio
async def test_persisted_analytical_query_skips_collectors(
    configured_settings: None,
) -> None:
    """Explicit persisted follow-ups must not construct fresh collectors."""

    release_run_id = uuid4()
    payload = AgentQueryRequest(
        query="Why is the risk score high?",
        release_run_id=release_run_id,
    )
    plan = build_plan(
        intent=AgentIntent.EXPLAIN_RISK_SCORE,
        release_run_id=release_run_id,
    )
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/agent/query",
            "headers": [],
        }
    )

    github_collector = await consume_dependency(
        agent_queries.get_agent_github_risk_collector(
            request=request,
            payload=payload,
            plan=plan,
        )
    )
    jira_collector = await consume_dependency(
        agent_queries.get_agent_jira_risk_collector(
            payload=payload,
            plan=plan,
        )
    )

    assert github_collector is None
    assert jira_collector is None
