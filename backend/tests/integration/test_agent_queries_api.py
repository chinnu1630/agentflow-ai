"""Integration tests for the AgentFlow natural-language query API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from app.api.routes.agent_queries import (
    get_agent_github_risk_collector,
    get_agent_jira_risk_collector,
)
from app.db.base import Base
from app.db.session import get_db_session
from app.main import app
from app.services.github_risk_collector import (
    GitHubRiskCollectionResult,
    RiskCollectionStatus,
)
from app.services.jira_risk_collector import (
    JiraRiskCollectionResult,
    JiraRiskCollectionStatus,
)


class FakeAgentGitHubRiskCollector:
    """Fake GitHub collector for agent execution integration tests."""

    async def collect_github_risks(
        self,
        *,
        run_id: str,
    ) -> GitHubRiskCollectionResult:
        """Return deterministic GitHub release-risk data."""

        return GitHubRiskCollectionResult(
            status=RiskCollectionStatus.SUCCESS,
            pull_request_count=2,
            risk_result_count=2,
            total_signal_count=3,
            high_risk_count=1,
            risk_results=[],
            collected_at=datetime.now(UTC),
            duration_ms=10.0,
        )


class FakeAgentJiraRiskCollector:
    """Fake Jira collector for agent execution integration tests."""

    async def collect(
        self,
        *,
        run_id: str,
    ) -> JiraRiskCollectionResult:
        """Return deterministic Jira release-risk data."""

        return JiraRiskCollectionResult(
            status=JiraRiskCollectionStatus.SUCCESS,
            issues=[],
            issue_results=[],
            signals=[],
            error_message=None,
            duration_ms=0.0,
        )


@pytest.fixture
async def agent_query_api_client() -> AsyncIterator[AsyncClient]:
    """Provide an isolated API client for query planning and execution."""

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async def override_get_db_session() -> AsyncIterator[AsyncSession]:
        """Provide an isolated database session."""

        async with session_factory() as session:
            yield session

    async def override_get_github_collector() -> FakeAgentGitHubRiskCollector:
        """Return the fake GitHub collector."""

        return FakeAgentGitHubRiskCollector()

    async def override_get_jira_collector() -> FakeAgentJiraRiskCollector:
        """Return the fake Jira collector."""

        return FakeAgentJiraRiskCollector()

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_agent_github_risk_collector] = override_get_github_collector
    app.dependency_overrides[get_agent_jira_risk_collector] = override_get_jira_collector

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            yield client
    finally:
        app.dependency_overrides.clear()

        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)

        await engine.dispose()


@pytest.mark.anyio
async def test_create_agent_query_plan_returns_structured_plan(
    agent_query_api_client: AsyncClient,
) -> None:
    """The API should convert natural language into a query plan."""

    response = await agent_query_api_client.post(
        "/api/v1/agent/query-plan",
        json={"query": "What are the biggest release risks this week?"},
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["intent"] == "release_risk_summary"
    assert payload["response_depth"] == "standard"
    assert payload["requires_current_snapshot"] is True
    assert payload["requires_historical_lookup"] is False
    assert payload["requires_human_approval"] is False
    assert payload["may_execute_side_effect"] is False


@pytest.mark.anyio
async def test_execute_agent_query_runs_release_risk_workflow(
    agent_query_api_client: AsyncClient,
) -> None:
    """The agent query endpoint should execute and persist the workflow."""

    response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "What are the biggest release risks this week?",
        },
    )

    assert response.status_code == 200

    payload = response.json()
    release_run = payload["release_run"]

    assert release_run["id"] is not None
    assert release_run["run_id"].startswith("release-run-")
    assert release_run["query"] == ("What are the biggest release risks this week?")
    assert release_run["requested_by"] == "agent-query-api"
    assert release_run["status"] in {
        "completed",
        "waiting_for_approval",
    }

    assert payload["github"]["status"] == "success"
    assert payload["jira"]["status"] == "success"
    assert payload["release_summary"]["source"] == "release"

    assert payload["risk_features"] is not None
    assert payload["risk_score"] is not None
    assert payload["risk_score"]["scoring_version"] == ("rule_based_release_risk_v1")

    events_response = await agent_query_api_client.get(
        f"/api/v1/release-runs/{release_run['id']}/events"
    )

    assert events_response.status_code == 200

    event_types = {event["event_type"] for event in events_response.json()["events"]}

    assert "workflow_started" in event_types
    assert "workflow_completed" in event_types
    assert "risk_features_extracted" in event_types
    assert "release_risk_scored" in event_types
    assert "release_risk_snapshot_created" in event_types


@pytest.mark.anyio
async def test_execute_agent_query_rejects_unsupported_intent(
    agent_query_api_client: AsyncClient,
) -> None:
    """Unsupported intents should not enter the release workflow."""

    github_override = app.dependency_overrides.pop(get_agent_github_risk_collector)
    jira_override = app.dependency_overrides.pop(get_agent_jira_risk_collector)

    try:
        response = await agent_query_api_client.post(
            "/api/v1/agent/query",
            json={
                "query": "Write a recipe for chocolate cake.",
            },
        )
    finally:
        app.dependency_overrides[get_agent_github_risk_collector] = github_override
        app.dependency_overrides[get_agent_jira_risk_collector] = jira_override

    assert response.status_code == 422

    error_payload = response.json()["error"]

    assert error_payload["code"] == "HTTP_ERROR"
    assert error_payload["message"] == ("This agent query intent is not executable yet.")


@pytest.mark.anyio
async def test_slack_action_requires_human_approval(
    agent_query_api_client: AsyncClient,
) -> None:
    """Slack action requests must remain behind the HITL gate."""

    response = await agent_query_api_client.post(
        "/api/v1/agent/query-plan",
        json={"query": "Can you send this to Slack?"},
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["intent"] == "action_request"
    assert payload["response_depth"] == "action_confirmation"
    assert payload["requires_human_approval"] is True
    assert payload["may_execute_side_effect"] is True


@pytest.mark.anyio
async def test_unrelated_query_is_marked_out_of_scope(
    agent_query_api_client: AsyncClient,
) -> None:
    """Unrelated questions must not enter the release workflow."""

    response = await agent_query_api_client.post(
        "/api/v1/agent/query-plan",
        json={"query": "Write a recipe for chocolate cake."},
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["intent"] == "out_of_scope"
    assert payload["response_depth"] == "brief"
    assert payload["may_execute_side_effect"] is False


@pytest.mark.anyio
async def test_empty_agent_query_is_rejected(
    agent_query_api_client: AsyncClient,
) -> None:
    """Pydantic should reject an empty natural-language query."""

    response = await agent_query_api_client.post(
        "/api/v1/agent/query-plan",
        json={"query": "   "},
    )

    assert response.status_code == 422
