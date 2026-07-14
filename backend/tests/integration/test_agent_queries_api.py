"""Integration tests for the AgentFlow query-planning API."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def agent_query_api_client() -> AsyncIterator[AsyncClient]:
    """Provide an async HTTP client for the Agent Query API.

    The endpoint only performs deterministic query classification, so this
    fixture does not require database setup or external-service overrides.
    """

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


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
