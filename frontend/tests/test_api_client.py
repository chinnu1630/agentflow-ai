"""Tests for the asynchronous AgentFlow API client."""

from __future__ import annotations

from typing import Any
from uuid import UUID

import httpx
import pytest
from pydantic import SecretStr

from agentflow_frontend.api_client import (
    AgentFlowAPIClient,
    AgentFlowAuthenticationError,
    AgentFlowAuthorizationError,
    AgentFlowRateLimitError,
    AgentFlowResponseValidationError,
    AgentFlowServiceUnavailableError,
)
from agentflow_frontend.api_models import AgentQueryRequest
from agentflow_frontend.config import FrontendSettings


def _settings() -> FrontendSettings:
    """Create deterministic test settings."""
    return FrontendSettings(
        backend_base_url="https://agentflow.example.test",
        connect_timeout_seconds=1.0,
        request_timeout_seconds=5.0,
        _env_file=None,
    )


def _successful_response() -> dict[str, Any]:
    """Return a minimal valid agent-query response."""
    return {
        "answer": "The release has one high-risk Jira blocker.",
        "plan": {
            "intent": "release_risk_summary",
            "response_depth": "detailed",
            "confidence": 0.98,
            "requires_current_snapshot": True,
            "requires_human_approval": True,
            "routing_reason_code": "fresh_release_risk_request",
        },
        "citations": [
            {
                "source": "jira",
                "source_type": "jira_issue",
                "source_id": "PAY-102",
                "title": "Payment rollback defect",
            }
        ],
        "approval_required": True,
    }


@pytest.mark.asyncio
async def test_execute_agent_query_sends_jwt_and_parses_response() -> None:
    """Send the expected API request and validate its typed response."""

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v1/agent/query"
        assert request.headers["Authorization"] == "Bearer signed-test-jwt"
        assert request.headers["Content-Type"] == "application/json"

        run_id = request.headers["X-Run-ID"]
        UUID(run_id)

        assert request.content
        assert request.content.decode().find(
            "What are the biggest release risks?"
        ) >= 0

        return httpx.Response(
            status_code=200,
            headers={"X-Run-ID": run_id},
            json=_successful_response(),
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        client = AgentFlowAPIClient(
            settings=_settings(),
            bearer_token=SecretStr("signed-test-jwt"),
            http_client=http_client,
        )

        result = await client.execute_agent_query(
            AgentQueryRequest(
                query="What are the biggest release risks?",
            )
        )

    assert result.response.approval_required is True
    assert result.response.citations[0].source_id == "PAY-102"
    UUID(result.run_id)


@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    [
        (401, AgentFlowAuthenticationError),
        (403, AgentFlowAuthorizationError),
        (429, AgentFlowRateLimitError),
        (503, AgentFlowServiceUnavailableError),
    ],
)
@pytest.mark.asyncio
async def test_execute_agent_query_maps_expected_http_errors(
    status_code: int,
    expected_error: type[Exception],
) -> None:
    """Convert important backend statuses into actionable frontend errors."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=status_code,
            headers={"X-Run-ID": request.headers["X-Run-ID"]},
            json={"detail": "Sensitive backend detail must not be exposed."},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        client = AgentFlowAPIClient(
            settings=_settings(),
            bearer_token=SecretStr("signed-test-jwt"),
            http_client=http_client,
        )

        with pytest.raises(expected_error) as exc_info:
            await client.execute_agent_query(
                AgentQueryRequest(query="Assess this release.")
            )

    assert "Sensitive backend detail" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_execute_agent_query_rejects_invalid_backend_response() -> None:
    """Fail closed when the backend violates its documented response model."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={"X-Run-ID": request.headers["X-Run-ID"]},
            json={"answer": "", "approval_required": False},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        client = AgentFlowAPIClient(
            settings=_settings(),
            bearer_token=SecretStr("signed-test-jwt"),
            http_client=http_client,
        )

        with pytest.raises(AgentFlowResponseValidationError):
            await client.execute_agent_query(
                AgentQueryRequest(query="Assess this release.")
            )


@pytest.mark.asyncio
async def test_execute_agent_query_maps_timeout_to_service_error() -> None:
    """Convert HTTP timeouts into a safe unavailable-service error."""

    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(
            "timed out",
            request=request,
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        client = AgentFlowAPIClient(
            settings=_settings(),
            bearer_token=SecretStr("signed-test-jwt"),
            http_client=http_client,
        )

        with pytest.raises(AgentFlowServiceUnavailableError):
            await client.execute_agent_query(
                AgentQueryRequest(query="Assess this release.")
            )


def test_api_client_rejects_blank_token() -> None:
    """Reject missing authentication before any network request occurs."""
    with pytest.raises(ValueError, match="must not be empty"):
        AgentFlowAPIClient(
            settings=_settings(),
            bearer_token=SecretStr("   "),
        )
