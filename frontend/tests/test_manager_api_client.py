"""Tests for manager-facing AgentFlow API client operations."""

from __future__ import annotations

from uuid import UUID

import httpx
import pytest
from pydantic import SecretStr

from agentflow_frontend.api_client import AgentFlowAPIClient
from agentflow_frontend.api_models import ReleaseRunApprovalDecisionRequest
from agentflow_frontend.config import FrontendSettings


def _settings() -> FrontendSettings:
    """Create deterministic frontend settings for API client tests."""
    return FrontendSettings(
        backend_base_url="https://agentflow.example.test",
        connect_timeout_seconds=1.0,
        request_timeout_seconds=5.0,
        _env_file=None,
    )


@pytest.mark.asyncio
async def test_list_pending_approvals_sends_pagination_and_parses_response() -> None:
    """Load the pending approval queue through the release-read API."""

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/v1/release-runs/approvals/pending"
        assert request.url.params["limit"] == "25"
        assert request.url.params["offset"] == "10"
        assert request.headers["Authorization"] == "Bearer signed-test-jwt"

        request_run_id = request.headers["X-Run-ID"]
        UUID(request_run_id)

        return httpx.Response(
            status_code=200,
            headers={"X-Run-ID": request_run_id},
            json={
                "approval_status": "pending",
                "approvals": [
                    {
                        "id": "3cc48c03-678b-458e-9418-941e914c220b",
                        "release_run_id": (
                            "14326708-c085-4e6d-9c32-47dc92b24841"
                        ),
                        "approval_status": "pending",
                        "approval_reason": "High release-risk score.",
                        "approval_policy_version": "hitl_policy_v1",
                        "created_at": "2026-07-27T12:00:00Z",
                    }
                ],
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
    ) as http_client:
        client = AgentFlowAPIClient(
            settings=_settings(),
            bearer_token=SecretStr("signed-test-jwt"),
            http_client=http_client,
        )

        result = await client.list_pending_approvals(
            limit=25,
            offset=10,
        )

    assert result.response.approval_status == "pending"
    assert len(result.response.approvals) == 1
    assert result.response.approvals[0].approval_status == "pending"
    UUID(result.run_id)


@pytest.mark.parametrize(
    ("limit", "offset", "message"),
    [
        (0, 0, "limit"),
        (501, 0, "limit"),
        (100, -1, "offset"),
    ],
)
@pytest.mark.asyncio
async def test_list_pending_approvals_rejects_invalid_pagination(
    limit: int,
    offset: int,
    message: str,
) -> None:
    """Reject invalid pagination before making an HTTP request."""
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: pytest.fail(
                f"Unexpected HTTP request: {request.url}"
            )
        )
    ) as http_client:
        client = AgentFlowAPIClient(
            settings=_settings(),
            bearer_token=SecretStr("signed-test-jwt"),
            http_client=http_client,
        )

        with pytest.raises(ValueError, match=message):
            await client.list_pending_approvals(
                limit=limit,
                offset=offset,
            )


@pytest.mark.asyncio
async def test_decide_release_run_approval_sends_validated_payload() -> None:
    """Approve a pending release through the scoped backend endpoint."""
    release_run_id = "14326708-c085-4e6d-9c32-47dc92b24841"
    approval_id = "3cc48c03-678b-458e-9418-941e914c220b"

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == (
            f"/api/v1/release-runs/{release_run_id}"
            f"/approvals/{approval_id}/decision"
        )
        assert request.headers["Authorization"] == "Bearer signed-test-jwt"
        assert request.content == (
            b'{"approval_status":"approved",'
            b'"decision_note":"Evidence reviewed."}'
        )

        request_run_id = request.headers["X-Run-ID"]

        return httpx.Response(
            status_code=200,
            headers={"X-Run-ID": request_run_id},
            json={
                "id": approval_id,
                "release_run_id": release_run_id,
                "approval_status": "approved",
                "approval_reason": "High release-risk score.",
                "approval_policy_version": "hitl_policy_v1",
                "decided_by": "manager@example.com",
                "decision_note": "Evidence reviewed.",
                "created_at": "2026-07-27T12:00:00Z",
                "decided_at": "2026-07-27T12:05:00Z",
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
    ) as http_client:
        client = AgentFlowAPIClient(
            settings=_settings(),
            bearer_token=SecretStr("signed-test-jwt"),
            http_client=http_client,
        )

        result = await client.decide_release_run_approval(
            release_run_id=release_run_id,
            approval_id=approval_id,
            decision=ReleaseRunApprovalDecisionRequest(
                approval_status="approved",
                decision_note="Evidence reviewed.",
            ),
        )

    assert result.response.approval_status == "approved"
    assert result.response.decided_by == "manager@example.com"
    UUID(result.run_id)


@pytest.mark.asyncio
async def test_get_release_run_status_parses_current_workflow_state() -> None:
    """Load the current persisted release-run status."""
    release_run_id = "14326708-c085-4e6d-9c32-47dc92b24841"

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == f"/api/v1/release-runs/{release_run_id}"

        return httpx.Response(
            status_code=200,
            headers={"X-Run-ID": request.headers["X-Run-ID"]},
            json={
                "id": release_run_id,
                "run_id": "release-run-demo",
                "query": "What are the biggest release risks?",
                "requested_by": "manager@example.com",
                "status": "waiting_for_approval",
                "created_at": "2026-07-27T12:00:00Z",
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
    ) as http_client:
        client = AgentFlowAPIClient(
            settings=_settings(),
            bearer_token=SecretStr("signed-test-jwt"),
            http_client=http_client,
        )

        result = await client.get_release_run_status(
            release_run_id=release_run_id,
        )

    assert result.response.status == "waiting_for_approval"
    UUID(result.run_id)


@pytest.mark.asyncio
async def test_list_release_run_events_parses_audit_timeline() -> None:
    """Load append-only workflow events for manager traceability."""
    release_run_id = "14326708-c085-4e6d-9c32-47dc92b24841"

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == (
            f"/api/v1/release-runs/{release_run_id}/events"
        )

        return httpx.Response(
            status_code=200,
            headers={"X-Run-ID": request.headers["X-Run-ID"]},
            json={
                "release_run_id": release_run_id,
                "events": [
                    {
                        "id": "a07a3fe4-cd3c-42ec-8178-37c27be23de2",
                        "release_run_id": release_run_id,
                        "event_type": "approval_request_created",
                        "event_status": "success",
                        "message": (
                            "Pending release approval request was created."
                        ),
                        "metadata_json": {
                            "approval_status": "pending",
                        },
                        "created_at": "2026-07-27T12:01:00Z",
                    }
                ],
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
    ) as http_client:
        client = AgentFlowAPIClient(
            settings=_settings(),
            bearer_token=SecretStr("signed-test-jwt"),
            http_client=http_client,
        )

        result = await client.list_release_run_events(
            release_run_id=release_run_id,
        )

    assert len(result.response.events) == 1
    assert result.response.events[0].event_type == "approval_request_created"
    UUID(result.run_id)


@pytest.mark.asyncio
async def test_send_release_run_slack_alert_parses_delivery_result() -> None:
    """Send an approved alert through the backend-controlled Slack action."""
    release_run_id = "14326708-c085-4e6d-9c32-47dc92b24841"

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == (
            f"/api/v1/release-runs/{release_run_id}/slack-alert"
        )
        assert request.headers["Authorization"] == "Bearer signed-test-jwt"
        assert request.content == b""

        return httpx.Response(
            status_code=200,
            headers={"X-Run-ID": request.headers["X-Run-ID"]},
            json={
                "sent": True,
                "slack_channel": "release-alerts",
                "slack_timestamp": "1722096000.000100",
                "risk_level": "high",
                "risk_score": 0.88,
                "recommended_action": "hold",
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
    ) as http_client:
        client = AgentFlowAPIClient(
            settings=_settings(),
            bearer_token=SecretStr("signed-test-jwt"),
            http_client=http_client,
        )

        result = await client.send_release_run_slack_alert(
            release_run_id=release_run_id,
        )

    assert result.response.sent is True
    assert result.response.slack_channel == "release-alerts"
    assert result.response.risk_score == 0.88
    UUID(result.run_id)
