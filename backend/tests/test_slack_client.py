"""Tests for Slack Web API client."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from app.integrations.slack_client import (
    SlackClient,
    SlackClientConfig,
    SlackClientError,
)
from app.services.slack_alert_payload_service import SlackReleaseRiskAlertPayload


def build_payload() -> SlackReleaseRiskAlertPayload:
    """Build a reusable Slack alert payload for client tests."""
    return SlackReleaseRiskAlertPayload(
        text="AgentFlow release risk alert: HIGH risk",
        blocks=[
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "Release risk summary.",
                },
            }
        ],
        metadata={
            "release_run_id": "release-run-id",
            "run_id": "release-run-001",
            "risk_level": "high",
            "risk_score": 0.8,
            "recommended_action": "review_required",
            "top_risk_count": 1,
        },
    )


def build_config() -> SlackClientConfig:
    """Build Slack client config for tests."""
    return SlackClientConfig(
        bot_token=SecretStr("xoxb-test-token"),
        channel_id="C1234567890",
        api_base_url="https://slack.test/api",
        timeout_seconds=5.0,
        max_retries=2,
        retry_base_delay_seconds=0.0,
    )


@pytest.mark.anyio
async def test_send_release_risk_alert_posts_payload_to_slack() -> None:
    """Client should send Slack-compatible JSON payload."""
    captured_request_json: dict[str, Any] = {}
    captured_authorization_header = ""

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_authorization_header

        captured_authorization_header = request.headers["Authorization"]
        captured_request_json.update(json.loads(request.content.decode()))

        return httpx.Response(
            status_code=200,
            json={
                "ok": True,
                "channel": "C1234567890",
                "ts": "12345.6789",
            },
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = SlackClient(
            http_client=http_client,
            config=build_config(),
            request_id="test-request-id",
        )

        result = await client.send_release_risk_alert(build_payload())

    assert result.ok is True
    assert result.channel == "C1234567890"
    assert result.timestamp == "12345.6789"
    assert captured_authorization_header == "Bearer xoxb-test-token"
    assert captured_request_json["channel"] == "C1234567890"
    assert captured_request_json["text"] == (
        "AgentFlow release risk alert: HIGH risk"
    )
    assert captured_request_json["metadata"]["event_type"] == (
        "agentflow_release_risk_alert"
    )
    assert captured_request_json["metadata"]["event_payload"]["risk_level"] == "high"
    assert "xoxb-test-token" not in str(captured_request_json)


@pytest.mark.anyio
async def test_send_release_risk_alert_retries_after_rate_limit() -> None:
    """Client should retry HTTP 429 responses."""
    call_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            return httpx.Response(
                status_code=429,
                headers={"Retry-After": "0"},
                json={"ok": False, "error": "ratelimited"},
            )

        return httpx.Response(
            status_code=200,
            json={
                "ok": True,
                "channel": "C1234567890",
                "ts": "12345.6789",
            },
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = SlackClient(
            http_client=http_client,
            config=build_config(),
            request_id="test-request-id",
        )

        result = await client.send_release_risk_alert(build_payload())

    assert result.ok is True
    assert call_count == 2


@pytest.mark.anyio
async def test_send_release_risk_alert_retries_after_server_error() -> None:
    """Client should retry temporary Slack 5xx responses."""
    call_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            return httpx.Response(
                status_code=500,
                json={"ok": False, "error": "internal_error"},
            )

        return httpx.Response(
            status_code=200,
            json={
                "ok": True,
                "channel": "C1234567890",
                "ts": "12345.6789",
            },
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = SlackClient(
            http_client=http_client,
            config=build_config(),
            request_id="test-request-id",
        )

        result = await client.send_release_risk_alert(build_payload())

    assert result.ok is True
    assert call_count == 2


@pytest.mark.anyio
async def test_send_release_risk_alert_raises_for_non_retryable_slack_error() -> None:
    """Client should raise when Slack rejects request with non-retryable error."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={"ok": False, "error": "invalid_auth"},
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = SlackClient(
            http_client=http_client,
            config=build_config(),
            request_id="test-request-id",
        )

        with pytest.raises(
            SlackClientError,
            match="Slack API rejected message: invalid_auth",
        ):
            await client.send_release_risk_alert(build_payload())


@pytest.mark.anyio
async def test_send_release_risk_alert_raises_after_retry_exhaustion() -> None:
    """Client should raise after exhausting retry attempts."""
    call_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1

        return httpx.Response(
            status_code=500,
            json={"ok": False, "error": "internal_error"},
        )

    transport = httpx.MockTransport(handler)
    config = build_config().model_copy(update={"max_retries": 1})

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = SlackClient(
            http_client=http_client,
            config=config,
            request_id="test-request-id",
        )

        with pytest.raises(
            SlackClientError,
            match="Slack message delivery failed after retryable HTTP response",
        ):
            await client.send_release_risk_alert(build_payload())

    assert call_count == 2


def test_slack_client_config_rejects_blank_channel() -> None:
    """Slack config should reject blank channel IDs."""
    with pytest.raises(ValidationError):
        SlackClientConfig(
            bot_token=SecretStr("xoxb-test-token"),
            channel_id=" ",
        )
