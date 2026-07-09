"""Tests for approved release Slack alert service."""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.integrations.slack_client import SlackClientError, SlackPostMessageResult
from app.services.slack_alert_payload_service import SlackReleaseRiskAlertPayload
from app.services.slack_release_alert_service import (
    SlackReleaseAlertNotApprovedError,
    SlackReleaseAlertRequest,
    SlackReleaseAlertService,
    SlackReleaseAlertServiceError,
)


class FakeSlackSender:
    """Fake Slack sender for unit tests."""

    def __init__(self) -> None:
        """Initialize fake sender."""
        self.sent_payloads: list[SlackReleaseRiskAlertPayload] = []

    async def send_release_risk_alert(
        self,
        payload: SlackReleaseRiskAlertPayload,
    ) -> SlackPostMessageResult:
        """Capture payload and return fake Slack result."""
        self.sent_payloads.append(payload)

        return SlackPostMessageResult(
            ok=True,
            channel="C1234567890",
            timestamp="12345.6789",
        )


class FailingSlackSender:
    """Fake Slack sender that simulates Slack delivery failure."""

    async def send_release_risk_alert(
        self,
        payload: SlackReleaseRiskAlertPayload,
    ) -> SlackPostMessageResult:
        """Raise a Slack client error."""
        raise SlackClientError("Slack failed.")


def build_approved_request() -> SlackReleaseAlertRequest:
    """Build reusable approved Slack alert request."""
    return SlackReleaseAlertRequest(
        release_run_id=uuid4(),
        run_id="release-run-001",
        release_run_status="approval_approved",
        requested_by="manager@example.com",
        approval_status="approved",
        approval_request_id=uuid4(),
        risk_score={
            "risk_level": "high",
            "score": 0.78,
            "recommended_action": "review_required",
        },
        release_summary={
            "top_risks": [
                {"title": "Payment API has failing CI"},
                {"title": "P1 checkout bug remains open"},
            ]
        },
    )


@pytest.mark.anyio
async def test_send_approved_release_alert_sends_payload() -> None:
    """Service should send Slack alert after approval."""
    service = SlackReleaseAlertService()
    sender = FakeSlackSender()

    result = await service.send_approved_release_alert(
        build_approved_request(),
        sender=sender,
        run_id="test-run-id",
    )

    assert result.sent is True
    assert result.slack_channel == "C1234567890"
    assert result.slack_timestamp == "12345.6789"
    assert result.risk_level == "high"
    assert result.risk_score == 0.78
    assert result.recommended_action == "review_required"

    assert len(sender.sent_payloads) == 1
    payload = sender.sent_payloads[0]

    assert payload.metadata["approval_status"] == "approved"
    assert payload.metadata["risk_level"] == "high"
    assert payload.metadata["top_risk_count"] == 2
    assert "Payment API has failing CI" in payload.blocks[3]["text"]["text"]


@pytest.mark.anyio
async def test_send_approved_release_alert_rejects_pending_approval() -> None:
    """Service should not send Slack alert before approval."""
    service = SlackReleaseAlertService()
    sender = FakeSlackSender()
    request = build_approved_request().model_copy(
        update={
            "release_run_status": "waiting_for_approval",
            "approval_status": "pending",
        }
    )

    with pytest.raises(
        SlackReleaseAlertNotApprovedError,
        match="Slack alert cannot be sent before approval.",
    ):
        await service.send_approved_release_alert(request, sender=sender)

    assert sender.sent_payloads == []


@pytest.mark.anyio
async def test_send_approved_release_alert_rejects_wrong_release_status() -> None:
    """Service should require approval_approved release-run status."""
    service = SlackReleaseAlertService()
    sender = FakeSlackSender()
    request = build_approved_request().model_copy(
        update={"release_run_status": "completed"}
    )

    with pytest.raises(
        SlackReleaseAlertNotApprovedError,
        match="Release run must be approval_approved before Slack alert.",
    ):
        await service.send_approved_release_alert(request, sender=sender)

    assert sender.sent_payloads == []


@pytest.mark.anyio
async def test_send_approved_release_alert_wraps_slack_client_error() -> None:
    """Service should wrap Slack client delivery errors."""
    service = SlackReleaseAlertService()

    with pytest.raises(
        SlackReleaseAlertServiceError,
        match="Failed to send approved release Slack alert.",
    ):
        await service.send_approved_release_alert(
            build_approved_request(),
            sender=FailingSlackSender(),
        )


@pytest.mark.anyio
async def test_send_approved_release_alert_defaults_unknown_risk_level_to_medium() -> None:
    """Service should safely normalize unexpected risk levels."""
    service = SlackReleaseAlertService()
    sender = FakeSlackSender()
    request = build_approved_request().model_copy(
        update={
            "risk_score": {
                "risk_level": "urgent",
                "score": 2.5,
                "recommended_action": "review_required",
            }
        }
    )

    result = await service.send_approved_release_alert(request, sender=sender)

    assert result.risk_level == "medium"
    assert result.risk_score == 1.0
    assert sender.sent_payloads[0].metadata["risk_level"] == "medium"
    assert sender.sent_payloads[0].metadata["risk_score"] == 1.0
