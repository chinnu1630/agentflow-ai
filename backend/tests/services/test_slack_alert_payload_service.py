"""Tests for Slack release-risk alert payload builder."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.services.slack_alert_payload_service import (
    SlackAlertPayloadService,
    SlackAlertSeverity,
    SlackReleaseRiskAlertRequest,
)


def test_build_release_risk_alert_payload_includes_safe_summary() -> None:
    """Payload builder should produce Slack-compatible safe message payload."""
    release_run_id = uuid4()
    approval_request_id = uuid4()
    service = SlackAlertPayloadService()

    payload = service.build_release_risk_alert(
        SlackReleaseRiskAlertRequest(
            release_run_id=release_run_id,
            run_id="release-run-001",
            release_run_status="approval_approved",
            requested_by="manager@example.com",
            risk_level=SlackAlertSeverity.HIGH,
            risk_score=0.78,
            recommended_action="review_required",
            approval_status="approved",
            approval_request_id=approval_request_id,
            top_risk_titles=[
                "Payment API has failing CI",
                "P1 checkout bug remains open",
            ],
        ),
        run_id="test-run-id",
    )

    assert payload.text == "AgentFlow release risk alert: HIGH risk"
    assert payload.metadata["release_run_id"] == str(release_run_id)
    assert payload.metadata["approval_request_id"] == str(approval_request_id)
    assert payload.metadata["risk_level"] == "high"
    assert payload.metadata["risk_score"] == 0.78
    assert payload.metadata["top_risk_count"] == 2
    assert len(payload.blocks) == 4
    assert payload.blocks[0]["type"] == "header"
    assert "Payment API has failing CI" in payload.blocks[3]["text"]["text"]


def test_build_release_risk_alert_payload_limits_top_risks() -> None:
    """Payload builder should cap top risks to five items."""
    service = SlackAlertPayloadService()

    payload = service.build_release_risk_alert(
        SlackReleaseRiskAlertRequest(
            release_run_id=uuid4(),
            run_id="release-run-002",
            release_run_status="approval_approved",
            requested_by="manager@example.com",
            risk_level=SlackAlertSeverity.CRITICAL,
            risk_score=0.95,
            recommended_action="block_release",
            approval_status="approved",
            top_risk_titles=[
                "Risk 1",
                "Risk 2",
                "Risk 3",
                "Risk 4",
                "Risk 5",
                "Risk 6",
            ],
        )
    )

    assert payload.metadata["top_risk_count"] == 5
    assert "Risk 5" in payload.blocks[3]["text"]["text"]
    assert "Risk 6" not in payload.blocks[3]["text"]["text"]


def test_build_release_risk_alert_payload_omits_empty_top_risk_block() -> None:
    """Payload should omit top-risk block when no usable risk titles exist."""
    service = SlackAlertPayloadService()

    payload = service.build_release_risk_alert(
        SlackReleaseRiskAlertRequest(
            release_run_id=uuid4(),
            run_id="release-run-003",
            release_run_status="completed",
            requested_by="manager@example.com",
            risk_level=SlackAlertSeverity.LOW,
            risk_score=0.1,
            recommended_action="proceed",
            top_risk_titles=["", "   "],
        )
    )

    assert payload.metadata["top_risk_count"] == 0
    assert len(payload.blocks) == 3


def test_slack_release_risk_alert_request_rejects_invalid_score() -> None:
    """Request model should reject risk scores outside 0..1."""
    with pytest.raises(ValidationError):
        SlackReleaseRiskAlertRequest(
            release_run_id=uuid4(),
            run_id="release-run-004",
            release_run_status="completed",
            requested_by="manager@example.com",
            risk_level=SlackAlertSeverity.MEDIUM,
            risk_score=1.5,
            recommended_action="review_required",
        )


def test_slack_release_risk_alert_request_rejects_unknown_severity() -> None:
    """Request model should reject unsupported severity values."""
    with pytest.raises(ValidationError):
        SlackReleaseRiskAlertRequest(
            release_run_id=uuid4(),
            run_id="release-run-005",
            release_run_status="completed",
            requested_by="manager@example.com",
            risk_level="urgent",
            risk_score=0.5,
            recommended_action="review_required",
        )
