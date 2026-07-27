"""Tests for manager approval, Slack, status, and audit API models."""

from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from agentflow_frontend.api_models import (
    PendingReleaseRunApprovalList,
    ReleaseRunApprovalDecisionRequest,
    ReleaseRunEventList,
    ReleaseRunStatus,
    SlackReleaseAlertResult,
)


def test_pending_approvals_parse_backend_contract() -> None:
    """Parse the manager approval queue without business-rule duplication."""
    release_run_id = "14326708-c085-4e6d-9c32-47dc92b24841"
    approval_id = "3cc48c03-678b-458e-9418-941e914c220b"

    response = PendingReleaseRunApprovalList.model_validate(
        {
            "approval_status": "pending",
            "approvals": [
                {
                    "id": approval_id,
                    "release_run_id": release_run_id,
                    "approval_status": "pending",
                    "approval_reason": "High release-risk score.",
                    "approval_policy_version": "hitl_policy_v1",
                    "requested_by": "manager@example.com",
                    "created_at": "2026-07-27T12:00:00Z",
                }
            ],
        }
    )

    assert response.approvals[0].id == UUID(approval_id)
    assert response.approvals[0].release_run_id == UUID(release_run_id)


@pytest.mark.parametrize("status", ["approved", "rejected"])
def test_approval_decision_accepts_terminal_status(status: str) -> None:
    """Allow only the two backend-supported terminal decisions."""
    request = ReleaseRunApprovalDecisionRequest(
        approval_status=status,
        decision_note="Reviewed supporting evidence.",
    )

    assert request.approval_status.value == status


def test_approval_decision_rejects_unknown_status() -> None:
    """Fail before sending an unsupported approval state."""
    with pytest.raises(ValidationError):
        ReleaseRunApprovalDecisionRequest(
            approval_status="pending",
        )


def test_approval_decision_rejects_unknown_fields() -> None:
    """Prevent client-side actor spoofing in approval payloads."""
    with pytest.raises(ValidationError):
        ReleaseRunApprovalDecisionRequest.model_validate(
            {
                "approval_status": "approved",
                "decided_by": "spoofed@example.com",
            }
        )


def test_release_status_and_audit_events_parse() -> None:
    """Parse workflow status and its append-only audit timeline."""
    release_run_id = "14326708-c085-4e6d-9c32-47dc92b24841"

    status = ReleaseRunStatus.model_validate(
        {
            "id": release_run_id,
            "run_id": "release-run-demo",
            "query": "What are the biggest release risks?",
            "requested_by": "manager@example.com",
            "status": "waiting_for_approval",
            "created_at": "2026-07-27T12:00:00Z",
        }
    )
    events = ReleaseRunEventList.model_validate(
        {
            "release_run_id": release_run_id,
            "events": [
                {
                    "id": "a07a3fe4-cd3c-42ec-8178-37c27be23de2",
                    "release_run_id": release_run_id,
                    "event_type": "approval_request_created",
                    "event_status": "success",
                    "message": "Pending release approval request was created.",
                    "metadata_json": {
                        "approval_status": "pending",
                    },
                    "created_at": "2026-07-27T12:01:00Z",
                }
            ],
        }
    )

    assert status.status == "waiting_for_approval"
    assert events.events[0].event_type == "approval_request_created"


def test_slack_alert_result_rejects_invalid_score() -> None:
    """Reject malformed Slack delivery results before UI rendering."""
    with pytest.raises(ValidationError):
        SlackReleaseAlertResult.model_validate(
            {
                "sent": True,
                "slack_channel": "release-alerts",
                "slack_timestamp": "1722096000.000100",
                "risk_level": "high",
                "risk_score": 1.2,
                "recommended_action": "hold",
            }
        )


def test_slack_alert_result_parses_success() -> None:
    """Parse a successful approval-gated Slack delivery result."""
    result = SlackReleaseAlertResult.model_validate(
        {
            "sent": True,
            "slack_channel": "release-alerts",
            "slack_timestamp": "1722096000.000100",
            "risk_level": "high",
            "risk_score": 0.88,
            "recommended_action": "hold",
        }
    )

    assert result.sent is True
    assert result.risk_score == 0.88
