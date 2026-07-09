"""Tests for approved release Slack alert service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

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


@dataclass(frozen=True)
class FakeApprovalRecord:
    """Fake approval record for snapshot-backed Slack tests."""

    id: UUID
    approval_status: str


@dataclass(frozen=True)
class FakeSnapshotRecord:
    """Fake snapshot record for snapshot-backed Slack tests."""

    id: UUID
    snapshot_version: int
    risk_payload_json: str


class FakeApprovalRepository:
    """Fake approval repository for Slack service tests."""

    def __init__(self, approval: FakeApprovalRecord | None) -> None:
        """Initialize fake repository with latest approval result."""
        self.approval = approval

    async def get_latest_by_release_run_id(
        self,
        release_run_id: UUID,
    ) -> FakeApprovalRecord | None:
        """Return configured approval record."""
        return self.approval


class FakeSnapshotRepository:
    """Fake risk snapshot repository for Slack service tests."""

    def __init__(self, snapshot: FakeSnapshotRecord | None) -> None:
        """Initialize fake repository with latest snapshot result."""
        self.snapshot = snapshot

    async def get_latest_by_release_run_id(
        self,
        release_run_id: UUID,
    ) -> FakeSnapshotRecord | None:
        """Return configured snapshot record."""
        return self.snapshot


def build_snapshot_payload(
    *,
    release_run_id: UUID,
    approval_request_id: UUID,
) -> dict[str, object]:
    """Build a valid ReleaseRunRiskResponse-shaped snapshot payload."""
    now = datetime.now(UTC).isoformat()

    return {
        "release_run": {
            "id": str(release_run_id),
            "run_id": "release-run-001",
            "query": "What are the biggest release risks this week?",
            "requested_by": "manager@example.com",
            "status": "waiting_for_approval",
            "created_at": now,
            "completed_at": now,
        },
        "github": {
            "source": "github",
            "status": "success",
            "pull_request_count": 1,
            "risk_result_count": 1,
            "total_signal_count": 1,
            "high_risk_count": 1,
            "risk_results": [],
            "error_type": None,
            "error_message": None,
            "collected_at": now,
            "duration_ms": 10.0,
        },
        "github_summary": {
            "source": "github",
            "collection_status": "success",
            "overall_severity": "high",
            "recommended_action": "review_required",
            "pull_request_count": 1,
            "risky_pull_request_count": 1,
            "total_signal_count": 1,
            "high_risk_count": 1,
            "top_risks": [],
            "summary_text": "GitHub risks require review.",
            "generated_at": now,
        },
        "jira": {
            "status": "success",
            "total_issues_analyzed": 0,
            "total_signals": 0,
            "issues": [],
            "signals": [],
            "error_message": None,
            "duration_ms": 0.0,
        },
        "jira_summary": {
            "source": "jira",
            "collection_status": "success",
            "overall_severity": "low",
            "recommended_action": "proceed",
            "issue_count": 0,
            "risky_issue_count": 0,
            "total_signal_count": 0,
            "high_risk_count": 0,
            "top_risks": [],
            "summary_text": "No Jira release blockers found.",
            "generated_at": now,
        },
        "release_summary": {
            "source": "release",
            "overall_severity": "high",
            "recommended_action": "review_required",
            "total_signal_count": 1,
            "high_risk_count": 1,
            "source_summary_count": 2,
            "top_risks": [
                {
                    "source": "github",
                    "source_type": "github_pull_request",
                    "source_id": "1",
                    "source_url": "https://github.example/pr/1",
                    "severity": "high",
                    "score": 0.78,
                    "title": "Payment API has failing CI",
                    "reason": "CI failed on a release-critical service.",
                    "evidence": {},
                }
            ],
            "source_summaries": [],
            "summary_text": "Release requires review before Slack notification.",
            "generated_at": now,
        },
        "knowledge_query": None,
        "knowledge_status": None,
        "knowledge_results": [],
        "knowledge_error": None,
        "risk_features": None,
        "risk_score": {
            "scoring_version": "rule_based_release_risk_v1",
            "feature_version": "release_risk_features_v1",
            "generated_at": now,
            "score": 0.78,
            "risk_level": "high",
            "recommended_action": "review_required",
            "reasons": ["High-risk GitHub signal detected."],
            "component_scores": {"github": 0.78},
        },
        "approval_required": True,
        "approval_reason": "High risk requires manager approval.",
        "approval_policy_version": "hitl_policy_v1",
        "approval_request_id": str(approval_request_id),
        "approval_status": "pending",
    }


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

@pytest.mark.anyio
async def test_send_approved_release_alert_from_snapshot_sends_stored_report() -> None:
    """Service should send Slack alert using the latest trusted snapshot."""
    service = SlackReleaseAlertService()
    sender = FakeSlackSender()
    release_run_id = uuid4()
    approval_id = uuid4()
    snapshot_payload = build_snapshot_payload(
        release_run_id=release_run_id,
        approval_request_id=approval_id,
    )

    result = await service.send_approved_release_alert_from_snapshot(
        release_run_id,
        approval_repository=FakeApprovalRepository(
            FakeApprovalRecord(
                id=approval_id,
                approval_status="approved",
            )
        ),
        risk_snapshot_repository=FakeSnapshotRepository(
            FakeSnapshotRecord(
                id=uuid4(),
                snapshot_version=1,
                risk_payload_json=json.dumps(snapshot_payload),
            )
        ),
        sender=sender,
        run_id="test-run-id",
    )

    assert result.sent is True
    assert result.risk_level == "high"
    assert result.risk_score == 0.78
    assert result.recommended_action == "review_required"

    assert len(sender.sent_payloads) == 1
    payload = sender.sent_payloads[0]

    assert payload.metadata["release_run_id"] == str(release_run_id)
    assert payload.metadata["approval_status"] == "approved"
    assert payload.metadata["approval_request_id"] == str(approval_id)
    assert payload.metadata["release_run_status"] == "approval_approved"
    assert payload.metadata["risk_level"] == "high"
    assert "Payment API has failing CI" in payload.blocks[3]["text"]["text"]


@pytest.mark.anyio
async def test_send_approved_release_alert_from_snapshot_rejects_pending_approval() -> None:
    """Service should reject snapshot-backed Slack send before approval."""
    service = SlackReleaseAlertService()
    sender = FakeSlackSender()
    release_run_id = uuid4()

    with pytest.raises(
        SlackReleaseAlertNotApprovedError,
        match="Slack alert cannot be sent before approval.",
    ):
        await service.send_approved_release_alert_from_snapshot(
            release_run_id,
            approval_repository=FakeApprovalRepository(
                FakeApprovalRecord(
                    id=uuid4(),
                    approval_status="pending",
                )
            ),
            risk_snapshot_repository=FakeSnapshotRepository(None),
            sender=sender,
        )

    assert sender.sent_payloads == []


@pytest.mark.anyio
async def test_send_approved_release_alert_from_snapshot_requires_snapshot() -> None:
    """Service should fail safely when approved release has no snapshot."""
    service = SlackReleaseAlertService()
    sender = FakeSlackSender()

    with pytest.raises(
        SlackReleaseAlertServiceError,
        match="No release-risk snapshot found for approved release run.",
    ):
        await service.send_approved_release_alert_from_snapshot(
            uuid4(),
            approval_repository=FakeApprovalRepository(
                FakeApprovalRecord(
                    id=uuid4(),
                    approval_status="approved",
                )
            ),
            risk_snapshot_repository=FakeSnapshotRepository(None),
            sender=sender,
        )

    assert sender.sent_payloads == []


@pytest.mark.anyio
async def test_send_approved_release_alert_from_snapshot_rejects_invalid_snapshot_json() -> None:
    """Service should reject corrupt stored snapshot JSON."""
    service = SlackReleaseAlertService()
    sender = FakeSlackSender()

    with pytest.raises(
        SlackReleaseAlertServiceError,
        match="Stored release-risk snapshot payload is invalid JSON.",
    ):
        await service.send_approved_release_alert_from_snapshot(
            uuid4(),
            approval_repository=FakeApprovalRepository(
                FakeApprovalRecord(
                    id=uuid4(),
                    approval_status="approved",
                )
            ),
            risk_snapshot_repository=FakeSnapshotRepository(
                FakeSnapshotRecord(
                    id=uuid4(),
                    snapshot_version=1,
                    risk_payload_json="{not-json",
                )
            ),
            sender=sender,
        )

    assert sender.sent_payloads == []
