"""Tests for durable approved Slack action orchestration."""

from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from app.integrations.slack_client import SlackPostMessageResult
from app.repositories.release_run_event_repository import (
    CreateReleaseRunEventCommand,
)
from app.repositories.release_run_slack_alert_repository import (
    CreateReleaseRunSlackAlertCommand,
    ReleaseRunSlackAlertAlreadySentError,
)
from app.services.slack_alert_payload_service import SlackReleaseRiskAlertPayload
from app.services.slack_release_alert_action_service import (
    SlackReleaseAlertActionService,
)
from app.services.slack_release_alert_service import (
    SlackReleaseAlertNotApprovedError,
)
from tests.services.test_slack_release_alert_service import (
    FakeApprovalRecord,
    FakeApprovalRepository,
    FakeSnapshotRecord,
    FakeSnapshotRepository,
    build_snapshot_payload,
)


class FakeSlackSender:
    """Capture Slack payloads without performing network I/O."""

    def __init__(self) -> None:
        self.sent_payloads: list[SlackReleaseRiskAlertPayload] = []

    async def send_release_risk_alert(
        self,
        payload: SlackReleaseRiskAlertPayload,
    ) -> SlackPostMessageResult:
        """Capture one Slack payload and return a deterministic result."""
        self.sent_payloads.append(payload)
        return SlackPostMessageResult(
            ok=True,
            channel="C1234567890",
            timestamp="12345.6789",
        )


class FakeSlackAlertRepository:
    """Persist a single successful Slack alert in memory."""

    def __init__(self) -> None:
        self.created_commands: list[CreateReleaseRunSlackAlertCommand] = []

    async def get_by_release_run_id(self, release_run_id: UUID) -> object | None:
        """Return no existing delivery record."""
        return None

    async def create_sent_alert(
        self,
        command: CreateReleaseRunSlackAlertCommand,
    ) -> object:
        """Capture the durable Slack record command."""
        self.created_commands.append(command)
        return SimpleNamespace(id=uuid4())


class FakeEventRepository:
    """Capture append-only release audit events."""

    def __init__(self) -> None:
        self.created_commands: list[CreateReleaseRunEventCommand] = []

    async def create(self, command: CreateReleaseRunEventCommand) -> object:
        """Capture one audit event command."""
        self.created_commands.append(command)
        return SimpleNamespace(id=uuid4())


@pytest.mark.anyio
async def test_execute_sends_persists_and_audits_approved_slack_action() -> None:
    """Approved action should send once and persist durable evidence."""
    release_run_id = uuid4()
    approval_id = uuid4()
    snapshot_id = uuid4()
    sender = FakeSlackSender()
    slack_repository = FakeSlackAlertRepository()
    event_repository = FakeEventRepository()

    snapshot_payload = build_snapshot_payload(
        release_run_id=release_run_id,
        approval_request_id=approval_id,
    )

    service = SlackReleaseAlertActionService(
        approval_repository=FakeApprovalRepository(
            FakeApprovalRecord(id=approval_id, approval_status="approved")
        ),
        risk_snapshot_repository=FakeSnapshotRepository(
            FakeSnapshotRecord(
                id=snapshot_id,
                snapshot_version=1,
                risk_payload_json=json.dumps(snapshot_payload),
            )
        ),
        slack_alert_repository=slack_repository,
        event_repository=event_repository,
        sender=sender,
        request_id="request-123",
    )

    result = await service.execute(release_run_id)

    assert result.sent is True
    assert result.slack_channel == "C1234567890"
    assert len(sender.sent_payloads) == 1

    assert len(slack_repository.created_commands) == 1
    persisted = slack_repository.created_commands[0]
    assert persisted.release_run_id == release_run_id
    assert persisted.approval_request_id == approval_id
    assert persisted.snapshot_id == snapshot_id
    assert persisted.snapshot_version == 1

    assert len(event_repository.created_commands) == 1
    audit_event = event_repository.created_commands[0]
    assert audit_event.event_type == "release_slack_alert_sent"
    assert audit_event.event_status == "success"

class FakeExistingSlackAlertRepository(FakeSlackAlertRepository):
    """Return an existing durable Slack delivery record."""

    async def get_by_release_run_id(self, release_run_id: UUID) -> object:
        """Return an existing Slack alert for duplicate protection."""
        return SimpleNamespace(
            id=uuid4(),
            slack_channel="C1234567890",
            slack_timestamp="12345.6789",
        )


@pytest.mark.anyio
async def test_execute_blocks_duplicate_slack_action() -> None:
    """Duplicate Slack actions should be blocked and audited."""
    release_run_id = uuid4()
    sender = FakeSlackSender()
    event_repository = FakeEventRepository()

    service = SlackReleaseAlertActionService(
        approval_repository=FakeApprovalRepository(None),
        risk_snapshot_repository=FakeSnapshotRepository(None),
        slack_alert_repository=FakeExistingSlackAlertRepository(),
        event_repository=event_repository,
        sender=sender,
        request_id="request-123",
    )

    with pytest.raises(
        ReleaseRunSlackAlertAlreadySentError,
        match="Slack alert already sent for this release run.",
    ):
        await service.execute(release_run_id)

    assert sender.sent_payloads == []
    assert len(event_repository.created_commands) == 1
    assert event_repository.created_commands[0].event_status == "blocked"


@pytest.mark.anyio
async def test_execute_blocks_slack_action_before_approval() -> None:
    """Pending approval should block delivery and create an audit event."""
    release_run_id = uuid4()
    approval_id = uuid4()
    sender = FakeSlackSender()
    event_repository = FakeEventRepository()

    service = SlackReleaseAlertActionService(
        approval_repository=FakeApprovalRepository(
            FakeApprovalRecord(id=approval_id, approval_status="pending")
        ),
        risk_snapshot_repository=FakeSnapshotRepository(None),
        slack_alert_repository=FakeSlackAlertRepository(),
        event_repository=event_repository,
        sender=sender,
        request_id="request-123",
    )

    with pytest.raises(
        SlackReleaseAlertNotApprovedError,
        match="Slack alert cannot be sent before approval.",
    ):
        await service.execute(release_run_id)

    assert sender.sent_payloads == []
    assert len(event_repository.created_commands) == 1
    assert event_repository.created_commands[0].event_status == "blocked"

