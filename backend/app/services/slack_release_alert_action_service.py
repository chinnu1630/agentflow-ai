"""Orchestrate durable, approved Slack release-alert actions."""

from __future__ import annotations

import logging
from typing import Protocol
from uuid import UUID

from app.observability.tracing import start_business_span
from app.repositories.release_run_event_repository import (
    CreateReleaseRunEventCommand,
)
from app.repositories.release_run_slack_alert_repository import (
    CreateReleaseRunSlackAlertCommand,
    ReleaseRunSlackAlertAlreadySentError,
)
from app.services.slack_release_alert_service import (
    ReleaseApprovalReader,
    ReleaseRiskSnapshotReader,
    SlackAlertSender,
    SlackReleaseAlertNotApprovedError,
    SlackReleaseAlertResult,
    SlackReleaseAlertService,
    SlackReleaseAlertServiceError,
)

logger = logging.getLogger(__name__)


class SlackAlertRecordProtocol(Protocol):
    """Persisted Slack alert fields required for idempotency handling."""

    @property
    def id(self) -> UUID:
        """Return the durable Slack alert identifier."""
        ...

    @property
    def slack_channel(self) -> str:
        """Return the destination Slack channel."""
        ...

    @property
    def slack_timestamp(self) -> str:
        """Return the Slack message timestamp."""
        ...


class SlackAlertRepositoryProtocol(Protocol):
    """Repository operations required by Slack action orchestration."""

    async def get_by_release_run_id(
        self,
        release_run_id: UUID,
    ) -> SlackAlertRecordProtocol | None:
        """Fetch an existing Slack alert for one release run."""
        ...

    async def create_sent_alert(
        self,
        command: CreateReleaseRunSlackAlertCommand,
    ) -> SlackAlertRecordProtocol:
        """Persist a successful Slack delivery."""
        ...


class ReleaseRunEventRepositoryProtocol(Protocol):
    """Audit-event persistence required by Slack action orchestration."""

    async def create(
        self,
        command: CreateReleaseRunEventCommand,
    ) -> object:
        """Persist one append-only release-run event."""
        ...


class SlackReleaseAlertActionService:
    """Execute approved Slack actions with persistence and audit evidence."""

    def __init__(
        self,
        *,
        approval_repository: ReleaseApprovalReader,
        risk_snapshot_repository: ReleaseRiskSnapshotReader,
        slack_alert_repository: SlackAlertRepositoryProtocol,
        event_repository: ReleaseRunEventRepositoryProtocol,
        sender: SlackAlertSender,
        request_id: str,
        delivery_service: SlackReleaseAlertService | None = None,
    ) -> None:
        """Initialize the Slack action orchestration service."""

        self._approval_repository = approval_repository
        self._risk_snapshot_repository = risk_snapshot_repository
        self._slack_alert_repository = slack_alert_repository
        self._event_repository = event_repository
        self._sender = sender
        self._request_id = request_id
        self._delivery_service = delivery_service or SlackReleaseAlertService()

    async def execute(
        self,
        release_run_id: UUID,
    ) -> SlackReleaseAlertResult:
        """Send one approved Slack alert and persist durable evidence."""

        with start_business_span(
            "slack.release_alert.duplicate_check",
            {
                "release_run_id": str(release_run_id),
                "run_id": self._request_id,
            },
        ) as span:
            existing_alert = (
                await self._slack_alert_repository.get_by_release_run_id(
                    release_run_id
                )
            )
            duplicate_found = existing_alert is not None
            span.set_attribute("slack.duplicate_found", duplicate_found)

        if existing_alert is not None:
            await self._record_event(
                release_run_id=release_run_id,
                event_status="blocked",
                message="Duplicate Slack alert send was blocked.",
                metadata_json={
                    "reason": "Slack alert already sent for this release run.",
                    "existing_slack_alert_id": str(existing_alert.id),
                    "existing_slack_channel": existing_alert.slack_channel,
                    "existing_slack_timestamp": existing_alert.slack_timestamp,
                },
            )
            raise ReleaseRunSlackAlertAlreadySentError(
                "Slack alert already sent for this release run."
            )

        latest_approval = (
            await self._approval_repository.get_latest_by_release_run_id(
                release_run_id
            )
        )
        latest_snapshot = (
            await self._risk_snapshot_repository.get_latest_by_release_run_id(
                release_run_id
            )
        )

        try:
            result = (
                await self._delivery_service.send_approved_release_alert_from_snapshot(
                    release_run_id,
                    approval_repository=self._approval_repository,
                    risk_snapshot_repository=self._risk_snapshot_repository,
                    sender=self._sender,
                    run_id=self._request_id,
                )
            )
        except SlackReleaseAlertNotApprovedError as exc:
            await self._record_event(
                release_run_id=release_run_id,
                event_status="blocked",
                message=(
                    "Slack alert was blocked because release is not approved."
                ),
                metadata_json={"reason": str(exc)},
            )
            raise
        except SlackReleaseAlertServiceError as exc:
            await self._record_event(
                release_run_id=release_run_id,
                event_status="failed",
                message="Approved release-risk Slack alert failed.",
                metadata_json={"error_type": exc.__class__.__name__},
            )
            raise

        alert_record = await self._slack_alert_repository.create_sent_alert(
            CreateReleaseRunSlackAlertCommand(
                release_run_id=release_run_id,
                approval_request_id=self._extract_uuid(
                    latest_approval,
                    "id",
                ),
                snapshot_id=self._extract_uuid(
                    latest_snapshot,
                    "id",
                ),
                snapshot_version=self._extract_snapshot_version(
                    latest_snapshot
                ),
                slack_channel=result.slack_channel,
                slack_timestamp=result.slack_timestamp,
                risk_level=result.risk_level,
                risk_score=result.risk_score,
                recommended_action=result.recommended_action,
            )
        )

        await self._record_event(
            release_run_id=release_run_id,
            event_status="success",
            message="Approved release-risk Slack alert was sent.",
            metadata_json={
                "slack_alert_id": str(alert_record.id),
                "slack_channel": result.slack_channel,
                "slack_timestamp": result.slack_timestamp,
                "risk_level": result.risk_level,
                "risk_score": result.risk_score,
                "recommended_action": result.recommended_action,
            },
        )

        logger.info(
            "slack_release_alert_action_completed",
            extra={
                "run_id": self._request_id,
                "release_run_id": str(release_run_id),
                "slack_channel": result.slack_channel,
                "risk_level": result.risk_level,
            },
        )

        return result

    async def _record_event(
        self,
        *,
        release_run_id: UUID,
        event_status: str,
        message: str,
        metadata_json: dict[str, object],
    ) -> None:
        """Persist one safe Slack action audit event."""

        await self._event_repository.create(
            CreateReleaseRunEventCommand(
                release_run_id=release_run_id,
                event_type="release_slack_alert_sent",
                event_status=event_status,
                message=message,
                metadata_json=metadata_json,
            )
        )

    @staticmethod
    def _extract_uuid(record: object | None, field_name: str) -> UUID | None:
        """Extract an optional UUID from a persisted record."""

        value = getattr(record, field_name, None)
        return value if isinstance(value, UUID) else None

    @staticmethod
    def _extract_snapshot_version(snapshot: object | None) -> int | None:
        """Extract a valid persisted snapshot version."""

        value = getattr(snapshot, "snapshot_version", None)

        if isinstance(value, int) and value >= 1:
            return value

        return None
