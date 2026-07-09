"""Approved release-risk Slack alert delivery service.

This service enforces AgentFlow's human-approval rule before Slack delivery:
nothing is sent unless the release has been approved.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from json import JSONDecodeError
from typing import Any, Protocol
from uuid import UUID

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.integrations.slack_client import SlackClientError, SlackPostMessageResult
from app.repositories.release_run_approval_repository import (
    ReleaseRunApprovalRepositoryError,
)
from app.repositories.release_run_risk_snapshot_repository import (
    ReleaseRunRiskSnapshotRepositoryError,
)
from app.schemas.risk import ReleaseRunRiskResponse
from app.services.slack_alert_payload_service import (
    SlackAlertPayloadService,
    SlackAlertSeverity,
    SlackReleaseRiskAlertPayload,
    SlackReleaseRiskAlertRequest,
)

logger = structlog.get_logger(__name__)


class SlackAlertSender(Protocol):
    """Protocol for a Slack sender dependency.

    The real implementation is SlackClient. Tests can provide a fake sender.
    """

    async def send_release_risk_alert(
        self,
        payload: SlackReleaseRiskAlertPayload,
    ) -> SlackPostMessageResult:
        """Send a release-risk alert payload to Slack."""


class ReleaseApprovalReader(Protocol):
    """Protocol for reading release-run approval state."""

    async def get_latest_by_release_run_id(self, release_run_id: UUID) -> object | None:
        """Fetch the latest approval request for one release run."""


class ReleaseRiskSnapshotReader(Protocol):
    """Protocol for reading persisted release-risk snapshots."""

    async def get_latest_by_release_run_id(self, release_run_id: UUID) -> object | None:
        """Fetch the latest trusted risk snapshot for one release run."""


class SlackReleaseAlertRequest(BaseModel):
    """Validated request for sending an approved release-risk Slack alert."""

    model_config = ConfigDict(extra="forbid")

    release_run_id: UUID
    run_id: str = Field(min_length=1, max_length=64)
    release_run_status: str = Field(min_length=1, max_length=64)
    requested_by: str = Field(min_length=1, max_length=255)
    approval_status: str | None = Field(default=None, max_length=32)
    approval_request_id: UUID | None = None
    risk_score: dict[str, Any] = Field(default_factory=dict)
    release_summary: dict[str, Any] = Field(default_factory=dict)


class SlackReleaseAlertResult(BaseModel):
    """Result returned after a release-risk Slack alert is sent."""

    model_config = ConfigDict(frozen=True)

    sent: bool
    slack_channel: str
    slack_timestamp: str
    risk_level: str
    risk_score: float
    recommended_action: str


class SlackReleaseAlertNotApprovedError(RuntimeError):
    """Raised when Slack delivery is attempted before approval."""


class SlackReleaseAlertServiceError(RuntimeError):
    """Raised when approved Slack alert delivery fails."""


class SlackReleaseAlertService:
    """Send approved release-risk Slack alerts."""

    def __init__(
        self,
        *,
        payload_service: SlackAlertPayloadService | None = None,
    ) -> None:
        """Initialize the service."""
        self._payload_service = payload_service or SlackAlertPayloadService()

    async def send_approved_release_alert_from_snapshot(
        self,
        release_run_id: UUID,
        *,
        approval_repository: ReleaseApprovalReader,
        risk_snapshot_repository: ReleaseRiskSnapshotReader,
        sender: SlackAlertSender,
        run_id: str | None = None,
    ) -> SlackReleaseAlertResult:
        """Send a Slack alert from the latest approved release-risk snapshot.

        Args:
            release_run_id: Release run database UUID.
            approval_repository: Repository used to verify HITL approval state.
            risk_snapshot_repository: Repository used to load trusted risk data.
            sender: Slack-compatible sender, usually SlackClient.
            run_id: Optional request/workflow ID for safe logs.

        Returns:
            Result describing the sent Slack message.

        Raises:
            SlackReleaseAlertNotApprovedError: If latest approval is not approved.
            SlackReleaseAlertServiceError: If snapshot loading or Slack sending fails.
        """
        try:
            approval = await approval_repository.get_latest_by_release_run_id(
                release_run_id
            )
            approval_id = self._extract_approved_approval_id(approval)

            snapshot = await risk_snapshot_repository.get_latest_by_release_run_id(
                release_run_id
            )

        except (
            ReleaseRunApprovalRepositoryError,
            ReleaseRunRiskSnapshotRepositoryError,
        ) as exc:
            logger.exception(
                "approved_release_slack_alert_snapshot_load_failed",
                run_id=run_id,
                release_run_id=str(release_run_id),
            )
            raise SlackReleaseAlertServiceError(
                "Failed to load approved release-risk snapshot."
            ) from exc

        if snapshot is None:
            logger.error(
                "approved_release_slack_alert_snapshot_missing",
                run_id=run_id,
                release_run_id=str(release_run_id),
            )
            raise SlackReleaseAlertServiceError(
                "No release-risk snapshot found for approved release run."
            )

        snapshot_response = self._load_snapshot_response(snapshot)

        request = self._build_alert_request_from_snapshot(
            response=snapshot_response,
            approval_id=approval_id,
        )

        logger.info(
            "approved_release_slack_alert_snapshot_loaded",
            run_id=run_id,
            release_run_id=str(release_run_id),
            snapshot_id=str(getattr(snapshot, "id", "unknown")),
            snapshot_version=getattr(snapshot, "snapshot_version", None),
            approval_id=str(approval_id),
        )

        return await self.send_approved_release_alert(
            request,
            sender=sender,
            run_id=run_id,
        )

    async def send_approved_release_alert(
        self,
        request: SlackReleaseAlertRequest,
        *,
        sender: SlackAlertSender,
        run_id: str | None = None,
    ) -> SlackReleaseAlertResult:
        """Send a Slack alert only when the release has been approved.

        Args:
            request: Approved release-risk alert request.
            sender: Slack-compatible sender, usually SlackClient.
            run_id: Optional request/workflow ID for safe logs.

        Returns:
            Result describing the sent Slack message.

        Raises:
            SlackReleaseAlertNotApprovedError: If release is not approved.
            SlackReleaseAlertServiceError: If Slack sending fails.
        """
        started_at = time.perf_counter()
        self._validate_approval_gate(request)

        risk_level = self._extract_risk_level(request.risk_score)
        risk_score = self._extract_risk_score(request.risk_score)
        recommended_action = self._extract_recommended_action(request.risk_score)
        top_risk_titles = self._extract_top_risk_titles(request.release_summary)

        payload = self._payload_service.build_release_risk_alert(
            SlackReleaseRiskAlertRequest(
                release_run_id=request.release_run_id,
                run_id=request.run_id,
                release_run_status=request.release_run_status,
                requested_by=request.requested_by,
                risk_level=SlackAlertSeverity(risk_level),
                risk_score=risk_score,
                recommended_action=recommended_action,
                approval_status=request.approval_status,
                approval_request_id=request.approval_request_id,
                top_risk_titles=top_risk_titles,
            ),
            run_id=run_id,
        )

        try:
            send_result = await sender.send_release_risk_alert(payload)
        except SlackClientError as exc:
            logger.exception(
                "approved_release_slack_alert_failed",
                run_id=run_id,
                release_run_id=str(request.release_run_id),
                release_run_status=request.release_run_status,
                approval_status=request.approval_status,
                risk_level=risk_level,
            )
            raise SlackReleaseAlertServiceError(
                "Failed to send approved release Slack alert."
            ) from exc

        logger.info(
            "approved_release_slack_alert_sent",
            run_id=run_id,
            release_run_id=str(request.release_run_id),
            release_run_status=request.release_run_status,
            approval_status=request.approval_status,
            risk_level=risk_level,
            risk_score=risk_score,
            recommended_action=recommended_action,
            slack_channel=send_result.channel,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
        )

        return SlackReleaseAlertResult(
            sent=True,
            slack_channel=send_result.channel,
            slack_timestamp=send_result.timestamp,
            risk_level=risk_level,
            risk_score=risk_score,
            recommended_action=recommended_action,
        )

    @staticmethod
    def _extract_approved_approval_id(approval: object | None) -> UUID:
        """Extract approval ID after verifying latest approval is approved."""
        if approval is None:
            raise SlackReleaseAlertNotApprovedError(
                "Slack alert cannot be sent before approval."
            )

        approval_status = getattr(approval, "approval_status", None)

        if approval_status != "approved":
            raise SlackReleaseAlertNotApprovedError(
                "Slack alert cannot be sent before approval."
            )

        approval_id = getattr(approval, "id", None)

        if not isinstance(approval_id, UUID):
            raise SlackReleaseAlertServiceError(
                "Approved release approval record is missing a valid ID."
            )

        return approval_id

    @staticmethod
    def _load_snapshot_response(snapshot: object) -> ReleaseRunRiskResponse:
        """Load and validate a stored release-risk snapshot payload."""
        risk_payload_json = getattr(snapshot, "risk_payload_json", None)

        if not isinstance(risk_payload_json, str) or not risk_payload_json.strip():
            raise SlackReleaseAlertServiceError(
                "Stored release-risk snapshot payload is empty."
            )

        try:
            payload = json.loads(risk_payload_json)
        except JSONDecodeError as exc:
            raise SlackReleaseAlertServiceError(
                "Stored release-risk snapshot payload is invalid JSON."
            ) from exc

        try:
            return ReleaseRunRiskResponse.model_validate(payload)
        except ValidationError as exc:
            raise SlackReleaseAlertServiceError(
                "Stored release-risk snapshot payload failed validation."
            ) from exc

    @staticmethod
    def _build_alert_request_from_snapshot(
        *,
        response: ReleaseRunRiskResponse,
        approval_id: UUID,
    ) -> SlackReleaseAlertRequest:
        """Build Slack alert input from a trusted release-risk snapshot."""
        risk_score_payload = (
            response.risk_score.model_dump(mode="json")
            if response.risk_score is not None
            else {}
        )

        return SlackReleaseAlertRequest(
            release_run_id=response.release_run.id,
            run_id=response.release_run.run_id,
            release_run_status="approval_approved",
            requested_by=response.release_run.requested_by,
            approval_status="approved",
            approval_request_id=approval_id,
            risk_score=risk_score_payload,
            release_summary=response.release_summary.model_dump(mode="json"),
        )

    @staticmethod
    def _validate_approval_gate(request: SlackReleaseAlertRequest) -> None:
        """Ensure Slack delivery is allowed by HITL approval state."""
        if request.approval_status != "approved":
            raise SlackReleaseAlertNotApprovedError(
                "Slack alert cannot be sent before approval."
            )

        if request.release_run_status != "approval_approved":
            raise SlackReleaseAlertNotApprovedError(
                "Release run must be approval_approved before Slack alert."
            )

    @staticmethod
    def _extract_risk_level(risk_score: Mapping[str, Any]) -> str:
        """Extract normalized risk level from risk score payload."""
        value = risk_score.get("risk_level", "low")
        enum_value = getattr(value, "value", None)
        normalized_value = str(enum_value if enum_value is not None else value).lower()

        if normalized_value not in {"low", "medium", "high", "critical"}:
            return "medium"

        return normalized_value

    @staticmethod
    def _extract_risk_score(risk_score: Mapping[str, Any]) -> float:
        """Extract bounded numeric risk score."""
        raw_score = risk_score.get("score", 0.0)

        try:
            numeric_score = float(raw_score)
        except (TypeError, ValueError):
            return 0.0

        return min(max(numeric_score, 0.0), 1.0)

    @staticmethod
    def _extract_recommended_action(risk_score: Mapping[str, Any]) -> str:
        """Extract recommended action from risk score payload."""
        value = risk_score.get("recommended_action", "review_required")
        enum_value = getattr(value, "value", None)
        normalized_value = str(enum_value if enum_value is not None else value).strip()

        return normalized_value or "review_required"

    @staticmethod
    def _extract_top_risk_titles(release_summary: Mapping[str, Any]) -> list[str]:
        """Extract safe top-risk titles from release summary payload."""
        top_risks = release_summary.get("top_risks", [])

        if not isinstance(top_risks, Sequence) or isinstance(top_risks, str):
            return []

        titles: list[str] = []

        for item in top_risks:
            title = SlackReleaseAlertService._extract_title_from_top_risk(item)

            if title:
                titles.append(title)

        return titles

    @staticmethod
    def _extract_title_from_top_risk(item: object) -> str:
        """Extract one display title from a top-risk item."""
        if isinstance(item, str):
            return item.strip()

        if isinstance(item, Mapping):
            for key in ("title", "summary", "description", "rule_id"):
                value = item.get(key)

                if isinstance(value, str) and value.strip():
                    return value.strip()

        return ""
