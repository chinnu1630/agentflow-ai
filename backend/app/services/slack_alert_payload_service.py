"""Slack alert payload builder for release-risk notifications.

This module does not send Slack messages. It only builds safe, deterministic
Slack-compatible payloads that can later be sent by a Slack client after human
approval.
"""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Any
from uuid import UUID

import structlog
from pydantic import BaseModel, ConfigDict, Field

logger = structlog.get_logger(__name__)


class SlackAlertSeverity(StrEnum):
    """Supported release-risk alert severities."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SlackReleaseRiskAlertRequest(BaseModel):
    """Validated input for building a Slack release-risk alert payload."""

    model_config = ConfigDict(extra="forbid")

    release_run_id: UUID
    run_id: str = Field(min_length=1, max_length=64)
    release_run_status: str = Field(min_length=1, max_length=64)
    requested_by: str = Field(min_length=1, max_length=255)
    risk_level: SlackAlertSeverity
    risk_score: float = Field(ge=0.0, le=1.0)
    recommended_action: str = Field(min_length=1, max_length=100)
    approval_status: str | None = Field(default=None, max_length=32)
    approval_request_id: UUID | None = None
    top_risk_titles: list[str] = Field(default_factory=list)


class SlackReleaseRiskAlertPayload(BaseModel):
    """Slack-compatible message payload.

    This shape is intentionally generic so a future Slack client can send it
    without needing to know how the message was constructed.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    blocks: list[dict[str, Any]]
    metadata: dict[str, Any]


class SlackAlertPayloadService:
    """Build safe Slack payloads for approved release-risk alerts."""

    def build_release_risk_alert(
        self,
        request: SlackReleaseRiskAlertRequest,
        *,
        run_id: str | None = None,
    ) -> SlackReleaseRiskAlertPayload:
        """Build a deterministic Slack payload for a release-risk result.

        Args:
            request: Validated release-risk alert data.
            run_id: Optional request/workflow ID for safe structured logging.

        Returns:
            Slack-compatible payload without sending it.
        """
        started_at = time.perf_counter()

        title = self._build_title(request)
        summary = self._build_summary(request)
        top_risks = self._safe_top_risks(request.top_risk_titles)

        blocks: list[dict[str, Any]] = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": title,
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Risk Level:* {request.risk_level.value.upper()}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Score:* {request.risk_score:.2f}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Action:* {request.recommended_action}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Status:* {request.release_run_status}",
                    },
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": summary,
                },
            },
        ]

        if top_risks:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": self._format_top_risks(top_risks),
                    },
                }
            )

        metadata = {
            "release_run_id": str(request.release_run_id),
            "run_id": request.run_id,
            "risk_level": request.risk_level.value,
            "risk_score": request.risk_score,
            "recommended_action": request.recommended_action,
            "release_run_status": request.release_run_status,
            "approval_status": request.approval_status,
            "approval_request_id": (
                str(request.approval_request_id)
                if request.approval_request_id is not None
                else None
            ),
            "top_risk_count": len(top_risks),
        }

        payload = SlackReleaseRiskAlertPayload(
            text=title,
            blocks=blocks,
            metadata=metadata,
        )

        logger.info(
            "slack_release_risk_alert_payload_built",
            run_id=run_id,
            release_run_id=str(request.release_run_id),
            risk_level=request.risk_level.value,
            risk_score=request.risk_score,
            recommended_action=request.recommended_action,
            release_run_status=request.release_run_status,
            approval_status=request.approval_status,
            top_risk_count=len(top_risks),
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
        )

        return payload

    @staticmethod
    def _build_title(request: SlackReleaseRiskAlertRequest) -> str:
        """Build a concise Slack fallback/title text."""
        return (
            "AgentFlow release risk alert: "
            f"{request.risk_level.value.upper()} risk"
        )

    @staticmethod
    def _build_summary(request: SlackReleaseRiskAlertRequest) -> str:
        """Build safe human-readable alert summary."""
        approval_text = (
            f" Approval status: `{request.approval_status}`."
            if request.approval_status is not None
            else ""
        )

        return (
            f"Release run `{request.run_id}` was requested by "
            f"`{request.requested_by}`. Recommended action: "
            f"`{request.recommended_action}`.{approval_text}"
        )

    @staticmethod
    def _safe_top_risks(top_risk_titles: list[str]) -> list[str]:
        """Normalize and truncate top risk titles for Slack."""
        safe_titles: list[str] = []

        for title in top_risk_titles[:5]:
            stripped_title = title.strip()

            if not stripped_title:
                continue

            safe_titles.append(stripped_title[:200])

        return safe_titles

    @staticmethod
    def _format_top_risks(top_risks: list[str]) -> str:
        """Format top risks as a compact Slack markdown list."""
        formatted_risks = "\n".join(
            f"{index}. {risk_title}"
            for index, risk_title in enumerate(top_risks, start=1)
        )

        return f"*Top risks:*\n{formatted_risks}"
