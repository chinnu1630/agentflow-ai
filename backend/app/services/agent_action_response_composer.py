"""Compose confirmations for approved AgentFlow actions."""

from __future__ import annotations

import logging
from typing import Protocol

from app.schemas.agent_query import AgentQueryPlan, AgentQueryResponse
from app.schemas.risk import ReleaseRunRiskResponse

logger = logging.getLogger(__name__)


class SlackActionResultProtocol(Protocol):
    """Slack delivery fields required for an action confirmation."""

    @property
    def sent(self) -> bool:
        """Return whether Slack delivery succeeded."""
        ...

    @property
    def slack_channel(self) -> str:
        """Return the destination Slack channel."""
        ...

    @property
    def slack_timestamp(self) -> str:
        """Return the Slack message timestamp."""
        ...

    @property
    def risk_level(self) -> str:
        """Return the delivered risk level."""
        ...

    @property
    def risk_score(self) -> float:
        """Return the delivered risk score."""
        ...

    @property
    def recommended_action(self) -> str:
        """Return the delivered recommendation."""
        ...


class AgentActionResponseComposerMixin:
    """Compose deterministic confirmations for executed agent actions."""

    _request_id: str

    def compose_slack_action_confirmation(
        self,
        *,
        plan: AgentQueryPlan,
        release_risk: ReleaseRunRiskResponse,
        slack_result: SlackActionResultProtocol,
    ) -> AgentQueryResponse:
        """Compose confirmation after an approved Slack alert is sent."""

        risk_percentage = round(slack_result.risk_score * 100)
        readable_risk_level = slack_result.risk_level.replace("_", " ")
        readable_action = slack_result.recommended_action.replace("_", " ")

        answer = (
            "Slack alert sent successfully. "
            f"Channel: {slack_result.slack_channel}. "
            f"Risk level: {readable_risk_level}. "
            f"Risk score: {risk_percentage}%. "
            f"Recommended action: {readable_action}."
        )

        logger.info(
            "agent_slack_action_confirmation_composed",
            extra={
                "run_id": self._request_id,
                "release_run_id": str(release_risk.release_run.id),
                "intent": plan.intent.value,
                "sent": slack_result.sent,
                "slack_channel": slack_result.slack_channel,
                "risk_level": slack_result.risk_level,
                "risk_score": slack_result.risk_score,
                "citation_count": 0,
            },
        )

        return AgentQueryResponse(
            answer=answer,
            plan=plan,
            release_risk=release_risk,
            citations=[],
            approval_required=release_risk.approval_required is True,
        )
