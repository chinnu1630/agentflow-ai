"""Compose persisted workflow, approval, and Slack status responses."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Protocol
from uuid import UUID

from app.schemas.agent_query import AgentQueryPlan, AgentQueryResponse
from app.schemas.risk import ReleaseRunRiskResponse

logger = logging.getLogger(__name__)


class ApprovalStatusRecordProtocol(Protocol):
    """Approval fields required to compose an approval-status response."""

    @property
    def id(self) -> UUID:
        """Return the durable approval identifier."""
        ...

    @property
    def approval_status(self) -> str:
        """Return the durable approval status."""
        ...

    @property
    def approval_reason(self) -> str:
        """Return the reason approval was required."""
        ...

    @property
    def approval_policy_version(self) -> str:
        """Return the approval policy version."""
        ...

    @property
    def requested_by(self) -> str | None:
        """Return who requested approval."""
        ...

    @property
    def decided_by(self) -> str | None:
        """Return who made the approval decision."""
        ...

    @property
    def decision_note(self) -> str | None:
        """Return the optional approval decision note."""
        ...

    @property
    def created_at(self) -> datetime:
        """Return when the approval request was created."""
        ...

    @property
    def decided_at(self) -> datetime | None:
        """Return when the approval decision was recorded."""
        ...


class SlackAlertStatusRecordProtocol(Protocol):
    """Slack delivery fields required to compose a status response."""

    @property
    def id(self) -> UUID:
        """Return the durable Slack alert identifier."""
        ...

    @property
    def delivery_status(self) -> str:
        """Return the Slack delivery status."""
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
        """Return the risk level included in the alert."""
        ...

    @property
    def risk_score(self) -> float:
        """Return the risk score included in the alert."""
        ...

    @property
    def recommended_action(self) -> str:
        """Return the recommended release action."""
        ...

    @property
    def created_at(self) -> datetime:
        """Return when the Slack alert was persisted."""
        ...


class AgentStatusResponseComposerMixin:
    """Compose deterministic persisted status responses."""

    _request_id: str

    def compose_approval_status(
        self,
        *,
        plan: AgentQueryPlan,
        release_risk: ReleaseRunRiskResponse,
        latest_approval: ApprovalStatusRecordProtocol | None,
    ) -> AgentQueryResponse:
        """Compose the latest durable HITL approval status.

        Args:
            plan: Validated approval-status query plan.
            release_risk: Trusted persisted release-risk snapshot.
            latest_approval: Latest durable approval record, when one exists.

        Returns:
            Approval-status response without evidence citations.
        """

        approval_reason: str | None
        approval_policy_version: str | None

        if latest_approval is not None:
            approval_status = latest_approval.approval_status
            approval_reason = latest_approval.approval_reason
            decided_by = latest_approval.decided_by
            decision_note = latest_approval.decision_note
            approval_policy_version = latest_approval.approval_policy_version
        else:
            approval_status = (
                release_risk.approval_status
                or (
                    "pending"
                    if release_risk.approval_required is True
                    else "not_required"
                )
            )
            approval_reason = release_risk.approval_reason
            decided_by = None
            decision_note = None
            approval_policy_version = release_risk.approval_policy_version

        readable_status = approval_status.replace("_", " ")
        answer = f"Approval status: {readable_status}."

        if decided_by:
            answer += f" Decided by: {decided_by}."

        if decision_note:
            answer += f" Decision note: {decision_note}"

            if not answer.endswith((".", "!", "?")):
                answer += "."

        if approval_reason:
            answer += f" Approval reason: {approval_reason}"

            if not answer.endswith((".", "!", "?")):
                answer += "."

        logger.info(
            "agent_approval_status_response_composed",
            extra={
                "run_id": self._request_id,
                "release_run_id": str(release_risk.release_run.id),
                "intent": plan.intent.value,
                "approval_status": approval_status,
                "approval_policy_version": approval_policy_version,
                "approval_record_found": latest_approval is not None,
                "decision_recorded": decided_by is not None,
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

    def compose_slack_status(
        self,
        *,
        plan: AgentQueryPlan,
        release_risk: ReleaseRunRiskResponse,
        slack_alert: SlackAlertStatusRecordProtocol | None,
    ) -> AgentQueryResponse:
        """Compose Slack delivery status from durable persisted state.

        Args:
            plan: Validated Slack-status query plan.
            release_risk: Trusted persisted release-risk snapshot.
            slack_alert: Persisted successful Slack delivery record, if present.

        Returns:
            Slack delivery response without evidence citations.
        """
        if slack_alert is None:
            answer = "No Slack alert has been sent for this release run."
            delivery_status = "not_sent"
        else:
            delivery_status = slack_alert.delivery_status
            readable_action = slack_alert.recommended_action.replace("_", " ")
            risk_percentage = round(slack_alert.risk_score * 100)

            answer = (
                f"Slack alert status: {delivery_status.replace('_', ' ')}. "
                f"Channel: {slack_alert.slack_channel}. "
                f"Risk level: {slack_alert.risk_level.replace('_', ' ')}. "
                f"Risk score: {risk_percentage}%. "
                f"Recommended action: {readable_action}."
            )

        logger.info(
            "agent_slack_status_response_composed",
            extra={
                "run_id": self._request_id,
                "release_run_id": str(release_risk.release_run.id),
                "intent": plan.intent.value,
                "delivery_status": delivery_status,
                "slack_alert_found": slack_alert is not None,
                "slack_channel": (
                    slack_alert.slack_channel
                    if slack_alert is not None
                    else None
                ),
                "citation_count": 0,
                "approval_required": release_risk.approval_required is True,
            },
        )

        return AgentQueryResponse(
            answer=answer,
            plan=plan,
            release_risk=release_risk,
            citations=[],
            approval_required=release_risk.approval_required is True,
        )

    def compose_workflow_status(
        self,
        *,
        plan: AgentQueryPlan,
        release_risk: ReleaseRunRiskResponse,
    ) -> AgentQueryResponse:
        """Compose workflow status from a trusted persisted snapshot.

        Args:
            plan: Validated workflow-status query plan.
            release_risk: Trusted persisted release-risk snapshot.

        Returns:
            Brief workflow-status response without evidence citations.
        """

        workflow_status_value = getattr(
            release_risk.release_run.status,
            "value",
            release_risk.release_run.status,
        )
        github_status_value = getattr(
            release_risk.github.status,
            "value",
            release_risk.github.status,
        )
        jira_status_value = getattr(
            release_risk.jira.status,
            "value",
            release_risk.jira.status,
        )

        workflow_status = str(workflow_status_value).replace("_", " ")
        github_status = str(github_status_value).replace("_", " ")
        jira_status = str(jira_status_value).replace("_", " ")
        knowledge_status = (
            release_risk.knowledge_status.replace("_", " ")
            if release_risk.knowledge_status
            else "not available"
        )
        approval_status = (
            release_risk.approval_status.replace("_", " ")
            if release_risk.approval_status
            else "not required"
        )

        answer = (
            f"Workflow status: {workflow_status}. "
            f"GitHub collection: {github_status}. "
            f"Jira collection: {jira_status}. "
            f"Knowledge retrieval: {knowledge_status}. "
            f"Approval status: {approval_status}."
        )

        logger.info(
            "agent_workflow_status_response_composed",
            extra={
                "run_id": self._request_id,
                "release_run_id": str(release_risk.release_run.id),
                "intent": plan.intent.value,
                "workflow_status": str(workflow_status_value),
                "github_status": str(github_status_value),
                "jira_status": str(jira_status_value),
                "knowledge_status": release_risk.knowledge_status,
                "approval_status": release_risk.approval_status,
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
