"""Resolve Jira tickets from trusted persisted release-risk context."""

from __future__ import annotations

import logging

from app.schemas.agent_query import AgentQueryPlan
from app.schemas.risk import (
    JiraIssueRiskResponse,
    ReleaseRunRiskResponse,
)

logger = logging.getLogger(__name__)


class AgentJiraTicketResolverError(RuntimeError):
    """Base error raised while resolving persisted Jira ticket context."""


class AgentJiraTicketNotFoundError(AgentJiraTicketResolverError):
    """Raised when the requested Jira ticket cannot be resolved."""


class AgentJiraTicketResolver:
    """Resolve one Jira issue from a validated persisted risk snapshot."""

    def __init__(self, request_id: str) -> None:
        """Initialize the resolver.

        Args:
            request_id: Request identifier included in structured logs.
        """

        self._request_id = request_id

    def resolve(
        self,
        *,
        plan: AgentQueryPlan,
        release_risk: ReleaseRunRiskResponse,
    ) -> JiraIssueRiskResponse:
        """Resolve one persisted Jira issue.

        Args:
            plan: Validated query plan containing extracted Jira issue keys.
            release_risk: Trusted persisted release-risk snapshot.

        Returns:
            Persisted Jira risk result for the requested issue.

        Raises:
            AgentJiraTicketNotFoundError: When no issue key was supplied or the
                requested issue does not exist in the persisted snapshot.
        """

        requested_keys = plan.entity_references.jira_issue_keys

        if not requested_keys:
            raise AgentJiraTicketNotFoundError(
                "No Jira issue key was provided."
            )

        requested_key = requested_keys[0].upper()

        issue = next(
            (
                persisted_issue
                for persisted_issue in release_risk.jira.issues
                if persisted_issue.issue_key.upper() == requested_key
            ),
            None,
        )

        if issue is None:
            logger.warning(
                "agent_jira_ticket_not_found",
                extra={
                    "run_id": self._request_id,
                    "release_run_id": str(release_risk.release_run.id),
                    "jira_issue_key": requested_key,
                    "persisted_issue_count": len(
                        release_risk.jira.issues
                    ),
                    "intent": plan.intent.value,
                },
            )
            raise AgentJiraTicketNotFoundError(
                "No persisted Jira issue matched the query."
            )

        logger.info(
            "agent_jira_ticket_resolved",
            extra={
                "run_id": self._request_id,
                "release_run_id": str(release_risk.release_run.id),
                "jira_issue_key": issue.issue_key,
                "signal_count": len(issue.signals),
                "intent": plan.intent.value,
            },
        )

        return issue
