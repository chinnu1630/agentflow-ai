"""Resolve GitHub pull requests from trusted persisted release-risk context."""

from __future__ import annotations

import logging

from app.schemas.agent_query import AgentQueryPlan
from app.schemas.risk import (
    PullRequestRiskResponse,
    ReleaseRunRiskResponse,
)

logger = logging.getLogger(__name__)


class AgentGitHubPRResolverError(RuntimeError):
    """Base error raised while resolving persisted GitHub PR context."""


class AgentGitHubPRNotFoundError(AgentGitHubPRResolverError):
    """Raised when the requested GitHub pull request cannot be resolved."""


class AgentGitHubPRResolver:
    """Resolve one GitHub PR from a validated persisted risk snapshot."""

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
    ) -> PullRequestRiskResponse:
        """Resolve one persisted GitHub pull-request result.

        Args:
            plan: Validated query plan containing extracted PR numbers.
            release_risk: Trusted persisted release-risk snapshot.

        Returns:
            Persisted GitHub risk result for the requested pull request.

        Raises:
            AgentGitHubPRNotFoundError: When no PR number was supplied or the
                requested PR does not exist in the persisted snapshot.
        """

        requested_numbers = plan.entity_references.pull_request_numbers

        if not requested_numbers:
            raise AgentGitHubPRNotFoundError(
                "No pull-request number was provided."
            )

        requested_number = requested_numbers[0]

        pull_request = next(
            (
                result
                for result in release_risk.github.risk_results
                if result.pull_request_number == requested_number
            ),
            None,
        )

        if pull_request is None:
            logger.warning(
                "agent_github_pr_not_found",
                extra={
                    "run_id": self._request_id,
                    "release_run_id": str(release_risk.release_run.id),
                    "pull_request_number": requested_number,
                    "persisted_pr_count": len(
                        release_risk.github.risk_results
                    ),
                    "intent": plan.intent.value,
                },
            )
            raise AgentGitHubPRNotFoundError(
                "No persisted GitHub pull request matched the query."
            )

        logger.info(
            "agent_github_pr_resolved",
            extra={
                "run_id": self._request_id,
                "release_run_id": str(release_risk.release_run.id),
                "pull_request_number": pull_request.pull_request_number,
                "source_id": pull_request.source_id,
                "signal_count": len(pull_request.signals),
                "intent": plan.intent.value,
            },
        )

        return pull_request
