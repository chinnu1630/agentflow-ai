"""Risk collection services for AgentFlow AI.

This module coordinates data collection from engineering systems and converts
raw integration data into structured release-risk results.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from enum import StrEnum
from time import perf_counter
from typing import Literal, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.integrations.github_client import GitHubClientError
from app.schemas.github import GitHubPullRequest
from app.services.github_risk_rules import (
    PullRequestRiskResult,
    RiskRuleEngine,
    RiskSeverity,
)

logger = logging.getLogger(__name__)


class RiskCollectionStatus(StrEnum):
    """Status of a risk collection operation."""

    SUCCESS = "success"
    DEGRADED = "degraded"


class GitHubPullRequestClient(Protocol):
    """Protocol for a client that can fetch open GitHub pull requests."""

    def list_open_pull_requests(self) -> list[GitHubPullRequest]:
        """Return open GitHub pull requests from a repository."""
        ...


class PullRequestRiskEvaluator(Protocol):
    """Protocol for a service that evaluates pull request risk."""

    def evaluate_pull_requests(
        self,
        pull_requests: Sequence[GitHubPullRequest],
        *,
        run_id: str,
    ) -> list[PullRequestRiskResult]:
        """Evaluate pull requests and return structured risk results."""
        ...


class GitHubRiskCollectionResult(BaseModel):
    """Collected GitHub risk result for one release-risk run."""

    model_config = ConfigDict(frozen=True)

    source: Literal["github"] = "github"
    status: RiskCollectionStatus
    pull_request_count: int = Field(ge=0)
    risk_result_count: int = Field(ge=0)
    total_signal_count: int = Field(ge=0)
    high_risk_count: int = Field(ge=0)
    risk_results: list[PullRequestRiskResult] = Field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None
    collected_at: datetime
    duration_ms: float = Field(ge=0.0)


class RiskCollector:
    """Collect release-risk signals from engineering systems."""

    def __init__(
        self,
        github_client: GitHubPullRequestClient,
        risk_rule_engine: PullRequestRiskEvaluator | None = None,
    ) -> None:
        """Initialize the collector with integration clients and evaluators."""
        self._github_client = github_client
        self._risk_rule_engine: PullRequestRiskEvaluator = (
            risk_rule_engine or RiskRuleEngine()
        )

    async def collect_github_risks(self, *, run_id: str) -> GitHubRiskCollectionResult:
        """Collect GitHub pull request risks for a release run.

        The GitHub client is currently synchronous, so this method runs it in a
        worker thread to avoid blocking the async FastAPI event loop.
        """
        started_at = perf_counter()

        logger.info(
            "risk_collector.collect_github_risks.started",
            extra={"run_id": run_id},
        )

        try:
            pull_requests = await asyncio.to_thread(
                self._github_client.list_open_pull_requests,
            )

            risk_results = self._risk_rule_engine.evaluate_pull_requests(
                pull_requests,
                run_id=run_id,
            )

            duration_ms = self._calculate_duration_ms(started_at)

            result = GitHubRiskCollectionResult(
                status=RiskCollectionStatus.SUCCESS,
                pull_request_count=len(pull_requests),
                risk_result_count=len(risk_results),
                total_signal_count=self._count_total_signals(risk_results),
                high_risk_count=self._count_high_risk_results(risk_results),
                risk_results=risk_results,
                collected_at=datetime.now(UTC),
                duration_ms=duration_ms,
            )

            logger.info(
                "risk_collector.collect_github_risks.completed",
                extra={
                    "run_id": run_id,
                    "status": result.status.value,
                    "pull_request_count": result.pull_request_count,
                    "risk_result_count": result.risk_result_count,
                    "total_signal_count": result.total_signal_count,
                    "high_risk_count": result.high_risk_count,
                    "duration_ms": result.duration_ms,
                },
            )

            return result

        except (GitHubClientError, TimeoutError, ConnectionError) as exc:
            duration_ms = self._calculate_duration_ms(started_at)

            logger.warning(
                "risk_collector.collect_github_risks.degraded",
                extra={
                    "run_id": run_id,
                    "status": RiskCollectionStatus.DEGRADED.value,
                    "error_type": exc.__class__.__name__,
                    "duration_ms": duration_ms,
                },
                exc_info=True,
            )

            return GitHubRiskCollectionResult(
                status=RiskCollectionStatus.DEGRADED,
                pull_request_count=0,
                risk_result_count=0,
                total_signal_count=0,
                high_risk_count=0,
                risk_results=[],
                error_type=exc.__class__.__name__,
                error_message=(
                    "GitHub risk collection failed. "
                    "The release-risk workflow should continue with partial data."
                ),
                collected_at=datetime.now(UTC),
                duration_ms=duration_ms,
            )

    def _count_total_signals(
        self,
        risk_results: Sequence[PullRequestRiskResult],
    ) -> int:
        """Count total risk signals across pull request risk results."""
        return sum(len(result.signals) for result in risk_results)

    def _count_high_risk_results(
        self,
        risk_results: Sequence[PullRequestRiskResult],
    ) -> int:
        """Count pull request results with high or critical max severity."""
        high_severities = {RiskSeverity.HIGH, RiskSeverity.CRITICAL}

        return sum(
            1
            for result in risk_results
            if result.max_severity in high_severities
        )

    def _calculate_duration_ms(self, started_at: float) -> float:
        """Calculate operation duration in milliseconds."""
        return round((perf_counter() - started_at) * 1000, 2)