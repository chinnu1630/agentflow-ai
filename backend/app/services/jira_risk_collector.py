"""Jira risk collection service for AgentFlow AI.

This module connects the Jira integration client to the deterministic Jira
risk rule engine. It does not contain business rules itself; it only
orchestrates fetching issues, evaluating rules, and returning a collection
result.

Architecture position:
FastAPI route/service -> JiraRiskCollector -> JiraClient + JiraRiskRuleEngine
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from time import perf_counter
from typing import Protocol
from uuid import UUID

from app.core.logging import get_logger
from app.integrations.jira_client import JiraClient, JiraClientError
from app.schemas.jira import JiraIssue
from app.services.jira_risk_rules import JiraIssueRiskResult, JiraRiskRuleEngine
from app.services.github_risk_rules import RiskSignal

logger = get_logger(__name__)


class JiraRiskCollectionStatus(StrEnum):
    """Allowed statuses for Jira risk collection."""

    SUCCESS = "success"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class JiraRiskCollectionResult:
    """Result returned after collecting Jira risk signals.

    Attributes:
        status: Whether Jira risk collection succeeded or failed.
        issues: Jira issues that were analyzed.
        issue_results: Per-issue deterministic risk results.
        signals: Flattened risk signals across all Jira issues.
        error_message: Safe user-facing error message when collection fails.
        duration_ms: Collection duration in milliseconds.
    """

    status: JiraRiskCollectionStatus
    issues: list[JiraIssue] = field(default_factory=list)
    issue_results: list[JiraIssueRiskResult] = field(default_factory=list)
    signals: list[RiskSignal] = field(default_factory=list)
    error_message: str | None = None
    duration_ms: float = 0.0

    @property
    def total_issues_analyzed(self) -> int:
        """Return the number of Jira issues analyzed."""

        return len(self.issues)

    @property
    def total_signals(self) -> int:
        """Return the number of Jira risk signals found."""

        return len(self.signals)


class JiraIssueClient(Protocol):
    """Protocol for clients that can fetch open Jira issues.

    This keeps the collector testable because tests can pass a fake client
    without calling the real Jira API.
    """

    async def list_open_issues(self) -> list[JiraIssue]:
        """Fetch normalized open Jira issues."""

        ...


class JiraIssueRuleEngine(Protocol):
    """Protocol for Jira risk rule engines."""

    def evaluate_issue(
        self,
        issue: JiraIssue,
        *,
        run_id: UUID | str | None,
    ) -> JiraIssueRiskResult:
        """Evaluate one Jira issue and return deterministic risk results."""

        ...


class JiraRiskCollector:
    """Collect deterministic Jira risk signals for a release run.

    The collector is intentionally thin. Fetching belongs to JiraClient.
    Rule logic belongs to JiraRiskRuleEngine. This class only coordinates
    those pieces and returns a single typed result.
    """

    def __init__(
        self,
        jira_client: JiraIssueClient | None = None,
        rule_engine: JiraIssueRuleEngine | None = None,
    ) -> None:
        """Initialize the Jira risk collector.

        Args:
            jira_client: Client used to fetch normalized Jira issues.
            rule_engine: Rule engine used to evaluate each issue.
        """

        self._jira_client = jira_client or JiraClient()
        self._rule_engine = rule_engine or JiraRiskRuleEngine()

    async def collect(
        self,
        *,
        run_id: UUID | str | None = None,
    ) -> JiraRiskCollectionResult:
        """Collect Jira risk signals for the current release context.

        Args:
            run_id: Optional release run identifier for structured logs.

        Returns:
            JiraRiskCollectionResult containing issue-level results and
            flattened risk signals.

        Security:
            This method logs counts and status only. It does not log Jira issue
            descriptions or raw payloads because those may contain internal data.
        """

        started_at = perf_counter()
        safe_run_id = str(run_id) if run_id is not None else None

        logger.info(
            "jira_risk_collection_started",
            extra={"run_id": safe_run_id},
        )

        try:
            issues = await self._jira_client.list_open_issues()
        except JiraClientError as exc:
            duration_ms = self._elapsed_ms(started_at)

            logger.warning(
                "jira_risk_collection_failed",
                extra={
                    "run_id": safe_run_id,
                    "error_type": exc.__class__.__name__,
                    "duration_ms": duration_ms,
                },
            )

            return JiraRiskCollectionResult(
                status=JiraRiskCollectionStatus.FAILED,
                error_message=(
                    "Jira risk collection failed. Release analysis can "
                    "continue with other available sources."
                ),
                duration_ms=duration_ms,
            )

        issue_results = [
            self._rule_engine.evaluate_issue(issue, run_id=safe_run_id)
            for issue in issues
        ]
        signals = self._flatten_signals(issue_results)
        duration_ms = self._elapsed_ms(started_at)

        logger.info(
            "jira_risk_collection_completed",
            extra={
                "run_id": safe_run_id,
                "status": JiraRiskCollectionStatus.SUCCESS.value,
                "issues_analyzed": len(issues),
                "signals_found": len(signals),
                "duration_ms": duration_ms,
            },
        )

        return JiraRiskCollectionResult(
            status=JiraRiskCollectionStatus.SUCCESS,
            issues=issues,
            issue_results=issue_results,
            signals=signals,
            duration_ms=duration_ms,
        )

    @staticmethod
    def _flatten_signals(
        issue_results: list[JiraIssueRiskResult],
    ) -> list[RiskSignal]:
        """Flatten issue-level risk signals into one list.

        Args:
            issue_results: Per-issue risk results.

        Returns:
            A flat list of risk signals.
        """

        return [
            signal
            for issue_result in issue_results
            for signal in issue_result.signals
        ]

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        """Return elapsed time in milliseconds."""

        return round((perf_counter() - started_at) * 1000, 2)