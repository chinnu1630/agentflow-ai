"""Rule-based Jira issue risk detection for AgentFlow AI.

This module converts normalized JiraIssue objects into deterministic,
explainable release-risk signals. These rules are intentionally simple,
auditable, and testable before we add ML scoring or LLM synthesis.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.jira import (
    JiraIssue,
    JiraIssuePriority,
    JiraIssueStatus,
    JiraIssueType,
)

logger = logging.getLogger(__name__)

EvidenceValue = str | int | float | bool


class JiraRiskSeverity(StrEnum):
    """Severity level for a detected Jira release risk."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class JiraRiskCategory(StrEnum):
    """Business category for a detected Jira risk signal."""

    OPEN_CRITICAL_BUG = "open_critical_bug"
    BLOCKED_JIRA_ISSUE = "blocked_jira_issue"
    RELEASE_BLOCKER_ISSUE = "release_blocker_issue"
    UNASSIGNED_HIGH_PRIORITY_ISSUE = "unassigned_high_priority_issue"
    DUE_SOON_ISSUE = "due_soon_issue"
    CRITICAL_SERVICE_ISSUE = "critical_service_issue"


class JiraRiskRuleEngineConfig(BaseModel):
    """Configuration thresholds for Jira issue risk rules."""

    model_config = ConfigDict(frozen=True)

    due_soon_days: int = Field(default=2, ge=1)
    critical_services: tuple[str, ...] = (
        "payment-service",
        "checkout-api",
        "auth-service",
        "billing-service",
    )


class JiraRiskSignal(BaseModel):
    """Single explainable risk signal generated from a Jira issue."""

    model_config = ConfigDict(frozen=True)

    source_type: Literal["jira_issue"] = "jira_issue"
    source_id: str = Field(min_length=1)
    source_url: str | None = None
    rule_id: str = Field(min_length=1)
    category: JiraRiskCategory
    severity: JiraRiskSeverity
    score: float = Field(ge=0.0, le=1.0)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    evidence: dict[str, EvidenceValue] = Field(default_factory=dict)


class JiraIssueRiskResult(BaseModel):
    """Risk evaluation result for one Jira issue."""

    model_config = ConfigDict(frozen=True)

    source_type: Literal["jira_issue"] = "jira_issue"
    source_id: str = Field(min_length=1)
    source_url: str | None = None
    issue_key: str = Field(min_length=1)
    total_score: float = Field(ge=0.0, le=1.0)
    max_severity: JiraRiskSeverity | None = None
    signals: list[JiraRiskSignal] = Field(default_factory=list)
    evaluated_at: datetime


class JiraRiskRuleEngine:
    """Evaluate Jira issues using deterministic release-risk rules."""

    def __init__(self, config: JiraRiskRuleEngineConfig | None = None) -> None:
        """Initialize the Jira rule engine with validated configuration."""
        self._config = config or JiraRiskRuleEngineConfig()
        self._critical_services = {
            service.lower() for service in self._config.critical_services
        }

    def evaluate_issues(
        self,
        issues: Sequence[JiraIssue],
        *,
        run_id: str,
        evaluated_at: datetime | None = None,
    ) -> list[JiraIssueRiskResult]:
        """Evaluate many Jira issues and return risk results in input order."""
        current_time = evaluated_at or datetime.now(UTC)

        logger.info(
            "jira_risk_rules.evaluate_issues.started",
            extra={
                "run_id": run_id,
                "issue_count": len(issues),
            },
        )

        results = [
            self.evaluate_issue(
                issue,
                run_id=run_id,
                evaluated_at=current_time,
            )
            for issue in issues
        ]

        logger.info(
            "jira_risk_rules.evaluate_issues.completed",
            extra={
                "run_id": run_id,
                "issue_count": len(results),
                "risk_signal_count": sum(len(result.signals) for result in results),
            },
        )

        return results

    def evaluate_issue(
        self,
        issue: JiraIssue,
        *,
        run_id: str,
        evaluated_at: datetime | None = None,
    ) -> JiraIssueRiskResult:
        """Evaluate one Jira issue and return explainable risk signals."""
        current_time = evaluated_at or datetime.now(UTC)

        logger.info(
            "jira_risk_rules.evaluate_issue.started",
            extra={
                "run_id": run_id,
                "issue_key": issue.issue_key,
                "priority": issue.priority.value,
                "status": issue.status.value,
            },
        )

        signals = [
            signal
            for signal in (
                self._evaluate_open_critical_bug(issue),
                self._evaluate_blocked_issue(issue),
                self._evaluate_release_blocker(issue),
                self._evaluate_unassigned_high_priority_issue(issue),
                self._evaluate_due_soon_issue(issue, evaluated_at=current_time),
                self._evaluate_critical_service_issue(issue),
            )
            if signal is not None
        ]

        result = JiraIssueRiskResult(
            source_id=issue.issue_key,
            source_url=issue.issue_url,
            issue_key=issue.issue_key,
            total_score=self._calculate_total_score(signals),
            max_severity=self._get_max_severity(signals),
            signals=signals,
            evaluated_at=current_time,
        )

        logger.info(
            "jira_risk_rules.evaluate_issue.completed",
            extra={
                "run_id": run_id,
                "issue_key": issue.issue_key,
                "risk_signal_count": len(signals),
                "total_score": result.total_score,
                "max_severity": result.max_severity.value
                if result.max_severity is not None
                else None,
            },
        )

        return result

    def _evaluate_open_critical_bug(self, issue: JiraIssue) -> JiraRiskSignal | None:
        """Detect open P0/P1 bugs or incidents."""
        if issue.status == JiraIssueStatus.DONE:
            return None

        if issue.issue_type not in {JiraIssueType.BUG, JiraIssueType.INCIDENT}:
            return None

        if issue.priority not in {JiraIssuePriority.P0, JiraIssuePriority.P1}:
            return None

        severity = (
            JiraRiskSeverity.CRITICAL
            if issue.priority == JiraIssuePriority.P0
            else JiraRiskSeverity.HIGH
        )
        score = 0.95 if issue.priority == JiraIssuePriority.P0 else 0.85

        return JiraRiskSignal(
            source_id=issue.issue_key,
            source_url=issue.issue_url,
            rule_id="jira_open_critical_bug",
            category=JiraRiskCategory.OPEN_CRITICAL_BUG,
            severity=severity,
            score=score,
            title="Open high-priority Jira bug may block release",
            description=(
                "A high-priority Jira bug or incident is still open during "
                "release validation."
            ),
            evidence={
                "issue_key": issue.issue_key,
                "issue_type": issue.issue_type.value,
                "priority": issue.priority.value,
                "status": issue.status.value,
            },
        )

    def _evaluate_blocked_issue(self, issue: JiraIssue) -> JiraRiskSignal | None:
        """Detect Jira issues currently marked blocked."""
        if issue.status != JiraIssueStatus.BLOCKED:
            return None

        return JiraRiskSignal(
            source_id=issue.issue_key,
            source_url=issue.issue_url,
            rule_id="jira_blocked_issue",
            category=JiraRiskCategory.BLOCKED_JIRA_ISSUE,
            severity=JiraRiskSeverity.HIGH,
            score=0.8,
            title="Jira issue is blocked",
            description=(
                "A Jira issue is blocked, which may delay release readiness "
                "or leave unresolved dependency risk."
            ),
            evidence={
                "issue_key": issue.issue_key,
                "status": issue.status.value,
                "priority": issue.priority.value,
            },
        )

    def _evaluate_release_blocker(self, issue: JiraIssue) -> JiraRiskSignal | None:
        """Detect issues explicitly marked as release blocking."""
        if not issue.is_blocking_release:
            return None

        return JiraRiskSignal(
            source_id=issue.issue_key,
            source_url=issue.issue_url,
            rule_id="jira_release_blocker",
            category=JiraRiskCategory.RELEASE_BLOCKER_ISSUE,
            severity=JiraRiskSeverity.CRITICAL,
            score=0.95,
            title="Jira issue is marked as a release blocker",
            description=(
                "The issue is explicitly marked as blocking release and should "
                "be reviewed before deployment approval."
            ),
            evidence={
                "issue_key": issue.issue_key,
                "is_blocking_release": issue.is_blocking_release,
                "priority": issue.priority.value,
                "status": issue.status.value,
            },
        )

    def _evaluate_unassigned_high_priority_issue(
        self,
        issue: JiraIssue,
    ) -> JiraRiskSignal | None:
        """Detect high-priority Jira issues without an assignee."""
        if issue.status == JiraIssueStatus.DONE:
            return None

        if issue.priority not in {JiraIssuePriority.P0, JiraIssuePriority.P1}:
            return None

        if issue.assignee:
            return None

        return JiraRiskSignal(
            source_id=issue.issue_key,
            source_url=issue.issue_url,
            rule_id="jira_unassigned_high_priority_issue",
            category=JiraRiskCategory.UNASSIGNED_HIGH_PRIORITY_ISSUE,
            severity=JiraRiskSeverity.HIGH,
            score=0.75,
            title="High-priority Jira issue has no assignee",
            description=(
                "A high-priority Jira issue is unassigned, which increases "
                "ownership and resolution risk before release."
            ),
            evidence={
                "issue_key": issue.issue_key,
                "priority": issue.priority.value,
                "status": issue.status.value,
                "assignee_missing": True,
            },
        )

    def _evaluate_due_soon_issue(
        self,
        issue: JiraIssue,
        *,
        evaluated_at: datetime,
    ) -> JiraRiskSignal | None:
        """Detect open Jira issues due soon."""
        if issue.status == JiraIssueStatus.DONE or issue.due_at is None:
            return None

        due_threshold = evaluated_at + timedelta(days=self._config.due_soon_days)

        if issue.due_at > due_threshold:
            return None

        return JiraRiskSignal(
            source_id=issue.issue_key,
            source_url=issue.issue_url,
            rule_id="jira_due_soon_issue",
            category=JiraRiskCategory.DUE_SOON_ISSUE,
            severity=JiraRiskSeverity.MEDIUM,
            score=0.55,
            title="Jira issue is due soon",
            description=(
                "The Jira issue is due soon and may require attention before "
                "release validation completes."
            ),
            evidence={
                "issue_key": issue.issue_key,
                "due_at": issue.due_at.isoformat(),
                "due_soon_days": self._config.due_soon_days,
                "status": issue.status.value,
            },
        )

    def _evaluate_critical_service_issue(
        self,
        issue: JiraIssue,
    ) -> JiraRiskSignal | None:
        """Detect open issues affecting critical services."""
        if issue.status == JiraIssueStatus.DONE:
            return None

        affected_services = {service.lower() for service in issue.affected_services}
        matched_services = sorted(affected_services & self._critical_services)

        if not matched_services:
            return None

        return JiraRiskSignal(
            source_id=issue.issue_key,
            source_url=issue.issue_url,
            rule_id="jira_critical_service_issue",
            category=JiraRiskCategory.CRITICAL_SERVICE_ISSUE,
            severity=JiraRiskSeverity.HIGH,
            score=0.8,
            title="Jira issue affects a critical service",
            description=(
                "The Jira issue affects a critical service that is sensitive "
                "during release readiness checks."
            ),
            evidence={
                "issue_key": issue.issue_key,
                "matched_service": matched_services[0],
                "priority": issue.priority.value,
                "status": issue.status.value,
            },
        )

    @staticmethod
    def _calculate_total_score(signals: Sequence[JiraRiskSignal]) -> float:
        """Calculate bounded total score from generated Jira risk signals."""
        if not signals:
            return 0.0

        return min(sum(signal.score for signal in signals), 1.0)

    @staticmethod
    def _get_max_severity(
        signals: Sequence[JiraRiskSignal],
    ) -> JiraRiskSeverity | None:
        """Return the highest severity across generated Jira risk signals."""
        if not signals:
            return None

        severity_rank = {
            JiraRiskSeverity.LOW: 1,
            JiraRiskSeverity.MEDIUM: 2,
            JiraRiskSeverity.HIGH: 3,
            JiraRiskSeverity.CRITICAL: 4,
        }

        return max(signals, key=lambda signal: severity_rank[signal.severity]).severity