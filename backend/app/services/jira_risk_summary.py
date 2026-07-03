"""Jira risk summary generation for AgentFlow AI.

This module turns deterministic Jira risk collection results into a
manager-friendly summary. It does not call Jira, does not use an LLM,
and does not mutate database state.

Architecture position:
JiraRiskCollector -> JiraRiskSummaryGenerator -> ReleaseRunService
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.services.github_risk_rules import RiskSeverity, RiskSignal
from app.services.jira_risk_collector import (
    JiraRiskCollectionResult,
    JiraRiskCollectionStatus,
)

logger = logging.getLogger(__name__)


class JiraRiskSummaryAction(StrEnum):
    """Recommended manager actions for Jira risk summaries."""

    PROCEED = "proceed"
    REVIEW_REQUIRED = "review_required"
    BLOCK_RELEASE = "block_release"
    PARTIAL_DATA_REVIEW = "partial_data_review"


class JiraRiskSummaryItem(BaseModel):
    """One manager-readable Jira risk item."""

    source_type: Literal["jira_issue"]
    source_id: str
    source_url: str | None = None
    severity: RiskSeverity
    score: float = Field(ge=0.0, le=1.0)
    title: str
    reason: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class JiraRiskSummary(BaseModel):
    """Manager-facing Jira release-risk summary."""

    source: Literal["jira"] = "jira"
    collection_status: JiraRiskCollectionStatus
    overall_severity: RiskSeverity
    recommended_action: JiraRiskSummaryAction
    issue_count: int = Field(ge=0)
    risky_issue_count: int = Field(ge=0)
    total_signal_count: int = Field(ge=0)
    high_risk_count: int = Field(ge=0)
    top_risks: list[JiraRiskSummaryItem] = Field(default_factory=list)
    summary_text: str
    generated_at: datetime


class JiraRiskSummaryGenerator:
    """Generate deterministic Jira risk summaries for managers.

    The summary generator is intentionally deterministic. LLMs will be used
    later for cross-source synthesis, but source-level risk summaries should
    stay predictable, testable, and explainable.
    """

    _SEVERITY_RANK: dict[RiskSeverity, int] = {
        RiskSeverity.LOW: 1,
        RiskSeverity.MEDIUM: 2,
        RiskSeverity.HIGH: 3,
        RiskSeverity.CRITICAL: 4,
    }

    _HIGH_RISK_SEVERITIES: frozenset[RiskSeverity] = frozenset(
        {
            RiskSeverity.HIGH,
            RiskSeverity.CRITICAL,
        }
    )

    def summarize_jira_risks(
        self,
        jira_result: JiraRiskCollectionResult,
        *,
        run_id: str,
    ) -> JiraRiskSummary:
        """Generate a manager-friendly summary from Jira risk collection.

        Args:
            jira_result: Jira risk collection result from JiraRiskCollector.
            run_id: Workflow run ID used for structured logs.

        Returns:
            Deterministic Jira risk summary.
        """

        logger.info(
            "jira_risk_summary_started",
            extra={
                "run_id": run_id,
                "collection_status": jira_result.status.value,
                "issue_count": jira_result.total_issues_analyzed,
                "total_signal_count": jira_result.total_signals,
            },
        )

        top_signals = self._select_top_signals(jira_result.signals)
        top_risks = [self._to_summary_item(signal) for signal in top_signals]

        overall_severity = self._calculate_overall_severity(jira_result.signals)
        high_risk_count = self._count_high_risk_signals(jira_result.signals)
        risky_issue_count = self._count_risky_issues(jira_result.signals)

        recommended_action = self._recommend_action(
            collection_status=jira_result.status,
            overall_severity=overall_severity,
            total_signal_count=jira_result.total_signals,
        )

        summary_text = self._build_summary_text(
            collection_status=jira_result.status,
            issue_count=jira_result.total_issues_analyzed,
            risky_issue_count=risky_issue_count,
            total_signal_count=jira_result.total_signals,
            high_risk_count=high_risk_count,
            recommended_action=recommended_action,
        )

        summary = JiraRiskSummary(
            collection_status=jira_result.status,
            overall_severity=overall_severity,
            recommended_action=recommended_action,
            issue_count=jira_result.total_issues_analyzed,
            risky_issue_count=risky_issue_count,
            total_signal_count=jira_result.total_signals,
            high_risk_count=high_risk_count,
            top_risks=top_risks,
            summary_text=summary_text,
            generated_at=datetime.now(UTC),
        )

        logger.info(
            "jira_risk_summary_completed",
            extra={
                "run_id": run_id,
                "overall_severity": summary.overall_severity.value,
                "recommended_action": summary.recommended_action.value,
                "risky_issue_count": summary.risky_issue_count,
                "high_risk_count": summary.high_risk_count,
            },
        )

        return summary

    def _select_top_signals(
        self,
        signals: list[RiskSignal],
        *,
        limit: int = 5,
    ) -> list[RiskSignal]:
        """Select the highest-risk Jira signals."""

        return sorted(
            signals,
            key=lambda signal: (
                self._SEVERITY_RANK[signal.severity],
                signal.score,
                signal.source_id,
            ),
            reverse=True,
        )[:limit]

    @staticmethod
    def _to_summary_item(signal: RiskSignal) -> JiraRiskSummaryItem:
        """Convert one risk signal into a manager summary item."""

        return JiraRiskSummaryItem(
            source_type="jira_issue",
            source_id=signal.source_id,
            source_url=signal.source_url,
            severity=signal.severity,
            score=signal.score,
            title=signal.title,
            reason=signal.description,
            evidence=signal.evidence,
        )

    def _calculate_overall_severity(self, signals: list[RiskSignal]) -> RiskSeverity:
        """Calculate the highest severity across Jira risk signals."""

        if not signals:
            return RiskSeverity.LOW

        return max(
            (signal.severity for signal in signals),
            key=lambda severity: self._SEVERITY_RANK[severity],
        )

    def _count_high_risk_signals(self, signals: list[RiskSignal]) -> int:
        """Count high and critical Jira risk signals."""

        return sum(
            1
            for signal in signals
            if signal.severity in self._HIGH_RISK_SEVERITIES
        )

    @staticmethod
    def _count_risky_issues(signals: list[RiskSignal]) -> int:
        """Count unique Jira issues that produced at least one signal."""

        return len({signal.source_id for signal in signals})

    @staticmethod
    def _recommend_action(
        *,
        collection_status: JiraRiskCollectionStatus,
        overall_severity: RiskSeverity,
        total_signal_count: int,
    ) -> JiraRiskSummaryAction:
        """Recommend a manager action from Jira risk severity."""

        if collection_status == JiraRiskCollectionStatus.FAILED:
            return JiraRiskSummaryAction.PARTIAL_DATA_REVIEW

        if total_signal_count == 0:
            return JiraRiskSummaryAction.PROCEED

        if overall_severity == RiskSeverity.CRITICAL:
            return JiraRiskSummaryAction.BLOCK_RELEASE

        return JiraRiskSummaryAction.REVIEW_REQUIRED

    @staticmethod
    def _build_summary_text(
        *,
        collection_status: JiraRiskCollectionStatus,
        issue_count: int,
        risky_issue_count: int,
        total_signal_count: int,
        high_risk_count: int,
        recommended_action: JiraRiskSummaryAction,
    ) -> str:
        """Build a concise manager-readable Jira summary."""

        if collection_status == JiraRiskCollectionStatus.FAILED:
            return (
                "Jira analysis was unavailable. Review GitHub and documentation "
                "results, then manually verify Jira before deployment."
            )

        if total_signal_count > 0:
            return (
                f"Jira analysis found {total_signal_count} risk signal(s) across "
                f"{risky_issue_count} issue(s), including {high_risk_count} high-risk "
                f"signal(s). Recommended action: {recommended_action.value}."
            )

        if issue_count == 0:
            return "Jira analysis found no open issues to review."

        return (
            f"Jira analysis reviewed {issue_count} issue(s) and found no "
            "release-risk signals."
        )