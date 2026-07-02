"""Deterministic risk summary generation for AgentFlow AI.

This module aggregates collected risk signals into a manager-friendly summary.
It does not call an LLM. Claude synthesis will come later and should use this
summary as grounded evidence.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.services.github_risk_collector import (
    GitHubRiskCollectionResult,
    RiskCollectionStatus,
)
from app.services.github_risk_rules import RiskSeverity, RiskSignal

logger = logging.getLogger(__name__)


class RiskSummaryAction(StrEnum):
    """Recommended release action based on deterministic risk summary."""

    PROCEED = "proceed"
    REVIEW_REQUIRED = "review_required"
    BLOCK_RELEASE = "block_release"
    PARTIAL_DATA_REVIEW = "partial_data_review"


class RiskSummaryItem(BaseModel):
    """One prioritized risk item for manager review."""

    model_config = ConfigDict(frozen=True)

    source_type: Literal["github_pull_request"] = "github_pull_request"
    source_id: str = Field(min_length=1)
    source_url: str | None = None
    severity: RiskSeverity
    score: float = Field(ge=0.0, le=1.0)
    title: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    evidence: dict[str, str | int | float | bool] = Field(default_factory=dict)


class GitHubRiskSummary(BaseModel):
    """Manager-friendly GitHub risk summary for a release run."""

    model_config = ConfigDict(frozen=True)

    source: Literal["github"] = "github"
    collection_status: RiskCollectionStatus
    overall_severity: RiskSeverity
    recommended_action: RiskSummaryAction
    pull_request_count: int = Field(ge=0)
    risky_pull_request_count: int = Field(ge=0)
    total_signal_count: int = Field(ge=0)
    high_risk_count: int = Field(ge=0)
    top_risks: list[RiskSummaryItem] = Field(default_factory=list)
    summary_text: str = Field(min_length=1)
    generated_at: datetime


class RiskSummaryGenerator:
    """Generate deterministic summaries from collected risk results."""

    def summarize_github_risks(
        self,
        github_result: GitHubRiskCollectionResult,
        *,
        run_id: str,
        max_top_risks: int = 5,
    ) -> GitHubRiskSummary:
        """Summarize GitHub risk collection results.

        Args:
            github_result: GitHub risk collection result from RiskCollector.
            run_id: Workflow run ID used for structured logs.
            max_top_risks: Maximum number of prioritized risks to include.

        Returns:
            GitHubRiskSummary with overall severity and recommended action.
        """
        logger.info(
            "risk_summary.github.started",
            extra={
                "run_id": run_id,
                "collection_status": github_result.status.value,
                "pull_request_count": github_result.pull_request_count,
                "total_signal_count": github_result.total_signal_count,
            },
        )

        all_signals = self._extract_signals(github_result)
        top_risks = self._build_top_risks(
            all_signals=all_signals,
            max_top_risks=max_top_risks,
        )
        overall_severity = self._calculate_overall_severity(
            collection_status=github_result.status,
            signals=all_signals,
        )
        recommended_action = self._determine_recommended_action(
            collection_status=github_result.status,
            overall_severity=overall_severity,
        )

        summary = GitHubRiskSummary(
            collection_status=github_result.status,
            overall_severity=overall_severity,
            recommended_action=recommended_action,
            pull_request_count=github_result.pull_request_count,
            risky_pull_request_count=self._count_risky_pull_requests(github_result),
            total_signal_count=github_result.total_signal_count,
            high_risk_count=github_result.high_risk_count,
            top_risks=top_risks,
            summary_text=self._build_summary_text(
                github_result=github_result,
                overall_severity=overall_severity,
                recommended_action=recommended_action,
            ),
            generated_at=datetime.now(UTC),
        )

        logger.info(
            "risk_summary.github.completed",
            extra={
                "run_id": run_id,
                "overall_severity": summary.overall_severity.value,
                "recommended_action": summary.recommended_action.value,
                "top_risk_count": len(summary.top_risks),
            },
        )

        return summary

    def _extract_signals(
        self,
        github_result: GitHubRiskCollectionResult,
    ) -> list[RiskSignal]:
        """Extract all risk signals from GitHub pull request risk results."""
        return [
            signal
            for risk_result in github_result.risk_results
            for signal in risk_result.signals
        ]

    def _build_top_risks(
        self,
        *,
        all_signals: Sequence[RiskSignal],
        max_top_risks: int,
    ) -> list[RiskSummaryItem]:
        """Build prioritized top risk items from raw risk signals."""
        sorted_signals = sorted(
            all_signals,
            key=lambda signal: (
                self._severity_rank(signal.severity),
                signal.score,
            ),
            reverse=True,
        )

        return [
            RiskSummaryItem(
                source_id=signal.source_id,
                source_url=signal.source_url,
                severity=signal.severity,
                score=signal.score,
                title=signal.title,
                reason=signal.description,
                evidence=signal.evidence,
            )
            for signal in sorted_signals[:max_top_risks]
        ]

    def _calculate_overall_severity(
        self,
        *,
        collection_status: RiskCollectionStatus,
        signals: Sequence[RiskSignal],
    ) -> RiskSeverity:
        """Calculate overall GitHub severity from collection status and signals."""
        if collection_status == RiskCollectionStatus.DEGRADED:
            return RiskSeverity.MEDIUM

        if not signals:
            return RiskSeverity.LOW

        return max(
            (signal.severity for signal in signals),
            key=self._severity_rank,
        )

    def _determine_recommended_action(
        self,
        *,
        collection_status: RiskCollectionStatus,
        overall_severity: RiskSeverity,
    ) -> RiskSummaryAction:
        """Choose release recommendation based on overall severity."""
        if collection_status == RiskCollectionStatus.DEGRADED:
            return RiskSummaryAction.PARTIAL_DATA_REVIEW

        if overall_severity == RiskSeverity.CRITICAL:
            return RiskSummaryAction.BLOCK_RELEASE

        if overall_severity == RiskSeverity.HIGH:
            return RiskSummaryAction.REVIEW_REQUIRED

        if overall_severity == RiskSeverity.MEDIUM:
            return RiskSummaryAction.REVIEW_REQUIRED

        return RiskSummaryAction.PROCEED

    def _count_risky_pull_requests(
        self,
        github_result: GitHubRiskCollectionResult,
    ) -> int:
        """Count pull requests that have at least one risk signal."""
        return sum(
            1
            for risk_result in github_result.risk_results
            if len(risk_result.signals) > 0
        )

    def _build_summary_text(
        self,
        *,
        github_result: GitHubRiskCollectionResult,
        overall_severity: RiskSeverity,
        recommended_action: RiskSummaryAction,
    ) -> str:
        """Build deterministic summary text for manager review."""
        if github_result.status == RiskCollectionStatus.DEGRADED:
            return (
                "GitHub risk collection is degraded. Continue release review "
                "with partial data and verify GitHub manually before approval."
            )

        if github_result.total_signal_count == 0:
            return (
                f"GitHub analysis evaluated {github_result.pull_request_count} "
                "open pull requests and found no release-risk signals."
            )

        return (
            f"GitHub analysis evaluated {github_result.pull_request_count} "
            f"open pull requests and found {github_result.total_signal_count} "
            f"risk signals. Overall severity is {overall_severity.value}. "
            f"Recommended action is {recommended_action.value}."
        )

    def _severity_rank(self, severity: RiskSeverity) -> int:
        """Return sortable rank for severity."""
        severity_rank = {
            RiskSeverity.LOW: 1,
            RiskSeverity.MEDIUM: 2,
            RiskSeverity.HIGH: 3,
            RiskSeverity.CRITICAL: 4,
        }

        return severity_rank[severity]