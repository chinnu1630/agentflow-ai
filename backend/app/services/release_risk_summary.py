"""Combined release risk summary generation for AgentFlow AI.

This module combines source-level GitHub and Jira summaries into one
manager-facing release-risk summary.

Architecture position:
GitHubRiskSummary + JiraRiskSummary -> ReleaseRiskSummaryGenerator
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from app.services.github_risk_rules import RiskSeverity
from app.services.github_risk_summary import GitHubRiskSummary
from app.services.jira_risk_summary import JiraRiskSummary

logger = logging.getLogger(__name__)


RiskItemSourceType = Literal["github_pull_request", "jira_issue"]
RiskSource = Literal["github", "jira"]


class ReleaseRiskSummaryAction(StrEnum):
    """Recommended manager action for the combined release summary."""

    PROCEED = "proceed"
    REVIEW_REQUIRED = "review_required"
    BLOCK_RELEASE = "block_release"
    PARTIAL_DATA_REVIEW = "partial_data_review"


class ReleaseRiskSummaryItem(BaseModel):
    """One combined release-risk item from GitHub or Jira."""

    source: RiskSource
    source_type: RiskItemSourceType
    source_id: str
    source_url: str | None = None
    severity: RiskSeverity
    score: float = Field(ge=0.0, le=1.0)
    title: str
    reason: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class ReleaseRiskSourceSummary(BaseModel):
    """Short source-level summary included in the release summary."""

    source: RiskSource
    overall_severity: RiskSeverity
    recommended_action: str
    total_signal_count: int = Field(ge=0)
    high_risk_count: int = Field(ge=0)
    summary_text: str


class ReleaseRiskSummary(BaseModel):
    """Manager-facing combined release-risk summary."""

    source: Literal["release"] = "release"
    overall_severity: RiskSeverity
    recommended_action: ReleaseRiskSummaryAction
    total_signal_count: int = Field(ge=0)
    high_risk_count: int = Field(ge=0)
    source_summary_count: int = Field(ge=0)
    top_risks: list[ReleaseRiskSummaryItem] = Field(default_factory=list)
    source_summaries: list[ReleaseRiskSourceSummary] = Field(default_factory=list)
    summary_text: str
    generated_at: datetime


class ReleaseRiskSummaryGenerator:
    """Generate deterministic release-level risk summaries.

    This generator combines GitHub and Jira summaries without using an LLM.
    LLM synthesis will come later after deterministic summaries are stable.
    """

    _SEVERITY_RANK: dict[RiskSeverity, int] = {
        RiskSeverity.LOW: 1,
        RiskSeverity.MEDIUM: 2,
        RiskSeverity.HIGH: 3,
        RiskSeverity.CRITICAL: 4,
    }

    def summarize_release_risks(
        self,
        *,
        github_summary: GitHubRiskSummary,
        jira_summary: JiraRiskSummary,
        run_id: str,
    ) -> ReleaseRiskSummary:
        """Generate one combined release-risk summary.

        Args:
            github_summary: Deterministic GitHub risk summary.
            jira_summary: Deterministic Jira risk summary.
            run_id: Workflow run ID used for structured logs.

        Returns:
            Combined release-risk summary for manager review.
        """

        logger.info(
            "release_risk_summary_started",
            extra={
                "run_id": run_id,
                "github_total_signal_count": github_summary.total_signal_count,
                "jira_total_signal_count": jira_summary.total_signal_count,
            },
        )

        top_risks = self._select_top_risks(
            [
                *self._to_release_risk_items(
                    source="github",
                    items=github_summary.top_risks,
                ),
                *self._to_release_risk_items(
                    source="jira",
                    items=jira_summary.top_risks,
                ),
            ]
        )

        overall_severity = self._calculate_overall_severity(
            [
                github_summary.overall_severity,
                jira_summary.overall_severity,
            ]
        )

        total_signal_count = (
            github_summary.total_signal_count + jira_summary.total_signal_count
        )
        high_risk_count = github_summary.high_risk_count + jira_summary.high_risk_count

        recommended_action = self._recommend_action(
            source_actions=[
                github_summary.recommended_action,
                jira_summary.recommended_action,
            ],
            total_signal_count=total_signal_count,
        )

        source_summaries = [
            self._to_source_summary(source="github", summary=github_summary),
            self._to_source_summary(source="jira", summary=jira_summary),
        ]

        summary_text = self._build_summary_text(
            recommended_action=recommended_action,
            total_signal_count=total_signal_count,
            high_risk_count=high_risk_count,
        )

        summary = ReleaseRiskSummary(
            overall_severity=overall_severity,
            recommended_action=recommended_action,
            total_signal_count=total_signal_count,
            high_risk_count=high_risk_count,
            source_summary_count=len(source_summaries),
            top_risks=top_risks,
            source_summaries=source_summaries,
            summary_text=summary_text,
            generated_at=datetime.now(UTC),
        )

        logger.info(
            "release_risk_summary_completed",
            extra={
                "run_id": run_id,
                "overall_severity": summary.overall_severity.value,
                "recommended_action": summary.recommended_action.value,
                "total_signal_count": summary.total_signal_count,
                "high_risk_count": summary.high_risk_count,
            },
        )

        return summary

    def _select_top_risks(
        self,
        risks: list[ReleaseRiskSummaryItem],
        *,
        limit: int = 10,
    ) -> list[ReleaseRiskSummaryItem]:
        """Select the highest-risk items across all sources."""

        return sorted(
            risks,
            key=lambda risk: (
                self._SEVERITY_RANK[risk.severity],
                risk.score,
                risk.source,
                risk.source_id,
            ),
            reverse=True,
        )[:limit]

    @staticmethod
    def _to_release_risk_items(
        *,
        source: RiskSource,
        items: list[Any],
    ) -> list[ReleaseRiskSummaryItem]:
        """Convert source-level summary items into release-level risk items."""

        release_items: list[ReleaseRiskSummaryItem] = []

        for item in items:
            source_type = item.source_type

            if source_type not in {"github_pull_request", "jira_issue"}:
                raise ValueError(f"Unsupported risk source_type: {source_type}")

            release_items.append(
                ReleaseRiskSummaryItem(
                    source=source,
                    source_type=cast(RiskItemSourceType, source_type),
                    source_id=item.source_id,
                    source_url=item.source_url,
                    severity=item.severity,
                    score=item.score,
                    title=item.title,
                    reason=item.reason,
                    evidence=item.evidence,
                )
            )

        return release_items

    def _calculate_overall_severity(
        self,
        severities: list[RiskSeverity],
    ) -> RiskSeverity:
        """Calculate highest severity across source summaries."""

        return max(
            severities,
            key=lambda severity: self._SEVERITY_RANK[severity],
        )

    @staticmethod
    def _recommend_action(
        *,
        source_actions: list[Any],
        total_signal_count: int,
    ) -> ReleaseRiskSummaryAction:
        """Recommend final release action from source-level actions."""

        action_values = {
            ReleaseRiskSummaryGenerator._enum_value(action)
            for action in source_actions
        }

        if ReleaseRiskSummaryAction.BLOCK_RELEASE.value in action_values:
            return ReleaseRiskSummaryAction.BLOCK_RELEASE

        if ReleaseRiskSummaryAction.PARTIAL_DATA_REVIEW.value in action_values:
            return ReleaseRiskSummaryAction.PARTIAL_DATA_REVIEW

        if (
            ReleaseRiskSummaryAction.REVIEW_REQUIRED.value in action_values
            or total_signal_count > 0
        ):
            return ReleaseRiskSummaryAction.REVIEW_REQUIRED

        return ReleaseRiskSummaryAction.PROCEED

    @staticmethod
    def _to_source_summary(
        *,
        source: RiskSource,
        summary: Any,
    ) -> ReleaseRiskSourceSummary:
        """Convert one source summary into a compact release source summary."""

        return ReleaseRiskSourceSummary(
            source=source,
            overall_severity=summary.overall_severity,
            recommended_action=ReleaseRiskSummaryGenerator._enum_value(
                summary.recommended_action
            ),
            total_signal_count=summary.total_signal_count,
            high_risk_count=summary.high_risk_count,
            summary_text=summary.summary_text,
        )

    @staticmethod
    def _build_summary_text(
        *,
        recommended_action: ReleaseRiskSummaryAction,
        total_signal_count: int,
        high_risk_count: int,
    ) -> str:
        """Build concise manager-readable release summary text."""

        if recommended_action == ReleaseRiskSummaryAction.BLOCK_RELEASE:
            return (
                f"Release should be blocked. Found {total_signal_count} total "
                f"risk signal(s), including {high_risk_count} high-risk signal(s)."
            )

        if recommended_action == ReleaseRiskSummaryAction.PARTIAL_DATA_REVIEW:
            return (
                "Release needs manual review because one or more data sources "
                "were unavailable or incomplete."
            )

        if recommended_action == ReleaseRiskSummaryAction.REVIEW_REQUIRED:
            return (
                f"Release requires manager review. Found {total_signal_count} total "
                f"risk signal(s), including {high_risk_count} high-risk signal(s)."
            )

        return "Release summary found no current GitHub or Jira release-risk signals."

    @staticmethod
    def _enum_value(value: Any) -> str:
        """Return enum value if value is an enum, otherwise return string form."""

        if hasattr(value, "value"):
            return str(value.value)

        return str(value)