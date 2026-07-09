"""Deterministic rule-based release-risk scoring for AgentFlow AI.

This module converts the stable ReleaseRiskFeatureVector into an explainable
release-risk score. It is intentionally deterministic because the project does
not yet have real historical release outcome labels for production ML training.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field

from app.services.risk_feature_extraction_service import ReleaseRiskFeatureVector

logger = structlog.get_logger(__name__)


class ReleaseRiskLevel(StrEnum):
    """Deterministic release-risk level."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReleaseRiskRecommendedAction(StrEnum):
    """Recommended release action from deterministic scoring."""

    PROCEED = "proceed"
    REVIEW_REQUIRED = "review_required"
    BLOCK_RELEASE = "block_release"
    PARTIAL_DATA_REVIEW = "partial_data_review"


class ReleaseRiskScore(BaseModel):
    """Explainable deterministic release-risk score."""

    model_config = ConfigDict(frozen=True)

    scoring_version: Literal["rule_based_release_risk_v1"] = (
        "rule_based_release_risk_v1"
    )
    feature_version: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    score: float = Field(ge=0.0, le=1.0)
    risk_level: ReleaseRiskLevel
    recommended_action: ReleaseRiskRecommendedAction
    reasons: list[str] = Field(default_factory=list)
    component_scores: dict[str, float] = Field(default_factory=dict)


class RuleBasedRiskScoringService:
    """Score release risk using explainable deterministic rules."""

    def score_release(
        self,
        features: ReleaseRiskFeatureVector,
        *,
        run_id: str | None = None,
    ) -> ReleaseRiskScore:
        """Create an explainable release-risk score from extracted features.

        Args:
            features: Stable numeric release-risk feature vector.
            run_id: Optional workflow run identifier used only in safe logs.

        Returns:
            Explainable deterministic release-risk score.

        This method is synchronous because it performs in-memory CPU work only.
        It does not call external APIs, databases, LLMs, or network services.
        """
        started_at = time.perf_counter()

        component_scores = self._calculate_component_scores(features)
        raw_score = min(1.0, round(sum(component_scores.values()), 4))

        risk_level = self._determine_risk_level(features, raw_score)
        adjusted_score = self._apply_risk_level_floor(raw_score, risk_level)
        recommended_action = self._determine_recommended_action(
            features=features,
            risk_level=risk_level,
        )
        reasons = self._build_reasons(
            features=features,
            risk_level=risk_level,
            recommended_action=recommended_action,
        )

        risk_score = ReleaseRiskScore(
            feature_version=features.feature_version,
            score=adjusted_score,
            risk_level=risk_level,
            recommended_action=recommended_action,
            reasons=reasons,
            component_scores=component_scores,
        )

        logger.info(
            "release_risk_scored",
            run_id=run_id,
            scoring_version=risk_score.scoring_version,
            feature_version=risk_score.feature_version,
            score=risk_score.score,
            risk_level=risk_score.risk_level,
            recommended_action=risk_score.recommended_action,
            reason_count=len(risk_score.reasons),
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
        )

        return risk_score

    def _calculate_component_scores(
        self,
        features: ReleaseRiskFeatureVector,
    ) -> dict[str, float]:
        """Calculate bounded score components from feature groups."""
        rule_score_component = round(features.max_rule_score * 0.45, 4)

        severity_pressure = min(
            0.30,
            (features.critical_risk_count * 0.12)
            + (features.high_risk_count * 0.08)
            + (features.medium_risk_count * 0.03)
            + (features.low_risk_count * 0.01),
        )

        category_pressure = min(
            0.20,
            (features.ci_failure_count * 0.10)
            + (features.open_critical_bug_count * 0.10)
            + (features.release_blocker_issue_count * 0.10)
            + (features.blocked_jira_issue_count * 0.08)
            + (features.critical_file_change_count * 0.07)
            + (features.review_blocked_count * 0.05),
        )

        data_quality_pressure = min(
            0.10,
            (0.04 if features.github_degraded else 0.0)
            + (0.04 if features.jira_degraded else 0.0)
            + (0.04 if features.knowledge_failed else 0.0)
            + (0.02 if features.knowledge_no_results else 0.0),
        )

        return {
            "rule_score_component": round(rule_score_component, 4),
            "severity_pressure": round(severity_pressure, 4),
            "category_pressure": round(category_pressure, 4),
            "data_quality_pressure": round(data_quality_pressure, 4),
        }

    def _determine_risk_level(
        self,
        features: ReleaseRiskFeatureVector,
        score: float,
    ) -> ReleaseRiskLevel:
        """Determine risk level using score thresholds and safety overrides."""
        if (
            score >= 0.85
            or features.critical_risk_count >= 2
            or (
                features.ci_failure_count > 0
                and (
                    features.open_critical_bug_count > 0
                    or features.release_blocker_issue_count > 0
                )
            )
        ):
            return ReleaseRiskLevel.CRITICAL

        if (
            score >= 0.65
            or features.critical_risk_count >= 1
            or features.high_risk_count >= 3
            or features.release_blocker_issue_count > 0
        ):
            return ReleaseRiskLevel.HIGH

        if score >= 0.35 or features.total_risk_count > 0:
            return ReleaseRiskLevel.MEDIUM

        return ReleaseRiskLevel.LOW

    def _apply_risk_level_floor(
        self,
        score: float,
        risk_level: ReleaseRiskLevel,
    ) -> float:
        """Keep numeric score aligned with safety override risk levels."""
        if risk_level == ReleaseRiskLevel.CRITICAL:
            return max(score, 0.85)

        if risk_level == ReleaseRiskLevel.HIGH:
            return max(score, 0.65)

        if risk_level == ReleaseRiskLevel.MEDIUM:
            return max(score, 0.35)

        return score

    def _determine_recommended_action(
        self,
        *,
        features: ReleaseRiskFeatureVector,
        risk_level: ReleaseRiskLevel,
    ) -> ReleaseRiskRecommendedAction:
        """Determine release action from risk level and data quality."""
        if risk_level == ReleaseRiskLevel.CRITICAL:
            return ReleaseRiskRecommendedAction.BLOCK_RELEASE

        if features.github_degraded or features.jira_degraded or features.knowledge_failed:
            return ReleaseRiskRecommendedAction.PARTIAL_DATA_REVIEW

        if risk_level in {ReleaseRiskLevel.HIGH, ReleaseRiskLevel.MEDIUM}:
            return ReleaseRiskRecommendedAction.REVIEW_REQUIRED

        return ReleaseRiskRecommendedAction.PROCEED

    def _build_reasons(
        self,
        *,
        features: ReleaseRiskFeatureVector,
        risk_level: ReleaseRiskLevel,
        recommended_action: ReleaseRiskRecommendedAction,
    ) -> list[str]:
        """Build safe human-readable scoring reasons without raw source content."""
        reasons: list[str] = [
            f"Release scored as {risk_level.value} with action {recommended_action.value}."
        ]

        if features.critical_risk_count > 0:
            reasons.append(
                f"Detected {features.critical_risk_count} critical risk signal(s)."
            )

        if features.high_risk_count > 0:
            reasons.append(f"Detected {features.high_risk_count} high risk signal(s).")

        if features.ci_failure_count > 0:
            reasons.append("Detected GitHub CI failure signal(s).")

        if features.review_blocked_count > 0:
            reasons.append("Detected blocked GitHub review signal(s).")

        if features.open_critical_bug_count > 0:
            reasons.append("Detected open critical Jira bug signal(s).")

        if features.release_blocker_issue_count > 0:
            reasons.append("Detected Jira release blocker signal(s).")

        if features.blocked_jira_issue_count > 0:
            reasons.append("Detected blocked Jira issue signal(s).")

        if features.critical_file_change_count > 0:
            reasons.append("Detected critical file change signal(s).")

        if features.github_degraded:
            reasons.append("GitHub collection was degraded.")

        if features.jira_degraded:
            reasons.append("Jira collection was degraded.")

        if features.knowledge_failed:
            reasons.append("Knowledge retrieval failed; score used available signals only.")
        elif features.knowledge_no_results:
            reasons.append("Knowledge retrieval completed but returned no matching context.")
        elif features.knowledge_result_count > 0:
            reasons.append(
                f"Knowledge retrieval returned {features.knowledge_result_count} context result(s)."
            )

        if features.total_risk_count == 0:
            reasons.append("No GitHub or Jira risk signals were detected.")

        return reasons
