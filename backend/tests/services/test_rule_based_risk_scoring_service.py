"""Tests for deterministic release-risk scoring."""

from __future__ import annotations

from typing import Any

from app.services.risk_feature_extraction_service import ReleaseRiskFeatureVector
from app.services.rule_based_risk_scoring_service import (
    ReleaseRiskLevel,
    ReleaseRiskRecommendedAction,
    RuleBasedRiskScoringService,
)


def _features(**overrides: Any) -> ReleaseRiskFeatureVector:
    """Build a feature vector with safe defaults for scoring tests."""
    values: dict[str, Any] = {
        "total_risk_count": 0,
        "github_risk_count": 0,
        "jira_risk_count": 0,
        "critical_risk_count": 0,
        "high_risk_count": 0,
        "medium_risk_count": 0,
        "low_risk_count": 0,
        "ci_failure_count": 0,
        "ci_pending_count": 0,
        "review_blocked_count": 0,
        "review_missing_count": 0,
        "stale_pr_count": 0,
        "large_changeset_count": 0,
        "draft_pull_request_count": 0,
        "missing_jira_link_count": 0,
        "critical_file_change_count": 0,
        "open_critical_bug_count": 0,
        "blocked_jira_issue_count": 0,
        "release_blocker_issue_count": 0,
        "unassigned_high_priority_issue_count": 0,
        "due_soon_issue_count": 0,
        "critical_service_issue_count": 0,
        "knowledge_result_count": 0,
        "knowledge_no_results": False,
        "knowledge_failed": False,
        "github_degraded": False,
        "jira_degraded": False,
        "max_rule_score": 0.0,
        "average_rule_score": 0.0,
    }
    values.update(overrides)
    return ReleaseRiskFeatureVector(**values)


def test_score_release_all_clear_proceeds() -> None:
    """A release with no detected risks should produce a low/proceed score."""
    service = RuleBasedRiskScoringService()

    result = service.score_release(_features(), run_id="test-run-1")

    assert result.scoring_version == "rule_based_release_risk_v1"
    assert result.risk_level == ReleaseRiskLevel.LOW
    assert result.recommended_action == ReleaseRiskRecommendedAction.PROCEED
    assert result.score == 0.0
    assert "No GitHub or Jira risk signals were detected." in result.reasons


def test_score_release_blocks_when_ci_failure_and_critical_jira_bug_exist() -> None:
    """Combined GitHub CI failure and critical Jira bug should block release."""
    service = RuleBasedRiskScoringService()

    result = service.score_release(
        _features(
            total_risk_count=2,
            github_risk_count=1,
            jira_risk_count=1,
            critical_risk_count=2,
            ci_failure_count=1,
            open_critical_bug_count=1,
            max_rule_score=0.95,
            average_rule_score=0.925,
        ),
        run_id="test-run-2",
    )

    assert result.risk_level == ReleaseRiskLevel.CRITICAL
    assert result.recommended_action == ReleaseRiskRecommendedAction.BLOCK_RELEASE
    assert result.score >= 0.85
    assert "Detected GitHub CI failure signal(s)." in result.reasons
    assert "Detected open critical Jira bug signal(s)." in result.reasons


def test_score_release_high_risk_requires_review() -> None:
    """A single critical signal should create a high risk review requirement."""
    service = RuleBasedRiskScoringService()

    result = service.score_release(
        _features(
            total_risk_count=1,
            github_risk_count=1,
            critical_risk_count=1,
            critical_file_change_count=1,
            max_rule_score=0.8,
            average_rule_score=0.8,
        )
    )

    assert result.risk_level == ReleaseRiskLevel.HIGH
    assert result.recommended_action == ReleaseRiskRecommendedAction.REVIEW_REQUIRED
    assert result.score >= 0.65
    assert "Detected critical file change signal(s)." in result.reasons


def test_score_release_degraded_data_requests_partial_data_review() -> None:
    """Dependency degradation should request partial review instead of blind proceed."""
    service = RuleBasedRiskScoringService()

    result = service.score_release(
        _features(
            github_degraded=True,
            jira_degraded=True,
            knowledge_failed=True,
        )
    )

    assert result.risk_level == ReleaseRiskLevel.LOW
    assert result.recommended_action == ReleaseRiskRecommendedAction.PARTIAL_DATA_REVIEW
    assert "GitHub collection was degraded." in result.reasons
    assert "Jira collection was degraded." in result.reasons
    assert "Knowledge retrieval failed; score used available signals only." in result.reasons


def test_score_release_knowledge_no_results_is_not_failure() -> None:
    """Knowledge no-results should be visible but should not be treated as failure."""
    service = RuleBasedRiskScoringService()

    result = service.score_release(
        _features(
            knowledge_no_results=True,
        )
    )

    assert result.risk_level == ReleaseRiskLevel.LOW
    assert result.recommended_action == ReleaseRiskRecommendedAction.PROCEED
    assert "Knowledge retrieval completed but returned no matching context." in result.reasons


def test_score_release_returns_safe_component_scores() -> None:
    """Scoring should expose component scores for audit/debugging."""
    service = RuleBasedRiskScoringService()

    result = service.score_release(
        _features(
            total_risk_count=2,
            high_risk_count=1,
            medium_risk_count=1,
            review_blocked_count=1,
            max_rule_score=0.7,
            average_rule_score=0.55,
        )
    )

    assert result.component_scores["rule_score_component"] == 0.315
    assert result.component_scores["severity_pressure"] > 0
    assert result.component_scores["category_pressure"] > 0
    assert result.component_scores["data_quality_pressure"] == 0.0
