"""Unit tests for deterministic similar-release matching."""

from __future__ import annotations

from uuid import uuid4

from app.schemas.risk import ReleaseRunRiskResponse
from app.services.agent_similar_release_matcher import (
    AgentSimilarReleaseMatcher,
)
from tests.services.test_slack_release_alert_service import (
    build_snapshot_payload,
)


def build_release_risk(
    *,
    run_id: str,
    feature_updates: dict[str, object],
) -> ReleaseRunRiskResponse:
    """Build a persisted release-risk response with selected feature values."""
    payload = build_snapshot_payload(
        release_run_id=uuid4(),
        approval_request_id=uuid4(),
    )
    payload["release_run"]["run_id"] = run_id
    payload["risk_features"] = {
        "feature_version": "release_risk_features_v1",
        "generated_at": payload["release_summary"]["generated_at"],
        "total_risk_count": 1,
        "github_risk_count": 1,
        "jira_risk_count": 0,
        "critical_risk_count": 0,
        "high_risk_count": 1,
        "medium_risk_count": 0,
        "low_risk_count": 0,
        "ci_failure_count": 1,
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
        "knowledge_no_results": True,
        "knowledge_failed": False,
        "github_degraded": False,
        "jira_degraded": False,
        "max_rule_score": 0.85,
        "average_rule_score": 0.85,
    }
    payload["risk_features"].update(feature_updates)

    return ReleaseRunRiskResponse.model_validate(payload)


def test_matches_release_with_smallest_feature_distance() -> None:
    """The closest persisted feature vector should be ranked first."""
    current = build_release_risk(
        run_id="release-run-current",
        feature_updates={
            "total_risk_count": 5,
            "github_risk_count": 2,
            "jira_risk_count": 3,
            "critical_risk_count": 1,
            "high_risk_count": 4,
            "ci_failure_count": 1,
            "blocked_jira_issue_count": 1,
            "release_blocker_issue_count": 1,
            "max_rule_score": 0.95,
            "average_rule_score": 0.76,
        },
    )
    dissimilar = build_release_risk(
        run_id="release-run-dissimilar",
        feature_updates={
            "total_risk_count": 1,
            "github_risk_count": 1,
            "jira_risk_count": 0,
            "critical_risk_count": 0,
            "high_risk_count": 0,
            "ci_failure_count": 0,
            "blocked_jira_issue_count": 0,
            "release_blocker_issue_count": 0,
            "max_rule_score": 0.20,
            "average_rule_score": 0.10,
        },
    )
    similar = build_release_risk(
        run_id="release-run-similar",
        feature_updates={
            "total_risk_count": 5,
            "github_risk_count": 2,
            "jira_risk_count": 3,
            "critical_risk_count": 1,
            "high_risk_count": 4,
            "ci_failure_count": 1,
            "blocked_jira_issue_count": 1,
            "release_blocker_issue_count": 1,
            "max_rule_score": 0.90,
            "average_rule_score": 0.72,
        },
    )

    matcher = AgentSimilarReleaseMatcher(request_id="request-123")

    match = matcher.match(
        current_release_risk=current,
        historical_release_risks=[dissimilar, similar],
    )

    assert match is not None
    assert match.release_risk.release_run.run_id == "release-run-similar"
    assert 0.0 <= match.similarity_score <= 1.0
    assert match.similarity_score > 0.9


def test_returns_none_when_no_historical_releases_exist() -> None:
    """Matching should degrade gracefully when no history exists."""
    matcher = AgentSimilarReleaseMatcher(request_id="request-123")
    current = build_release_risk(
        run_id="release-run-current",
        feature_updates={},
    )

    match = matcher.match(
        current_release_risk=current,
        historical_release_risks=[],
    )

    assert match is None
