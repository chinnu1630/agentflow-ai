"""Tests for release-risk feature extraction."""

from __future__ import annotations

import pytest

from app.services.risk_feature_extraction_service import (
    ReleaseRiskFeatureExtractionRequest,
    RiskFeatureExtractionService,
)


def _signal(
    *,
    category: str,
    severity: str,
    score: float,
    source_type: str = "github_pull_request",
) -> dict[str, object]:
    """Build a minimal risk signal test payload."""
    return {
        "source_type": source_type,
        "source_id": "SOURCE-1",
        "rule_id": f"rule_{category}",
        "category": category,
        "severity": severity,
        "score": score,
        "title": f"{category} risk",
        "description": f"{category} risk description",
        "evidence": {},
    }


def test_extract_features_counts_github_jira_knowledge_and_rule_scores() -> None:
    """Feature extraction should count signals, categories, severities, and scores."""
    service = RiskFeatureExtractionService()

    request = ReleaseRiskFeatureExtractionRequest(
        github={
            "status": "success",
            "risk_results": [
                {
                    "signals": [
                        _signal(
                            category="ci_failure",
                            severity="critical",
                            score=0.95,
                        ),
                        _signal(
                            category="review_blocked",
                            severity="high",
                            score=0.8,
                        ),
                    ]
                },
                {
                    "signals": [
                        _signal(
                            category="stale_pull_request",
                            severity="medium",
                            score=0.45,
                        )
                    ]
                },
            ],
        },
        jira={
            "status": "success",
            "signals": [
                _signal(
                    category="open_critical_bug",
                    severity="critical",
                    score=0.9,
                    source_type="jira_issue",
                ),
                _signal(
                    category="release_blocker_issue",
                    severity="high",
                    score=0.7,
                    source_type="jira_issue",
                ),
            ],
        },
        knowledge_status="completed",
        knowledge_results=[
            {
                "title": "Payment Runbook",
                "score": 42.0,
                "content": "Raw document content is not used as a feature.",
            }
        ],
    )

    features = service.extract_features(request, run_id="test-run-1")

    assert features.feature_version == "release_risk_features_v1"
    assert features.total_risk_count == 5
    assert features.github_risk_count == 3
    assert features.jira_risk_count == 2

    assert features.critical_risk_count == 2
    assert features.high_risk_count == 2
    assert features.medium_risk_count == 1
    assert features.low_risk_count == 0

    assert features.ci_failure_count == 1
    assert features.review_blocked_count == 1
    assert features.stale_pr_count == 1
    assert features.open_critical_bug_count == 1
    assert features.release_blocker_issue_count == 1

    assert features.knowledge_result_count == 1
    assert features.knowledge_no_results is False
    assert features.knowledge_failed is False

    assert features.github_degraded is False
    assert features.jira_degraded is False

    assert features.max_rule_score == 0.95
    assert features.average_rule_score == pytest.approx(0.76)


def test_extract_features_prefers_jira_top_level_signals_to_avoid_duplicates() -> None:
    """Jira extraction should not double-count top-level and issue-level signals."""
    service = RiskFeatureExtractionService()

    request = ReleaseRiskFeatureExtractionRequest(
        jira={
            "status": "success",
            "signals": [
                _signal(
                    category="blocked_jira_issue",
                    severity="high",
                    score=0.85,
                    source_type="jira_issue",
                )
            ],
            "issues": [
                {
                    "issue_key": "PAY-101",
                    "signals": [
                        _signal(
                            category="blocked_jira_issue",
                            severity="high",
                            score=0.85,
                            source_type="jira_issue",
                        )
                    ],
                }
            ],
        }
    )

    features = service.extract_features(request)

    assert features.total_risk_count == 1
    assert features.jira_risk_count == 1
    assert features.blocked_jira_issue_count == 1


def test_extract_features_marks_knowledge_no_results_without_failure() -> None:
    """No Knowledge results should be represented separately from failure."""
    service = RiskFeatureExtractionService()

    request = ReleaseRiskFeatureExtractionRequest(
        knowledge_status="no_results",
        knowledge_results=[],
    )

    features = service.extract_features(request)

    assert features.knowledge_result_count == 0
    assert features.knowledge_no_results is True
    assert features.knowledge_failed is False
    assert features.max_rule_score == 0.0
    assert features.average_rule_score == 0.0


def test_extract_features_marks_knowledge_failure_and_degraded_sources() -> None:
    """Recoverable dependency issues should become safe boolean features."""
    service = RiskFeatureExtractionService()

    request = ReleaseRiskFeatureExtractionRequest(
        github={"status": "degraded", "risk_results": []},
        jira={"status": "degraded", "signals": []},
        knowledge_status="failed",
        knowledge_error="retrieval failed",
    )

    features = service.extract_features(request)

    assert features.github_degraded is True
    assert features.jira_degraded is True
    assert features.knowledge_failed is True
    assert features.knowledge_no_results is False


def test_extract_from_payload_accepts_full_release_risk_payload() -> None:
    """The service should extract features from the full API/workflow payload."""
    service = RiskFeatureExtractionService()

    features = service.extract_from_payload(
        {
            "github": {
                "status": "success",
                "risk_results": [
                    {
                        "signals": [
                            _signal(
                                category="critical_file_change",
                                severity="high",
                                score=0.75,
                            )
                        ]
                    }
                ],
            },
            "jira": {
                "status": "success",
                "signals": [
                    _signal(
                        category="due_soon_issue",
                        severity="medium",
                        score=0.4,
                        source_type="jira_issue",
                    )
                ],
            },
            "knowledge_status": "completed",
            "knowledge_results": [{"title": "Release Checklist", "score": 5.0}],
        },
        run_id="test-run-2",
    )

    assert features.total_risk_count == 2
    assert features.critical_file_change_count == 1
    assert features.due_soon_issue_count == 1
    assert features.knowledge_result_count == 1
