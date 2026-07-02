"""Tests for risk API schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from app.schemas.risk import (
    GitHubRiskCollectionResponse,
    PullRequestRiskResponse,
    ReleaseRunRiskResponse,
    ReleaseRunSummaryResponse,
    RiskCategoryResponse,
    RiskCollectionStatusResponse,
    RiskSeverityResponse,
    RiskSignalResponse,
)


def test_risk_signal_response_accepts_valid_payload() -> None:
    """RiskSignalResponse should validate a valid risk signal payload."""
    signal = RiskSignalResponse(
        source_type="github_pull_request",
        source_id="PR-42",
        source_url="https://github.com/acme/backend/pull/42",
        rule_id="github_ci_failure",
        category=RiskCategoryResponse.CI_FAILURE,
        severity=RiskSeverityResponse.HIGH,
        score=0.85,
        title="Pull request has failing CI",
        description="The pull request has a failing CI status.",
        evidence={"ci_status": "failure"},
    )

    assert signal.source_id == "PR-42"
    assert signal.category == RiskCategoryResponse.CI_FAILURE
    assert signal.severity == RiskSeverityResponse.HIGH
    assert signal.score == 0.85


def test_pull_request_risk_response_accepts_nested_signals() -> None:
    """PullRequestRiskResponse should validate nested risk signal responses."""
    signal = RiskSignalResponse(
        source_type="github_pull_request",
        source_id="PR-42",
        source_url=None,
        rule_id="github_review_missing",
        category=RiskCategoryResponse.REVIEW_MISSING,
        severity=RiskSeverityResponse.MEDIUM,
        score=0.5,
        title="Pull request is missing approval",
        description="The pull request does not have approval yet.",
        evidence={"review_state": "pending"},
    )

    result = PullRequestRiskResponse(
        source_type="github_pull_request",
        source_id="PR-42",
        source_url=None,
        pull_request_number=42,
        total_score=0.5,
        max_severity=RiskSeverityResponse.MEDIUM,
        signals=[signal],
        evaluated_at=datetime.now(UTC),
    )

    assert result.pull_request_number == 42
    assert result.total_score == 0.5
    assert len(result.signals) == 1


def test_release_run_risk_response_accepts_nested_github_result() -> None:
    """ReleaseRunRiskResponse should validate release run and GitHub risk data."""
    release_run = ReleaseRunSummaryResponse(
        id=uuid4(),
        run_id="release-run-test123",
        query="What are the biggest release risks this week?",
        requested_by="manager@example.com",
        status="completed",
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )

    github = GitHubRiskCollectionResponse(
        source="github",
        status=RiskCollectionStatusResponse.SUCCESS,
        pull_request_count=2,
        risk_result_count=2,
        total_signal_count=3,
        high_risk_count=1,
        risk_results=[],
        collected_at=datetime.now(UTC),
        duration_ms=12.5,
    )

    response = ReleaseRunRiskResponse(
        release_run=release_run,
        github=github,
    )

    assert response.release_run.status == "completed"
    assert response.github.status == RiskCollectionStatusResponse.SUCCESS
    assert response.github.high_risk_count == 1