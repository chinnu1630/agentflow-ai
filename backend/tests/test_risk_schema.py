"""Tests for risk API schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from app.schemas.risk import (
    GitHubRiskCollectionResponse,
    GitHubRiskSummaryResponse,
    PullRequestRiskResponse,
    ReleaseRunRiskResponse,
    ReleaseRunSummaryResponse,
    RiskCategoryResponse,
    RiskCollectionStatusResponse,
    RiskSeverityResponse,
    RiskSignalResponse,
    RiskSummaryActionResponse,
    RiskSummaryItemResponse,
)


def test_risk_signal_response_accepts_valid_payload() -> None:
    """RiskSignalResponse should validate a valid GitHub risk signal payload."""
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

    assert signal.source_type == "github_pull_request"
    assert signal.source_id == "PR-42"
    assert signal.category == RiskCategoryResponse.CI_FAILURE
    assert signal.severity == RiskSeverityResponse.HIGH
    assert signal.score == 0.85


def test_risk_signal_response_accepts_jira_issue_payload() -> None:
    """RiskSignalResponse should validate a Jira issue risk signal payload."""
    signal = RiskSignalResponse(
        source_type="jira_issue",
        source_id="PAY-102",
        source_url="https://jira.example.com/browse/PAY-102",
        rule_id="jira_open_critical_bug",
        category=RiskCategoryResponse.OPEN_CRITICAL_BUG,
        severity=RiskSeverityResponse.HIGH,
        score=0.9,
        title="Open high-priority Jira bug may block release",
        description="A P1 Jira bug is still open during release validation.",
        evidence={
            "issue_key": "PAY-102",
            "priority": "p1",
            "status": "in_progress",
        },
    )

    assert signal.source_type == "jira_issue"
    assert signal.source_id == "PAY-102"
    assert signal.category == RiskCategoryResponse.OPEN_CRITICAL_BUG
    assert signal.severity == RiskSeverityResponse.HIGH
    assert signal.score == 0.9


def test_risk_summary_item_response_accepts_jira_issue_source() -> None:
    """RiskSummaryItemResponse should validate Jira issue summary items."""
    item = RiskSummaryItemResponse(
        source_type="jira_issue",
        source_id="PAY-103",
        source_url="https://jira.example.com/browse/PAY-103",
        severity=RiskSeverityResponse.CRITICAL,
        score=0.95,
        title="Release blocker Jira issue is unresolved",
        reason="The issue is explicitly marked as release blocking.",
        evidence={
            "issue_key": "PAY-103",
            "is_blocking_release": True,
        },
    )

    assert item.source_type == "jira_issue"
    assert item.source_id == "PAY-103"
    assert item.severity == RiskSeverityResponse.CRITICAL
    assert item.score == 0.95


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
    """ReleaseRunRiskResponse should validate release run, GitHub risk data, and summary."""
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

    github_summary = GitHubRiskSummaryResponse(
        source="github",
        collection_status=RiskCollectionStatusResponse.SUCCESS,
        overall_severity=RiskSeverityResponse.HIGH,
        recommended_action=RiskSummaryActionResponse.REVIEW_REQUIRED,
        pull_request_count=2,
        risky_pull_request_count=1,
        total_signal_count=3,
        high_risk_count=1,
        top_risks=[],
        summary_text="GitHub analysis found 3 risk signals.",
        generated_at=datetime.now(UTC),
    )

    response = ReleaseRunRiskResponse(
        release_run=release_run,
        github=github,
        github_summary=github_summary,
    )

    assert response.release_run.status == "completed"
    assert response.github.status == RiskCollectionStatusResponse.SUCCESS
    assert response.github.high_risk_count == 1
    assert response.github_summary.source == "github"
    assert response.github_summary.collection_status == RiskCollectionStatusResponse.SUCCESS
    assert response.github_summary.overall_severity == RiskSeverityResponse.HIGH
    assert (
        response.github_summary.recommended_action
        == RiskSummaryActionResponse.REVIEW_REQUIRED
    )
    assert response.github_summary.summary_text