"""Tests for Jira risk summary generation."""

from __future__ import annotations

from app.services.github_risk_rules import RiskCategory, RiskSeverity, RiskSignal
from app.services.jira_risk_collector import (
    JiraRiskCollectionResult,
    JiraRiskCollectionStatus,
)
from app.services.jira_risk_summary import (
    JiraRiskSummaryAction,
    JiraRiskSummaryGenerator,
)


def _build_signal(
    *,
    source_id: str = "PAY-102",
    rule_id: str = "jira_open_critical_bug",
    category: RiskCategory = RiskCategory.OPEN_CRITICAL_BUG,
    severity: RiskSeverity = RiskSeverity.HIGH,
    score: float = 0.9,
    title: str = "Open high-priority Jira bug may block release",
) -> RiskSignal:
    """Build a Jira risk signal for tests."""

    return RiskSignal(
        source_type="jira_issue",
        source_id=source_id,
        source_url=f"https://jira.example.com/browse/{source_id}",
        rule_id=rule_id,
        category=category,
        severity=severity,
        score=score,
        title=title,
        description="A Jira issue is risky for the release.",
        evidence={"issue_key": source_id},
    )


def test_summarize_jira_risks_returns_proceed_when_no_issues_exist() -> None:
    """Summary should recommend proceed when Jira has no open issues."""

    generator = JiraRiskSummaryGenerator()
    result = JiraRiskCollectionResult(
        status=JiraRiskCollectionStatus.SUCCESS,
        issues=[],
        issue_results=[],
        signals=[],
        error_message=None,
        duration_ms=1.0,
    )

    summary = generator.summarize_jira_risks(result, run_id="release-run-test")

    assert summary.source == "jira"
    assert summary.collection_status == JiraRiskCollectionStatus.SUCCESS
    assert summary.overall_severity == RiskSeverity.LOW
    assert summary.recommended_action == JiraRiskSummaryAction.PROCEED
    assert summary.issue_count == 0
    assert summary.risky_issue_count == 0
    assert summary.total_signal_count == 0
    assert summary.high_risk_count == 0
    assert summary.top_risks == []
    assert summary.summary_text == "Jira analysis found no open issues to review."


def test_summarize_jira_risks_blocks_release_for_critical_signal() -> None:
    """Summary should block release when a critical Jira signal exists."""

    signal = _build_signal(
        source_id="PAY-103",
        rule_id="jira_release_blocker_issue",
        category=RiskCategory.RELEASE_BLOCKER_ISSUE,
        severity=RiskSeverity.CRITICAL,
        score=0.98,
        title="Release blocker Jira issue is unresolved",
    )
    generator = JiraRiskSummaryGenerator()
    result = JiraRiskCollectionResult(
        status=JiraRiskCollectionStatus.SUCCESS,
        issues=[],
        issue_results=[],
        signals=[signal],
        error_message=None,
        duration_ms=2.0,
    )

    summary = generator.summarize_jira_risks(result, run_id="release-run-test")

    assert summary.overall_severity == RiskSeverity.CRITICAL
    assert summary.recommended_action == JiraRiskSummaryAction.BLOCK_RELEASE
    assert summary.risky_issue_count == 1
    assert summary.total_signal_count == 1
    assert summary.high_risk_count == 1
    assert len(summary.top_risks) == 1
    assert summary.top_risks[0].source_id == "PAY-103"
    assert summary.top_risks[0].severity == RiskSeverity.CRITICAL


def test_summarize_jira_risks_recommends_review_for_high_signal() -> None:
    """Summary should recommend review when high Jira signals exist."""

    signal = _build_signal(
        source_id="PAY-102",
        severity=RiskSeverity.HIGH,
        score=0.9,
    )
    generator = JiraRiskSummaryGenerator()
    result = JiraRiskCollectionResult(
        status=JiraRiskCollectionStatus.SUCCESS,
        issues=[],
        issue_results=[],
        signals=[signal],
        error_message=None,
        duration_ms=2.0,
    )

    summary = generator.summarize_jira_risks(result, run_id="release-run-test")

    assert summary.overall_severity == RiskSeverity.HIGH
    assert summary.recommended_action == JiraRiskSummaryAction.REVIEW_REQUIRED
    assert summary.risky_issue_count == 1
    assert summary.high_risk_count == 1
    assert "Recommended action: review_required" in summary.summary_text


def test_summarize_jira_risks_returns_partial_review_when_collection_failed() -> None:
    """Summary should recommend partial review when Jira collection fails."""

    generator = JiraRiskSummaryGenerator()
    result = JiraRiskCollectionResult(
        status=JiraRiskCollectionStatus.FAILED,
        issues=[],
        issue_results=[],
        signals=[],
        error_message="Jira API unavailable.",
        duration_ms=5.0,
    )

    summary = generator.summarize_jira_risks(result, run_id="release-run-test")

    assert summary.collection_status == JiraRiskCollectionStatus.FAILED
    assert summary.overall_severity == RiskSeverity.LOW
    assert summary.recommended_action == JiraRiskSummaryAction.PARTIAL_DATA_REVIEW
    assert "Jira analysis was unavailable" in summary.summary_text


def test_summarize_jira_risks_sorts_top_risks_by_severity_then_score() -> None:
    """Summary should rank critical signals above high and medium signals."""

    medium_signal = _build_signal(
        source_id="PAY-101",
        severity=RiskSeverity.MEDIUM,
        score=0.99,
        title="Medium risk issue",
    )
    high_signal = _build_signal(
        source_id="PAY-102",
        severity=RiskSeverity.HIGH,
        score=0.7,
        title="High risk issue",
    )
    critical_signal = _build_signal(
        source_id="PAY-103",
        severity=RiskSeverity.CRITICAL,
        score=0.6,
        title="Critical risk issue",
    )
    generator = JiraRiskSummaryGenerator()
    result = JiraRiskCollectionResult(
        status=JiraRiskCollectionStatus.SUCCESS,
        issues=[],
        issue_results=[],
        signals=[medium_signal, high_signal, critical_signal],
        error_message=None,
        duration_ms=2.0,
    )

    summary = generator.summarize_jira_risks(result, run_id="release-run-test")

    assert [risk.source_id for risk in summary.top_risks] == [
        "PAY-103",
        "PAY-102",
        "PAY-101",
    ]
    assert summary.overall_severity == RiskSeverity.CRITICAL
    assert summary.recommended_action == JiraRiskSummaryAction.BLOCK_RELEASE