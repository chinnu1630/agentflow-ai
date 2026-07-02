"""Tests for deterministic risk summary generation."""

from __future__ import annotations

from datetime import UTC, datetime

from app.services.github_risk_collector import (
    GitHubRiskCollectionResult,
    RiskCollectionStatus,
)
from app.services.github_risk_rules import (
    PullRequestRiskResult,
    RiskCategory,
    RiskSeverity,
    RiskSignal,
)
from app.services.github_risk_summary import (
    RiskSummaryAction,
    RiskSummaryGenerator,
)


def _signal(
    *,
    source_id: str,
    severity: RiskSeverity,
    score: float,
    title: str,
) -> RiskSignal:
    return RiskSignal(
        source_id=source_id,
        source_url=None,
        rule_id=f"test_{title.lower().replace(' ', '_')}",
        category=RiskCategory.CI_FAILURE,
        severity=severity,
        score=score,
        title=title,
        description=f"{title} description.",
        evidence={"test": True},
    )


def _risk_result(
    *,
    pull_request_number: int,
    signals: list[RiskSignal],
) -> PullRequestRiskResult:
    max_severity = None
    if signals:
        severity_rank = {
            RiskSeverity.LOW: 1,
            RiskSeverity.MEDIUM: 2,
            RiskSeverity.HIGH: 3,
            RiskSeverity.CRITICAL: 4,
        }
        max_severity = max(
            (signal.severity for signal in signals),
            key=lambda severity: severity_rank[severity],
        )

    return PullRequestRiskResult(
        source_id=f"PR-{pull_request_number}",
        source_url=None,
        pull_request_number=pull_request_number,
        total_score=max((signal.score for signal in signals), default=0.0),
        max_severity=max_severity,
        signals=signals,
        evaluated_at=datetime.now(UTC),
    )


def _github_result(
    *,
    status: RiskCollectionStatus,
    risk_results: list[PullRequestRiskResult],
) -> GitHubRiskCollectionResult:
    total_signal_count = sum(len(result.signals) for result in risk_results)
    high_risk_count = sum(
        1
        for result in risk_results
        if result.max_severity in {RiskSeverity.HIGH, RiskSeverity.CRITICAL}
    )

    return GitHubRiskCollectionResult(
        status=status,
        pull_request_count=len(risk_results),
        risk_result_count=len(risk_results),
        total_signal_count=total_signal_count,
        high_risk_count=high_risk_count,
        risk_results=risk_results,
        collected_at=datetime.now(UTC),
        duration_ms=10.0,
    )


def test_summary_returns_proceed_when_no_risks_found() -> None:
    github_result = _github_result(
        status=RiskCollectionStatus.SUCCESS,
        risk_results=[
            _risk_result(pull_request_number=1, signals=[]),
            _risk_result(pull_request_number=2, signals=[]),
        ],
    )

    summary = RiskSummaryGenerator().summarize_github_risks(
        github_result,
        run_id="test-run-001",
    )

    assert summary.overall_severity == RiskSeverity.LOW
    assert summary.recommended_action == RiskSummaryAction.PROCEED
    assert summary.pull_request_count == 2
    assert summary.risky_pull_request_count == 0
    assert summary.total_signal_count == 0
    assert summary.top_risks == []


def test_summary_requires_review_when_high_risk_signal_exists() -> None:
    high_signal = _signal(
        source_id="PR-42",
        severity=RiskSeverity.HIGH,
        score=0.85,
        title="Pull request has failing CI",
    )
    medium_signal = _signal(
        source_id="PR-43",
        severity=RiskSeverity.MEDIUM,
        score=0.45,
        title="Pull request is missing approval",
    )

    github_result = _github_result(
        status=RiskCollectionStatus.SUCCESS,
        risk_results=[
            _risk_result(pull_request_number=42, signals=[high_signal]),
            _risk_result(pull_request_number=43, signals=[medium_signal]),
        ],
    )

    summary = RiskSummaryGenerator().summarize_github_risks(
        github_result,
        run_id="test-run-002",
    )

    assert summary.overall_severity == RiskSeverity.HIGH
    assert summary.recommended_action == RiskSummaryAction.REVIEW_REQUIRED
    assert summary.risky_pull_request_count == 2
    assert summary.total_signal_count == 2
    assert summary.high_risk_count == 1
    assert summary.top_risks[0].source_id == "PR-42"


def test_summary_blocks_release_when_critical_signal_exists() -> None:
    critical_signal = _signal(
        source_id="PR-99",
        severity=RiskSeverity.CRITICAL,
        score=0.98,
        title="Critical release blocker",
    )

    github_result = _github_result(
        status=RiskCollectionStatus.SUCCESS,
        risk_results=[
            _risk_result(pull_request_number=99, signals=[critical_signal]),
        ],
    )

    summary = RiskSummaryGenerator().summarize_github_risks(
        github_result,
        run_id="test-run-003",
    )

    assert summary.overall_severity == RiskSeverity.CRITICAL
    assert summary.recommended_action == RiskSummaryAction.BLOCK_RELEASE
    assert summary.top_risks[0].severity == RiskSeverity.CRITICAL


def test_summary_uses_partial_data_review_when_collection_degraded() -> None:
    github_result = _github_result(
        status=RiskCollectionStatus.DEGRADED,
        risk_results=[],
    )

    summary = RiskSummaryGenerator().summarize_github_risks(
        github_result,
        run_id="test-run-004",
    )

    assert summary.overall_severity == RiskSeverity.MEDIUM
    assert summary.recommended_action == RiskSummaryAction.PARTIAL_DATA_REVIEW
    assert "degraded" in summary.summary_text.lower()


def test_summary_limits_top_risks() -> None:
    signals = [
        _signal(
            source_id=f"PR-{number}",
            severity=RiskSeverity.MEDIUM,
            score=0.5,
            title=f"Risk {number}",
        )
        for number in range(1, 6)
    ]

    github_result = _github_result(
        status=RiskCollectionStatus.SUCCESS,
        risk_results=[
            _risk_result(
                pull_request_number=index + 1,
                signals=[signal],
            )
            for index, signal in enumerate(signals)
        ],
    )

    summary = RiskSummaryGenerator().summarize_github_risks(
        github_result,
        run_id="test-run-005",
        max_top_risks=3,
    )

    assert len(summary.top_risks) == 3