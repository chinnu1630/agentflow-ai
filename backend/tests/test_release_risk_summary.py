"""Tests for combined release risk summary generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from app.services.github_risk_rules import RiskSeverity
from app.services.release_risk_summary import (
    ReleaseRiskSummaryAction,
    ReleaseRiskSummaryGenerator,
)


@dataclass(frozen=True)
class FakeRiskItem:
    """Small fake risk item used to test release summary logic."""

    source_type: str
    source_id: str
    source_url: str | None
    severity: RiskSeverity
    score: float
    title: str
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FakeSourceSummary:
    """Small fake source summary used to test release summary logic."""

    source: str
    overall_severity: RiskSeverity
    recommended_action: str
    total_signal_count: int
    high_risk_count: int
    summary_text: str
    top_risks: list[FakeRiskItem] = field(default_factory=list)


def _summary(
    *,
    source: str,
    overall_severity: RiskSeverity = RiskSeverity.LOW,
    recommended_action: str = "proceed",
    total_signal_count: int = 0,
    high_risk_count: int = 0,
    top_risks: list[FakeRiskItem] | None = None,
) -> FakeSourceSummary:
    """Build a fake source summary for release summary tests."""

    return FakeSourceSummary(
        source=source,
        overall_severity=overall_severity,
        recommended_action=recommended_action,
        total_signal_count=total_signal_count,
        high_risk_count=high_risk_count,
        summary_text=f"{source} summary text",
        top_risks=top_risks or [],
    )


def _risk_item(
    *,
    source_type: str,
    source_id: str,
    severity: RiskSeverity,
    score: float,
) -> FakeRiskItem:
    """Build a fake top risk item."""

    return FakeRiskItem(
        source_type=source_type,
        source_id=source_id,
        source_url=None,
        severity=severity,
        score=score,
        title=f"Risk {source_id}",
        reason=f"Reason for {source_id}",
        evidence={"source_id": source_id},
    )


def test_release_summary_proceeds_when_all_sources_are_clean() -> None:
    """Release summary should proceed when GitHub and Jira have no risks."""
    generator = ReleaseRiskSummaryGenerator()

    summary = generator.summarize_release_risks(
        github_summary=cast(
            Any,
            _summary(source="github"),
        ),
        jira_summary=cast(
            Any,
            _summary(source="jira"),
        ),
        run_id="release-run-test",
    )

    assert summary.source == "release"
    assert summary.overall_severity == RiskSeverity.LOW
    assert summary.recommended_action == ReleaseRiskSummaryAction.PROCEED
    assert summary.total_signal_count == 0
    assert summary.high_risk_count == 0
    assert summary.source_summary_count == 2
    assert summary.top_risks == []
    assert summary.summary_text


def test_release_summary_blocks_when_any_source_blocks_release() -> None:
    """Release summary should block when GitHub or Jira has a blocking risk."""
    generator = ReleaseRiskSummaryGenerator()

    summary = generator.summarize_release_risks(
        github_summary=cast(
            Any,
            _summary(source="github"),
        ),
        jira_summary=cast(
            Any,
            _summary(
                source="jira",
                overall_severity=RiskSeverity.CRITICAL,
                recommended_action="block_release",
                total_signal_count=1,
                high_risk_count=1,
            ),
        ),
        run_id="release-run-test",
    )

    assert summary.overall_severity == RiskSeverity.CRITICAL
    assert summary.recommended_action == ReleaseRiskSummaryAction.BLOCK_RELEASE
    assert summary.total_signal_count == 1
    assert summary.high_risk_count == 1
    assert "blocked" in summary.summary_text


def test_release_summary_requires_partial_review_when_source_data_is_partial() -> None:
    """Release summary should require review when a source has partial data."""
    generator = ReleaseRiskSummaryGenerator()

    summary = generator.summarize_release_risks(
        github_summary=cast(
            Any,
            _summary(
                source="github",
                recommended_action="partial_data_review",
            ),
        ),
        jira_summary=cast(
            Any,
            _summary(source="jira"),
        ),
        run_id="release-run-test",
    )

    assert summary.recommended_action == ReleaseRiskSummaryAction.PARTIAL_DATA_REVIEW
    assert "manual review" in summary.summary_text


def test_release_summary_requires_review_when_non_blocking_signals_exist() -> None:
    """Release summary should require review when risk signals exist."""
    generator = ReleaseRiskSummaryGenerator()

    summary = generator.summarize_release_risks(
        github_summary=cast(
            Any,
            _summary(
                source="github",
                overall_severity=RiskSeverity.HIGH,
                recommended_action="review_required",
                total_signal_count=2,
                high_risk_count=1,
            ),
        ),
        jira_summary=cast(
            Any,
            _summary(source="jira"),
        ),
        run_id="release-run-test",
    )

    assert summary.overall_severity == RiskSeverity.HIGH
    assert summary.recommended_action == ReleaseRiskSummaryAction.REVIEW_REQUIRED
    assert summary.total_signal_count == 2
    assert summary.high_risk_count == 1


def test_release_summary_sorts_top_risks_across_sources() -> None:
    """Release summary should rank top risks by severity and score."""
    generator = ReleaseRiskSummaryGenerator()

    github_risk = _risk_item(
        source_type="github_pull_request",
        source_id="PR-42",
        severity=RiskSeverity.HIGH,
        score=0.8,
    )
    jira_risk = _risk_item(
        source_type="jira_issue",
        source_id="PAY-103",
        severity=RiskSeverity.CRITICAL,
        score=0.7,
    )

    summary = generator.summarize_release_risks(
        github_summary=cast(
            Any,
            _summary(
                source="github",
                overall_severity=RiskSeverity.HIGH,
                recommended_action="review_required",
                total_signal_count=1,
                high_risk_count=1,
                top_risks=[github_risk],
            ),
        ),
        jira_summary=cast(
            Any,
            _summary(
                source="jira",
                overall_severity=RiskSeverity.CRITICAL,
                recommended_action="block_release",
                total_signal_count=1,
                high_risk_count=1,
                top_risks=[jira_risk],
            ),
        ),
        run_id="release-run-test",
    )

    assert len(summary.top_risks) == 2
    assert summary.top_risks[0].source == "jira"
    assert summary.top_risks[0].source_id == "PAY-103"
    assert summary.top_risks[1].source == "github"
    assert summary.top_risks[1].source_id == "PR-42"