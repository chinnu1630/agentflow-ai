"""Tests for strict Claude release-risk synthesis schemas."""

import pytest
from pydantic import ValidationError

from app.schemas.llm_risk_synthesis import (
    ClaudeReleaseRiskReport,
    SynthesisEvidenceCitation,
    SynthesisEvidenceSource,
    SynthesizedReleaseRisk,
)
from app.schemas.risk import (
    RiskSeverityResponse,
    RiskSummaryActionResponse,
)


def _build_evidence() -> SynthesisEvidenceCitation:
    """Create reusable trusted evidence for schema tests."""
    return SynthesisEvidenceCitation(
        source=SynthesisEvidenceSource.JIRA_ISSUE,
        source_id="PAY-102",
        title="Payment release blocker",
        source_url="https://jira.example.com/browse/PAY-102",
        supporting_fact="PAY-102 is an unresolved release-blocking issue.",
    )


def _build_risk(
    *,
    rank: int = 1,
    severity: RiskSeverityResponse = RiskSeverityResponse.CRITICAL,
) -> SynthesizedReleaseRisk:
    """Create one reusable synthesized release risk."""
    return SynthesizedReleaseRisk(
        rank=rank,
        title="Unresolved payment blocker",
        severity=severity,
        confidence=0.95,
        explanation="The unresolved payment issue blocks a safe release.",
        evidence=[_build_evidence()],
        mitigations=["Resolve PAY-102 before deployment."],
    )


def test_accepts_valid_block_release_report() -> None:
    """A grounded critical-risk report should pass validation."""
    report = ClaudeReleaseRiskReport(
        recommendation=RiskSummaryActionResponse.BLOCK_RELEASE,
        confidence=0.94,
        executive_summary="The release should be blocked until PAY-102 is resolved.",
        risks=[_build_risk()],
        missing_information=[],
        degraded_sources=[],
        requires_human_review=True,
    )

    assert report.schema_version == "claude_release_risk_report_v1"
    assert report.risks[0].evidence[0].source_id == "PAY-102"


def test_rejects_non_sequential_risk_ranks() -> None:
    """Risk ranks must start at one and remain sequential."""
    with pytest.raises(ValidationError, match="risk ranks must be sequential"):
        ClaudeReleaseRiskReport(
            recommendation=RiskSummaryActionResponse.BLOCK_RELEASE,
            confidence=0.90,
            executive_summary="Two release risks were identified.",
            risks=[
                _build_risk(rank=1),
                _build_risk(rank=3),
            ],
            requires_human_review=True,
        )


def test_rejects_high_risk_without_human_review() -> None:
    """High-impact risks must never bypass human review."""
    with pytest.raises(
        ValidationError,
        match="high or critical risks must require human review",
    ):
        ClaudeReleaseRiskReport(
            recommendation=RiskSummaryActionResponse.BLOCK_RELEASE,
            confidence=0.92,
            executive_summary="A critical blocker was identified.",
            risks=[_build_risk()],
            requires_human_review=False,
        )


def test_rejects_non_proceed_report_without_risks() -> None:
    """A blocking or review recommendation must contain supporting risks."""
    with pytest.raises(
        ValidationError,
        match="non-proceed recommendations must include at least one risk",
    ):
        ClaudeReleaseRiskReport(
            recommendation=RiskSummaryActionResponse.REVIEW_REQUIRED,
            confidence=0.70,
            executive_summary="The release requires review.",
            risks=[],
            requires_human_review=True,
        )


def test_rejects_duplicate_mitigations() -> None:
    """Duplicate mitigation steps should be rejected."""
    with pytest.raises(
        ValidationError,
        match="mitigations must not contain duplicate values",
    ):
        SynthesizedReleaseRisk(
            rank=1,
            title="Unresolved payment blocker",
            severity=RiskSeverityResponse.HIGH,
            confidence=0.88,
            explanation="The payment issue increases release risk.",
            evidence=[_build_evidence()],
            mitigations=[
                "Resolve PAY-102.",
                "Resolve PAY-102.",
            ],
        )
