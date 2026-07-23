"""Tests for deterministic release-risk synthesis groundedness."""

from __future__ import annotations

from app.schemas.llm_risk_synthesis import (
    ClaudeReleaseRiskReport,
    SynthesisEvidenceCitation,
    SynthesisEvidenceSource,
    SynthesizedReleaseRisk,
)
from app.schemas.risk_enums import (
    RiskSeverityResponse,
    RiskSummaryActionResponse,
)
from app.services.release_risk_synthesis_groundedness_evaluation_service import (
    ReleaseRiskSynthesisGroundednessEvaluationService,
)
from tests.fixtures.release_risk_synthesis_evidence import (
    build_release_risk_with_full_evidence,
)


def _build_report(
    *,
    source_id: str = "42",
    supporting_fact: str = (
        "Three required payment integration checks are failing."
    ),
    explanation: str = (
        "Required payment integration checks failed on the payment API."
    ),
) -> ClaudeReleaseRiskReport:
    """Build one structured release-risk report for groundedness tests."""
    return ClaudeReleaseRiskReport(
        recommendation=RiskSummaryActionResponse.REVIEW_REQUIRED,
        confidence=0.94,
        executive_summary="The payment release requires human review.",
        risks=[
            SynthesizedReleaseRisk(
                rank=1,
                title="Payment API CI failure",
                severity=RiskSeverityResponse.HIGH,
                confidence=0.93,
                explanation=explanation,
                evidence=[
                    SynthesisEvidenceCitation(
                        source=(
                            SynthesisEvidenceSource.GITHUB_PULL_REQUEST
                        ),
                        source_id=source_id,
                        title="Payment API has failing CI",
                        supporting_fact=supporting_fact,
                    )
                ],
                mitigations=[
                    "Restore all required payment integration checks."
                ],
            )
        ],
        requires_human_review=True,
    )


def test_grounded_report_passes_with_complete_metrics() -> None:
    """Supported claims and verified citations should pass the quality gate."""
    report = (
        ReleaseRiskSynthesisGroundednessEvaluationService().evaluate(
            report=_build_report(),
            release_risk=build_release_risk_with_full_evidence(),
        )
    )

    assert report.passed is True
    assert report.total_risks == 1
    assert report.grounded_risks == 1
    assert report.total_citations == 1
    assert report.verified_citations == 1
    assert report.grounded_citations == 1
    assert report.citation_validity_rate == 1.0
    assert report.citation_groundedness_rate == 1.0
    assert report.risk_groundedness_rate == 1.0
    assert report.failure_details == []


def test_unverified_citation_fails_safely() -> None:
    """An invented source ID should fail without exposing evidence content."""
    report = (
        ReleaseRiskSynthesisGroundednessEvaluationService().evaluate(
            report=_build_report(source_id="9999"),
            release_risk=build_release_risk_with_full_evidence(),
        )
    )

    assert report.passed is False
    assert report.verified_citations == 0
    assert report.citation_validity_rate == 0.0
    assert report.failure_details[0].reason == "unverified_citation"
    assert report.failure_details[0].source_id == "9999"


def test_unsupported_supporting_fact_fails() -> None:
    """A valid citation must not support an unrelated fabricated fact."""
    report = (
        ReleaseRiskSynthesisGroundednessEvaluationService().evaluate(
            report=_build_report(
                supporting_fact=(
                    "A database migration deleted customer invoices."
                )
            ),
            release_risk=build_release_risk_with_full_evidence(),
        )
    )

    assert report.passed is False
    assert report.verified_citations == 1
    assert report.grounded_citations == 0
    assert report.failure_details[0].reason == (
        "unsupported_supporting_fact"
    )
    assert report.failure_details[0].overlap_score == 0.0


def test_unsupported_risk_explanation_fails() -> None:
    """A risk explanation must be supported by its combined cited evidence."""
    report = (
        ReleaseRiskSynthesisGroundednessEvaluationService().evaluate(
            report=_build_report(
                explanation=(
                    "A regional network outage corrupted billing invoices."
                )
            ),
            release_risk=build_release_risk_with_full_evidence(),
        )
    )

    assert report.passed is False
    assert report.grounded_citations == 1
    assert report.grounded_risks == 0
    assert report.failure_details[0].reason == (
        "unsupported_risk_explanation"
    )


def test_empty_proceed_report_passes_as_vacuously_grounded() -> None:
    """A valid no-risk proceed report should produce safe empty-set metrics."""
    synthesis_report = ClaudeReleaseRiskReport(
        recommendation=RiskSummaryActionResponse.PROCEED,
        confidence=0.9,
        executive_summary="No material release risks were identified.",
        risks=[],
        requires_human_review=False,
    )

    report = (
        ReleaseRiskSynthesisGroundednessEvaluationService().evaluate(
            report=synthesis_report,
            release_risk=build_release_risk_with_full_evidence(),
        )
    )

    assert report.passed is True
    assert report.total_risks == 0
    assert report.total_citations == 0
    assert report.citation_validity_rate == 1.0
    assert report.citation_groundedness_rate == 1.0
    assert report.risk_groundedness_rate == 1.0
