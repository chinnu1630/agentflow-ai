"""CI quality gate for release-risk synthesis groundedness."""

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
    source: SynthesisEvidenceSource,
    source_id: str,
    title: str,
    supporting_fact: str,
    explanation: str,
) -> ClaudeReleaseRiskReport:
    """Build one deterministic grounded synthesis evaluation report."""
    return ClaudeReleaseRiskReport(
        recommendation=RiskSummaryActionResponse.REVIEW_REQUIRED,
        confidence=0.95,
        executive_summary="Trusted evidence requires release review.",
        risks=[
            SynthesizedReleaseRisk(
                rank=1,
                title=title,
                severity=RiskSeverityResponse.HIGH,
                confidence=0.93,
                explanation=explanation,
                evidence=[
                    SynthesisEvidenceCitation(
                        source=source,
                        source_id=source_id,
                        title=title,
                        supporting_fact=supporting_fact,
                    )
                ],
                mitigations=["Resolve the cited risk before deployment."],
            )
        ],
        requires_human_review=True,
    )


def test_release_risk_synthesis_groundedness_quality_gate() -> None:
    """Every golden synthesis case must remain completely grounded."""
    release_risk = build_release_risk_with_full_evidence()
    knowledge_result = release_risk.knowledge_results[0]

    assert knowledge_result.chunk_id is not None

    cases = [
        _build_report(
            source=SynthesisEvidenceSource.GITHUB_PULL_REQUEST,
            source_id="42",
            title="Payment API CI failure",
            supporting_fact=(
                "Three required payment integration checks are failing."
            ),
            explanation=(
                "Required payment integration checks failed on the payment API."
            ),
        ),
        _build_report(
            source=SynthesisEvidenceSource.JIRA_ISSUE,
            source_id="PAY-102",
            title="Checkout authorization blocker",
            supporting_fact=(
                "PAY-102 blocks successful checkout authorization."
            ),
            explanation=(
                "Checkout authorization fails because PAY-102 remains blocking."
            ),
        ),
        _build_report(
            source=SynthesisEvidenceSource.DETERMINISTIC_RISK_RULE,
            source_id="CI_FAILURE",
            title="Deterministic CI failure",
            supporting_fact=(
                "Three required payment integration checks are failing."
            ),
            explanation=(
                "The CI failure rule detected failing payment integration checks."
            ),
        ),
        _build_report(
            source=SynthesisEvidenceSource.ENGINEERING_DOCUMENT,
            source_id=str(knowledge_result.chunk_id),
            title="Payment rollback guidance",
            supporting_fact=(
                "Rollback requires deploying the previous stable image."
            ),
            explanation=(
                "The runbook requires the previous stable image and checkout "
                "health validation."
            ),
        ),
    ]

    evaluator = ReleaseRiskSynthesisGroundednessEvaluationService()
    reports = [
        evaluator.evaluate(
            report=case,
            release_risk=release_risk,
            run_id="groundedness-quality-gate",
        )
        for case in cases
    ]

    assert all(report.passed for report in reports)
    assert all(report.citation_validity_rate == 1.0 for report in reports)
    assert all(
        report.citation_groundedness_rate == 1.0
        for report in reports
    )
    assert all(report.risk_groundedness_rate == 1.0 for report in reports)
    assert all(not report.failure_details for report in reports)
