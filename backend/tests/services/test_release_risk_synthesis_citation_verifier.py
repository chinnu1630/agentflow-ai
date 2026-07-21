"""Tests for Claude synthesis citation verification."""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.schemas.llm_risk_synthesis import (
    ClaudeReleaseRiskReport,
    SynthesisEvidenceCitation,
    SynthesisEvidenceSource,
    SynthesizedReleaseRisk,
)
from app.schemas.risk import (
    ReleaseRunRiskResponse,
    RiskSeverityResponse,
    RiskSummaryActionResponse,
)
from app.services.release_risk_synthesis_citation_verifier import (
    ReleaseRiskSynthesisCitationVerifier,
    SynthesisCitationVerificationError,
)
from tests.services.test_slack_release_alert_service import (
    build_snapshot_payload,
)


def _build_release_risk() -> ReleaseRunRiskResponse:
    """Build trusted deterministic release-risk evidence."""
    return ReleaseRunRiskResponse.model_validate(
        build_snapshot_payload(
            release_run_id=uuid4(),
            approval_request_id=uuid4(),
        )
    )


def _build_report(
    *,
    source: SynthesisEvidenceSource,
    source_id: str,
) -> ClaudeReleaseRiskReport:
    """Build one structured Claude report with a selected citation."""
    return ClaudeReleaseRiskReport(
        recommendation=RiskSummaryActionResponse.REVIEW_REQUIRED,
        confidence=0.9,
        executive_summary="The release requires human review.",
        risks=[
            SynthesizedReleaseRisk(
                rank=1,
                title="Payment release risk",
                severity=RiskSeverityResponse.HIGH,
                confidence=0.92,
                explanation="Trusted evidence indicates elevated risk.",
                evidence=[
                    SynthesisEvidenceCitation(
                        source=source,
                        source_id=source_id,
                        title="Trusted release evidence",
                        supporting_fact="A trusted release risk was detected.",
                    )
                ],
                mitigations=["Resolve the release risk before deployment."],
            )
        ],
        requires_human_review=True,
    )


def test_accepts_exact_trusted_source_type_and_source_id() -> None:
    """Verifier should accept a citation present in deterministic evidence."""
    release_risk = _build_release_risk()
    report = _build_report(
        source=SynthesisEvidenceSource.GITHUB_PULL_REQUEST,
        source_id="1",
    )

    verified = ReleaseRiskSynthesisCitationVerifier().verify(
        report=report,
        release_risk=release_risk,
    )

    assert verified is report


def test_rejects_invented_source_id() -> None:
    """Verifier should reject a source ID absent from workflow evidence."""
    release_risk = _build_release_risk()
    report = _build_report(
        source=SynthesisEvidenceSource.GITHUB_PULL_REQUEST,
        source_id="9999",
    )

    with pytest.raises(
        SynthesisCitationVerificationError,
        match="unverified evidence citation",
    ):
        ReleaseRiskSynthesisCitationVerifier().verify(
            report=report,
            release_risk=release_risk,
        )


def test_rejects_valid_id_with_wrong_source_type() -> None:
    """Verifier should validate the source type and source ID together."""
    release_risk = _build_release_risk()
    report = _build_report(
        source=SynthesisEvidenceSource.JIRA_ISSUE,
        source_id="1",
    )

    with pytest.raises(SynthesisCitationVerificationError):
        ReleaseRiskSynthesisCitationVerifier().verify(
            report=report,
            release_risk=release_risk,
        )
