"""Verify Claude synthesis citations against trusted workflow evidence."""

from __future__ import annotations

from app.schemas.llm_risk_synthesis import ClaudeReleaseRiskReport
from app.schemas.risk import ReleaseRunRiskResponse
from app.services.release_risk_synthesis_evidence_index import (
    ReleaseRiskSynthesisEvidenceIndex,
)


class SynthesisCitationVerificationError(ValueError):
    """Raised when Claude cites evidence absent from trusted workflow data."""


class ReleaseRiskSynthesisCitationVerifier:
    """Validate every Claude citation against deterministic AgentFlow evidence."""

    def __init__(
        self,
        evidence_index: ReleaseRiskSynthesisEvidenceIndex | None = None,
    ) -> None:
        """Initialize the verifier with the trusted evidence index."""
        self._evidence_index = evidence_index or ReleaseRiskSynthesisEvidenceIndex()

    def verify(
        self,
        *,
        report: ClaudeReleaseRiskReport,
        release_risk: ReleaseRunRiskResponse,
    ) -> ClaudeReleaseRiskReport:
        """Return the report only when every citation references trusted evidence.

        Complexity:
            Time: O(e + c), where e is trusted evidence and c is citations.
            Space: O(e) for the citation-specific trusted evidence index.
        """
        trusted_evidence = self._evidence_index.build(release_risk)

        for risk in report.risks:
            for citation in risk.evidence:
                citation_key = (citation.source, citation.source_id)

                if citation_key not in trusted_evidence:
                    raise SynthesisCitationVerificationError(
                        "Claude synthesis contains an unverified evidence citation."
                    )

        return report
