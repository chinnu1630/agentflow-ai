"""Verify Claude synthesis citations against trusted workflow evidence."""

from __future__ import annotations

from app.schemas.llm_risk_synthesis import (
    ClaudeReleaseRiskReport,
    SynthesisEvidenceSource,
)
from app.schemas.risk import ReleaseRunRiskResponse


class SynthesisCitationVerificationError(ValueError):
    """Raised when Claude cites evidence absent from trusted workflow data."""


class ReleaseRiskSynthesisCitationVerifier:
    """Validate every Claude citation against deterministic AgentFlow evidence."""

    def verify(
        self,
        *,
        report: ClaudeReleaseRiskReport,
        release_risk: ReleaseRunRiskResponse,
    ) -> ClaudeReleaseRiskReport:
        """Return the report only when every citation references trusted evidence.

        Complexity:
            Time: O(e + c), where e is trusted evidence and c is citations.
            Space: O(e) for the trusted citation allowlist.
        """
        trusted_citations = self._build_trusted_citation_allowlist(release_risk)

        for risk in report.risks:
            for citation in risk.evidence:
                citation_key = (citation.source, citation.source_id)

                if citation_key not in trusted_citations:
                    raise SynthesisCitationVerificationError(
                        "Claude synthesis contains an unverified evidence citation."
                    )

        return report

    @staticmethod
    def _build_trusted_citation_allowlist(
        release_risk: ReleaseRunRiskResponse,
    ) -> set[tuple[SynthesisEvidenceSource, str]]:
        """Build trusted source-type and source-ID pairs from workflow evidence."""
        trusted: set[tuple[SynthesisEvidenceSource, str]] = set()

        for risk in release_risk.release_summary.top_risks:
            source = (
                SynthesisEvidenceSource.GITHUB_PULL_REQUEST
                if risk.source_type == "github_pull_request"
                else SynthesisEvidenceSource.JIRA_ISSUE
            )
            trusted.add((source, risk.source_id))

        for pull_request in release_risk.github.risk_results:
            trusted.add(
                (
                    SynthesisEvidenceSource.GITHUB_PULL_REQUEST,
                    pull_request.source_id,
                )
            )

            for signal in pull_request.signals:
                trusted.add(
                    (
                        SynthesisEvidenceSource.DETERMINISTIC_RISK_RULE,
                        signal.rule_id,
                    )
                )

        for issue in release_risk.jira.issues:
            trusted.add(
                (
                    SynthesisEvidenceSource.JIRA_ISSUE,
                    issue.issue_key,
                )
            )

            for signal in issue.signals:
                trusted.add(
                    (
                        SynthesisEvidenceSource.DETERMINISTIC_RISK_RULE,
                        signal.rule_id,
                    )
                )

        for signal in release_risk.jira.signals:
            trusted.add(
                (
                    SynthesisEvidenceSource.DETERMINISTIC_RISK_RULE,
                    signal.rule_id,
                )
            )

        for result in release_risk.knowledge_results:
            if result.chunk_id is not None:
                trusted.add(
                    (
                        SynthesisEvidenceSource.ENGINEERING_DOCUMENT,
                        str(result.chunk_id),
                    )
                )

            if result.document_id is not None:
                trusted.add(
                    (
                        SynthesisEvidenceSource.ENGINEERING_DOCUMENT,
                        str(result.document_id),
                    )
                )

        return trusted
