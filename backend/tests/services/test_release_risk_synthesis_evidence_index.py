"""Tests for citation-specific release-risk synthesis evidence."""

from __future__ import annotations

from app.schemas.llm_risk_synthesis import SynthesisEvidenceSource
from app.services.release_risk_synthesis_evidence_index import (
    ReleaseRiskSynthesisEvidenceIndex,
)
from tests.fixtures.release_risk_synthesis_evidence import (
    build_release_risk_with_full_evidence,
)


def test_builds_citation_specific_text_for_all_trusted_sources() -> None:
    """Index should expose evidence text under every valid citation identity."""
    evidence = ReleaseRiskSynthesisEvidenceIndex().build(
        build_release_risk_with_full_evidence()
    )

    pull_request_text = evidence[
        (SynthesisEvidenceSource.GITHUB_PULL_REQUEST, "42")
    ]
    assert "Payment API has failing CI" in pull_request_text
    assert "Three required payment integration checks are failing." in (
        pull_request_text
    )
    assert "failed_check_count" in pull_request_text

    github_rule_text = evidence[
        (SynthesisEvidenceSource.DETERMINISTIC_RISK_RULE, "CI_FAILURE")
    ]
    assert "Payment integration checks failed" in github_rule_text
    assert "required_check" in github_rule_text

    jira_issue_text = evidence[
        (SynthesisEvidenceSource.JIRA_ISSUE, "PAY-102")
    ]
    assert "Checkout authorization fails" in jira_issue_text
    assert "PAY-102 blocks successful checkout authorization." in jira_issue_text

    jira_rule_text = evidence[
        (
            SynthesisEvidenceSource.DETERMINISTIC_RISK_RULE,
            "OPEN_CRITICAL_BUG",
        )
    ]
    assert "Critical checkout defect remains open" in jira_rule_text
    assert "priority P1" in jira_rule_text

    document_keys = [
        key
        for key in evidence
        if key[0] is SynthesisEvidenceSource.ENGINEERING_DOCUMENT
    ]
    assert len(document_keys) == 2

    for key in document_keys:
        assert "Payment Service Runbook" in evidence[key]
        assert "previous stable image" in evidence[key]
        assert "service payments" in evidence[key]
