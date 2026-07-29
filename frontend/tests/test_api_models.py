"""Tests for typed AgentFlow frontend API models."""

from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from agentflow_frontend.api_models import AgentQueryRequest, AgentQueryResponse


def test_agent_query_request_normalizes_valid_query() -> None:
    """Strip surrounding whitespace while preserving the manager question."""
    request = AgentQueryRequest(query="  What are the biggest release risks?  ")

    assert request.query == "What are the biggest release risks?"


@pytest.mark.parametrize("query", ["", "   ", "?!..."])
def test_agent_query_request_rejects_invalid_query(query: str) -> None:
    """Reject empty and control-only manager input."""
    with pytest.raises(ValidationError):
        AgentQueryRequest(query=query)


def test_agent_query_request_rejects_unknown_fields() -> None:
    """Prevent unreviewed request properties from crossing the API boundary."""
    with pytest.raises(ValidationError):
        AgentQueryRequest.model_validate(
            {
                "query": "What are the biggest release risks?",
                "requested_by": "spoofed-manager",
            }
        )


def test_agent_query_response_parses_manager_dashboard_fields() -> None:
    """Parse answer, risks, citations, degradation, and approval metadata."""
    release_run_id = "14326708-c085-4e6d-9c32-47dc92b24841"
    approval_id = "3cc48c03-678b-458e-9418-941e914c220b"

    response = AgentQueryResponse.model_validate(
        {
            "answer": "Two high-priority risks require review.",
            "plan": {
                "intent": "release_risk_summary",
                "response_depth": "detailed",
                "confidence": 0.97,
                "release_run_id": release_run_id,
                "requires_current_snapshot": True,
                "requires_human_approval": True,
                "routing_reason_code": "fresh_release_risk_request",
                "filters": {},
            },
            "release_risk": {
                "release_run": {
                    "id": release_run_id,
                    "run_id": "release-run-demo",
                    "query": "What are the biggest release risks?",
                    "requested_by": "manager@example.com",
                    "status": "waiting_for_approval",
                    "created_at": "2026-07-27T12:00:00Z",
                },
                "github": {
                    "status": "degraded",
                    "error_type": "GitHubUnavailableError",
                    "error_message": "GitHub was unavailable.",
                },
                "jira": {
                    "status": "success",
                },
                "release_summary": {
                    "overall_severity": "high",
                    "recommended_action": "hold",
                    "total_signal_count": 4,
                    "high_risk_count": 2,
                    "summary_text": "Release should be reviewed before deployment.",
                    "top_risks": [
                        {
                            "source": "jira",
                            "source_type": "jira_issue",
                            "source_id": "PAY-102",
                            "severity": "high",
                            "score": 0.91,
                            "title": "Payment rollback defect",
                            "reason": "Open release-blocking issue.",
                            "evidence": {"priority": "P1"},
                        }
                    ],
                },
                "risk_score": {
                    "score": 0.88,
                    "risk_level": "high",
                    "recommended_action": "hold",
                    "reasons": ["Open P1 release blocker"],
                },
                "approval_required": True,
                "approval_request_id": approval_id,
                "approval_status": "pending",
                "backend_future_field": "ignored",
            },
            "citations": [
                {
                    "source": "jira",
                    "source_type": "jira_issue",
                    "source_id": "PAY-102",
                    "title": "Payment rollback defect",
                    "source_url": "https://jira.example.test/browse/PAY-102",
                }
            ],
            "approval_required": True,
            "future_response_field": "ignored",
        }
    )

    assert response.release_risk is not None
    assert response.release_risk.release_run.id == UUID(release_run_id)
    assert response.release_risk.github.status == "degraded"
    assert response.release_risk.release_summary.top_risks[0].source_id == "PAY-102"
    assert response.release_risk.approval_request_id == UUID(approval_id)
    assert response.citations[0].source_id == "PAY-102"


def test_agent_query_response_rejects_out_of_range_risk_score() -> None:
    """Reject malformed backend data before rendering it in Streamlit."""
    with pytest.raises(ValidationError):
        AgentQueryResponse.model_validate(
            {
                "answer": "Invalid response",
                "plan": {
                    "intent": "release_risk_summary",
                    "response_depth": "brief",
                    "confidence": 1.5,
                    "routing_reason_code": "invalid",
                },
                "approval_required": False,
            }
        )

def test_agent_query_request_accepts_validated_entity_context() -> None:
    """Serialize one typed PR reference for a contextual follow-up."""

    request = AgentQueryRequest.model_validate(
        {
            "query": "Why is it risky?",
            "context_entity_references": {
                "pull_request_numbers": [4],
            },
        }
    )

    assert request.context_entity_references is not None
    assert request.context_entity_references.pull_request_numbers == [4]
    assert request.context_entity_references.jira_issue_keys == []
