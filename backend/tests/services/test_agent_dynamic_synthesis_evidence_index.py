"""Tests for citation-specific dynamic synthesis evidence indexing."""

from __future__ import annotations

from app.schemas.agent_execution_result import (
    AgentExecutionResult,
    AgentExecutionStatus,
)
from app.schemas.agent_query import AgentIntent
from app.schemas.agent_tool import (
    AgentToolEvidence,
    AgentToolExecutionStatus,
    AgentToolName,
    AgentToolResult,
)
from app.services.agent_dynamic_synthesis_evidence_index import (
    AgentDynamicSynthesisEvidenceIndex,
)


def _build_execution_result(
    *tool_results: AgentToolResult,
) -> AgentExecutionResult:
    """Build one successful dynamic execution result."""
    return AgentExecutionResult(
        intent=AgentIntent.KNOWLEDGE_DOC_QUESTION,
        objective="Answer using trusted tool evidence.",
        plan_reason_code="trusted_evidence_required",
        status=AgentExecutionStatus.SUCCESS,
        tool_results=list(tool_results),
        requires_synthesis=True,
        duration_ms=10,
    )


def test_indexes_single_evidence_tool_output() -> None:
    """A one-evidence tool should map its complete bounded output."""
    result = AgentToolResult(
        step_id="lookup_pr",
        tool_name=AgentToolName.LOOKUP_GITHUB_PULL_REQUEST,
        status=AgentToolExecutionStatus.SUCCESS,
        output={
            "pull_request_number": 42,
            "title": "Fix payment retries",
            "ci_status": "failed",
        },
        evidence=[
            AgentToolEvidence(
                source_type="github_pull_request",
                source_id="pr:42",
                title="GitHub pull request #42",
            )
        ],
        duration_ms=2,
    )

    index = AgentDynamicSynthesisEvidenceIndex().build(
        _build_execution_result(result)
    )

    evidence_text = index[("github_pull_request", "pr:42")]

    assert "Fix payment retries" in evidence_text
    assert "failed" in evidence_text
    assert "GitHub pull request #42" in evidence_text


def test_indexes_matching_engineering_document_chunk_only() -> None:
    """Knowledge citations must not borrow content from sibling chunks."""
    result = AgentToolResult(
        step_id="search_docs",
        tool_name=AgentToolName.SEARCH_ENGINEERING_KNOWLEDGE,
        status=AgentToolExecutionStatus.SUCCESS,
        output={
            "result_count": 2,
            "results": [
                {
                    "chunk_id": "chunk-1",
                    "title": "Payment Runbook",
                    "content": "Rollback payment service using deployment v41.",
                },
                {
                    "chunk_id": "chunk-2",
                    "title": "Database Runbook",
                    "content": "Rotate database credentials.",
                },
            ],
        },
        evidence=[
            AgentToolEvidence(
                source_type="engineering_document_chunk",
                source_id="chunk-1",
                title="Payment Runbook",
            ),
            AgentToolEvidence(
                source_type="engineering_document_chunk",
                source_id="chunk-2",
                title="Database Runbook",
            ),
        ],
        duration_ms=4,
    )

    index = AgentDynamicSynthesisEvidenceIndex().build(
        _build_execution_result(result)
    )

    payment_text = index[
        ("engineering_document_chunk", "chunk-1")
    ]

    assert "Rollback payment service" in payment_text
    assert "Rotate database credentials" not in payment_text


def test_indexes_matching_historical_release_only() -> None:
    """Historical citations should map by release_run_id."""
    result = AgentToolResult(
        step_id="history",
        tool_name=AgentToolName.LOOKUP_RELEASE_HISTORY,
        status=AgentToolExecutionStatus.SUCCESS,
        output={
            "release_count": 2,
            "releases": [
                {
                    "release_run_id": "release-1",
                    "risk_level": "high",
                },
                {
                    "release_run_id": "release-2",
                    "risk_level": "low",
                },
            ],
        },
        evidence=[
            AgentToolEvidence(
                source_type="historical_release_risk",
                source_id="release-1",
                title="release-run-1",
            ),
            AgentToolEvidence(
                source_type="historical_release_risk",
                source_id="release-2",
                title="release-run-2",
            ),
        ],
        duration_ms=5,
    )

    index = AgentDynamicSynthesisEvidenceIndex().build(
        _build_execution_result(result)
    )

    release_text = index[
        ("historical_release_risk", "release-1")
    ]

    assert "high" in release_text
    assert "low" not in release_text


def test_indexes_similar_release_payload() -> None:
    """Similar-release evidence should map its nested release object."""
    result = AgentToolResult(
        step_id="similar",
        tool_name=AgentToolName.LOOKUP_SIMILAR_RELEASE,
        status=AgentToolExecutionStatus.SUCCESS,
        output={
            "found": True,
            "similarity_score": 0.91,
            "release": {
                "release_run_id": "release-9",
                "risk_level": "critical",
            },
        },
        evidence=[
            AgentToolEvidence(
                source_type="historical_release_risk",
                source_id="release-9",
                title="release-run-9",
            )
        ],
        duration_ms=3,
    )

    index = AgentDynamicSynthesisEvidenceIndex().build(
        _build_execution_result(result)
    )

    evidence_text = index[
        ("historical_release_risk", "release-9")
    ]

    assert "critical" in evidence_text
    assert "release-9" in evidence_text
