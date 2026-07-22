"""Tests for dynamic-agent synthesis citation verification."""

import pytest

from app.schemas.agent_dynamic_synthesis import (
    AgentDynamicAnswer,
    AgentDynamicAnswerCitation,
)
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
from app.services.agent_dynamic_synthesis_citation_verifier import (
    AgentDynamicSynthesisCitationVerificationError,
    AgentDynamicSynthesisCitationVerifier,
)


def _build_execution_result() -> AgentExecutionResult:
    """Build one trusted tool execution result."""
    return AgentExecutionResult(
        intent=AgentIntent.KNOWLEDGE_DOC_QUESTION,
        objective="Answer from trusted engineering documents.",
        plan_reason_code="knowledge_lookup_required",
        status=AgentExecutionStatus.SUCCESS,
        tool_results=[
            AgentToolResult(
                step_id="search_docs",
                tool_name=AgentToolName.SEARCH_ENGINEERING_KNOWLEDGE,
                status=AgentToolExecutionStatus.SUCCESS,
                output={"result_count": 1},
                evidence=[
                    AgentToolEvidence(
                        source_type="engineering_document_chunk",
                        source_id="chunk-123",
                        title="Payment Service Runbook",
                    )
                ],
                duration_ms=3,
            )
        ],
        requires_synthesis=True,
        duration_ms=3,
    )


def _build_answer(
    *,
    source_type: str,
    source_id: str,
) -> AgentDynamicAnswer:
    """Build one dynamic answer with a selected citation."""
    return AgentDynamicAnswer(
        answer="Follow the documented payment rollback procedure.",
        confidence=0.95,
        citations=[
            AgentDynamicAnswerCitation(
                source_type=source_type,
                source_id=source_id,
                title="Payment Service Runbook",
                supporting_fact="The runbook defines the rollback steps.",
            )
        ],
        requires_human_review=False,
    )


def test_accepts_exact_trusted_dynamic_citation() -> None:
    """Verifier should accept an exact tool-evidence citation."""
    answer = _build_answer(
        source_type="engineering_document_chunk",
        source_id="chunk-123",
    )

    verified = AgentDynamicSynthesisCitationVerifier().verify(
        answer=answer,
        execution_result=_build_execution_result(),
    )

    assert verified is answer


def test_rejects_invented_dynamic_source_id() -> None:
    """Verifier should reject a source ID absent from tool evidence."""
    answer = _build_answer(
        source_type="engineering_document_chunk",
        source_id="chunk-999",
    )

    with pytest.raises(
        AgentDynamicSynthesisCitationVerificationError,
        match="unverified citation",
    ):
        AgentDynamicSynthesisCitationVerifier().verify(
            answer=answer,
            execution_result=_build_execution_result(),
        )


def test_rejects_valid_id_with_wrong_source_type() -> None:
    """Verifier must validate source type and source ID together."""
    answer = _build_answer(
        source_type="jira_issue",
        source_id="chunk-123",
    )

    with pytest.raises(
        AgentDynamicSynthesisCitationVerificationError
    ):
        AgentDynamicSynthesisCitationVerifier().verify(
            answer=answer,
            execution_result=_build_execution_result(),
        )
