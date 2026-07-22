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


def _build_partial_execution_result() -> AgentExecutionResult:
    """Build one execution containing trusted evidence and a failed step."""
    return AgentExecutionResult(
        intent=AgentIntent.KNOWLEDGE_DOC_QUESTION,
        objective="Answer using all available trusted evidence.",
        plan_reason_code="knowledge_lookup_required",
        status=AgentExecutionStatus.PARTIAL,
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
            ),
            AgentToolResult(
                step_id="load_snapshot",
                tool_name=AgentToolName.LOAD_CURRENT_RISK_SNAPSHOT,
                status=AgentToolExecutionStatus.FAILED,
                error_code="snapshot_unavailable",
                error_message="Snapshot service unavailable.",
                duration_ms=4,
            ),
        ],
        requires_synthesis=True,
        duration_ms=7,
    )


def _build_failed_execution_result() -> AgentExecutionResult:
    """Build one execution where the required tool failed."""
    return AgentExecutionResult(
        intent=AgentIntent.KNOWLEDGE_DOC_QUESTION,
        objective="Answer from trusted engineering documents.",
        plan_reason_code="knowledge_lookup_required",
        status=AgentExecutionStatus.FAILED,
        tool_results=[
            AgentToolResult(
                step_id="search_docs",
                tool_name=AgentToolName.SEARCH_ENGINEERING_KNOWLEDGE,
                status=AgentToolExecutionStatus.FAILED,
                error_code="knowledge_unavailable",
                error_message="Knowledge service unavailable.",
                duration_ms=3,
            )
        ],
        requires_synthesis=True,
        duration_ms=3,
    )


def test_rejects_uncited_answer_when_trusted_evidence_exists() -> None:
    """Evidence-backed synthesis must cite at least one trusted source."""
    answer = AgentDynamicAnswer(
        answer="Follow the documented rollback procedure.",
        confidence=0.9,
        requires_human_review=False,
    )

    with pytest.raises(
        AgentDynamicSynthesisCitationVerificationError,
        match="must include at least one verified citation",
    ):
        AgentDynamicSynthesisCitationVerifier().verify(
            answer=answer,
            execution_result=_build_execution_result(),
        )


@pytest.mark.parametrize(
    "execution_result",
    [
        pytest.param(
            _build_partial_execution_result(),
            id="partial",
        ),
        pytest.param(
            _build_failed_execution_result(),
            id="failed",
        ),
    ],
)
def test_degraded_execution_requires_human_review(
    execution_result: AgentExecutionResult,
) -> None:
    """Partial and failed executions must fail closed without review."""
    answer = AgentDynamicAnswer.model_construct(
        answer="Only limited information was available.",
        confidence=0.4,
        citations=(
            [
                AgentDynamicAnswerCitation(
                    source_type="engineering_document_chunk",
                    source_id="chunk-123",
                    title="Payment Service Runbook",
                    supporting_fact="The runbook contains rollback guidance.",
                )
            ]
            if execution_result.status is AgentExecutionStatus.PARTIAL
            else []
        ),
        degraded_steps=[
            result.step_id
            for result in execution_result.tool_results
            if result.status is not AgentToolExecutionStatus.SUCCESS
        ],
        requires_human_review=False,
    )

    with pytest.raises(
        AgentDynamicSynthesisCitationVerificationError,
        match="must require human review",
    ):
        AgentDynamicSynthesisCitationVerifier().verify(
            answer=answer,
            execution_result=execution_result,
        )


def test_rejects_hidden_degraded_execution_step() -> None:
    """Synthesis must disclose every exact degraded execution step."""
    answer = AgentDynamicAnswer(
        answer="The runbook was found, but execution was degraded.",
        confidence=0.6,
        citations=[
            AgentDynamicAnswerCitation(
                source_type="engineering_document_chunk",
                source_id="chunk-123",
                title="Payment Service Runbook",
                supporting_fact="The runbook contains rollback guidance.",
            )
        ],
        degraded_steps=["different_step"],
        requires_human_review=True,
    )

    with pytest.raises(
        AgentDynamicSynthesisCitationVerificationError,
        match="must exactly match degraded execution steps",
    ):
        AgentDynamicSynthesisCitationVerifier().verify(
            answer=answer,
            execution_result=_build_partial_execution_result(),
        )
