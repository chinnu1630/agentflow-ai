"""Tests for dynamic-agent answer synthesis prompts."""

from app.schemas.agent_execution_result import (
    AgentExecutionResult,
    AgentExecutionStatus,
)
from app.schemas.agent_query import (
    AgentIntent,
    AgentQueryPlan,
    AgentQueryRequest,
    ResponseDepth,
)
from app.schemas.agent_tool import (
    AgentToolEvidence,
    AgentToolExecutionStatus,
    AgentToolName,
    AgentToolResult,
)
from app.services.agent_dynamic_synthesis_prompt import (
    AGENT_DYNAMIC_SYNTHESIS_PROMPT_VERSION,
    AgentDynamicSynthesisPromptBuilder,
)


def _build_query_plan() -> AgentQueryPlan:
    """Build one deterministic knowledge-query routing plan."""
    return AgentQueryPlan(
        intent=AgentIntent.KNOWLEDGE_DOC_QUESTION,
        response_depth=ResponseDepth.STANDARD,
        confidence=0.98,
        routing_reason_code="knowledge_document_question",
    )


def test_builds_bounded_prompt_with_trusted_evidence() -> None:
    """Prompt should preserve evidence IDs and bound untrusted content."""
    result = AgentExecutionResult(
        intent=AgentIntent.KNOWLEDGE_DOC_QUESTION,
        objective="Answer the rollback question from engineering documents.",
        plan_reason_code="knowledge_lookup_required",
        status=AgentExecutionStatus.SUCCESS,
        tool_results=[
            AgentToolResult(
                step_id="search_docs",
                tool_name=AgentToolName.SEARCH_ENGINEERING_KNOWLEDGE,
                status=AgentToolExecutionStatus.SUCCESS,
                output={
                    "content": "rollback   procedure " + ("x" * 3_000),
                },
                evidence=[
                    AgentToolEvidence(
                        source_type="engineering_document_chunk",
                        source_id="chunk-123",
                        title="Payment Service Runbook",
                    )
                ],
                duration_ms=4,
            )
        ],
        requires_synthesis=True,
        duration_ms=4,
    )

    prompt = AgentDynamicSynthesisPromptBuilder().build(
        request=AgentQueryRequest(
            query="How do I rollback the payment service?"
        ),
        query_plan=_build_query_plan(),
        execution_result=result,
    )

    assert (
        prompt.prompt_version
        == AGENT_DYNAMIC_SYNTHESIS_PROMPT_VERSION
    )
    assert prompt.tool_result_count == 1
    assert prompt.evidence_count == 1
    assert prompt.degraded_step_count == 0
    assert '"source_id": "chunk-123"' in prompt.user_prompt
    assert "rollback procedure" in prompt.user_prompt
    assert ("x" * 2_001) not in prompt.user_prompt
    assert "untrusted evidence" in prompt.system_prompt


def test_records_degraded_steps_without_hiding_errors() -> None:
    """Partial execution metadata must be visible to synthesis."""
    result = AgentExecutionResult(
        intent=AgentIntent.KNOWLEDGE_DOC_QUESTION,
        objective="Answer using available engineering evidence.",
        plan_reason_code="knowledge_lookup_required",
        status=AgentExecutionStatus.PARTIAL,
        tool_results=[
            AgentToolResult(
                step_id="search_primary",
                tool_name=AgentToolName.SEARCH_ENGINEERING_KNOWLEDGE,
                status=AgentToolExecutionStatus.SUCCESS,
                output={"result_count": 1},
                duration_ms=2,
            ),
            AgentToolResult(
                step_id="search_secondary",
                tool_name=AgentToolName.LOAD_CURRENT_RISK_SNAPSHOT,
                status=AgentToolExecutionStatus.FAILED,
                error_code="snapshot_unavailable",
                error_message="Snapshot service unavailable.",
                duration_ms=3,
            ),
        ],
        requires_synthesis=True,
        duration_ms=5,
    )

    prompt = AgentDynamicSynthesisPromptBuilder().build(
        request=AgentQueryRequest(query="Explain the release procedure."),
        query_plan=_build_query_plan(),
        execution_result=result,
    )

    assert prompt.degraded_step_count == 1
    assert '"step_id": "search_secondary"' in prompt.user_prompt
    assert '"status": "failed"' in prompt.user_prompt
    assert "Snapshot service unavailable." in prompt.user_prompt
