"""CI quality gate for dynamic agent synthesis groundedness."""

from __future__ import annotations

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
from app.services.agent_dynamic_synthesis_groundedness_evaluation_service import (
    AgentDynamicSynthesisGroundednessEvaluationService,
)


def test_dynamic_synthesis_groundedness_quality_gate() -> None:
    """Golden dynamic answer must remain fully grounded in tool evidence."""
    execution_result = AgentExecutionResult(
        intent=AgentIntent.KNOWLEDGE_DOC_QUESTION,
        objective="Explain the trusted payment rollback procedure.",
        plan_reason_code="knowledge_lookup_required",
        status=AgentExecutionStatus.SUCCESS,
        tool_results=[
            AgentToolResult(
                step_id="search_payment_runbook",
                tool_name=AgentToolName.SEARCH_ENGINEERING_KNOWLEDGE,
                status=AgentToolExecutionStatus.SUCCESS,
                output={
                    "result_count": 1,
                    "results": [
                        {
                            "document_id": "payment-runbook",
                            "chunk_id": "payment-rollback-chunk",
                            "title": "Payment Service Runbook",
                            "content": (
                                "Rollback the payment service by restoring "
                                "deployment version 41, then verify checkout "
                                "and payment authorization health checks."
                            ),
                        }
                    ],
                },
                evidence=[
                    AgentToolEvidence(
                        source_type="engineering_document_chunk",
                        source_id="payment-rollback-chunk",
                        title="Payment Service Runbook",
                    )
                ],
                duration_ms=4,
            )
        ],
        requires_synthesis=True,
        duration_ms=4,
    )
    answer = AgentDynamicAnswer(
        answer=(
            "Restore deployment version 41, then verify checkout and payment "
            "authorization health checks."
        ),
        confidence=0.96,
        citations=[
            AgentDynamicAnswerCitation(
                source_type="engineering_document_chunk",
                source_id="payment-rollback-chunk",
                title="Payment Service Runbook",
                supporting_fact=(
                    "The rollback restores deployment version 41 and verifies "
                    "checkout and payment authorization health checks."
                ),
            )
        ],
        requires_human_review=False,
    )

    report = (
        AgentDynamicSynthesisGroundednessEvaluationService().evaluate(
            answer=answer,
            execution_result=execution_result,
            run_id="dynamic-groundedness-quality-gate",
        )
    )

    assert report.passed is True
    assert report.citation_validity_rate == 1.0
    assert report.citation_groundedness_rate == 1.0
    assert report.answer_grounded is True
