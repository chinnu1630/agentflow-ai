"""Tests for deterministic dynamic synthesis groundedness evaluation."""

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


def _build_execution_result() -> AgentExecutionResult:
    """Build one trusted engineering-knowledge execution result."""
    return AgentExecutionResult(
        intent=AgentIntent.KNOWLEDGE_DOC_QUESTION,
        objective="Answer from trusted engineering documentation.",
        plan_reason_code="knowledge_lookup_required",
        status=AgentExecutionStatus.SUCCESS,
        tool_results=[
            AgentToolResult(
                step_id="search_docs",
                tool_name=AgentToolName.SEARCH_ENGINEERING_KNOWLEDGE,
                status=AgentToolExecutionStatus.SUCCESS,
                output={
                    "result_count": 1,
                    "results": [
                        {
                            "document_id": "document-1",
                            "chunk_id": "chunk-123",
                            "title": "Payment Service Runbook",
                            "content": (
                                "Rollback the payment service by restoring "
                                "deployment version 41 and verifying checkout."
                            ),
                        }
                    ],
                },
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
    source_id: str = "chunk-123",
    supporting_fact: str = (
        "Rollback the payment service by restoring deployment version 41."
    ),
    answer_text: str = (
        "Restore deployment version 41 to rollback the payment service."
    ),
) -> AgentDynamicAnswer:
    """Build one dynamic answer with a selected citation."""
    return AgentDynamicAnswer(
        answer=answer_text,
        confidence=0.95,
        citations=[
            AgentDynamicAnswerCitation(
                source_type="engineering_document_chunk",
                source_id=source_id,
                title="Payment Service Runbook",
                supporting_fact=supporting_fact,
            )
        ],
        requires_human_review=False,
    )


def test_passes_grounded_dynamic_answer() -> None:
    """Supported citations and answer claims should pass."""
    report = (
        AgentDynamicSynthesisGroundednessEvaluationService().evaluate(
            answer=_build_answer(),
            execution_result=_build_execution_result(),
            run_id="run-grounded",
        )
    )

    assert report.passed is True
    assert report.total_citations == 1
    assert report.verified_citations == 1
    assert report.grounded_citations == 1
    assert report.answer_grounded is True
    assert report.failure_details == []


def test_rejects_invented_dynamic_citation() -> None:
    """A citation absent from executed evidence should fail."""
    report = (
        AgentDynamicSynthesisGroundednessEvaluationService().evaluate(
            answer=_build_answer(source_id="chunk-999"),
            execution_result=_build_execution_result(),
        )
    )

    assert report.passed is False
    assert report.verified_citations == 0
    assert report.answer_grounded is False
    assert report.failure_details[0].reason == "unverified_citation"


def test_rejects_unsupported_supporting_fact() -> None:
    """A citation fact must overlap its exact evidence object."""
    report = (
        AgentDynamicSynthesisGroundednessEvaluationService().evaluate(
            answer=_build_answer(
                supporting_fact=(
                    "The service requires rotating database credentials."
                )
            ),
            execution_result=_build_execution_result(),
        )
    )

    assert report.passed is False
    assert report.verified_citations == 1
    assert report.grounded_citations == 0
    assert any(
        failure.reason == "unsupported_supporting_fact"
        for failure in report.failure_details
    )


def test_rejects_unsupported_overall_answer() -> None:
    """The overall answer must be supported by its cited evidence union."""
    report = (
        AgentDynamicSynthesisGroundednessEvaluationService().evaluate(
            answer=_build_answer(
                answer_text=(
                    "Delete the production database and rotate all secrets."
                )
            ),
            execution_result=_build_execution_result(),
        )
    )

    assert report.passed is False
    assert report.grounded_citations == 1
    assert report.answer_grounded is False
    assert any(
        failure.reason == "unsupported_answer"
        for failure in report.failure_details
    )


def test_rejects_uncited_answer_when_evidence_exists() -> None:
    """Evidence-backed answers must not omit citations."""
    answer = AgentDynamicAnswer(
        answer="Restore deployment version 41.",
        confidence=0.8,
        requires_human_review=False,
    )

    report = (
        AgentDynamicSynthesisGroundednessEvaluationService().evaluate(
            answer=answer,
            execution_result=_build_execution_result(),
        )
    )

    assert report.passed is False
    assert report.total_citations == 0
    assert report.answer_grounded is False
    assert report.failure_details[0].reason == "missing_citation"
