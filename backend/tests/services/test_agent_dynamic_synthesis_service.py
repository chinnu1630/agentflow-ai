"""Tests for dynamic answer synthesis orchestration."""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from app.integrations.anthropic_dynamic_synthesis_client import (
    ClaudeDynamicSynthesisResult,
)
from app.schemas.agent_dynamic_synthesis import (
    AgentDynamicAnswer,
    AgentDynamicAnswerCitation,
)
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
from app.services.agent_dynamic_synthesis_citation_verifier import (
    AgentDynamicSynthesisCitationVerificationError,
)
from app.services.agent_dynamic_synthesis_service import (
    AgentDynamicSynthesisService,
)


class FakeDynamicSynthesisClient:
    """Return one deterministic Claude synthesis result."""

    def __init__(self, answer: AgentDynamicAnswer) -> None:
        self._answer = answer
        self.call_count = 0
        self.prompt_version: str | None = None

    async def synthesize_dynamic_answer(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        prompt_version: str,
    ) -> ClaudeDynamicSynthesisResult:
        """Capture bounded prompts and return the configured answer."""
        assert system_prompt
        assert user_prompt

        self.call_count += 1
        self.prompt_version = prompt_version

        return ClaudeDynamicSynthesisResult(
            answer=self._answer,
            message_id="msg-synthesis-123",
            model="test-claude-model",
            input_tokens=300,
            output_tokens=120,
            stop_reason="end_turn",
            duration_ms=20.5,
            prompt_version=prompt_version,
        )


def _build_execution_result() -> AgentExecutionResult:
    """Build one successful execution with trusted evidence."""
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


def _build_query_plan() -> AgentQueryPlan:
    """Build the matching deterministic routing plan."""
    return AgentQueryPlan(
        intent=AgentIntent.KNOWLEDGE_DOC_QUESTION,
        response_depth=ResponseDepth.STANDARD,
        confidence=0.98,
        routing_reason_code="matched_knowledge_question",
    )


@pytest.mark.anyio
async def test_synthesizes_and_verifies_dynamic_answer() -> None:
    """Service should build prompts and verify returned citations."""
    answer = AgentDynamicAnswer(
        answer="Follow the documented payment rollback procedure.",
        confidence=0.94,
        citations=[
            AgentDynamicAnswerCitation(
                source_type="engineering_document_chunk",
                source_id="chunk-123",
                title="Payment Service Runbook",
                supporting_fact="The runbook defines rollback steps.",
            )
        ],
        requires_human_review=False,
    )
    client = FakeDynamicSynthesisClient(answer)
    service = AgentDynamicSynthesisService(
        client=client,
        request_id="request-123",
    )

    result = await service.synthesize(
        request=AgentQueryRequest(
            query="How do I rollback the payment service?"
        ),
        query_plan=_build_query_plan(),
        execution_result=_build_execution_result(),
    )

    assert client.call_count == 1
    assert client.prompt_version == "agent-dynamic-synthesis-v1"
    assert result.answer is answer
    assert result.message_id == "msg-synthesis-123"
    assert result.input_tokens == 300


@pytest.mark.anyio
async def test_fails_closed_when_trusted_evidence_is_not_cited() -> None:
    """Service must reject an uncited answer when evidence exists."""
    answer = AgentDynamicAnswer(
        answer="Follow the documented rollback procedure.",
        confidence=0.9,
        requires_human_review=False,
    )
    client = FakeDynamicSynthesisClient(answer)
    service = AgentDynamicSynthesisService(
        client=client,
        request_id="request-uncited",
    )

    with pytest.raises(
        AgentDynamicSynthesisCitationVerificationError,
        match="must include at least one verified citation",
    ):
        await service.synthesize(
            request=AgentQueryRequest(
                query="How do I rollback the payment service?"
            ),
            query_plan=_build_query_plan(),
            execution_result=_build_execution_result(),
        )

    assert client.call_count == 1


@pytest.mark.anyio
async def test_fails_closed_when_degraded_step_is_hidden() -> None:
    """Service must reject synthesis that hides a failed tool step."""
    execution_result = AgentExecutionResult(
        intent=AgentIntent.KNOWLEDGE_DOC_QUESTION,
        objective="Answer using available engineering evidence.",
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
    answer = AgentDynamicAnswer(
        answer="Use the documented rollback procedure.",
        confidence=0.6,
        citations=[
            AgentDynamicAnswerCitation(
                source_type="engineering_document_chunk",
                source_id="chunk-123",
                title="Payment Service Runbook",
                supporting_fact="The runbook contains rollback guidance.",
            )
        ],
        requires_human_review=True,
    )
    client = FakeDynamicSynthesisClient(answer)
    service = AgentDynamicSynthesisService(
        client=client,
        request_id="request-hidden-degradation",
    )

    with pytest.raises(
        AgentDynamicSynthesisCitationVerificationError,
        match="must exactly match degraded execution steps",
    ):
        await service.synthesize(
            request=AgentQueryRequest(
                query="Explain the release procedure."
            ),
            query_plan=_build_query_plan(),
            execution_result=execution_result,
        )

    assert client.call_count == 1



class CapturingSpan:
    """Capture dynamic synthesis span attributes and status."""

    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}
        self.status: object | None = None

    def set_attribute(self, key: str, value: object) -> None:
        """Store one safe span attribute."""
        self.attributes[key] = value

    def set_status(self, status: object) -> None:
        """Store the assigned span status."""
        self.status = status


@pytest.mark.anyio
async def test_synthesis_span_records_safe_usage_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthesis span should expose safe usage and grounding metadata."""
    captured_name: str | None = None
    span = CapturingSpan()

    @contextmanager
    def capture_span(
        span_name: str,
        attributes: dict[str, object],
    ) -> Iterator[CapturingSpan]:
        nonlocal captured_name
        captured_name = span_name
        span.attributes.update(attributes)
        yield span

    monkeypatch.setattr(
        "app.services.agent_dynamic_synthesis_service.start_business_span",
        capture_span,
    )

    answer = AgentDynamicAnswer(
        answer="Follow the documented payment rollback procedure.",
        confidence=0.94,
        citations=[
            AgentDynamicAnswerCitation(
                source_type="engineering_document_chunk",
                source_id="chunk-123",
                title="Payment Service Runbook",
                supporting_fact="The runbook defines rollback steps.",
            )
        ],
        requires_human_review=False,
    )
    execution_result = _build_execution_result()
    service = AgentDynamicSynthesisService(
        client=FakeDynamicSynthesisClient(answer),
        request_id="request-synthesis-trace",
    )

    await service.synthesize(
        request=AgentQueryRequest(
            query="How do I rollback the payment service?"
        ),
        query_plan=_build_query_plan(),
        execution_result=execution_result,
    )

    assert captured_name == "agent.dynamic_synthesis"
    assert span.attributes == {
        "run_id": "request-synthesis-trace",
        "intent": "knowledge_doc_question",
        "execution_id": str(execution_result.execution_id),
        "execution_status": "success",
        "step_count": 1,
        "prompt_version": "agent-dynamic-synthesis-v1",
        "model_name": "test-claude-model",
        "input_token_count": 300,
        "output_token_count": 120,
        "citation_count": 1,
        "requires_human_review": False,
    }
    assert "query" not in span.attributes
    assert "answer" not in span.attributes
    assert "tool_results" not in span.attributes


@pytest.mark.anyio
async def test_synthesis_span_records_safe_grounding_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Grounding failures should expose stage and type without content."""
    span = CapturingSpan()

    @contextmanager
    def capture_span(
        span_name: str,
        attributes: dict[str, object],
    ) -> Iterator[CapturingSpan]:
        assert span_name == "agent.dynamic_synthesis"
        span.attributes.update(attributes)
        yield span

    monkeypatch.setattr(
        "app.services.agent_dynamic_synthesis_service.start_business_span",
        capture_span,
    )

    answer = AgentDynamicAnswer(
        answer="Follow the documented rollback procedure.",
        confidence=0.9,
        requires_human_review=False,
    )
    service = AgentDynamicSynthesisService(
        client=FakeDynamicSynthesisClient(answer),
        request_id="request-synthesis-grounding-failure",
    )

    with pytest.raises(AgentDynamicSynthesisCitationVerificationError):
        await service.synthesize(
            request=AgentQueryRequest(
                query="How do I rollback the payment service?"
            ),
            query_plan=_build_query_plan(),
            execution_result=_build_execution_result(),
        )

    assert span.attributes["failure_stage"] == "grounding_verification"
    assert span.attributes["exception_type"] == (
        "AgentDynamicSynthesisCitationVerificationError"
    )
    assert span.attributes["execution_status"] == "success"
    assert span.status is not None
    assert "must include at least one verified citation" not in str(
        span.attributes
    )
    assert "must include at least one verified citation" not in str(
        span.status
    )
