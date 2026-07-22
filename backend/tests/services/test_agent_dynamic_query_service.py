"""Tests for bounded dynamic query orchestration."""

from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from uuid import UUID

import pytest

from app.integrations.anthropic_dynamic_synthesis_client import (
    ClaudeDynamicSynthesisResult,
)
from app.schemas.agent_dynamic_synthesis import AgentDynamicAnswer
from app.schemas.agent_execution_plan import (
    AgentExecutionPlan,
    AgentExecutionStep,
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
    AgentToolExecutionStatus,
    AgentToolInvocation,
    AgentToolName,
    AgentToolResult,
)
from app.services.agent_dynamic_query_service import (
    AgentDynamicQueryService,
)
from app.services.agent_dynamic_synthesis_citation_verifier import (
    AgentDynamicSynthesisCitationVerificationError,
)
from app.services.agent_execution_planner_service import (
    AgentExecutionPlannerResult,
)
from app.services.agent_llm_cost_estimator import (
    AgentLLMCostEstimator,
    AgentLLMCostRates,
)


def _build_execution_plan() -> AgentExecutionPlan:
    """Create one reusable read-only execution plan."""
    return AgentExecutionPlan(
        objective="Search trusted payment rollback guidance.",
        intent=AgentIntent.KNOWLEDGE_DOC_QUESTION,
        response_depth=ResponseDepth.STANDARD,
        steps=[
            AgentExecutionStep(
                step_id="search_knowledge",
                invocation=AgentToolInvocation(
                    step_id="search_knowledge",
                    tool_name=(
                        AgentToolName.SEARCH_ENGINEERING_KNOWLEDGE
                    ),
                    arguments={"query": "payment rollback"},
                    timeout_seconds=30,
                ),
            )
        ],
        plan_reason_code="search_engineering_knowledge",
    )


class FakeDynamicPlanner:
    """Return a deterministic planner result."""

    def __init__(self, plan: AgentExecutionPlan) -> None:
        self._plan = plan
        self.call_count = 0

    async def create_plan(
        self,
        *,
        request: AgentQueryRequest,
        query_plan: AgentQueryPlan,
    ) -> AgentExecutionPlannerResult:
        """Return safe planner metadata."""
        del request, query_plan
        self.call_count += 1

        return AgentExecutionPlannerResult(
            plan=self._plan,
            prompt_version="agent-execution-planner-v1",
            model="test-claude-model",
            message_id="msg_dynamic_123",
            input_tokens=250,
            output_tokens=100,
            duration_ms=25.5,
        )


class FakeDynamicExecutor:
    """Return a deterministic execution result."""

    def __init__(self, plan: AgentExecutionPlan) -> None:
        self._plan = plan
        self.call_count = 0
        self.has_release_run_context: bool | None = None
        self.allow_side_effects: bool | None = None
        self.human_approval_granted: bool | None = None

    async def execute(
        self,
        plan: AgentExecutionPlan,
        *,
        has_release_run_context: bool,
        allow_side_effects: bool = False,
        human_approval_granted: bool = False,
    ) -> AgentExecutionResult:
        """Capture policy arguments and return one successful result."""
        assert plan == self._plan

        self.call_count += 1
        self.has_release_run_context = has_release_run_context
        self.allow_side_effects = allow_side_effects
        self.human_approval_granted = human_approval_granted

        return AgentExecutionResult(
            intent=plan.intent,
            objective=plan.objective,
            plan_reason_code=plan.plan_reason_code,
            status=AgentExecutionStatus.SUCCESS,
            tool_results=[
                AgentToolResult(
                    step_id="search_knowledge",
                    tool_name=(
                        AgentToolName.SEARCH_ENGINEERING_KNOWLEDGE
                    ),
                    status=AgentToolExecutionStatus.SUCCESS,
                    output={"result_count": 1},
                    duration_ms=10,
                )
            ],
            requires_synthesis=True,
            duration_ms=12,
        )



class FakeDynamicSynthesizer:
    """Return one deterministic evidence-grounded answer."""

    def __init__(self) -> None:
        self.call_count = 0

    async def synthesize(
        self,
        *,
        request: AgentQueryRequest,
        query_plan: AgentQueryPlan,
        execution_result: AgentExecutionResult,
    ) -> ClaudeDynamicSynthesisResult:
        """Return synthesis metadata for the executed result."""
        del request

        assert query_plan.intent is execution_result.intent
        self.call_count += 1

        return ClaudeDynamicSynthesisResult(
            answer=AgentDynamicAnswer(
                answer="Follow the trusted payment rollback procedure.",
                confidence=0.94,
                requires_human_review=False,
            ),
            message_id="msg-synthesis-123",
            model="test-claude-model",
            input_tokens=300,
            output_tokens=120,
            stop_reason="end_turn",
            duration_ms=20.5,
            prompt_version="agent-dynamic-synthesis-v1",
        )


@pytest.mark.anyio
async def test_executes_read_only_dynamic_pipeline() -> None:
    """The service should preserve metadata and disable side effects."""
    plan = _build_execution_plan()
    planner = FakeDynamicPlanner(plan)
    executor = FakeDynamicExecutor(plan)
    synthesizer = FakeDynamicSynthesizer()
    service = AgentDynamicQueryService(
        planner=planner,
        executor=executor,
        synthesizer=synthesizer,
        request_id="request-123",
        cost_estimator=AgentLLMCostEstimator(
            rates=AgentLLMCostRates(
                planning_input_per_million_usd=Decimal("3"),
                planning_output_per_million_usd=Decimal("15"),
                synthesis_input_per_million_usd=Decimal("3"),
                synthesis_output_per_million_usd=Decimal("15"),
            )
        ),
    )
    request = AgentQueryRequest(
        query="How do I rollback the payment service?"
    )
    query_plan = AgentQueryPlan(
        intent=AgentIntent.KNOWLEDGE_DOC_QUESTION,
        response_depth=ResponseDepth.STANDARD,
        confidence=0.98,
        routing_reason_code="matched_knowledge_question",
    )

    response = await service.execute(
        request=request,
        query_plan=query_plan,
    )

    assert planner.call_count == 1
    assert executor.call_count == 1
    assert synthesizer.call_count == 1
    assert executor.has_release_run_context is False
    assert executor.allow_side_effects is False
    assert executor.human_approval_granted is False
    assert response.execution_plan == plan
    assert response.execution_result.status is (
        AgentExecutionStatus.SUCCESS
    )
    assert response.prompt_version == "agent-execution-planner-v1"
    assert response.answer.confidence == 0.94
    assert response.synthesis_prompt_version == (
        "agent-dynamic-synthesis-v1"
    )
    assert response.cost_estimate.total_cost_usd == Decimal(
        "0.004950"
    )


@pytest.mark.anyio
async def test_forwards_release_context_availability() -> None:
    """Trusted release-run context should be passed to validation."""
    plan = _build_execution_plan()
    planner = FakeDynamicPlanner(plan)
    executor = FakeDynamicExecutor(plan)
    synthesizer = FakeDynamicSynthesizer()
    service = AgentDynamicQueryService(
        planner=planner,
        executor=executor,
        synthesizer=synthesizer,
        request_id="request-456",
    )
    release_run_id = UUID("11111111-1111-1111-1111-111111111111")
    request = AgentQueryRequest(
        query="Explain the current release.",
        release_run_id=release_run_id,
    )
    query_plan = AgentQueryPlan(
        intent=AgentIntent.KNOWLEDGE_DOC_QUESTION,
        response_depth=ResponseDepth.STANDARD,
        confidence=0.95,
        release_run_id=release_run_id,
        routing_reason_code="matched_knowledge_question",
    )

    await service.execute(
        request=request,
        query_plan=query_plan,
    )

    assert executor.has_release_run_context is True



class FakeRejectedDynamicSynthesizer:
    """Reject synthesized output that violates grounding policy."""

    async def synthesize(
        self,
        *,
        request: AgentQueryRequest,
        query_plan: AgentQueryPlan,
        execution_result: AgentExecutionResult,
    ) -> ClaudeDynamicSynthesisResult:
        """Raise the verifier error after tool execution."""
        del request, query_plan, execution_result

        raise AgentDynamicSynthesisCitationVerificationError(
            "Untrusted Claude output must not be exposed."
        )


@pytest.mark.anyio
async def test_logs_safe_audit_metadata_when_synthesis_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Grounding rejection should preserve safe execution audit metadata."""
    plan = _build_execution_plan()
    planner = FakeDynamicPlanner(plan)
    executor = FakeDynamicExecutor(plan)
    logged_events: list[tuple[str, dict[str, object]]] = []

    def capture_error(event: str, **metadata: object) -> None:
        """Capture structured error metadata without logging answer content."""
        logged_events.append((event, metadata))

    monkeypatch.setattr(
        "app.services.agent_dynamic_query_service.logger.error",
        capture_error,
    )

    service = AgentDynamicQueryService(
        planner=planner,
        executor=executor,
        synthesizer=FakeRejectedDynamicSynthesizer(),
        request_id="request-grounding-rejected",
    )
    request = AgentQueryRequest(
        query="How do I rollback the payment service?"
    )
    query_plan = AgentQueryPlan(
        intent=AgentIntent.KNOWLEDGE_DOC_QUESTION,
        response_depth=ResponseDepth.STANDARD,
        confidence=0.98,
        routing_reason_code="matched_knowledge_question",
    )

    with pytest.raises(
        AgentDynamicSynthesisCitationVerificationError,
        match="Untrusted Claude output",
    ):
        await service.execute(
            request=request,
            query_plan=query_plan,
        )

    assert len(logged_events) == 1
    event, metadata = logged_events[0]
    assert event == "agent_dynamic_synthesis_rejected"
    assert metadata["run_id"] == "request-grounding-rejected"
    assert metadata["intent"] == "knowledge_doc_question"
    assert metadata["execution_status"] == "success"
    assert metadata["step_count"] == 1
    assert metadata["error_type"] == (
        "AgentDynamicSynthesisCitationVerificationError"
    )
    assert "Untrusted Claude output" not in str(metadata)



class CapturingSpan:
    """Capture safe attributes and status assigned to a business span."""

    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}
        self.status: object | None = None

    def set_attribute(self, key: str, value: object) -> None:
        """Store one safe scalar span attribute."""
        self.attributes[key] = value

    def set_status(self, status: object) -> None:
        """Store the final span status."""
        self.status = status


@pytest.mark.anyio
async def test_dynamic_pipeline_span_records_safe_success_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pipeline span should contain only safe operational metadata."""
    captured_span_name: str | None = None
    span = CapturingSpan()

    @contextmanager
    def capture_business_span(
        span_name: str,
        attributes: dict[str, object],
    ) -> Iterator[CapturingSpan]:
        """Capture span creation without exporting a real trace."""
        nonlocal captured_span_name
        captured_span_name = span_name
        span.attributes.update(attributes)
        yield span

    monkeypatch.setattr(
        "app.services.agent_dynamic_query_service.start_business_span",
        capture_business_span,
    )

    plan = _build_execution_plan()
    service = AgentDynamicQueryService(
        planner=FakeDynamicPlanner(plan),
        executor=FakeDynamicExecutor(plan),
        synthesizer=FakeDynamicSynthesizer(),
        request_id="request-trace-success",
    )

    response = await service.execute(
        request=AgentQueryRequest(
            query="How do I rollback the payment service?"
        ),
        query_plan=AgentQueryPlan(
            intent=AgentIntent.KNOWLEDGE_DOC_QUESTION,
            response_depth=ResponseDepth.STANDARD,
            confidence=0.98,
            routing_reason_code="matched_knowledge_question",
        ),
    )

    assert captured_span_name == "agent.dynamic_query_pipeline"
    assert span.attributes == {
        "run_id": "request-trace-success",
        "intent": "knowledge_doc_question",
        "execution_id": str(response.execution_result.execution_id),
        "execution_status": "success",
        "step_count": 1,
        "citation_count": 0,
        "requires_human_review": False,
        "estimated_cost_usd": "0.000000",
    }
    assert "query" not in span.attributes
    assert "answer" not in span.attributes
    assert "tool_results" not in span.attributes


@pytest.mark.anyio
async def test_dynamic_pipeline_span_records_safe_grounding_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Grounding rejection should be traceable without exception content."""
    captured_span_name: str | None = None
    span = CapturingSpan()

    @contextmanager
    def capture_business_span(
        span_name: str,
        attributes: dict[str, object],
    ) -> Iterator[CapturingSpan]:
        """Capture span creation without exporting a real trace."""
        nonlocal captured_span_name
        captured_span_name = span_name
        span.attributes.update(attributes)
        yield span

    monkeypatch.setattr(
        "app.services.agent_dynamic_query_service.start_business_span",
        capture_business_span,
    )

    plan = _build_execution_plan()
    service = AgentDynamicQueryService(
        planner=FakeDynamicPlanner(plan),
        executor=FakeDynamicExecutor(plan),
        synthesizer=FakeRejectedDynamicSynthesizer(),
        request_id="request-trace-grounding-failure",
    )

    with pytest.raises(AgentDynamicSynthesisCitationVerificationError):
        await service.execute(
            request=AgentQueryRequest(
                query="How do I rollback the payment service?"
            ),
            query_plan=AgentQueryPlan(
                intent=AgentIntent.KNOWLEDGE_DOC_QUESTION,
                response_depth=ResponseDepth.STANDARD,
                confidence=0.98,
                routing_reason_code="matched_knowledge_question",
            ),
        )

    assert captured_span_name == "agent.dynamic_query_pipeline"
    assert span.attributes["run_id"] == (
        "request-trace-grounding-failure"
    )
    assert span.attributes["intent"] == "knowledge_doc_question"
    assert span.attributes["execution_status"] == "success"
    assert span.attributes["failure_stage"] == "grounding_verification"
    assert span.attributes["exception_type"] == (
        "AgentDynamicSynthesisCitationVerificationError"
    )
    assert span.status is not None
    assert "Untrusted Claude output" not in str(span.attributes)
    assert "Untrusted Claude output" not in str(span.status)
