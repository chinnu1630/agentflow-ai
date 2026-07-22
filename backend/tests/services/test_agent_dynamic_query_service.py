"""Tests for bounded dynamic query orchestration."""

from uuid import UUID

import pytest

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
from app.services.agent_execution_planner_service import (
    AgentExecutionPlannerResult,
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


@pytest.mark.anyio
async def test_executes_read_only_dynamic_pipeline() -> None:
    """The service should preserve metadata and disable side effects."""
    plan = _build_execution_plan()
    planner = FakeDynamicPlanner(plan)
    executor = FakeDynamicExecutor(plan)
    service = AgentDynamicQueryService(
        planner=planner,
        executor=executor,
        request_id="request-123",
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
    assert executor.has_release_run_context is False
    assert executor.allow_side_effects is False
    assert executor.human_approval_granted is False
    assert response.execution_plan == plan
    assert response.execution_result.status is (
        AgentExecutionStatus.SUCCESS
    )
    assert response.prompt_version == "agent-execution-planner-v1"


@pytest.mark.anyio
async def test_forwards_release_context_availability() -> None:
    """Trusted release-run context should be passed to validation."""
    plan = _build_execution_plan()
    planner = FakeDynamicPlanner(plan)
    executor = FakeDynamicExecutor(plan)
    service = AgentDynamicQueryService(
        planner=planner,
        executor=executor,
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
