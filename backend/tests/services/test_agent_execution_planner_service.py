"""Tests for bounded AgentFlow execution-planner orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import pytest

from app.integrations.anthropic_execution_planner_client import (
    ClaudeExecutionPlanResult,
)
from app.schemas.agent_execution_plan import (
    AgentExecutionPlan,
    AgentExecutionStep,
)
from app.schemas.agent_query import (
    AgentIntent,
    AgentQueryPlan,
    AgentQueryRequest,
    ResponseDepth,
)
from app.schemas.agent_tool import (
    AgentToolInvocation,
    AgentToolName,
)
from app.services.agent_execution_plan_validator import (
    AgentExecutionContextRequiredError,
    AgentExecutionPlanValidator,
    AgentExecutionToolNotAllowedError,
)
from app.services.agent_execution_planner_service import (
    AgentExecutionPlanIntentMismatchError,
    AgentExecutionPlannerService,
)
from app.services.agent_tool_registry import AgentToolRegistry


@dataclass
class FakePlannerClient:
    """Return one predefined structured Claude planning result."""

    result: ClaudeExecutionPlanResult
    call_count: int = 0

    async def create_execution_plan(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        prompt_version: str,
    ) -> ClaudeExecutionPlanResult:
        """Return the predefined result while recording invocation count."""
        assert system_prompt
        assert user_prompt
        assert prompt_version == "agent-execution-planner-v1"

        self.call_count += 1
        return self.result


def _build_query_plan(
    *,
    intent: AgentIntent = AgentIntent.KNOWLEDGE_DOC_QUESTION,
    release_run_id: UUID | None = None,
) -> AgentQueryPlan:
    """Create a trusted deterministic routing plan."""
    return AgentQueryPlan(
        intent=intent,
        response_depth=ResponseDepth.STANDARD,
        confidence=0.98,
        release_run_id=release_run_id,
        requires_current_snapshot=release_run_id is not None,
        routing_reason_code="matched_test_intent",
    )


def _build_execution_plan(
    *,
    intent: AgentIntent = AgentIntent.KNOWLEDGE_DOC_QUESTION,
    tool_name: AgentToolName = (
        AgentToolName.SEARCH_ENGINEERING_KNOWLEDGE
    ),
    timeout_seconds: int = 30,
) -> AgentExecutionPlan:
    """Create one reusable planner-generated execution plan."""
    step_id = "execute_tool"

    return AgentExecutionPlan(
        objective="Answer the manager's release workflow question.",
        intent=intent,
        response_depth=ResponseDepth.STANDARD,
        steps=[
            AgentExecutionStep(
                step_id=step_id,
                invocation=AgentToolInvocation(
                    step_id=step_id,
                    tool_name=tool_name,
                    arguments={"query": "payment rollback"},
                    timeout_seconds=timeout_seconds,
                ),
            )
        ],
        plan_reason_code="execute_approved_tool",
    )


def _build_claude_result(
    plan: AgentExecutionPlan,
) -> ClaudeExecutionPlanResult:
    """Wrap one execution plan in safe Claude usage metadata."""
    return ClaudeExecutionPlanResult(
        plan=plan,
        message_id="msg_plan_test",
        model="test-claude-model",
        input_tokens=250,
        output_tokens=100,
        stop_reason="end_turn",
        duration_ms=25.5,
        prompt_version="agent-execution-planner-v1",
    )


def _build_service(
    planner_client: FakePlannerClient,
) -> AgentExecutionPlannerService:
    """Create the planner service with trusted deterministic policy."""
    return AgentExecutionPlannerService(
        planner_client=planner_client,
        plan_validator=AgentExecutionPlanValidator(
            registry=AgentToolRegistry(),
            request_id="request-plan-test",
        ),
        request_id="request-plan-test",
    )


@pytest.mark.anyio
async def test_returns_policy_validated_execution_plan() -> None:
    """A valid read-only Claude plan should pass deterministic policy."""
    execution_plan = _build_execution_plan()
    planner_client = FakePlannerClient(
        result=_build_claude_result(execution_plan)
    )
    service = _build_service(planner_client)

    result = await service.create_plan(
        request=AgentQueryRequest(
            query="How do I rollback the payment service?"
        ),
        query_plan=_build_query_plan(),
    )

    assert result.plan is execution_plan
    assert result.model == "test-claude-model"
    assert result.input_tokens == 250
    assert result.output_tokens == 100
    assert planner_client.call_count == 1


@pytest.mark.anyio
async def test_rejects_claude_intent_change() -> None:
    """Claude must not override the deterministic routed intent."""
    execution_plan = _build_execution_plan(
        intent=AgentIntent.RELEASE_RISK_SUMMARY,
    )
    planner_client = FakePlannerClient(
        result=_build_claude_result(execution_plan)
    )
    service = _build_service(planner_client)

    with pytest.raises(
        AgentExecutionPlanIntentMismatchError,
        match="changed the deterministic intent",
    ):
        await service.create_plan(
            request=AgentQueryRequest(
                query="How do I rollback the payment service?"
            ),
            query_plan=_build_query_plan(
                intent=AgentIntent.KNOWLEDGE_DOC_QUESTION,
            ),
        )


@pytest.mark.anyio
async def test_rejects_side_effecting_planner_output() -> None:
    """Automatic planning must remain read-only even for valid tool names."""
    execution_plan = _build_execution_plan(
        intent=AgentIntent.ACTION_REQUEST,
        tool_name=AgentToolName.SEND_APPROVED_SLACK_ALERT,
    )
    planner_client = FakePlannerClient(
        result=_build_claude_result(execution_plan)
    )
    service = _build_service(planner_client)

    release_run_id = UUID(
        "22222222-2222-2222-2222-222222222222"
    )

    with pytest.raises(
        AgentExecutionToolNotAllowedError,
        match="disabled for this execution",
    ):
        await service.create_plan(
            request=AgentQueryRequest(
                query="Send the release report to Slack.",
                release_run_id=release_run_id,
            ),
            query_plan=_build_query_plan(
                intent=AgentIntent.ACTION_REQUEST,
                release_run_id=release_run_id,
            ),
        )


@pytest.mark.anyio
async def test_rejects_context_tool_without_release_run_context() -> None:
    """Claude cannot select context-dependent tools without trusted context."""
    execution_plan = _build_execution_plan(
        intent=AgentIntent.APPROVAL_STATUS_QUESTION,
        tool_name=AgentToolName.LOOKUP_APPROVAL_STATUS,
        timeout_seconds=10,
    )
    planner_client = FakePlannerClient(
        result=_build_claude_result(execution_plan)
    )
    service = _build_service(planner_client)

    with pytest.raises(
        AgentExecutionContextRequiredError,
        match="requires release-run context",
    ):
        await service.create_plan(
            request=AgentQueryRequest(
                query="Is the release approved?"
            ),
            query_plan=_build_query_plan(
                intent=AgentIntent.APPROVAL_STATUS_QUESTION,
            ),
        )


@pytest.mark.anyio
async def test_accepts_context_tool_with_release_run_context() -> None:
    """Trusted release context permits registered context-dependent tools."""
    release_run_id = UUID(
        "11111111-1111-1111-1111-111111111111"
    )
    execution_plan = _build_execution_plan(
        intent=AgentIntent.APPROVAL_STATUS_QUESTION,
        tool_name=AgentToolName.LOOKUP_APPROVAL_STATUS,
        timeout_seconds=10,
    )
    planner_client = FakePlannerClient(
        result=_build_claude_result(execution_plan)
    )
    service = _build_service(planner_client)

    result = await service.create_plan(
        request=AgentQueryRequest(
            query="Is the release approved?",
            release_run_id=release_run_id,
        ),
        query_plan=_build_query_plan(
            intent=AgentIntent.APPROVAL_STATUS_QUESTION,
            release_run_id=release_run_id,
        ),
    )

    assert result.plan.steps[0].invocation.tool_name is (
        AgentToolName.LOOKUP_APPROVAL_STATUS
    )
