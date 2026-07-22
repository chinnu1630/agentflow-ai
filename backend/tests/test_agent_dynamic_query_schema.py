"""Tests for bounded dynamic agent API response contracts."""

import pytest
from pydantic import ValidationError

from app.schemas.agent_dynamic_query import AgentDynamicQueryResponse
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
    ResponseDepth,
)
from app.schemas.agent_tool import (
    AgentToolExecutionStatus,
    AgentToolInvocation,
    AgentToolName,
    AgentToolResult,
)


def test_accepts_auditable_dynamic_query_response() -> None:
    """The response should combine routing, planning, and execution metadata."""
    query_plan = AgentQueryPlan(
        intent=AgentIntent.KNOWLEDGE_DOC_QUESTION,
        response_depth=ResponseDepth.STANDARD,
        confidence=0.98,
        routing_reason_code="matched_knowledge_question",
    )
    execution_plan = AgentExecutionPlan(
        objective="Find trusted payment rollback guidance.",
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
    execution_result = AgentExecutionResult(
        intent=AgentIntent.KNOWLEDGE_DOC_QUESTION,
        objective=execution_plan.objective,
        plan_reason_code=execution_plan.plan_reason_code,
        status=AgentExecutionStatus.SUCCESS,
        tool_results=[
            AgentToolResult(
                step_id="search_knowledge",
                tool_name=(
                    AgentToolName.SEARCH_ENGINEERING_KNOWLEDGE
                ),
                status=AgentToolExecutionStatus.SUCCESS,
                output={"result_count": 1},
                duration_ms=12,
            )
        ],
        requires_synthesis=True,
        duration_ms=15,
    )

    response = AgentDynamicQueryResponse(
        query_plan=query_plan,
        execution_plan=execution_plan,
        execution_result=execution_result,
        prompt_version="agent-execution-planner-v1",
        model="test-claude-model",
        message_id="msg_dynamic_123",
        input_tokens=250,
        output_tokens=100,
        planning_duration_ms=25.5,
        answer=AgentDynamicAnswer(
            answer="Follow the trusted payment rollback procedure.",
            confidence=0.94,
            requires_human_review=False,
        ),
        synthesis_prompt_version="agent-dynamic-synthesis-v1",
        synthesis_model="test-claude-model",
        synthesis_message_id="msg_synthesis_123",
        synthesis_input_tokens=300,
        synthesis_output_tokens=120,
        synthesis_duration_ms=20.5,
    )

    assert response.query_plan.intent is (
        AgentIntent.KNOWLEDGE_DOC_QUESTION
    )
    assert response.execution_plan.steps[0].step_id == (
        "search_knowledge"
    )
    assert response.execution_result.status is (
        AgentExecutionStatus.SUCCESS
    )
    assert response.input_tokens == 250



def test_rejects_mismatched_execution_intent() -> None:
    """The API response must preserve the deterministic routed intent."""
    query_plan = AgentQueryPlan(
        intent=AgentIntent.KNOWLEDGE_DOC_QUESTION,
        response_depth=ResponseDepth.STANDARD,
        confidence=0.98,
        routing_reason_code="matched_knowledge_question",
    )
    execution_plan = AgentExecutionPlan(
        objective="Find trusted payment rollback guidance.",
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
    execution_result = AgentExecutionResult(
        intent=AgentIntent.APPROVAL_STATUS_QUESTION,
        objective=execution_plan.objective,
        plan_reason_code=execution_plan.plan_reason_code,
        status=AgentExecutionStatus.SUCCESS,
        tool_results=[
            AgentToolResult(
                step_id="search_knowledge",
                tool_name=(
                    AgentToolName.SEARCH_ENGINEERING_KNOWLEDGE
                ),
                status=AgentToolExecutionStatus.SUCCESS,
                output={"result_count": 1},
                duration_ms=12,
            )
        ],
        requires_synthesis=True,
        duration_ms=15,
    )

    with pytest.raises(
        ValidationError,
        match="execution result intent must match",
    ):
        AgentDynamicQueryResponse(
            query_plan=query_plan,
            execution_plan=execution_plan,
            execution_result=execution_result,
            prompt_version="agent-execution-planner-v1",
            model="test-claude-model",
            message_id="msg_dynamic_123",
            input_tokens=250,
            output_tokens=100,
            planning_duration_ms=25.5,
            answer=AgentDynamicAnswer(
                answer="Follow the trusted payment rollback procedure.",
                confidence=0.94,
                requires_human_review=False,
            ),
            synthesis_prompt_version="agent-dynamic-synthesis-v1",
            synthesis_model="test-claude-model",
            synthesis_message_id="msg_synthesis_123",
            synthesis_input_tokens=300,
            synthesis_output_tokens=120,
            synthesis_duration_ms=20.5,
        )
