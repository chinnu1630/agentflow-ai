"""Tests for deterministic dynamic-agent evaluation."""

import pytest

from app.schemas.agent_execution_plan import (
    AgentExecutionPlan,
    AgentExecutionStep,
)
from app.schemas.agent_query import AgentIntent, AgentQueryPlan, AgentQueryRequest
from app.schemas.agent_tool import AgentToolInvocation, AgentToolName
from app.services.agent_dynamic_evaluation_service import (
    DynamicAgentEvaluationService,
)
from app.services.agent_query_router import AgentQueryRouter
from tests.fixtures.agent_dynamic_eval_cases import (
    build_dynamic_agent_eval_cases,
)


class GoldenPlanner:
    """Return the expected minimal read-only tool for routed intents."""

    _TOOLS = {
        AgentIntent.KNOWLEDGE_DOC_QUESTION: (
            AgentToolName.SEARCH_ENGINEERING_KNOWLEDGE
        ),
        AgentIntent.GITHUB_PR_QUESTION: (
            AgentToolName.LOOKUP_GITHUB_PULL_REQUEST
        ),
        AgentIntent.JIRA_TICKET_QUESTION: (
            AgentToolName.LOOKUP_JIRA_ISSUE
        ),
        AgentIntent.APPROVAL_STATUS_QUESTION: (
            AgentToolName.LOOKUP_APPROVAL_STATUS
        ),
        AgentIntent.SLACK_STATUS_QUESTION: (
            AgentToolName.LOOKUP_SLACK_STATUS
        ),
    }

    async def create_plan(
        self,
        *,
        request: AgentQueryRequest,
        query_plan: AgentQueryPlan,
    ) -> AgentExecutionPlan:
        """Create one deterministic minimal plan."""
        del request

        tool_name = self._TOOLS[query_plan.intent]
        step_id = "evaluate_tool"

        return AgentExecutionPlan(
            objective="Evaluate the routed AgentFlow request.",
            intent=query_plan.intent,
            response_depth=query_plan.response_depth,
            steps=[
                AgentExecutionStep(
                    step_id=step_id,
                    invocation=AgentToolInvocation(
                        step_id=step_id,
                        tool_name=tool_name,
                        arguments={},
                        timeout_seconds=10,
                    ),
                )
            ],
            plan_reason_code="evaluate_golden_case",
        )


class WrongToolPlanner(GoldenPlanner):
    """Return an incorrect but read-only tool for failure testing."""

    async def create_plan(
        self,
        *,
        request: AgentQueryRequest,
        query_plan: AgentQueryPlan,
    ) -> AgentExecutionPlan:
        """Return a mismatched tool plan."""
        del request

        step_id = "wrong_tool"
        return AgentExecutionPlan(
            objective="Return an intentionally incorrect tool.",
            intent=query_plan.intent,
            response_depth=query_plan.response_depth,
            steps=[
                AgentExecutionStep(
                    step_id=step_id,
                    invocation=AgentToolInvocation(
                        step_id=step_id,
                        tool_name=AgentToolName.LOOKUP_SLACK_STATUS,
                        arguments={},
                        timeout_seconds=10,
                    ),
                )
            ],
            plan_reason_code="evaluate_wrong_tool",
        )


@pytest.mark.anyio
async def test_evaluates_golden_dynamic_agent_cases() -> None:
    """Golden cases should satisfy all deterministic quality metrics."""
    service = DynamicAgentEvaluationService(
        router=AgentQueryRouter(),
        planner=GoldenPlanner(),
    )

    report = await service.evaluate(build_dynamic_agent_eval_cases())

    assert report.total_cases == 7
    assert report.passed_cases == 7
    assert report.failed_cases == 0
    assert report.routing_accuracy == 1.0
    assert report.tool_accuracy == 1.0
    assert report.safety_accuracy == 1.0
    assert report.overall_accuracy == 1.0


@pytest.mark.anyio
async def test_failure_details_exclude_raw_manager_query() -> None:
    """Evaluation failures must remain safe for CI logs."""
    case = build_dynamic_agent_eval_cases()[0]
    service = DynamicAgentEvaluationService(
        router=AgentQueryRouter(),
        planner=WrongToolPlanner(),
    )

    report = await service.evaluate([case])

    assert report.failed_cases == 1
    failure = report.failed_case_details[0]
    assert failure.case_name == "knowledge_runbook_question"
    assert failure.reason == "tool_mismatch"
    assert failure.query_length == len(case.query)
    assert case.query not in str(failure.model_dump())
