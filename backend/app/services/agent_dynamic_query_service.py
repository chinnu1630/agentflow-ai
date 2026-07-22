"""Coordinate bounded planning and read-only dynamic execution."""

from __future__ import annotations

from typing import Protocol

import structlog

from app.schemas.agent_dynamic_query import AgentDynamicQueryResponse
from app.schemas.agent_execution_plan import AgentExecutionPlan
from app.schemas.agent_execution_result import AgentExecutionResult
from app.schemas.agent_query import AgentQueryPlan, AgentQueryRequest
from app.services.agent_execution_planner_service import (
    AgentExecutionPlannerResult,
)

logger = structlog.get_logger(__name__)


class DynamicPlannerProtocol(Protocol):
    """Planning capability required by the dynamic query service."""

    async def create_plan(
        self,
        *,
        request: AgentQueryRequest,
        query_plan: AgentQueryPlan,
    ) -> AgentExecutionPlannerResult:
        """Create one policy-validated execution plan."""

        ...


class DynamicExecutorProtocol(Protocol):
    """Execution capability required by the dynamic query service."""

    async def execute(
        self,
        plan: AgentExecutionPlan,
        *,
        has_release_run_context: bool,
        allow_side_effects: bool = False,
        human_approval_granted: bool = False,
    ) -> AgentExecutionResult:
        """Execute one validated read-only plan."""

        ...


class AgentDynamicQueryService:
    """Run the complete bounded dynamic query pipeline."""

    def __init__(
        self,
        *,
        planner: DynamicPlannerProtocol,
        executor: DynamicExecutorProtocol,
        request_id: str,
    ) -> None:
        """Initialize the dynamic query orchestration service."""
        self._planner = planner
        self._executor = executor
        self._request_id = request_id

    async def execute(
        self,
        *,
        request: AgentQueryRequest,
        query_plan: AgentQueryPlan,
    ) -> AgentDynamicQueryResponse:
        """Plan and execute one read-only dynamic manager query."""
        planner_result = await self._planner.create_plan(
            request=request,
            query_plan=query_plan,
        )
        has_release_run_context = (
            request.release_run_id is not None
            or query_plan.release_run_id is not None
        )

        execution_result = await self._executor.execute(
            planner_result.plan,
            has_release_run_context=has_release_run_context,
            allow_side_effects=False,
            human_approval_granted=False,
        )

        response = AgentDynamicQueryResponse(
            query_plan=query_plan,
            execution_plan=planner_result.plan,
            execution_result=execution_result,
            prompt_version=planner_result.prompt_version,
            model=planner_result.model,
            message_id=planner_result.message_id,
            input_tokens=planner_result.input_tokens,
            output_tokens=planner_result.output_tokens,
            planning_duration_ms=planner_result.duration_ms,
        )

        logger.info(
            "agent_dynamic_query_completed",
            run_id=self._request_id,
            intent=query_plan.intent.value,
            execution_id=str(execution_result.execution_id),
            execution_status=execution_result.status.value,
            step_count=len(execution_result.tool_results),
            prompt_version=planner_result.prompt_version,
            model=planner_result.model,
            input_tokens=planner_result.input_tokens,
            output_tokens=planner_result.output_tokens,
        )

        return response
