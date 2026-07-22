"""Coordinate bounded planning and read-only dynamic execution."""

from __future__ import annotations

from typing import Protocol

import structlog

from app.integrations.anthropic_dynamic_synthesis_client import (
    ClaudeDynamicSynthesisResult,
)
from app.observability.tracing import (
    record_business_span_failure,
    set_safe_span_attributes,
    start_business_span,
)
from app.schemas.agent_dynamic_query import AgentDynamicQueryResponse
from app.schemas.agent_execution_plan import AgentExecutionPlan
from app.schemas.agent_execution_result import AgentExecutionResult
from app.schemas.agent_query import AgentQueryPlan, AgentQueryRequest
from app.services.agent_dynamic_synthesis_citation_verifier import (
    AgentDynamicSynthesisCitationVerificationError,
)
from app.services.agent_execution_planner_service import (
    AgentExecutionPlannerResult,
)
from app.services.agent_llm_cost_estimator import AgentLLMCostEstimator

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


class DynamicSynthesizerProtocol(Protocol):
    """Synthesis capability required by the dynamic query service."""

    async def synthesize(
        self,
        *,
        request: AgentQueryRequest,
        query_plan: AgentQueryPlan,
        execution_result: AgentExecutionResult,
    ) -> ClaudeDynamicSynthesisResult:
        """Produce one verified evidence-grounded manager answer."""

        ...


class AgentDynamicQueryService:
    """Run the complete bounded dynamic query pipeline."""

    def __init__(
        self,
        *,
        planner: DynamicPlannerProtocol,
        executor: DynamicExecutorProtocol,
        synthesizer: DynamicSynthesizerProtocol,
        request_id: str,
        cost_estimator: AgentLLMCostEstimator | None = None,
    ) -> None:
        """Initialize the dynamic query orchestration service."""
        self._planner = planner
        self._executor = executor
        self._synthesizer = synthesizer
        self._request_id = request_id
        self._cost_estimator = cost_estimator or AgentLLMCostEstimator()

    async def execute(
        self,
        *,
        request: AgentQueryRequest,
        query_plan: AgentQueryPlan,
    ) -> AgentDynamicQueryResponse:
        """Plan and execute one read-only dynamic manager query."""
        with start_business_span(
            "agent.dynamic_query_pipeline",
            {
                "run_id": self._request_id,
                "intent": query_plan.intent.value,
            },
        ) as span:
            try:
                planner_result = await self._planner.create_plan(
                    request=request,
                    query_plan=query_plan,
                )
            except Exception as exc:
                record_business_span_failure(
                    span,
                    failure_stage="dynamic_planning",
                    exception=exc,
                )
                raise

            has_release_run_context = (
                request.release_run_id is not None
                or query_plan.release_run_id is not None
            )

            try:
                execution_result = await self._executor.execute(
                    planner_result.plan,
                    has_release_run_context=has_release_run_context,
                    allow_side_effects=False,
                    human_approval_granted=False,
                )
            except Exception as exc:
                record_business_span_failure(
                    span,
                    failure_stage="tool_execution",
                    exception=exc,
                )
                raise

            set_safe_span_attributes(
                span,
                {
                    "execution_id": str(execution_result.execution_id),
                    "execution_status": execution_result.status.value,
                    "step_count": len(execution_result.tool_results),
                },
            )

            try:
                synthesis_result = await self._synthesizer.synthesize(
                    request=request,
                    query_plan=query_plan,
                    execution_result=execution_result,
                )
            except AgentDynamicSynthesisCitationVerificationError as exc:
                record_business_span_failure(
                    span,
                    failure_stage="grounding_verification",
                    exception=exc,
                    execution_status=execution_result.status.value,
                )
                logger.error(
                    "agent_dynamic_synthesis_rejected",
                    run_id=self._request_id,
                    intent=query_plan.intent.value,
                    execution_id=str(execution_result.execution_id),
                    execution_status=execution_result.status.value,
                    step_count=len(execution_result.tool_results),
                    error_type=type(exc).__name__,
                )
                raise
            except Exception as exc:
                record_business_span_failure(
                    span,
                    failure_stage="dynamic_synthesis",
                    exception=exc,
                    execution_status=execution_result.status.value,
                )
                raise

            set_safe_span_attributes(
                span,
                {
                    "citation_count": len(
                        synthesis_result.answer.citations
                    ),
                    "requires_human_review": (
                        synthesis_result.answer.requires_human_review
                    ),
                },
            )

            cost_estimate = self._cost_estimator.estimate(
                planning_input_tokens=planner_result.input_tokens,
                planning_output_tokens=planner_result.output_tokens,
                synthesis_input_tokens=synthesis_result.input_tokens,
                synthesis_output_tokens=synthesis_result.output_tokens,
            )

            set_safe_span_attributes(
                span,
                {
                    "estimated_cost_usd": str(
                        cost_estimate.total_cost_usd
                    ),
                },
            )

            response = AgentDynamicQueryResponse(
                query_plan=query_plan,
                execution_plan=planner_result.plan,
                execution_result=execution_result,
                answer=synthesis_result.answer,
                prompt_version=planner_result.prompt_version,
                model=planner_result.model,
                message_id=planner_result.message_id,
                input_tokens=planner_result.input_tokens,
                output_tokens=planner_result.output_tokens,
                planning_duration_ms=planner_result.duration_ms,
                synthesis_prompt_version=synthesis_result.prompt_version,
                synthesis_model=synthesis_result.model,
                synthesis_message_id=synthesis_result.message_id,
                synthesis_input_tokens=synthesis_result.input_tokens,
                synthesis_output_tokens=synthesis_result.output_tokens,
                synthesis_duration_ms=synthesis_result.duration_ms,
                cost_estimate=cost_estimate,
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
                synthesis_prompt_version=synthesis_result.prompt_version,
                synthesis_model=synthesis_result.model,
                synthesis_input_tokens=synthesis_result.input_tokens,
                synthesis_output_tokens=synthesis_result.output_tokens,
                synthesis_citation_count=len(
                    synthesis_result.answer.citations
                ),
                estimated_cost_usd=str(cost_estimate.total_cost_usd),
            )

            return response
