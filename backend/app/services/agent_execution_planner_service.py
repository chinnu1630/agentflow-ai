"""Create and validate bounded dynamic AgentFlow execution plans."""

from __future__ import annotations

from typing import Protocol

import structlog
from pydantic import BaseModel, ConfigDict, Field

from app.integrations.anthropic_execution_planner_client import (
    ClaudeExecutionPlanResult,
)
from app.observability.tracing import start_business_span
from app.schemas.agent_execution_plan import AgentExecutionPlan
from app.schemas.agent_query import AgentQueryPlan, AgentQueryRequest
from app.services.agent_execution_plan_validator import (
    AgentExecutionPlanValidator,
)
from app.services.agent_execution_planner_prompt import (
    AgentExecutionPlannerPromptBuilder,
)

logger = structlog.get_logger(__name__)


class ExecutionPlannerClientProtocol(Protocol):
    """Claude operation required by the execution-planner service."""

    async def create_execution_plan(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        prompt_version: str,
    ) -> ClaudeExecutionPlanResult:
        """Generate one validated structured execution plan."""

        ...


class AgentExecutionPlannerServiceError(RuntimeError):
    """Raised when a bounded execution plan cannot be created."""


class AgentExecutionPlanIntentMismatchError(
    AgentExecutionPlannerServiceError
):
    """Raised when Claude changes the deterministic routed intent."""


class AgentExecutionPlannerResult(BaseModel):
    """Validated plan and safe LLMOps metadata returned by the service."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan: AgentExecutionPlan
    prompt_version: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1)
    message_id: str = Field(min_length=1)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    duration_ms: float = Field(ge=0.0)


class AgentExecutionPlannerService:
    """Create policy-validated plans without granting execution authority."""

    def __init__(
        self,
        *,
        planner_client: ExecutionPlannerClientProtocol,
        plan_validator: AgentExecutionPlanValidator,
        request_id: str,
        prompt_builder: AgentExecutionPlannerPromptBuilder | None = None,
    ) -> None:
        """Initialize the bounded execution-planner service.

        Args:
            planner_client: Structured Claude planning client.
            plan_validator: Deterministic enterprise policy validator.
            request_id: Request identifier used for logs and traces.
            prompt_builder: Optional injected prompt builder for testing.
        """
        self._planner_client = planner_client
        self._plan_validator = plan_validator
        self._request_id = request_id
        self._prompt_builder = (
            prompt_builder or AgentExecutionPlannerPromptBuilder()
        )

    async def create_plan(
        self,
        *,
        request: AgentQueryRequest,
        query_plan: AgentQueryPlan,
    ) -> AgentExecutionPlannerResult:
        """Create and validate one bounded dynamic execution plan.

        The LLM may select approved read-only tools, but it cannot change the
        deterministic intent, authorize side effects, or bypass tool policy.

        Args:
            request: Original validated manager query.
            query_plan: Trusted deterministic routing result.

        Returns:
            Policy-validated execution plan and safe Claude metadata.

        Raises:
            AgentExecutionPlanIntentMismatchError: If Claude changes intent.
            AgentExecutionPlanValidationError: If tool policy is violated.
        """
        prompt = self._prompt_builder.build(
            request=request,
            query_plan=query_plan,
        )

        with start_business_span(
            "agent.execution_planning",
            {
                "run_id": self._request_id,
                "intent": query_plan.intent.value,
                "prompt_version": prompt.prompt_version,
                "approved_tool_count": prompt.approved_tool_count,
                "release_run_context_available": (
                    prompt.release_run_context_available
                ),
            },
        ) as span:
            claude_result = (
                await self._planner_client.create_execution_plan(
                    system_prompt=prompt.system_prompt,
                    user_prompt=prompt.user_prompt,
                    prompt_version=prompt.prompt_version,
                )
            )

            if claude_result.plan.intent is not query_plan.intent:
                raise AgentExecutionPlanIntentMismatchError(
                    "Claude execution plan changed the deterministic intent."
                )

            validated_plan = self._plan_validator.validate(
                claude_result.plan,
                has_release_run_context=(
                    prompt.release_run_context_available
                ),
                allow_side_effects=False,
                human_approval_granted=False,
            )

            span.set_attribute(
                "agent.execution_plan.step_count",
                len(validated_plan.steps),
            )
            span.set_attribute(
                "llm.model",
                claude_result.model,
            )
            span.set_attribute(
                "llm.input_tokens",
                claude_result.input_tokens,
            )
            span.set_attribute(
                "llm.output_tokens",
                claude_result.output_tokens,
            )

        logger.info(
            "agent_execution_plan_created",
            run_id=self._request_id,
            intent=validated_plan.intent.value,
            step_count=len(validated_plan.steps),
            prompt_version=claude_result.prompt_version,
            model=claude_result.model,
            input_tokens=claude_result.input_tokens,
            output_tokens=claude_result.output_tokens,
            duration_ms=claude_result.duration_ms,
        )

        return AgentExecutionPlannerResult(
            plan=validated_plan,
            prompt_version=claude_result.prompt_version,
            model=claude_result.model,
            message_id=claude_result.message_id,
            input_tokens=claude_result.input_tokens,
            output_tokens=claude_result.output_tokens,
            duration_ms=claude_result.duration_ms,
        )
