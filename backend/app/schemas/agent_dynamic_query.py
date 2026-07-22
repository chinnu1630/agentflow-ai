"""Public API contracts for bounded dynamic AgentFlow queries."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.agent_dynamic_synthesis import AgentDynamicAnswer
from app.schemas.agent_execution_plan import AgentExecutionPlan
from app.schemas.agent_execution_result import AgentExecutionResult
from app.schemas.agent_query import AgentQueryPlan


class AgentDynamicQueryCostEstimate(BaseModel):
    """Estimated dynamic-agent LLM cost using configured pricing rates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    currency: Literal["USD"] = "USD"
    planning_input_cost_usd: Decimal = Field(ge=0, decimal_places=6)
    planning_output_cost_usd: Decimal = Field(ge=0, decimal_places=6)
    synthesis_input_cost_usd: Decimal = Field(ge=0, decimal_places=6)
    synthesis_output_cost_usd: Decimal = Field(ge=0, decimal_places=6)
    total_cost_usd: Decimal = Field(ge=0, decimal_places=6)


class AgentDynamicQueryResponse(BaseModel):
    """Auditable response from read-only dynamic agent execution."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    query_plan: AgentQueryPlan
    execution_plan: AgentExecutionPlan
    execution_result: AgentExecutionResult
    answer: AgentDynamicAnswer
    prompt_version: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=255)
    message_id: str = Field(min_length=1, max_length=255)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    planning_duration_ms: float = Field(ge=0.0)
    synthesis_prompt_version: str = Field(min_length=1, max_length=100)
    synthesis_model: str = Field(min_length=1, max_length=255)
    synthesis_message_id: str = Field(min_length=1, max_length=255)
    synthesis_input_tokens: int = Field(ge=0)
    synthesis_output_tokens: int = Field(ge=0)
    synthesis_duration_ms: float = Field(ge=0.0)
    cost_estimate: AgentDynamicQueryCostEstimate

    @model_validator(mode="after")
    def validate_pipeline_consistency(self) -> AgentDynamicQueryResponse:
        """Require routing, planning, and execution to retain one intent."""
        routed_intent = self.query_plan.intent

        if self.execution_plan.intent is not routed_intent:
            raise ValueError(
                "execution plan intent must match the routed query intent"
            )

        if self.execution_result.intent is not routed_intent:
            raise ValueError(
                "execution result intent must match the routed query intent"
            )

        if (
            self.execution_result.objective
            != self.execution_plan.objective
        ):
            raise ValueError(
                "execution result objective must match the execution plan"
            )

        if (
            self.execution_result.plan_reason_code
            != self.execution_plan.plan_reason_code
        ):
            raise ValueError(
                "execution result reason code must match the execution plan"
            )

        return self
