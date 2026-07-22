"""Estimate dynamic-agent LLM costs from configured token rates."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.agent_dynamic_query import AgentDynamicQueryCostEstimate

_TOKEN_RATE_DIVISOR = Decimal("1000000")
_COST_QUANTUM = Decimal("0.000001")


class AgentLLMCostRates(BaseModel):
    """Configured USD prices per one million tokens."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    planning_input_per_million_usd: Decimal = Field(ge=0)
    planning_output_per_million_usd: Decimal = Field(ge=0)
    synthesis_input_per_million_usd: Decimal = Field(ge=0)
    synthesis_output_per_million_usd: Decimal = Field(ge=0)


class AgentLLMCostEstimator:
    """Calculate an auditable O(1) cost estimate from token usage."""

    def __init__(self, *, rates: AgentLLMCostRates | None = None) -> None:
        """Initialize the estimator with explicit or zero-cost rates."""
        self._rates = rates or AgentLLMCostRates(
            planning_input_per_million_usd=Decimal("0"),
            planning_output_per_million_usd=Decimal("0"),
            synthesis_input_per_million_usd=Decimal("0"),
            synthesis_output_per_million_usd=Decimal("0"),
        )

    def estimate(
        self,
        *,
        planning_input_tokens: int,
        planning_output_tokens: int,
        synthesis_input_tokens: int,
        synthesis_output_tokens: int,
    ) -> AgentDynamicQueryCostEstimate:
        """Return a six-decimal USD estimate for one dynamic query."""
        token_counts = (
            planning_input_tokens,
            planning_output_tokens,
            synthesis_input_tokens,
            synthesis_output_tokens,
        )
        if any(token_count < 0 for token_count in token_counts):
            raise ValueError("token counts must be non-negative")

        planning_input_cost = self._calculate_cost(
            planning_input_tokens,
            self._rates.planning_input_per_million_usd,
        )
        planning_output_cost = self._calculate_cost(
            planning_output_tokens,
            self._rates.planning_output_per_million_usd,
        )
        synthesis_input_cost = self._calculate_cost(
            synthesis_input_tokens,
            self._rates.synthesis_input_per_million_usd,
        )
        synthesis_output_cost = self._calculate_cost(
            synthesis_output_tokens,
            self._rates.synthesis_output_per_million_usd,
        )
        total_cost = (
            planning_input_cost
            + planning_output_cost
            + synthesis_input_cost
            + synthesis_output_cost
        ).quantize(_COST_QUANTUM, rounding=ROUND_HALF_UP)

        return AgentDynamicQueryCostEstimate(
            planning_input_cost_usd=planning_input_cost,
            planning_output_cost_usd=planning_output_cost,
            synthesis_input_cost_usd=synthesis_input_cost,
            synthesis_output_cost_usd=synthesis_output_cost,
            total_cost_usd=total_cost,
        )

    @staticmethod
    def _calculate_cost(
        token_count: int,
        rate_per_million_usd: Decimal,
    ) -> Decimal:
        """Calculate one token component using decimal-safe arithmetic."""
        return (
            Decimal(token_count)
            * rate_per_million_usd
            / _TOKEN_RATE_DIVISOR
        ).quantize(_COST_QUANTUM, rounding=ROUND_HALF_UP)
