"""Tests for configurable dynamic-agent LLM cost estimation."""

from decimal import Decimal

import pytest

from app.services.agent_llm_cost_estimator import (
    AgentLLMCostEstimator,
    AgentLLMCostRates,
)


def test_estimates_planning_and_synthesis_costs() -> None:
    """Estimator should calculate components using decimal arithmetic."""
    estimator = AgentLLMCostEstimator(
        rates=AgentLLMCostRates(
            planning_input_per_million_usd=Decimal("3"),
            planning_output_per_million_usd=Decimal("15"),
            synthesis_input_per_million_usd=Decimal("3"),
            synthesis_output_per_million_usd=Decimal("15"),
        )
    )

    result = estimator.estimate(
        planning_input_tokens=250,
        planning_output_tokens=100,
        synthesis_input_tokens=300,
        synthesis_output_tokens=120,
    )

    assert result.planning_input_cost_usd == Decimal("0.000750")
    assert result.planning_output_cost_usd == Decimal("0.001500")
    assert result.synthesis_input_cost_usd == Decimal("0.000900")
    assert result.synthesis_output_cost_usd == Decimal("0.001800")
    assert result.total_cost_usd == Decimal("0.004950")


def test_defaults_to_zero_when_pricing_is_not_configured() -> None:
    """Unknown pricing should produce an explicit zero-cost estimate."""
    result = AgentLLMCostEstimator().estimate(
        planning_input_tokens=250,
        planning_output_tokens=100,
        synthesis_input_tokens=300,
        synthesis_output_tokens=120,
    )

    assert result.total_cost_usd == Decimal("0.000000")


def test_rejects_negative_token_counts() -> None:
    """Invalid token usage must never produce a cost estimate."""
    with pytest.raises(
        ValueError,
        match="token counts must be non-negative",
    ):
        AgentLLMCostEstimator().estimate(
            planning_input_tokens=-1,
            planning_output_tokens=0,
            synthesis_input_tokens=0,
            synthesis_output_tokens=0,
        )
