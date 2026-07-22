"""Tests for fail-closed dynamic-agent cost policy."""

from decimal import Decimal

import pytest

from app.schemas.agent_dynamic_query import AgentDynamicQueryCostEstimate
from app.services.agent_llm_cost_policy import (
    AgentLLMCostLimitExceededError,
    AgentLLMCostPolicy,
)


def _build_cost(total: str) -> AgentDynamicQueryCostEstimate:
    """Build a valid cost estimate with the requested total."""
    return AgentDynamicQueryCostEstimate(
        planning_input_cost_usd="0",
        planning_output_cost_usd="0",
        synthesis_input_cost_usd="0",
        synthesis_output_cost_usd=total,
        total_cost_usd=total,
    )


def test_allows_cost_at_configured_limit() -> None:
    """A request equal to its cost ceiling should be allowed."""
    policy = AgentLLMCostPolicy(
        max_estimated_cost_usd=Decimal("0.010000")
    )

    policy.enforce(_build_cost("0.010000"))


def test_rejects_cost_above_configured_limit() -> None:
    """A request above its cost ceiling must fail closed."""
    policy = AgentLLMCostPolicy(
        max_estimated_cost_usd=Decimal("0.010000")
    )

    with pytest.raises(AgentLLMCostLimitExceededError):
        policy.enforce(_build_cost("0.010001"))


def test_disabled_policy_allows_any_estimated_cost() -> None:
    """No configured ceiling should leave enforcement disabled."""
    AgentLLMCostPolicy().enforce(_build_cost("999.000000"))
