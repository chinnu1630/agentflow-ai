"""Enforce configurable cost limits for dynamic-agent requests."""

from __future__ import annotations

from decimal import Decimal

from app.schemas.agent_dynamic_query import AgentDynamicQueryCostEstimate


class AgentLLMCostLimitExceededError(RuntimeError):
    """Raised when estimated dynamic-query cost exceeds policy."""


class AgentLLMCostPolicy:
    """Apply an O(1) fail-closed cost threshold to one query."""

    def __init__(
        self,
        *,
        max_estimated_cost_usd: Decimal | None = None,
    ) -> None:
        """Initialize the policy with an optional positive USD ceiling."""
        if (
            max_estimated_cost_usd is not None
            and max_estimated_cost_usd <= 0
        ):
            raise ValueError(
                "max_estimated_cost_usd must be greater than zero"
            )

        self._max_estimated_cost_usd = max_estimated_cost_usd

    @property
    def max_estimated_cost_usd(self) -> Decimal | None:
        """Return the configured request ceiling."""
        return self._max_estimated_cost_usd

    def enforce(
        self,
        cost_estimate: AgentDynamicQueryCostEstimate,
    ) -> None:
        """Reject a result whose estimated cost exceeds policy."""
        if (
            self._max_estimated_cost_usd is not None
            and cost_estimate.total_cost_usd
            > self._max_estimated_cost_usd
        ):
            raise AgentLLMCostLimitExceededError(
                "Dynamic query exceeded the configured cost limit."
            )
