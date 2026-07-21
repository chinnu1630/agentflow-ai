"""Deterministic policy validation for dynamic AgentFlow plans."""

from __future__ import annotations

import logging

from app.schemas.agent_execution_plan import AgentExecutionPlan
from app.schemas.agent_query import AgentIntent
from app.schemas.agent_tool import AgentToolEffect
from app.services.agent_tool_registry import (
    AgentToolNotRegisteredError,
    AgentToolRegistry,
)

logger = logging.getLogger(__name__)


class AgentExecutionPlanValidationError(RuntimeError):
    """Raised when a dynamic execution plan violates trusted policy."""


class AgentExecutionToolNotAllowedError(AgentExecutionPlanValidationError):
    """Raised when a plan requests an unavailable or forbidden tool."""


class AgentExecutionContextRequiredError(AgentExecutionPlanValidationError):
    """Raised when a tool requires missing release-run context."""


class AgentExecutionApprovalRequiredError(AgentExecutionPlanValidationError):
    """Raised when a side effect lacks durable human approval."""


class AgentExecutionTimeoutViolationError(AgentExecutionPlanValidationError):
    """Raised when a planner requests an excessive tool timeout."""


class AgentExecutionPlanValidator:
    """Validate planner output against deterministic enterprise policy."""

    def __init__(
        self,
        registry: AgentToolRegistry,
        request_id: str,
    ) -> None:
        """Initialize the execution-plan validator.

        Args:
            registry: Trusted registry of approved AgentFlow tools.
            request_id: Request identifier used for structured logging.
        """
        self._registry = registry
        self._request_id = request_id

    def validate(
        self,
        plan: AgentExecutionPlan,
        *,
        has_release_run_context: bool,
        allow_side_effects: bool = False,
        human_approval_granted: bool = False,
    ) -> AgentExecutionPlan:
        """Validate one bounded dynamic execution plan.

        Args:
            plan: Strict planner-generated execution plan.
            has_release_run_context: Whether trusted release context exists.
            allow_side_effects: Whether the caller permits action tools.
            human_approval_granted: Whether durable human approval exists.

        Returns:
            The original validated plan.

        Raises:
            AgentExecutionToolNotAllowedError: For unknown or forbidden tools.
            AgentExecutionContextRequiredError: For missing release context.
            AgentExecutionApprovalRequiredError: For unapproved side effects.
            AgentExecutionTimeoutViolationError: For excessive timeouts.
        """
        for step in plan.steps:
            try:
                definition = self._registry.get_definition(
                    step.invocation.tool_name
                )
            except AgentToolNotRegisteredError as exc:
                raise AgentExecutionToolNotAllowedError(
                    "Execution plan requested an unregistered tool."
                ) from exc

            if (
                definition.requires_release_run_context
                and not has_release_run_context
            ):
                raise AgentExecutionContextRequiredError(
                    f"Tool requires release-run context: "
                    f"{definition.name.value}"
                )

            if (
                step.invocation.timeout_seconds
                > definition.default_timeout_seconds
            ):
                raise AgentExecutionTimeoutViolationError(
                    f"Tool timeout exceeds the trusted registry limit: "
                    f"{definition.name.value}"
                )

            if definition.effect is AgentToolEffect.SIDE_EFFECT:
                self._validate_side_effect(
                    plan=plan,
                    allow_side_effects=allow_side_effects,
                    human_approval_granted=human_approval_granted,
                )

        logger.info(
            "agent_execution_plan_validated",
            extra={
                "run_id": self._request_id,
                "intent": plan.intent.value,
                "step_count": len(plan.steps),
                "allow_side_effects": allow_side_effects,
                "human_approval_granted": human_approval_granted,
            },
        )

        return plan

    @staticmethod
    def _validate_side_effect(
        *,
        plan: AgentExecutionPlan,
        allow_side_effects: bool,
        human_approval_granted: bool,
    ) -> None:
        """Enforce deterministic policy for consequential actions."""
        if plan.intent is not AgentIntent.ACTION_REQUEST:
            raise AgentExecutionToolNotAllowedError(
                "Side-effecting tools require an action-request intent."
            )

        if not allow_side_effects:
            raise AgentExecutionToolNotAllowedError(
                "Side-effecting tools are disabled for this execution."
            )

        if not human_approval_granted:
            raise AgentExecutionApprovalRequiredError(
                "Side-effecting tools require durable human approval."
            )
