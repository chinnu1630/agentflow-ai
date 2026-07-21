"""Tests for deterministic AgentFlow execution-plan validation."""

import pytest

from app.schemas.agent_execution_plan import (
    AgentExecutionPlan,
    AgentExecutionStep,
)
from app.schemas.agent_query import AgentIntent, ResponseDepth
from app.schemas.agent_tool import (
    AgentToolInvocation,
    AgentToolName,
)
from app.services.agent_execution_plan_validator import (
    AgentExecutionApprovalRequiredError,
    AgentExecutionContextRequiredError,
    AgentExecutionPlanValidator,
    AgentExecutionTimeoutViolationError,
    AgentExecutionToolNotAllowedError,
)
from app.services.agent_tool_registry import AgentToolRegistry


def _build_plan(
    tool_name: AgentToolName,
    *,
    intent: AgentIntent = AgentIntent.RELEASE_RISK_SUMMARY,
    timeout_seconds: int = 10,
) -> AgentExecutionPlan:
    """Create one reusable execution plan."""
    step_id = "tool_step"

    return AgentExecutionPlan(
        objective="Answer the manager's release question.",
        intent=intent,
        response_depth=ResponseDepth.STANDARD,
        steps=[
            AgentExecutionStep(
                step_id=step_id,
                invocation=AgentToolInvocation(
                    step_id=step_id,
                    tool_name=tool_name,
                    arguments={},
                    timeout_seconds=timeout_seconds,
                ),
            )
        ],
        plan_reason_code="execute_trusted_tool",
    )


def _build_validator() -> AgentExecutionPlanValidator:
    """Create a validator backed by the trusted registry."""
    return AgentExecutionPlanValidator(
        registry=AgentToolRegistry(),
        request_id="request-123",
    )


def test_accepts_read_only_tool_with_required_context() -> None:
    """A valid read-only plan should pass unchanged."""
    validator = _build_validator()
    plan = _build_plan(
        AgentToolName.LOAD_CURRENT_RISK_SNAPSHOT,
    )

    validated_plan = validator.validate(
        plan,
        has_release_run_context=True,
    )

    assert validated_plan is plan


def test_rejects_tool_when_release_context_is_missing() -> None:
    """Context-dependent tools must fail closed without trusted context."""
    validator = _build_validator()
    plan = _build_plan(
        AgentToolName.LOOKUP_JIRA_ISSUE,
    )

    with pytest.raises(
        AgentExecutionContextRequiredError,
        match="requires release-run context",
    ):
        validator.validate(
            plan,
            has_release_run_context=False,
        )


def test_rejects_timeout_above_registry_limit() -> None:
    """Planner-requested timeouts may not exceed trusted limits."""
    validator = _build_validator()
    plan = _build_plan(
        AgentToolName.LOOKUP_APPROVAL_STATUS,
        timeout_seconds=11,
    )

    with pytest.raises(
        AgentExecutionTimeoutViolationError,
        match="timeout exceeds the trusted registry limit",
    ):
        validator.validate(
            plan,
            has_release_run_context=True,
        )


def test_rejects_side_effect_for_non_action_intent() -> None:
    """Consequential tools require an explicit action intent."""
    validator = _build_validator()
    plan = _build_plan(
        AgentToolName.SEND_APPROVED_SLACK_ALERT,
        intent=AgentIntent.RELEASE_RISK_SUMMARY,
        timeout_seconds=30,
    )

    with pytest.raises(
        AgentExecutionToolNotAllowedError,
        match="require an action-request intent",
    ):
        validator.validate(
            plan,
            has_release_run_context=True,
            allow_side_effects=True,
            human_approval_granted=True,
        )


def test_rejects_side_effect_when_actions_are_disabled() -> None:
    """Read-only execution mode must exclude action tools."""
    validator = _build_validator()
    plan = _build_plan(
        AgentToolName.SEND_APPROVED_SLACK_ALERT,
        intent=AgentIntent.ACTION_REQUEST,
        timeout_seconds=30,
    )

    with pytest.raises(
        AgentExecutionToolNotAllowedError,
        match="disabled for this execution",
    ):
        validator.validate(
            plan,
            has_release_run_context=True,
            allow_side_effects=False,
            human_approval_granted=True,
        )


def test_rejects_side_effect_without_durable_approval() -> None:
    """An action tool must never bypass persisted human approval."""
    validator = _build_validator()
    plan = _build_plan(
        AgentToolName.SEND_APPROVED_SLACK_ALERT,
        intent=AgentIntent.ACTION_REQUEST,
        timeout_seconds=30,
    )

    with pytest.raises(
        AgentExecutionApprovalRequiredError,
        match="require durable human approval",
    ):
        validator.validate(
            plan,
            has_release_run_context=True,
            allow_side_effects=True,
            human_approval_granted=False,
        )


def test_accepts_approved_action_plan() -> None:
    """An explicitly allowed and durably approved action may pass."""
    validator = _build_validator()
    plan = _build_plan(
        AgentToolName.SEND_APPROVED_SLACK_ALERT,
        intent=AgentIntent.ACTION_REQUEST,
        timeout_seconds=30,
    )

    validated_plan = validator.validate(
        plan,
        has_release_run_context=True,
        allow_side_effects=True,
        human_approval_granted=True,
    )

    assert validated_plan is plan
