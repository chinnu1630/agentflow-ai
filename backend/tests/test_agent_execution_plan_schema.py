"""Tests for bounded dynamic AgentFlow execution plans."""

import pytest
from pydantic import ValidationError

from app.schemas.agent_execution_plan import (
    AgentExecutionBudget,
    AgentExecutionPlan,
    AgentExecutionStep,
)
from app.schemas.agent_query import AgentIntent, ResponseDepth
from app.schemas.agent_tool import (
    AgentToolInvocation,
    AgentToolName,
)


def _build_step(
    step_id: str,
    tool_name: AgentToolName,
    *,
    depends_on: list[str] | None = None,
) -> AgentExecutionStep:
    """Create one reusable execution-plan step."""
    return AgentExecutionStep(
        step_id=step_id,
        invocation=AgentToolInvocation(
            step_id=step_id,
            tool_name=tool_name,
            arguments={},
            timeout_seconds=20,
        ),
        depends_on=depends_on or [],
    )


def test_accepts_valid_parallel_execution_plan() -> None:
    """Independent read-only tools may be planned in parallel."""
    plan = AgentExecutionPlan(
        objective="Assess current release risks.",
        intent=AgentIntent.RELEASE_RISK_SUMMARY,
        response_depth=ResponseDepth.STANDARD,
        steps=[
            _build_step(
                "github",
                AgentToolName.LOOKUP_GITHUB_PULL_REQUEST,
            ),
            _build_step(
                "jira",
                AgentToolName.LOOKUP_JIRA_ISSUE,
            ),
            _build_step(
                "knowledge",
                AgentToolName.SEARCH_ENGINEERING_KNOWLEDGE,
            ),
        ],
        plan_reason_code="collect_release_evidence",
    )

    assert plan.schema_version == "agent_execution_plan_v1"
    assert len(plan.steps) == 3


def test_rejects_duplicate_step_ids() -> None:
    """Execution step identifiers must remain unique."""
    with pytest.raises(
        ValidationError,
        match="execution plan step IDs must be unique",
    ):
        AgentExecutionPlan(
            objective="Assess current release risks.",
            intent=AgentIntent.RELEASE_RISK_SUMMARY,
            response_depth=ResponseDepth.STANDARD,
            steps=[
                _build_step(
                    "lookup",
                    AgentToolName.LOOKUP_GITHUB_PULL_REQUEST,
                ),
                _build_step(
                    "lookup",
                    AgentToolName.LOOKUP_JIRA_ISSUE,
                ),
            ],
            plan_reason_code="collect_release_evidence",
        )


def test_rejects_unknown_dependency() -> None:
    """Every dependency must reference a step in the same plan."""
    with pytest.raises(
        ValidationError,
        match="unknown step dependencies",
    ):
        AgentExecutionPlan(
            objective="Assess current release risks.",
            intent=AgentIntent.RELEASE_RISK_SUMMARY,
            response_depth=ResponseDepth.STANDARD,
            steps=[
                _build_step(
                    "synthesize",
                    AgentToolName.LOAD_CURRENT_RISK_SNAPSHOT,
                    depends_on=["missing_step"],
                ),
            ],
            plan_reason_code="load_release_context",
        )


def test_rejects_dependency_cycle() -> None:
    """Dynamic execution plans must always remain acyclic."""
    with pytest.raises(
        ValidationError,
        match="dependency graph must be acyclic",
    ):
        AgentExecutionPlan(
            objective="Assess current release risks.",
            intent=AgentIntent.RELEASE_RISK_SUMMARY,
            response_depth=ResponseDepth.STANDARD,
            steps=[
                _build_step(
                    "github",
                    AgentToolName.LOOKUP_GITHUB_PULL_REQUEST,
                    depends_on=["jira"],
                ),
                _build_step(
                    "jira",
                    AgentToolName.LOOKUP_JIRA_ISSUE,
                    depends_on=["github"],
                ),
            ],
            plan_reason_code="collect_release_evidence",
        )


def test_rejects_plan_exceeding_step_budget() -> None:
    """The planner may not exceed deterministic execution limits."""
    with pytest.raises(
        ValidationError,
        match="exceeds its configured step budget",
    ):
        AgentExecutionPlan(
            objective="Assess current release risks.",
            intent=AgentIntent.RELEASE_RISK_SUMMARY,
            response_depth=ResponseDepth.STANDARD,
            steps=[
                _build_step(
                    "github",
                    AgentToolName.LOOKUP_GITHUB_PULL_REQUEST,
                ),
                _build_step(
                    "jira",
                    AgentToolName.LOOKUP_JIRA_ISSUE,
                ),
            ],
            budget=AgentExecutionBudget(max_steps=1),
            plan_reason_code="collect_release_evidence",
        )


def test_rejects_mismatched_step_and_invocation_ids() -> None:
    """Step identity must match its nested invocation identity."""
    with pytest.raises(
        ValidationError,
        match="execution step ID must match invocation step ID",
    ):
        AgentExecutionStep(
            step_id="github",
            invocation=AgentToolInvocation(
                step_id="jira",
                tool_name=AgentToolName.LOOKUP_GITHUB_PULL_REQUEST,
                arguments={},
                timeout_seconds=20,
            ),
        )
