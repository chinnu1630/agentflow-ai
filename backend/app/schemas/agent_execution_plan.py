"""Strict schemas for bounded dynamic AgentFlow execution plans.

An execution plan is a validated directed acyclic graph of approved tool
invocations. The schema describes intended execution only; authorization and
tool availability are enforced later by deterministic policy services.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.agent_query import AgentIntent, ResponseDepth
from app.schemas.agent_tool import AgentToolInvocation


class AgentStepFailurePolicy(StrEnum):
    """Allowed behavior when one execution-plan step fails."""

    FAIL_PLAN = "fail_plan"
    CONTINUE_WITH_PARTIAL_RESULTS = "continue_with_partial_results"


class AgentExecutionBudget(BaseModel):
    """Deterministic resource limits for one dynamic agent execution."""

    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(default=10, ge=1, le=20)
    max_parallel_steps: int = Field(default=3, ge=1, le=5)
    max_total_duration_seconds: int = Field(default=180, ge=1, le=300)
    max_replans: int = Field(default=1, ge=0, le=3)


class AgentExecutionStep(BaseModel):
    """One tool invocation and its dependency requirements."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    step_id: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$",
    )
    invocation: AgentToolInvocation
    depends_on: list[str] = Field(
        default_factory=list,
        max_length=10,
    )
    failure_policy: AgentStepFailurePolicy = (
        AgentStepFailurePolicy.CONTINUE_WITH_PARTIAL_RESULTS
    )

    @field_validator("depends_on")
    @classmethod
    def validate_dependencies(cls, values: list[str]) -> list[str]:
        """Reject blank or duplicate dependency identifiers."""
        normalized_values = [value.strip() for value in values]

        if any(not value for value in normalized_values):
            raise ValueError("step dependencies must not be blank")

        if len(normalized_values) != len(set(normalized_values)):
            raise ValueError("step dependencies must not contain duplicates")

        return normalized_values

    @model_validator(mode="after")
    def validate_invocation_identity(self) -> AgentExecutionStep:
        """Require the step and invocation to use the same identifier."""
        if self.step_id != self.invocation.step_id:
            raise ValueError(
                "execution step ID must match invocation step ID"
            )

        if self.step_id in self.depends_on:
            raise ValueError("execution steps must not depend on themselves")

        return self


class AgentExecutionPlan(BaseModel):
    """Validated bounded plan produced for dynamic agent execution."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    schema_version: Literal["agent_execution_plan_v1"] = (
        "agent_execution_plan_v1"
    )
    objective: str = Field(min_length=1, max_length=500)
    intent: AgentIntent
    response_depth: ResponseDepth
    steps: list[AgentExecutionStep] = Field(
        min_length=1,
        max_length=20,
    )
    budget: AgentExecutionBudget = Field(
        default_factory=AgentExecutionBudget
    )
    requires_synthesis: bool = True
    plan_reason_code: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[a-z0-9_]+$",
    )

    @model_validator(mode="after")
    def validate_execution_graph(self) -> AgentExecutionPlan:
        """Validate step limits, dependency ownership, and graph acyclicity."""
        if len(self.steps) > self.budget.max_steps:
            raise ValueError(
                "execution plan exceeds its configured step budget"
            )

        step_ids = [step.step_id for step in self.steps]

        if len(step_ids) != len(set(step_ids)):
            raise ValueError(
                "execution plan step IDs must be unique"
            )

        known_step_ids = set(step_ids)

        for step in self.steps:
            unknown_dependencies = (
                set(step.depends_on) - known_step_ids
            )

            if unknown_dependencies:
                raise ValueError(
                    "execution plan contains unknown step dependencies"
                )

        self._reject_dependency_cycles()
        return self

    def _reject_dependency_cycles(self) -> None:
        """Reject cyclic tool dependencies using depth-first search."""
        dependencies_by_step = {
            step.step_id: tuple(step.depends_on)
            for step in self.steps
        }
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(step_id: str) -> None:
            if step_id in visiting:
                raise ValueError(
                    "execution plan dependency graph must be acyclic"
                )

            if step_id in visited:
                return

            visiting.add(step_id)

            for dependency_id in dependencies_by_step[step_id]:
                visit(dependency_id)

            visiting.remove(step_id)
            visited.add(step_id)

        for current_step_id in dependencies_by_step:
            visit(current_step_id)
