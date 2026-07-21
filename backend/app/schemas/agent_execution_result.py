"""Strict result contract for bounded dynamic AgentFlow executions."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.agent_query import AgentIntent
from app.schemas.agent_tool import (
    AgentToolExecutionStatus,
    AgentToolResult,
)


class AgentExecutionStatus(StrEnum):
    """Normalized outcome of an entire dynamic execution plan."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class AgentExecutionResult(BaseModel):
    """Auditable result produced after executing a validated tool DAG."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    execution_id: UUID = Field(default_factory=uuid4)
    intent: AgentIntent
    objective: str = Field(min_length=1, max_length=500)
    plan_reason_code: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[a-z0-9_]+$",
    )
    status: AgentExecutionStatus
    tool_results: list[AgentToolResult] = Field(
        min_length=1,
        max_length=20,
    )
    requires_synthesis: bool
    duration_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_execution_consistency(self) -> AgentExecutionResult:
        """Reject duplicate steps and contradictory aggregate statuses."""
        step_ids = [result.step_id for result in self.tool_results]

        if len(step_ids) != len(set(step_ids)):
            raise ValueError(
                "execution results must contain unique step IDs"
            )

        tool_statuses = [
            result.status for result in self.tool_results
        ]
        has_degraded_result = any(
            status is not AgentToolExecutionStatus.SUCCESS
            for status in tool_statuses
        )
        has_usable_result = any(
            status
            in {
                AgentToolExecutionStatus.SUCCESS,
                AgentToolExecutionStatus.PARTIAL,
            }
            for status in tool_statuses
        )
        has_failed_result = any(
            status is AgentToolExecutionStatus.FAILED
            for status in tool_statuses
        )

        if (
            self.status is AgentExecutionStatus.SUCCESS
            and has_degraded_result
        ):
            raise ValueError(
                "successful executions require every tool to succeed"
            )

        if self.status is AgentExecutionStatus.PARTIAL:
            if not has_degraded_result or not has_usable_result:
                raise ValueError(
                    "partial executions require usable and degraded results"
                )

        if (
            self.status is AgentExecutionStatus.FAILED
            and not has_failed_result
        ):
            raise ValueError(
                "failed executions require at least one failed tool result"
            )

        return self
