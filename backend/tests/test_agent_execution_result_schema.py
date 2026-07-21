"""Tests for dynamic AgentFlow execution-result contracts."""

from uuid import UUID

import pytest
from pydantic import ValidationError

from app.schemas.agent_execution_result import (
    AgentExecutionResult,
    AgentExecutionStatus,
)
from app.schemas.agent_query import AgentIntent
from app.schemas.agent_tool import (
    AgentToolExecutionStatus,
    AgentToolName,
    AgentToolResult,
)


def _build_tool_result(
    *,
    step_id: str = "load_snapshot",
    status: AgentToolExecutionStatus = (
        AgentToolExecutionStatus.SUCCESS
    ),
) -> AgentToolResult:
    """Create one reusable normalized tool result."""
    error_code = None
    error_message = None

    if status is AgentToolExecutionStatus.FAILED:
        error_code = "snapshot_unavailable"
        error_message = "The persisted snapshot could not be loaded."

    return AgentToolResult(
        step_id=step_id,
        tool_name=AgentToolName.LOAD_CURRENT_RISK_SNAPSHOT,
        status=status,
        output={"release_run_id": "release-123"},
        error_code=error_code,
        error_message=error_message,
        duration_ms=12,
    )


def _build_execution_result(
    *,
    status: AgentExecutionStatus,
    tool_results: list[AgentToolResult],
) -> AgentExecutionResult:
    """Create one execution result for schema validation tests."""
    return AgentExecutionResult(
        execution_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        intent=AgentIntent.EXPLAIN_RISK_SCORE,
        objective="Explain the persisted release-risk score.",
        plan_reason_code="load_trusted_snapshot",
        status=status,
        tool_results=tool_results,
        requires_synthesis=True,
        duration_ms=25,
    )


def test_accepts_successful_execution_result() -> None:
    """All-success tool results produce a successful execution."""
    result = _build_execution_result(
        status=AgentExecutionStatus.SUCCESS,
        tool_results=[_build_tool_result()],
    )

    assert result.status is AgentExecutionStatus.SUCCESS
    assert result.tool_results[0].step_id == "load_snapshot"


def test_rejects_duplicate_step_results() -> None:
    """Every planned step may appear only once in execution output."""
    with pytest.raises(
        ValidationError,
        match="execution results must contain unique step IDs",
    ):
        _build_execution_result(
            status=AgentExecutionStatus.SUCCESS,
            tool_results=[
                _build_tool_result(),
                _build_tool_result(),
            ],
        )


def test_rejects_success_when_a_tool_failed() -> None:
    """Aggregate success cannot hide a failed tool."""
    with pytest.raises(
        ValidationError,
        match="successful executions require every tool to succeed",
    ):
        _build_execution_result(
            status=AgentExecutionStatus.SUCCESS,
            tool_results=[
                _build_tool_result(
                    status=AgentToolExecutionStatus.FAILED,
                )
            ],
        )


def test_rejects_partial_when_every_tool_succeeded() -> None:
    """Partial status requires an actual degraded result."""
    with pytest.raises(
        ValidationError,
        match="partial executions require usable and degraded results",
    ):
        _build_execution_result(
            status=AgentExecutionStatus.PARTIAL,
            tool_results=[_build_tool_result()],
        )


def test_rejects_failed_execution_without_failed_tool() -> None:
    """Aggregate failure must be supported by a failed tool result."""
    with pytest.raises(
        ValidationError,
        match="failed executions require at least one failed tool result",
    ):
        _build_execution_result(
            status=AgentExecutionStatus.FAILED,
            tool_results=[
                _build_tool_result(
                    status=AgentToolExecutionStatus.PARTIAL,
                )
            ],
        )
