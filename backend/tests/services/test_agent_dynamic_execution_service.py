"""Tests for bounded dynamic AgentFlow DAG execution."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest

from app.schemas.agent_execution_plan import (
    AgentExecutionBudget,
    AgentExecutionPlan,
    AgentExecutionStep,
    AgentStepFailurePolicy,
)
from app.schemas.agent_execution_result import AgentExecutionStatus
from app.schemas.agent_query import AgentIntent, ResponseDepth
from app.schemas.agent_tool import (
    AgentToolExecutionStatus,
    AgentToolInvocation,
    AgentToolName,
    AgentToolResult,
)
from app.services.agent_dynamic_execution_service import (
    AgentDynamicExecutionService,
    AgentToolAdapterError,
)
from app.services.agent_execution_plan_validator import (
    AgentExecutionPlanValidator,
    AgentExecutionToolNotAllowedError,
)
from app.services.agent_tool_registry import AgentToolRegistry


class RecordingAdapter:
    """Record dependencies and bounded concurrent executions."""

    def __init__(
        self,
        *,
        status: AgentToolExecutionStatus = (
            AgentToolExecutionStatus.SUCCESS
        ),
        delay_seconds: float = 0.0,
        raise_adapter_error: bool = False,
    ) -> None:
        self.status = status
        self.delay_seconds = delay_seconds
        self.raise_adapter_error = raise_adapter_error
        self.received_dependencies: list[set[str]] = []
        self.active_count = 0
        self.max_active_count = 0

    async def execute(
        self,
        *,
        invocation: AgentToolInvocation,
        dependency_results: Mapping[str, AgentToolResult],
    ) -> AgentToolResult:
        """Return a configured normalized result."""
        self.received_dependencies.append(
            set(dependency_results)
        )
        self.active_count += 1
        self.max_active_count = max(
            self.max_active_count,
            self.active_count,
        )

        try:
            if self.delay_seconds:
                await asyncio.sleep(self.delay_seconds)

            if self.raise_adapter_error:
                raise AgentToolAdapterError("adapter failed")

            error_code = None
            error_message = None

            if self.status is AgentToolExecutionStatus.FAILED:
                error_code = "configured_failure"
                error_message = "The configured adapter failed."

            return AgentToolResult(
                step_id=invocation.step_id,
                tool_name=invocation.tool_name,
                status=self.status,
                output={"step_id": invocation.step_id},
                error_code=error_code,
                error_message=error_message,
                duration_ms=0,
            )
        finally:
            self.active_count -= 1


def _build_step(
    *,
    step_id: str,
    tool_name: AgentToolName,
    depends_on: list[str] | None = None,
    timeout_seconds: int = 10,
    failure_policy: AgentStepFailurePolicy = (
        AgentStepFailurePolicy.CONTINUE_WITH_PARTIAL_RESULTS
    ),
) -> AgentExecutionStep:
    """Create one reusable execution step."""
    return AgentExecutionStep(
        step_id=step_id,
        invocation=AgentToolInvocation(
            step_id=step_id,
            tool_name=tool_name,
            timeout_seconds=timeout_seconds,
        ),
        depends_on=depends_on or [],
        failure_policy=failure_policy,
    )


def _build_plan(
    *,
    steps: list[AgentExecutionStep],
    max_parallel_steps: int = 3,
    max_total_duration_seconds: int = 30,
    intent: AgentIntent = AgentIntent.EXPLAIN_RISK_SCORE,
) -> AgentExecutionPlan:
    """Create one reusable validated execution plan."""
    return AgentExecutionPlan(
        objective="Answer the manager using trusted tools.",
        intent=intent,
        response_depth=ResponseDepth.DEEP,
        steps=steps,
        budget=AgentExecutionBudget(
            max_steps=10,
            max_parallel_steps=max_parallel_steps,
            max_total_duration_seconds=max_total_duration_seconds,
            max_replans=0,
        ),
        requires_synthesis=True,
        plan_reason_code="trusted_tool_execution",
    )


def _build_service(
    adapters: Mapping[AgentToolName, RecordingAdapter],
) -> AgentDynamicExecutionService:
    """Create the execution service with deterministic policy."""
    registry = AgentToolRegistry()
    validator = AgentExecutionPlanValidator(
        registry=registry,
        request_id="request-123",
    )
    return AgentDynamicExecutionService(
        adapters=adapters,
        plan_validator=validator,
        request_id="request-123",
    )


@pytest.mark.asyncio
async def test_executes_dependencies_in_order() -> None:
    """A dependent step receives the completed dependency result."""
    snapshot_adapter = RecordingAdapter()
    jira_adapter = RecordingAdapter()
    service = _build_service(
        {
            AgentToolName.LOAD_CURRENT_RISK_SNAPSHOT: (
                snapshot_adapter
            ),
            AgentToolName.LOOKUP_JIRA_ISSUE: jira_adapter,
        }
    )
    plan = _build_plan(
        steps=[
            _build_step(
                step_id="snapshot",
                tool_name=(
                    AgentToolName.LOAD_CURRENT_RISK_SNAPSHOT
                ),
            ),
            _build_step(
                step_id="jira",
                tool_name=AgentToolName.LOOKUP_JIRA_ISSUE,
                depends_on=["snapshot"],
            ),
        ]
    )

    result = await service.execute(
        plan,
        has_release_run_context=True,
    )

    assert result.status is AgentExecutionStatus.SUCCESS
    assert jira_adapter.received_dependencies == [{"snapshot"}]
    assert [
        tool_result.step_id
        for tool_result in result.tool_results
    ] == ["snapshot", "jira"]


@pytest.mark.asyncio
async def test_runs_independent_steps_with_bounded_parallelism() -> None:
    """Independent steps execute concurrently within the plan limit."""
    shared_adapter = RecordingAdapter(delay_seconds=0.02)
    service = _build_service(
        {
            AgentToolName.LOOKUP_APPROVAL_STATUS: shared_adapter,
            AgentToolName.LOOKUP_SLACK_STATUS: shared_adapter,
            AgentToolName.LOAD_CURRENT_RISK_SNAPSHOT: shared_adapter,
        }
    )
    plan = _build_plan(
        steps=[
            _build_step(
                step_id="approval",
                tool_name=AgentToolName.LOOKUP_APPROVAL_STATUS,
            ),
            _build_step(
                step_id="slack",
                tool_name=AgentToolName.LOOKUP_SLACK_STATUS,
            ),
            _build_step(
                step_id="snapshot",
                tool_name=(
                    AgentToolName.LOAD_CURRENT_RISK_SNAPSHOT
                ),
            ),
        ],
        max_parallel_steps=2,
    )

    result = await service.execute(
        plan,
        has_release_run_context=True,
    )

    assert result.status is AgentExecutionStatus.SUCCESS
    assert shared_adapter.max_active_count == 2


@pytest.mark.asyncio
async def test_normalizes_tool_timeout() -> None:
    """A per-tool timeout becomes a safe failed result."""
    adapter = RecordingAdapter(delay_seconds=1.1)
    service = _build_service(
        {AgentToolName.LOAD_CURRENT_RISK_SNAPSHOT: adapter}
    )
    plan = _build_plan(
        steps=[
            _build_step(
                step_id="snapshot",
                tool_name=(
                    AgentToolName.LOAD_CURRENT_RISK_SNAPSHOT
                ),
                timeout_seconds=1,
            )
        ]
    )

    result = await service.execute(
        plan,
        has_release_run_context=True,
    )

    assert result.status is AgentExecutionStatus.FAILED
    assert result.tool_results[0].error_code == "tool_timeout"


@pytest.mark.asyncio
async def test_continues_with_partial_results_after_failure() -> None:
    """Continue policy preserves usable results from other tools."""
    failed_adapter = RecordingAdapter(
        raise_adapter_error=True
    )
    successful_adapter = RecordingAdapter()
    service = _build_service(
        {
            AgentToolName.LOAD_CURRENT_RISK_SNAPSHOT: (
                failed_adapter
            ),
            AgentToolName.LOOKUP_APPROVAL_STATUS: (
                successful_adapter
            ),
        }
    )
    plan = _build_plan(
        steps=[
            _build_step(
                step_id="snapshot",
                tool_name=(
                    AgentToolName.LOAD_CURRENT_RISK_SNAPSHOT
                ),
            ),
            _build_step(
                step_id="approval",
                tool_name=AgentToolName.LOOKUP_APPROVAL_STATUS,
            ),
        ]
    )

    result = await service.execute(
        plan,
        has_release_run_context=True,
    )

    assert result.status is AgentExecutionStatus.PARTIAL
    assert {
        tool_result.status
        for tool_result in result.tool_results
    } == {
        AgentToolExecutionStatus.SUCCESS,
        AgentToolExecutionStatus.FAILED,
    }


@pytest.mark.asyncio
async def test_fail_plan_policy_aborts_remaining_steps() -> None:
    """A required failed step creates auditable aborted results."""
    failed_adapter = RecordingAdapter(
        raise_adapter_error=True
    )
    downstream_adapter = RecordingAdapter()
    service = _build_service(
        {
            AgentToolName.LOAD_CURRENT_RISK_SNAPSHOT: (
                failed_adapter
            ),
            AgentToolName.LOOKUP_JIRA_ISSUE: downstream_adapter,
        }
    )
    plan = _build_plan(
        steps=[
            _build_step(
                step_id="snapshot",
                tool_name=(
                    AgentToolName.LOAD_CURRENT_RISK_SNAPSHOT
                ),
                failure_policy=AgentStepFailurePolicy.FAIL_PLAN,
            ),
            _build_step(
                step_id="jira",
                tool_name=AgentToolName.LOOKUP_JIRA_ISSUE,
                depends_on=["snapshot"],
            ),
        ]
    )

    result = await service.execute(
        plan,
        has_release_run_context=True,
    )

    assert result.status is AgentExecutionStatus.FAILED
    assert result.tool_results[1].error_code == "plan_aborted"
    assert downstream_adapter.received_dependencies == []


@pytest.mark.asyncio
async def test_revalidates_side_effect_policy_before_execution() -> None:
    """The executor cannot run an unapproved side-effecting tool."""
    slack_adapter = RecordingAdapter()
    service = _build_service(
        {
            AgentToolName.SEND_APPROVED_SLACK_ALERT: (
                slack_adapter
            )
        }
    )
    plan = _build_plan(
        steps=[
            _build_step(
                step_id="send_slack",
                tool_name=(
                    AgentToolName.SEND_APPROVED_SLACK_ALERT
                ),
                timeout_seconds=30,
            )
        ],
        intent=AgentIntent.ACTION_REQUEST,
    )

    with pytest.raises(
        AgentExecutionToolNotAllowedError,
        match="Side-effecting tools are disabled",
    ):
        await service.execute(
            plan,
            has_release_run_context=True,
            allow_side_effects=False,
            human_approval_granted=True,
        )

    assert slack_adapter.received_dependencies == []
