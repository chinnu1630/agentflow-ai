"""Execute validated AgentFlow tool plans with bounded async concurrency."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Protocol

import structlog
from pydantic import ValidationError

from app.observability.tracing import start_business_span
from app.schemas.agent_execution_plan import (
    AgentExecutionPlan,
    AgentExecutionStep,
    AgentStepFailurePolicy,
)
from app.schemas.agent_execution_result import (
    AgentExecutionResult,
    AgentExecutionStatus,
)
from app.schemas.agent_tool import (
    AgentToolExecutionStatus,
    AgentToolInvocation,
    AgentToolName,
    AgentToolResult,
)
from app.schemas.agent_tool_arguments import (
    validate_agent_tool_arguments,
)
from app.services.agent_execution_plan_validator import (
    AgentExecutionPlanValidator,
)

logger = structlog.get_logger(__name__)


class AgentToolAdapterError(RuntimeError):
    """Base error raised by an AgentFlow tool adapter."""


class AgentToolAdapter(Protocol):
    """Execute one approved tool invocation."""

    async def execute(
        self,
        *,
        invocation: AgentToolInvocation,
        dependency_results: Mapping[str, AgentToolResult],
    ) -> AgentToolResult:
        """Return one normalized tool result."""
        ...


class AgentDynamicExecutionService:
    """Execute a validated tool DAG under deterministic enterprise limits."""

    def __init__(
        self,
        *,
        adapters: Mapping[AgentToolName, AgentToolAdapter],
        plan_validator: AgentExecutionPlanValidator,
        request_id: str,
    ) -> None:
        """Initialize the dynamic execution service.

        Args:
            adapters: Approved runtime adapters keyed by typed tool name.
            plan_validator: Deterministic policy validator.
            request_id: Request identifier used for logs and traces.
        """
        self._adapters = dict(adapters)
        self._plan_validator = plan_validator
        self._request_id = request_id

    async def execute(
        self,
        plan: AgentExecutionPlan,
        *,
        has_release_run_context: bool,
        allow_side_effects: bool = False,
        human_approval_granted: bool = False,
    ) -> AgentExecutionResult:
        """Execute one validated DAG and return an auditable aggregate result.

        Policy is revalidated immediately before execution. Independent
        dependency-ready steps run concurrently, bounded by the plan budget.
        Tool failures are normalized instead of escaping the execution engine.

        Args:
            plan: Planner-generated strict execution plan.
            has_release_run_context: Whether trusted release context exists.
            allow_side_effects: Whether the caller permits action tools.
            human_approval_granted: Whether durable approval was verified.

        Returns:
            Immutable aggregate execution result.

        Raises:
            AgentExecutionPlanValidationError: If deterministic policy fails.
        """
        validated_plan = self._plan_validator.validate(
            plan,
            has_release_run_context=has_release_run_context,
            allow_side_effects=allow_side_effects,
            human_approval_granted=human_approval_granted,
        )
        started_at = time.perf_counter()
        results_by_step: dict[str, AgentToolResult] = {}

        with start_business_span(
            "agent.dynamic_execution",
            {
                "run_id": self._request_id,
                "intent": validated_plan.intent.value,
                "step_count": len(validated_plan.steps),
                "max_parallel_steps": (
                    validated_plan.budget.max_parallel_steps
                ),
                "max_total_duration_seconds": (
                    validated_plan.budget.max_total_duration_seconds
                ),
                "allow_side_effects": allow_side_effects,
            },
        ) as span:
            try:
                async with asyncio.timeout(
                    validated_plan.budget.max_total_duration_seconds
                ):
                    await self._execute_dag(
                        plan=validated_plan,
                        results_by_step=results_by_step,
                    )
            except TimeoutError:
                self._mark_unexecuted_steps_failed(
                    plan=validated_plan,
                    results_by_step=results_by_step,
                    error_code="execution_timeout",
                    error_message=(
                        "The execution exceeded its total duration budget."
                    ),
                )

            ordered_results = [
                results_by_step[step.step_id]
                for step in validated_plan.steps
            ]
            execution_status = self._derive_execution_status(
                ordered_results
            )
            duration_ms = self._elapsed_ms(started_at)

            span.set_attribute(
                "agent.execution.status",
                execution_status.value,
            )
            span.set_attribute(
                "agent.execution.duration_ms",
                duration_ms,
            )
            span.set_attribute(
                "agent.execution.failed_step_count",
                sum(
                    result.status is AgentToolExecutionStatus.FAILED
                    for result in ordered_results
                ),
            )

        logger.info(
            "agent_dynamic_execution_completed",
            run_id=self._request_id,
            intent=validated_plan.intent.value,
            status=execution_status.value,
            step_count=len(ordered_results),
            failed_step_count=sum(
                result.status is AgentToolExecutionStatus.FAILED
                for result in ordered_results
            ),
            duration_ms=duration_ms,
        )

        return AgentExecutionResult(
            intent=validated_plan.intent,
            objective=validated_plan.objective,
            plan_reason_code=validated_plan.plan_reason_code,
            status=execution_status,
            tool_results=ordered_results,
            requires_synthesis=validated_plan.requires_synthesis,
            duration_ms=duration_ms,
        )

    async def _execute_dag(
        self,
        *,
        plan: AgentExecutionPlan,
        results_by_step: dict[str, AgentToolResult],
    ) -> None:
        """Execute dependency-ready DAG steps until completion or abort."""
        steps_by_id = {
            step.step_id: step
            for step in plan.steps
        }

        while len(results_by_step) < len(plan.steps):
            ready_steps = [
                step
                for step in plan.steps
                if (
                    step.step_id not in results_by_step
                    and all(
                        dependency_id in results_by_step
                        for dependency_id in step.depends_on
                    )
                )
            ]

            if not ready_steps:
                raise RuntimeError(
                    "Validated execution plan made no scheduling progress."
                )

            executable_steps: list[AgentExecutionStep] = []

            for step in ready_steps:
                failed_dependencies = [
                    dependency_id
                    for dependency_id in step.depends_on
                    if (
                        results_by_step[dependency_id].status
                        is AgentToolExecutionStatus.FAILED
                    )
                ]

                if failed_dependencies:
                    results_by_step[step.step_id] = (
                        self._build_failed_result(
                            step=step,
                            error_code="dependency_failed",
                            error_message=(
                                "A required dependency did not complete "
                                "successfully."
                            ),
                        )
                    )

                    if (
                        step.failure_policy
                        is AgentStepFailurePolicy.FAIL_PLAN
                    ):
                        self._mark_unexecuted_steps_failed(
                            plan=plan,
                            results_by_step=results_by_step,
                            error_code="plan_aborted",
                            error_message=(
                                "Execution stopped after a required step "
                                "failed."
                            ),
                        )
                        return
                else:
                    executable_steps.append(step)

            parallel_limit = plan.budget.max_parallel_steps

            for start_index in range(
                0,
                len(executable_steps),
                parallel_limit,
            ):
                batch = executable_steps[
                    start_index : start_index + parallel_limit
                ]
                batch_results = await self._execute_batch(
                    batch=batch,
                    results_by_step=results_by_step,
                )

                for result in batch_results:
                    results_by_step[result.step_id] = result

                must_abort = any(
                    (
                        result.status
                        is AgentToolExecutionStatus.FAILED
                        and steps_by_id[result.step_id].failure_policy
                        is AgentStepFailurePolicy.FAIL_PLAN
                    )
                    for result in batch_results
                )

                if must_abort:
                    self._mark_unexecuted_steps_failed(
                        plan=plan,
                        results_by_step=results_by_step,
                        error_code="plan_aborted",
                        error_message=(
                            "Execution stopped after a required step failed."
                        ),
                    )
                    return

    async def _execute_batch(
        self,
        *,
        batch: list[AgentExecutionStep],
        results_by_step: Mapping[str, AgentToolResult],
    ) -> list[AgentToolResult]:
        """Execute one bounded parallel batch using structured concurrency."""
        tasks: dict[str, asyncio.Task[AgentToolResult]] = {}

        async with asyncio.TaskGroup() as task_group:
            for step in batch:
                dependency_results = {
                    dependency_id: results_by_step[dependency_id]
                    for dependency_id in step.depends_on
                }
                tasks[step.step_id] = task_group.create_task(
                    self._execute_step(
                        step=step,
                        dependency_results=dependency_results,
                    )
                )

        return [
            tasks[step.step_id].result()
            for step in batch
        ]

    async def _execute_step(
        self,
        *,
        step: AgentExecutionStep,
        dependency_results: Mapping[str, AgentToolResult],
    ) -> AgentToolResult:
        """Execute one adapter with timeout, tracing, and safe normalization."""
        started_at = time.perf_counter()
        invocation = step.invocation

        try:
            validated_arguments = validate_agent_tool_arguments(
                tool_name=invocation.tool_name,
                arguments=dict(invocation.arguments),
            )
        except ValidationError:
            return self._build_failed_result(
                step=step,
                error_code="invalid_tool_arguments",
                error_message=(
                    "The planner supplied invalid arguments for this tool."
                ),
                duration_ms=self._elapsed_ms(started_at),
            )

        safe_invocation = invocation.model_copy(
            update={
                "arguments": validated_arguments.model_dump(mode="json")
            }
        )
        adapter = self._adapters.get(safe_invocation.tool_name)

        if adapter is None:
            return self._build_failed_result(
                step=step,
                error_code="tool_adapter_unavailable",
                error_message=(
                    "No runtime adapter is registered for this tool."
                ),
                duration_ms=self._elapsed_ms(started_at),
            )

        with start_business_span(
            "agent.tool_execution",
            {
                "run_id": self._request_id,
                "step_id": step.step_id,
                "tool_name": invocation.tool_name.value,
                "timeout_seconds": invocation.timeout_seconds,
                "dependency_count": len(step.depends_on),
            },
        ):
            try:
                async with asyncio.timeout(
                    invocation.timeout_seconds
                ):
                    result = await adapter.execute(
                        invocation=safe_invocation,
                        dependency_results=dependency_results,
                    )
            except TimeoutError:
                return self._build_failed_result(
                    step=step,
                    error_code="tool_timeout",
                    error_message=(
                        "The tool exceeded its execution timeout."
                    ),
                    duration_ms=self._elapsed_ms(started_at),
                )
            except AgentToolAdapterError:
                return self._build_failed_result(
                    step=step,
                    error_code="tool_execution_failed",
                    error_message=(
                        "The tool adapter could not complete the request."
                    ),
                    duration_ms=self._elapsed_ms(started_at),
                )
            except ValidationError:
                return self._build_failed_result(
                    step=step,
                    error_code="invalid_tool_result",
                    error_message=(
                        "The tool returned an invalid normalized result."
                    ),
                    duration_ms=self._elapsed_ms(started_at),
                )
            except Exception:
                logger.exception(
                    "agent_tool_unexpected_error",
                    run_id=self._request_id,
                    step_id=step.step_id,
                    tool_name=invocation.tool_name.value,
                )
                return self._build_failed_result(
                    step=step,
                    error_code="unexpected_tool_error",
                    error_message=(
                        "The tool encountered an unexpected internal error."
                    ),
                    duration_ms=self._elapsed_ms(started_at),
                )

        if (
            result.step_id != step.step_id
            or result.tool_name is not invocation.tool_name
        ):
            return self._build_failed_result(
                step=step,
                error_code="tool_result_identity_mismatch",
                error_message=(
                    "The tool result did not match the planned invocation."
                ),
                duration_ms=self._elapsed_ms(started_at),
            )

        return result.model_copy(
            update={"duration_ms": self._elapsed_ms(started_at)}
        )

    @staticmethod
    def _derive_execution_status(
        results: list[AgentToolResult],
    ) -> AgentExecutionStatus:
        """Derive aggregate status from normalized tool outcomes."""
        statuses = [result.status for result in results]

        if all(
            status is AgentToolExecutionStatus.SUCCESS
            for status in statuses
        ):
            return AgentExecutionStatus.SUCCESS

        if any(
            status
            in {
                AgentToolExecutionStatus.SUCCESS,
                AgentToolExecutionStatus.PARTIAL,
            }
            for status in statuses
        ):
            return AgentExecutionStatus.PARTIAL

        return AgentExecutionStatus.FAILED

    @classmethod
    def _mark_unexecuted_steps_failed(
        cls,
        *,
        plan: AgentExecutionPlan,
        results_by_step: dict[str, AgentToolResult],
        error_code: str,
        error_message: str,
    ) -> None:
        """Create auditable failure records for every unexecuted step."""
        for step in plan.steps:
            if step.step_id not in results_by_step:
                results_by_step[step.step_id] = cls._build_failed_result(
                    step=step,
                    error_code=error_code,
                    error_message=error_message,
                )

    @staticmethod
    def _build_failed_result(
        *,
        step: AgentExecutionStep,
        error_code: str,
        error_message: str,
        duration_ms: int = 0,
    ) -> AgentToolResult:
        """Build one safe normalized failed tool result."""
        return AgentToolResult(
            step_id=step.step_id,
            tool_name=step.invocation.tool_name,
            status=AgentToolExecutionStatus.FAILED,
            error_code=error_code,
            error_message=error_message,
            duration_ms=duration_ms,
        )

    @staticmethod
    def _elapsed_ms(started_at: float) -> int:
        """Return non-negative elapsed milliseconds."""
        return max(0, round((time.perf_counter() - started_at) * 1_000))
