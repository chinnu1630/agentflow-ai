"""Deterministic evaluation for bounded dynamic-agent behavior."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.core.logging import get_logger
from app.schemas.agent_execution_plan import AgentExecutionPlan
from app.schemas.agent_query import (
    AgentIntent,
    AgentQueryPlan,
    AgentQueryRequest,
)
from app.schemas.agent_tool import AgentToolEffect, AgentToolName
from app.services.agent_query_router import AgentQueryRouter
from app.services.agent_tool_registry import (
    AgentToolNotRegisteredError,
    AgentToolRegistry,
)

logger = get_logger(__name__)

DynamicAgentEvaluationFailureReason = Literal[
    "intent_mismatch",
    "route_policy_mismatch",
    "planner_error",
    "tool_mismatch",
    "unsafe_tool",
]


class DynamicAgentEvaluationError(RuntimeError):
    """Raised when a dynamic-agent evaluation dependency fails."""


class DynamicAgentEvalCase(BaseModel):
    """One safe deterministic dynamic-agent golden case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=100)
    query: str = Field(min_length=1, max_length=2_000)
    expected_intent: AgentIntent
    expected_tool_name: AgentToolName | None = None
    release_run_context_available: bool = False
    dynamic_planning_allowed: bool = True
    expected_requires_human_approval: bool = False
    expected_may_execute_side_effect: bool = False


class DynamicAgentEvalFailureDetail(BaseModel):
    """Safe failure metadata without raw manager query content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_name: str
    query_length: int = Field(ge=0)
    reason: DynamicAgentEvaluationFailureReason
    expected_intent: AgentIntent
    actual_intent: AgentIntent | None = None
    expected_tool_name: AgentToolName | None = None
    actual_tool_names: list[AgentToolName] = Field(default_factory=list)
    error_type: str | None = None


class DynamicAgentEvaluationReport(BaseModel):
    """Aggregate deterministic dynamic-agent evaluation metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_cases: int = Field(ge=0)
    passed_cases: int = Field(ge=0)
    failed_cases: int = Field(ge=0)
    routing_accuracy: float = Field(ge=0.0, le=1.0)
    tool_accuracy: float = Field(ge=0.0, le=1.0)
    safety_accuracy: float = Field(ge=0.0, le=1.0)
    overall_accuracy: float = Field(ge=0.0, le=1.0)
    duration_ms: float = Field(ge=0.0)
    failed_case_details: list[DynamicAgentEvalFailureDetail] = Field(
        default_factory=list
    )


class DynamicAgentEvaluationPlanner(Protocol):
    """Planner capability required by deterministic evaluation."""

    async def create_plan(
        self,
        *,
        request: AgentQueryRequest,
        query_plan: AgentQueryPlan,
    ) -> AgentExecutionPlan:
        """Return one policy-validated execution plan."""


class _DynamicAgentCaseOutcome(BaseModel):
    """Internal result for one golden case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    routing_passed: bool
    tool_passed: bool
    safety_passed: bool
    overall_passed: bool
    failure_detail: DynamicAgentEvalFailureDetail | None = None


class DynamicAgentEvaluationService:
    """Evaluate routing, planning, and safety contracts in O(n) time."""

    _EVALUATION_RELEASE_RUN_ID = UUID(
        "11111111-1111-1111-1111-111111111111"
    )

    def __init__(
        self,
        *,
        router: AgentQueryRouter,
        planner: DynamicAgentEvaluationPlanner,
        registry: AgentToolRegistry | None = None,
    ) -> None:
        """Initialize the evaluator with deterministic dependencies."""
        self._router = router
        self._planner = planner
        self._registry = registry or AgentToolRegistry()

    async def evaluate(
        self,
        cases: Sequence[DynamicAgentEvalCase],
        *,
        run_id: UUID | None = None,
    ) -> DynamicAgentEvaluationReport:
        """Evaluate golden cases and return safe aggregate metrics."""
        started_at = time.perf_counter()
        outcomes = [
            await self._evaluate_case(case)
            for case in cases
        ]

        total_cases = len(outcomes)
        routing_hits = sum(
            outcome.routing_passed for outcome in outcomes
        )
        tool_hits = sum(outcome.tool_passed for outcome in outcomes)
        safety_hits = sum(
            outcome.safety_passed for outcome in outcomes
        )
        overall_hits = sum(
            outcome.overall_passed for outcome in outcomes
        )
        failures = [
            outcome.failure_detail
            for outcome in outcomes
            if outcome.failure_detail is not None
        ]

        report = DynamicAgentEvaluationReport(
            total_cases=total_cases,
            passed_cases=overall_hits,
            failed_cases=total_cases - overall_hits,
            routing_accuracy=self._safe_ratio(
                routing_hits,
                total_cases,
            ),
            tool_accuracy=self._safe_ratio(tool_hits, total_cases),
            safety_accuracy=self._safe_ratio(
                safety_hits,
                total_cases,
            ),
            overall_accuracy=self._safe_ratio(
                overall_hits,
                total_cases,
            ),
            duration_ms=round(
                (time.perf_counter() - started_at) * 1_000,
                3,
            ),
            failed_case_details=failures,
        )

        logger.info(
            "dynamic_agent_evaluation_completed",
            extra={
                "run_id": str(run_id) if run_id else None,
                "total_cases": report.total_cases,
                "passed_cases": report.passed_cases,
                "failed_cases": report.failed_cases,
                "routing_accuracy": report.routing_accuracy,
                "tool_accuracy": report.tool_accuracy,
                "safety_accuracy": report.safety_accuracy,
                "overall_accuracy": report.overall_accuracy,
                "duration_ms": report.duration_ms,
            },
        )

        return report

    async def _evaluate_case(
        self,
        case: DynamicAgentEvalCase,
    ) -> _DynamicAgentCaseOutcome:
        """Evaluate one golden case without exposing its raw query."""
        request = AgentQueryRequest(
            query=case.query,
            release_run_id=(
                self._EVALUATION_RELEASE_RUN_ID
                if case.release_run_context_available
                else None
            ),
        )
        query_plan = await self._router.create_plan(request)

        routing_passed = query_plan.intent is case.expected_intent
        safety_passed = (
            query_plan.requires_human_approval
            is case.expected_requires_human_approval
            and query_plan.may_execute_side_effect
            is case.expected_may_execute_side_effect
        )

        if not routing_passed:
            return self._failure(
                case=case,
                query_plan=query_plan,
                reason="intent_mismatch",
                routing_passed=False,
                tool_passed=False,
                safety_passed=safety_passed,
            )

        if not safety_passed:
            return self._failure(
                case=case,
                query_plan=query_plan,
                reason="route_policy_mismatch",
                routing_passed=True,
                tool_passed=False,
                safety_passed=False,
            )

        if not case.dynamic_planning_allowed:
            tool_passed = case.expected_tool_name is None
            return _DynamicAgentCaseOutcome(
                routing_passed=True,
                tool_passed=tool_passed,
                safety_passed=True,
                overall_passed=tool_passed,
                failure_detail=None,
            )

        try:
            execution_plan = await self._planner.create_plan(
                request=request,
                query_plan=query_plan,
            )
        except Exception as exc:
            return self._failure(
                case=case,
                query_plan=query_plan,
                reason="planner_error",
                routing_passed=True,
                tool_passed=False,
                safety_passed=True,
                error_type=type(exc).__name__,
            )

        actual_tool_names = [
            step.invocation.tool_name
            for step in execution_plan.steps
        ]
        expected_tools = (
            [case.expected_tool_name]
            if case.expected_tool_name is not None
            else []
        )
        tool_passed = actual_tool_names == expected_tools

        if not tool_passed:
            return self._failure(
                case=case,
                query_plan=query_plan,
                reason="tool_mismatch",
                routing_passed=True,
                tool_passed=False,
                safety_passed=True,
                actual_tool_names=actual_tool_names,
            )

        try:
            safe_tools = all(
                self._registry.get_definition(tool_name).effect
                is AgentToolEffect.READ_ONLY
                for tool_name in actual_tool_names
            )
        except AgentToolNotRegisteredError as exc:
            return self._failure(
                case=case,
                query_plan=query_plan,
                reason="unsafe_tool",
                routing_passed=True,
                tool_passed=True,
                safety_passed=False,
                actual_tool_names=actual_tool_names,
                error_type=type(exc).__name__,
            )

        if not safe_tools:
            return self._failure(
                case=case,
                query_plan=query_plan,
                reason="unsafe_tool",
                routing_passed=True,
                tool_passed=True,
                safety_passed=False,
                actual_tool_names=actual_tool_names,
            )

        return _DynamicAgentCaseOutcome(
            routing_passed=True,
            tool_passed=True,
            safety_passed=True,
            overall_passed=True,
        )

    @staticmethod
    def _failure(
        *,
        case: DynamicAgentEvalCase,
        query_plan: AgentQueryPlan,
        reason: DynamicAgentEvaluationFailureReason,
        routing_passed: bool,
        tool_passed: bool,
        safety_passed: bool,
        actual_tool_names: list[AgentToolName] | None = None,
        error_type: str | None = None,
    ) -> _DynamicAgentCaseOutcome:
        """Build a safe failed-case outcome."""
        return _DynamicAgentCaseOutcome(
            routing_passed=routing_passed,
            tool_passed=tool_passed,
            safety_passed=safety_passed,
            overall_passed=False,
            failure_detail=DynamicAgentEvalFailureDetail(
                case_name=case.name,
                query_length=len(case.query),
                reason=reason,
                expected_intent=case.expected_intent,
                actual_intent=query_plan.intent,
                expected_tool_name=case.expected_tool_name,
                actual_tool_names=actual_tool_names or [],
                error_type=error_type,
            ),
        )

    @staticmethod
    def _safe_ratio(numerator: int, denominator: int) -> float:
        """Return a division-safe evaluation ratio."""
        if denominator == 0:
            return 0.0

        return numerator / denominator
