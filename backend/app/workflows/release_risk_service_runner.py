"""Application-facing runner for the service-backed release-risk workflow.

This module provides a clean boundary for executing the LangGraph workflow
that calls the existing ReleaseRunService.collect_release_risks() method.

Current scope:
- Build validated initial workflow state
- Execute the service-backed LangGraph graph asynchronously
- Return validated final workflow state

Future scope:
- Add OpenTelemetry spans around workflow execution
- Add structured logs with run_id and workflow status
- Add durable checkpointing for human approval
- Connect this runner from FastAPI or a dedicated orchestration service
"""

from __future__ import annotations

from typing import Protocol, cast
from uuid import UUID

from app.workflows.release_risk_service_graph import (
    build_release_risk_service_graph,
)
from app.workflows.release_risk_service_nodes import ReleaseRiskCollectionService
from app.workflows.release_risk_state import ReleaseRiskState


class AsyncWorkflowGraph(Protocol):
    """Minimal protocol for a compiled LangGraph graph with async execution."""

    async def ainvoke(self, input_data: dict[str, object]) -> object:
        """Execute the compiled graph asynchronously."""


class ReleaseRiskServiceWorkflowRunner:
    """Runner for executing the service-backed release-risk workflow.

    The runner compiles the graph once during initialization and reuses it for
    multiple workflow executions. This avoids recompiling the graph on every
    request while keeping per-request state isolated.
    """

    def __init__(self, service: ReleaseRiskCollectionService) -> None:
        """Initialize the runner with a release-risk collection service."""
        self._graph = cast(
            AsyncWorkflowGraph,
            build_release_risk_service_graph(service),
        )

    async def run(
        self,
        *,
        release_run_id: UUID,
        run_id: str,
        manager_query: str = "What are the biggest release risks this week?",
        requested_by: str | None = None,
    ) -> ReleaseRiskState:
        """Run the service-backed release-risk workflow.

        Args:
            release_run_id: Database UUID of the release run.
            run_id: Correlation ID used for logs and tracing.
            manager_query: Original manager question.
            requested_by: Optional user or system actor.

        Returns:
            Final validated workflow state.

        Raises:
            pydantic.ValidationError: If initial or final workflow state is invalid.
            TypeError: If the service returns an unsupported payload shape.
        """
        initial_state = ReleaseRiskState(
            release_run_id=release_run_id,
            run_id=run_id,
            manager_query=manager_query,
            requested_by=requested_by,
        )

        graph_input = cast(dict[str, object], initial_state.model_dump(mode="python"))
        raw_result = await self._graph.ainvoke(graph_input)

        return ReleaseRiskState.model_validate(raw_result)


async def run_release_risk_service_workflow(
    *,
    service: ReleaseRiskCollectionService,
    release_run_id: UUID,
    run_id: str,
    manager_query: str = "What are the biggest release risks this week?",
    requested_by: str | None = None,
) -> ReleaseRiskState:
    """Execute the service-backed release-risk workflow with a temporary runner.

    This convenience function is useful for simple integration points and tests.
    For high-throughput production paths, prefer creating one runner instance
    and reusing it through FastAPI dependency injection.
    """
    runner = ReleaseRiskServiceWorkflowRunner(service)

    return await runner.run(
        release_run_id=release_run_id,
        run_id=run_id,
        manager_query=manager_query,
        requested_by=requested_by,
    )