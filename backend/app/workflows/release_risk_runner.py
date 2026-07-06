"""Workflow runner for AgentFlow AI release-risk orchestration.

This module provides the application-facing entry point for executing the
release-risk LangGraph workflow.

Current scope:
- Build validated initial workflow state
- Execute compiled LangGraph workflow asynchronously
- Return validated final workflow state

Future scope:
- Add OpenTelemetry spans around workflow execution
- Add durable LangGraph checkpointing for HITL approval
- Connect ReleaseRunService to this runner
- Add real GitHub/Jira/RAG/ML/synthesis nodes
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol, cast
from uuid import UUID

from app.workflows.release_risk_graph import build_release_risk_graph
from app.workflows.release_risk_state import ReleaseRiskState


class AsyncWorkflowGraph(Protocol):
    """Minimal protocol for a compiled graph with async invocation support."""

    async def ainvoke(self, input_data: dict[str, object]) -> object:
        """Execute the compiled graph asynchronously."""


@lru_cache(maxsize=1)
def get_release_risk_graph() -> AsyncWorkflowGraph:
    """Return a cached compiled release-risk workflow graph.

    The graph definition is static and does not store per-request state, so it
    is safe to compile once and reuse across workflow runs.
    """
    return cast(AsyncWorkflowGraph, build_release_risk_graph())


async def run_release_risk_workflow(
    *,
    release_run_id: UUID,
    run_id: str,
    manager_query: str = "What are the biggest release risks this week?",
    requested_by: str | None = None,
) -> ReleaseRiskState:
    """Execute the release-risk workflow and return the final validated state.

    Args:
        release_run_id: Database ID of the release run being analyzed.
        run_id: Correlation ID used for tracing and structured logs.
        manager_query: Original manager question.
        requested_by: Optional user or system actor that started the workflow.

    Returns:
        Final validated release-risk workflow state.

    Raises:
        pydantic.ValidationError: If the initial or final workflow state is invalid.
    """
    initial_state = ReleaseRiskState(
        release_run_id=release_run_id,
        run_id=run_id,
        manager_query=manager_query,
        requested_by=requested_by,
    )

    graph = get_release_risk_graph()
    graph_input = cast(dict[str, object], initial_state.model_dump(mode="python"))
    raw_result = await graph.ainvoke(graph_input)

    return ReleaseRiskState.model_validate(raw_result)