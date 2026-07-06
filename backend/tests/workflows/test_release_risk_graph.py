"""Tests for release-risk LangGraph assembly."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.workflows.release_risk_graph import (
    ReleaseRiskGraphNodeSet,
    WorkflowStateInput,
    WorkflowStateUpdate,
    build_release_risk_graph,
)
from app.workflows.release_risk_state import (
    ReleaseRiskState,
    ReleaseRiskWorkflowStage,
    ReleaseRiskWorkflowStatus,
)


@pytest.fixture
def anyio_backend() -> str:
    """Use asyncio backend for async graph tests."""
    return "asyncio"


def _validate_test_state(state: WorkflowStateInput) -> ReleaseRiskState:
    """Validate graph state for test helper nodes."""
    if isinstance(state, ReleaseRiskState):
        return state

    return ReleaseRiskState.model_validate(state)


def _custom_start_node(state: WorkflowStateInput) -> WorkflowStateUpdate:
    """Custom sync test node proving the graph supports node-set injection."""
    validated_state = _validate_test_state(state)

    return validated_state.add_completed_node("custom_start_node").model_dump(
        mode="python"
    )


async def _async_custom_prepare_github_node(
    state: WorkflowStateInput,
) -> WorkflowStateUpdate:
    """Custom async test node proving the graph supports async nodes."""
    validated_state = _validate_test_state(state)
    updated_state = validated_state.mark_running(
        ReleaseRiskWorkflowStage.COLLECTING_GITHUB_RISKS
    ).add_completed_node("async_custom_prepare_github_node")

    return updated_state.model_dump(mode="python")


def test_release_risk_graph_runs_linear_workflow() -> None:
    """Compiled graph should execute the initial workflow from start to complete."""
    graph = build_release_risk_graph()
    initial_state = ReleaseRiskState(
        release_run_id=uuid4(),
        run_id="test-graph-run-001",
    )

    result = graph.invoke(initial_state.model_dump(mode="python"))
    final_state = ReleaseRiskState.model_validate(result)

    assert final_state.status == ReleaseRiskWorkflowStatus.SUCCEEDED
    assert final_state.stage == ReleaseRiskWorkflowStage.COMPLETED
    assert final_state.is_terminal is True
    assert final_state.has_errors is False
    assert final_state.completed_nodes == [
        "start_release_risk_workflow",
        "prepare_github_risk_collection",
        "prepare_jira_risk_collection",
        "prepare_release_summary",
        "complete_release_risk_workflow",
    ]


def test_release_risk_graph_preserves_request_identity() -> None:
    """Compiled graph should preserve release_run_id, run_id, and requester metadata."""
    graph = build_release_risk_graph()
    release_run_id = uuid4()

    initial_state = ReleaseRiskState(
        release_run_id=release_run_id,
        run_id="test-graph-run-002",
        manager_query="What are the biggest release risks this week?",
        requested_by="engineering-manager@example.com",
    )

    result = graph.invoke(initial_state.model_dump(mode="python"))
    final_state = ReleaseRiskState.model_validate(result)

    assert final_state.release_run_id == release_run_id
    assert final_state.run_id == "test-graph-run-002"
    assert final_state.manager_query == "What are the biggest release risks this week?"
    assert final_state.requested_by == "engineering-manager@example.com"


def test_release_risk_graph_rejects_invalid_initial_state() -> None:
    """Compiled graph should reject invalid workflow input."""
    graph = build_release_risk_graph()

    with pytest.raises(ValidationError):
        graph.invoke(
            {
                "release_run_id": uuid4(),
                "run_id": "   ",
            }
        )


def test_release_risk_graph_accepts_custom_sync_node_set() -> None:
    """Graph builder should support injected sync node functions."""
    custom_nodes = ReleaseRiskGraphNodeSet(
        start=_custom_start_node,
    )
    graph = build_release_risk_graph(custom_nodes)

    initial_state = ReleaseRiskState(
        release_run_id=uuid4(),
        run_id="test-graph-run-003",
    )

    result: Any = graph.invoke(initial_state.model_dump(mode="python"))
    final_state = ReleaseRiskState.model_validate(result)

    assert final_state.status == ReleaseRiskWorkflowStatus.SUCCEEDED
    assert final_state.stage == ReleaseRiskWorkflowStage.COMPLETED
    assert final_state.completed_nodes == [
        "custom_start_node",
        "prepare_github_risk_collection",
        "prepare_jira_risk_collection",
        "prepare_release_summary",
        "complete_release_risk_workflow",
    ]


@pytest.mark.anyio
async def test_release_risk_graph_accepts_custom_async_node_set() -> None:
    """Graph builder should support injected async node functions."""
    custom_nodes = ReleaseRiskGraphNodeSet(
        prepare_github=_async_custom_prepare_github_node,
    )
    graph = build_release_risk_graph(custom_nodes)

    initial_state = ReleaseRiskState(
        release_run_id=uuid4(),
        run_id="test-graph-run-004",
    )

    result: Any = await graph.ainvoke(initial_state.model_dump(mode="python"))
    final_state = ReleaseRiskState.model_validate(result)

    assert final_state.status == ReleaseRiskWorkflowStatus.SUCCEEDED
    assert final_state.stage == ReleaseRiskWorkflowStage.COMPLETED
    assert final_state.completed_nodes == [
        "start_release_risk_workflow",
        "async_custom_prepare_github_node",
        "prepare_jira_risk_collection",
        "prepare_release_summary",
        "complete_release_risk_workflow",
    ]