"""Tests for release-risk LangGraph assembly."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.workflows.release_risk_graph import build_release_risk_graph
from app.workflows.release_risk_state import (
    ReleaseRiskState,
    ReleaseRiskWorkflowStage,
    ReleaseRiskWorkflowStatus,
)


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