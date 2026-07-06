"""Tests for release-risk workflow runner."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.workflows.release_risk_runner import (
    get_release_risk_graph,
    run_release_risk_workflow,
)
from app.workflows.release_risk_state import (
    ReleaseRiskWorkflowStage,
    ReleaseRiskWorkflowStatus,
)


@pytest.fixture
def anyio_backend() -> str:
    """Use asyncio backend for async workflow tests."""
    return "asyncio"


@pytest.mark.anyio
async def test_run_release_risk_workflow_returns_completed_state() -> None:
    """Workflow runner should execute the graph and return completed state."""
    release_run_id = uuid4()

    final_state = await run_release_risk_workflow(
        release_run_id=release_run_id,
        run_id="test-runner-001",
    )

    assert final_state.release_run_id == release_run_id
    assert final_state.run_id == "test-runner-001"
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


@pytest.mark.anyio
async def test_run_release_risk_workflow_preserves_manager_context() -> None:
    """Workflow runner should preserve manager query and requester metadata."""
    release_run_id = uuid4()

    final_state = await run_release_risk_workflow(
        release_run_id=release_run_id,
        run_id="test-runner-002",
        manager_query="Are there any risky payment-service changes?",
        requested_by="manager@example.com",
    )

    assert final_state.release_run_id == release_run_id
    assert final_state.run_id == "test-runner-002"
    assert final_state.manager_query == "Are there any risky payment-service changes?"
    assert final_state.requested_by == "manager@example.com"


@pytest.mark.anyio
async def test_run_release_risk_workflow_rejects_invalid_run_id() -> None:
    """Workflow runner should reject invalid initial state before graph execution."""
    with pytest.raises(ValidationError):
        await run_release_risk_workflow(
            release_run_id=uuid4(),
            run_id="   ",
        )


def test_get_release_risk_graph_returns_cached_graph() -> None:
    """Compiled workflow graph should be cached and reused."""
    first_graph = get_release_risk_graph()
    second_graph = get_release_risk_graph()

    assert first_graph is second_graph