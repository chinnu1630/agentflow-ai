"""Tests for release-risk workflow node functions."""

from __future__ import annotations

from uuid import uuid4

from app.workflows.release_risk_nodes import (
    complete_release_risk_workflow,
    fail_release_risk_workflow,
    prepare_github_risk_collection,
    prepare_jira_risk_collection,
    prepare_release_summary,
    start_release_risk_workflow,
)
from app.workflows.release_risk_state import (
    ReleaseRiskState,
    ReleaseRiskWorkflowStage,
    ReleaseRiskWorkflowStatus,
)


def test_start_release_risk_workflow_marks_state_running() -> None:
    """Start node should move the workflow into running state."""
    state = ReleaseRiskState(
        release_run_id=uuid4(),
        run_id="test-run-001",
    )

    updated_state = start_release_risk_workflow(state)

    assert updated_state.status == ReleaseRiskWorkflowStatus.RUNNING
    assert updated_state.stage == ReleaseRiskWorkflowStage.INITIALIZED
    assert updated_state.completed_nodes == ["start_release_risk_workflow"]
    assert state.completed_nodes == []


def test_prepare_github_risk_collection_updates_stage() -> None:
    """GitHub preparation node should set the GitHub collection stage."""
    state = ReleaseRiskState(
        release_run_id=uuid4(),
        run_id="test-run-002",
    )

    updated_state = prepare_github_risk_collection(state)

    assert updated_state.status == ReleaseRiskWorkflowStatus.RUNNING
    assert updated_state.stage == ReleaseRiskWorkflowStage.COLLECTING_GITHUB_RISKS
    assert updated_state.completed_nodes == ["prepare_github_risk_collection"]


def test_prepare_jira_risk_collection_updates_stage() -> None:
    """Jira preparation node should set the Jira collection stage."""
    state = ReleaseRiskState(
        release_run_id=uuid4(),
        run_id="test-run-003",
    )

    updated_state = prepare_jira_risk_collection(state)

    assert updated_state.status == ReleaseRiskWorkflowStatus.RUNNING
    assert updated_state.stage == ReleaseRiskWorkflowStage.COLLECTING_JIRA_RISKS
    assert updated_state.completed_nodes == ["prepare_jira_risk_collection"]


def test_prepare_release_summary_updates_stage() -> None:
    """Release summary preparation node should set summary building stage."""
    state = ReleaseRiskState(
        release_run_id=uuid4(),
        run_id="test-run-004",
    )

    updated_state = prepare_release_summary(state)

    assert updated_state.status == ReleaseRiskWorkflowStatus.RUNNING
    assert updated_state.stage == ReleaseRiskWorkflowStage.BUILDING_RELEASE_SUMMARY
    assert updated_state.completed_nodes == ["prepare_release_summary"]


def test_complete_release_risk_workflow_marks_state_succeeded() -> None:
    """Completion node should mark workflow as succeeded and terminal."""
    state = ReleaseRiskState(
        release_run_id=uuid4(),
        run_id="test-run-005",
    )

    updated_state = complete_release_risk_workflow(state)

    assert updated_state.status == ReleaseRiskWorkflowStatus.SUCCEEDED
    assert updated_state.stage == ReleaseRiskWorkflowStage.COMPLETED
    assert updated_state.is_terminal is True
    assert updated_state.completed_nodes == ["complete_release_risk_workflow"]


def test_fail_release_risk_workflow_marks_state_failed() -> None:
    """Failure helper should mark workflow as failed with one error."""
    state = ReleaseRiskState(
        release_run_id=uuid4(),
        run_id="test-run-006",
    )

    updated_state = fail_release_risk_workflow(
        state,
        source="workflow_router",
        message="Invalid workflow transition",
        details={"stage": "unknown"},
    )

    assert updated_state.status == ReleaseRiskWorkflowStatus.FAILED
    assert updated_state.stage == ReleaseRiskWorkflowStage.FAILED
    assert updated_state.is_terminal is True
    assert updated_state.has_errors is True
    assert updated_state.failed_nodes == ["workflow_router"]
    assert len(updated_state.errors) == 1
    assert updated_state.errors[0].source == "workflow_router"
    assert updated_state.errors[0].message == "Invalid workflow transition"
    assert updated_state.errors[0].recoverable is False
    assert updated_state.errors[0].details == {"stage": "unknown"}


def test_nodes_can_be_chained_in_expected_order() -> None:
    """Workflow nodes should support simple sequential state transitions."""
    state = ReleaseRiskState(
        release_run_id=uuid4(),
        run_id="test-run-007",
    )

    updated_state = start_release_risk_workflow(state)
    updated_state = prepare_github_risk_collection(updated_state)
    updated_state = prepare_jira_risk_collection(updated_state)
    updated_state = prepare_release_summary(updated_state)
    updated_state = complete_release_risk_workflow(updated_state)

    assert updated_state.status == ReleaseRiskWorkflowStatus.SUCCEEDED
    assert updated_state.stage == ReleaseRiskWorkflowStage.COMPLETED
    assert updated_state.completed_nodes == [
        "start_release_risk_workflow",
        "prepare_github_risk_collection",
        "prepare_jira_risk_collection",
        "prepare_release_summary",
        "complete_release_risk_workflow",
    ]