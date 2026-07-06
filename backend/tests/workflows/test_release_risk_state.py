"""Tests for release-risk workflow state models."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.workflows.release_risk_state import (
    ReleaseRiskState,
    ReleaseRiskWorkflowStage,
    ReleaseRiskWorkflowStatus,
)


def test_release_risk_state_defaults_to_initialized() -> None:
    """ReleaseRiskState should start in a safe initialized state."""
    state = ReleaseRiskState(
        release_run_id=uuid4(),
        run_id="test-run-001",
    )

    assert state.status == ReleaseRiskWorkflowStatus.NOT_STARTED
    assert state.stage == ReleaseRiskWorkflowStage.INITIALIZED
    assert state.github is None
    assert state.github_summary is None
    assert state.jira is None
    assert state.jira_summary is None
    assert state.release_summary is None
    assert state.completed_nodes == []
    assert state.failed_nodes == []
    assert state.errors == []
    assert state.has_errors is False
    assert state.is_terminal is False


def test_release_risk_state_accepts_existing_service_outputs() -> None:
    """ReleaseRiskState should hold current GitHub/Jira/release summary outputs."""
    state = ReleaseRiskState(
        release_run_id=uuid4(),
        run_id="test-run-002",
        github={"status": "completed", "risk_count": 2},
        github_summary={"total_risks": 2, "highest_severity": "high"},
        jira={"status": "completed", "risk_count": 1},
        jira_summary={"total_risks": 1, "highest_severity": "medium"},
        release_summary={
            "total_risks": 3,
            "highest_severity": "high",
            "recommendation": "review_before_deploy",
        },
    )

    assert state.github is not None
    assert state.github["risk_count"] == 2
    assert state.jira is not None
    assert state.jira["risk_count"] == 1
    assert state.release_summary is not None
    assert state.release_summary["total_risks"] == 3


def test_release_risk_state_rejects_blank_run_id() -> None:
    """ReleaseRiskState should reject blank run IDs."""
    with pytest.raises(ValidationError):
        ReleaseRiskState(
            release_run_id=uuid4(),
            run_id="   ",
        )


def test_release_risk_state_rejects_blank_manager_query() -> None:
    """ReleaseRiskState should reject blank manager queries."""
    with pytest.raises(ValidationError):
        ReleaseRiskState(
            release_run_id=uuid4(),
            run_id="test-run-003",
            manager_query="   ",
        )


def test_release_risk_state_rejects_unknown_fields() -> None:
    """ReleaseRiskState should reject unexpected fields to protect the contract."""
    with pytest.raises(ValidationError):
        ReleaseRiskState(
            release_run_id=uuid4(),
            run_id="test-run-004",
            unexpected_field="not allowed",
        )


def test_release_risk_state_can_mark_running() -> None:
    """ReleaseRiskState should support safe transition to running state."""
    state = ReleaseRiskState(
        release_run_id=uuid4(),
        run_id="test-run-005",
    )

    updated_state = state.mark_running(
        ReleaseRiskWorkflowStage.COLLECTING_GITHUB_RISKS,
    )

    assert updated_state.status == ReleaseRiskWorkflowStatus.RUNNING
    assert updated_state.stage == ReleaseRiskWorkflowStage.COLLECTING_GITHUB_RISKS
    assert state.status == ReleaseRiskWorkflowStatus.NOT_STARTED
    assert state.stage == ReleaseRiskWorkflowStage.INITIALIZED


def test_release_risk_state_can_mark_succeeded() -> None:
    """ReleaseRiskState should support safe transition to completed state."""
    state = ReleaseRiskState(
        release_run_id=uuid4(),
        run_id="test-run-006",
    ).mark_running(ReleaseRiskWorkflowStage.BUILDING_RELEASE_SUMMARY)

    updated_state = state.mark_succeeded()

    assert updated_state.status == ReleaseRiskWorkflowStatus.SUCCEEDED
    assert updated_state.stage == ReleaseRiskWorkflowStage.COMPLETED
    assert updated_state.is_terminal is True


def test_release_risk_state_can_track_completed_nodes() -> None:
    """ReleaseRiskState should track completed workflow nodes."""
    state = ReleaseRiskState(
        release_run_id=uuid4(),
        run_id="test-run-007",
    )

    updated_state = state.add_completed_node("collect_github_risks")

    assert updated_state.completed_nodes == ["collect_github_risks"]
    assert state.completed_nodes == []


def test_release_risk_state_rejects_blank_completed_node_name() -> None:
    """ReleaseRiskState should reject blank completed node names."""
    state = ReleaseRiskState(
        release_run_id=uuid4(),
        run_id="test-run-008",
    )

    with pytest.raises(ValueError, match="node_name must not be blank"):
        state.add_completed_node("   ")


def test_release_risk_state_adds_recoverable_error_as_partial() -> None:
    """Recoverable errors should move the workflow to PARTIAL status."""
    state = ReleaseRiskState(
        release_run_id=uuid4(),
        run_id="test-run-009",
    )

    updated_state = state.add_error(
        source="github_collector",
        message="GitHub API timeout",
        recoverable=True,
        details={"http_status": 504},
    )

    assert updated_state.status == ReleaseRiskWorkflowStatus.PARTIAL
    assert updated_state.stage == ReleaseRiskWorkflowStage.INITIALIZED
    assert updated_state.has_errors is True
    assert updated_state.is_terminal is False
    assert len(updated_state.errors) == 1
    assert updated_state.errors[0].source == "github_collector"
    assert updated_state.errors[0].recoverable is True
    assert updated_state.failed_nodes == ["github_collector"]


def test_release_risk_state_adds_non_recoverable_error_as_failed() -> None:
    """Non-recoverable errors should move the workflow to FAILED status."""
    state = ReleaseRiskState(
        release_run_id=uuid4(),
        run_id="test-run-010",
    ).mark_running(ReleaseRiskWorkflowStage.COLLECTING_JIRA_RISKS)

    updated_state = state.add_error(
        source="jira_collector",
        message="Invalid Jira configuration",
        recoverable=False,
        details={"config_key": "jira_base_url"},
    )

    assert updated_state.status == ReleaseRiskWorkflowStatus.FAILED
    assert updated_state.stage == ReleaseRiskWorkflowStage.FAILED
    assert updated_state.has_errors is True
    assert updated_state.is_terminal is True
    assert len(updated_state.errors) == 1
    assert updated_state.errors[0].source == "jira_collector"
    assert updated_state.errors[0].recoverable is False
    assert updated_state.failed_nodes == ["jira_collector"]