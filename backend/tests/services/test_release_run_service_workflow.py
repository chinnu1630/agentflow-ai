"""Tests for ReleaseRunService LangGraph workflow integration."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from app.services.release_run_service import ReleaseRunService
from app.workflows.release_risk_state import (
    ReleaseRiskWorkflowStage,
    ReleaseRiskWorkflowStatus,
)


class WorkflowOnlyReleaseRunService(ReleaseRunService):
    """Minimal ReleaseRunService subclass for workflow integration tests."""

    def __init__(self, result: object | None) -> None:
        """Initialize fake service with a fixed release-risk result."""
        self._request_id = "test-request-id"
        self.result = result
        self.called_with_release_run_id: UUID | None = None
        self.call_count = 0

    async def collect_release_risks(self, release_run_id: UUID) -> object | None:
        """Return a fake release-risk result for the workflow node."""
        self.called_with_release_run_id = release_run_id
        self.call_count += 1

        return self.result


@pytest.fixture
def anyio_backend() -> str:
    """Use asyncio backend for async service workflow tests."""
    return "asyncio"


@pytest.mark.anyio
async def test_release_run_service_runs_langgraph_workflow_successfully() -> None:
    """ReleaseRunService should execute the service-backed LangGraph workflow."""
    release_run_id = uuid4()
    service = WorkflowOnlyReleaseRunService(
        {
            "release_run": {"id": str(release_run_id), "status": "completed"},
            "github": {"status": "completed", "risk_count": 2},
            "github_summary": {"total_risks": 2, "highest_severity": "high"},
            "jira": {"status": "completed", "risk_count": 1},
            "jira_summary": {"total_risks": 1, "highest_severity": "medium"},
            "release_summary": {
                "total_risks": 3,
                "highest_severity": "high",
                "recommendation": "review_before_deploy",
            },
        }
    )

    final_state = await service.run_release_risk_workflow(
        release_run_id=release_run_id,
        manager_query="What are the biggest release risks this week?",
        requested_by="manager@example.com",
    )

    assert service.called_with_release_run_id == release_run_id
    assert service.call_count == 1
    assert final_state.release_run_id == release_run_id
    assert final_state.run_id == "test-request-id"
    assert final_state.manager_query == "What are the biggest release risks this week?"
    assert final_state.requested_by == "manager@example.com"
    assert final_state.status == ReleaseRiskWorkflowStatus.SUCCEEDED
    assert final_state.stage == ReleaseRiskWorkflowStage.COMPLETED
    assert final_state.is_terminal is True
    assert final_state.has_errors is False
    assert final_state.release_summary == {
        "total_risks": 3,
        "highest_severity": "high",
        "recommendation": "review_before_deploy",
    }
    assert final_state.completed_nodes == [
        "start_release_risk_workflow",
        "collect_release_risks",
        "score_release_risk",
        "complete_release_risk_workflow",
    ]
    assert final_state.risk_features is not None
    assert final_state.risk_score is not None


@pytest.mark.anyio
async def test_release_run_service_workflow_handles_missing_release_run() -> None:
    """ReleaseRunService workflow should return failed state when release run is missing."""
    release_run_id = uuid4()
    service = WorkflowOnlyReleaseRunService(None)

    final_state = await service.run_release_risk_workflow(
        release_run_id=release_run_id,
    )

    assert service.called_with_release_run_id == release_run_id
    assert service.call_count == 1
    assert final_state.status == ReleaseRiskWorkflowStatus.FAILED
    assert final_state.stage == ReleaseRiskWorkflowStage.FAILED
    assert final_state.is_terminal is True
    assert final_state.has_errors is True
    assert final_state.failed_nodes == ["release_run_service"]
    assert final_state.completed_nodes == ["start_release_risk_workflow"]
    assert len(final_state.errors) == 1
    assert final_state.errors[0].message == "Release run was not found."


@pytest.mark.anyio
async def test_release_run_service_workflow_rejects_invalid_manager_query() -> None:
    """ReleaseRunService workflow should validate input before calling collectors."""
    service = WorkflowOnlyReleaseRunService(None)

    with pytest.raises(ValidationError):
        await service.run_release_risk_workflow(
            release_run_id=uuid4(),
            manager_query="   ",
        )

    assert service.call_count == 0