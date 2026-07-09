"""Tests for the service-backed release-risk workflow runner."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from app.workflows.release_risk_service_runner import (
    ReleaseRiskServiceWorkflowRunner,
    run_release_risk_service_workflow,
)
from app.workflows.release_risk_state import (
    ReleaseRiskWorkflowStage,
    ReleaseRiskWorkflowStatus,
)


class FakeReleaseRiskResult(BaseModel):
    """Fake service result matching the release risk response shape."""

    model_config = ConfigDict(extra="forbid")

    release_run: dict[str, object]
    github: dict[str, object]
    github_summary: dict[str, object]
    jira: dict[str, object]
    jira_summary: dict[str, object]
    release_summary: dict[str, object]


class FakeReleaseRiskService:
    """Fake service implementing the release-risk collection protocol."""

    def __init__(self, result: object | None) -> None:
        """Initialize fake service with a predefined result."""
        self.result = result
        self.called_with_release_run_id: UUID | None = None
        self.call_count = 0

    async def collect_release_risks(self, release_run_id: UUID) -> object | None:
        """Return the fake release-risk result."""
        self.called_with_release_run_id = release_run_id
        self.call_count += 1

        return self.result


@pytest.fixture
def anyio_backend() -> str:
    """Use asyncio backend for async runner tests."""
    return "asyncio"


@pytest.mark.anyio
async def test_service_workflow_runner_returns_completed_state() -> None:
    """Runner should execute the service-backed graph successfully."""
    release_run_id = uuid4()
    fake_service = FakeReleaseRiskService(
        FakeReleaseRiskResult(
            release_run={"id": str(release_run_id), "status": "created"},
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
    )
    runner = ReleaseRiskServiceWorkflowRunner(fake_service)

    final_state = await runner.run(
        release_run_id=release_run_id,
        run_id="test-service-runner-001",
    )

    assert fake_service.called_with_release_run_id == release_run_id
    assert fake_service.call_count == 1
    assert final_state.release_run_id == release_run_id
    assert final_state.run_id == "test-service-runner-001"
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
async def test_service_workflow_runner_handles_missing_release_run() -> None:
    """Runner should return failed state when service returns None."""
    release_run_id = uuid4()
    fake_service = FakeReleaseRiskService(None)
    runner = ReleaseRiskServiceWorkflowRunner(fake_service)

    final_state = await runner.run(
        release_run_id=release_run_id,
        run_id="test-service-runner-002",
    )

    assert fake_service.called_with_release_run_id == release_run_id
    assert final_state.status == ReleaseRiskWorkflowStatus.FAILED
    assert final_state.stage == ReleaseRiskWorkflowStage.FAILED
    assert final_state.is_terminal is True
    assert final_state.has_errors is True
    assert final_state.failed_nodes == ["release_run_service"]
    assert final_state.completed_nodes == ["start_release_risk_workflow"]
    assert len(final_state.errors) == 1
    assert final_state.errors[0].message == "Release run was not found."


@pytest.mark.anyio
async def test_service_workflow_runner_preserves_manager_context() -> None:
    """Runner should preserve manager query and requester metadata."""
    release_run_id = uuid4()
    fake_service = FakeReleaseRiskService(
        {
            "release_run": {"id": str(release_run_id)},
            "github": {"risk_count": 0},
            "github_summary": {"total_risks": 0},
            "jira": {"risk_count": 0},
            "jira_summary": {"total_risks": 0},
            "release_summary": {"total_risks": 0},
        }
    )
    runner = ReleaseRiskServiceWorkflowRunner(fake_service)

    final_state = await runner.run(
        release_run_id=release_run_id,
        run_id="test-service-runner-003",
        manager_query="Are payment-service changes risky?",
        requested_by="manager@example.com",
    )

    assert final_state.release_run_id == release_run_id
    assert final_state.run_id == "test-service-runner-003"
    assert final_state.manager_query == "Are payment-service changes risky?"
    assert final_state.requested_by == "manager@example.com"
    assert final_state.status == ReleaseRiskWorkflowStatus.SUCCEEDED


@pytest.mark.anyio
async def test_service_workflow_runner_rejects_invalid_run_id() -> None:
    """Runner should reject invalid initial state before calling the service."""
    fake_service = FakeReleaseRiskService(None)
    runner = ReleaseRiskServiceWorkflowRunner(fake_service)

    with pytest.raises(ValidationError):
        await runner.run(
            release_run_id=uuid4(),
            run_id="   ",
        )

    assert fake_service.call_count == 0


@pytest.mark.anyio
async def test_run_release_risk_service_workflow_convenience_function() -> None:
    """Convenience function should execute the service-backed workflow."""
    release_run_id = uuid4()
    fake_service = FakeReleaseRiskService(
        {
            "release_run": {"id": str(release_run_id)},
            "github": {"risk_count": 0},
            "github_summary": {"total_risks": 0},
            "jira": {"risk_count": 0},
            "jira_summary": {"total_risks": 0},
            "release_summary": {"total_risks": 0},
        }
    )

    final_state = await run_release_risk_service_workflow(
        service=fake_service,
        release_run_id=release_run_id,
        run_id="test-service-runner-004",
    )

    assert fake_service.called_with_release_run_id == release_run_id
    assert final_state.status == ReleaseRiskWorkflowStatus.SUCCEEDED
    assert final_state.stage == ReleaseRiskWorkflowStage.COMPLETED