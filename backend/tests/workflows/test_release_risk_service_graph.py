"""Tests for the service-backed release-risk LangGraph workflow."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ConfigDict

from app.workflows.release_risk_service_graph import (
    build_release_risk_service_graph,
)
from app.workflows.release_risk_state import (
    ReleaseRiskState,
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

    async def collect_release_risks(self, release_run_id: UUID) -> object | None:
        """Return the fake release-risk result."""
        self.called_with_release_run_id = release_run_id

        return self.result


@pytest.fixture
def anyio_backend() -> str:
    """Use asyncio backend for async graph tests."""
    return "asyncio"


@pytest.mark.anyio
async def test_service_graph_collects_release_risks_and_completes() -> None:
    """Service graph should collect release risks and finish successfully."""
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
    graph = build_release_risk_service_graph(fake_service)
    initial_state = ReleaseRiskState(
        release_run_id=release_run_id,
        run_id="test-service-graph-001",
    )

    result = await graph.ainvoke(initial_state.model_dump(mode="python"))
    final_state = ReleaseRiskState.model_validate(result)

    assert fake_service.called_with_release_run_id == release_run_id
    assert final_state.status == ReleaseRiskWorkflowStatus.SUCCEEDED
    assert final_state.stage == ReleaseRiskWorkflowStage.COMPLETED
    assert final_state.is_terminal is True
    assert final_state.has_errors is False
    assert final_state.release_run == {
        "id": str(release_run_id),
        "status": "created",
    }
    assert final_state.github == {"status": "completed", "risk_count": 2}
    assert final_state.github_summary == {
        "total_risks": 2,
        "highest_severity": "high",
    }
    assert final_state.jira == {"status": "completed", "risk_count": 1}
    assert final_state.jira_summary == {
        "total_risks": 1,
        "highest_severity": "medium",
    }
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
async def test_service_graph_stops_when_release_run_is_missing() -> None:
    """Service graph should stop as failed when the release run is not found."""
    release_run_id = uuid4()
    fake_service = FakeReleaseRiskService(None)
    graph = build_release_risk_service_graph(fake_service)
    initial_state = ReleaseRiskState(
        release_run_id=release_run_id,
        run_id="test-service-graph-002",
    )

    result = await graph.ainvoke(initial_state.model_dump(mode="python"))
    final_state = ReleaseRiskState.model_validate(result)

    assert fake_service.called_with_release_run_id == release_run_id
    assert final_state.status == ReleaseRiskWorkflowStatus.FAILED
    assert final_state.stage == ReleaseRiskWorkflowStage.FAILED
    assert final_state.is_terminal is True
    assert final_state.has_errors is True
    assert final_state.release_summary is None
    assert final_state.failed_nodes == ["release_run_service"]
    assert final_state.completed_nodes == [
        "start_release_risk_workflow",
    ]
    assert "complete_release_risk_workflow" not in final_state.completed_nodes
    assert len(final_state.errors) == 1
    assert final_state.errors[0].source == "release_run_service"
    assert final_state.errors[0].message == "Release run was not found."


@pytest.mark.anyio
async def test_service_graph_preserves_request_context() -> None:
    """Service graph should preserve run identity and requester metadata."""
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
    graph = build_release_risk_service_graph(fake_service)
    initial_state = ReleaseRiskState(
        release_run_id=release_run_id,
        run_id="test-service-graph-003",
        manager_query="What are the biggest release risks this week?",
        requested_by="manager@example.com",
    )

    result = await graph.ainvoke(initial_state.model_dump(mode="python"))
    final_state = ReleaseRiskState.model_validate(result)

    assert final_state.release_run_id == release_run_id
    assert final_state.run_id == "test-service-graph-003"
    assert final_state.manager_query == "What are the biggest release risks this week?"
    assert final_state.requested_by == "manager@example.com"
    assert final_state.status == ReleaseRiskWorkflowStatus.SUCCEEDED