"""Tests for service-backed release-risk workflow nodes."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ConfigDict

from app.workflows.release_risk_service_nodes import (
    create_collect_release_risks_node,
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
    """Fake release-risk service for workflow node tests."""

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
    """Use asyncio backend for async workflow node tests."""
    return "asyncio"


@pytest.mark.anyio
async def test_collect_release_risks_node_updates_state_from_pydantic_result() -> None:
    """Service-backed node should store release-risk outputs in workflow state."""
    release_run_id = uuid4()
    fake_result = FakeReleaseRiskResult(
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
    fake_service = FakeReleaseRiskService(fake_result)
    node = create_collect_release_risks_node(fake_service)
    initial_state = ReleaseRiskState(
        release_run_id=release_run_id,
        run_id="test-service-node-001",
    )

    result = await node(initial_state.model_dump(mode="python"))
    final_state = ReleaseRiskState.model_validate(result)

    assert fake_service.called_with_release_run_id == release_run_id
    assert final_state.status == ReleaseRiskWorkflowStatus.RUNNING
    assert final_state.stage == ReleaseRiskWorkflowStage.BUILDING_RELEASE_SUMMARY
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
    assert final_state.completed_nodes == ["collect_release_risks"]
    assert final_state.has_errors is False


@pytest.mark.anyio
async def test_collect_release_risks_node_updates_state_from_dict_result() -> None:
    """Service-backed node should also accept dictionary service results."""
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
    node = create_collect_release_risks_node(fake_service)
    initial_state = ReleaseRiskState(
        release_run_id=release_run_id,
        run_id="test-service-node-002",
    )

    result = await node(initial_state.model_dump(mode="python"))
    final_state = ReleaseRiskState.model_validate(result)

    assert final_state.release_run == {"id": str(release_run_id)}
    assert final_state.github == {"risk_count": 0}
    assert final_state.release_summary == {"total_risks": 0}
    assert final_state.completed_nodes == ["collect_release_risks"]


@pytest.mark.anyio
async def test_collect_release_risks_node_marks_missing_release_run_as_failed() -> None:
    """Missing release run should become a non-recoverable workflow failure."""
    release_run_id = uuid4()
    fake_service = FakeReleaseRiskService(None)
    node = create_collect_release_risks_node(fake_service)
    initial_state = ReleaseRiskState(
        release_run_id=release_run_id,
        run_id="test-service-node-003",
    )

    result = await node(initial_state.model_dump(mode="python"))
    final_state = ReleaseRiskState.model_validate(result)

    assert final_state.status == ReleaseRiskWorkflowStatus.FAILED
    assert final_state.stage == ReleaseRiskWorkflowStage.FAILED
    assert final_state.is_terminal is True
    assert final_state.has_errors is True
    assert final_state.failed_nodes == ["release_run_service"]
    assert len(final_state.errors) == 1
    assert final_state.errors[0].source == "release_run_service"
    assert final_state.errors[0].recoverable is False
    assert final_state.errors[0].details == {
        "release_run_id": str(release_run_id),
    }


@pytest.mark.anyio
async def test_collect_release_risks_node_rejects_invalid_payload_type() -> None:
    """Unexpected service result types should fail loudly in tests."""
    fake_service = FakeReleaseRiskService(object())
    node = create_collect_release_risks_node(fake_service)
    initial_state = ReleaseRiskState(
        release_run_id=uuid4(),
        run_id="test-service-node-004",
    )

    with pytest.raises(
        TypeError,
        match="collect_release_risks\\(\\) must return",
    ):
        await node(initial_state.model_dump(mode="python"))


@pytest.mark.anyio
async def test_collect_release_risks_node_rejects_invalid_nested_payload() -> None:
    """Nested result values must be dictionaries, Pydantic models, or None."""
    fake_service = FakeReleaseRiskService(
        {
            "release_run": {"id": "release-run-001"},
            "github": ["not", "a", "dict"],
            "github_summary": None,
            "jira": None,
            "jira_summary": None,
            "release_summary": None,
        }
    )
    node = create_collect_release_risks_node(fake_service)
    initial_state = ReleaseRiskState(
        release_run_id=uuid4(),
        run_id="test-service-node-005",
    )

    with pytest.raises(
        TypeError,
        match="github must be a dictionary",
    ):
        await node(initial_state.model_dump(mode="python"))