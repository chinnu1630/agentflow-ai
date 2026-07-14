"""Unit tests for the AgentFlow query executor."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

import pytest

import app.services.agent_query_executor as executor_module
from app.schemas.agent_query import (
    AgentIntent,
    AgentQueryPlan,
    AgentQueryRequest,
    ResponseDepth,
)
from app.schemas.risk import ReleaseRunRiskResponse
from app.services.agent_query_executor import (
    AgentQueryContextMismatchError,
    AgentQueryExecutor,
    AgentQueryResultError,
    UnsupportedAgentQueryIntentError,
)
from app.services.release_run_service import ReleaseRunResult


@pytest.fixture
def anyio_backend() -> str:
    """Use asyncio for async executor tests."""

    return "asyncio"


def build_plan(
    *,
    intent: AgentIntent = AgentIntent.RELEASE_RISK_SUMMARY,
    release_run_id: UUID | None = None,
) -> AgentQueryPlan:
    """Build a valid agent query plan."""

    return AgentQueryPlan(
        intent=intent,
        response_depth=ResponseDepth.STANDARD,
        confidence=1.0,
        release_run_id=release_run_id,
        requires_current_snapshot=True,
        routing_reason_code="test_route",
    )


def build_service(workflow_state: object) -> Mock:
    """Build a mocked release-run workflow service."""

    service = Mock()
    service.start_release_run = AsyncMock()
    service.run_release_risk_workflow = AsyncMock(
        return_value=workflow_state,
    )
    return service


def patch_response_mapping(
    monkeypatch: pytest.MonkeyPatch,
    expected_response: ReleaseRunRiskResponse,
) -> None:
    """Patch workflow extraction and public response mapping."""

    def extract_result(workflow_state: object) -> object:
        return workflow_state

    def map_response(result: object) -> ReleaseRunRiskResponse:
        del result
        return expected_response

    monkeypatch.setattr(
        executor_module,
        "extract_risk_result_from_workflow_state",
        extract_result,
    )
    monkeypatch.setattr(
        executor_module,
        "to_release_run_risk_response",
        map_response,
    )


@pytest.mark.anyio
async def test_executes_release_risk_summary_for_existing_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Executor should reuse an existing release run."""

    release_run_id = uuid4()
    workflow_state = {"result": "workflow-result"}
    service = build_service(workflow_state)

    expected_response = cast(
        ReleaseRunRiskResponse,
        Mock(spec=ReleaseRunRiskResponse),
    )
    expected_response.approval_required = False

    patch_response_mapping(monkeypatch, expected_response)

    executor = AgentQueryExecutor(
        release_run_service=service,
        request_id="request-123",
    )
    request = AgentQueryRequest(
        query="What are the biggest release risks this week?",
        release_run_id=release_run_id,
    )
    plan = build_plan(release_run_id=release_run_id)

    response = await executor.execute(
        request,
        plan,
        requested_by="manager@example.com",
    )

    assert response is expected_response
    service.start_release_run.assert_not_awaited()
    service.run_release_risk_workflow.assert_awaited_once_with(
        release_run_id,
        manager_query=request.query,
        requested_by="manager@example.com",
    )


@pytest.mark.anyio
async def test_creates_release_run_when_context_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Executor should create a release run when no run ID is provided."""

    created_release_run_id = uuid4()
    workflow_state = {"result": "workflow-result"}
    service = build_service(workflow_state)
    service.start_release_run.return_value = ReleaseRunResult(
        id=created_release_run_id,
        run_id="release-run-test",
        query="What are the biggest release risks this week?",
        requested_by="manager@example.com",
        status="created",
        created_at=datetime.now(UTC),
    )

    expected_response = cast(
        ReleaseRunRiskResponse,
        Mock(spec=ReleaseRunRiskResponse),
    )
    expected_response.approval_required = True

    patch_response_mapping(monkeypatch, expected_response)

    executor = AgentQueryExecutor(
        release_run_service=service,
        request_id="request-123",
    )
    request = AgentQueryRequest(query="What are the biggest release risks this week?")
    plan = build_plan()

    response = await executor.execute(
        request,
        plan,
        requested_by="manager@example.com",
    )

    assert response is expected_response
    service.start_release_run.assert_awaited_once()
    start_command = service.start_release_run.await_args.args[0]
    assert start_command.query == request.query
    assert start_command.requested_by == "manager@example.com"

    service.run_release_risk_workflow.assert_awaited_once_with(
        created_release_run_id,
        manager_query=request.query,
        requested_by="manager@example.com",
    )


@pytest.mark.anyio
async def test_rejects_unsupported_intent() -> None:
    """Executor should reject intents not implemented in this milestone."""

    service = build_service({"result": "unused"})
    executor = AgentQueryExecutor(
        release_run_service=service,
        request_id="request-123",
    )

    request = AgentQueryRequest(
        query="Why is the risk score high?",
        release_run_id=uuid4(),
    )
    plan = build_plan(
        intent=AgentIntent.EXPLAIN_RISK_SCORE,
        release_run_id=request.release_run_id,
    )

    with pytest.raises(
        UnsupportedAgentQueryIntentError,
        match="explain_risk_score",
    ):
        await executor.execute(
            request,
            plan,
            requested_by="manager@example.com",
        )

    service.start_release_run.assert_not_awaited()
    service.run_release_risk_workflow.assert_not_awaited()


@pytest.mark.anyio
async def test_rejects_mismatched_release_run_ids() -> None:
    """Executor should reject conflicting request and plan context."""

    service = build_service({"result": "unused"})
    executor = AgentQueryExecutor(
        release_run_service=service,
        request_id="request-123",
    )

    request = AgentQueryRequest(
        query="What are the biggest release risks?",
        release_run_id=uuid4(),
    )
    plan = build_plan(release_run_id=uuid4())

    with pytest.raises(
        AgentQueryContextMismatchError,
        match="do not match",
    ):
        await executor.execute(
            request,
            plan,
            requested_by="manager@example.com",
        )

    service.start_release_run.assert_not_awaited()
    service.run_release_risk_workflow.assert_not_awaited()


@pytest.mark.anyio
async def test_rejects_empty_workflow_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Executor should fail when workflow state has no usable result."""

    release_run_id = uuid4()
    service = build_service({"status": "failed"})

    monkeypatch.setattr(
        executor_module,
        "extract_risk_result_from_workflow_state",
        lambda workflow_state: None,
    )

    executor = AgentQueryExecutor(
        release_run_service=service,
        request_id="request-123",
    )
    request = AgentQueryRequest(
        query="What are the biggest release risks?",
        release_run_id=release_run_id,
    )
    plan = build_plan(release_run_id=release_run_id)

    with pytest.raises(
        AgentQueryResultError,
        match="no usable result",
    ):
        await executor.execute(
            request,
            plan,
            requested_by="manager@example.com",
        )
