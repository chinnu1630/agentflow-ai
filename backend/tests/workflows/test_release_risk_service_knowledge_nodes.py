"""Tests for Knowledge Agent integration in the release-risk service workflow."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from uuid import UUID, uuid4

import pytest

from app.workflows import release_risk_service_nodes
from app.workflows.release_risk_service_graph import build_release_risk_service_graph
from app.workflows.release_risk_service_nodes import (
    create_retrieve_knowledge_context_node,
)
from app.workflows.release_risk_state import (
    KnowledgeRetrievalStatus,
    ReleaseRiskState,
    ReleaseRiskWorkflowStatus,
)


class CapturingLogger:
    """Capture structured log calls for workflow node tests."""

    def __init__(self) -> None:
        """Initialize captured log event storage."""
        self.info_events: list[tuple[str, dict[str, Any]]] = []
        self.warning_events: list[tuple[str, dict[str, Any]]] = []

    def info(self, event: str, **fields: Any) -> None:
        """Capture an info-level structured log event."""
        self.info_events.append((event, fields))

    def warning(self, event: str, **fields: Any) -> None:
        """Capture a warning-level structured log event."""
        self.warning_events.append((event, fields))


class FakeKnowledgeRetrievalService:
    """Fake Knowledge retrieval service used by workflow tests."""

    def __init__(self) -> None:
        """Initialize the fake service with captured request state."""
        self.received_query: str | None = None
        self.received_run_id: str | None = None

    async def retrieve_relevant_chunks(
        self,
        retrieval_request: object,
        *,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Return deterministic fake Knowledge retrieval results."""
        self.received_query = retrieval_request.query
        self.received_run_id = run_id

        return {
            "query": self.received_query,
            "total_candidates": 1,
            "results": [
                {
                    "document_id": uuid4(),
                    "chunk_id": uuid4(),
                    "title": "Payment Service Runbook",
                    "source_type": "runbook",
                    "source_uri": "docs/payment-service-runbook.md",
                    "chunk_index": 0,
                    "score": 1.25,
                    "content": "Redis latency can increase checkout failure risk.",
                    "token_count": 7,
                    "metadata_json": {"team": "payments"},
                }
            ],
        }



class EmptyKnowledgeRetrievalService:
    """Fake Knowledge retrieval service that returns no matching chunks."""

    async def retrieve_relevant_chunks(
        self,
        retrieval_request: object,
        *,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Return an empty but successful Knowledge retrieval response."""
        return {
            "query": retrieval_request.query,
            "total_candidates": 0,
            "results": [],
        }


class FailingKnowledgeRetrievalService:
    """Fake Knowledge retrieval service that raises a recoverable error."""

    async def retrieve_relevant_chunks(
        self,
        retrieval_request: object,
        *,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Raise a deterministic retrieval error."""
        raise ValueError("simulated retrieval failure")


class FakeReleaseRiskCollectionService:
    """Fake release-risk collection service used by graph tests."""

    def __init__(self) -> None:
        """Initialize the fake service with captured release run ID."""
        self.received_release_run_id: UUID | None = None

    async def collect_release_risks(
        self,
        release_run_id: UUID,
    ) -> dict[str, Any]:
        """Return deterministic fake release-risk collection output."""
        self.received_release_run_id = release_run_id

        return {
            "release_run": {
                "id": release_run_id,
                "name": "Weekly Payments Release",
                "status": "running",
            },
            "github": {
                "risks": [
                    {
                        "title": "Redis checkout failure risk",
                        "severity": "high",
                        "description": "Payment checkout path has Redis latency risk.",
                    }
                ]
            },
            "github_summary": {
                "overall_status": "high_risk",
                "recommended_action": "review_before_deploy",
                "summary": "Redis checkout failure risk detected.",
            },
            "jira": {
                "risks": [
                    {
                        "title": "Checkout failures reported in Jira",
                        "severity": "high",
                        "description": "P1 ticket mentions checkout failures.",
                    }
                ]
            },
            "jira_summary": {
                "overall_status": "high_risk",
                "recommended_action": "investigate_p1_tickets",
                "summary": "Jira has high-priority checkout failure tickets.",
            },
            "release_summary": {
                "overall_status": "high_risk",
                "recommended_action": "do_not_deploy_without_review",
                "summary": "Release has Redis checkout failure risk.",
            },
        }


def _workflow_state() -> ReleaseRiskState:
    """Build a valid workflow state for Knowledge node tests."""
    return ReleaseRiskState(
        release_run_id=uuid4(),
        run_id="test-run-id",
        manager_query="What are the biggest release risks this week?",
        github_summary={
            "summary": "Redis checkout failure risk detected.",
            "recommended_action": "review_before_deploy",
        },
        jira_summary={
            "summary": "P1 checkout failures are open.",
            "recommended_action": "investigate_p1_tickets",
        },
        release_summary={
            "summary": "Release has Redis checkout failure risk.",
            "overall_status": "high_risk",
        },
    )


@pytest.mark.asyncio
async def test_retrieve_knowledge_context_node_adds_results_to_state() -> None:
    """Knowledge node should retrieve document evidence and store it in state."""
    knowledge_service = FakeKnowledgeRetrievalService()
    node = create_retrieve_knowledge_context_node(knowledge_service)

    result = await node(_workflow_state().model_dump(mode="python"))

    final_state = ReleaseRiskState.model_validate(result)

    assert final_state.knowledge_status == KnowledgeRetrievalStatus.COMPLETED
    assert final_state.knowledge_error is None
    assert len(final_state.knowledge_results) == 1
    assert final_state.knowledge_results[0]["title"] == "Payment Service Runbook"
    assert "retrieve_knowledge_context" in final_state.completed_nodes
    assert knowledge_service.received_run_id == "test-run-id"
    assert knowledge_service.received_query is not None
    assert "Redis checkout failure" in knowledge_service.received_query



@pytest.mark.asyncio
async def test_retrieve_knowledge_context_node_uses_validated_span_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Knowledge span should receive identifiers from validated workflow state."""
    captured_span_name: str | None = None
    captured_attributes: dict[str, object] = {}

    @contextmanager
    def capture_business_span(
        span_name: str,
        attributes: dict[str, object],
    ) -> Iterator[None]:
        """Capture business span arguments without starting a real trace."""
        nonlocal captured_span_name, captured_attributes

        captured_span_name = span_name
        captured_attributes = attributes

        yield

    monkeypatch.setattr(
        release_risk_service_nodes,
        "start_business_span",
        capture_business_span,
    )

    workflow_state = _workflow_state()
    node = create_retrieve_knowledge_context_node(
        FakeKnowledgeRetrievalService()
    )

    await node(workflow_state.model_dump(mode="python"))

    assert captured_span_name == "knowledge.retrieve"
    assert captured_attributes == {
        "release_run_id": str(workflow_state.release_run_id),
        "run_id": workflow_state.run_id,
        "query_present": True,
    }


@pytest.mark.asyncio
async def test_retrieve_knowledge_context_node_sets_no_results_when_empty() -> None:
    """Knowledge node should distinguish empty retrieval from retrieval failure."""
    node = create_retrieve_knowledge_context_node(
        EmptyKnowledgeRetrievalService()
    )

    result = await node(_workflow_state().model_dump(mode="python"))

    final_state = ReleaseRiskState.model_validate(result)

    assert final_state.knowledge_status == KnowledgeRetrievalStatus.NO_RESULTS
    assert final_state.knowledge_error is None
    assert final_state.knowledge_results == []
    assert final_state.has_errors is False
    assert "retrieve_knowledge_context" in final_state.completed_nodes



@pytest.mark.asyncio
async def test_retrieve_knowledge_context_node_logs_safe_success_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Knowledge node should log safe metadata when retrieval succeeds."""
    capturing_logger = CapturingLogger()
    monkeypatch.setattr(
        release_risk_service_nodes,
        "logger",
        capturing_logger,
    )

    node = create_retrieve_knowledge_context_node(FakeKnowledgeRetrievalService())

    result = await node(_workflow_state().model_dump(mode="python"))
    final_state = ReleaseRiskState.model_validate(result)

    assert final_state.knowledge_status == KnowledgeRetrievalStatus.COMPLETED

    assert len(capturing_logger.info_events) == 1
    event_name, fields = capturing_logger.info_events[0]

    assert event_name == "knowledge_retrieval_node_completed"
    assert fields["run_id"] == "test-run-id"
    assert fields["release_run_id"] == str(final_state.release_run_id)
    assert fields["knowledge_status"] == "completed"
    assert fields["result_count"] == 1
    assert fields["query_length"] > 0

    serialized_fields = str(fields)

    assert "Redis checkout failure" not in serialized_fields
    assert "Redis latency can increase checkout failure risk" not in serialized_fields
    assert "query" not in fields
    assert "content" not in fields


@pytest.mark.asyncio
async def test_retrieve_knowledge_context_node_degrades_gracefully_on_failure() -> None:
    """Knowledge node failure should be recoverable and keep workflow usable."""
    node = create_retrieve_knowledge_context_node(FailingKnowledgeRetrievalService())

    result = await node(_workflow_state().model_dump(mode="python"))

    final_state = ReleaseRiskState.model_validate(result)

    assert final_state.status == ReleaseRiskWorkflowStatus.PARTIAL
    assert final_state.knowledge_status == KnowledgeRetrievalStatus.FAILED
    assert final_state.knowledge_error == "Knowledge retrieval failed."
    assert final_state.knowledge_results == []
    assert final_state.has_errors is True
    assert final_state.errors[0].source == "knowledge_retrieval"
    assert final_state.errors[0].recoverable is True



@pytest.mark.asyncio
async def test_retrieve_knowledge_context_node_logs_safe_failure_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Knowledge node should log safe metadata when retrieval fails."""
    capturing_logger = CapturingLogger()
    monkeypatch.setattr(
        release_risk_service_nodes,
        "logger",
        capturing_logger,
    )

    node = create_retrieve_knowledge_context_node(FailingKnowledgeRetrievalService())

    result = await node(_workflow_state().model_dump(mode="python"))
    final_state = ReleaseRiskState.model_validate(result)

    assert final_state.knowledge_status == KnowledgeRetrievalStatus.FAILED

    assert len(capturing_logger.warning_events) == 1
    event_name, fields = capturing_logger.warning_events[0]

    assert event_name == "knowledge_retrieval_node_failed"
    assert fields["run_id"] == "test-run-id"
    assert fields["release_run_id"] == str(final_state.release_run_id)
    assert fields["error_type"] == "ValueError"
    assert fields["query_length"] > 0

    serialized_fields = str(fields)

    assert "simulated retrieval failure" not in serialized_fields
    assert "Redis checkout failure" not in serialized_fields
    assert "query" not in fields
    assert "content" not in fields
    assert "error_message" not in fields


@pytest.mark.asyncio
async def test_service_graph_runs_optional_knowledge_node_when_configured() -> None:
    """Service graph should run Knowledge retrieval when service is provided."""
    release_service = FakeReleaseRiskCollectionService()
    knowledge_service = FakeKnowledgeRetrievalService()
    graph = build_release_risk_service_graph(
        release_service,
        knowledge_service=knowledge_service,
    )

    initial_state = ReleaseRiskState(
        release_run_id=uuid4(),
        run_id="test-run-id",
        manager_query="What are the biggest release risks this week?",
    )

    raw_result = await graph.ainvoke(initial_state.model_dump(mode="python"))
    final_state = ReleaseRiskState.model_validate(raw_result)

    assert final_state.status == ReleaseRiskWorkflowStatus.SUCCEEDED
    assert final_state.knowledge_status == KnowledgeRetrievalStatus.COMPLETED
    assert len(final_state.knowledge_results) == 1
    assert "collect_release_risks" in final_state.completed_nodes
    assert "retrieve_knowledge_context" in final_state.completed_nodes
    assert release_service.received_release_run_id == initial_state.release_run_id
    assert knowledge_service.received_query is not None
    assert "Redis checkout failure" in knowledge_service.received_query



@pytest.mark.asyncio
async def test_service_graph_completes_when_knowledge_returns_no_results() -> None:
    """Service graph should complete when Knowledge retrieval returns no results."""
    graph = build_release_risk_service_graph(
        FakeReleaseRiskCollectionService(),
        knowledge_service=EmptyKnowledgeRetrievalService(),
    )

    initial_state = ReleaseRiskState(
        release_run_id=uuid4(),
        run_id="test-run-id",
        manager_query="What are the biggest release risks this week?",
    )

    raw_result = await graph.ainvoke(initial_state.model_dump(mode="python"))
    final_state = ReleaseRiskState.model_validate(raw_result)

    assert final_state.status == ReleaseRiskWorkflowStatus.SUCCEEDED
    assert final_state.knowledge_status == KnowledgeRetrievalStatus.NO_RESULTS
    assert final_state.knowledge_error is None
    assert final_state.knowledge_results == []
    assert final_state.has_errors is False
    assert "collect_release_risks" in final_state.completed_nodes
    assert "retrieve_knowledge_context" in final_state.completed_nodes


@pytest.mark.asyncio
async def test_service_graph_still_completes_when_knowledge_node_fails() -> None:
    """Service graph should complete with PARTIAL status if Knowledge fails."""
    graph = build_release_risk_service_graph(
        FakeReleaseRiskCollectionService(),
        knowledge_service=FailingKnowledgeRetrievalService(),
    )

    initial_state = ReleaseRiskState(
        release_run_id=uuid4(),
        run_id="test-run-id",
    )

    raw_result = await graph.ainvoke(initial_state.model_dump(mode="python"))
    final_state = ReleaseRiskState.model_validate(raw_result)

    assert final_state.status == ReleaseRiskWorkflowStatus.SUCCEEDED
    assert final_state.knowledge_status == KnowledgeRetrievalStatus.FAILED
    assert final_state.knowledge_results == []
    assert final_state.has_errors is True
    assert final_state.errors[0].source == "knowledge_retrieval"
