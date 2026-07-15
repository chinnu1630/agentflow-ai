"""Integration tests for release run API routes."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.api.routes.release_runs import (
    get_jira_risk_collector,
    get_risk_collector,
    get_slack_alert_sender,
)
from app.db.base import Base
from app.db.session import get_db_session
from app.integrations.slack_client import SlackPostMessageResult
from app.main import app
from app.repositories.release_run_risk_snapshot_repository import (
    CreateReleaseRunRiskSnapshotCommand,
    ReleaseRunRiskSnapshotRepository,
)
from app.services.engineering_document_retrieval_service import EngineeringDocumentRetrievalService
from app.services.github_risk_collector import GitHubRiskCollectionResult, RiskCollectionStatus
from app.services.jira_risk_collector import (
    JiraRiskCollectionResult,
    JiraRiskCollectionStatus,
)
from app.services.release_run_service import ReleaseRunService
from app.services.slack_alert_payload_service import SlackReleaseRiskAlertPayload


class FakeRiskCollector:
    """Fake GitHub risk collector used to avoid real GitHub API calls."""

    async def collect_github_risks(
        self,
        *,
        run_id: str,
    ) -> GitHubRiskCollectionResult:
        """Return a deterministic fake GitHub risk collection result."""

        return GitHubRiskCollectionResult(
            status=RiskCollectionStatus.SUCCESS,
            pull_request_count=2,
            risk_result_count=2,
            total_signal_count=3,
            high_risk_count=1,
            risk_results=[],
            collected_at=datetime.now(UTC),
            duration_ms=10.5,
        )


class FakeDegradedRiskCollector:
    """Fake GitHub collector that simulates degraded GitHub availability."""

    async def collect_github_risks(
        self,
        *,
        run_id: str,
    ) -> GitHubRiskCollectionResult:
        """Return degraded GitHub risk collection output."""

        return GitHubRiskCollectionResult(
            status=RiskCollectionStatus.DEGRADED,
            pull_request_count=0,
            risk_result_count=0,
            total_signal_count=0,
            high_risk_count=0,
            risk_results=[],
            error_type="GitHubClientError",
            error_message="GitHub unavailable.",
            collected_at=datetime.now(UTC),
            duration_ms=1.0,
        )


class FakeJiraRiskCollector:
    """Fake Jira risk collector used to avoid real Jira API calls."""

    async def collect(self, *, run_id: str) -> JiraRiskCollectionResult:
        """Return an empty successful Jira risk collection result."""

        return JiraRiskCollectionResult(
            status=JiraRiskCollectionStatus.SUCCESS,
            issues=[],
            issue_results=[],
            signals=[],
            error_message=None,
            duration_ms=0.0,
        )


class FakeSlackAlertSender:
    """Fake Slack sender used to avoid real Slack API calls."""

    def __init__(self) -> None:
        """Initialize fake sender with captured payload list."""
        self.sent_payloads: list[SlackReleaseRiskAlertPayload] = []

    async def send_release_risk_alert(
        self,
        payload: SlackReleaseRiskAlertPayload,
    ) -> SlackPostMessageResult:
        """Capture Slack payload and return deterministic fake result."""
        self.sent_payloads.append(payload)

        return SlackPostMessageResult(
            ok=True,
            channel="C1234567890",
            timestamp="12345.6789",
        )


def override_external_collectors_for_test() -> None:
    """Override external GitHub and Jira collectors for API tests."""

    async def override_get_risk_collector() -> FakeRiskCollector:
        """Override GitHub collector dependency for API tests."""

        return FakeRiskCollector()

    async def override_get_jira_risk_collector() -> FakeJiraRiskCollector:
        """Override Jira collector dependency for API tests."""

        return FakeJiraRiskCollector()

    app.dependency_overrides[get_risk_collector] = override_get_risk_collector
    app.dependency_overrides[get_jira_risk_collector] = override_get_jira_risk_collector


def override_slack_alert_sender_for_test(
    sender: FakeSlackAlertSender,
) -> None:
    """Override Slack sender dependency for API tests."""

    async def override_get_slack_alert_sender() -> FakeSlackAlertSender:
        """Return fake Slack sender instead of real Slack client."""
        return sender

    app.dependency_overrides[get_slack_alert_sender] = override_get_slack_alert_sender


def _assert_risk_scoring_response(response_data: dict[str, Any]) -> None:
    """Assert release-risk API response includes deterministic scoring output."""

    risk_features = response_data["risk_features"]
    risk_score = response_data["risk_score"]

    assert risk_features["feature_version"] == "release_risk_features_v1"
    assert isinstance(risk_features["total_risk_count"], int)
    assert isinstance(risk_features["github_risk_count"], int)
    assert isinstance(risk_features["jira_risk_count"], int)
    assert isinstance(risk_features["knowledge_result_count"], int)
    assert isinstance(risk_features["knowledge_failed"], bool)

    assert risk_score["scoring_version"] == "rule_based_release_risk_v1"
    assert risk_score["feature_version"] == "release_risk_features_v1"
    assert 0.0 <= risk_score["score"] <= 1.0
    assert risk_score["risk_level"] in {
        "low",
        "medium",
        "high",
        "critical",
    }
    assert risk_score["recommended_action"] in {
        "proceed",
        "review_required",
        "block_release",
        "partial_data_review",
    }
    assert isinstance(risk_score["reasons"], list)
    assert risk_score["reasons"]
    assert isinstance(risk_score["component_scores"], dict)


def override_degraded_github_collector_for_test() -> None:
    """Override GitHub collector with degraded output for approval tests."""

    async def override_get_risk_collector() -> FakeDegradedRiskCollector:
        """Override GitHub collector dependency with degraded fake."""

        return FakeDegradedRiskCollector()

    async def override_get_jira_risk_collector() -> FakeJiraRiskCollector:
        """Override Jira collector dependency for API tests."""

        return FakeJiraRiskCollector()

    app.dependency_overrides[get_risk_collector] = override_get_risk_collector
    app.dependency_overrides[get_jira_risk_collector] = override_get_jira_risk_collector


@pytest.fixture
async def release_run_api_client() -> AsyncIterator[AsyncClient]:
    """Create an API client with an isolated in-memory test database."""

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async def override_get_db_session() -> AsyncIterator[AsyncSession]:
        """Override FastAPI database dependency for tests."""

        async with session_factory() as session:
            try:
                yield session
            finally:
                await session.close()

    app.dependency_overrides[get_db_session] = override_get_db_session

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client

    app.dependency_overrides.clear()

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.mark.anyio
async def test_start_release_run_api_creates_release_run(
    release_run_api_client: AsyncClient,
) -> None:
    """POST /release-runs should create a release run."""

    response = await release_run_api_client.post(
        "/api/v1/release-runs",
        json={
            "query": "What are the biggest release risks this week?",
            "requested_by": "manager@example.com",
        },
    )

    assert response.status_code == 201

    response_data = response.json()

    assert response_data["id"] is not None
    assert response_data["run_id"].startswith("release-run-")
    assert response_data["query"] == "What are the biggest release risks this week?"
    assert response_data["requested_by"] == "manager@example.com"
    assert response_data["status"] == "created"
    assert response_data["completed_at"] is None


@pytest.mark.anyio
async def test_get_release_run_api_returns_created_release_run(
    release_run_api_client: AsyncClient,
) -> None:
    """GET /release-runs/{id} should return an existing release run."""

    create_response = await release_run_api_client.post(
        "/api/v1/release-runs",
        json={
            "query": "Check release readiness for this week.",
            "requested_by": "manager@example.com",
        },
    )

    assert create_response.status_code == 201

    created_release_run = create_response.json()
    release_run_id = created_release_run["id"]

    get_response = await release_run_api_client.get(
        f"/api/v1/release-runs/{release_run_id}",
    )

    assert get_response.status_code == 200

    response_data = get_response.json()

    assert response_data["id"] == release_run_id
    assert response_data["run_id"] == created_release_run["run_id"]
    assert response_data["query"] == "Check release readiness for this week."
    assert response_data["status"] == "created"


@pytest.mark.anyio
async def test_get_release_run_api_returns_404_when_missing(
    release_run_api_client: AsyncClient,
) -> None:
    """GET /release-runs/{id} should return 404 for a missing release run."""

    response = await release_run_api_client.get(
        f"/api/v1/release-runs/{uuid4()}",
    )

    assert response.status_code == 404

    response_data = response.json()

    assert response_data["error"]["message"] == "Release run not found."


@pytest.mark.anyio
async def test_start_release_run_api_rejects_invalid_payload(
    release_run_api_client: AsyncClient,
) -> None:
    """POST /release-runs should reject invalid request payload."""

    response = await release_run_api_client.post(
        "/api/v1/release-runs",
        json={
            "query": "bad",
            "requested_by": "me",
        },
    )

    assert response.status_code == 422


@pytest.mark.anyio
async def test_collect_release_risks_api_returns_full_release_risk_summary(
    release_run_api_client: AsyncClient,
) -> None:
    """POST /release-runs/{id}/risks should collect GitHub and Jira risks."""

    override_external_collectors_for_test()

    create_response = await release_run_api_client.post(
        "/api/v1/release-runs",
        json={
            "query": "What are the biggest release risks this week?",
            "requested_by": "manager@example.com",
        },
    )

    assert create_response.status_code == 201

    release_run_id = create_response.json()["id"]

    risk_response = await release_run_api_client.post(
        f"/api/v1/release-runs/{release_run_id}/risks",
    )

    assert risk_response.status_code == 200

    response_data = risk_response.json()

    assert response_data["release_run"]["id"] == release_run_id
    assert response_data["release_run"]["status"] == "completed"

    assert response_data["github"]["source"] == "github"
    assert response_data["github"]["status"] == "success"
    assert response_data["github"]["pull_request_count"] == 2
    assert response_data["github"]["risk_result_count"] == 2
    assert response_data["github"]["total_signal_count"] == 3
    assert response_data["github"]["high_risk_count"] == 1

    assert response_data["github_summary"]["source"] == "github"
    assert response_data["github_summary"]["collection_status"] == "success"
    assert response_data["github_summary"]["overall_severity"] in {
        "low",
        "medium",
        "high",
        "critical",
    }
    assert response_data["github_summary"]["recommended_action"] in {
        "proceed",
        "review_required",
        "block_release",
        "partial_data_review",
    }
    assert isinstance(response_data["github_summary"]["top_risks"], list)
    assert isinstance(response_data["github_summary"]["summary_text"], str)
    assert response_data["github_summary"]["summary_text"]

    assert response_data["jira"]["status"] == "success"
    assert response_data["jira"]["total_issues_analyzed"] == 0
    assert response_data["jira"]["total_signals"] == 0
    assert response_data["jira"]["issues"] == []
    assert response_data["jira"]["signals"] == []

    assert response_data["jira_summary"]["source"] == "jira"
    assert response_data["jira_summary"]["collection_status"] == "success"
    assert response_data["jira_summary"]["overall_severity"] == "low"
    assert response_data["jira_summary"]["recommended_action"] == "proceed"
    assert response_data["jira_summary"]["issue_count"] == 0
    assert response_data["jira_summary"]["risky_issue_count"] == 0
    assert response_data["jira_summary"]["total_signal_count"] == 0

    assert response_data["release_summary"]["source"] == "release"
    assert response_data["release_summary"]["overall_severity"] in {
        "low",
        "medium",
        "high",
        "critical",
    }
    assert response_data["release_summary"]["recommended_action"] in {
        "proceed",
        "review_required",
        "block_release",
        "partial_data_review",
    }
    assert isinstance(response_data["release_summary"]["summary_text"], str)
    assert response_data["release_summary"]["summary_text"]

    _assert_risk_scoring_response(response_data)


@pytest.mark.anyio
async def test_collect_github_risks_api_returns_github_risk_summary(
    release_run_api_client: AsyncClient,
) -> None:
    """POST /release-runs/{id}/github-risks should collect GitHub and Jira risks."""

    override_external_collectors_for_test()

    create_response = await release_run_api_client.post(
        "/api/v1/release-runs",
        json={
            "query": "What are the biggest release risks this week?",
            "requested_by": "manager@example.com",
        },
    )

    assert create_response.status_code == 201

    release_run_id = create_response.json()["id"]

    risk_response = await release_run_api_client.post(
        f"/api/v1/release-runs/{release_run_id}/github-risks",
    )

    assert risk_response.status_code == 200

    response_data = risk_response.json()

    assert response_data["release_run"]["id"] == release_run_id
    assert response_data["release_run"]["status"] == "completed"

    assert response_data["github"]["source"] == "github"
    assert response_data["github"]["status"] == "success"
    assert response_data["github"]["pull_request_count"] == 2
    assert response_data["github"]["risk_result_count"] == 2
    assert response_data["github"]["total_signal_count"] == 3
    assert response_data["github"]["high_risk_count"] == 1

    assert response_data["github_summary"]["source"] == "github"
    assert response_data["github_summary"]["collection_status"] == "success"
    assert response_data["github_summary"]["overall_severity"] in {
        "low",
        "medium",
        "high",
        "critical",
    }
    assert response_data["github_summary"]["recommended_action"] in {
        "proceed",
        "review_required",
        "block_release",
        "partial_data_review",
    }
    assert isinstance(response_data["github_summary"]["top_risks"], list)
    assert isinstance(response_data["github_summary"]["summary_text"], str)
    assert response_data["github_summary"]["summary_text"]

    assert response_data["jira"]["status"] == "success"
    assert response_data["jira"]["total_issues_analyzed"] == 0
    assert response_data["jira"]["total_signals"] == 0
    assert response_data["jira"]["issues"] == []
    assert response_data["jira"]["signals"] == []

    assert response_data["jira_summary"]["source"] == "jira"
    assert response_data["jira_summary"]["collection_status"] == "success"
    assert response_data["jira_summary"]["overall_severity"] == "low"
    assert response_data["jira_summary"]["recommended_action"] == "proceed"
    assert response_data["jira_summary"]["issue_count"] == 0
    assert response_data["jira_summary"]["risky_issue_count"] == 0
    assert response_data["jira_summary"]["total_signal_count"] == 0

    assert response_data["release_summary"]["source"] == "release"
    assert response_data["release_summary"]["overall_severity"] in {
        "low",
        "medium",
        "high",
        "critical",
    }
    assert response_data["release_summary"]["recommended_action"] in {
        "proceed",
        "review_required",
        "block_release",
        "partial_data_review",
    }
    assert isinstance(response_data["release_summary"]["summary_text"], str)
    assert response_data["release_summary"]["summary_text"]


@pytest.mark.anyio
async def test_collect_release_risks_api_uses_langgraph_workflow_path(
    release_run_api_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /release-runs/{id}/risks should enter through LangGraph workflow."""

    override_external_collectors_for_test()

    workflow_called = False
    original_run_release_risk_workflow = ReleaseRunService.run_release_risk_workflow

    async def spy_run_release_risk_workflow(
        self: ReleaseRunService,
        release_run_id: UUID,
    ) -> Any:
        """Record that the preferred API route entered the LangGraph path."""

        nonlocal workflow_called
        workflow_called = True

        return await original_run_release_risk_workflow(
            self,
            release_run_id,
        )

    monkeypatch.setattr(
        ReleaseRunService,
        "run_release_risk_workflow",
        spy_run_release_risk_workflow,
    )

    create_response = await release_run_api_client.post(
        "/api/v1/release-runs",
        json={
            "query": "What are the biggest release risks this week?",
            "requested_by": "manager@example.com",
        },
    )

    assert create_response.status_code == 201

    release_run_id = create_response.json()["id"]

    risk_response = await release_run_api_client.post(
        f"/api/v1/release-runs/{release_run_id}/risks",
    )

    assert risk_response.status_code == 200
    assert workflow_called is True

    response_data = risk_response.json()

    assert response_data["release_run"]["id"] == release_run_id
    assert response_data["release_run"]["status"] == "completed"
    assert response_data["github"]["status"] == "success"
    assert response_data["jira"]["status"] == "success"
    assert response_data["release_summary"]["source"] == "release"


@pytest.mark.anyio
async def test_legacy_github_risks_api_keeps_direct_service_path(
    release_run_api_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /release-runs/{id}/github-risks should not use LangGraph workflow."""

    override_external_collectors_for_test()

    direct_collector_called = False
    original_collect_release_risks = ReleaseRunService.collect_release_risks

    async def spy_collect_release_risks(
        self: ReleaseRunService,
        release_run_id: UUID,
    ) -> Any:
        """Record that the legacy API route used the direct service path."""

        nonlocal direct_collector_called
        direct_collector_called = True

        return await original_collect_release_risks(
            self,
            release_run_id,
        )

    async def fail_if_workflow_is_used(
        self: ReleaseRunService,
        release_run_id: UUID,
    ) -> Any:
        """Fail the test if the legacy endpoint accidentally uses LangGraph."""

        raise AssertionError(
            "Legacy /github-risks endpoint should not call run_release_risk_workflow()."
        )

    monkeypatch.setattr(
        ReleaseRunService,
        "collect_release_risks",
        spy_collect_release_risks,
    )
    monkeypatch.setattr(
        ReleaseRunService,
        "run_release_risk_workflow",
        fail_if_workflow_is_used,
    )

    create_response = await release_run_api_client.post(
        "/api/v1/release-runs",
        json={
            "query": "What are the biggest release risks this week?",
            "requested_by": "manager@example.com",
        },
    )

    assert create_response.status_code == 201

    release_run_id = create_response.json()["id"]

    risk_response = await release_run_api_client.post(
        f"/api/v1/release-runs/{release_run_id}/github-risks",
    )

    assert risk_response.status_code == 200
    assert direct_collector_called is True

    response_data = risk_response.json()

    assert response_data["release_run"]["id"] == release_run_id
    assert response_data["release_run"]["status"] == "completed"
    assert response_data["github"]["status"] == "success"
    assert response_data["jira"]["status"] == "success"

@pytest.mark.anyio
async def test_collect_release_risks_api_returns_knowledge_context_when_docs_match(
    release_run_api_client: AsyncClient,
) -> None:
    """POST /release-runs/{id}/risks should expose retrieved Knowledge Agent context.

    This protects the end-to-end explainability path:

    engineering document ingestion
    -> preferred LangGraph /risks workflow
    -> EngineeringDocumentRetrievalService
    -> knowledge_results in public API response
    """

    override_external_collectors_for_test()

    ingest_response = await release_run_api_client.post(
        "/api/v1/engineering-documents/ingest",
        json={
            "title": "Checkout Redis Incident Runbook",
            "source_type": "runbook",
            "source_uri": "test-knowledge-base",
            "raw_content": (
                "Redis checkout failure is a known release risk. "
                "If checkout latency increases after deployment, review Redis "
                "connection pool saturation, payment retry queues, and rollback "
                "the checkout feature flag before continuing the release."
            ),
            "metadata_json": {
                "service": "checkout",
                "risk": "Redis checkout failure",
            },
        },
    )

    assert ingest_response.status_code in {200, 201}, ingest_response.text

    create_response = await release_run_api_client.post(
        "/api/v1/release-runs",
        json={
            "query": (
                "What are the biggest release risks this week for Redis "
                "checkout failure?"
            ),
            "requested_by": "manager@example.com",
        },
    )

    assert create_response.status_code == 201

    release_run_id = create_response.json()["id"]

    risk_response = await release_run_api_client.post(
        f"/api/v1/release-runs/{release_run_id}/risks",
    )

    assert risk_response.status_code == 200, risk_response.text

    response_data = risk_response.json()

    assert response_data["release_run"]["id"] == release_run_id
    assert response_data["release_run"]["status"] == "completed"

    assert response_data["github"]["status"] == "success"
    assert response_data["jira"]["status"] == "success"
    assert response_data["release_summary"]["source"] == "release"

    assert response_data["knowledge_status"] == "completed"
    assert response_data["knowledge_error"] is None
    assert response_data["knowledge_query"]

    knowledge_results = response_data["knowledge_results"]

    assert isinstance(knowledge_results, list)
    assert len(knowledge_results) >= 1

    serialized_results = str(knowledge_results).lower()

    assert "redis" in serialized_results
    assert "checkout" in serialized_results

    events_response = await release_run_api_client.get(
        f"/api/v1/release-runs/{release_run_id}/events",
    )

    assert events_response.status_code == 200, events_response.text

    events_data = events_response.json()
    event_types = [
        event["event_type"]
        for event in events_data["events"]
    ]

    assert "knowledge_retrieval_completed" in event_types
    assert "workflow_completed" in event_types

    knowledge_event = next(
        event
        for event in events_data["events"]
        if event["event_type"] == "knowledge_retrieval_completed"
    )

    assert knowledge_event["event_status"] == "success"
    assert knowledge_event["metadata_json"]["result_count"] >= 1
    assert knowledge_event["metadata_json"]["query_length"] > 0



@pytest.mark.anyio
async def test_collect_release_risks_api_audits_knowledge_retrieval_no_results(
    release_run_api_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preferred /risks endpoint should audit successful empty Knowledge retrieval."""

    override_external_collectors_for_test()

    async def retrieve_no_relevant_chunks(
        self: EngineeringDocumentRetrievalService,
        retrieval_request: Any,
        *,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Simulate successful Knowledge retrieval with no matching chunks."""

        return {
            "query": retrieval_request.query,
            "total_candidates": 0,
            "results": [],
        }

    monkeypatch.setattr(
        EngineeringDocumentRetrievalService,
        "retrieve_relevant_chunks",
        retrieve_no_relevant_chunks,
    )

    create_response = await release_run_api_client.post(
        "/api/v1/release-runs",
        json={
            "query": "What are the biggest release risks this week?",
            "requested_by": "manager@example.com",
        },
    )

    assert create_response.status_code == 201

    release_run_id = create_response.json()["id"]

    risk_response = await release_run_api_client.post(
        f"/api/v1/release-runs/{release_run_id}/risks",
    )

    assert risk_response.status_code == 200, risk_response.text

    response_data = risk_response.json()

    assert response_data["release_run"]["id"] == release_run_id
    assert response_data["release_run"]["status"] == "completed"
    assert response_data["github"]["status"] == "success"
    assert response_data["jira"]["status"] == "success"
    assert response_data["release_summary"]["source"] == "release"

    assert response_data["knowledge_status"] == "no_results"
    assert response_data["knowledge_results"] == []
    assert response_data["knowledge_error"] is None
    assert response_data["knowledge_query"]

    events_response = await release_run_api_client.get(
        f"/api/v1/release-runs/{release_run_id}/events",
    )

    assert events_response.status_code == 200, events_response.text

    events_data = events_response.json()
    event_types = [
        event["event_type"]
        for event in events_data["events"]
    ]

    assert "knowledge_retrieval_completed" in event_types
    assert "knowledge_retrieval_failed" not in event_types
    assert "workflow_completed" in event_types
    assert "workflow_failed" not in event_types

    knowledge_event = next(
        event
        for event in events_data["events"]
        if event["event_type"] == "knowledge_retrieval_completed"
    )

    assert knowledge_event["event_status"] == "success"
    assert knowledge_event["metadata_json"]["result_count"] == 0
    assert knowledge_event["metadata_json"]["query_length"] > 0
    assert knowledge_event["metadata_json"]["knowledge_status"] == "no_results"
    assert knowledge_event["metadata_json"]["error_present"] is False


@pytest.mark.anyio
async def test_collect_release_risks_api_audits_knowledge_retrieval_failure(
    release_run_api_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preferred /risks endpoint should audit recoverable Knowledge Agent failure."""

    override_external_collectors_for_test()

    async def fail_retrieve_relevant_chunks(
        self: EngineeringDocumentRetrievalService,
        retrieval_request: Any,
        *,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Simulate a recoverable Knowledge Agent retrieval failure."""

        raise ValueError("Simulated retrieval failure.")

    monkeypatch.setattr(
        EngineeringDocumentRetrievalService,
        "retrieve_relevant_chunks",
        fail_retrieve_relevant_chunks,
    )

    create_response = await release_run_api_client.post(
        "/api/v1/release-runs",
        json={
            "query": "What are the biggest release risks this week?",
            "requested_by": "manager@example.com",
        },
    )

    assert create_response.status_code == 201

    release_run_id = create_response.json()["id"]

    risk_response = await release_run_api_client.post(
        f"/api/v1/release-runs/{release_run_id}/risks",
    )

    assert risk_response.status_code == 200, risk_response.text

    response_data = risk_response.json()

    assert response_data["release_run"]["id"] == release_run_id
    assert response_data["release_run"]["status"] == "waiting_for_approval"
    assert response_data["approval_required"] is True
    assert response_data["approval_status"] == "pending"
    assert response_data["approval_request_id"] is not None
    assert response_data["github"]["status"] == "success"
    assert response_data["jira"]["status"] == "success"
    assert response_data["release_summary"]["source"] == "release"

    assert response_data["knowledge_status"] == "failed"
    assert response_data["knowledge_results"] == []
    assert response_data["knowledge_error"] == "Knowledge retrieval failed."

    events_response = await release_run_api_client.get(
        f"/api/v1/release-runs/{release_run_id}/events",
    )

    assert events_response.status_code == 200, events_response.text

    events_data = events_response.json()
    event_types = [
        event["event_type"]
        for event in events_data["events"]
    ]

    assert "knowledge_retrieval_failed" in event_types
    assert "workflow_completed" in event_types
    assert "workflow_failed" not in event_types

    knowledge_event = next(
        event
        for event in events_data["events"]
        if event["event_type"] == "knowledge_retrieval_failed"
    )

    assert knowledge_event["event_status"] == "failed"
    assert knowledge_event["metadata_json"]["result_count"] == 0
    assert knowledge_event["metadata_json"]["query_length"] > 0
    assert knowledge_event["metadata_json"]["error_present"] is True


@pytest.mark.anyio
async def test_collect_release_risks_api_persists_release_risk_snapshot(
    release_run_api_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /risks should persist a trusted backend-generated risk snapshot."""

    override_external_collectors_for_test()

    created_snapshot_commands: list[CreateReleaseRunRiskSnapshotCommand] = []
    original_create_snapshot = ReleaseRunRiskSnapshotRepository.create_snapshot

    async def spy_create_snapshot(
        self: ReleaseRunRiskSnapshotRepository,
        command: CreateReleaseRunRiskSnapshotCommand,
    ) -> object:
        """Record snapshot persistence while preserving real repository behavior."""
        created_snapshot_commands.append(command)
        return await original_create_snapshot(self, command)

    monkeypatch.setattr(
        ReleaseRunRiskSnapshotRepository,
        "create_snapshot",
        spy_create_snapshot,
    )

    create_response = await release_run_api_client.post(
        "/api/v1/release-runs",
        json={
            "query": "What are the biggest release risks this week?",
            "requested_by": "manager@example.com",
        },
    )

    assert create_response.status_code == 201

    release_run_id = create_response.json()["id"]

    risk_response = await release_run_api_client.post(
        f"/api/v1/release-runs/{release_run_id}/risks",
    )

    assert risk_response.status_code == 200, risk_response.text

    response_data = risk_response.json()

    assert len(created_snapshot_commands) == 1

    snapshot_command = created_snapshot_commands[0]

    assert str(snapshot_command.release_run_id) == release_run_id
    assert snapshot_command.overall_severity == (
        response_data["release_summary"]["overall_severity"]
    )
    assert snapshot_command.approval_required == response_data["approval_required"]
    assert snapshot_command.approval_status_at_snapshot == (
        response_data["approval_status"] or "not_required"
    )

    snapshot_payload = snapshot_command.risk_payload

    assert snapshot_payload["release_run"]["id"] == release_run_id
    assert snapshot_payload["github"]["status"] == response_data["github"]["status"]
    assert snapshot_payload["jira"]["status"] == response_data["jira"]["status"]
    assert snapshot_payload["release_summary"]["overall_severity"] == (
        response_data["release_summary"]["overall_severity"]
    )
    assert snapshot_payload["risk_score"]["score"] == response_data["risk_score"]["score"]

    events_response = await release_run_api_client.get(
        f"/api/v1/release-runs/{release_run_id}/events",
    )

    assert events_response.status_code == 200, events_response.text

    events_data = events_response.json()
    snapshot_events = [
        event
        for event in events_data["events"]
        if event["event_type"] == "release_risk_snapshot_created"
    ]

    assert len(snapshot_events) == 1
    assert snapshot_events[0]["event_status"] == "success"
    assert snapshot_events[0]["metadata_json"]["snapshot_version"] == 1
    assert snapshot_events[0]["metadata_json"]["overall_severity"] == (
        response_data["release_summary"]["overall_severity"]
    )


@pytest.mark.anyio
async def test_collect_release_risks_api_creates_new_snapshot_version_on_second_run(
    release_run_api_client: AsyncClient,
) -> None:
    """Repeated POST /risks calls should create append-only snapshot versions."""

    override_external_collectors_for_test()

    create_response = await release_run_api_client.post(
        "/api/v1/release-runs",
        json={
            "query": "What are the biggest release risks this week?",
            "requested_by": "manager@example.com",
        },
    )

    assert create_response.status_code == 201

    release_run_id = create_response.json()["id"]

    first_risk_response = await release_run_api_client.post(
        f"/api/v1/release-runs/{release_run_id}/risks",
    )
    second_risk_response = await release_run_api_client.post(
        f"/api/v1/release-runs/{release_run_id}/risks",
    )

    assert first_risk_response.status_code == 200, first_risk_response.text
    assert second_risk_response.status_code == 200, second_risk_response.text

    events_response = await release_run_api_client.get(
        f"/api/v1/release-runs/{release_run_id}/events",
    )

    assert events_response.status_code == 200, events_response.text

    events_data = events_response.json()
    snapshot_versions = [
        event["metadata_json"]["snapshot_version"]
        for event in events_data["events"]
        if event["event_type"] == "release_risk_snapshot_created"
    ]

    assert snapshot_versions == [1, 2]


@pytest.mark.anyio
async def test_send_release_run_slack_alert_api_sends_after_approval(
    release_run_api_client: AsyncClient,
) -> None:
    """POST /slack-alert should send only after approved snapshot exists."""

    override_degraded_github_collector_for_test()
    slack_sender = FakeSlackAlertSender()
    override_slack_alert_sender_for_test(slack_sender)

    create_response = await release_run_api_client.post(
        "/api/v1/release-runs",
        json={
            "query": "What are the biggest release risks this week?",
            "requested_by": "manager@example.com",
        },
    )

    assert create_response.status_code == 201, create_response.text

    release_run_id = create_response.json()["id"]

    risk_response = await release_run_api_client.post(
        f"/api/v1/release-runs/{release_run_id}/risks",
    )

    assert risk_response.status_code == 200, risk_response.text

    risk_response_data = risk_response.json()
    approval_request_id = risk_response_data["approval_request_id"]

    assert approval_request_id is not None
    assert risk_response_data["approval_status"] == "pending"

    decision_response = await release_run_api_client.post(
        (
            f"/api/v1/release-runs/{release_run_id}"
            f"/approvals/{approval_request_id}/decision"
        ),
        json={
            "approval_status": "approved",
            "decided_by": "director@example.com",
            "decision_note": "Approved after reviewing rollback plan.",
        },
    )

    assert decision_response.status_code == 200, decision_response.text
    assert decision_response.json()["approval_status"] == "approved"

    slack_response = await release_run_api_client.post(
        f"/api/v1/release-runs/{release_run_id}/slack-alert",
    )

    assert slack_response.status_code == 200, slack_response.text

    slack_response_data = slack_response.json()

    assert slack_response_data["sent"] is True
    assert slack_response_data["slack_channel"] == "C1234567890"
    assert slack_response_data["slack_timestamp"] == "12345.6789"

    assert len(slack_sender.sent_payloads) == 1

    payload = slack_sender.sent_payloads[0]

    assert payload.metadata["release_run_id"] == release_run_id
    assert payload.metadata["approval_status"] == "approved"
    assert payload.metadata["approval_request_id"] == approval_request_id
    assert payload.metadata["release_run_status"] == "approval_approved"

    events_response = await release_run_api_client.get(
        f"/api/v1/release-runs/{release_run_id}/events",
    )

    assert events_response.status_code == 200, events_response.text

    events_data = events_response.json()
    slack_events = [
        event
        for event in events_data["events"]
        if event["event_type"] == "release_slack_alert_sent"
    ]

    assert len(slack_events) == 1
    assert slack_events[0]["event_status"] == "success"
    assert slack_events[0]["metadata_json"]["slack_channel"] == "C1234567890"


@pytest.mark.anyio
async def test_send_release_run_slack_alert_api_blocks_duplicate_send(
    release_run_api_client: AsyncClient,
) -> None:
    """POST /slack-alert should not send duplicate alerts for one release run."""

    override_degraded_github_collector_for_test()
    slack_sender = FakeSlackAlertSender()
    override_slack_alert_sender_for_test(slack_sender)

    create_response = await release_run_api_client.post(
        "/api/v1/release-runs",
        json={
            "query": "What are the biggest release risks this week?",
            "requested_by": "manager@example.com",
        },
    )

    assert create_response.status_code == 201, create_response.text

    release_run_id = create_response.json()["id"]

    risk_response = await release_run_api_client.post(
        f"/api/v1/release-runs/{release_run_id}/risks",
    )

    assert risk_response.status_code == 200, risk_response.text

    approval_request_id = risk_response.json()["approval_request_id"]

    decision_response = await release_run_api_client.post(
        (
            f"/api/v1/release-runs/{release_run_id}"
            f"/approvals/{approval_request_id}/decision"
        ),
        json={
            "approval_status": "approved",
            "decided_by": "director@example.com",
            "decision_note": "Approved after reviewing rollback plan.",
        },
    )

    assert decision_response.status_code == 200, decision_response.text

    first_slack_response = await release_run_api_client.post(
        f"/api/v1/release-runs/{release_run_id}/slack-alert",
    )
    second_slack_response = await release_run_api_client.post(
        f"/api/v1/release-runs/{release_run_id}/slack-alert",
    )

    assert first_slack_response.status_code == 200, first_slack_response.text
    assert second_slack_response.status_code == 409, second_slack_response.text
    assert "Slack alert already sent" in second_slack_response.text

    assert len(slack_sender.sent_payloads) == 1

    events_response = await release_run_api_client.get(
        f"/api/v1/release-runs/{release_run_id}/events",
    )

    assert events_response.status_code == 200, events_response.text

    events_data = events_response.json()
    slack_events = [
        event
        for event in events_data["events"]
        if event["event_type"] == "release_slack_alert_sent"
    ]

    assert [event["event_status"] for event in slack_events] == [
        "success",
        "blocked",
    ]


@pytest.mark.anyio
async def test_send_release_run_slack_alert_api_blocks_before_approval(
    release_run_api_client: AsyncClient,
) -> None:
    """POST /slack-alert should reject release runs still waiting for approval."""

    override_degraded_github_collector_for_test()
    slack_sender = FakeSlackAlertSender()
    override_slack_alert_sender_for_test(slack_sender)

    create_response = await release_run_api_client.post(
        "/api/v1/release-runs",
        json={
            "query": "What are the biggest release risks this week?",
            "requested_by": "manager@example.com",
        },
    )

    assert create_response.status_code == 201, create_response.text

    release_run_id = create_response.json()["id"]

    risk_response = await release_run_api_client.post(
        f"/api/v1/release-runs/{release_run_id}/risks",
    )

    assert risk_response.status_code == 200, risk_response.text
    assert risk_response.json()["approval_status"] == "pending"

    slack_response = await release_run_api_client.post(
        f"/api/v1/release-runs/{release_run_id}/slack-alert",
    )

    assert slack_response.status_code == 409, slack_response.text
    assert "Slack alert cannot be sent before approval" in slack_response.text
    assert slack_sender.sent_payloads == []

    events_response = await release_run_api_client.get(
        f"/api/v1/release-runs/{release_run_id}/events",
    )

    assert events_response.status_code == 200, events_response.text

    events_data = events_response.json()
    slack_events = [
        event
        for event in events_data["events"]
        if event["event_type"] == "release_slack_alert_sent"
    ]

    assert len(slack_events) == 1
    assert slack_events[0]["event_status"] == "blocked"
    assert "not approved" in slack_events[0]["message"].lower()


@pytest.mark.anyio
async def test_collect_release_risks_api_creates_pending_approval_request_when_required(
    release_run_api_client: AsyncClient,
) -> None:
    """POST /risks should create/reuse pending approval when approval is required."""

    override_degraded_github_collector_for_test()

    create_response = await release_run_api_client.post(
        "/api/v1/release-runs",
        json={
            "query": "What are the biggest release risks this week?",
            "requested_by": "manager@example.com",
        },
    )

    assert create_response.status_code == 201

    release_run_id = create_response.json()["id"]

    first_risk_response = await release_run_api_client.post(
        f"/api/v1/release-runs/{release_run_id}/risks",
    )

    assert first_risk_response.status_code == 200

    first_response_data = first_risk_response.json()

    assert first_response_data["approval_required"] is True
    assert first_response_data["approval_policy_version"] == "hitl_policy_v1"
    assert first_response_data["approval_request_id"] is not None
    assert first_response_data["approval_status"] == "pending"
    assert first_response_data["release_run"]["status"] == "waiting_for_approval"

    second_risk_response = await release_run_api_client.post(
        f"/api/v1/release-runs/{release_run_id}/risks",
    )

    assert second_risk_response.status_code == 200

    second_response_data = second_risk_response.json()

    assert second_response_data["approval_required"] is True
    assert second_response_data["approval_request_id"] == (
        first_response_data["approval_request_id"]
    )
    assert second_response_data["approval_status"] == "pending"

    events_response = await release_run_api_client.get(
        f"/api/v1/release-runs/{release_run_id}/events",
    )

    assert events_response.status_code == 200

    event_types = [
        event["event_type"]
        for event in events_response.json()["events"]
    ]

    assert "approval_request_created" in event_types
    assert event_types.count("approval_request_created") == 1
    assert "release_run_waiting_for_approval" in event_types
