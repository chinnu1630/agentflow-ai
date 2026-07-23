"""Integration tests proving release-risk workflows write audit events."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.api.dependencies.security import get_current_principal
from app.api.routes.release_runs import get_jira_risk_collector, get_risk_collector
from app.core.security import AuthenticatedPrincipal
from app.db.base import Base
from app.db.session import get_db_session
from app.main import app
from app.services.engineering_document_embedding_provider import (
    get_engineering_document_embedding_provider,
)
from app.services.engineering_document_reranker import (
    get_engineering_document_reranker,
)
from app.services.github_risk_collector import GitHubRiskCollectionResult, RiskCollectionStatus
from app.services.jira_risk_collector import (
    JiraRiskCollectionResult,
    JiraRiskCollectionStatus,
)


class FakeAuditEmbeddingProvider:
    """Generate deterministic embeddings without loading model weights."""

    async def embed_texts(
        self,
        texts: Sequence[str],
        *,
        run_id: str | None = None,
    ) -> list[list[float]]:
        """Return one 384-dimensional embedding per supplied text."""
        return [[float(index + 1)] * 384 for index, _text in enumerate(texts)]



class FakeAuditReranker:
    """Return deterministic scores without loading a cross-encoder."""

    async def score_candidates(
        self,
        *,
        query: str,
        candidate_contents: Sequence[str],
        run_id: str | None = None,
    ) -> list[float]:
        """Return descending scores in candidate order."""
        return [
            float(len(candidate_contents) - index)
            for index, _content in enumerate(candidate_contents)
        ]


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


@pytest.fixture
async def audit_workflow_api_client() -> AsyncIterator[AsyncClient]:
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

    async def override_get_risk_collector() -> FakeRiskCollector:
        """Override GitHub collector dependency for API tests."""

        return FakeRiskCollector()

    async def override_get_jira_risk_collector() -> FakeJiraRiskCollector:
        """Override Jira collector dependency for API tests."""

        return FakeJiraRiskCollector()

    async def override_get_current_principal() -> AuthenticatedPrincipal:
        """Return a trusted manager for release API tests."""
        return AuthenticatedPrincipal(
            subject="manager-123",
            email="manager@example.com",
            roles=frozenset({"release_manager"}),
            scopes=frozenset(
                {
                    "release:read",
                    "release:write",
                    "release:approve",
                    "release:notify",
                }
            ),
        )

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[
        get_current_principal
    ] = override_get_current_principal
    app.dependency_overrides[get_risk_collector] = override_get_risk_collector
    app.dependency_overrides[get_jira_risk_collector] = override_get_jira_risk_collector
    app.dependency_overrides[
        get_engineering_document_embedding_provider
    ] = FakeAuditEmbeddingProvider
    app.dependency_overrides[
        get_engineering_document_reranker
    ] = FakeAuditReranker

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
async def test_release_risk_workflow_writes_audit_timeline(
    audit_workflow_api_client: AsyncClient,
) -> None:
    """Preferred /risks workflow should persist an end-to-end audit timeline."""

    create_response = await audit_workflow_api_client.post(
        "/api/v1/release-runs",
        json={
            "query": "What are the biggest release risks this week?",
        },
    )

    assert create_response.status_code == 201

    release_run_id = create_response.json()["id"]

    risk_response = await audit_workflow_api_client.post(
        f"/api/v1/release-runs/{release_run_id}/risks",
    )

    assert risk_response.status_code == 200

    events_response = await audit_workflow_api_client.get(
        f"/api/v1/release-runs/{release_run_id}/events",
    )

    assert events_response.status_code == 200

    response_data = events_response.json()
    event_types = [event["event_type"] for event in response_data["events"]]

    assert response_data["release_run_id"] == release_run_id

    assert "release_run_started" in event_types
    assert "workflow_started" in event_types
    assert "risk_collection_started" in event_types
    assert "github_collection_completed" in event_types
    assert "jira_collection_completed" in event_types
    assert "release_summary_created" in event_types
    assert "risk_features_extracted" in event_types
    assert "release_risk_scored" in event_types
    assert "approval_requirement_determined" in event_types
    assert "risk_collection_completed" in event_types
    assert "workflow_completed" in event_types

    github_event = next(
        event
        for event in response_data["events"]
        if event["event_type"] == "github_collection_completed"
    )

    assert github_event["event_status"] == "success"
    assert github_event["metadata_json"]["pull_request_count"] == 2
    assert github_event["metadata_json"]["risk_result_count"] == 2
    assert github_event["metadata_json"]["high_risk_count"] == 1

    jira_event = next(
        event
        for event in response_data["events"]
        if event["event_type"] == "jira_collection_completed"
    )

    assert jira_event["event_status"] == "success"
    assert jira_event["metadata_json"]["total_issues_analyzed"] == 0
    assert jira_event["metadata_json"]["total_signals"] == 0

    feature_event = next(
        event
        for event in response_data["events"]
        if event["event_type"] == "risk_features_extracted"
    )

    assert feature_event["event_status"] == "success"
    assert feature_event["metadata_json"]["feature_version"] == (
        "release_risk_features_v1"
    )
    assert isinstance(feature_event["metadata_json"]["total_risk_count"], int)
    assert isinstance(feature_event["metadata_json"]["github_risk_count"], int)
    assert isinstance(feature_event["metadata_json"]["jira_risk_count"], int)
    assert isinstance(feature_event["metadata_json"]["knowledge_failed"], bool)
    assert "content" not in feature_event["metadata_json"]
    assert "knowledge_results" not in feature_event["metadata_json"]

    scoring_event = next(
        event
        for event in response_data["events"]
        if event["event_type"] == "release_risk_scored"
    )

    assert scoring_event["event_status"] == "success"
    assert scoring_event["metadata_json"]["scoring_version"] == (
        "rule_based_release_risk_v1"
    )
    assert scoring_event["metadata_json"]["feature_version"] == (
        "release_risk_features_v1"
    )
    assert 0.0 <= scoring_event["metadata_json"]["score"] <= 1.0
    assert scoring_event["metadata_json"]["risk_level"] in {
        "low",
        "medium",
        "high",
        "critical",
    }
    assert scoring_event["metadata_json"]["recommended_action"] in {
        "proceed",
        "review_required",
        "block_release",
        "partial_data_review",
    }
    assert scoring_event["metadata_json"]["reason_count"] >= 1
    assert "raw_query" not in scoring_event["metadata_json"]
    assert "stack_trace" not in scoring_event["metadata_json"]

    approval_event = next(
        event
        for event in response_data["events"]
        if event["event_type"] == "approval_requirement_determined"
    )

    assert approval_event["event_status"] == "success"
    assert approval_event["metadata_json"]["approval_policy_version"] == (
        "hitl_policy_v1"
    )
    assert isinstance(approval_event["metadata_json"]["approval_required"], bool)
    assert isinstance(
        approval_event["metadata_json"]["approval_reason_present"],
        bool,
    )
    assert approval_event["metadata_json"]["risk_level"] in {
        "low",
        "medium",
        "high",
        "critical",
    }
    assert approval_event["metadata_json"]["recommended_action"] in {
        "proceed",
        "review_required",
        "block_release",
        "partial_data_review",
    }
    assert "approval_reason" not in approval_event["metadata_json"]
    assert "raw_query" not in approval_event["metadata_json"]
    assert "content" not in approval_event["metadata_json"]
