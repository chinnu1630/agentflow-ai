from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.api.routes.release_runs import get_risk_collector
from app.db.base import Base
from app.db.session import get_db_session
from app.main import app
from app.services.risk_collector import GitHubRiskCollectionResult, RiskCollectionStatus


class FakeRiskCollector:
    """Fake risk collector used to avoid real GitHub calls in API tests."""

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


@pytest.fixture
async def release_run_api_client() -> AsyncIterator[AsyncClient]:
    """Create an API client with an isolated test database."""
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
async def test_collect_github_risks_api_returns_github_risk_summary(
    release_run_api_client: AsyncClient,
) -> None:
    """POST /release-runs/{id}/github-risks should collect GitHub risks."""

    async def override_get_risk_collector() -> FakeRiskCollector:
        """Override GitHub collector dependency for API tests."""
        return FakeRiskCollector()

    app.dependency_overrides[get_risk_collector] = override_get_risk_collector

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


@pytest.mark.anyio
async def test_collect_github_risks_api_returns_404_when_release_run_missing(
    release_run_api_client: AsyncClient,
) -> None:
    """POST /release-runs/{id}/github-risks should return 404 if missing."""

    async def override_get_risk_collector() -> FakeRiskCollector:
        """Override GitHub collector dependency for API tests."""
        return FakeRiskCollector()

    app.dependency_overrides[get_risk_collector] = override_get_risk_collector

    response = await release_run_api_client.post(
        f"/api/v1/release-runs/{uuid4()}/github-risks",
    )

    assert response.status_code == 404

    response_data = response.json()

    assert response_data["error"]["message"] == "Release run not found."