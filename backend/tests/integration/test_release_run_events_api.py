"""Integration tests for release-run audit event API routes."""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.api.dependencies.security import get_current_principal
from app.core.security import AuthenticatedPrincipal
from app.db.base import Base
from app.db.session import get_db_session
from app.main import app
from app.repositories.release_run_event_repository import (
    CreateReleaseRunEventCommand,
    ReleaseRunEventRepository,
)


@pytest.fixture
async def release_run_events_api_client() -> AsyncIterator[AsyncClient]:
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
async def test_list_release_run_events_api_returns_audit_timeline(
    release_run_events_api_client: AsyncClient,
) -> None:
    """GET /release-runs/{id}/events should return audit events in order."""

    create_response = await release_run_events_api_client.post(
        "/api/v1/release-runs",
        json={
            "query": "What are the biggest release risks this week?",
        },
    )

    assert create_response.status_code == 201

    release_run_id = create_response.json()["id"]

    async for session in app.dependency_overrides[get_db_session]():
        event_repository = ReleaseRunEventRepository(
            session=session,
            request_id="test-request-id",
        )

        await event_repository.create(
            CreateReleaseRunEventCommand(
                release_run_id=release_run_id,
                event_type="workflow_started",
                event_status="started",
                message="Workflow started.",
                metadata_json={"source": "test"},
            )
        )
        await event_repository.create(
            CreateReleaseRunEventCommand(
                release_run_id=release_run_id,
                event_type="workflow_completed",
                event_status="success",
                message="Workflow completed.",
                metadata_json={"risk_count": 3},
            )
        )

        await session.commit()
        break

    response = await release_run_events_api_client.get(
        f"/api/v1/release-runs/{release_run_id}/events",
    )

    assert response.status_code == 200

    response_data = response.json()

    assert response_data["release_run_id"] == release_run_id
    assert len(response_data["events"]) == 3

    assert response_data["events"][0]["event_type"] == "release_run_started"
    assert response_data["events"][0]["event_status"] == "success"

    assert response_data["events"][1]["event_type"] == "workflow_started"
    assert response_data["events"][1]["event_status"] == "started"
    assert response_data["events"][1]["message"] == "Workflow started."
    assert response_data["events"][1]["metadata_json"] == {"source": "test"}

    assert response_data["events"][2]["event_type"] == "workflow_completed"
    assert response_data["events"][2]["event_status"] == "success"
    assert response_data["events"][2]["metadata_json"] == {"risk_count": 3}


@pytest.mark.anyio
async def test_list_release_run_events_api_returns_start_event_after_creation(
    release_run_events_api_client: AsyncClient,
) -> None:
    """GET /release-runs/{id}/events should include the creation audit event."""

    create_response = await release_run_events_api_client.post(
        "/api/v1/release-runs",
        json={
            "query": "Check release readiness for this week.",
        },
    )

    assert create_response.status_code == 201

    release_run_id = create_response.json()["id"]

    response = await release_run_events_api_client.get(
        f"/api/v1/release-runs/{release_run_id}/events",
    )

    assert response.status_code == 200

    response_data = response.json()

    assert response_data["release_run_id"] == release_run_id
    assert len(response_data["events"]) == 1
    assert response_data["events"][0]["event_type"] == "release_run_started"
    assert response_data["events"][0]["event_status"] == "success"


@pytest.mark.anyio
async def test_list_release_run_events_api_returns_404_when_release_run_missing(
    release_run_events_api_client: AsyncClient,
) -> None:
    """GET /release-runs/{id}/events should return 404 for missing release run."""

    response = await release_run_events_api_client.get(
        f"/api/v1/release-runs/{uuid4()}/events",
    )

    assert response.status_code == 404

    response_data = response.json()

    assert response_data["error"]["message"] == "Release run not found."
