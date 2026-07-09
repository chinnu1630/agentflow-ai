"""Integration tests for release-run approval API routes."""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.session import get_db_session
from app.main import app
from app.repositories.release_run_approval_repository import (
    CreateReleaseRunApprovalCommand,
    ReleaseRunApprovalRepository,
)


@pytest.fixture
async def release_run_approvals_api_client() -> AsyncIterator[AsyncClient]:
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


async def _create_release_run(client: AsyncClient) -> str:
    """Create a release run through the public API and return its ID."""
    response = await client.post(
        "/api/v1/release-runs",
        json={
            "query": "What are the biggest release risks this week?",
            "requested_by": "manager@example.com",
        },
    )

    assert response.status_code == 201

    return str(response.json()["id"])


async def _create_pending_approval(release_run_id: str) -> str:
    """Create a pending approval request directly through the repository."""
    async for session in app.dependency_overrides[get_db_session]():
        repository = ReleaseRunApprovalRepository(
            session=session,
            request_id="test-request-id",
        )
        approval = await repository.create_pending(
            CreateReleaseRunApprovalCommand(
                release_run_id=release_run_id,
                approval_reason="High release risk requires manager approval.",
                approval_policy_version="hitl_policy_v1",
                requested_by="manager@example.com",
            )
        )
        await session.commit()
        return str(approval.id)

    raise AssertionError("Database session override did not yield a session.")


@pytest.mark.anyio
async def test_list_release_run_approvals_api_returns_approval_history(
    release_run_approvals_api_client: AsyncClient,
) -> None:
    """GET /release-runs/{id}/approvals should return approval requests."""
    release_run_id = await _create_release_run(release_run_approvals_api_client)
    approval_id = await _create_pending_approval(release_run_id)

    response = await release_run_approvals_api_client.get(
        f"/api/v1/release-runs/{release_run_id}/approvals",
    )

    assert response.status_code == 200

    response_data = response.json()

    assert response_data["release_run_id"] == release_run_id
    assert len(response_data["approvals"]) == 1
    assert response_data["approvals"][0]["id"] == approval_id
    assert response_data["approvals"][0]["approval_status"] == "pending"
    assert response_data["approvals"][0]["approval_policy_version"] == (
        "hitl_policy_v1"
    )


@pytest.mark.anyio
async def test_list_release_run_approvals_api_returns_404_when_release_run_missing(
    release_run_approvals_api_client: AsyncClient,
) -> None:
    """GET /release-runs/{id}/approvals should return 404 for missing run."""
    response = await release_run_approvals_api_client.get(
        f"/api/v1/release-runs/{uuid4()}/approvals",
    )

    assert response.status_code == 404
    assert response.json()["error"]["message"] == "Release run not found."


@pytest.mark.anyio
async def test_decide_release_run_approval_api_approves_pending_request(
    release_run_approvals_api_client: AsyncClient,
) -> None:
    """POST /approvals/{id}/decision should approve a pending approval."""
    release_run_id = await _create_release_run(release_run_approvals_api_client)
    approval_id = await _create_pending_approval(release_run_id)

    response = await release_run_approvals_api_client.post(
        f"/api/v1/release-runs/{release_run_id}/approvals/{approval_id}/decision",
        json={
            "approval_status": "approved",
            "decided_by": "director@example.com",
            "decision_note": "Approved after reviewing rollback plan.",
        },
    )

    assert response.status_code == 200

    response_data = response.json()

    assert response_data["id"] == approval_id
    assert response_data["release_run_id"] == release_run_id
    assert response_data["approval_status"] == "approved"
    assert response_data["decided_by"] == "director@example.com"
    assert response_data["decision_note"] == "Approved after reviewing rollback plan."
    assert response_data["decided_at"] is not None

    events_response = await release_run_approvals_api_client.get(
        f"/api/v1/release-runs/{release_run_id}/events",
    )

    assert events_response.status_code == 200

    event_types = [
        event["event_type"]
        for event in events_response.json()["events"]
    ]

    assert "approval_request_decided" in event_types


@pytest.mark.anyio
async def test_decide_release_run_approval_api_rejects_pending_status(
    release_run_approvals_api_client: AsyncClient,
) -> None:
    """Decision API should reject pending as a terminal decision."""
    release_run_id = await _create_release_run(release_run_approvals_api_client)
    approval_id = await _create_pending_approval(release_run_id)

    response = await release_run_approvals_api_client.post(
        f"/api/v1/release-runs/{release_run_id}/approvals/{approval_id}/decision",
        json={
            "approval_status": "pending",
            "decided_by": "director@example.com",
        },
    )

    assert response.status_code == 422


@pytest.mark.anyio
async def test_decide_release_run_approval_api_rejects_double_decision(
    release_run_approvals_api_client: AsyncClient,
) -> None:
    """Decision API should reject deciding an already-decided approval."""
    release_run_id = await _create_release_run(release_run_approvals_api_client)
    approval_id = await _create_pending_approval(release_run_id)

    first_response = await release_run_approvals_api_client.post(
        f"/api/v1/release-runs/{release_run_id}/approvals/{approval_id}/decision",
        json={
            "approval_status": "approved",
            "decided_by": "director@example.com",
        },
    )

    assert first_response.status_code == 200

    second_response = await release_run_approvals_api_client.post(
        f"/api/v1/release-runs/{release_run_id}/approvals/{approval_id}/decision",
        json={
            "approval_status": "rejected",
            "decided_by": "director@example.com",
        },
    )

    assert second_response.status_code == 409
    assert second_response.json()["error"]["message"] == (
        "Only pending approval requests can be decided."
    )


@pytest.mark.anyio
async def test_decide_release_run_approval_api_returns_404_for_wrong_release_run(
    release_run_approvals_api_client: AsyncClient,
) -> None:
    """Decision API should reject approval IDs from another release run."""
    release_run_id = await _create_release_run(release_run_approvals_api_client)
    other_release_run_id = await _create_release_run(release_run_approvals_api_client)
    approval_id = await _create_pending_approval(other_release_run_id)

    response = await release_run_approvals_api_client.post(
        f"/api/v1/release-runs/{release_run_id}/approvals/{approval_id}/decision",
        json={
            "approval_status": "approved",
            "decided_by": "director@example.com",
        },
    )

    assert response.status_code == 404
    assert response.json()["error"]["message"] == "Approval request not found."
