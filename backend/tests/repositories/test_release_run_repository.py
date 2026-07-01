from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.release_run import ReleaseRun
from app.repositories.release_run_repository import ReleaseRunRepository

@pytest.mark.anyio
async def test_create_release_run_persists_record(
    db_session: AsyncSession,
) -> None:
    """Repository should create and return a release run."""
    repository = ReleaseRunRepository(
        session=db_session,
        request_id="test-request-id",
    )

    release_run = ReleaseRun(
        run_id="test-run-001",
        query="What are the biggest release risks this week?",
        requested_by="manager@example.com",
        status="created",
    )

    created_release_run = await repository.create(release_run)

    assert created_release_run.id is not None
    assert created_release_run.run_id == "test-run-001"
    assert created_release_run.query == "What are the biggest release risks this week?"
    assert created_release_run.requested_by == "manager@example.com"
    assert created_release_run.status == "created"
    assert created_release_run.completed_at is None


@pytest.mark.anyio
async def test_get_by_id_returns_release_run_when_found(
    db_session: AsyncSession,
) -> None:
    """Repository should return a release run by ID when it exists."""
    repository = ReleaseRunRepository(
        session=db_session,
        request_id="test-request-id",
    )

    release_run = ReleaseRun(
        run_id="test-run-002",
        query="Check release readiness",
        requested_by="manager@example.com",
        status="created",
    )

    created_release_run = await repository.create(release_run)

    fetched_release_run = await repository.get_by_id(created_release_run.id)

    assert fetched_release_run is not None
    assert fetched_release_run.id == created_release_run.id
    assert fetched_release_run.run_id == "test-run-002"


@pytest.mark.anyio
async def test_get_by_id_returns_none_when_not_found(
    db_session: AsyncSession,
) -> None:
    """Repository should return None when the release run does not exist."""
    repository = ReleaseRunRepository(
        session=db_session,
        request_id="test-request-id",
    )

    fetched_release_run = await repository.get_by_id(uuid4())

    assert fetched_release_run is None


@pytest.mark.anyio
async def test_list_recent_returns_release_runs(
    db_session: AsyncSession,
) -> None:
    """Repository should list recent release runs."""
    repository = ReleaseRunRepository(
        session=db_session,
        request_id="test-request-id",
    )

    first_release_run = ReleaseRun(
        run_id="test-run-003",
        query="First release risk query",
        requested_by="manager@example.com",
        status="created",
    )
    second_release_run = ReleaseRun(
        run_id="test-run-004",
        query="Second release risk query",
        requested_by="manager@example.com",
        status="created",
    )

    await repository.create(first_release_run)
    await repository.create(second_release_run)

    release_runs = await repository.list_recent(limit=10, offset=0)

    assert len(release_runs) >= 2
    assert any(run.run_id == "test-run-003" for run in release_runs)
    assert any(run.run_id == "test-run-004" for run in release_runs)


@pytest.mark.anyio
async def test_list_recent_rejects_invalid_limit(
    db_session: AsyncSession,
) -> None:
    """Repository should reject invalid pagination limit."""
    repository = ReleaseRunRepository(
        session=db_session,
        request_id="test-request-id",
    )

    with pytest.raises(ValueError, match="limit must be greater than 0"):
        await repository.list_recent(limit=0)


@pytest.mark.anyio
async def test_list_recent_rejects_invalid_offset(
    db_session: AsyncSession,
) -> None:
    """Repository should reject invalid pagination offset."""
    repository = ReleaseRunRepository(
        session=db_session,
        request_id="test-request-id",
    )

    with pytest.raises(ValueError, match="offset cannot be negative"):
        await repository.list_recent(offset=-1)


@pytest.mark.anyio
async def test_update_status_updates_release_run(
    db_session: AsyncSession,
) -> None:
    """Repository should update release run status."""
    repository = ReleaseRunRepository(
        session=db_session,
        request_id="test-request-id",
    )

    release_run = ReleaseRun(
        run_id="test-run-005",
        query="Check deployment risk",
        requested_by="manager@example.com",
        status="created",
    )

    created_release_run = await repository.create(release_run)

    updated_release_run = await repository.update_status(
        release_run_id=created_release_run.id,
        status="running",
    )

    assert updated_release_run is not None
    assert updated_release_run.status == "running"
    assert updated_release_run.completed_at is None


@pytest.mark.anyio
async def test_update_status_sets_completed_at_for_terminal_status(
    db_session: AsyncSession,
) -> None:
    """Repository should set completed_at for terminal statuses."""
    repository = ReleaseRunRepository(
        session=db_session,
        request_id="test-request-id",
    )

    release_run = ReleaseRun(
        run_id="test-run-006",
        query="Check final release risk",
        requested_by="manager@example.com",
        status="created",
    )

    created_release_run = await repository.create(release_run)

    updated_release_run = await repository.update_status(
        release_run_id=created_release_run.id,
        status="completed",
    )

    assert updated_release_run is not None
    assert updated_release_run.status == "completed"
    assert updated_release_run.completed_at is not None


@pytest.mark.anyio
async def test_update_status_returns_none_when_release_run_missing(
    db_session: AsyncSession,
) -> None:
    """Repository should return None when updating a missing release run."""
    repository = ReleaseRunRepository(
        session=db_session,
        request_id="test-request-id",
    )

    updated_release_run = await repository.update_status(
        release_run_id=uuid4(),
        status="running",
    )

    assert updated_release_run is None