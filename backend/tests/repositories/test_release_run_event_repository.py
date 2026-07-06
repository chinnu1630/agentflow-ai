from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models.release_run import ReleaseRun
from app.models.release_run_event import ReleaseRunEvent
from app.repositories.release_run_event_repository import (
    CreateReleaseRunEventCommand,
    ReleaseRunEventRepository,
)


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """Create an isolated async database session for repository tests."""

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

    async with session_factory() as db_session:
        yield db_session

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)

    await engine.dispose()


async def create_test_release_run(session: AsyncSession) -> ReleaseRun:
    """Create a parent release run for audit event tests."""

    release_run = ReleaseRun(
        run_id=f"release-run-{uuid4().hex[:12]}",
        query="What are the biggest release risks this week?",
        requested_by="manager@example.com",
        status="created",
    )

    session.add(release_run)
    await session.flush()
    await session.refresh(release_run)

    return release_run


@pytest.mark.anyio
async def test_create_release_run_event_persists_audit_event(
    session: AsyncSession,
) -> None:
    """Repository should persist a release-run audit event."""

    release_run = await create_test_release_run(session)
    repository = ReleaseRunEventRepository(
        session=session,
        request_id="test-request-id",
    )

    event = await repository.create(
        CreateReleaseRunEventCommand(
            release_run_id=release_run.id,
            event_type="github_collection_completed",
            event_status="success",
            message="GitHub risks collected successfully.",
            metadata_json={
                "pull_request_count": 2,
                "risk_result_count": 2,
                "high_risk_count": 1,
            },
        )
    )

    await session.commit()

    assert isinstance(event, ReleaseRunEvent)
    assert event.id is not None
    assert event.release_run_id == release_run.id
    assert event.event_type == "github_collection_completed"
    assert event.event_status == "success"
    assert event.message == "GitHub risks collected successfully."
    assert event.metadata_json["pull_request_count"] == 2
    assert event.created_at is not None


@pytest.mark.anyio
async def test_list_by_release_run_id_returns_events_in_created_order(
    session: AsyncSession,
) -> None:
    """Repository should list audit events for one release run in order."""

    release_run = await create_test_release_run(session)
    other_release_run = await create_test_release_run(session)

    repository = ReleaseRunEventRepository(
        session=session,
        request_id="test-request-id",
    )

    await repository.create(
        CreateReleaseRunEventCommand(
            release_run_id=release_run.id,
            event_type="workflow_started",
            event_status="success",
            message="Workflow started.",
        )
    )
    await repository.create(
        CreateReleaseRunEventCommand(
            release_run_id=release_run.id,
            event_type="github_collection_completed",
            event_status="success",
            message="GitHub collection completed.",
        )
    )
    await repository.create(
        CreateReleaseRunEventCommand(
            release_run_id=other_release_run.id,
            event_type="workflow_started",
            event_status="success",
            message="Other workflow started.",
        )
    )

    await session.commit()

    events = await repository.list_by_release_run_id(release_run.id)

    assert len(events) == 2
    assert events[0].event_type == "workflow_started"
    assert events[1].event_type == "github_collection_completed"


@pytest.mark.anyio
async def test_list_by_release_run_id_supports_limit_and_offset(
    session: AsyncSession,
) -> None:
    """Repository should support pagination for audit events."""

    release_run = await create_test_release_run(session)
    repository = ReleaseRunEventRepository(
        session=session,
        request_id="test-request-id",
    )

    for index in range(3):
        await repository.create(
            CreateReleaseRunEventCommand(
                release_run_id=release_run.id,
                event_type=f"event_{index}",
                event_status="success",
                message=f"Event {index}.",
            )
        )

    await session.commit()

    events = await repository.list_by_release_run_id(
        release_run.id,
        limit=1,
        offset=1,
    )

    assert len(events) == 1
    assert events[0].event_type == "event_1"


@pytest.mark.anyio
async def test_list_by_release_run_id_rejects_invalid_limit(
    session: AsyncSession,
) -> None:
    """Repository should reject invalid pagination limits."""

    repository = ReleaseRunEventRepository(
        session=session,
        request_id="test-request-id",
    )

    with pytest.raises(ValueError, match="limit must be greater than 0"):
        await repository.list_by_release_run_id(uuid4(), limit=0)


@pytest.mark.anyio
async def test_list_by_release_run_id_rejects_invalid_offset(
    session: AsyncSession,
) -> None:
    """Repository should reject invalid pagination offsets."""

    repository = ReleaseRunEventRepository(
        session=session,
        request_id="test-request-id",
    )

    with pytest.raises(ValueError, match="offset cannot be negative"):
        await repository.list_by_release_run_id(uuid4(), offset=-1)


def test_create_release_run_event_command_rejects_invalid_payload() -> None:
    """CreateReleaseRunEventCommand should validate audit event input."""

    with pytest.raises(ValidationError):
        CreateReleaseRunEventCommand(
            release_run_id=uuid4(),
            event_type="",
            event_status="success",
            message="Invalid event.",
        )
