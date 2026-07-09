"""Tests for release-run Slack alert repository."""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models.release_run import ReleaseRun
from app.models.release_run_slack_alert import ReleaseRunSlackAlert
from app.repositories.release_run_slack_alert_repository import (
    CreateReleaseRunSlackAlertCommand,
    ReleaseRunSlackAlertAlreadySentError,
    ReleaseRunSlackAlertRepository,
)


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """Create an isolated async database session for Slack alert repository tests."""
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
    """Create a parent release run for Slack alert repository tests."""
    release_run = ReleaseRun(
        run_id=f"release-run-{uuid4().hex[:12]}",
        query="What are the biggest release risks this week?",
        requested_by="manager@example.com",
        status="approval_approved",
    )

    session.add(release_run)
    await session.flush()
    await session.refresh(release_run)

    return release_run


def build_command(release_run_id: object) -> CreateReleaseRunSlackAlertCommand:
    """Build a reusable valid Slack alert creation command."""
    return CreateReleaseRunSlackAlertCommand(
        release_run_id=release_run_id,
        approval_request_id=uuid4(),
        snapshot_id=uuid4(),
        snapshot_version=1,
        slack_channel="C1234567890",
        slack_timestamp="12345.6789",
        risk_level="high",
        risk_score=0.82,
        recommended_action="review_required",
    )


@pytest.mark.anyio
async def test_create_sent_alert_persists_slack_alert_record(
    session: AsyncSession,
) -> None:
    """Repository should persist one successful Slack alert record."""
    release_run = await create_test_release_run(session)
    repository = ReleaseRunSlackAlertRepository(
        session=session,
        request_id="test-request-id",
    )

    alert = await repository.create_sent_alert(build_command(release_run.id))

    await session.commit()

    assert isinstance(alert, ReleaseRunSlackAlert)
    assert alert.id is not None
    assert alert.release_run_id == release_run.id
    assert alert.delivery_status == "sent"
    assert alert.slack_channel == "C1234567890"
    assert alert.slack_timestamp == "12345.6789"
    assert alert.risk_level == "high"
    assert alert.risk_score == 0.82
    assert alert.recommended_action == "review_required"


@pytest.mark.anyio
async def test_get_by_release_run_id_returns_existing_alert(
    session: AsyncSession,
) -> None:
    """Repository should fetch the Slack alert record for a release run."""
    release_run = await create_test_release_run(session)
    repository = ReleaseRunSlackAlertRepository(
        session=session,
        request_id="test-request-id",
    )

    created_alert = await repository.create_sent_alert(build_command(release_run.id))

    await session.commit()

    fetched_alert = await repository.get_by_release_run_id(release_run.id)

    assert fetched_alert is not None
    assert fetched_alert.id == created_alert.id
    assert fetched_alert.release_run_id == release_run.id


@pytest.mark.anyio
async def test_get_by_release_run_id_returns_none_when_missing(
    session: AsyncSession,
) -> None:
    """Repository should return None when no Slack alert exists."""
    repository = ReleaseRunSlackAlertRepository(
        session=session,
        request_id="test-request-id",
    )

    fetched_alert = await repository.get_by_release_run_id(uuid4())

    assert fetched_alert is None


@pytest.mark.anyio
async def test_create_sent_alert_blocks_duplicate_release_run_alert(
    session: AsyncSession,
) -> None:
    """Repository should enforce one Slack alert per release run."""
    release_run = await create_test_release_run(session)
    repository = ReleaseRunSlackAlertRepository(
        session=session,
        request_id="test-request-id",
    )

    await repository.create_sent_alert(build_command(release_run.id))

    with pytest.raises(
        ReleaseRunSlackAlertAlreadySentError,
        match="Slack alert already sent for this release run.",
    ):
        await repository.create_sent_alert(
            build_command(release_run.id).model_copy(
                update={"slack_timestamp": "22222.3333"}
            )
        )


@pytest.mark.anyio
async def test_list_by_status_returns_matching_alerts(
    session: AsyncSession,
) -> None:
    """Repository should list Slack alert records by delivery status."""
    first_release_run = await create_test_release_run(session)
    second_release_run = await create_test_release_run(session)
    repository = ReleaseRunSlackAlertRepository(
        session=session,
        request_id="test-request-id",
    )

    await repository.create_sent_alert(build_command(first_release_run.id))
    await repository.create_sent_alert(build_command(second_release_run.id))

    await session.commit()

    alerts = await repository.list_by_status("sent")

    assert len(alerts) == 2
    assert {alert.release_run_id for alert in alerts} == {
        first_release_run.id,
        second_release_run.id,
    }


@pytest.mark.anyio
async def test_list_by_status_validates_pagination_and_status(
    session: AsyncSession,
) -> None:
    """Repository should reject invalid list input."""
    repository = ReleaseRunSlackAlertRepository(
        session=session,
        request_id="test-request-id",
    )

    with pytest.raises(ValueError, match="limit must be greater than 0"):
        await repository.list_by_status("sent", limit=0)

    with pytest.raises(ValueError, match="offset cannot be negative"):
        await repository.list_by_status("sent", offset=-1)

    with pytest.raises(ValueError, match="delivery_status must not be blank"):
        await repository.list_by_status("   ")


def test_create_slack_alert_command_rejects_invalid_score() -> None:
    """Command should reject out-of-range risk scores."""
    with pytest.raises(ValidationError):
        CreateReleaseRunSlackAlertCommand(
            release_run_id=uuid4(),
            approval_request_id=uuid4(),
            snapshot_id=uuid4(),
            snapshot_version=1,
            slack_channel="C1234567890",
            slack_timestamp="12345.6789",
            risk_level="high",
            risk_score=2.0,
            recommended_action="review_required",
        )


def test_create_slack_alert_command_normalizes_text_fields() -> None:
    """Command should strip text fields before persistence."""
    command = CreateReleaseRunSlackAlertCommand(
        release_run_id=uuid4(),
        approval_request_id=uuid4(),
        snapshot_id=uuid4(),
        snapshot_version=1,
        slack_channel=" C1234567890 ",
        slack_timestamp=" 12345.6789 ",
        risk_level=" high ",
        risk_score=0.82,
        recommended_action=" review_required ",
        delivery_status=" sent ",
    )

    assert command.slack_channel == "C1234567890"
    assert command.slack_timestamp == "12345.6789"
    assert command.risk_level == "high"
    assert command.recommended_action == "review_required"
    assert command.delivery_status == "sent"
