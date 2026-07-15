"""Tests for release-risk snapshot repository."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models.release_run import ReleaseRun
from app.models.release_run_risk_snapshot import ReleaseRunRiskSnapshot
from app.repositories.release_run_risk_snapshot_repository import (
    CreateReleaseRunRiskSnapshotCommand,
    ReleaseRunRiskSnapshotRepository,
)


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """Create an isolated async database session for snapshot repository tests."""
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
    """Create a parent release run for snapshot repository tests."""
    release_run = ReleaseRun(
        run_id=f"release-run-{uuid4().hex[:12]}",
        query="What are the biggest release risks this week?",
        requested_by="manager@example.com",
        status="completed",
    )

    session.add(release_run)
    await session.flush()
    await session.refresh(release_run)

    return release_run


@pytest.mark.anyio
async def test_create_snapshot_persists_release_risk_report(
    session: AsyncSession,
) -> None:
    """Repository should persist a trusted backend-generated risk snapshot."""
    release_run = await create_test_release_run(session)
    repository = ReleaseRunRiskSnapshotRepository(
        session=session,
        request_id="test-request-id",
    )

    snapshot = await repository.create_snapshot(
        CreateReleaseRunRiskSnapshotCommand(
            release_run_id=release_run.id,
            risk_payload={
                "release_run_id": str(release_run.id),
                "overall_severity": "critical",
                "top_risks": [
                    {
                        "source": "github",
                        "title": "Payment PR has failing CI",
                        "severity": "critical",
                    }
                ],
            },
            overall_severity="critical",
            approval_required=True,
            approval_status_at_snapshot="pending",
        )
    )

    await session.commit()

    assert isinstance(snapshot, ReleaseRunRiskSnapshot)
    assert snapshot.id is not None
    assert snapshot.release_run_id == release_run.id
    assert snapshot.snapshot_version == 1
    assert snapshot.overall_severity == "critical"
    assert snapshot.approval_required is True
    assert snapshot.approval_status_at_snapshot == "pending"

    payload = json.loads(snapshot.risk_payload_json)
    assert payload["overall_severity"] == "critical"
    assert payload["top_risks"][0]["source"] == "github"


@pytest.mark.anyio
async def test_create_snapshot_increments_version_per_release_run(
    session: AsyncSession,
) -> None:
    """Repository should increment snapshot version for the same release run."""
    release_run = await create_test_release_run(session)
    repository = ReleaseRunRiskSnapshotRepository(
        session=session,
        request_id="test-request-id",
    )

    first_snapshot = await repository.create_snapshot(
        CreateReleaseRunRiskSnapshotCommand(
            release_run_id=release_run.id,
            risk_payload={"overall_severity": "high", "version": 1},
            overall_severity="high",
            approval_required=True,
            approval_status_at_snapshot="pending",
        )
    )
    second_snapshot = await repository.create_snapshot(
        CreateReleaseRunRiskSnapshotCommand(
            release_run_id=release_run.id,
            risk_payload={"overall_severity": "critical", "version": 2},
            overall_severity="critical",
            approval_required=True,
            approval_status_at_snapshot="pending",
        )
    )

    await session.commit()

    assert first_snapshot.snapshot_version == 1
    assert second_snapshot.snapshot_version == 2


@pytest.mark.anyio
async def test_snapshot_version_is_scoped_to_release_run(
    session: AsyncSession,
) -> None:
    """Repository should restart snapshot versioning for each release run."""
    first_release_run = await create_test_release_run(session)
    second_release_run = await create_test_release_run(session)
    repository = ReleaseRunRiskSnapshotRepository(
        session=session,
        request_id="test-request-id",
    )

    first_release_snapshot = await repository.create_snapshot(
        CreateReleaseRunRiskSnapshotCommand(
            release_run_id=first_release_run.id,
            risk_payload={"release": "first"},
            overall_severity="high",
            approval_required=True,
            approval_status_at_snapshot="pending",
        )
    )
    second_release_snapshot = await repository.create_snapshot(
        CreateReleaseRunRiskSnapshotCommand(
            release_run_id=second_release_run.id,
            risk_payload={"release": "second"},
            overall_severity="low",
            approval_required=False,
            approval_status_at_snapshot="not_required",
        )
    )

    await session.commit()

    assert first_release_snapshot.snapshot_version == 1
    assert second_release_snapshot.snapshot_version == 1


@pytest.mark.anyio
async def test_get_latest_by_release_run_id_returns_highest_version(
    session: AsyncSession,
) -> None:
    """Repository should fetch the newest snapshot for a release run."""
    release_run = await create_test_release_run(session)
    repository = ReleaseRunRiskSnapshotRepository(
        session=session,
        request_id="test-request-id",
    )

    await repository.create_snapshot(
        CreateReleaseRunRiskSnapshotCommand(
            release_run_id=release_run.id,
            risk_payload={"overall_severity": "medium", "version": 1},
            overall_severity="medium",
            approval_required=False,
            approval_status_at_snapshot="not_required",
        )
    )
    await repository.create_snapshot(
        CreateReleaseRunRiskSnapshotCommand(
            release_run_id=release_run.id,
            risk_payload={"overall_severity": "critical", "version": 2},
            overall_severity="critical",
            approval_required=True,
            approval_status_at_snapshot="pending",
        )
    )

    await session.commit()

    latest_snapshot = await repository.get_latest_by_release_run_id(release_run.id)

    assert latest_snapshot is not None
    assert latest_snapshot.snapshot_version == 2
    assert latest_snapshot.overall_severity == "critical"

    payload = json.loads(latest_snapshot.risk_payload_json)
    assert payload["version"] == 2


@pytest.mark.anyio
async def test_get_latest_by_release_run_id_returns_none_when_missing(
    session: AsyncSession,
) -> None:
    """Repository should return None when no snapshot exists for a release run."""
    repository = ReleaseRunRiskSnapshotRepository(
        session=session,
        request_id="test-request-id",
    )

    latest_snapshot = await repository.get_latest_by_release_run_id(uuid4())

    assert latest_snapshot is None


@pytest.mark.anyio
async def test_list_by_release_run_id_returns_snapshots_in_version_order(
    session: AsyncSession,
) -> None:
    """Repository should list snapshot history in ascending version order."""
    release_run = await create_test_release_run(session)
    other_release_run = await create_test_release_run(session)
    repository = ReleaseRunRiskSnapshotRepository(
        session=session,
        request_id="test-request-id",
    )

    await repository.create_snapshot(
        CreateReleaseRunRiskSnapshotCommand(
            release_run_id=release_run.id,
            risk_payload={"version": 1},
            overall_severity="low",
            approval_required=False,
            approval_status_at_snapshot="not_required",
        )
    )
    await repository.create_snapshot(
        CreateReleaseRunRiskSnapshotCommand(
            release_run_id=release_run.id,
            risk_payload={"version": 2},
            overall_severity="high",
            approval_required=True,
            approval_status_at_snapshot="pending",
        )
    )
    await repository.create_snapshot(
        CreateReleaseRunRiskSnapshotCommand(
            release_run_id=other_release_run.id,
            risk_payload={"version": 1, "other_release": True},
            overall_severity="critical",
            approval_required=True,
            approval_status_at_snapshot="pending",
        )
    )

    await session.commit()

    snapshots = await repository.list_by_release_run_id(release_run.id)

    assert len(snapshots) == 2
    assert [snapshot.snapshot_version for snapshot in snapshots] == [1, 2]


@pytest.mark.anyio
async def test_list_by_release_run_id_validates_limit_and_offset(
    session: AsyncSession,
) -> None:
    """Repository should reject invalid pagination input."""
    repository = ReleaseRunRiskSnapshotRepository(
        session=session,
        request_id="test-request-id",
    )

    with pytest.raises(ValueError, match="limit must be greater than 0"):
        await repository.list_by_release_run_id(uuid4(), limit=0)

    with pytest.raises(ValueError, match="offset cannot be negative"):
        await repository.list_by_release_run_id(uuid4(), offset=-1)


def test_create_snapshot_command_rejects_non_json_serializable_payload() -> None:
    """Snapshot command should reject payloads that cannot be stored as JSON."""
    with pytest.raises(ValidationError, match="risk_payload must be JSON serializable"):
        CreateReleaseRunRiskSnapshotCommand(
            release_run_id=uuid4(),
            risk_payload={"bad_value": object()},
            overall_severity="critical",
            approval_required=True,
            approval_status_at_snapshot="pending",
        )


def test_create_snapshot_command_normalizes_text_fields() -> None:
    """Snapshot command should normalize text fields before persistence."""
    command = CreateReleaseRunRiskSnapshotCommand(
        release_run_id=uuid4(),
        risk_payload={"overall_severity": "high"},
        overall_severity=" high ",
        approval_required=True,
        approval_status_at_snapshot=" pending ",
    )

    assert command.overall_severity == "high"
    assert command.approval_status_at_snapshot == "pending"

@pytest.mark.anyio
async def test_list_latest_previous_release_snapshots_returns_latest_per_run(
    session: AsyncSession,
) -> None:
    """Repository should return the newest snapshot from each previous run."""
    current_release_run = await create_test_release_run(session)
    older_release_run = await create_test_release_run(session)
    newer_release_run = await create_test_release_run(session)
    oldest_release_run = await create_test_release_run(session)

    repository = ReleaseRunRiskSnapshotRepository(
        session=session,
        request_id="test-request-id",
    )
    reference_time = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)

    current_snapshot = await repository.create_snapshot(
        CreateReleaseRunRiskSnapshotCommand(
            release_run_id=current_release_run.id,
            risk_payload={"release": "current"},
            overall_severity="critical",
            approval_required=True,
            approval_status_at_snapshot="pending",
        )
    )
    older_version_one = await repository.create_snapshot(
        CreateReleaseRunRiskSnapshotCommand(
            release_run_id=older_release_run.id,
            risk_payload={"release": "older", "version": 1},
            overall_severity="medium",
            approval_required=False,
            approval_status_at_snapshot="not_required",
        )
    )
    older_version_two = await repository.create_snapshot(
        CreateReleaseRunRiskSnapshotCommand(
            release_run_id=older_release_run.id,
            risk_payload={"release": "older", "version": 2},
            overall_severity="high",
            approval_required=True,
            approval_status_at_snapshot="pending",
        )
    )
    newer_snapshot = await repository.create_snapshot(
        CreateReleaseRunRiskSnapshotCommand(
            release_run_id=newer_release_run.id,
            risk_payload={"release": "newer"},
            overall_severity="critical",
            approval_required=True,
            approval_status_at_snapshot="pending",
        )
    )
    oldest_snapshot = await repository.create_snapshot(
        CreateReleaseRunRiskSnapshotCommand(
            release_run_id=oldest_release_run.id,
            risk_payload={"release": "oldest"},
            overall_severity="low",
            approval_required=False,
            approval_status_at_snapshot="not_required",
        )
    )

    current_snapshot.created_at = reference_time
    older_version_one.created_at = reference_time - timedelta(hours=4)
    older_version_two.created_at = reference_time - timedelta(hours=3)
    newer_snapshot.created_at = reference_time - timedelta(hours=1)
    oldest_snapshot.created_at = reference_time - timedelta(hours=5)

    await session.commit()

    snapshots = await repository.list_latest_previous_release_snapshots(
        exclude_release_run_id=current_release_run.id,
        limit=2,
    )

    assert [snapshot.id for snapshot in snapshots] == [
        newer_snapshot.id,
        older_version_two.id,
    ]
    assert older_version_one.id not in {snapshot.id for snapshot in snapshots}
    assert current_snapshot.id not in {snapshot.id for snapshot in snapshots}

@pytest.mark.anyio
async def test_list_latest_previous_release_snapshots_validates_limit(
    session: AsyncSession,
) -> None:
    """Repository should reject unsafe historical lookup limits."""
    repository = ReleaseRunRiskSnapshotRepository(
        session=session,
        request_id="test-request-id",
    )

    with pytest.raises(ValueError, match="limit must be greater than 0"):
        await repository.list_latest_previous_release_snapshots(
            exclude_release_run_id=uuid4(),
            limit=0,
        )

    with pytest.raises(ValueError, match="limit cannot exceed 100"):
        await repository.list_latest_previous_release_snapshots(
            exclude_release_run_id=uuid4(),
            limit=101,
        )
