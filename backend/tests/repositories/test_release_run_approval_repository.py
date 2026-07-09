"""Tests for release-run approval repository."""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models.release_run import ReleaseRun
from app.models.release_run_approval import ReleaseRunApproval
from app.repositories.release_run_approval_repository import (
    CreateReleaseRunApprovalCommand,
    DecideReleaseRunApprovalCommand,
    ReleaseRunApprovalRepository,
    ReleaseRunApprovalStatus,
)


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """Create an isolated async database session for approval repository tests."""
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
    """Create a parent release run for approval repository tests."""
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
async def test_create_pending_persists_release_run_approval(
    session: AsyncSession,
) -> None:
    """Repository should persist a pending approval request."""
    release_run = await create_test_release_run(session)
    repository = ReleaseRunApprovalRepository(
        session=session,
        request_id="test-request-id",
    )

    approval = await repository.create_pending(
        CreateReleaseRunApprovalCommand(
            release_run_id=release_run.id,
            approval_reason="Critical release risk requires manager approval.",
            approval_policy_version="hitl_policy_v1",
            requested_by="manager@example.com",
        )
    )

    await session.commit()

    assert isinstance(approval, ReleaseRunApproval)
    assert approval.id is not None
    assert approval.release_run_id == release_run.id
    assert approval.approval_status == ReleaseRunApprovalStatus.PENDING.value
    assert approval.approval_reason == (
        "Critical release risk requires manager approval."
    )
    assert approval.approval_policy_version == "hitl_policy_v1"
    assert approval.requested_by == "manager@example.com"
    assert approval.decided_by is None
    assert approval.decided_at is None


@pytest.mark.anyio
async def test_get_latest_by_release_run_id_returns_latest_approval(
    session: AsyncSession,
) -> None:
    """Repository should return the newest approval request for a release run."""
    release_run = await create_test_release_run(session)
    repository = ReleaseRunApprovalRepository(
        session=session,
        request_id="test-request-id",
    )

    first = await repository.create_pending(
        CreateReleaseRunApprovalCommand(
            release_run_id=release_run.id,
            approval_reason="First approval reason.",
            approval_policy_version="hitl_policy_v1",
        )
    )
    second = await repository.create_pending(
        CreateReleaseRunApprovalCommand(
            release_run_id=release_run.id,
            approval_reason="Second approval reason.",
            approval_policy_version="hitl_policy_v1",
        )
    )

    await session.commit()

    latest = await repository.get_latest_by_release_run_id(release_run.id)

    assert latest is not None
    assert latest.id == second.id
    assert latest.id != first.id
    assert latest.approval_reason == "Second approval reason."


@pytest.mark.anyio
async def test_list_by_release_run_id_returns_approvals_in_created_order(
    session: AsyncSession,
) -> None:
    """Repository should list approvals for one release run in order."""
    release_run = await create_test_release_run(session)
    other_release_run = await create_test_release_run(session)
    repository = ReleaseRunApprovalRepository(
        session=session,
        request_id="test-request-id",
    )

    await repository.create_pending(
        CreateReleaseRunApprovalCommand(
            release_run_id=release_run.id,
            approval_reason="First approval.",
            approval_policy_version="hitl_policy_v1",
        )
    )
    await repository.create_pending(
        CreateReleaseRunApprovalCommand(
            release_run_id=release_run.id,
            approval_reason="Second approval.",
            approval_policy_version="hitl_policy_v1",
        )
    )
    await repository.create_pending(
        CreateReleaseRunApprovalCommand(
            release_run_id=other_release_run.id,
            approval_reason="Other release approval.",
            approval_policy_version="hitl_policy_v1",
        )
    )

    await session.commit()

    approvals = await repository.list_by_release_run_id(release_run.id)

    assert len(approvals) == 2
    assert approvals[0].approval_reason == "First approval."
    assert approvals[1].approval_reason == "Second approval."


@pytest.mark.anyio
async def test_decide_approves_pending_release_run_approval(
    session: AsyncSession,
) -> None:
    """Repository should approve a pending approval request."""
    release_run = await create_test_release_run(session)
    repository = ReleaseRunApprovalRepository(
        session=session,
        request_id="test-request-id",
    )
    approval = await repository.create_pending(
        CreateReleaseRunApprovalCommand(
            release_run_id=release_run.id,
            approval_reason="Critical risk requires approval.",
            approval_policy_version="hitl_policy_v1",
        )
    )

    decided = await repository.decide(
        DecideReleaseRunApprovalCommand(
            approval_id=approval.id,
            approval_status=ReleaseRunApprovalStatus.APPROVED,
            decided_by="director@example.com",
            decision_note="Approved after reviewing rollback plan.",
        )
    )

    await session.commit()

    assert decided is not None
    assert decided.id == approval.id
    assert decided.approval_status == ReleaseRunApprovalStatus.APPROVED.value
    assert decided.decided_by == "director@example.com"
    assert decided.decision_note == "Approved after reviewing rollback plan."
    assert decided.decided_at is not None


@pytest.mark.anyio
async def test_decide_rejects_missing_approval(
    session: AsyncSession,
) -> None:
    """Repository should return None when approval request is missing."""
    repository = ReleaseRunApprovalRepository(
        session=session,
        request_id="test-request-id",
    )

    decided = await repository.decide(
        DecideReleaseRunApprovalCommand(
            approval_id=uuid4(),
            approval_status=ReleaseRunApprovalStatus.REJECTED,
            decided_by="director@example.com",
        )
    )

    assert decided is None


@pytest.mark.anyio
async def test_decide_rejects_non_pending_approval(
    session: AsyncSession,
) -> None:
    """Repository should not allow approving or rejecting twice."""
    release_run = await create_test_release_run(session)
    repository = ReleaseRunApprovalRepository(
        session=session,
        request_id="test-request-id",
    )
    approval = await repository.create_pending(
        CreateReleaseRunApprovalCommand(
            release_run_id=release_run.id,
            approval_reason="Critical risk requires approval.",
            approval_policy_version="hitl_policy_v1",
        )
    )

    await repository.decide(
        DecideReleaseRunApprovalCommand(
            approval_id=approval.id,
            approval_status=ReleaseRunApprovalStatus.APPROVED,
            decided_by="director@example.com",
        )
    )

    with pytest.raises(
        ValueError,
        match="Only pending approval requests can be decided.",
    ):
        await repository.decide(
            DecideReleaseRunApprovalCommand(
                approval_id=approval.id,
                approval_status=ReleaseRunApprovalStatus.REJECTED,
                decided_by="director@example.com",
            )
        )


@pytest.mark.anyio
async def test_list_by_release_run_id_rejects_invalid_pagination(
    session: AsyncSession,
) -> None:
    """Repository should reject invalid pagination input."""
    repository = ReleaseRunApprovalRepository(
        session=session,
        request_id="test-request-id",
    )

    with pytest.raises(ValueError, match="limit must be greater than 0"):
        await repository.list_by_release_run_id(uuid4(), limit=0)

    with pytest.raises(ValueError, match="offset cannot be negative"):
        await repository.list_by_release_run_id(uuid4(), offset=-1)


def test_create_release_run_approval_command_rejects_invalid_payload() -> None:
    """Create command should reject blank approval metadata."""
    with pytest.raises(ValidationError):
        CreateReleaseRunApprovalCommand(
            release_run_id=uuid4(),
            approval_reason=" ",
            approval_policy_version="hitl_policy_v1",
        )


def test_decide_release_run_approval_command_rejects_pending_decision() -> None:
    """Decision command should reject pending as a terminal decision."""
    with pytest.raises(ValidationError):
        DecideReleaseRunApprovalCommand(
            approval_id=uuid4(),
            approval_status=ReleaseRunApprovalStatus.PENDING,
            decided_by="director@example.com",
        )


@pytest.mark.anyio
async def test_list_by_status_returns_only_matching_approvals(
    session: AsyncSession,
) -> None:
    """Repository should list approvals filtered by status."""
    release_run = await create_test_release_run(session)
    other_release_run = await create_test_release_run(session)
    repository = ReleaseRunApprovalRepository(
        session=session,
        request_id="test-request-id",
    )

    pending_approval = await repository.create_pending(
        CreateReleaseRunApprovalCommand(
            release_run_id=release_run.id,
            approval_reason="Pending approval.",
            approval_policy_version="hitl_policy_v1",
        )
    )
    approved_approval = await repository.create_pending(
        CreateReleaseRunApprovalCommand(
            release_run_id=other_release_run.id,
            approval_reason="Approval to decide.",
            approval_policy_version="hitl_policy_v1",
        )
    )

    await repository.decide(
        DecideReleaseRunApprovalCommand(
            approval_id=approved_approval.id,
            approval_status=ReleaseRunApprovalStatus.APPROVED,
            decided_by="director@example.com",
        )
    )

    await session.commit()

    pending_approvals = await repository.list_by_status(
        ReleaseRunApprovalStatus.PENDING
    )

    assert len(pending_approvals) == 1
    assert pending_approvals[0].id == pending_approval.id
    assert pending_approvals[0].approval_status == "pending"


@pytest.mark.anyio
async def test_list_by_status_supports_limit_and_offset(
    session: AsyncSession,
) -> None:
    """Repository should paginate approvals filtered by status."""
    release_run = await create_test_release_run(session)
    repository = ReleaseRunApprovalRepository(
        session=session,
        request_id="test-request-id",
    )

    for index in range(3):
        await repository.create_pending(
            CreateReleaseRunApprovalCommand(
                release_run_id=release_run.id,
                approval_reason=f"Pending approval {index}.",
                approval_policy_version="hitl_policy_v1",
            )
        )

    await session.commit()

    approvals = await repository.list_by_status(
        ReleaseRunApprovalStatus.PENDING,
        limit=1,
        offset=1,
    )

    assert len(approvals) == 1
    assert approvals[0].approval_reason == "Pending approval 1."


@pytest.mark.anyio
async def test_list_by_status_rejects_invalid_pagination(
    session: AsyncSession,
) -> None:
    """Repository should reject invalid status-list pagination input."""
    repository = ReleaseRunApprovalRepository(
        session=session,
        request_id="test-request-id",
    )

    with pytest.raises(ValueError, match="limit must be greater than 0"):
        await repository.list_by_status(
            ReleaseRunApprovalStatus.PENDING,
            limit=0,
        )

    with pytest.raises(ValueError, match="offset cannot be negative"):
        await repository.list_by_status(
            ReleaseRunApprovalStatus.PENDING,
            offset=-1,
        )
