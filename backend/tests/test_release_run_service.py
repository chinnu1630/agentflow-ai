from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from app.models.release_run import ReleaseRun
from app.repositories.release_run_repository import ReleaseRunRepositoryError
from app.services.release_run_service import (
    ReleaseRunService,
    ReleaseRunServiceError,
    StartReleaseRunCommand,
)


class FakeReleaseRunRepository:
    """Fake repository used to unit test ReleaseRunService without a database."""

    def __init__(self, should_fail: bool = False) -> None:
        """Initialize the fake repository.

        Args:
            should_fail: Whether repository methods should raise an error.
        """
        self.should_fail = should_fail
        self.release_runs: dict[UUID, ReleaseRun] = {}

    async def create(self, release_run: ReleaseRun) -> ReleaseRun:
        """Fake creation of a release run."""
        if self.should_fail:
            raise ReleaseRunRepositoryError("Database create failed.")

        release_run.id = uuid4()
        release_run.created_at = datetime.now(UTC)
        release_run.completed_at = None

        self.release_runs[release_run.id] = release_run

        return release_run

    async def get_by_id(self, release_run_id: UUID) -> ReleaseRun | None:
        """Fake lookup of a release run by ID."""
        if self.should_fail:
            raise ReleaseRunRepositoryError("Database lookup failed.")

        return self.release_runs.get(release_run_id)

    async def update_status(
        self,
        release_run_id: UUID,
        status: str,
    ) -> ReleaseRun | None:
        """Fake status update for a release run."""
        if self.should_fail:
            raise ReleaseRunRepositoryError("Database update failed.")

        release_run = self.release_runs.get(release_run_id)

        if release_run is None:
            return None

        release_run.status = status

        if status in {"completed", "failed", "cancelled"}:
            release_run.completed_at = datetime.now(UTC)

        return release_run


@pytest.mark.anyio
async def test_start_release_run_creates_release_run() -> None:
    """Service should start a release run with created status."""
    repository = FakeReleaseRunRepository()
    service = ReleaseRunService(
        repository=repository,
        request_id="test-request-id",
    )

    command = StartReleaseRunCommand(
        query="What are the biggest release risks this week?",
        requested_by="manager@example.com",
    )

    result = await service.start_release_run(command)

    assert result.id is not None
    assert result.run_id.startswith("release-run-")
    assert result.query == "What are the biggest release risks this week?"
    assert result.requested_by == "manager@example.com"
    assert result.status == "created"
    assert result.completed_at is None


@pytest.mark.anyio
async def test_get_release_run_returns_result_when_found() -> None:
    """Service should return a release run when it exists."""
    repository = FakeReleaseRunRepository()
    service = ReleaseRunService(
        repository=repository,
        request_id="test-request-id",
    )

    command = StartReleaseRunCommand(
        query="Check release readiness",
        requested_by="manager@example.com",
    )

    created_result = await service.start_release_run(command)

    fetched_result = await service.get_release_run(created_result.id)

    assert fetched_result is not None
    assert fetched_result.id == created_result.id
    assert fetched_result.run_id == created_result.run_id


@pytest.mark.anyio
async def test_get_release_run_returns_none_when_missing() -> None:
    """Service should return None when release run does not exist."""
    repository = FakeReleaseRunRepository()
    service = ReleaseRunService(
        repository=repository,
        request_id="test-request-id",
    )

    result = await service.get_release_run(uuid4())

    assert result is None


@pytest.mark.anyio
async def test_mark_running_updates_status() -> None:
    """Service should mark a release run as running."""
    repository = FakeReleaseRunRepository()
    service = ReleaseRunService(
        repository=repository,
        request_id="test-request-id",
    )

    command = StartReleaseRunCommand(
        query="Check deployment risk",
        requested_by="manager@example.com",
    )

    created_result = await service.start_release_run(command)

    updated_result = await service.mark_running(created_result.id)

    assert updated_result is not None
    assert updated_result.status == "running"
    assert updated_result.completed_at is None


@pytest.mark.anyio
async def test_mark_completed_sets_completed_status() -> None:
    """Service should mark a release run as completed."""
    repository = FakeReleaseRunRepository()
    service = ReleaseRunService(
        repository=repository,
        request_id="test-request-id",
    )

    command = StartReleaseRunCommand(
        query="Check final release status",
        requested_by="manager@example.com",
    )

    created_result = await service.start_release_run(command)

    updated_result = await service.mark_completed(created_result.id)

    assert updated_result is not None
    assert updated_result.status == "completed"
    assert updated_result.completed_at is not None


@pytest.mark.anyio
async def test_mark_failed_sets_failed_status() -> None:
    """Service should mark a release run as failed."""
    repository = FakeReleaseRunRepository()
    service = ReleaseRunService(
        repository=repository,
        request_id="test-request-id",
    )

    command = StartReleaseRunCommand(
        query="Check failed release workflow",
        requested_by="manager@example.com",
    )

    created_result = await service.start_release_run(command)

    updated_result = await service.mark_failed(created_result.id)

    assert updated_result is not None
    assert updated_result.status == "failed"
    assert updated_result.completed_at is not None


@pytest.mark.anyio
async def test_mark_running_returns_none_when_release_run_missing() -> None:
    """Service should return None when updating a missing release run."""
    repository = FakeReleaseRunRepository()
    service = ReleaseRunService(
        repository=repository,
        request_id="test-request-id",
    )

    result = await service.mark_running(uuid4())

    assert result is None


@pytest.mark.anyio
async def test_start_release_run_raises_service_error_when_repository_fails() -> None:
    """Service should wrap repository create failures."""
    repository = FakeReleaseRunRepository(should_fail=True)
    service = ReleaseRunService(
        repository=repository,
        request_id="test-request-id",
    )

    command = StartReleaseRunCommand(
        query="What are the biggest release risks this week?",
        requested_by="manager@example.com",
    )

    with pytest.raises(
        ReleaseRunServiceError,
        match="Failed to start release-risk workflow.",
    ):
        await service.start_release_run(command)


@pytest.mark.anyio
async def test_get_release_run_raises_service_error_when_repository_fails() -> None:
    """Service should wrap repository lookup failures."""
    repository = FakeReleaseRunRepository(should_fail=True)
    service = ReleaseRunService(
        repository=repository,
        request_id="test-request-id",
    )

    with pytest.raises(
        ReleaseRunServiceError,
        match="Failed to fetch release run.",
    ):
        await service.get_release_run(uuid4())


@pytest.mark.anyio
async def test_mark_running_raises_service_error_when_repository_fails() -> None:
    """Service should wrap repository update failures."""
    repository = FakeReleaseRunRepository(should_fail=True)
    service = ReleaseRunService(
        repository=repository,
        request_id="test-request-id",
    )

    with pytest.raises(
        ReleaseRunServiceError,
        match="Failed to update release run status.",
    ):
        await service.mark_running(uuid4())


def test_start_release_run_command_rejects_invalid_input() -> None:
    """Command should reject invalid manager query input."""
    with pytest.raises(ValidationError):
        StartReleaseRunCommand(
            query="bad",
            requested_by="me",
        )