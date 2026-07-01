import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.release_run import ReleaseRun

logger = logging.getLogger(__name__)


class ReleaseRunRepositoryError(RuntimeError):
    """Raised when release run database operations fail."""


class ReleaseRunRepository:
    """Repository for release run database operations."""

    def __init__(self, session: AsyncSession, request_id: str) -> None:
        """Initialize the repository.

        Args:
            session: Async SQLAlchemy database session.
            request_id: Request-level ID used for structured logging.
        """
        self._session = session
        self._request_id = request_id

    async def create(self, release_run: ReleaseRun) -> ReleaseRun:
        """Create a new release run.

        Args:
            release_run: ReleaseRun model instance to persist.

        Returns:
            Persisted ReleaseRun instance.

        Raises:
            ReleaseRunRepositoryError: If the database operation fails.
        """
        try:
            self._session.add(release_run)
            await self._session.flush()
            await self._session.refresh(release_run)

            logger.info(
                "release_run_created",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(release_run.id),
                    "release_run_run_id": release_run.run_id,
                },
            )

            return release_run

        except SQLAlchemyError as exc:
            logger.exception(
                "release_run_create_failed",
                extra={"request_id": self._request_id},
            )
            raise ReleaseRunRepositoryError(
                "Failed to create release run."
            ) from exc

    async def get_by_id(self, release_run_id: UUID) -> ReleaseRun | None:
        """Fetch a release run by database UUID.

        Args:
            release_run_id: Release run primary key.

        Returns:
            ReleaseRun if found, otherwise None.

        Raises:
            ReleaseRunRepositoryError: If the database operation fails.
        """
        try:
            statement = select(ReleaseRun).where(ReleaseRun.id == release_run_id)
            result = await self._session.execute(statement)
            release_run = result.scalar_one_or_none()

            logger.info(
                "release_run_fetched",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(release_run_id),
                    "found": release_run is not None,
                },
            )

            return release_run

        except SQLAlchemyError as exc:
            logger.exception(
                "release_run_fetch_failed",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(release_run_id),
                },
            )
            raise ReleaseRunRepositoryError(
                "Failed to fetch release run."
            ) from exc

    async def list_recent(self, limit: int = 20, offset: int = 0) -> Sequence[ReleaseRun]:
        """List recent release runs ordered by creation time.

        Args:
            limit: Maximum number of records to return.
            offset: Number of records to skip.

        Returns:
            Sequence of ReleaseRun records.

        Raises:
            ValueError: If limit or offset is invalid.
            ReleaseRunRepositoryError: If the database operation fails.
        """
        if limit <= 0:
            raise ValueError("limit must be greater than 0.")

        if offset < 0:
            raise ValueError("offset cannot be negative.")

        try:
            statement = (
                select(ReleaseRun)
                .order_by(ReleaseRun.created_at.desc())
                .limit(limit)
                .offset(offset)
            )

            result = await self._session.execute(statement)
            release_runs = result.scalars().all()

            logger.info(
                "recent_release_runs_listed",
                extra={
                    "request_id": self._request_id,
                    "limit": limit,
                    "offset": offset,
                    "count": len(release_runs),
                },
            )

            return release_runs

        except SQLAlchemyError as exc:
            logger.exception(
                "recent_release_runs_list_failed",
                extra={
                    "request_id": self._request_id,
                    "limit": limit,
                    "offset": offset,
                },
            )
            raise ReleaseRunRepositoryError(
                "Failed to list recent release runs."
            ) from exc

    async def update_status(
        self,
        release_run_id: UUID,
        status: str,
    ) -> ReleaseRun | None:
        """Update the status of a release run.

        Args:
            release_run_id: Release run primary key.
            status: New release run status.

        Returns:
            Updated ReleaseRun if found, otherwise None.

        Raises:
            ReleaseRunRepositoryError: If the database operation fails.
        """
        try:
            release_run = await self.get_by_id(release_run_id)

            if release_run is None:
                logger.warning(
                    "release_run_status_update_skipped_not_found",
                    extra={
                        "request_id": self._request_id,
                        "release_run_id": str(release_run_id),
                        "status": status,
                    },
                )
                return None

            release_run.status = status

            if status in {"completed", "failed", "cancelled"}:
                release_run.completed_at = datetime.now(UTC)

            await self._session.flush()
            await self._session.refresh(release_run)

            logger.info(
                "release_run_status_updated",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(release_run_id),
                    "status": status,
                },
            )

            return release_run

        except SQLAlchemyError as exc:
            logger.exception(
                "release_run_status_update_failed",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(release_run_id),
                    "status": status,
                },
            )
            raise ReleaseRunRepositoryError(
                "Failed to update release run status."
            ) from exc