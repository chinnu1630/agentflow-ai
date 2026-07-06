from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.release_run_event import ReleaseRunEvent

logger = logging.getLogger(__name__)


class CreateReleaseRunEventCommand(BaseModel):
    """Validated input for creating a release-run audit event."""

    model_config = ConfigDict(extra="forbid")

    release_run_id: UUID
    event_type: str = Field(min_length=1, max_length=100)
    event_status: str = Field(default="success", min_length=1, max_length=32)
    message: str = Field(min_length=1)
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class ReleaseRunEventRepositoryError(RuntimeError):
    """Raised when release-run event database operations fail."""


class ReleaseRunEventRepository:
    """Repository for release-run audit event database operations."""

    def __init__(self, session: AsyncSession, request_id: str) -> None:
        """Initialize the repository.

        Args:
            session: Async SQLAlchemy database session.
            request_id: Request-level ID used for structured logging.
        """

        self._session = session
        self._request_id = request_id

    async def create(
        self,
        command: CreateReleaseRunEventCommand,
    ) -> ReleaseRunEvent:
        """Create a new release-run audit event.

        Args:
            command: Validated command containing audit event data.

        Returns:
            Persisted ReleaseRunEvent instance.

        Raises:
            ReleaseRunEventRepositoryError: If the database operation fails.
        """

        try:
            event = ReleaseRunEvent(
                release_run_id=command.release_run_id,
                event_type=command.event_type,
                event_status=command.event_status,
                message=command.message,
                metadata_json=command.metadata_json,
            )

            self._session.add(event)
            await self._session.flush()
            await self._session.refresh(event)

            logger.info(
                "release_run_event_created",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(command.release_run_id),
                    "release_run_event_id": str(event.id),
                    "event_type": command.event_type,
                    "event_status": command.event_status,
                },
            )

            return event

        except SQLAlchemyError as exc:
            logger.exception(
                "release_run_event_create_failed",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(command.release_run_id),
                    "event_type": command.event_type,
                    "event_status": command.event_status,
                },
            )
            raise ReleaseRunEventRepositoryError(
                "Failed to create release-run event."
            ) from exc

    async def list_by_release_run_id(
        self,
        release_run_id: UUID,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[ReleaseRunEvent]:
        """List audit events for a release run ordered by creation time.

        Args:
            release_run_id: Release run primary key.
            limit: Maximum number of events to return.
            offset: Number of events to skip.

        Returns:
            Sequence of ReleaseRunEvent records.

        Raises:
            ValueError: If limit or offset is invalid.
            ReleaseRunEventRepositoryError: If the database operation fails.
        """

        if limit <= 0:
            raise ValueError("limit must be greater than 0.")

        if offset < 0:
            raise ValueError("offset cannot be negative.")

        try:
            statement = (
                select(ReleaseRunEvent)
                .where(ReleaseRunEvent.release_run_id == release_run_id)
                .order_by(ReleaseRunEvent.created_at.asc())
                .limit(limit)
                .offset(offset)
            )

            result = await self._session.execute(statement)
            events = result.scalars().all()

            logger.info(
                "release_run_events_listed",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(release_run_id),
                    "limit": limit,
                    "offset": offset,
                    "count": len(events),
                },
            )

            return events

        except SQLAlchemyError as exc:
            logger.exception(
                "release_run_events_list_failed",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(release_run_id),
                    "limit": limit,
                    "offset": offset,
                },
            )
            raise ReleaseRunEventRepositoryError(
                "Failed to list release-run events."
            ) from exc
