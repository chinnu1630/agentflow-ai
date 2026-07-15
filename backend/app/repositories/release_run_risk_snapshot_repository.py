"""Repository for persisted release-risk report snapshots."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.release_run_risk_snapshot import ReleaseRunRiskSnapshot

logger = logging.getLogger(__name__)


class CreateReleaseRunRiskSnapshotCommand(BaseModel):
    """Validated input for creating a release-risk report snapshot."""

    model_config = ConfigDict(extra="forbid")

    release_run_id: UUID
    risk_payload: dict[str, Any] = Field(min_length=1)
    overall_severity: str = Field(min_length=1, max_length=32)
    approval_required: bool
    approval_status_at_snapshot: str = Field(
        default="not_required",
        min_length=1,
        max_length=32,
    )

    @field_validator("overall_severity", "approval_status_at_snapshot")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """Normalize text fields and reject blank values."""
        stripped_value = value.strip()

        if not stripped_value:
            raise ValueError("value must not be blank")

        return stripped_value

    @field_validator("risk_payload")
    @classmethod
    def validate_json_serializable_payload(
        cls,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        """Ensure the risk payload can be safely stored as JSON."""
        try:
            json.dumps(value, sort_keys=True)
        except TypeError as exc:
            raise ValueError("risk_payload must be JSON serializable") from exc

        return value


class ReleaseRunRiskSnapshotRepositoryError(RuntimeError):
    """Raised when release-risk snapshot database operations fail."""


class ReleaseRunRiskSnapshotRepository:
    """Repository for release-risk snapshot persistence."""

    def __init__(self, session: AsyncSession, request_id: str) -> None:
        """Initialize the repository.

        Args:
            session: Async SQLAlchemy database session.
            request_id: Request-level ID used for structured logging.
        """
        self._session = session
        self._request_id = request_id

    async def create_snapshot(
        self,
        command: CreateReleaseRunRiskSnapshotCommand,
    ) -> ReleaseRunRiskSnapshot:
        """Create a new release-risk snapshot with the next version number.

        Args:
            command: Validated snapshot creation command.

        Returns:
            Persisted release-risk snapshot.

        Raises:
            ReleaseRunRiskSnapshotRepositoryError: If the database operation fails.
        """
        try:
            latest_version_statement = select(
                func.max(ReleaseRunRiskSnapshot.snapshot_version)
            ).where(
                ReleaseRunRiskSnapshot.release_run_id == command.release_run_id
            )

            latest_version_result = await self._session.execute(
                latest_version_statement
            )
            latest_version = latest_version_result.scalar_one_or_none()
            next_version = (latest_version or 0) + 1

            snapshot = ReleaseRunRiskSnapshot(
                release_run_id=command.release_run_id,
                snapshot_version=next_version,
                risk_payload_json=json.dumps(command.risk_payload, sort_keys=True),
                overall_severity=command.overall_severity,
                approval_required=command.approval_required,
                approval_status_at_snapshot=command.approval_status_at_snapshot,
            )

            self._session.add(snapshot)
            await self._session.flush()
            await self._session.refresh(snapshot)

            logger.info(
                "release_run_risk_snapshot_created",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(command.release_run_id),
                    "snapshot_id": str(snapshot.id),
                    "snapshot_version": snapshot.snapshot_version,
                    "overall_severity": snapshot.overall_severity,
                    "approval_required": snapshot.approval_required,
                    "approval_status_at_snapshot": snapshot.approval_status_at_snapshot,
                },
            )

            return snapshot

        except SQLAlchemyError as exc:
            logger.exception(
                "release_run_risk_snapshot_create_failed",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(command.release_run_id),
                },
            )
            raise ReleaseRunRiskSnapshotRepositoryError(
                "Failed to create release-risk snapshot."
            ) from exc

    async def get_latest_by_release_run_id(
        self,
        release_run_id: UUID,
    ) -> ReleaseRunRiskSnapshot | None:
        """Fetch the latest snapshot for one release run."""
        try:
            statement = (
                select(ReleaseRunRiskSnapshot)
                .where(ReleaseRunRiskSnapshot.release_run_id == release_run_id)
                .order_by(ReleaseRunRiskSnapshot.snapshot_version.desc())
                .limit(1)
            )

            result = await self._session.execute(statement)
            snapshot = result.scalar_one_or_none()

            logger.info(
                "release_run_latest_risk_snapshot_fetched",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(release_run_id),
                    "found": snapshot is not None,
                },
            )

            return snapshot

        except SQLAlchemyError as exc:
            logger.exception(
                "release_run_latest_risk_snapshot_fetch_failed",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(release_run_id),
                },
            )
            raise ReleaseRunRiskSnapshotRepositoryError(
                "Failed to fetch latest release-risk snapshot."
            ) from exc

    async def list_latest_previous_release_snapshots(
        self,
        *,
        exclude_release_run_id: UUID,
        limit: int = 10,
    ) -> Sequence[ReleaseRunRiskSnapshot]:
        """List the latest snapshot from each previous release run.

        Args:
            exclude_release_run_id: Current release run to exclude.
            limit: Maximum number of previous release snapshots to return.

        Returns:
            Latest snapshot per previous release run, newest first.

        Raises:
            ValueError: If limit is outside the supported range.
            ReleaseRunRiskSnapshotRepositoryError: If the query fails.
        """
        if limit <= 0:
            raise ValueError("limit must be greater than 0.")

        if limit > 100:
            raise ValueError("limit cannot exceed 100.")

        try:
            ranked_snapshots = (
                select(
                    ReleaseRunRiskSnapshot.id.label("snapshot_id"),
                    func.row_number()
                    .over(
                        partition_by=ReleaseRunRiskSnapshot.release_run_id,
                        order_by=ReleaseRunRiskSnapshot.snapshot_version.desc(),
                    )
                    .label("snapshot_rank"),
                )
                .where(
                    ReleaseRunRiskSnapshot.release_run_id
                    != exclude_release_run_id
                )
                .subquery()
            )

            statement = (
                select(ReleaseRunRiskSnapshot)
                .join(
                    ranked_snapshots,
                    ReleaseRunRiskSnapshot.id
                    == ranked_snapshots.c.snapshot_id,
                )
                .where(ranked_snapshots.c.snapshot_rank == 1)
                .order_by(
                    ReleaseRunRiskSnapshot.created_at.desc(),
                    ReleaseRunRiskSnapshot.release_run_id.asc(),
                )
                .limit(limit)
            )

            result = await self._session.execute(statement)
            snapshots = result.scalars().all()

            logger.info(
                "latest_previous_release_risk_snapshots_listed",
                extra={
                    "request_id": self._request_id,
                    "exclude_release_run_id": str(exclude_release_run_id),
                    "limit": limit,
                    "count": len(snapshots),
                },
            )

            return snapshots

        except SQLAlchemyError as exc:
            logger.exception(
                "latest_previous_release_risk_snapshots_list_failed",
                extra={
                    "request_id": self._request_id,
                    "exclude_release_run_id": str(exclude_release_run_id),
                    "limit": limit,
                },
            )
            raise ReleaseRunRiskSnapshotRepositoryError(
                "Failed to list previous release-risk snapshots."
            ) from exc

    async def list_by_release_run_id(
        self,
        release_run_id: UUID,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[ReleaseRunRiskSnapshot]:
        """List snapshots for one release run ordered by version ascending."""
        if limit <= 0:
            raise ValueError("limit must be greater than 0.")

        if offset < 0:
            raise ValueError("offset cannot be negative.")

        try:
            statement = (
                select(ReleaseRunRiskSnapshot)
                .where(ReleaseRunRiskSnapshot.release_run_id == release_run_id)
                .order_by(ReleaseRunRiskSnapshot.snapshot_version.asc())
                .limit(limit)
                .offset(offset)
            )

            result = await self._session.execute(statement)
            snapshots = result.scalars().all()

            logger.info(
                "release_run_risk_snapshots_listed",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(release_run_id),
                    "limit": limit,
                    "offset": offset,
                    "count": len(snapshots),
                },
            )

            return snapshots

        except SQLAlchemyError as exc:
            logger.exception(
                "release_run_risk_snapshots_list_failed",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(release_run_id),
                    "limit": limit,
                    "offset": offset,
                },
            )
            raise ReleaseRunRiskSnapshotRepositoryError(
                "Failed to list release-risk snapshots."
            ) from exc
