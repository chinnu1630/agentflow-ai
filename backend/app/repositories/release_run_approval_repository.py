"""Repository for HITL release-run approval requests."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.release_run_approval import ReleaseRunApproval

logger = logging.getLogger(__name__)


class ReleaseRunApprovalStatus(StrEnum):
    """Allowed lifecycle states for a release-run approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class CreateReleaseRunApprovalCommand(BaseModel):
    """Validated input for creating a pending release approval request."""

    model_config = ConfigDict(extra="forbid")

    release_run_id: UUID
    approval_reason: str = Field(min_length=1, max_length=1_000)
    approval_policy_version: str = Field(min_length=1, max_length=100)
    requested_by: str | None = Field(default=None, max_length=255)

    @field_validator("approval_reason", "approval_policy_version", "requested_by")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        """Normalize optional text and reject blank provided values."""
        if value is None:
            return None

        stripped_value = value.strip()

        if not stripped_value:
            raise ValueError("value must not be blank")

        return stripped_value


class DecideReleaseRunApprovalCommand(BaseModel):
    """Validated input for approving or rejecting a release approval request."""

    model_config = ConfigDict(extra="forbid")

    approval_id: UUID
    approval_status: ReleaseRunApprovalStatus
    decided_by: str = Field(min_length=1, max_length=255)
    decision_note: str | None = Field(default=None, max_length=1_000)

    @field_validator("approval_status")
    @classmethod
    def validate_terminal_status(
        cls,
        value: ReleaseRunApprovalStatus,
    ) -> ReleaseRunApprovalStatus:
        """Only approved/rejected decisions are valid terminal decisions."""
        if value == ReleaseRunApprovalStatus.PENDING:
            raise ValueError("approval_status must be approved or rejected")

        return value

    @field_validator("decided_by", "decision_note")
    @classmethod
    def validate_text(cls, value: str | None) -> str | None:
        """Normalize decision text and reject blank provided values."""
        if value is None:
            return None

        stripped_value = value.strip()

        if not stripped_value:
            raise ValueError("value must not be blank")

        return stripped_value


class ReleaseRunApprovalRepositoryError(RuntimeError):
    """Raised when release-run approval database operations fail."""


class ReleaseRunApprovalRepository:
    """Repository for release-run HITL approval persistence."""

    def __init__(self, session: AsyncSession, request_id: str) -> None:
        """Initialize the repository.

        Args:
            session: Async SQLAlchemy database session.
            request_id: Request-level ID used for structured logging.
        """
        self._session = session
        self._request_id = request_id

    async def create_pending(
        self,
        command: CreateReleaseRunApprovalCommand,
    ) -> ReleaseRunApproval:
        """Create a pending approval request.

        Args:
            command: Validated approval creation command.

        Returns:
            Persisted pending approval request.

        Raises:
            ReleaseRunApprovalRepositoryError: If the database operation fails.
        """
        try:
            approval = ReleaseRunApproval(
                release_run_id=command.release_run_id,
                approval_status=ReleaseRunApprovalStatus.PENDING.value,
                approval_reason=command.approval_reason,
                approval_policy_version=command.approval_policy_version,
                requested_by=command.requested_by,
            )

            self._session.add(approval)
            await self._session.flush()
            await self._session.refresh(approval)

            logger.info(
                "release_run_approval_created",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(command.release_run_id),
                    "approval_id": str(approval.id),
                    "approval_status": approval.approval_status,
                    "approval_policy_version": approval.approval_policy_version,
                },
            )

            return approval

        except SQLAlchemyError as exc:
            logger.exception(
                "release_run_approval_create_failed",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(command.release_run_id),
                },
            )
            raise ReleaseRunApprovalRepositoryError(
                "Failed to create release-run approval request."
            ) from exc

    async def get_by_id(self, approval_id: UUID) -> ReleaseRunApproval | None:
        """Fetch one approval request by ID."""
        try:
            approval = await self._session.get(ReleaseRunApproval, approval_id)

            logger.info(
                "release_run_approval_fetched",
                extra={
                    "request_id": self._request_id,
                    "approval_id": str(approval_id),
                    "found": approval is not None,
                },
            )

            return approval

        except SQLAlchemyError as exc:
            logger.exception(
                "release_run_approval_fetch_failed",
                extra={
                    "request_id": self._request_id,
                    "approval_id": str(approval_id),
                },
            )
            raise ReleaseRunApprovalRepositoryError(
                "Failed to fetch release-run approval request."
            ) from exc

    async def get_latest_by_release_run_id(
        self,
        release_run_id: UUID,
    ) -> ReleaseRunApproval | None:
        """Fetch latest approval request for one release run."""
        try:
            statement = (
                select(ReleaseRunApproval)
                .where(ReleaseRunApproval.release_run_id == release_run_id)
                .order_by(ReleaseRunApproval.created_at.desc())
                .limit(1)
            )

            result = await self._session.execute(statement)
            approval = result.scalar_one_or_none()

            logger.info(
                "release_run_latest_approval_fetched",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(release_run_id),
                    "found": approval is not None,
                },
            )

            return approval

        except SQLAlchemyError as exc:
            logger.exception(
                "release_run_latest_approval_fetch_failed",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(release_run_id),
                },
            )
            raise ReleaseRunApprovalRepositoryError(
                "Failed to fetch latest release-run approval request."
            ) from exc

    async def list_by_release_run_id(
        self,
        release_run_id: UUID,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[ReleaseRunApproval]:
        """List approval requests for a release run ordered by creation time."""
        if limit <= 0:
            raise ValueError("limit must be greater than 0.")

        if offset < 0:
            raise ValueError("offset cannot be negative.")

        try:
            statement = (
                select(ReleaseRunApproval)
                .where(ReleaseRunApproval.release_run_id == release_run_id)
                .order_by(ReleaseRunApproval.created_at.asc())
                .limit(limit)
                .offset(offset)
            )

            result = await self._session.execute(statement)
            approvals = result.scalars().all()

            logger.info(
                "release_run_approvals_listed",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(release_run_id),
                    "limit": limit,
                    "offset": offset,
                    "count": len(approvals),
                },
            )

            return approvals

        except SQLAlchemyError as exc:
            logger.exception(
                "release_run_approvals_list_failed",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(release_run_id),
                    "limit": limit,
                    "offset": offset,
                },
            )
            raise ReleaseRunApprovalRepositoryError(
                "Failed to list release-run approval requests."
            ) from exc


    async def list_by_status(
        self,
        approval_status: ReleaseRunApprovalStatus,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[ReleaseRunApproval]:
        """List approval requests by lifecycle status.

        This is used by manager dashboards to show pending approvals without
        requiring the UI to already know individual release_run_id values.
        """
        if limit <= 0:
            raise ValueError("limit must be greater than 0.")

        if offset < 0:
            raise ValueError("offset cannot be negative.")

        try:
            statement = (
                select(ReleaseRunApproval)
                .where(
                    ReleaseRunApproval.approval_status
                    == approval_status.value
                )
                .order_by(ReleaseRunApproval.created_at.asc())
                .limit(limit)
                .offset(offset)
            )

            result = await self._session.execute(statement)
            approvals = result.scalars().all()

            logger.info(
                "release_run_approvals_listed_by_status",
                extra={
                    "request_id": self._request_id,
                    "approval_status": approval_status.value,
                    "limit": limit,
                    "offset": offset,
                    "count": len(approvals),
                },
            )

            return approvals

        except SQLAlchemyError as exc:
            logger.exception(
                "release_run_approvals_list_by_status_failed",
                extra={
                    "request_id": self._request_id,
                    "approval_status": approval_status.value,
                    "limit": limit,
                    "offset": offset,
                },
            )
            raise ReleaseRunApprovalRepositoryError(
                "Failed to list release-run approvals by status."
            ) from exc

    async def decide(
        self,
        command: DecideReleaseRunApprovalCommand,
    ) -> ReleaseRunApproval | None:
        """Approve or reject a pending approval request.

        Returns None when the approval request does not exist.
        """
        try:
            approval = await self._session.get(
                ReleaseRunApproval,
                command.approval_id,
            )

            if approval is None:
                return None

            if approval.approval_status != ReleaseRunApprovalStatus.PENDING.value:
                raise ValueError("Only pending approval requests can be decided.")

            approval.approval_status = command.approval_status.value
            approval.decided_by = command.decided_by
            approval.decision_note = command.decision_note
            approval.decided_at = datetime.now(UTC)

            await self._session.flush()
            await self._session.refresh(approval)

            logger.info(
                "release_run_approval_decided",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(approval.release_run_id),
                    "approval_id": str(approval.id),
                    "approval_status": approval.approval_status,
                },
            )

            return approval

        except SQLAlchemyError as exc:
            logger.exception(
                "release_run_approval_decide_failed",
                extra={
                    "request_id": self._request_id,
                    "approval_id": str(command.approval_id),
                    "approval_status": command.approval_status.value,
                },
            )
            raise ReleaseRunApprovalRepositoryError(
                "Failed to decide release-run approval request."
            ) from exc
