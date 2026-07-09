"""Repository for release-run Slack alert idempotency records."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.release_run_slack_alert import ReleaseRunSlackAlert

logger = logging.getLogger(__name__)


class CreateReleaseRunSlackAlertCommand(BaseModel):
    """Validated input for recording a successful Slack alert."""

    model_config = ConfigDict(extra="forbid")

    release_run_id: UUID
    approval_request_id: UUID | None = None
    snapshot_id: UUID | None = None
    snapshot_version: int | None = Field(default=None, ge=1)
    slack_channel: str = Field(min_length=1, max_length=255)
    slack_timestamp: str = Field(min_length=1, max_length=100)
    risk_level: str = Field(min_length=1, max_length=32)
    risk_score: float = Field(ge=0.0, le=1.0)
    recommended_action: str = Field(min_length=1, max_length=100)
    delivery_status: str = Field(default="sent", min_length=1, max_length=32)

    @field_validator(
        "slack_channel",
        "slack_timestamp",
        "risk_level",
        "recommended_action",
        "delivery_status",
    )
    @classmethod
    def validate_text(cls, value: str) -> str:
        """Normalize text fields and reject blank values."""
        stripped_value = value.strip()

        if not stripped_value:
            raise ValueError("value must not be blank")

        return stripped_value


class ReleaseRunSlackAlertRepositoryError(RuntimeError):
    """Raised when release-run Slack alert database operations fail."""


class ReleaseRunSlackAlertAlreadySentError(RuntimeError):
    """Raised when a Slack alert already exists for a release run."""


class ReleaseRunSlackAlertRepository:
    """Repository for release-run Slack alert idempotency persistence."""

    def __init__(self, session: AsyncSession, request_id: str) -> None:
        """Initialize the repository.

        Args:
            session: Async SQLAlchemy database session.
            request_id: Request-level ID used for structured logging.
        """
        self._session = session
        self._request_id = request_id

    async def create_sent_alert(
        self,
        command: CreateReleaseRunSlackAlertCommand,
    ) -> ReleaseRunSlackAlert:
        """Persist a successful Slack alert if one does not already exist.

        Args:
            command: Validated successful Slack alert record.

        Returns:
            Persisted Slack alert record.

        Raises:
            ReleaseRunSlackAlertAlreadySentError: If release already sent Slack alert.
            ReleaseRunSlackAlertRepositoryError: If database operation fails.
        """
        try:
            existing_alert = await self.get_by_release_run_id(command.release_run_id)

            if existing_alert is not None:
                raise ReleaseRunSlackAlertAlreadySentError(
                    "Slack alert already sent for this release run."
                )

            alert = ReleaseRunSlackAlert(
                release_run_id=command.release_run_id,
                approval_request_id=command.approval_request_id,
                snapshot_id=command.snapshot_id,
                snapshot_version=command.snapshot_version,
                delivery_status=command.delivery_status,
                slack_channel=command.slack_channel,
                slack_timestamp=command.slack_timestamp,
                risk_level=command.risk_level,
                risk_score=command.risk_score,
                recommended_action=command.recommended_action,
            )

            self._session.add(alert)
            await self._session.flush()
            await self._session.refresh(alert)

            logger.info(
                "release_run_slack_alert_recorded",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(command.release_run_id),
                    "slack_alert_id": str(alert.id),
                    "delivery_status": alert.delivery_status,
                    "slack_channel": alert.slack_channel,
                    "risk_level": alert.risk_level,
                },
            )

            return alert

        except ReleaseRunSlackAlertAlreadySentError:
            logger.info(
                "release_run_slack_alert_duplicate_blocked",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(command.release_run_id),
                },
            )
            raise

        except IntegrityError as exc:
            logger.warning(
                "release_run_slack_alert_integrity_duplicate",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(command.release_run_id),
                },
            )
            raise ReleaseRunSlackAlertAlreadySentError(
                "Slack alert already sent for this release run."
            ) from exc

        except SQLAlchemyError as exc:
            logger.exception(
                "release_run_slack_alert_create_failed",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(command.release_run_id),
                },
            )
            raise ReleaseRunSlackAlertRepositoryError(
                "Failed to record release-run Slack alert."
            ) from exc

    async def get_by_release_run_id(
        self,
        release_run_id: UUID,
    ) -> ReleaseRunSlackAlert | None:
        """Fetch the Slack alert record for one release run."""
        try:
            statement = (
                select(ReleaseRunSlackAlert)
                .where(ReleaseRunSlackAlert.release_run_id == release_run_id)
                .limit(1)
            )

            result = await self._session.execute(statement)
            alert = result.scalar_one_or_none()

            logger.info(
                "release_run_slack_alert_fetched",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(release_run_id),
                    "found": alert is not None,
                },
            )

            return alert

        except SQLAlchemyError as exc:
            logger.exception(
                "release_run_slack_alert_fetch_failed",
                extra={
                    "request_id": self._request_id,
                    "release_run_id": str(release_run_id),
                },
            )
            raise ReleaseRunSlackAlertRepositoryError(
                "Failed to fetch release-run Slack alert."
            ) from exc

    async def list_by_status(
        self,
        delivery_status: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[ReleaseRunSlackAlert]:
        """List Slack alert records by delivery status."""
        if limit <= 0:
            raise ValueError("limit must be greater than 0.")

        if offset < 0:
            raise ValueError("offset cannot be negative.")

        normalized_status = delivery_status.strip()

        if not normalized_status:
            raise ValueError("delivery_status must not be blank.")

        try:
            statement = (
                select(ReleaseRunSlackAlert)
                .where(ReleaseRunSlackAlert.delivery_status == normalized_status)
                .order_by(ReleaseRunSlackAlert.created_at.asc())
                .limit(limit)
                .offset(offset)
            )

            result = await self._session.execute(statement)
            alerts = result.scalars().all()

            logger.info(
                "release_run_slack_alerts_listed",
                extra={
                    "request_id": self._request_id,
                    "delivery_status": normalized_status,
                    "limit": limit,
                    "offset": offset,
                    "count": len(alerts),
                },
            )

            return alerts

        except SQLAlchemyError as exc:
            logger.exception(
                "release_run_slack_alerts_list_failed",
                extra={
                    "request_id": self._request_id,
                    "delivery_status": normalized_status,
                    "limit": limit,
                    "offset": offset,
                },
            )
            raise ReleaseRunSlackAlertRepositoryError(
                "Failed to list release-run Slack alerts."
            ) from exc
