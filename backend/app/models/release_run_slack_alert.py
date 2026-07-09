"""Database model for release-run Slack alert delivery records."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Uuid

from app.db.base import Base


class ReleaseRunSlackAlert(Base):
    """Database model representing one successful Slack alert for a release run.

    This table enforces idempotency for manual Slack alert sending. Audit events
    record what happened, while this table stores the durable business state
    that a release run already sent a Slack alert.
    """

    __tablename__ = "release_run_slack_alerts"

    __table_args__ = (
        UniqueConstraint(
            "release_run_id",
            name="uq_release_run_slack_alerts_release_run_id",
        ),
        Index(
            "ix_release_run_slack_alerts_release_run_id_created_at",
            "release_run_id",
            "created_at",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    release_run_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("release_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    approval_request_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        nullable=True,
        index=True,
    )

    snapshot_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        nullable=True,
        index=True,
    )

    snapshot_version: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    delivery_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="sent",
        index=True,
    )

    slack_channel: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    slack_timestamp: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    risk_level: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
    )

    risk_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    recommended_action: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        index=True,
    )
