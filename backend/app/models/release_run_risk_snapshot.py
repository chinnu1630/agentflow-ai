"""Database model for persisted release-risk report snapshots."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Uuid

from app.db.base import Base


class ReleaseRunRiskSnapshot(Base):
    """Database model representing one backend-generated release-risk report snapshot.

    A snapshot stores the trusted risk report produced by the backend workflow.
    Slack alerts should later read from this table instead of accepting client-
    supplied risk payloads.
    """

    __tablename__ = "release_run_risk_snapshots"

    __table_args__ = (
        UniqueConstraint(
            "release_run_id",
            "snapshot_version",
            name="uq_release_run_risk_snapshots_release_run_id_snapshot_version",
        ),
        Index(
            "ix_release_run_risk_snapshots_release_run_id_created_at",
            "release_run_id",
            "created_at",
        ),
        Index(
            "ix_release_run_risk_snapshots_release_run_id_snapshot_version",
            "release_run_id",
            "snapshot_version",
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

    snapshot_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    risk_payload_json: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    overall_severity: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
    )

    approval_required: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    approval_status_at_snapshot: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="not_required",
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        index=True,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
