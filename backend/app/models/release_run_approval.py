"""Database model for HITL approval requests tied to release runs."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Uuid

from app.db.base import Base


class ReleaseRunApproval(Base):
    """Database model representing one human approval request for a release run.

    This table stores durable HITL approval state. It allows the backend to
    remember that a release requires approval even if the API server restarts
    before a manager reviews the request.
    """

    __tablename__ = "release_run_approvals"

    __table_args__ = (
        Index(
            "ix_release_run_approvals_release_run_id_created_at",
            "release_run_id",
            "created_at",
        ),
        Index(
            "ix_release_run_approvals_release_run_id_status",
            "release_run_id",
            "approval_status",
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

    approval_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="pending",
        index=True,
    )

    approval_reason: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    approval_policy_version: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    requested_by: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    decided_by: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    decision_note: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        index=True,
    )

    decided_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
