"""add release run slack alerts

Revision ID: 5d0b4c2a7e91
Revises: dbb3d55a9914
Create Date: 2026-07-09
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "5d0b4c2a7e91"
down_revision: str | None = "dbb3d55a9914"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create release-run Slack alert idempotency table."""
    op.create_table(
        "release_run_slack_alerts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("release_run_id", sa.Uuid(), nullable=False),
        sa.Column("approval_request_id", sa.Uuid(), nullable=True),
        sa.Column("snapshot_id", sa.Uuid(), nullable=True),
        sa.Column("snapshot_version", sa.Integer(), nullable=True),
        sa.Column("delivery_status", sa.String(length=32), nullable=False),
        sa.Column("slack_channel", sa.String(length=255), nullable=False),
        sa.Column("slack_timestamp", sa.String(length=100), nullable=False),
        sa.Column("risk_level", sa.String(length=32), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("recommended_action", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["release_run_id"],
            ["release_runs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "release_run_id",
            name="uq_release_run_slack_alerts_release_run_id",
        ),
    )

    op.create_index(
        op.f("ix_release_run_slack_alerts_release_run_id"),
        "release_run_slack_alerts",
        ["release_run_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_release_run_slack_alerts_approval_request_id"),
        "release_run_slack_alerts",
        ["approval_request_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_release_run_slack_alerts_snapshot_id"),
        "release_run_slack_alerts",
        ["snapshot_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_release_run_slack_alerts_delivery_status"),
        "release_run_slack_alerts",
        ["delivery_status"],
        unique=False,
    )
    op.create_index(
        op.f("ix_release_run_slack_alerts_risk_level"),
        "release_run_slack_alerts",
        ["risk_level"],
        unique=False,
    )
    op.create_index(
        op.f("ix_release_run_slack_alerts_created_at"),
        "release_run_slack_alerts",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        "ix_release_run_slack_alerts_release_run_id_created_at",
        "release_run_slack_alerts",
        ["release_run_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    """Drop release-run Slack alert idempotency table."""
    op.drop_index(
        "ix_release_run_slack_alerts_release_run_id_created_at",
        table_name="release_run_slack_alerts",
    )
    op.drop_index(
        op.f("ix_release_run_slack_alerts_created_at"),
        table_name="release_run_slack_alerts",
    )
    op.drop_index(
        op.f("ix_release_run_slack_alerts_risk_level"),
        table_name="release_run_slack_alerts",
    )
    op.drop_index(
        op.f("ix_release_run_slack_alerts_delivery_status"),
        table_name="release_run_slack_alerts",
    )
    op.drop_index(
        op.f("ix_release_run_slack_alerts_snapshot_id"),
        table_name="release_run_slack_alerts",
    )
    op.drop_index(
        op.f("ix_release_run_slack_alerts_approval_request_id"),
        table_name="release_run_slack_alerts",
    )
    op.drop_index(
        op.f("ix_release_run_slack_alerts_release_run_id"),
        table_name="release_run_slack_alerts",
    )
    op.drop_table("release_run_slack_alerts")
