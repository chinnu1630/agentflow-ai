"""add release run risk snapshots

Revision ID: dbb3d55a9914
Revises: 7c3f2a1b9e4d
Create Date: 2026-07-09
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "dbb3d55a9914"
down_revision: str | None = "7c3f2a1b9e4d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create release-risk snapshot table."""
    op.create_table(
        "release_run_risk_snapshots",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("release_run_id", sa.Uuid(), nullable=False),
        sa.Column("snapshot_version", sa.Integer(), nullable=False),
        sa.Column("risk_payload_json", sa.Text(), nullable=False),
        sa.Column("overall_severity", sa.String(length=32), nullable=False),
        sa.Column("approval_required", sa.Boolean(), nullable=False),
        sa.Column("approval_status_at_snapshot", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["release_run_id"],
            ["release_runs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "release_run_id",
            "snapshot_version",
            name="uq_release_run_risk_snapshots_release_run_id_snapshot_version",
        ),
    )

    op.create_index(
        op.f("ix_release_run_risk_snapshots_release_run_id"),
        "release_run_risk_snapshots",
        ["release_run_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_release_run_risk_snapshots_overall_severity"),
        "release_run_risk_snapshots",
        ["overall_severity"],
        unique=False,
    )
    op.create_index(
        op.f("ix_release_run_risk_snapshots_approval_required"),
        "release_run_risk_snapshots",
        ["approval_required"],
        unique=False,
    )
    op.create_index(
        op.f("ix_release_run_risk_snapshots_approval_status_at_snapshot"),
        "release_run_risk_snapshots",
        ["approval_status_at_snapshot"],
        unique=False,
    )
    op.create_index(
        op.f("ix_release_run_risk_snapshots_created_at"),
        "release_run_risk_snapshots",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        "ix_release_run_risk_snapshots_release_run_id_created_at",
        "release_run_risk_snapshots",
        ["release_run_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_release_run_risk_snapshots_release_run_id_snapshot_version",
        "release_run_risk_snapshots",
        ["release_run_id", "snapshot_version"],
        unique=False,
    )


def downgrade() -> None:
    """Drop release-risk snapshot table."""
    op.drop_index(
        "ix_release_run_risk_snapshots_release_run_id_snapshot_version",
        table_name="release_run_risk_snapshots",
    )
    op.drop_index(
        "ix_release_run_risk_snapshots_release_run_id_created_at",
        table_name="release_run_risk_snapshots",
    )
    op.drop_index(
        op.f("ix_release_run_risk_snapshots_created_at"),
        table_name="release_run_risk_snapshots",
    )
    op.drop_index(
        op.f("ix_release_run_risk_snapshots_approval_status_at_snapshot"),
        table_name="release_run_risk_snapshots",
    )
    op.drop_index(
        op.f("ix_release_run_risk_snapshots_approval_required"),
        table_name="release_run_risk_snapshots",
    )
    op.drop_index(
        op.f("ix_release_run_risk_snapshots_overall_severity"),
        table_name="release_run_risk_snapshots",
    )
    op.drop_index(
        op.f("ix_release_run_risk_snapshots_release_run_id"),
        table_name="release_run_risk_snapshots",
    )
    op.drop_table("release_run_risk_snapshots")
