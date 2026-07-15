"""add release run approvals

Revision ID: 7c3f2a1b9e4d
Revises: 40ff991fa2a6
Create Date: 2026-07-09

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "7c3f2a1b9e4d"
down_revision: str | None = "40ff991fa2a6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create release-run approval request table."""
    op.create_table(
        "release_run_approvals",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("release_run_id", sa.Uuid(), nullable=False),
        sa.Column("approval_status", sa.String(length=32), nullable=False),
        sa.Column("approval_reason", sa.Text(), nullable=False),
        sa.Column("approval_policy_version", sa.String(length=100), nullable=False),
        sa.Column("requested_by", sa.String(length=255), nullable=True),
        sa.Column("decided_by", sa.String(length=255), nullable=True),
        sa.Column("decision_note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["release_run_id"],
            ["release_runs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "ix_release_run_approvals_release_run_id",
        "release_run_approvals",
        ["release_run_id"],
        unique=False,
    )
    op.create_index(
        "ix_release_run_approvals_approval_status",
        "release_run_approvals",
        ["approval_status"],
        unique=False,
    )
    op.create_index(
        "ix_release_run_approvals_created_at",
        "release_run_approvals",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        "ix_release_run_approvals_release_run_id_created_at",
        "release_run_approvals",
        ["release_run_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_release_run_approvals_release_run_id_status",
        "release_run_approvals",
        ["release_run_id", "approval_status"],
        unique=False,
    )


def downgrade() -> None:
    """Drop release-run approval request table."""
    op.drop_index(
        "ix_release_run_approvals_release_run_id_status",
        table_name="release_run_approvals",
    )
    op.drop_index(
        "ix_release_run_approvals_release_run_id_created_at",
        table_name="release_run_approvals",
    )
    op.drop_index(
        "ix_release_run_approvals_created_at",
        table_name="release_run_approvals",
    )
    op.drop_index(
        "ix_release_run_approvals_approval_status",
        table_name="release_run_approvals",
    )
    op.drop_index(
        "ix_release_run_approvals_release_run_id",
        table_name="release_run_approvals",
    )
    op.drop_table("release_run_approvals")
