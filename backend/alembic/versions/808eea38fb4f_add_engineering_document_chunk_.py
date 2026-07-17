"""add engineering document chunk embeddings

Revision ID: 808eea38fb4f
Revises: 5d0b4c2a7e91
Create Date: 2026-07-16 14:57:42.853229
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

revision: str = "808eea38fb4f"
down_revision: str | None = "5d0b4c2a7e91"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add nullable chunk embeddings with database-specific storage."""
    connection = op.get_bind()

    if connection.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
        embedding_type: sa.types.TypeEngine[object] = Vector(384)
    else:
        embedding_type = sa.JSON()

    op.add_column(
        "engineering_document_chunks",
        sa.Column(
            "embedding",
            embedding_type,
            nullable=True,
        ),
    )


def downgrade() -> None:
    """Remove engineering-document chunk embeddings."""
    op.drop_column(
        "engineering_document_chunks",
        "embedding",
    )
