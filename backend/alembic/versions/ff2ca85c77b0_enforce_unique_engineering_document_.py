"""Enforce one engineering document per authoritative source URI.

Revision ID: ff2ca85c77b0
Revises: 808eea38fb4f
Create Date: 2026-07-30 13:23:17.300786
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "ff2ca85c77b0"
down_revision: str | Sequence[str] | None = "808eea38fb4f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INDEX_NAME = "ix_engineering_documents_source_uri"
_TABLE_NAME = "engineering_documents"


def upgrade() -> None:
    """Replace the source URI index with a uniqueness-enforcing index."""
    connection = op.get_bind()

    duplicate_rows = connection.execute(
        sa.text(
            """
            SELECT source_uri
            FROM engineering_documents
            GROUP BY source_uri
            HAVING COUNT(*) > 1
            ORDER BY source_uri
            LIMIT 10
            """
        )
    ).all()

    if duplicate_rows:
        duplicate_source_uris = ", ".join(
            str(row[0]) for row in duplicate_rows
        )
        raise RuntimeError(
            "Cannot enforce unique engineering document source URIs because "
            f"duplicate values exist: {duplicate_source_uris}"
        )

    op.drop_index(
        _INDEX_NAME,
        table_name=_TABLE_NAME,
    )
    op.create_index(
        _INDEX_NAME,
        _TABLE_NAME,
        ["source_uri"],
        unique=True,
    )


def downgrade() -> None:
    """Restore the original non-unique source URI index."""
    op.drop_index(
        _INDEX_NAME,
        table_name=_TABLE_NAME,
    )
    op.create_index(
        _INDEX_NAME,
        _TABLE_NAME,
        ["source_uri"],
        unique=False,
    )
