"""SQLAlchemy model for chunks of engineering knowledge documents.

Document chunks are the retrieval units used by the future Knowledge Agent.
Each chunk belongs to exactly one EngineeringDocument and preserves its order
inside the original source document.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.engineering_document import EngineeringDocument


class EngineeringDocumentChunk(Base):
    """Persisted chunk of an engineering document.

    A chunk is a smaller section of a larger engineering document. Future RAG
    retrieval, embeddings, reranking, and citations will operate at this level.
    """

    __tablename__ = "engineering_document_chunks"

    __table_args__ = (
        CheckConstraint("chunk_index >= 0", name="ck_document_chunk_index_non_negative"),
        CheckConstraint("token_count >= 0", name="ck_document_chunk_token_count_non_negative"),
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_document_chunk_document_id_chunk_index",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    document_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("engineering_documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metadata_json: Mapped[dict[str, object]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"),
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    document: Mapped[EngineeringDocument] = relationship(
        "EngineeringDocument",
        back_populates="chunks",
    )


Index(
    "ix_engineering_document_chunks_document_id",
    EngineeringDocumentChunk.document_id,
)

Index(
    "ix_engineering_document_chunks_document_id_chunk_index",
    EngineeringDocumentChunk.document_id,
    EngineeringDocumentChunk.chunk_index,
)
