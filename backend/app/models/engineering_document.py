"""SQLAlchemy model for engineering knowledge-base documents.

Engineering documents are the source-of-truth records used by the future
Knowledge Agent / RAG pipeline. Each document can later be split into smaller
chunks for retrieval, embeddings, citations, and release-risk evidence.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, Enum, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.engineering_document_chunk import EngineeringDocumentChunk


class EngineeringDocumentSourceType(StrEnum):
    """Supported source categories for engineering knowledge documents."""

    RUNBOOK = "runbook"
    RELEASE_CHECKLIST = "release_checklist"
    INCIDENT_POSTMORTEM = "incident_postmortem"
    ARCHITECTURE_DOC = "architecture_doc"
    OTHER = "other"


class EngineeringDocument(Base):
    """Persisted engineering document used as a RAG source.

    A document is the parent record for one or more document chunks.
    The full document metadata is stored here so downstream RAG answers can
    cite the original source reliably.
    """

    __tablename__ = "engineering_documents"

    id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    source_type: Mapped[EngineeringDocumentSourceType] = mapped_column(
        Enum(
            EngineeringDocumentSourceType,
            name="engineering_document_source_type",
            values_callable=lambda enum_class: [member.value for member in enum_class],
        ),
        nullable=False,
    )
    source_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    raw_content: Mapped[str] = mapped_column(Text, nullable=False)
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
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    chunks: Mapped[list[EngineeringDocumentChunk]] = relationship(
        "EngineeringDocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="EngineeringDocumentChunk.chunk_index",
    )


Index(
    "ix_engineering_documents_source_type",
    EngineeringDocument.source_type,
)

Index(
    "ix_engineering_documents_source_uri",
    EngineeringDocument.source_uri,
)
