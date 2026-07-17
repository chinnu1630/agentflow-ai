"""Pydantic schemas for engineering knowledge-base documents.

These schemas validate data before it reaches the repository layer. They are
used for the Knowledge Agent / RAG storage foundation and will later support
document ingestion, chunking, retrieval, and citations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.models.engineering_document import EngineeringDocumentSourceType


class EngineeringDocumentCreate(BaseModel):
    """Input schema for creating an engineering document."""

    title: str = Field(min_length=1, max_length=255)
    source_type: EngineeringDocumentSourceType
    source_uri: str = Field(min_length=1, max_length=1024)
    content_hash: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[a-fA-F0-9]{64}$",
    )
    raw_content: str = Field(min_length=1)
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class EngineeringDocumentChunkCreate(BaseModel):
    """Input schema for creating a chunk of an engineering document."""

    document_id: UUID
    chunk_index: int = Field(ge=0)
    content: str = Field(min_length=1)
    token_count: int = Field(default=0, ge=0)
    embedding: list[float] | None = None
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class EngineeringDocumentRead(BaseModel):
    """Read schema for an engineering document."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    source_type: EngineeringDocumentSourceType
    source_uri: str
    content_hash: str
    raw_content: str
    metadata_json: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class EngineeringDocumentChunkRead(BaseModel):
    """Read schema for an engineering document chunk."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    document_id: UUID
    chunk_index: int
    content: str
    token_count: int
    metadata_json: dict[str, Any]
    created_at: datetime
