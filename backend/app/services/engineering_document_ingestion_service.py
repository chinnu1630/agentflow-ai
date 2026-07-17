"""Service for ingesting engineering documents into the Knowledge Agent store."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any, Protocol
from uuid import UUID

import structlog
from pydantic import BaseModel, Field

from app.models.engineering_document import EngineeringDocumentSourceType
from app.repositories.engineering_document_repository import EngineeringDocumentRepository
from app.schemas.engineering_document import (
    EngineeringDocumentChunkCreate,
    EngineeringDocumentCreate,
)
from app.services.document_chunker import (
    DocumentChunker,
    DocumentChunkingConfig,
    DocumentChunkingInput,
)

logger = structlog.get_logger(__name__)


class EngineeringDocumentEmbeddingProvider(Protocol):
    """Protocol for generating dense embeddings during ingestion."""

    async def embed_texts(
        self,
        texts: Sequence[str],
        *,
        run_id: str | None = None,
    ) -> list[list[float]]:
        """Return one embedding for every supplied text."""
        ...


class EngineeringDocumentIngestionRequest(BaseModel):
    """Validated input for ingesting one engineering document."""

    title: str = Field(min_length=1, max_length=255)
    source_type: EngineeringDocumentSourceType
    source_uri: str = Field(min_length=1, max_length=1024)
    raw_content: str = Field(min_length=1)
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    chunking_config: DocumentChunkingConfig = Field(
        default_factory=DocumentChunkingConfig
    )


class EngineeringDocumentIngestionResult(BaseModel):
    """Result returned after ingesting an engineering document."""

    document_id: UUID
    content_hash: str
    chunk_count: int
    created_document: bool
    created_chunks: bool
    duplicate_document: bool


class EngineeringDocumentIngestionService:
    """Coordinate document persistence and deterministic chunk creation.

    This service owns the document-ingestion workflow for the Knowledge Agent.
    API routes, background jobs, or LangGraph nodes should call this service
    instead of manually coordinating repositories and chunking logic.
    """

    def __init__(
        self,
        repository: EngineeringDocumentRepository,
        *,
        chunker: DocumentChunker | None = None,
        embedding_provider: EngineeringDocumentEmbeddingProvider | None = None,
    ) -> None:
        """Initialize the ingestion service.

        Args:
            repository: Repository for engineering documents and chunks.
            chunker: Optional deterministic chunker dependency.
            embedding_provider: Optional provider for semantic chunk embeddings.
        """
        self._repository = repository
        self._chunker = chunker or DocumentChunker()
        self._embedding_provider = embedding_provider

    async def ingest_document(
        self,
        ingestion_request: EngineeringDocumentIngestionRequest,
        *,
        run_id: str | None = None,
    ) -> EngineeringDocumentIngestionResult:
        """Ingest one engineering document and its chunks.

        The operation is idempotent by content hash. If a document with the same
        content already exists, this service will not create a duplicate.

        Args:
            ingestion_request: Validated document ingestion payload.
            run_id: Optional workflow/request identifier for structured logs.

        Returns:
            A structured ingestion result with document and chunk counts.
        """
        content_hash = self._hash_content(ingestion_request.raw_content)
        existing_document = await self._repository.get_document_by_content_hash(
            content_hash,
            run_id=run_id,
        )

        if existing_document is not None:
            existing_chunks = await self._repository.list_chunks_by_document_id(
                existing_document.id,
                run_id=run_id,
            )

            if existing_chunks:
                chunks_without_embeddings = [
                    chunk for chunk in existing_chunks if chunk.embedding is None
                ]

                if self._embedding_provider is not None and chunks_without_embeddings:
                    embeddings = await self._embedding_provider.embed_texts(
                        [chunk.content for chunk in chunks_without_embeddings],
                        run_id=run_id,
                    )

                    if len(embeddings) != len(chunks_without_embeddings):
                        raise ValueError(
                            "embedding provider must return one embedding per document chunk"
                        )

                    updated_count = await self._repository.update_chunk_embeddings(
                        {
                            chunk.id: embedding
                            for chunk, embedding in zip(
                                chunks_without_embeddings,
                                embeddings,
                                strict=True,
                            )
                        },
                        run_id=run_id,
                    )

                    logger.info(
                        "engineering_document_ingestion_embeddings_backfilled",
                        run_id=run_id,
                        document_id=str(existing_document.id),
                        updated_count=updated_count,
                    )

                logger.info(
                    "engineering_document_ingestion_skipped_duplicate",
                    run_id=run_id,
                    document_id=str(existing_document.id),
                    content_hash=content_hash,
                    chunk_count=len(existing_chunks),
                )
                return EngineeringDocumentIngestionResult(
                    document_id=existing_document.id,
                    content_hash=content_hash,
                    chunk_count=len(existing_chunks),
                    created_document=False,
                    created_chunks=False,
                    duplicate_document=True,
                )

            chunk_payloads = self._chunker.chunk_document(
                DocumentChunkingInput(
                    document_id=existing_document.id,
                    raw_content=existing_document.raw_content,
                    metadata_json=existing_document.metadata_json,
                ),
                config=ingestion_request.chunking_config,
                run_id=run_id,
            )
            chunk_payloads = await self._attach_embeddings(
                chunk_payloads,
                run_id=run_id,
            )
            created_chunks = await self._repository.create_chunks(
                chunk_payloads,
                run_id=run_id,
            )

            logger.info(
                "engineering_document_ingestion_completed_existing_document",
                run_id=run_id,
                document_id=str(existing_document.id),
                content_hash=content_hash,
                chunk_count=len(created_chunks),
            )

            return EngineeringDocumentIngestionResult(
                document_id=existing_document.id,
                content_hash=content_hash,
                chunk_count=len(created_chunks),
                created_document=False,
                created_chunks=True,
                duplicate_document=True,
            )

        document = await self._repository.create_document(
            EngineeringDocumentCreate(
                title=ingestion_request.title,
                source_type=ingestion_request.source_type,
                source_uri=ingestion_request.source_uri,
                content_hash=content_hash,
                raw_content=ingestion_request.raw_content,
                metadata_json=ingestion_request.metadata_json,
            ),
            run_id=run_id,
        )

        chunk_payloads = self._chunker.chunk_document(
            DocumentChunkingInput(
                document_id=document.id,
                raw_content=document.raw_content,
                metadata_json=document.metadata_json,
            ),
            config=ingestion_request.chunking_config,
            run_id=run_id,
        )

        chunk_payloads = await self._attach_embeddings(
            chunk_payloads,
            run_id=run_id,
        )
        created_chunks = await self._repository.create_chunks(
            chunk_payloads,
            run_id=run_id,
        )

        logger.info(
            "engineering_document_ingestion_completed",
            run_id=run_id,
            document_id=str(document.id),
            content_hash=content_hash,
            chunk_count=len(created_chunks),
        )

        return EngineeringDocumentIngestionResult(
            document_id=document.id,
            content_hash=content_hash,
            chunk_count=len(created_chunks),
            created_document=True,
            created_chunks=True,
            duplicate_document=False,
        )

    async def _attach_embeddings(
        self,
        chunk_payloads: list[EngineeringDocumentChunkCreate],
        *,
        run_id: str | None,
    ) -> list[EngineeringDocumentChunkCreate]:
        """Attach generated embeddings while preserving chunk order."""
        if self._embedding_provider is None or not chunk_payloads:
            return chunk_payloads

        embeddings = await self._embedding_provider.embed_texts(
            [chunk.content for chunk in chunk_payloads],
            run_id=run_id,
        )

        if len(embeddings) != len(chunk_payloads):
            raise ValueError(
                "embedding provider must return one embedding per document chunk"
            )

        return [
            chunk.model_copy(update={"embedding": embedding})
            for chunk, embedding in zip(chunk_payloads, embeddings, strict=True)
        ]

    def _hash_content(self, raw_content: str) -> str:
        """Return a SHA-256 content hash for duplicate detection."""
        return hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
