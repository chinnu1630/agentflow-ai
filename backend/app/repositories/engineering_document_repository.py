"""Repository for engineering knowledge-base documents and chunks."""

from __future__ import annotations

from uuid import UUID

import structlog
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.engineering_document import EngineeringDocument
from app.models.engineering_document_chunk import EngineeringDocumentChunk
from app.schemas.engineering_document import (
    EngineeringDocumentChunkCreate,
    EngineeringDocumentCreate,
)

logger = structlog.get_logger(__name__)


class EngineeringDocumentRepositoryError(RuntimeError):
    """Raised when engineering document persistence fails."""


class EngineeringDocumentRepository:
    """Repository for engineering documents and document chunks.

    This repository owns all database access for Knowledge Agent source
    documents. Services and LangGraph nodes should call this class instead of
    writing SQLAlchemy queries directly.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the repository with an async database session."""
        self._session = session

    async def create_document(
        self,
        document_create: EngineeringDocumentCreate,
        *,
        run_id: str | None = None,
    ) -> EngineeringDocument:
        """Create and persist a new engineering document.

        Args:
            document_create: Validated document creation payload.
            run_id: Optional workflow/request identifier for structured logs.

        Returns:
            The persisted EngineeringDocument ORM object.

        Raises:
            EngineeringDocumentRepositoryError: If the document cannot be saved.
        """
        document = EngineeringDocument(
            title=document_create.title,
            source_type=document_create.source_type,
            source_uri=document_create.source_uri,
            content_hash=document_create.content_hash.lower(),
            raw_content=document_create.raw_content,
            metadata_json=document_create.metadata_json,
        )

        self._session.add(document)

        try:
            await self._session.flush()
            await self._session.refresh(document)
        except IntegrityError as exc:
            await self._session.rollback()
            logger.warning(
                "engineering_document_create_conflict",
                run_id=run_id,
                content_hash=document_create.content_hash.lower(),
                source_uri=document_create.source_uri,
            )
            raise EngineeringDocumentRepositoryError(
                "Engineering document could not be created because it violates a database constraint."
            ) from exc

        logger.info(
            "engineering_document_created",
            run_id=run_id,
            document_id=str(document.id),
            source_type=document.source_type.value,
            source_uri=document.source_uri,
        )

        return document

    async def get_document_by_id(
        self,
        document_id: UUID,
        *,
        run_id: str | None = None,
    ) -> EngineeringDocument | None:
        """Return an engineering document by ID, or None when it does not exist."""
        result = await self._session.execute(
            select(EngineeringDocument).where(EngineeringDocument.id == document_id)
        )
        document = result.scalar_one_or_none()

        logger.info(
            "engineering_document_fetched_by_id",
            run_id=run_id,
            document_id=str(document_id),
            found=document is not None,
        )

        return document

    async def get_document_by_content_hash(
        self,
        content_hash: str,
        *,
        run_id: str | None = None,
    ) -> EngineeringDocument | None:
        """Return an engineering document by content hash, or None when missing."""
        normalized_hash = content_hash.lower()

        result = await self._session.execute(
            select(EngineeringDocument).where(
                EngineeringDocument.content_hash == normalized_hash
            )
        )
        document = result.scalar_one_or_none()

        logger.info(
            "engineering_document_fetched_by_content_hash",
            run_id=run_id,
            content_hash=normalized_hash,
            found=document is not None,
        )

        return document

    async def list_documents(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        run_id: str | None = None,
    ) -> list[EngineeringDocument]:
        """List engineering documents ordered by newest first.

        Args:
            limit: Maximum number of documents to return.
            offset: Number of documents to skip.
            run_id: Optional workflow/request identifier for structured logs.

        Returns:
            A list of EngineeringDocument ORM objects.

        Raises:
            ValueError: If limit or offset is invalid.
        """
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")

        if offset < 0:
            raise ValueError("offset must be greater than or equal to 0")

        result = await self._session.execute(
            select(EngineeringDocument)
            .order_by(EngineeringDocument.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        documents = list(result.scalars().all())

        logger.info(
            "engineering_documents_listed",
            run_id=run_id,
            limit=limit,
            offset=offset,
            count=len(documents),
        )

        return documents

    async def create_chunk(
        self,
        chunk_create: EngineeringDocumentChunkCreate,
        *,
        run_id: str | None = None,
    ) -> EngineeringDocumentChunk:
        """Create and persist a single engineering document chunk.

        Args:
            chunk_create: Validated chunk creation payload.
            run_id: Optional workflow/request identifier for structured logs.

        Returns:
            The persisted EngineeringDocumentChunk ORM object.

        Raises:
            EngineeringDocumentRepositoryError: If the chunk cannot be saved.
        """
        chunk = EngineeringDocumentChunk(
            document_id=chunk_create.document_id,
            chunk_index=chunk_create.chunk_index,
            content=chunk_create.content,
            token_count=chunk_create.token_count,
            metadata_json=chunk_create.metadata_json,
        )

        self._session.add(chunk)

        try:
            await self._session.flush()
            await self._session.refresh(chunk)
        except IntegrityError as exc:
            await self._session.rollback()
            logger.warning(
                "engineering_document_chunk_create_conflict",
                run_id=run_id,
                document_id=str(chunk_create.document_id),
                chunk_index=chunk_create.chunk_index,
            )
            raise EngineeringDocumentRepositoryError(
                "Engineering document chunk could not be created because it violates a database constraint."
            ) from exc

        logger.info(
            "engineering_document_chunk_created",
            run_id=run_id,
            chunk_id=str(chunk.id),
            document_id=str(chunk.document_id),
            chunk_index=chunk.chunk_index,
            token_count=chunk.token_count,
        )

        return chunk

    async def create_chunks(
        self,
        chunk_creates: list[EngineeringDocumentChunkCreate],
        *,
        run_id: str | None = None,
    ) -> list[EngineeringDocumentChunk]:
        """Create and persist multiple chunks.

        Args:
            chunk_creates: Validated chunk creation payloads.
            run_id: Optional workflow/request identifier for structured logs.

        Returns:
            Persisted EngineeringDocumentChunk ORM objects.
        """
        chunks: list[EngineeringDocumentChunk] = []

        for chunk_create in chunk_creates:
            chunk = EngineeringDocumentChunk(
                document_id=chunk_create.document_id,
                chunk_index=chunk_create.chunk_index,
                content=chunk_create.content,
                token_count=chunk_create.token_count,
                metadata_json=chunk_create.metadata_json,
            )
            self._session.add(chunk)
            chunks.append(chunk)

        try:
            await self._session.flush()
            for chunk in chunks:
                await self._session.refresh(chunk)
        except IntegrityError as exc:
            await self._session.rollback()
            logger.warning(
                "engineering_document_chunks_create_conflict",
                run_id=run_id,
                chunk_count=len(chunk_creates),
            )
            raise EngineeringDocumentRepositoryError(
                "Engineering document chunks could not be created because one or more chunks violate a database constraint."
            ) from exc

        logger.info(
            "engineering_document_chunks_created",
            run_id=run_id,
            chunk_count=len(chunks),
        )

        return chunks

    async def list_chunks_by_document_id(
        self,
        document_id: UUID,
        *,
        run_id: str | None = None,
    ) -> list[EngineeringDocumentChunk]:
        """List chunks for a document ordered by chunk_index."""
        result = await self._session.execute(
            select(EngineeringDocumentChunk)
            .where(EngineeringDocumentChunk.document_id == document_id)
            .order_by(EngineeringDocumentChunk.chunk_index.asc())
        )
        chunks = list(result.scalars().all())

        logger.info(
            "engineering_document_chunks_listed",
            run_id=run_id,
            document_id=str(document_id),
            count=len(chunks),
        )

        return chunks

    async def list_chunks_by_document_ids(
        self,
        document_ids: list[UUID],
        *,
        run_id: str | None = None,
    ) -> list[EngineeringDocumentChunk]:
        """List chunks for multiple documents in one database query.

        Args:
            document_ids: Engineering document IDs whose chunks should be loaded.
            run_id: Optional workflow/request identifier for structured logs.

        Returns:
            Engineering document chunks ordered by document ID and chunk index.

        Raises:
            ValueError: If too many document IDs are requested at once.
        """
        if not document_ids:
            logger.info(
                "engineering_document_chunks_batch_listed",
                run_id=run_id,
                document_count=0,
                count=0,
            )
            return []

        unique_document_ids = list(dict.fromkeys(document_ids))

        if len(unique_document_ids) > 100:
            raise ValueError("document_ids must contain at most 100 IDs")

        result = await self._session.execute(
            select(EngineeringDocumentChunk)
            .where(EngineeringDocumentChunk.document_id.in_(unique_document_ids))
            .order_by(
                EngineeringDocumentChunk.document_id.asc(),
                EngineeringDocumentChunk.chunk_index.asc(),
            )
        )
        chunks = list(result.scalars().all())

        logger.info(
            "engineering_document_chunks_batch_listed",
            run_id=run_id,
            document_count=len(unique_document_ids),
            count=len(chunks),
        )

        return chunks
