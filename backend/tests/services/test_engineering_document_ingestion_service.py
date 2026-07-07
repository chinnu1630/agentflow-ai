"""Tests for engineering document ingestion service."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import app.models  # noqa: F401 - ensures all SQLAlchemy models are registered
from app.db.base import Base
from app.models.engineering_document import EngineeringDocumentSourceType
from app.repositories.engineering_document_repository import EngineeringDocumentRepository
from app.services.document_chunker import DocumentChunkingConfig
from app.services.engineering_document_ingestion_service import (
    EngineeringDocumentIngestionRequest,
    EngineeringDocumentIngestionService,
)


def _numbered_tokens(count: int) -> str:
    """Return deterministic whitespace-separated tokens for ingestion tests."""
    return " ".join(f"token-{index}" for index in range(1, count + 1))


@pytest_asyncio.fixture
async def async_session() -> AsyncGenerator[AsyncSession, None]:
    """Create an isolated in-memory SQLite async session for service tests."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with session_factory() as session:
        yield session

    await engine.dispose()


def _ingestion_request(
    *,
    raw_content: str = "Payment service depends on Redis and Postgres.",
) -> EngineeringDocumentIngestionRequest:
    """Build a valid ingestion request for tests."""
    return EngineeringDocumentIngestionRequest(
        title="Payment Service Runbook",
        source_type=EngineeringDocumentSourceType.RUNBOOK,
        source_uri="docs/payment-service-runbook.md",
        raw_content=raw_content,
        metadata_json={"team": "payments", "service": "payment-api"},
        chunking_config=DocumentChunkingConfig(
            max_tokens_per_chunk=20,
            overlap_tokens=5,
        ),
    )


@pytest.mark.asyncio
async def test_ingest_document_creates_document_and_chunks(
    async_session: AsyncSession,
) -> None:
    """ingest_document should create a document and its chunks."""
    repository = EngineeringDocumentRepository(async_session)
    service = EngineeringDocumentIngestionService(repository)

    result = await service.ingest_document(
        _ingestion_request(raw_content=_numbered_tokens(45)),
        run_id="test-run-id",
    )

    assert result.created_document is True
    assert result.created_chunks is True
    assert result.duplicate_document is False
    assert result.chunk_count == 3

    stored_document = await repository.get_document_by_id(result.document_id)
    assert stored_document is not None
    assert stored_document.title == "Payment Service Runbook"

    stored_chunks = await repository.list_chunks_by_document_id(result.document_id)
    assert len(stored_chunks) == 3
    assert [chunk.chunk_index for chunk in stored_chunks] == [0, 1, 2]


@pytest.mark.asyncio
async def test_ingest_document_is_idempotent_for_same_content(
    async_session: AsyncSession,
) -> None:
    """ingest_document should not duplicate a document with the same content."""
    repository = EngineeringDocumentRepository(async_session)
    service = EngineeringDocumentIngestionService(repository)
    request = _ingestion_request(raw_content=_numbered_tokens(45))

    first_result = await service.ingest_document(request)
    second_result = await service.ingest_document(request)

    assert first_result.document_id == second_result.document_id
    assert first_result.created_document is True
    assert second_result.created_document is False
    assert second_result.created_chunks is False
    assert second_result.duplicate_document is True
    assert second_result.chunk_count == first_result.chunk_count

    documents = await repository.list_documents()
    assert len(documents) == 1

    chunks = await repository.list_chunks_by_document_id(first_result.document_id)
    assert len(chunks) == first_result.chunk_count


@pytest.mark.asyncio
async def test_ingest_document_uses_sha256_content_hash(
    async_session: AsyncSession,
) -> None:
    """ingest_document should return the SHA-256 hash of the raw content."""
    repository = EngineeringDocumentRepository(async_session)
    service = EngineeringDocumentIngestionService(repository)
    raw_content = "Payment service depends on Redis and Postgres."
    expected_hash = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()

    result = await service.ingest_document(
        _ingestion_request(raw_content=raw_content),
    )

    assert result.content_hash == expected_hash

    stored_document = await repository.get_document_by_id(result.document_id)
    assert stored_document is not None
    assert stored_document.content_hash == expected_hash


@pytest.mark.asyncio
async def test_ingest_document_preserves_metadata_on_chunks(
    async_session: AsyncSession,
) -> None:
    """ingest_document should pass document metadata into chunk metadata."""
    repository = EngineeringDocumentRepository(async_session)
    service = EngineeringDocumentIngestionService(repository)

    result = await service.ingest_document(
        _ingestion_request(raw_content=_numbered_tokens(25)),
    )

    chunks = await repository.list_chunks_by_document_id(result.document_id)

    assert chunks[0].metadata_json["team"] == "payments"
    assert chunks[0].metadata_json["service"] == "payment-api"
    assert chunks[0].metadata_json["chunking_strategy"] == "fixed_token_window"


def test_ingestion_request_rejects_empty_title() -> None:
    """EngineeringDocumentIngestionRequest should reject empty titles."""
    with pytest.raises(ValidationError):
        EngineeringDocumentIngestionRequest(
            title="",
            source_type=EngineeringDocumentSourceType.RUNBOOK,
            source_uri="docs/payment-service-runbook.md",
            raw_content="Valid content.",
        )


def test_ingestion_request_rejects_empty_raw_content() -> None:
    """EngineeringDocumentIngestionRequest should reject empty raw content."""
    with pytest.raises(ValidationError):
        EngineeringDocumentIngestionRequest(
            title="Payment Service Runbook",
            source_type=EngineeringDocumentSourceType.RUNBOOK,
            source_uri="docs/payment-service-runbook.md",
            raw_content="",
        )
