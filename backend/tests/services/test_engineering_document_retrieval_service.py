"""Tests for engineering document keyword retrieval service."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from uuid import UUID

import pytest
import pytest_asyncio
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import app.models  # noqa: F401 - ensures all SQLAlchemy models are registered
from app.db.base import Base
from app.models.engineering_document import EngineeringDocumentSourceType
from app.models.engineering_document_chunk import EngineeringDocumentChunk
from app.repositories.engineering_document_repository import EngineeringDocumentRepository
from app.services.document_chunker import DocumentChunkingConfig
from app.services.engineering_document_ingestion_service import (
    EngineeringDocumentIngestionRequest,
    EngineeringDocumentIngestionService,
)
from app.services.engineering_document_retrieval_service import (
    EngineeringDocumentRetrievalRequest,
    EngineeringDocumentRetrievalService,
)


@pytest_asyncio.fixture
async def async_session() -> AsyncGenerator[AsyncSession, None]:
    """Create an isolated in-memory SQLite async session for retrieval tests."""
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
    title: str,
    source_type: EngineeringDocumentSourceType,
    source_uri: str,
    raw_content: str,
) -> EngineeringDocumentIngestionRequest:
    """Build a valid ingestion request for retrieval tests."""
    return EngineeringDocumentIngestionRequest(
        title=title,
        source_type=source_type,
        source_uri=source_uri,
        raw_content=raw_content,
        metadata_json={"team": "platform"},
        chunking_config=DocumentChunkingConfig(
            max_tokens_per_chunk=40,
            overlap_tokens=5,
        ),
    )


@pytest.mark.asyncio
async def test_retrieve_relevant_chunks_returns_matching_chunk(
    async_session: AsyncSession,
) -> None:
    """retrieve_relevant_chunks should return chunks matching query terms."""
    repository = EngineeringDocumentRepository(async_session)
    ingestion_service = EngineeringDocumentIngestionService(repository)
    retrieval_service = EngineeringDocumentRetrievalService(repository)

    await ingestion_service.ingest_document(
        _ingestion_request(
            title="Payment Service Runbook",
            source_type=EngineeringDocumentSourceType.RUNBOOK,
            source_uri="docs/payment-runbook.md",
            raw_content=(
                "Payment service depends on Redis and Postgres. "
                "Redis latency can increase checkout failure risk."
            ),
        )
    )

    response = await retrieval_service.retrieve_relevant_chunks(
        EngineeringDocumentRetrievalRequest(query="Redis checkout failure", top_k=3),
        run_id="test-run-id",
    )

    assert response.query == "Redis checkout failure"
    assert response.total_candidates >= 1
    assert len(response.results) == 1
    assert response.results[0].title == "Payment Service Runbook"
    assert "Redis latency" in response.results[0].content
    assert response.results[0].score > 0


@pytest.mark.asyncio
async def test_retrieve_relevant_chunks_orders_results_by_score(
    async_session: AsyncSession,
) -> None:
    """retrieve_relevant_chunks should rank stronger keyword matches first."""
    repository = EngineeringDocumentRepository(async_session)
    ingestion_service = EngineeringDocumentIngestionService(repository)
    retrieval_service = EngineeringDocumentRetrievalService(repository)

    await ingestion_service.ingest_document(
        _ingestion_request(
            title="Payment Runbook",
            source_type=EngineeringDocumentSourceType.RUNBOOK,
            source_uri="docs/payment-runbook.md",
            raw_content=(
                "Redis Redis checkout failure failure payment outage. "
                "Rollback when checkout failure increases."
            ),
        )
    )
    await ingestion_service.ingest_document(
        _ingestion_request(
            title="General Release Checklist",
            source_type=EngineeringDocumentSourceType.RELEASE_CHECKLIST,
            source_uri="docs/release-checklist.md",
            raw_content="Release checklist requires monitoring dashboards and approvals.",
        )
    )

    response = await retrieval_service.retrieve_relevant_chunks(
        EngineeringDocumentRetrievalRequest(query="Redis checkout failure", top_k=5),
    )

    assert response.results[0].title == "Payment Runbook"
    assert response.results[0].score >= response.results[-1].score


@pytest.mark.asyncio
async def test_retrieve_relevant_chunks_respects_source_type_filter(
    async_session: AsyncSession,
) -> None:
    """retrieve_relevant_chunks should filter documents by source type."""
    repository = EngineeringDocumentRepository(async_session)
    ingestion_service = EngineeringDocumentIngestionService(repository)
    retrieval_service = EngineeringDocumentRetrievalService(repository)

    await ingestion_service.ingest_document(
        _ingestion_request(
            title="Payment Runbook",
            source_type=EngineeringDocumentSourceType.RUNBOOK,
            source_uri="docs/payment-runbook.md",
            raw_content="Redis checkout failure rollback procedure.",
        )
    )
    await ingestion_service.ingest_document(
        _ingestion_request(
            title="Release Checklist",
            source_type=EngineeringDocumentSourceType.RELEASE_CHECKLIST,
            source_uri="docs/release-checklist.md",
            raw_content="Redis checkout failure must be reviewed before release.",
        )
    )

    response = await retrieval_service.retrieve_relevant_chunks(
        EngineeringDocumentRetrievalRequest(
            query="Redis checkout failure",
            source_type=EngineeringDocumentSourceType.RELEASE_CHECKLIST,
        )
    )

    assert len(response.results) == 1
    assert response.results[0].title == "Release Checklist"
    assert (
        response.results[0].source_type
        == EngineeringDocumentSourceType.RELEASE_CHECKLIST
    )


@pytest.mark.asyncio
async def test_retrieve_relevant_chunks_returns_empty_results_when_no_match(
    async_session: AsyncSession,
) -> None:
    """retrieve_relevant_chunks should return no results when nothing matches."""
    repository = EngineeringDocumentRepository(async_session)
    ingestion_service = EngineeringDocumentIngestionService(repository)
    retrieval_service = EngineeringDocumentRetrievalService(repository)

    await ingestion_service.ingest_document(
        _ingestion_request(
            title="Payment Runbook",
            source_type=EngineeringDocumentSourceType.RUNBOOK,
            source_uri="docs/payment-runbook.md",
            raw_content="Payment service depends on Redis and Postgres.",
        )
    )

    response = await retrieval_service.retrieve_relevant_chunks(
        EngineeringDocumentRetrievalRequest(query="kubernetes autoscaling incident"),
    )

    assert response.total_candidates == 0
    assert response.results == []


def test_retrieval_request_rejects_empty_query() -> None:
    """EngineeringDocumentRetrievalRequest should reject empty query text."""
    with pytest.raises(ValidationError):
        EngineeringDocumentRetrievalRequest(query="")


def test_retrieval_request_rejects_whitespace_query() -> None:
    """EngineeringDocumentRetrievalRequest should reject whitespace-only query."""
    with pytest.raises(ValidationError):
        EngineeringDocumentRetrievalRequest(query="   \n\t   ")


def test_retrieval_request_rejects_invalid_top_k() -> None:
    """EngineeringDocumentRetrievalRequest should reject invalid top_k values."""
    with pytest.raises(ValidationError):
        EngineeringDocumentRetrievalRequest(query="redis failure", top_k=0)


@pytest.mark.anyio
async def test_retrieval_service_uses_batch_chunk_loading(
    async_session: AsyncSession,
) -> None:
    """Retrieval should batch-load chunks to avoid an N+1 query pattern."""
    repository = EngineeringDocumentRepository(async_session)
    ingestion_service = EngineeringDocumentIngestionService(repository=repository)
    retrieval_service = EngineeringDocumentRetrievalService(repository=repository)

    await ingestion_service.ingest_document(
        EngineeringDocumentIngestionRequest(
            title="Payment Redis Runbook",
            source_type=EngineeringDocumentSourceType.RUNBOOK,
            source_uri="internal://runbooks/payment-redis-batch-loading",
            raw_content="Redis checkout failure rollback guidance.",
            metadata_json={"team": "payments"},
        )
    )

    original_batch_loader = repository.list_chunks_by_document_ids
    original_single_loader = repository.list_chunks_by_document_id

    batch_loader_calls = 0
    single_loader_calls = 0

    async def counting_batch_loader(
        document_ids: list[UUID],
        *,
        run_id: str | None = None,
    ) -> list[EngineeringDocumentChunk]:
        nonlocal batch_loader_calls
        batch_loader_calls += 1
        return await original_batch_loader(document_ids, run_id=run_id)

    async def counting_single_loader(
        document_id: UUID,
        *,
        run_id: str | None = None,
    ) -> list[EngineeringDocumentChunk]:
        nonlocal single_loader_calls
        single_loader_calls += 1
        return await original_single_loader(document_id, run_id=run_id)

    repository.list_chunks_by_document_ids = counting_batch_loader  # type: ignore[method-assign]
    repository.list_chunks_by_document_id = counting_single_loader  # type: ignore[method-assign]

    response = await retrieval_service.retrieve_relevant_chunks(
        EngineeringDocumentRetrievalRequest(
            query="Redis checkout failure",
            top_k=1,
        )
    )

    assert len(response.results) == 1
    assert batch_loader_calls == 1
    assert single_loader_calls == 0


@pytest.mark.asyncio
async def test_bm25_ranks_full_context_above_repeated_single_keyword(
    async_session: AsyncSession,
) -> None:
    """BM25 retrieval should prefer full risk context over repeated single terms."""
    repository = EngineeringDocumentRepository(async_session)
    ingestion_service = EngineeringDocumentIngestionService(repository)
    retrieval_service = EngineeringDocumentRetrievalService(repository)

    await ingestion_service.ingest_document(
        _ingestion_request(
            title="Payment Redis Incident Runbook",
            source_type=EngineeringDocumentSourceType.RUNBOOK,
            source_uri="docs/payment-redis-incident-runbook.md",
            raw_content=(
                "Redis checkout failure caused payment release risk. "
                "Rollback checkout service if Redis latency increases."
            ),
        )
    )

    await ingestion_service.ingest_document(
        _ingestion_request(
            title="Generic Redis Notes",
            source_type=EngineeringDocumentSourceType.RUNBOOK,
            source_uri="docs/generic-redis-notes.md",
            raw_content=(
                "Redis Redis Redis Redis Redis Redis Redis. "
                "General cache notes for engineering teams."
            ),
        )
    )

    response = await retrieval_service.retrieve_relevant_chunks(
        EngineeringDocumentRetrievalRequest(
            query="Redis checkout failure",
            top_k=2,
        )
    )

    assert len(response.results) == 2
    assert response.results[0].title == "Payment Redis Incident Runbook"
    assert response.results[0].score > response.results[1].score


@pytest.mark.asyncio
async def test_retrieve_relevant_chunks_respects_top_k_limit(
    async_session: AsyncSession,
) -> None:
    """retrieve_relevant_chunks should not return more than the requested top_k."""
    repository = EngineeringDocumentRepository(async_session)
    ingestion_service = EngineeringDocumentIngestionService(repository)
    retrieval_service = EngineeringDocumentRetrievalService(repository)

    await ingestion_service.ingest_document(
        _ingestion_request(
            title="Payment Redis Runbook",
            source_type=EngineeringDocumentSourceType.RUNBOOK,
            source_uri="docs/payment-redis-top-k.md",
            raw_content="Redis checkout failure rollback guidance.",
        )
    )

    await ingestion_service.ingest_document(
        _ingestion_request(
            title="Checkout Failure Notes",
            source_type=EngineeringDocumentSourceType.RUNBOOK,
            source_uri="docs/checkout-failure-top-k.md",
            raw_content="Checkout failure release risk and rollback notes.",
        )
    )

    response = await retrieval_service.retrieve_relevant_chunks(
        EngineeringDocumentRetrievalRequest(
            query="Redis checkout failure",
            top_k=1,
        )
    )

    assert len(response.results) == 1
