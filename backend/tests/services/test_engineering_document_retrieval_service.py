"""Tests for engineering document keyword retrieval service."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
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
from app.repositories.engineering_document_repository import (
    EngineeringDocumentRepository,
    EngineeringDocumentSemanticMatch,
)
from app.services.document_chunker import DocumentChunkingConfig
from app.services.engineering_document_embedding_provider import (
    EngineeringDocumentEmbeddingError,
)
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "expected_top_title"),
    [
        (
            "Redis checkout failure",
            "Payment Redis Incident Runbook",
        ),
        (
            "release approval checklist",
            "Release Readiness Checklist",
        ),
        (
            "rollback after payment outage",
            "Payment Rollback Procedure",
        ),
    ],
)
async def test_retrieval_quality_eval_cases_return_expected_top_document(
    async_session: AsyncSession,
    query: str,
    expected_top_title: str,
) -> None:
    """BM25 retrieval should pass deterministic AgentFlow quality eval cases."""
    repository = EngineeringDocumentRepository(async_session)
    ingestion_service = EngineeringDocumentIngestionService(repository)
    retrieval_service = EngineeringDocumentRetrievalService(repository)

    documents = [
        _ingestion_request(
            title="Payment Redis Incident Runbook",
            source_type=EngineeringDocumentSourceType.RUNBOOK,
            source_uri="docs/payment-redis-incident-runbook.md",
            raw_content=(
                "Redis checkout failure caused payment release risk. "
                "Redis latency increased during checkout. "
                "Use the payment rollback procedure when checkout failures spike."
            ),
        ),
        _ingestion_request(
            title="Release Readiness Checklist",
            source_type=EngineeringDocumentSourceType.RELEASE_CHECKLIST,
            source_uri="docs/release-readiness-checklist.md",
            raw_content=(
                "Release approval checklist requires Jira P1 review, "
                "GitHub pull request approval, CI validation, rollback readiness, "
                "and engineering manager sign off."
            ),
        ),
        _ingestion_request(
            title="Payment Rollback Procedure",
            source_type=EngineeringDocumentSourceType.RUNBOOK,
            source_uri="docs/payment-rollback-procedure.md",
            raw_content=(
                "Payment outage rollback procedure explains how to revert "
                "payment service deployments after checkout errors, failed releases, "
                "or customer-impacting payment incidents."
            ),
        ),
    ]

    for document in documents:
        await ingestion_service.ingest_document(document)

    response = await retrieval_service.retrieve_relevant_chunks(
        EngineeringDocumentRetrievalRequest(
            query=query,
            top_k=3,
        )
    )

    assert response.results
    assert response.results[0].title == expected_top_title



class FakeQueryEmbeddingProvider:
    """Generate a deterministic query embedding for hybrid retrieval tests."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed_texts(
        self,
        texts: Sequence[str],
        *,
        run_id: str | None = None,
    ) -> list[list[float]]:
        """Return one deterministic 384-dimensional vector per query."""
        self.calls.append(list(texts))
        return [[0.5] * 384 for _text in texts]


@pytest.mark.asyncio
async def test_hybrid_retrieval_returns_semantic_only_match(
    async_session: AsyncSession,
) -> None:
    """Hybrid retrieval should recover relevant chunks without keyword overlap."""
    repository = EngineeringDocumentRepository(async_session)
    ingestion_service = EngineeringDocumentIngestionService(repository)

    result = await ingestion_service.ingest_document(
        _ingestion_request(
            title="Production Release Policy",
            source_type=EngineeringDocumentSourceType.RELEASE_CHECKLIST,
            source_uri="docs/production-release-policy.md",
            raw_content=(
                "Release manager approval is mandatory before production deployment."
            ),
        )
    )

    chunks = await repository.list_chunks_by_document_id(result.document_id)
    semantic_chunk = chunks[0]

    async def fake_semantic_search(
        *,
        query_embedding: list[float],
        limit: int = 20,
        document_ids: list[UUID] | None = None,
        run_id: str | None = None,
    ) -> list[EngineeringDocumentSemanticMatch]:
        assert query_embedding == [0.5] * 384
        assert limit >= 1

        return [
            EngineeringDocumentSemanticMatch(
                chunk=semantic_chunk,
                similarity_score=0.92,
            )
        ]

    repository.search_chunks_by_embedding = fake_semantic_search  # type: ignore[method-assign]
    embedding_provider = FakeQueryEmbeddingProvider()
    retrieval_service = EngineeringDocumentRetrievalService(
        repository,
        embedding_provider=embedding_provider,
    )

    response = await retrieval_service.retrieve_relevant_chunks(
        EngineeringDocumentRetrievalRequest(
            query="Can we ship?",
            top_k=3,
        ),
        run_id="hybrid-semantic-test",
    )

    assert embedding_provider.calls == [["Can we ship?"]]
    assert response.results
    assert response.results[0].title == "Production Release Policy"
    assert response.results[0].score > 0



@pytest.mark.asyncio
async def test_hybrid_retrieval_fuses_duplicate_lexical_and_semantic_match(
    async_session: AsyncSession,
) -> None:
    """RRF should merge one chunk found by both retrieval strategies."""
    repository = EngineeringDocumentRepository(async_session)
    ingestion_service = EngineeringDocumentIngestionService(repository)

    result = await ingestion_service.ingest_document(
        _ingestion_request(
            title="Release Approval Policy",
            source_type=EngineeringDocumentSourceType.RELEASE_CHECKLIST,
            source_uri="docs/release-approval-policy.md",
            raw_content="Release manager approval is required before deployment.",
        )
    )

    chunks = await repository.list_chunks_by_document_id(result.document_id)
    matching_chunk = chunks[0]

    async def fake_semantic_search(
        *,
        query_embedding: list[float],
        limit: int = 20,
        document_ids: list[UUID] | None = None,
        run_id: str | None = None,
    ) -> list[EngineeringDocumentSemanticMatch]:
        return [
            EngineeringDocumentSemanticMatch(
                chunk=matching_chunk,
                similarity_score=0.95,
            )
        ]

    repository.search_chunks_by_embedding = fake_semantic_search  # type: ignore[method-assign]

    retrieval_service = EngineeringDocumentRetrievalService(
        repository,
        embedding_provider=FakeQueryEmbeddingProvider(),
    )

    response = await retrieval_service.retrieve_relevant_chunks(
        EngineeringDocumentRetrievalRequest(
            query="release manager approval",
            top_k=5,
        )
    )

    assert response.total_candidates == 1
    assert len(response.results) == 1
    assert response.results[0].title == "Release Approval Policy"
    assert response.results[0].score == pytest.approx(2 / 61, abs=1e-6)


class FailingQueryEmbeddingProvider:
    """Simulate an unavailable local embedding model."""

    async def embed_texts(
        self,
        texts: Sequence[str],
        *,
        run_id: str | None = None,
    ) -> list[list[float]]:
        """Raise the provider's expected operational error."""
        raise EngineeringDocumentEmbeddingError("Embedding model unavailable.")


@pytest.mark.asyncio
async def test_semantic_failure_degrades_to_bm25_results(
    async_session: AsyncSession,
) -> None:
    """Embedding failure should not prevent deterministic lexical retrieval."""
    repository = EngineeringDocumentRepository(async_session)
    ingestion_service = EngineeringDocumentIngestionService(repository)

    await ingestion_service.ingest_document(
        _ingestion_request(
            title="Payment Failure Runbook",
            source_type=EngineeringDocumentSourceType.RUNBOOK,
            source_uri="docs/payment-failure-runbook.md",
            raw_content="Redis checkout failure requires payment service rollback.",
        )
    )

    retrieval_service = EngineeringDocumentRetrievalService(
        repository,
        embedding_provider=FailingQueryEmbeddingProvider(),
    )

    response = await retrieval_service.retrieve_relevant_chunks(
        EngineeringDocumentRetrievalRequest(
            query="Redis checkout failure",
            top_k=3,
        ),
        run_id="semantic-fallback-test",
    )

    assert response.total_candidates == 1
    assert len(response.results) == 1
    assert response.results[0].title == "Payment Failure Runbook"
    assert response.results[0].score > 0


class FakeCandidateReranker:
    """Return deterministic cross-encoder scores for retrieval tests."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    async def score_candidates(
        self,
        *,
        query: str,
        candidate_contents: Sequence[str],
        run_id: str | None = None,
    ) -> list[float]:
        """Return configured scores in candidate order."""
        self.calls.append((query, list(candidate_contents)))
        return list(self._scores)


class FailingCandidateReranker:
    """Simulate a cross-encoder inference failure."""

    async def score_candidates(
        self,
        *,
        query: str,
        candidate_contents: Sequence[str],
        run_id: str | None = None,
    ) -> list[float]:
        """Raise the expected reranker domain error."""
        from app.services.engineering_document_reranker import (
            EngineeringDocumentRerankerError,
        )

        raise EngineeringDocumentRerankerError("Reranker unavailable.")


@pytest.mark.asyncio
async def test_cross_encoder_reranker_changes_fused_result_order(
    async_session: AsyncSession,
) -> None:
    """Cross-encoder scores should determine final candidate ordering."""
    repository = EngineeringDocumentRepository(async_session)
    ingestion_service = EngineeringDocumentIngestionService(repository)

    await ingestion_service.ingest_document(
        _ingestion_request(
            title="Lexical Release Notes",
            source_type=EngineeringDocumentSourceType.RUNBOOK,
            source_uri="docs/lexical-release-notes.md",
            raw_content="Release approval release approval checklist guidance.",
        )
    )
    await ingestion_service.ingest_document(
        _ingestion_request(
            title="Production Deployment Policy",
            source_type=EngineeringDocumentSourceType.RELEASE_CHECKLIST,
            source_uri="docs/production-deployment-policy.md",
            raw_content="A manager must authorize the release before production deployment.",
        )
    )

    reranker = FakeCandidateReranker(scores=[0.1, 0.9])
    retrieval_service = EngineeringDocumentRetrievalService(
        repository,
        reranker=reranker,
    )

    response = await retrieval_service.retrieve_relevant_chunks(
        EngineeringDocumentRetrievalRequest(
            query="release approval",
            top_k=2,
        ),
        run_id="reranker-order-test",
    )

    assert len(response.results) == 2
    assert response.results[0].title == "Production Deployment Policy"
    assert response.results[0].score == 0.9
    assert len(reranker.calls) == 1


@pytest.mark.asyncio
async def test_reranker_failure_preserves_fused_ranking(
    async_session: AsyncSession,
) -> None:
    """Reranker failure should preserve the deterministic fallback ranking."""
    repository = EngineeringDocumentRepository(async_session)
    ingestion_service = EngineeringDocumentIngestionService(repository)

    await ingestion_service.ingest_document(
        _ingestion_request(
            title="Payment Release Runbook",
            source_type=EngineeringDocumentSourceType.RUNBOOK,
            source_uri="docs/payment-release-runbook.md",
            raw_content="Redis checkout failure requires immediate rollback.",
        )
    )

    retrieval_service = EngineeringDocumentRetrievalService(
        repository,
        reranker=FailingCandidateReranker(),
    )

    response = await retrieval_service.retrieve_relevant_chunks(
        EngineeringDocumentRetrievalRequest(
            query="Redis checkout failure",
            top_k=3,
        ),
        run_id="reranker-fallback-test",
    )

    assert len(response.results) == 1
    assert response.results[0].title == "Payment Release Runbook"
    assert response.results[0].score > 0
