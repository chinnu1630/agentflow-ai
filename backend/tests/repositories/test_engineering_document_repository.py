"""Tests for the engineering document repository."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import Float
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import app.models  # noqa: F401 - ensures all SQLAlchemy models are registered
from app.db.base import Base
from app.models.engineering_document import EngineeringDocumentSourceType
from app.repositories.engineering_document_repository import (
    EngineeringDocumentRepository,
    EngineeringDocumentRepositoryError,
)
from app.schemas.engineering_document import (
    EngineeringDocumentChunkCreate,
    EngineeringDocumentCreate,
)


def _sha256(content: str) -> str:
    """Return a deterministic SHA-256 hash for test document content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _document_create(
    *,
    title: str = "Payment Service Runbook",
    raw_content: str = "Payment service depends on Postgres, Redis, and Stripe.",
    source_uri: str = "docs/payment-service-runbook.md",
) -> EngineeringDocumentCreate:
    """Build a valid EngineeringDocumentCreate payload for tests."""
    return EngineeringDocumentCreate(
        title=title,
        source_type=EngineeringDocumentSourceType.RUNBOOK,
        source_uri=source_uri,
        content_hash=_sha256(raw_content),
        raw_content=raw_content,
        metadata_json={
            "team": "payments",
            "service": "payment-api",
        },
    )


@pytest_asyncio.fixture
async def async_session() -> AsyncGenerator[AsyncSession, None]:
    """Create an isolated in-memory SQLite async session for repository tests."""
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


@pytest.mark.asyncio
async def test_create_document_persists_engineering_document(
    async_session: AsyncSession,
) -> None:
    """create_document should persist a validated engineering document."""
    repository = EngineeringDocumentRepository(async_session)
    document_create = _document_create()

    document = await repository.create_document(
        document_create,
        run_id="test-run-id",
    )

    assert document.id is not None
    assert document.title == "Payment Service Runbook"
    assert document.source_type == EngineeringDocumentSourceType.RUNBOOK
    assert document.source_uri == "docs/payment-service-runbook.md"
    assert document.content_hash == document_create.content_hash
    assert document.raw_content == document_create.raw_content
    assert document.metadata_json["team"] == "payments"
    assert document.created_at is not None
    assert document.updated_at is not None


@pytest.mark.asyncio
async def test_get_document_by_id_returns_document_when_found(
    async_session: AsyncSession,
) -> None:
    """get_document_by_id should return a document when the ID exists."""
    repository = EngineeringDocumentRepository(async_session)
    created_document = await repository.create_document(_document_create())

    found_document = await repository.get_document_by_id(created_document.id)

    assert found_document is not None
    assert found_document.id == created_document.id
    assert found_document.title == created_document.title


@pytest.mark.asyncio
async def test_get_document_by_id_returns_none_when_missing(
    async_session: AsyncSession,
) -> None:
    """get_document_by_id should return None when no document exists."""
    repository = EngineeringDocumentRepository(async_session)

    found_document = await repository.get_document_by_id(uuid4())

    assert found_document is None


@pytest.mark.asyncio
async def test_get_document_by_content_hash_returns_document_when_found(
    async_session: AsyncSession,
) -> None:
    """get_document_by_content_hash should return a matching document."""
    repository = EngineeringDocumentRepository(async_session)
    document_create = _document_create()
    created_document = await repository.create_document(document_create)

    found_document = await repository.get_document_by_content_hash(
        document_create.content_hash.upper()
    )

    assert found_document is not None
    assert found_document.id == created_document.id
    assert found_document.content_hash == document_create.content_hash


@pytest.mark.asyncio
async def test_create_document_rejects_duplicate_content_hash(
    async_session: AsyncSession,
) -> None:
    """create_document should raise a repository error for duplicate content."""
    repository = EngineeringDocumentRepository(async_session)
    document_create = _document_create()

    await repository.create_document(document_create)

    duplicate_document_create = _document_create(
        title="Duplicate Payment Runbook",
        raw_content=document_create.raw_content,
        source_uri="docs/duplicate-payment-service-runbook.md",
    )

    with pytest.raises(EngineeringDocumentRepositoryError):
        await repository.create_document(duplicate_document_create)


@pytest.mark.asyncio
async def test_list_documents_returns_documents(
    async_session: AsyncSession,
) -> None:
    """list_documents should return persisted documents."""
    repository = EngineeringDocumentRepository(async_session)

    first_document = await repository.create_document(
        _document_create(
            title="Payment Service Runbook",
            raw_content="Payment runbook content.",
            source_uri="docs/payment-runbook.md",
        )
    )
    second_document = await repository.create_document(
        _document_create(
            title="Release Readiness Checklist",
            raw_content="Release checklist content.",
            source_uri="docs/release-checklist.md",
        )
    )

    documents = await repository.list_documents(limit=10, offset=0)

    document_ids = {document.id for document in documents}
    assert first_document.id in document_ids
    assert second_document.id in document_ids


@pytest.mark.asyncio
async def test_list_documents_rejects_invalid_limit(
    async_session: AsyncSession,
) -> None:
    """list_documents should reject an invalid limit."""
    repository = EngineeringDocumentRepository(async_session)

    with pytest.raises(ValueError, match="limit must be between 1 and 100"):
        await repository.list_documents(limit=0)


@pytest.mark.asyncio
async def test_list_documents_rejects_invalid_offset(
    async_session: AsyncSession,
) -> None:
    """list_documents should reject a negative offset."""
    repository = EngineeringDocumentRepository(async_session)

    with pytest.raises(ValueError, match="offset must be greater than or equal to 0"):
        await repository.list_documents(offset=-1)


@pytest.mark.asyncio
async def test_create_chunk_persists_engineering_document_chunk(
    async_session: AsyncSession,
) -> None:
    """create_chunk should persist a chunk for an engineering document."""
    repository = EngineeringDocumentRepository(async_session)
    document = await repository.create_document(_document_create())

    chunk = await repository.create_chunk(
        EngineeringDocumentChunkCreate(
            document_id=document.id,
            chunk_index=0,
            content="Payment service depends on Redis.",
            token_count=6,
            embedding=[0.1, 0.2, 0.3],
            metadata_json={"section": "dependencies"},
        )
    )

    assert chunk.id is not None
    assert chunk.document_id == document.id
    assert chunk.chunk_index == 0
    assert chunk.content == "Payment service depends on Redis."
    assert chunk.token_count == 6
    assert chunk.embedding == [0.1, 0.2, 0.3]
    assert chunk.metadata_json["section"] == "dependencies"


@pytest.mark.asyncio
async def test_create_chunks_persists_multiple_chunks(
    async_session: AsyncSession,
) -> None:
    """create_chunks should persist multiple chunks in one repository call."""
    repository = EngineeringDocumentRepository(async_session)
    document = await repository.create_document(_document_create())

    chunks = await repository.create_chunks(
        [
            EngineeringDocumentChunkCreate(
                document_id=document.id,
                chunk_index=0,
                content="Payment service overview.",
                token_count=3,
                metadata_json={"section": "overview"},
            ),
            EngineeringDocumentChunkCreate(
                document_id=document.id,
                chunk_index=1,
                content="Payment service rollback procedure.",
                token_count=4,
                metadata_json={"section": "rollback"},
            ),
        ]
    )

    assert len(chunks) == 2
    assert chunks[0].document_id == document.id
    assert chunks[1].document_id == document.id


@pytest.mark.asyncio
async def test_list_chunks_by_document_id_returns_chunks_in_index_order(
    async_session: AsyncSession,
) -> None:
    """list_chunks_by_document_id should return chunks ordered by chunk_index."""
    repository = EngineeringDocumentRepository(async_session)
    document = await repository.create_document(_document_create())

    await repository.create_chunks(
        [
            EngineeringDocumentChunkCreate(
                document_id=document.id,
                chunk_index=2,
                content="Rollback procedure.",
                token_count=2,
            ),
            EngineeringDocumentChunkCreate(
                document_id=document.id,
                chunk_index=0,
                content="Overview.",
                token_count=1,
            ),
            EngineeringDocumentChunkCreate(
                document_id=document.id,
                chunk_index=1,
                content="Dependencies.",
                token_count=1,
            ),
        ]
    )

    chunks = await repository.list_chunks_by_document_id(document.id)

    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert [chunk.content for chunk in chunks] == [
        "Overview.",
        "Dependencies.",
        "Rollback procedure.",
    ]


@pytest.mark.asyncio
async def test_create_chunk_rejects_duplicate_document_chunk_index(
    async_session: AsyncSession,
) -> None:
    """create_chunk should reject duplicate chunk indexes for one document."""
    repository = EngineeringDocumentRepository(async_session)
    document = await repository.create_document(_document_create())

    chunk_create = EngineeringDocumentChunkCreate(
        document_id=document.id,
        chunk_index=0,
        content="Original chunk.",
        token_count=2,
    )

    await repository.create_chunk(chunk_create)

    duplicate_chunk_create = EngineeringDocumentChunkCreate(
        document_id=document.id,
        chunk_index=0,
        content="Duplicate chunk.",
        token_count=2,
    )

    with pytest.raises(EngineeringDocumentRepositoryError):
        await repository.create_chunk(duplicate_chunk_create)


@pytest.mark.anyio
async def test_list_chunks_by_document_ids_returns_chunks_for_multiple_documents(
    async_session: AsyncSession,
) -> None:
    """Repository should batch-load chunks for multiple engineering documents."""
    repository = EngineeringDocumentRepository(async_session)

    first_document = await repository.create_document(
        EngineeringDocumentCreate(
            title="Payment Redis Runbook",
            source_type=EngineeringDocumentSourceType.RUNBOOK,
            source_uri="internal://runbooks/payment-redis",
            content_hash="a" * 64,
            raw_content="Redis checkout failure rollback guidance.",
            metadata_json={"team": "payments"},
        )
    )

    second_document = await repository.create_document(
        EngineeringDocumentCreate(
            title="Checkout Release Checklist",
            source_type=EngineeringDocumentSourceType.RUNBOOK,
            source_uri="internal://checklists/checkout-release",
            content_hash="b" * 64,
            raw_content="Checkout release readiness checklist.",
            metadata_json={"team": "checkout"},
        )
    )

    await repository.create_chunks(
        [
            EngineeringDocumentChunkCreate(
                document_id=first_document.id,
                chunk_index=0,
                content="Redis checkout failure.",
                token_count=3,
                metadata_json={"chunk": 0},
            ),
            EngineeringDocumentChunkCreate(
                document_id=second_document.id,
                chunk_index=0,
                content="Checkout release readiness.",
                token_count=3,
                metadata_json={"chunk": 0},
            ),
        ]
    )

    chunks = await repository.list_chunks_by_document_ids(
        [first_document.id, second_document.id]
    )

    assert len(chunks) == 2
    assert {chunk.document_id for chunk in chunks} == {
        first_document.id,
        second_document.id,
    }


@pytest.mark.anyio
async def test_list_chunks_by_document_ids_returns_empty_list_for_empty_input(
    async_session: AsyncSession,
) -> None:
    """Repository should safely return no chunks when no document IDs are provided."""
    repository = EngineeringDocumentRepository(async_session)

    chunks = await repository.list_chunks_by_document_ids([])

    assert chunks == []



@pytest.mark.asyncio
async def test_update_chunk_embeddings_backfills_existing_chunks(
    async_session: AsyncSession,
) -> None:
    """Repository should add embeddings to chunks created before vector support."""
    repository = EngineeringDocumentRepository(async_session)
    document = await repository.create_document(_document_create())

    chunks = await repository.create_chunks(
        [
            EngineeringDocumentChunkCreate(
                document_id=document.id,
                chunk_index=0,
                content="Payment service overview.",
                token_count=3,
            ),
            EngineeringDocumentChunkCreate(
                document_id=document.id,
                chunk_index=1,
                content="Payment rollback procedure.",
                token_count=3,
            ),
        ]
    )

    updated_count = await repository.update_chunk_embeddings(
        {
            chunks[0].id: [1.0] * 384,
            chunks[1].id: [2.0] * 384,
        },
        run_id="embedding-backfill-test",
    )

    stored_chunks = await repository.list_chunks_by_document_id(document.id)

    assert updated_count == 2
    assert stored_chunks[0].embedding == [1.0] * 384
    assert stored_chunks[1].embedding == [2.0] * 384


@pytest.mark.asyncio
async def test_search_chunks_by_embedding_returns_empty_on_sqlite(
    async_session: AsyncSession,
) -> None:
    """Semantic search should degrade safely when pgvector is unavailable."""
    repository = EngineeringDocumentRepository(async_session)

    matches = await repository.search_chunks_by_embedding(
        query_embedding=[0.1] * 384,
        limit=5,
        run_id="semantic-search-sqlite-test",
    )

    assert matches == []



@pytest.mark.asyncio
async def test_semantic_search_distance_column_uses_float_result_type() -> None:
    """Cosine distance must use FLOAT instead of the vector result processor."""
    captured_statements: list[Any] = []

    class FakeResult:
        """Return no semantic matches."""

        def all(self) -> list[Any]:
            """Return an empty row collection."""
            return []

    class FakePostgresSession:
        """Capture the semantic-search statement without a real database."""

        def get_bind(self) -> Any:
            """Return a PostgreSQL-like bind."""
            return SimpleNamespace(
                dialect=SimpleNamespace(name="postgresql"),
            )

        async def execute(self, statement: Any) -> FakeResult:
            """Capture the statement and return an empty result."""
            captured_statements.append(statement)
            return FakeResult()

    repository = EngineeringDocumentRepository(
        FakePostgresSession(),  # type: ignore[arg-type]
    )

    matches = await repository.search_chunks_by_embedding(
        query_embedding=[0.1] * 384,
        limit=5,
        run_id="semantic-distance-type-test",
    )

    assert matches == []
    assert len(captured_statements) == 1

    selected_columns = list(captured_statements[0].selected_columns)
    distance_column = selected_columns[-1]

    assert distance_column.key == "cosine_distance"
    assert isinstance(distance_column.type, Float)
