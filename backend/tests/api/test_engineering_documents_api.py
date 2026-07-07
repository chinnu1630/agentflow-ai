"""API tests for engineering document Knowledge Agent endpoints."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import httpx
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import app.models  # noqa: F401 - ensures all SQLAlchemy models are registered
from app.db.base import Base
from app.db.session import get_db_session
from app.main import app


@pytest_asyncio.fixture
async def async_session() -> AsyncGenerator[AsyncSession, None]:
    """Create an isolated in-memory SQLite async session for API tests."""
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


@pytest_asyncio.fixture
async def api_client(
    async_session: AsyncSession,
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create an HTTP client with the database dependency overridden."""
    async def override_get_db_session() -> AsyncGenerator[AsyncSession, None]:
        yield async_session

    app.dependency_overrides[get_db_session] = override_get_db_session

    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        yield client

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_ingest_engineering_document_returns_created(
    api_client: httpx.AsyncClient,
) -> None:
    """POST /engineering-documents/ingest should ingest a new document."""
    response = await api_client.post(
        "/api/v1/engineering-documents/ingest",
        json={
            "title": "Payment Service Runbook",
            "source_type": "runbook",
            "source_uri": "docs/payment-service-runbook.md",
            "raw_content": (
                "Payment service depends on Redis and Postgres. "
                "Redis latency can increase checkout failure risk."
            ),
            "metadata_json": {
                "team": "payments",
                "service": "payment-api",
            },
            "chunking_config": {
                "max_tokens_per_chunk": 20,
                "overlap_tokens": 5,
            },
        },
    )

    assert response.status_code == 201

    data = response.json()

    assert data["document_id"]
    assert len(data["content_hash"]) == 64
    assert data["chunk_count"] >= 1
    assert data["created_document"] is True
    assert data["created_chunks"] is True
    assert data["duplicate_document"] is False


@pytest.mark.asyncio
async def test_ingest_engineering_document_is_idempotent(
    api_client: httpx.AsyncClient,
) -> None:
    """POST /engineering-documents/ingest should not duplicate same content."""
    payload = {
        "title": "Payment Service Runbook",
        "source_type": "runbook",
        "source_uri": "docs/payment-service-runbook.md",
        "raw_content": "Payment service depends on Redis and Postgres.",
        "metadata_json": {
            "team": "payments",
        },
        "chunking_config": {
            "max_tokens_per_chunk": 20,
            "overlap_tokens": 5,
        },
    }

    first_response = await api_client.post(
        "/api/v1/engineering-documents/ingest",
        json=payload,
    )
    second_response = await api_client.post(
        "/api/v1/engineering-documents/ingest",
        json=payload,
    )

    assert first_response.status_code == 201
    assert second_response.status_code == 200

    first_data = first_response.json()
    second_data = second_response.json()

    assert first_data["document_id"] == second_data["document_id"]
    assert second_data["created_document"] is False
    assert second_data["created_chunks"] is False
    assert second_data["duplicate_document"] is True


@pytest.mark.asyncio
async def test_retrieve_engineering_document_chunks_returns_matches(
    api_client: httpx.AsyncClient,
) -> None:
    """POST /engineering-documents/retrieve should return matching chunks."""
    ingest_response = await api_client.post(
        "/api/v1/engineering-documents/ingest",
        json={
            "title": "Payment Service Runbook",
            "source_type": "runbook",
            "source_uri": "docs/payment-service-runbook.md",
            "raw_content": (
                "Payment service depends on Redis and Postgres. "
                "Redis latency can increase checkout failure risk. "
                "Rollback is required when checkout failures exceed threshold."
            ),
            "metadata_json": {
                "team": "payments",
                "service": "payment-api",
            },
            "chunking_config": {
                "max_tokens_per_chunk": 20,
                "overlap_tokens": 5,
            },
        },
    )

    assert ingest_response.status_code == 201

    retrieve_response = await api_client.post(
        "/api/v1/engineering-documents/retrieve",
        json={
            "query": "Redis checkout failure",
            "top_k": 3,
        },
    )

    assert retrieve_response.status_code == 200

    data = retrieve_response.json()

    assert data["query"] == "Redis checkout failure"
    assert data["total_candidates"] >= 1
    assert len(data["results"]) >= 1
    assert data["results"][0]["title"] == "Payment Service Runbook"
    assert data["results"][0]["score"] > 0
    assert "Redis" in data["results"][0]["content"]


@pytest.mark.asyncio
async def test_retrieve_engineering_document_chunks_respects_source_type_filter(
    api_client: httpx.AsyncClient,
) -> None:
    """POST /engineering-documents/retrieve should support source_type filtering."""
    await api_client.post(
        "/api/v1/engineering-documents/ingest",
        json={
            "title": "Payment Runbook",
            "source_type": "runbook",
            "source_uri": "docs/payment-runbook.md",
            "raw_content": "Redis checkout failure rollback procedure.",
            "metadata_json": {"team": "payments"},
            "chunking_config": {
                "max_tokens_per_chunk": 20,
                "overlap_tokens": 5,
            },
        },
    )

    await api_client.post(
        "/api/v1/engineering-documents/ingest",
        json={
            "title": "Release Checklist",
            "source_type": "release_checklist",
            "source_uri": "docs/release-checklist.md",
            "raw_content": "Redis checkout failure requires release manager approval.",
            "metadata_json": {"team": "release"},
            "chunking_config": {
                "max_tokens_per_chunk": 20,
                "overlap_tokens": 5,
            },
        },
    )

    response = await api_client.post(
        "/api/v1/engineering-documents/retrieve",
        json={
            "query": "Redis checkout failure",
            "top_k": 5,
            "source_type": "release_checklist",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["results"]) == 1
    assert data["results"][0]["title"] == "Release Checklist"
    assert data["results"][0]["source_type"] == "release_checklist"


@pytest.mark.asyncio
async def test_ingest_engineering_document_rejects_invalid_payload(
    api_client: httpx.AsyncClient,
) -> None:
    """POST /engineering-documents/ingest should reject invalid input."""
    response = await api_client.post(
        "/api/v1/engineering-documents/ingest",
        json={
            "title": "",
            "source_type": "runbook",
            "source_uri": "docs/payment-service-runbook.md",
            "raw_content": "",
        },
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_retrieve_engineering_document_chunks_rejects_invalid_query(
    api_client: httpx.AsyncClient,
) -> None:
    """POST /engineering-documents/retrieve should reject invalid queries."""
    response = await api_client.post(
        "/api/v1/engineering-documents/retrieve",
        json={
            "query": "",
            "top_k": 3,
        },
    )

    assert response.status_code == 422
