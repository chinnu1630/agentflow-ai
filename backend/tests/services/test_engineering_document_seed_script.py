"""Tests for the engineering-document ingestion script."""

from __future__ import annotations

import argparse
import sys
from collections.abc import AsyncGenerator, Sequence
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

import app.models  # noqa: F401 - register SQLAlchemy models
from app.db.base import Base
from app.models.engineering_document import EngineeringDocumentSourceType
from app.repositories.engineering_document_repository import (
    EngineeringDocumentRepository,
)
from scripts import ingest_engineering_documents as seed_script


@pytest_asyncio.fixture
async def session_factory(
) -> AsyncGenerator[async_sessionmaker[AsyncSession], None]:
    """Create an isolated database session factory for script tests."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    yield factory
    await engine.dispose()


class FakeEmbeddingProvider:
    """Generate deterministic embeddings without loading model weights."""

    async def embed_texts(
        self,
        texts: Sequence[str],
        *,
        run_id: str | None = None,
    ) -> list[list[float]]:
        """Return one 384-dimensional embedding per text."""
        return [[float(index + 1)] * 384 for index, _text in enumerate(texts)]


def _write_document(
    document_path: Path,
    *,
    document_type: str = "RUNBOOK",
) -> None:
    """Write a valid engineering document fixture."""
    document_path.write_text(
        f"""---
document_id: payment-service-test-v1
title: Payment Service Test Document
document_type: {document_type}
service: payment-service
tags:
  - payments
  - production
---
# Payment Service

Rollback when payment failures exceed the production threshold.
""",
        encoding="utf-8",
    )


def test_parse_front_matter_returns_metadata_and_body() -> None:
    """Front matter should be separated from the Markdown body."""
    metadata, body = seed_script.parse_front_matter(
        """---
title: Payment Runbook
document_type: RUNBOOK
tags:
  - payments
  - rollback
---
# Runbook

Rollback instructions.
"""
    )

    assert metadata["title"] == "Payment Runbook"
    assert metadata["document_type"] == "RUNBOOK"
    assert metadata["tags"] == ["payments", "rollback"]
    assert body == "# Runbook\n\nRollback instructions."


@pytest.mark.parametrize(
    ("raw_content", "expected_message"),
    [
        ("# Missing front matter", "must begin"),
        ("---\ntitle: Runbook", "is not closed"),
        ("---\n- orphan\n---\nBody", "has no parent key"),
        ("---\ntitle: Runbook\n---", "body must not be empty"),
    ],
)
def test_parse_front_matter_rejects_invalid_documents(
    raw_content: str,
    expected_message: str,
) -> None:
    """Invalid front matter should fail before database ingestion."""
    with pytest.raises(ValueError, match=expected_message):
        seed_script.parse_front_matter(raw_content)


@pytest.mark.parametrize(
    ("document_type", "expected_source_type"),
    [
        ("RUNBOOK", EngineeringDocumentSourceType.RUNBOOK),
        ("runbook", EngineeringDocumentSourceType.RUNBOOK),
        ("CHECKLIST", EngineeringDocumentSourceType.RELEASE_CHECKLIST),
    ],
)
def test_resolve_source_type_maps_supported_types(
    document_type: str,
    expected_source_type: EngineeringDocumentSourceType,
) -> None:
    """Supported front-matter types should map to model enum values."""
    assert seed_script.resolve_source_type(document_type) == expected_source_type


def test_resolve_source_type_rejects_unsupported_type() -> None:
    """Unsupported document types should not silently map to OTHER."""
    with pytest.raises(ValueError, match="Unsupported engineering document type"):
        seed_script.resolve_source_type("SPEC")


def test_build_ingestion_request_loads_document(tmp_path: Path) -> None:
    """A Markdown file should produce a validated ingestion request."""
    document_path = tmp_path / "payment-runbook.md"
    _write_document(document_path)

    request = seed_script.build_ingestion_request(document_path)

    assert request.title == "Payment Service Test Document"
    assert request.source_type == EngineeringDocumentSourceType.RUNBOOK
    assert request.source_uri == str(document_path)
    assert request.raw_content.startswith("# Payment Service")
    assert "---" not in request.raw_content
    assert request.metadata_json["service"] == "payment-service"
    assert request.metadata_json["tags"] == ["payments", "production"]
    assert request.metadata_json["source_filename"] == document_path.name


def test_build_ingestion_request_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """A missing document should fail with its path in the error."""
    document_path = tmp_path / "missing.md"

    with pytest.raises(FileNotFoundError, match="Engineering document not found"):
        seed_script.build_ingestion_request(document_path)


@pytest.mark.asyncio
async def test_ingest_documents_persists_and_deduplicates_documents(
    tmp_path: Path,
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The script should persist documents and remain idempotent."""
    document_path = tmp_path / "payment-runbook.md"
    _write_document(document_path)

    monkeypatch.setattr(
        seed_script,
        "get_session_factory",
        lambda: session_factory,
    )
    monkeypatch.setattr(
        seed_script,
        "get_engineering_document_embedding_provider",
        lambda: FakeEmbeddingProvider(),
    )

    first_results = await seed_script.ingest_documents([document_path])
    second_results = await seed_script.ingest_documents([document_path])

    assert len(first_results) == 1
    assert first_results[0].duplicate_document is False
    assert first_results[0].chunk_count > 0
    assert second_results[0].duplicate_document is True
    assert second_results[0].document_id == first_results[0].document_id

    async with session_factory() as session:
        repository = EngineeringDocumentRepository(session)
        documents = await repository.list_documents()
        chunks = await repository.list_chunks_by_document_id(
            first_results[0].document_id
        )

    assert len(documents) == 1
    assert chunks
    assert all(chunk.embedding is not None for chunk in chunks)


def test_parse_arguments_returns_supplied_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI arguments should be converted into document paths."""
    document_path = tmp_path / "runbook.md"
    monkeypatch.setattr(
        sys,
        "argv",
        ["ingest_engineering_documents", str(document_path)],
    )

    arguments = seed_script.parse_arguments()

    assert arguments.documents == [document_path]


def test_main_runs_ingestion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI entry point should execute ingestion and return success."""
    document_path = Path("runbook.md")
    captured_paths: list[Path] = []

    async def fake_ingest_documents(
        document_paths: Sequence[Path],
    ) -> list[object]:
        captured_paths.extend(document_paths)
        return []

    monkeypatch.setattr(
        seed_script,
        "parse_arguments",
        lambda: argparse.Namespace(documents=[document_path]),
    )
    monkeypatch.setattr(
        seed_script,
        "ingest_documents",
        fake_ingest_documents,
    )

    assert seed_script.main() == 0
    assert captured_paths == [document_path]
