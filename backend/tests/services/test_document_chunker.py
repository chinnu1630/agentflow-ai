"""Tests for deterministic document chunking."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.services.document_chunker import (
    DocumentChunker,
    DocumentChunkingConfig,
    DocumentChunkingInput,
)


def _numbered_tokens(count: int) -> str:
    """Return deterministic whitespace-separated tokens for chunking tests."""
    return " ".join(f"token-{index}" for index in range(1, count + 1))


def test_chunk_document_creates_single_chunk_for_short_document() -> None:
    """Short documents should produce one chunk."""
    document_id = uuid4()
    chunker = DocumentChunker()

    chunks = chunker.chunk_document(
        DocumentChunkingInput(
            document_id=document_id,
            raw_content="Payment service depends on Redis and Postgres.",
            metadata_json={"service": "payment-api"},
        ),
        config=DocumentChunkingConfig(max_tokens_per_chunk=20, overlap_tokens=0),
        run_id="test-run-id",
    )

    assert len(chunks) == 1
    assert chunks[0].document_id == document_id
    assert chunks[0].chunk_index == 0
    assert chunks[0].content == "Payment service depends on Redis and Postgres."
    assert chunks[0].token_count == 7
    assert chunks[0].metadata_json["service"] == "payment-api"
    assert chunks[0].metadata_json["chunking_strategy"] == "fixed_token_window"


def test_chunk_document_creates_multiple_chunks_with_overlap() -> None:
    """Long documents should produce overlapping ordered chunks."""
    document_id = uuid4()
    chunker = DocumentChunker()

    chunks = chunker.chunk_document(
        DocumentChunkingInput(
            document_id=document_id,
            raw_content=_numbered_tokens(50),
        ),
        config=DocumentChunkingConfig(max_tokens_per_chunk=20, overlap_tokens=5),
    )

    assert len(chunks) == 3
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert chunks[0].content == _numbered_tokens(20)
    assert chunks[1].content == " ".join(
        f"token-{index}" for index in range(16, 36)
    )
    assert chunks[2].content == " ".join(
        f"token-{index}" for index in range(31, 51)
    )
    assert [chunk.token_count for chunk in chunks] == [20, 20, 20]


def test_chunk_document_is_deterministic_for_same_input() -> None:
    """Same input and config should produce the same chunk content."""
    document_id = uuid4()
    chunker = DocumentChunker()
    chunking_input = DocumentChunkingInput(
        document_id=document_id,
        raw_content=_numbered_tokens(45),
    )
    config = DocumentChunkingConfig(max_tokens_per_chunk=20, overlap_tokens=5)

    first_chunks = chunker.chunk_document(chunking_input, config=config)
    second_chunks = chunker.chunk_document(chunking_input, config=config)

    assert [chunk.content for chunk in first_chunks] == [
        chunk.content for chunk in second_chunks
    ]
    assert [chunk.chunk_index for chunk in first_chunks] == [
        chunk.chunk_index for chunk in second_chunks
    ]
    assert [chunk.token_count for chunk in first_chunks] == [
        chunk.token_count for chunk in second_chunks
    ]


def test_chunking_config_rejects_overlap_equal_to_max_tokens() -> None:
    """Config should reject overlap that prevents forward progress."""
    with pytest.raises(ValidationError):
        DocumentChunkingConfig(max_tokens_per_chunk=20, overlap_tokens=20)


def test_chunking_input_rejects_whitespace_only_content() -> None:
    """Chunking input should reject documents without usable text."""
    with pytest.raises(ValidationError):
        DocumentChunkingInput(
            document_id=uuid4(),
            raw_content="   \n\t   ",
        )


def test_chunk_metadata_includes_token_boundaries() -> None:
    """Chunk metadata should preserve token boundaries for traceability."""
    chunker = DocumentChunker()

    chunks = chunker.chunk_document(
        DocumentChunkingInput(
            document_id=uuid4(),
            raw_content=_numbered_tokens(25),
        ),
        config=DocumentChunkingConfig(max_tokens_per_chunk=20, overlap_tokens=5),
    )

    assert chunks[0].metadata_json["start_token_index"] == 0
    assert chunks[0].metadata_json["end_token_index"] == 20
    assert chunks[1].metadata_json["start_token_index"] == 15
    assert chunks[1].metadata_json["end_token_index"] == 25
