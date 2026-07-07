"""Deterministic document chunking service for the Knowledge Agent.

This service converts raw engineering document text into ordered chunk creation
payloads. It intentionally does not use embeddings, pgvector, Claude, or any
external API. Chunking must be deterministic and testable before we add RAG.
"""

from __future__ import annotations

import re
from typing import Any, Self
from uuid import UUID

import structlog
from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.engineering_document import EngineeringDocumentChunkCreate

logger = structlog.get_logger(__name__)


class DocumentChunkingConfig(BaseModel):
    """Configuration for deterministic document chunking."""

    max_tokens_per_chunk: int = Field(default=200, ge=20, le=2_000)
    overlap_tokens: int = Field(default=40, ge=0, le=500)
    strategy: str = Field(default="fixed_token_window")

    @model_validator(mode="after")
    def validate_overlap_is_smaller_than_chunk_size(self) -> Self:
        """Ensure overlap cannot make the chunking loop invalid."""
        if self.overlap_tokens >= self.max_tokens_per_chunk:
            raise ValueError("overlap_tokens must be smaller than max_tokens_per_chunk")

        return self


class DocumentChunkingInput(BaseModel):
    """Input payload for chunking a single engineering document."""

    document_id: UUID
    raw_content: str = Field(min_length=1)
    metadata_json: dict[str, Any] = Field(default_factory=dict)

    @field_validator("raw_content")
    @classmethod
    def validate_raw_content_has_text(cls, value: str) -> str:
        """Reject empty or whitespace-only document content."""
        if not value.strip():
            raise ValueError("raw_content must contain non-whitespace text")

        return value


class DocumentChunker:
    """Split engineering documents into ordered retrieval chunks.

    The chunker uses a simple token-window strategy based on whitespace tokens.
    This is not a model tokenizer. It is a deterministic approximation that is
    good enough for storage and testing before we add model-specific tokenizers.
    """

    def chunk_document(
        self,
        chunking_input: DocumentChunkingInput,
        *,
        config: DocumentChunkingConfig | None = None,
        run_id: str | None = None,
    ) -> list[EngineeringDocumentChunkCreate]:
        """Split one document into ordered chunk creation payloads.

        Args:
            chunking_input: Validated document content and metadata.
            config: Optional chunking configuration.
            run_id: Optional workflow/request identifier for structured logs.

        Returns:
            Ordered chunk creation payloads.

        Raises:
            ValueError: If tokenization produces no usable tokens.
        """
        resolved_config = config or DocumentChunkingConfig()
        tokens = self._tokenize(chunking_input.raw_content)

        if not tokens:
            raise ValueError("raw_content must produce at least one token")

        chunks: list[EngineeringDocumentChunkCreate] = []
        start_index = 0
        chunk_index = 0
        step_size = (
            resolved_config.max_tokens_per_chunk - resolved_config.overlap_tokens
        )

        while start_index < len(tokens):
            end_index = min(
                start_index + resolved_config.max_tokens_per_chunk,
                len(tokens),
            )
            chunk_tokens = tokens[start_index:end_index]
            chunk_content = " ".join(chunk_tokens)

            chunk_metadata = {
                **chunking_input.metadata_json,
                "chunking_strategy": resolved_config.strategy,
                "start_token_index": start_index,
                "end_token_index": end_index,
                "overlap_tokens": resolved_config.overlap_tokens,
            }

            chunks.append(
                EngineeringDocumentChunkCreate(
                    document_id=chunking_input.document_id,
                    chunk_index=chunk_index,
                    content=chunk_content,
                    token_count=len(chunk_tokens),
                    metadata_json=chunk_metadata,
                )
            )

            if end_index >= len(tokens):
                break

            start_index += step_size
            chunk_index += 1

        logger.info(
            "engineering_document_chunking_completed",
            run_id=run_id,
            document_id=str(chunking_input.document_id),
            chunk_count=len(chunks),
            total_tokens=len(tokens),
            max_tokens_per_chunk=resolved_config.max_tokens_per_chunk,
            overlap_tokens=resolved_config.overlap_tokens,
            strategy=resolved_config.strategy,
        )

        return chunks

    def _tokenize(self, raw_content: str) -> list[str]:
        """Tokenize text using deterministic whitespace token extraction."""
        return re.findall(r"\S+", raw_content)
