"""Keyword retrieval service for engineering document chunks.

This service provides the first deterministic retrieval layer for the Knowledge
Agent. It intentionally avoids embeddings, pgvector, rerankers, and LLM calls so
retrieval behavior can be tested before semantic search is introduced.
"""

from __future__ import annotations

import re
from typing import Any
from uuid import UUID

import structlog
from pydantic import BaseModel, Field, field_validator

from app.models.engineering_document import EngineeringDocumentSourceType
from app.repositories.engineering_document_repository import EngineeringDocumentRepository

logger = structlog.get_logger(__name__)


class EngineeringDocumentRetrievalRequest(BaseModel):
    """Validated request for retrieving relevant engineering document chunks."""

    query: str = Field(min_length=1, max_length=1_000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_limit: int = Field(default=100, ge=1, le=100)
    source_type: EngineeringDocumentSourceType | None = None

    @field_validator("query")
    @classmethod
    def validate_query_contains_text(cls, value: str) -> str:
        """Reject whitespace-only retrieval queries."""
        if not value.strip():
            raise ValueError("query must contain non-whitespace text")

        return value


class EngineeringDocumentRetrievalResult(BaseModel):
    """One retrieved document chunk with citation metadata."""

    document_id: UUID
    chunk_id: UUID
    title: str
    source_type: EngineeringDocumentSourceType
    source_uri: str
    chunk_index: int
    score: float
    content: str
    token_count: int
    metadata_json: dict[str, Any]


class EngineeringDocumentRetrievalResponse(BaseModel):
    """Ranked retrieval response for a Knowledge Agent query."""

    query: str
    total_candidates: int
    results: list[EngineeringDocumentRetrievalResult]


class EngineeringDocumentRetrievalService:
    """Retrieve relevant engineering document chunks using keyword scoring.

    This is a baseline retrieval implementation. It scans a limited number of
    stored documents and ranks chunks by lexical overlap with the query.
    """

    def __init__(self, repository: EngineeringDocumentRepository) -> None:
        """Initialize the retrieval service.

        Args:
            repository: Repository used to read engineering documents and chunks.
        """
        self._repository = repository

    async def retrieve_relevant_chunks(
        self,
        retrieval_request: EngineeringDocumentRetrievalRequest,
        *,
        run_id: str | None = None,
    ) -> EngineeringDocumentRetrievalResponse:
        """Return top-k relevant chunks for a query.

        Args:
            retrieval_request: Validated retrieval request.
            run_id: Optional workflow/request identifier for structured logs.

        Returns:
            Ranked retrieval response containing matching chunks.
        """
        query_terms = self._tokenize(retrieval_request.query)
        query_phrase = retrieval_request.query.strip().lower()

        documents = await self._repository.list_documents(
            limit=retrieval_request.document_limit,
            offset=0,
            run_id=run_id,
        )

        candidates: list[EngineeringDocumentRetrievalResult] = []

        for document in documents:
            if (
                retrieval_request.source_type is not None
                and document.source_type != retrieval_request.source_type
            ):
                continue

            chunks = await self._repository.list_chunks_by_document_id(
                document.id,
                run_id=run_id,
            )

            for chunk in chunks:
                score = self._score_chunk(
                    query_terms=query_terms,
                    query_phrase=query_phrase,
                    chunk_content=chunk.content,
                )

                if score <= 0:
                    continue

                candidates.append(
                    EngineeringDocumentRetrievalResult(
                        document_id=document.id,
                        chunk_id=chunk.id,
                        title=document.title,
                        source_type=document.source_type,
                        source_uri=document.source_uri,
                        chunk_index=chunk.chunk_index,
                        score=score,
                        content=chunk.content,
                        token_count=chunk.token_count,
                        metadata_json=chunk.metadata_json,
                    )
                )

        ranked_results = sorted(
            candidates,
            key=lambda result: (-result.score, result.title, result.chunk_index),
        )[: retrieval_request.top_k]

        logger.info(
            "engineering_document_retrieval_completed",
            run_id=run_id,
            query_length=len(retrieval_request.query),
            source_type=(
                retrieval_request.source_type.value
                if retrieval_request.source_type is not None
                else None
            ),
            total_candidates=len(candidates),
            returned_results=len(ranked_results),
            top_k=retrieval_request.top_k,
        )

        return EngineeringDocumentRetrievalResponse(
            query=retrieval_request.query,
            total_candidates=len(candidates),
            results=ranked_results,
        )

    def _score_chunk(
        self,
        *,
        query_terms: set[str],
        query_phrase: str,
        chunk_content: str,
    ) -> float:
        """Return a deterministic keyword score for one chunk."""
        chunk_content_lower = chunk_content.lower()
        chunk_terms = self._tokenize(chunk_content_lower)

        if not query_terms or not chunk_terms:
            return 0.0

        matched_terms = query_terms.intersection(chunk_terms)
        term_overlap_score = len(matched_terms) / len(query_terms)

        phrase_bonus = 1.0 if query_phrase in chunk_content_lower else 0.0
        frequency_bonus = sum(
            chunk_content_lower.count(term) for term in matched_terms
        ) * 0.05

        return round(term_overlap_score + phrase_bonus + frequency_bonus, 4)

    def _tokenize(self, text: str) -> set[str]:
        """Tokenize text into lowercase alphanumeric terms."""
        return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))
