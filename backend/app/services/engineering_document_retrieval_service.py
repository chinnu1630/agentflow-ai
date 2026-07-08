"""BM25-style retrieval service for engineering document chunks.

This service provides the first deterministic retrieval layer for the Knowledge
Agent. It intentionally avoids embeddings, pgvector, rerankers, and LLM calls so
retrieval behavior can be tested before semantic search is introduced.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import structlog
from pydantic import BaseModel, Field, field_validator

from app.models.engineering_document import (
    EngineeringDocument,
    EngineeringDocumentSourceType,
)
from app.models.engineering_document_chunk import EngineeringDocumentChunk

logger = structlog.get_logger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


@dataclass(frozen=True, slots=True)
class _CandidateChunk:
    """Internal candidate used while ranking engineering document chunks."""

    document: EngineeringDocument
    chunk: EngineeringDocumentChunk
    terms: list[str]


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
    """Retrieve relevant engineering document chunks using BM25-style scoring.

    This service scans a bounded set of engineering documents, scores each chunk
    with deterministic lexical relevance, and returns the top-k chunks. BM25 is
    used as a production-friendly baseline before pgvector or embeddings are
    introduced.
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

        candidate_chunks = await self._build_candidate_chunks(
            documents=documents,
            source_type=retrieval_request.source_type,
            run_id=run_id,
        )

        document_frequency_by_term = self._calculate_document_frequencies(
            candidate_chunks
        )
        average_document_length = self._calculate_average_document_length(
            candidate_chunks
        )

        candidates: list[EngineeringDocumentRetrievalResult] = []

        for candidate in candidate_chunks:
            bm25_score = self._calculate_bm25_score(
                query_terms=query_terms,
                chunk_terms=candidate.terms,
                document_frequency_by_term=document_frequency_by_term,
                total_documents=len(candidate_chunks),
                average_document_length=average_document_length,
            )

            phrase_bonus = self._calculate_phrase_bonus(
                query_phrase=query_phrase,
                chunk_content=candidate.chunk.content,
            )

            score = round(bm25_score + phrase_bonus, 4)

            if score <= 0:
                continue

            candidates.append(
                EngineeringDocumentRetrievalResult(
                    document_id=candidate.document.id,
                    chunk_id=candidate.chunk.id,
                    title=candidate.document.title,
                    source_type=candidate.document.source_type,
                    source_uri=candidate.document.source_uri,
                    chunk_index=candidate.chunk.chunk_index,
                    score=score,
                    content=candidate.chunk.content,
                    token_count=candidate.chunk.token_count,
                    metadata_json=candidate.chunk.metadata_json or {},
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
            retrieval_strategy="bm25_keyword",
            total_chunks_scanned=len(candidate_chunks),
            total_candidates=len(candidates),
            returned_results=len(ranked_results),
            top_k=retrieval_request.top_k,
        )

        return EngineeringDocumentRetrievalResponse(
            query=retrieval_request.query,
            total_candidates=len(candidates),
            results=ranked_results,
        )

    async def _build_candidate_chunks(
        self,
        *,
        documents: list[EngineeringDocument],
        source_type: EngineeringDocumentSourceType | None,
        run_id: str | None,
    ) -> list[_CandidateChunk]:
        """Load document chunks that are eligible for retrieval ranking."""
        candidate_chunks: list[_CandidateChunk] = []

        for document in documents:
            if source_type is not None and document.source_type != source_type:
                continue

            chunks = await self._repository.list_chunks_by_document_id(
                document.id,
                run_id=run_id,
            )

            for chunk in chunks:
                chunk_terms = self._tokenize(chunk.content)

                if not chunk_terms:
                    continue

                candidate_chunks.append(
                    _CandidateChunk(
                        document=document,
                        chunk=chunk,
                        terms=chunk_terms,
                    )
                )

        return candidate_chunks

    def _calculate_document_frequencies(
        self,
        candidate_chunks: list[_CandidateChunk],
    ) -> dict[str, int]:
        """Count how many chunks contain each term."""
        document_frequency_by_term: dict[str, int] = {}

        for candidate in candidate_chunks:
            for term in set(candidate.terms):
                document_frequency_by_term[term] = (
                    document_frequency_by_term.get(term, 0) + 1
                )

        return document_frequency_by_term

    def _calculate_average_document_length(
        self,
        candidate_chunks: list[_CandidateChunk],
    ) -> float:
        """Calculate average chunk length used for BM25 length normalization."""
        if not candidate_chunks:
            return 1.0

        total_terms = sum(len(candidate.terms) for candidate in candidate_chunks)

        return total_terms / len(candidate_chunks)

    def _calculate_bm25_score(
        self,
        *,
        query_terms: list[str],
        chunk_terms: list[str],
        document_frequency_by_term: dict[str, int],
        total_documents: int,
        average_document_length: float,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:
        """Calculate a BM25-style relevance score for one chunk.

        BM25 rewards chunks that contain important query terms while reducing
        the advantage of very long chunks that match only by accident.
        """
        if not query_terms or not chunk_terms or total_documents <= 0:
            return 0.0

        if average_document_length <= 0:
            average_document_length = 1.0

        term_frequency_by_term = Counter(chunk_terms)
        chunk_length = len(chunk_terms)
        score = 0.0

        for term in set(query_terms):
            term_frequency = term_frequency_by_term.get(term, 0)

            if term_frequency == 0:
                continue

            document_frequency = document_frequency_by_term.get(term, 0)

            inverse_document_frequency = math.log(
                1
                + (
                    (total_documents - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
            )

            denominator = term_frequency + k1 * (
                1 - b + b * (chunk_length / average_document_length)
            )

            score += inverse_document_frequency * (
                (term_frequency * (k1 + 1)) / denominator
            )

        return score

    def _calculate_phrase_bonus(
        self,
        *,
        query_phrase: str,
        chunk_content: str,
    ) -> float:
        """Return a small deterministic bonus for exact phrase matches."""
        if len(query_phrase) < 3:
            return 0.0

        if query_phrase in chunk_content.lower():
            return 0.25

        return 0.0

    def _score_chunk(
        self,
        *,
        query_terms: set[str],
        query_phrase: str,
        chunk_content: str,
    ) -> float:
        """Return a deterministic keyword score for one chunk.

        This method is kept for backward-compatible unit tests that may call the
        previous private scorer directly. The main retrieval path uses
        collection-level BM25 scoring because BM25 needs document-frequency
        statistics across all candidate chunks.
        """
        chunk_terms = self._tokenize(chunk_content)

        if not query_terms or not chunk_terms:
            return 0.0

        document_frequency_by_term = {term: 1 for term in set(chunk_terms)}
        bm25_score = self._calculate_bm25_score(
            query_terms=list(query_terms),
            chunk_terms=chunk_terms,
            document_frequency_by_term=document_frequency_by_term,
            total_documents=1,
            average_document_length=float(len(chunk_terms)),
        )

        phrase_bonus = self._calculate_phrase_bonus(
            query_phrase=query_phrase,
            chunk_content=chunk_content,
        )

        return round(bm25_score + phrase_bonus, 4)

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into lowercase alphanumeric terms while preserving frequency."""
        return [token.lower() for token in _TOKEN_PATTERN.findall(text)]