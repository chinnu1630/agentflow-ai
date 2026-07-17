"""Hybrid retrieval service for AgentFlow engineering documents.

The Knowledge Agent combines BM25 lexical retrieval, pgvector semantic search,
reciprocal-rank fusion, and local cross-encoder reranking. Semantic and reranker
failures degrade gracefully to deterministic lexical results.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

import structlog
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.exc import SQLAlchemyError

from app.models.engineering_document import (
    EngineeringDocument,
    EngineeringDocumentSourceType,
)
from app.models.engineering_document_chunk import EngineeringDocumentChunk
from app.repositories.engineering_document_repository import (
    EngineeringDocumentRepository,
    EngineeringDocumentSemanticMatch,
)
from app.services.engineering_document_embedding_provider import (
    EngineeringDocumentEmbeddingError,
)
from app.services.engineering_document_reranker import (
    EngineeringDocumentRerankerError,
)

logger = structlog.get_logger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


@dataclass(frozen=True, slots=True)
class _CandidateChunk:
    """Internal candidate used while ranking engineering document chunks."""

    document: EngineeringDocument
    chunk: EngineeringDocumentChunk
    terms: list[str]



class EngineeringDocumentCandidateReranker(Protocol):
    """Protocol for cross-encoder candidate reranking."""

    async def score_candidates(
        self,
        *,
        query: str,
        candidate_contents: Sequence[str],
        run_id: str | None = None,
    ) -> list[float]:
        """Return one relevance score for each candidate."""
        ...


class EngineeringDocumentQueryEmbeddingProvider(Protocol):
    """Protocol for generating query embeddings during semantic retrieval."""

    async def embed_texts(
        self,
        texts: Sequence[str],
        *,
        run_id: str | None = None,
    ) -> list[list[float]]:
        """Return one embedding for every supplied query."""
        ...


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
    """Retrieve relevant chunks using hybrid search and local reranking.

    The service generates bounded lexical and semantic candidate sets, merges
    them with reciprocal-rank fusion, and optionally reranks the fused pool with
    a cross-encoder before returning the requested top-k results.
    """

    def __init__(
        self,
        repository: EngineeringDocumentRepository,
        *,
        embedding_provider: EngineeringDocumentQueryEmbeddingProvider | None = None,
        reranker: EngineeringDocumentCandidateReranker | None = None,
    ) -> None:
        """Initialize the retrieval service.

        Args:
            repository: Repository used to read engineering documents and chunks.
            embedding_provider: Optional provider for semantic query embeddings.
            reranker: Optional cross-encoder used to rerank fused candidates.
        """
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._reranker = reranker

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
        retrieval_started_at = time.perf_counter()
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

        lexical_candidates: list[EngineeringDocumentRetrievalResult] = []

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

            lexical_candidates.append(
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

        lexical_ranked = sorted(
            lexical_candidates,
            key=lambda result: (-result.score, result.title, result.chunk_index),
        )

        semantic_matches: list[EngineeringDocumentSemanticMatch] = []

        if self._embedding_provider is not None and candidate_chunks:
            try:
                query_embeddings = await self._embedding_provider.embed_texts(
                    [retrieval_request.query],
                    run_id=run_id,
                )

                if len(query_embeddings) != 1:
                    raise ValueError(
                        "embedding provider must return one query embedding"
                    )

                semantic_matches = (
                    await self._repository.search_chunks_by_embedding(
                        query_embedding=query_embeddings[0],
                        limit=min(max(retrieval_request.top_k * 4, 20), 100),
                        document_ids=list(
                            dict.fromkeys(
                                candidate.document.id
                                for candidate in candidate_chunks
                            )
                        ),
                        run_id=run_id,
                    )
                )
            except (
                EngineeringDocumentEmbeddingError,
                SQLAlchemyError,
                ValueError,
            ) as exc:
                logger.warning(
                    "engineering_document_semantic_retrieval_failed",
                    run_id=run_id,
                    error_type=exc.__class__.__name__,
                )

        documents_by_id = {document.id: document for document in documents}
        candidate_pool_size = min(max(retrieval_request.top_k * 4, 20), 100)

        fused_results = self._fuse_ranked_results(
            lexical_results=lexical_ranked,
            semantic_matches=semantic_matches,
            documents_by_id=documents_by_id,
            top_k=candidate_pool_size,
        )
        ranked_results = await self._rerank_results(
            query=retrieval_request.query,
            candidates=fused_results,
            top_k=retrieval_request.top_k,
            run_id=run_id,
        )

        total_candidates = len(
            {
                *(result.chunk_id for result in lexical_ranked),
                *(match.chunk.id for match in semantic_matches),
            }
        )
        semantic_results_used = bool(semantic_matches)

        if semantic_results_used and self._reranker is not None:
            retrieval_strategy = "hybrid_bm25_pgvector_rrf_cross_encoder"
        elif semantic_results_used:
            retrieval_strategy = "hybrid_bm25_pgvector_rrf"
        elif self._reranker is not None:
            retrieval_strategy = "bm25_cross_encoder"
        else:
            retrieval_strategy = "bm25_keyword"

        logger.info(
            "engineering_document_retrieval_completed",
            run_id=run_id,
            query_length=len(retrieval_request.query),
            source_type=(
                retrieval_request.source_type.value
                if retrieval_request.source_type is not None
                else None
            ),
            retrieval_strategy=retrieval_strategy,
            total_chunks_scanned=len(candidate_chunks),
            total_candidates=total_candidates,
            returned_results=len(ranked_results),
            top_k=retrieval_request.top_k,
            duration_ms=round((time.perf_counter() - retrieval_started_at) * 1000, 2),
        )

        return EngineeringDocumentRetrievalResponse(
            query=retrieval_request.query,
            total_candidates=total_candidates,
            results=ranked_results,
        )

    async def _rerank_results(
        self,
        *,
        query: str,
        candidates: list[EngineeringDocumentRetrievalResult],
        top_k: int,
        run_id: str | None,
    ) -> list[EngineeringDocumentRetrievalResult]:
        """Rerank fused candidates and degrade gracefully on model failure."""
        if self._reranker is None or not candidates:
            return candidates[:top_k]

        try:
            reranker_scores = await self._reranker.score_candidates(
                query=query,
                candidate_contents=[candidate.content for candidate in candidates],
                run_id=run_id,
            )

            if len(reranker_scores) != len(candidates):
                raise ValueError(
                    "reranker must return one score for every candidate"
                )
        except (EngineeringDocumentRerankerError, ValueError) as exc:
            logger.warning(
                "engineering_document_reranking_degraded",
                run_id=run_id,
                candidate_count=len(candidates),
                error_type=exc.__class__.__name__,
            )
            return candidates[:top_k]

        scored_candidates = [
            candidate.model_copy(update={"score": round(score, 6)})
            for candidate, score in zip(
                candidates,
                reranker_scores,
                strict=True,
            )
        ]

        return sorted(
            scored_candidates,
            key=lambda result: (
                -result.score,
                result.title,
                result.chunk_index,
            ),
        )[:top_k]

    def _fuse_ranked_results(
        self,
        *,
        lexical_results: list[EngineeringDocumentRetrievalResult],
        semantic_matches: list[EngineeringDocumentSemanticMatch],
        documents_by_id: dict[UUID, EngineeringDocument],
        top_k: int,
    ) -> list[EngineeringDocumentRetrievalResult]:
        """Fuse lexical and semantic rankings using reciprocal-rank fusion."""
        if not semantic_matches:
            return lexical_results[:top_k]

        reciprocal_rank_constant = 60
        fused_scores: dict[UUID, float] = {}
        results_by_chunk_id: dict[UUID, EngineeringDocumentRetrievalResult] = {}

        for rank, result in enumerate(lexical_results, start=1):
            results_by_chunk_id[result.chunk_id] = result
            fused_scores[result.chunk_id] = (
                fused_scores.get(result.chunk_id, 0.0)
                + 1.0 / (reciprocal_rank_constant + rank)
            )

        for rank, match in enumerate(semantic_matches, start=1):
            document = documents_by_id.get(match.chunk.document_id)

            if document is None:
                continue

            if match.chunk.id not in results_by_chunk_id:
                results_by_chunk_id[match.chunk.id] = (
                    EngineeringDocumentRetrievalResult(
                        document_id=document.id,
                        chunk_id=match.chunk.id,
                        title=document.title,
                        source_type=document.source_type,
                        source_uri=document.source_uri,
                        chunk_index=match.chunk.chunk_index,
                        score=match.similarity_score,
                        content=match.chunk.content,
                        token_count=match.chunk.token_count,
                        metadata_json=match.chunk.metadata_json or {},
                    )
                )

            fused_scores[match.chunk.id] = (
                fused_scores.get(match.chunk.id, 0.0)
                + 1.0 / (reciprocal_rank_constant + rank)
            )

        ranked_chunk_ids = sorted(
            results_by_chunk_id,
            key=lambda chunk_id: (
                -fused_scores.get(chunk_id, 0.0),
                -results_by_chunk_id[chunk_id].score,
                results_by_chunk_id[chunk_id].title,
                results_by_chunk_id[chunk_id].chunk_index,
            ),
        )[:top_k]

        return [
            results_by_chunk_id[chunk_id].model_copy(
                update={"score": round(fused_scores[chunk_id], 6)}
            )
            for chunk_id in ranked_chunk_ids
        ]

    async def _build_candidate_chunks(
        self,
        *,
        documents: list[EngineeringDocument],
        source_type: EngineeringDocumentSourceType | None,
        run_id: str | None,
    ) -> list[_CandidateChunk]:
        """Load document chunks that are eligible for retrieval ranking.

        This method batch-loads chunks for all eligible documents to avoid the
        N+1 query pattern during Knowledge Agent retrieval.
        """
        eligible_documents = [
            document
            for document in documents
            if source_type is None or document.source_type == source_type
        ]

        if not eligible_documents:
            return []

        documents_by_id = {document.id: document for document in eligible_documents}

        chunks = await self._repository.list_chunks_by_document_ids(
            list(documents_by_id.keys()),
            run_id=run_id,
        )

        candidate_chunks: list[_CandidateChunk] = []

        for chunk in chunks:
            document = documents_by_id.get(chunk.document_id)

            if document is None:
                continue

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