"""Engineering document API routes for AgentFlow AI.

This module exposes the Knowledge Agent foundation endpoints for ingesting
engineering documents and retrieving relevant document chunks.

Architecture position:
FastAPI route -> Knowledge services -> EngineeringDocumentRepository -> database
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.security import require_scopes
from app.core.security import AuthenticatedPrincipal
from app.db.session import get_db_session
from app.repositories.engineering_document_repository import (
    EngineeringDocumentRepository,
    EngineeringDocumentRepositoryError,
)
from app.services.engineering_document_embedding_provider import (
    SentenceTransformerEmbeddingProvider,
    get_engineering_document_embedding_provider,
)
from app.services.engineering_document_ingestion_service import (
    EngineeringDocumentIngestionRequest,
    EngineeringDocumentIngestionResult,
    EngineeringDocumentIngestionService,
)
from app.services.engineering_document_reranker import (
    CrossEncoderEngineeringDocumentReranker,
    get_engineering_document_reranker,
)
from app.services.engineering_document_retrieval_service import (
    EngineeringDocumentRetrievalRequest,
    EngineeringDocumentRetrievalResponse,
    EngineeringDocumentRetrievalService,
)

router = APIRouter(prefix="/engineering-documents", tags=["engineering-documents"])

KnowledgeReadPrincipalDependency = Annotated[
    AuthenticatedPrincipal,
    Depends(require_scopes("knowledge:read")),
]
KnowledgeWritePrincipalDependency = Annotated[
    AuthenticatedPrincipal,
    Depends(require_scopes("knowledge:write")),
]


@router.post(
    "/ingest",
    response_model=EngineeringDocumentIngestionResult,
)
async def ingest_engineering_document(
    command: EngineeringDocumentIngestionRequest,
    request: Request,
    response: Response,
    _principal: KnowledgeWritePrincipalDependency,
    session: AsyncSession = Depends(get_db_session),
    embedding_provider: SentenceTransformerEmbeddingProvider = Depends(
        get_engineering_document_embedding_provider
    ),
) -> EngineeringDocumentIngestionResult:
    """Ingest one engineering document into the Knowledge Agent store.

    This endpoint stores the original document, chunks it deterministically,
    persists ordered chunks and local semantic embeddings, and returns
    ingestion metadata. It does not call Claude, Slack, or external AI APIs.
    """

    request_id = str(getattr(request.state, "request_id", "unknown-request-id"))

    repository = EngineeringDocumentRepository(session=session)
    service = EngineeringDocumentIngestionService(
        repository=repository,
        embedding_provider=embedding_provider,
    )

    try:
        result = await service.ingest_document(
            command,
            run_id=request_id,
        )
        await session.commit()

        response.status_code = (
            status.HTTP_201_CREATED
            if result.created_document
            else status.HTTP_200_OK
        )

        return result

    except EngineeringDocumentRepositoryError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Engineering document could not be ingested.",
        ) from exc

    except ValueError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid engineering document ingestion request.",
        ) from exc

    except SQLAlchemyError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while ingesting engineering document.",
        ) from exc


@router.post(
    "/retrieve",
    response_model=EngineeringDocumentRetrievalResponse,
)
async def retrieve_engineering_document_chunks(
    command: EngineeringDocumentRetrievalRequest,
    request: Request,
    _principal: KnowledgeReadPrincipalDependency,
    session: AsyncSession = Depends(get_db_session),
    embedding_provider: SentenceTransformerEmbeddingProvider = Depends(
        get_engineering_document_embedding_provider
    ),
    reranker: CrossEncoderEngineeringDocumentReranker = Depends(
        get_engineering_document_reranker
    ),
) -> EngineeringDocumentRetrievalResponse:
    """Retrieve relevant engineering document chunks for a query.

    This endpoint combines deterministic BM25 keyword retrieval with pgvector
    semantic retrieval and reciprocal-rank fusion. It does not perform LLM
    synthesis or send engineering data to an external AI service.
    """

    request_id = str(getattr(request.state, "request_id", "unknown-request-id"))

    repository = EngineeringDocumentRepository(session=session)
    service = EngineeringDocumentRetrievalService(
        repository=repository,
        embedding_provider=embedding_provider,
        reranker=reranker,
    )

    try:
        return await service.retrieve_relevant_chunks(
            command,
            run_id=request_id,
        )

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while retrieving engineering document chunks.",
        ) from exc
