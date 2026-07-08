"""CI-style quality gate for Knowledge Agent retrieval.

This test protects AgentFlow AI from retrieval regressions. If a future
BM25, pgvector, embedding, or reranker change makes known release-risk
queries stop returning the expected engineering document, this test fails.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.engineering_document_repository import EngineeringDocumentRepository
from app.services.engineering_document_ingestion_service import (
    EngineeringDocumentIngestionService,
)
from app.services.engineering_document_retrieval_service import (
    EngineeringDocumentRetrievalService,
)
from app.services.knowledge_retrieval_evaluation_service import (
    EngineeringDocumentRetrievalEvaluationAdapter,
    KnowledgeRetrievalEvaluationReport,
    KnowledgeRetrievalEvaluationService,
)
from tests.fixtures.knowledge_retrieval_eval_cases import (
    build_knowledge_retrieval_eval_cases,
    seed_knowledge_retrieval_eval_documents,
)


MIN_TOP_1_ACCURACY = 1.0
MIN_TOP_K_ACCURACY = 1.0


def _format_safe_failure_summary(
    report: KnowledgeRetrievalEvaluationReport,
) -> str:
    """Build a safe failure summary without raw document content or raw chunks."""
    failures = [
        {
            "case_name": failure.case_name,
            "expected_document_title": failure.expected_document_title,
            "reason": failure.reason,
            "retrieved_documents": [
                {
                    "rank": document.rank,
                    "document_title": document.document_title,
                    "source_type": document.source_type,
                }
                for document in failure.retrieved_documents
            ],
        }
        for failure in report.failed_case_details
    ]

    return (
        "Knowledge retrieval quality gate failed. "
        f"top_1_accuracy={report.top_1_accuracy}, "
        f"top_k_accuracy={report.top_k_accuracy}, "
        f"failures={failures}"
    )


@pytest.mark.anyio
async def test_knowledge_retrieval_quality_gate_passes_for_bm25_baseline(
    db_session: AsyncSession,
) -> None:
    """BM25 retrieval should satisfy AgentFlow deterministic quality gates."""
    repository = EngineeringDocumentRepository(db_session)
    ingestion_service = EngineeringDocumentIngestionService(repository)
    retrieval_service = EngineeringDocumentRetrievalService(repository)
    adapter = EngineeringDocumentRetrievalEvaluationAdapter(retrieval_service)
    evaluation_service = KnowledgeRetrievalEvaluationService(adapter)

    await seed_knowledge_retrieval_eval_documents(ingestion_service)

    report = await evaluation_service.evaluate(
        build_knowledge_retrieval_eval_cases(top_k=3),
        run_id=uuid4(),
    )

    failure_summary = _format_safe_failure_summary(report)

    assert report.total_cases == 3
    assert report.top_1_accuracy >= MIN_TOP_1_ACCURACY, failure_summary
    assert report.top_k_accuracy >= MIN_TOP_K_ACCURACY, failure_summary
    assert report.failed_cases == 0, failure_summary
