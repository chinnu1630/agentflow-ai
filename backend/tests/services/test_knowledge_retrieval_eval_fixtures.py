"""Tests for reusable Knowledge Agent retrieval evaluation fixtures."""

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
    KnowledgeRetrievalEvaluationService,
)
from tests.fixtures.knowledge_retrieval_eval_cases import (
    build_knowledge_retrieval_eval_cases,
    build_knowledge_retrieval_eval_documents,
    seed_knowledge_retrieval_eval_documents,
)


def test_reusable_eval_fixtures_define_expected_benchmark_cases() -> None:
    """Reusable eval fixtures should define stable benchmark cases."""
    cases = build_knowledge_retrieval_eval_cases()

    assert [case.name for case in cases] == [
        "redis_checkout_failure",
        "release_approval_checklist",
        "rollback_after_payment_outage",
    ]
    assert [case.expected_document_title for case in cases] == [
        "Payment Redis Incident Runbook",
        "Release Readiness Checklist",
        "Payment Rollback Procedure",
    ]


def test_reusable_eval_documents_match_expected_titles() -> None:
    """Reusable eval documents should contain the expected benchmark documents."""
    documents = build_knowledge_retrieval_eval_documents()

    assert [document.title for document in documents] == [
        "Payment Redis Incident Runbook",
        "Release Readiness Checklist",
        "Payment Rollback Procedure",
    ]


@pytest.mark.anyio
async def test_reusable_eval_fixtures_pass_against_real_retrieval_service(
    db_session: AsyncSession,
) -> None:
    """Reusable eval fixtures should pass against the real BM25 retriever."""
    repository = EngineeringDocumentRepository(db_session)
    ingestion_service = EngineeringDocumentIngestionService(repository)
    retrieval_service = EngineeringDocumentRetrievalService(repository)
    adapter = EngineeringDocumentRetrievalEvaluationAdapter(retrieval_service)
    evaluation_service = KnowledgeRetrievalEvaluationService(adapter)

    await seed_knowledge_retrieval_eval_documents(ingestion_service)

    report = await evaluation_service.evaluate(
        build_knowledge_retrieval_eval_cases(),
        run_id=uuid4(),
    )

    assert report.total_cases == 3
    assert report.passed_cases == 3
    assert report.failed_cases == 0
    assert report.top_1_accuracy == 1.0
    assert report.top_k_accuracy == 1.0
    assert report.failed_case_details == []
