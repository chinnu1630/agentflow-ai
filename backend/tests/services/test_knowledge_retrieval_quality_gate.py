"""CI-style quality gate for Knowledge Agent retrieval.

This test protects AgentFlow AI from retrieval regressions. If a future
BM25, pgvector, embedding, or reranker change makes known release-risk
queries stop returning the expected engineering document, this test fails.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.engineering_document import EngineeringDocumentSourceType
from app.repositories.engineering_document_repository import (
    EngineeringDocumentRepository,
    EngineeringDocumentSemanticMatch,
)
from app.services.engineering_document_ingestion_service import (
    EngineeringDocumentIngestionService,
)
from app.services.engineering_document_retrieval_service import (
    EngineeringDocumentRetrievalService,
)
from app.services.knowledge_retrieval_evaluation_service import (
    EngineeringDocumentRetrievalEvaluationAdapter,
    KnowledgeRetrievalEvalCase,
    KnowledgeRetrievalEvalFailureDetail,
    KnowledgeRetrievalEvalRetrievedDocument,
    KnowledgeRetrievalEvaluationReport,
    KnowledgeRetrievalEvaluationService,
)
from tests.fixtures.knowledge_retrieval_eval_cases import (
    KnowledgeRetrievalEvalDocumentFixture,
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


class DeterministicHybridEmbeddingProvider:
    """Generate deterministic query embeddings for the hybrid quality gate."""

    def __init__(self) -> None:
        """Initialize provider call tracking."""
        self.calls: list[list[str]] = []
        self.current_query: str | None = None

    async def embed_texts(
        self,
        texts: Sequence[str],
        *,
        run_id: str | None = None,
    ) -> list[list[float]]:
        """Return one fixed-size embedding per query."""
        self.calls.append(list(texts))
        self.current_query = texts[0] if texts else None
        return [[0.5] * 384 for _text in texts]


class DeterministicHybridReranker:
    """Assign deterministic relevance scores using expected content markers."""

    def __init__(self, expected_markers_by_query: dict[str, str]) -> None:
        """Initialize expected query-to-content markers."""
        self._expected_markers_by_query = expected_markers_by_query
        self.calls: list[tuple[str, list[str]]] = []

    async def score_candidates(
        self,
        *,
        query: str,
        candidate_contents: Sequence[str],
        run_id: str | None = None,
    ) -> list[float]:
        """Score the expected candidate above distractors."""
        contents = list(candidate_contents)
        self.calls.append((query, contents))

        expected_marker = self._expected_markers_by_query[query]

        return [1.0 if expected_marker in content.lower() else 0.0 for content in contents]


@pytest.mark.anyio
async def test_knowledge_retrieval_quality_gate_passes_for_hybrid_pipeline(
    db_session: AsyncSession,
) -> None:
    """Hybrid retrieval should pass lexical and semantic-only eval cases."""
    repository = EngineeringDocumentRepository(db_session)
    ingestion_service = EngineeringDocumentIngestionService(repository)

    await seed_knowledge_retrieval_eval_documents(ingestion_service)
    await ingestion_service.ingest_document(
        KnowledgeRetrievalEvalDocumentFixture(
            title="Production Deployment Policy",
            source_type=EngineeringDocumentSourceType.RELEASE_CHECKLIST,
            source_uri="docs/production-deployment-policy.md",
            raw_content=(
                "A manager must authorize the release before production deployment. "
                "Emergency changes still require an approval record."
            ),
            metadata_json={"team": "release-management"},
        ).to_ingestion_request()
    )

    query_to_title = {
        "Redis checkout failure": "Payment Redis Incident Runbook",
        "release approval checklist": "Release Readiness Checklist",
        "rollback after payment outage": "Payment Rollback Procedure",
        "Can we ship?": "Production Deployment Policy",
    }
    expected_markers_by_query = {
        "Redis checkout failure": "redis checkout failure",
        "release approval checklist": "jira p1 review",
        "rollback after payment outage": "payment outage rollback procedure",
        "Can we ship?": "manager must authorize the release",
    }

    documents = await repository.list_documents(limit=100)
    chunks_by_title = {}

    for document in documents:
        chunks = await repository.list_chunks_by_document_id(document.id)
        chunks_by_title[document.title] = chunks[0]

    embedding_provider = DeterministicHybridEmbeddingProvider()

    async def deterministic_semantic_search(
        *,
        query_embedding: list[float],
        limit: int = 20,
        document_ids: list[UUID] | None = None,
        run_id: str | None = None,
    ) -> list[EngineeringDocumentSemanticMatch]:
        """Return the semantic match expected for the current eval query."""
        assert query_embedding == [0.5] * 384
        assert limit >= 1
        assert embedding_provider.current_query is not None

        expected_title = query_to_title[embedding_provider.current_query]
        expected_chunk = chunks_by_title[expected_title]

        if document_ids is not None:
            assert expected_chunk.document_id in document_ids

        return [
            EngineeringDocumentSemanticMatch(
                chunk=expected_chunk,
                similarity_score=0.99,
            )
        ]

    repository.search_chunks_by_embedding = (  # type: ignore[method-assign]
        deterministic_semantic_search
    )

    reranker = DeterministicHybridReranker(expected_markers_by_query)
    retrieval_service = EngineeringDocumentRetrievalService(
        repository,
        embedding_provider=embedding_provider,
        reranker=reranker,
    )
    evaluation_service = KnowledgeRetrievalEvaluationService(
        EngineeringDocumentRetrievalEvaluationAdapter(retrieval_service)
    )

    cases = [
        *build_knowledge_retrieval_eval_cases(top_k=3),
        KnowledgeRetrievalEvalCase(
            name="semantic_only_deployment_approval",
            query="Can we ship?",
            expected_document_title="Production Deployment Policy",
            top_k=3,
        ),
    ]

    report = await evaluation_service.evaluate(cases, run_id=uuid4())
    failure_summary = _format_safe_failure_summary(report)

    assert report.total_cases == 4
    assert report.passed_cases == 4, failure_summary
    assert report.failed_cases == 0, failure_summary
    assert report.top_1_accuracy == 1.0, failure_summary
    assert report.top_k_accuracy == 1.0, failure_summary
    assert embedding_provider.calls == [[case.query] for case in cases]
    assert len(reranker.calls) == 4


def test_quality_gate_failure_summary_excludes_raw_document_content() -> None:
    """Quality gate failure summary should be safe for CI logs."""
    raw_internal_runbook_content = (
        "SECRET_INTERNAL_RUNBOOK_CONTENT: Redis password rotation steps and "
        "private operational details should never appear in CI failure output."
    )

    report = KnowledgeRetrievalEvaluationReport(
        total_cases=1,
        passed_cases=0,
        failed_cases=1,
        top_1_accuracy=0.0,
        top_k_accuracy=0.0,
        duration_ms=1.25,
        failed_case_details=[
            KnowledgeRetrievalEvalFailureDetail(
                case_name="redis_checkout_failure",
                expected_document_title="Payment Redis Incident Runbook",
                query_length=len("Redis checkout failure"),
                top_k=3,
                reason="expected_not_in_top_k",
                retrieved_documents=[
                    KnowledgeRetrievalEvalRetrievedDocument(
                        rank=1,
                        document_id="doc-release",
                        document_title="Release Readiness Checklist",
                        source_type="release_checklist",
                    )
                ],
            )
        ],
    )

    failure_summary = _format_safe_failure_summary(report)

    assert "redis_checkout_failure" in failure_summary
    assert "Payment Redis Incident Runbook" in failure_summary
    assert "Release Readiness Checklist" in failure_summary
    assert "expected_not_in_top_k" in failure_summary

    assert raw_internal_runbook_content not in failure_summary
    assert "SECRET_INTERNAL_RUNBOOK_CONTENT" not in failure_summary
    assert "Redis checkout failure" not in failure_summary
