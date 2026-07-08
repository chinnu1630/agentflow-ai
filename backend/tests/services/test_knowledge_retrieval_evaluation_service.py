"""Tests for deterministic Knowledge Agent retrieval evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID, uuid4

import pytest

from app.services.knowledge_retrieval_evaluation_service import (
    KnowledgeRetrievalEvalCandidate,
    KnowledgeRetrievalEvalCase,
    KnowledgeRetrievalEvaluationError,
    KnowledgeRetrievalEvaluationService,
)


class FakeKnowledgeRetrievalEvalRetriever:
    """Fake retriever for deterministic evaluator unit tests."""

    def __init__(
        self,
        responses_by_case_name: dict[str, list[KnowledgeRetrievalEvalCandidate]],
    ) -> None:
        """Initialize fake responses keyed by eval case name."""
        self._responses_by_case_name = responses_by_case_name

    async def retrieve_for_evaluation(
        self,
        case: KnowledgeRetrievalEvalCase,
        *,
        run_id: UUID | None,
    ) -> Sequence[KnowledgeRetrievalEvalCandidate]:
        """Return fake candidates for an eval case."""
        return self._responses_by_case_name.get(case.name, [])


class FailingKnowledgeRetrievalEvalRetriever:
    """Fake retriever that simulates a retrieval dependency failure."""

    async def retrieve_for_evaluation(
        self,
        case: KnowledgeRetrievalEvalCase,
        *,
        run_id: UUID | None,
    ) -> Sequence[KnowledgeRetrievalEvalCandidate]:
        """Raise a controlled retrieval evaluation error."""
        raise KnowledgeRetrievalEvaluationError("retrieval dependency failed")


@pytest.mark.anyio
async def test_expected_top_document_passes() -> None:
    """Evaluator should pass when expected document is ranked first."""
    case = KnowledgeRetrievalEvalCase(
        name="redis_checkout_failure",
        query="Redis checkout failure",
        expected_document_title="Payment Redis Incident Runbook",
        top_k=3,
    )

    retriever = FakeKnowledgeRetrievalEvalRetriever(
        responses_by_case_name={
            "redis_checkout_failure": [
                KnowledgeRetrievalEvalCandidate(
                    document_id="doc-redis",
                    document_title="Payment Redis Incident Runbook",
                    source_type="incident_postmortem",
                ),
                KnowledgeRetrievalEvalCandidate(
                    document_id="doc-release",
                    document_title="Release Readiness Checklist",
                    source_type="checklist",
                ),
            ]
        }
    )

    service = KnowledgeRetrievalEvaluationService(retriever=retriever)

    report = await service.evaluate([case], run_id=uuid4())

    assert report.total_cases == 1
    assert report.passed_cases == 1
    assert report.failed_cases == 0
    assert report.top_1_accuracy == 1.0
    assert report.top_k_accuracy == 1.0
    assert report.failed_case_details == []


@pytest.mark.anyio
async def test_missing_expected_document_fails() -> None:
    """Evaluator should fail when expected document is absent from top-k."""
    case = KnowledgeRetrievalEvalCase(
        name="release_approval_checklist",
        query="release approval checklist",
        expected_document_title="Release Readiness Checklist",
        top_k=2,
    )

    retriever = FakeKnowledgeRetrievalEvalRetriever(
        responses_by_case_name={
            "release_approval_checklist": [
                KnowledgeRetrievalEvalCandidate(
                    document_id="doc-redis",
                    document_title="Payment Redis Incident Runbook",
                    source_type="incident_postmortem",
                ),
                KnowledgeRetrievalEvalCandidate(
                    document_id="doc-rollback",
                    document_title="Payment Rollback Procedure",
                    source_type="runbook",
                ),
            ]
        }
    )

    service = KnowledgeRetrievalEvaluationService(retriever=retriever)

    report = await service.evaluate([case], run_id=uuid4())

    assert report.total_cases == 1
    assert report.passed_cases == 0
    assert report.failed_cases == 1
    assert report.top_1_accuracy == 0.0
    assert report.top_k_accuracy == 0.0

    failure = report.failed_case_details[0]
    assert failure.case_name == "release_approval_checklist"
    assert failure.expected_document_title == "Release Readiness Checklist"
    assert failure.reason == "expected_not_in_top_k"
    assert failure.query_length == len("release approval checklist")
    assert failure.retrieved_documents[0].document_title == (
        "Payment Redis Incident Runbook"
    )


@pytest.mark.anyio
async def test_empty_retrieval_fails_safely() -> None:
    """Evaluator should fail safely when retrieval returns no documents."""
    case = KnowledgeRetrievalEvalCase(
        name="rollback_after_payment_outage",
        query="rollback after payment outage",
        expected_document_title="Payment Rollback Procedure",
        top_k=3,
    )

    retriever = FakeKnowledgeRetrievalEvalRetriever(responses_by_case_name={})
    service = KnowledgeRetrievalEvaluationService(retriever=retriever)

    report = await service.evaluate([case], run_id=uuid4())

    assert report.total_cases == 1
    assert report.passed_cases == 0
    assert report.failed_cases == 1
    assert report.top_1_accuracy == 0.0
    assert report.top_k_accuracy == 0.0

    failure = report.failed_case_details[0]
    assert failure.reason == "empty_results"
    assert failure.retrieved_documents == []


@pytest.mark.anyio
async def test_metric_calculations_are_correct() -> None:
    """Evaluator should calculate top-1 and top-k accuracy correctly."""
    cases = [
        KnowledgeRetrievalEvalCase(
            name="case_top_1_pass",
            query="Redis checkout failure",
            expected_document_title="Payment Redis Incident Runbook",
            top_k=3,
        ),
        KnowledgeRetrievalEvalCase(
            name="case_top_k_pass",
            query="rollback after payment outage",
            expected_document_title="Payment Rollback Procedure",
            top_k=3,
        ),
        KnowledgeRetrievalEvalCase(
            name="case_fail",
            query="release approval checklist",
            expected_document_title="Release Readiness Checklist",
            top_k=3,
        ),
    ]

    retriever = FakeKnowledgeRetrievalEvalRetriever(
        responses_by_case_name={
            "case_top_1_pass": [
                KnowledgeRetrievalEvalCandidate(
                    document_id="doc-redis",
                    document_title="Payment Redis Incident Runbook",
                    source_type="incident_postmortem",
                )
            ],
            "case_top_k_pass": [
                KnowledgeRetrievalEvalCandidate(
                    document_id="doc-other",
                    document_title="Payment Redis Incident Runbook",
                    source_type="incident_postmortem",
                ),
                KnowledgeRetrievalEvalCandidate(
                    document_id="doc-rollback",
                    document_title="Payment Rollback Procedure",
                    source_type="runbook",
                ),
            ],
            "case_fail": [
                KnowledgeRetrievalEvalCandidate(
                    document_id="doc-redis",
                    document_title="Payment Redis Incident Runbook",
                    source_type="incident_postmortem",
                )
            ],
        }
    )

    service = KnowledgeRetrievalEvaluationService(retriever=retriever)

    report = await service.evaluate(cases, run_id=uuid4())

    assert report.total_cases == 3
    assert report.passed_cases == 2
    assert report.failed_cases == 1
    assert report.top_1_accuracy == pytest.approx(1 / 3)
    assert report.top_k_accuracy == pytest.approx(2 / 3)
    assert len(report.failed_case_details) == 1
    assert report.failed_case_details[0].case_name == "case_fail"


@pytest.mark.anyio
async def test_retrieval_error_fails_safely_without_raw_query() -> None:
    """Evaluator should convert controlled retrieval errors into safe failures."""
    case = KnowledgeRetrievalEvalCase(
        name="dependency_failure_case",
        query="Redis checkout failure",
        expected_document_title="Payment Redis Incident Runbook",
        top_k=3,
    )

    service = KnowledgeRetrievalEvaluationService(
        retriever=FailingKnowledgeRetrievalEvalRetriever()
    )

    report = await service.evaluate([case], run_id=uuid4())

    assert report.total_cases == 1
    assert report.passed_cases == 0
    assert report.failed_cases == 1

    failure = report.failed_case_details[0]
    assert failure.reason == "retrieval_error"
    assert failure.error_type == "KnowledgeRetrievalEvaluationError"
    assert failure.query_length == len("Redis checkout failure")

    failure_json = failure.model_dump_json()
    assert "Redis checkout failure" not in failure_json

from collections.abc import AsyncGenerator as _AsyncGenerator

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import app.models  # noqa: F401 - ensures SQLAlchemy models are registered
from app.db.base import Base
from app.models.engineering_document import EngineeringDocumentSourceType
from app.repositories.engineering_document_repository import EngineeringDocumentRepository
from app.services.document_chunker import DocumentChunkingConfig
from app.services.engineering_document_ingestion_service import (
    EngineeringDocumentIngestionRequest,
    EngineeringDocumentIngestionService,
)
from app.services.engineering_document_retrieval_service import (
    EngineeringDocumentRetrievalService,
)
from app.services.knowledge_retrieval_evaluation_service import (
    EngineeringDocumentRetrievalEvaluationAdapter,
)


@pytest_asyncio.fixture
async def evaluation_async_session() -> _AsyncGenerator[AsyncSession, None]:
    """Create an isolated SQLite session for real retrieval adapter tests."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with session_factory() as session:
        yield session

    await engine.dispose()


def _evaluation_ingestion_request(
    *,
    title: str,
    source_type: EngineeringDocumentSourceType,
    source_uri: str,
    raw_content: str,
) -> EngineeringDocumentIngestionRequest:
    """Build an engineering document ingestion request for eval tests."""
    return EngineeringDocumentIngestionRequest(
        title=title,
        source_type=source_type,
        source_uri=source_uri,
        raw_content=raw_content,
        metadata_json={"team": "platform"},
        chunking_config=DocumentChunkingConfig(
            max_tokens_per_chunk=40,
            overlap_tokens=5,
        ),
    )


async def _seed_retrieval_eval_documents(
    ingestion_service: EngineeringDocumentIngestionService,
) -> None:
    """Seed deterministic AgentFlow Knowledge retrieval eval documents."""
    documents = [
        _evaluation_ingestion_request(
            title="Payment Redis Incident Runbook",
            source_type=EngineeringDocumentSourceType.RUNBOOK,
            source_uri="docs/payment-redis-incident-runbook.md",
            raw_content=(
                "Redis checkout failure caused payment release risk. "
                "Redis latency increased during checkout. "
                "Use the payment rollback procedure when checkout failures spike."
            ),
        ),
        _evaluation_ingestion_request(
            title="Release Readiness Checklist",
            source_type=EngineeringDocumentSourceType.RELEASE_CHECKLIST,
            source_uri="docs/release-readiness-checklist.md",
            raw_content=(
                "Release approval checklist requires Jira P1 review, "
                "GitHub pull request approval, CI validation, rollback readiness, "
                "and engineering manager sign off."
            ),
        ),
        _evaluation_ingestion_request(
            title="Payment Rollback Procedure",
            source_type=EngineeringDocumentSourceType.RUNBOOK,
            source_uri="docs/payment-rollback-procedure.md",
            raw_content=(
                "Payment outage rollback procedure explains how to revert "
                "payment service deployments after checkout errors, failed releases, "
                "or customer-impacting payment incidents."
            ),
        ),
    ]

    for document in documents:
        await ingestion_service.ingest_document(document)


@pytest.mark.asyncio
async def test_real_engineering_document_retrieval_adapter_passes_eval_cases(
    evaluation_async_session: AsyncSession,
) -> None:
    """Real BM25 retrieval should pass deterministic Knowledge eval cases."""
    repository = EngineeringDocumentRepository(evaluation_async_session)
    ingestion_service = EngineeringDocumentIngestionService(repository)
    retrieval_service = EngineeringDocumentRetrievalService(repository)
    adapter = EngineeringDocumentRetrievalEvaluationAdapter(retrieval_service)
    evaluation_service = KnowledgeRetrievalEvaluationService(adapter)

    await _seed_retrieval_eval_documents(ingestion_service)

    cases = [
        KnowledgeRetrievalEvalCase(
            name="redis_checkout_failure",
            query="Redis checkout failure",
            expected_document_title="Payment Redis Incident Runbook",
            top_k=3,
        ),
        KnowledgeRetrievalEvalCase(
            name="release_approval_checklist",
            query="release approval checklist",
            expected_document_title="Release Readiness Checklist",
            top_k=3,
        ),
        KnowledgeRetrievalEvalCase(
            name="rollback_after_payment_outage",
            query="rollback after payment outage",
            expected_document_title="Payment Rollback Procedure",
            top_k=3,
        ),
    ]

    report = await evaluation_service.evaluate(cases, run_id=uuid4())

    assert report.total_cases == 3
    assert report.passed_cases == 3
    assert report.failed_cases == 0
    assert report.top_1_accuracy == 1.0
    assert report.top_k_accuracy == 1.0
    assert report.failed_case_details == []


@pytest.mark.asyncio
async def test_real_engineering_document_retrieval_adapter_returns_safe_metadata(
    evaluation_async_session: AsyncSession,
) -> None:
    """Adapter should expose safe metadata and drop raw retrieved content."""
    repository = EngineeringDocumentRepository(evaluation_async_session)
    ingestion_service = EngineeringDocumentIngestionService(repository)
    retrieval_service = EngineeringDocumentRetrievalService(repository)
    adapter = EngineeringDocumentRetrievalEvaluationAdapter(retrieval_service)

    await ingestion_service.ingest_document(
        _evaluation_ingestion_request(
            title="Payment Redis Incident Runbook",
            source_type=EngineeringDocumentSourceType.RUNBOOK,
            source_uri="docs/payment-redis-incident-runbook.md",
            raw_content="Redis checkout failure caused payment release risk.",
        )
    )

    candidates = await adapter.retrieve_for_evaluation(
        KnowledgeRetrievalEvalCase(
            name="safe_metadata_case",
            query="Redis checkout failure",
            expected_document_title="Payment Redis Incident Runbook",
            top_k=1,
        ),
        run_id=uuid4(),
    )

    assert len(candidates) == 1
    assert candidates[0].document_id is not None
    assert candidates[0].document_title == "Payment Redis Incident Runbook"
    assert candidates[0].source_type == "runbook"
    assert not hasattr(candidates[0], "content")


@pytest.mark.asyncio
async def test_real_engineering_document_retrieval_adapter_invalid_source_type_fails_safely(
    evaluation_async_session: AsyncSession,
) -> None:
    """Invalid source type should become a safe evaluation failure."""
    repository = EngineeringDocumentRepository(evaluation_async_session)
    retrieval_service = EngineeringDocumentRetrievalService(repository)
    adapter = EngineeringDocumentRetrievalEvaluationAdapter(retrieval_service)
    evaluation_service = KnowledgeRetrievalEvaluationService(adapter)

    report = await evaluation_service.evaluate(
        [
            KnowledgeRetrievalEvalCase(
                name="invalid_source_type_case",
                query="Redis checkout failure",
                expected_document_title="Payment Redis Incident Runbook",
                top_k=3,
                source_types=["not_a_valid_source_type"],
            )
        ],
        run_id=uuid4(),
    )

    assert report.total_cases == 1
    assert report.passed_cases == 0
    assert report.failed_cases == 1
    assert report.top_1_accuracy == 0.0
    assert report.top_k_accuracy == 0.0

    failure = report.failed_case_details[0]
    assert failure.reason == "retrieval_error"
    assert failure.error_type == "KnowledgeRetrievalEvaluationError"
    assert failure.retrieved_documents == []


@pytest.mark.anyio
async def test_evaluation_report_includes_duration_ms() -> None:
    """Evaluation report should include safe duration metadata."""
    case = KnowledgeRetrievalEvalCase(
        name="redis_checkout_failure",
        query="Redis checkout failure",
        expected_document_title="Payment Redis Incident Runbook",
        top_k=3,
    )

    retriever = FakeKnowledgeRetrievalEvalRetriever(
        responses_by_case_name={
            "redis_checkout_failure": [
                KnowledgeRetrievalEvalCandidate(
                    document_id="doc-redis",
                    document_title="Payment Redis Incident Runbook",
                    source_type="incident_postmortem",
                )
            ]
        }
    )

    service = KnowledgeRetrievalEvaluationService(retriever=retriever)

    report = await service.evaluate([case], run_id=uuid4())

    assert report.duration_ms >= 0.0
    assert report.total_cases == 1
    assert report.passed_cases == 1
