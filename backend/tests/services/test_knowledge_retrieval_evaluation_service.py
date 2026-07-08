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
