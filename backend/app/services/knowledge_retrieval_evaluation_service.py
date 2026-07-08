"""Deterministic evaluation service for Knowledge Agent retrieval quality.

This module measures whether the Knowledge Agent returns the expected
engineering document in the top result or within top-k results.

It intentionally avoids embeddings, LLM-as-judge, RAGAS, rerankers, and
raw document content so evaluation remains deterministic, cheap, and safe.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.core.logging import get_logger

logger = get_logger(__name__)


FailureReason = Literal[
    "empty_results",
    "expected_not_in_top_k",
    "retrieval_error",
]


class KnowledgeRetrievalEvaluationError(Exception):
    """Raised when the retrieval dependency fails during evaluation."""


class KnowledgeRetrievalEvalCase(BaseModel):
    """A deterministic retrieval evaluation case."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    expected_document_title: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=20)
    source_types: list[str] | None = None


class KnowledgeRetrievalEvalCandidate(BaseModel):
    """Safe metadata for one retrieved document candidate.

    This intentionally does not include raw document content or retrieved chunks.
    """

    model_config = ConfigDict(frozen=True)

    document_id: str | None = None
    document_title: str = Field(..., min_length=1)
    source_type: str | None = None


class KnowledgeRetrievalEvalRetrievedDocument(BaseModel):
    """Safe ranked document metadata included in failed eval details."""

    model_config = ConfigDict(frozen=True)

    rank: int = Field(..., ge=1)
    document_id: str | None = None
    document_title: str
    source_type: str | None = None


class KnowledgeRetrievalEvalFailureDetail(BaseModel):
    """Safe failure detail for one failed retrieval evaluation case."""

    model_config = ConfigDict(frozen=True)

    case_name: str
    expected_document_title: str
    query_length: int = Field(..., ge=0)
    top_k: int = Field(..., ge=1)
    reason: FailureReason
    error_type: str | None = None
    retrieved_documents: list[KnowledgeRetrievalEvalRetrievedDocument] = Field(
        default_factory=list
    )


class KnowledgeRetrievalEvaluationReport(BaseModel):
    """Aggregate retrieval evaluation report."""

    model_config = ConfigDict(frozen=True)

    total_cases: int = Field(..., ge=0)
    passed_cases: int = Field(..., ge=0)
    failed_cases: int = Field(..., ge=0)
    top_1_accuracy: float = Field(..., ge=0.0, le=1.0)
    top_k_accuracy: float = Field(..., ge=0.0, le=1.0)
    failed_case_details: list[KnowledgeRetrievalEvalFailureDetail] = Field(
        default_factory=list
    )


class KnowledgeRetrievalEvalRetriever(Protocol):
    """Protocol for retrieval services that can be evaluated."""

    async def retrieve_for_evaluation(
        self,
        case: KnowledgeRetrievalEvalCase,
        *,
        run_id: UUID | None,
    ) -> Sequence[KnowledgeRetrievalEvalCandidate]:
        """Return safe document candidates for one evaluation case."""


class _CaseEvaluationOutcome(BaseModel):
    """Internal outcome for one evaluation case."""

    model_config = ConfigDict(frozen=True)

    top_1_passed: bool
    top_k_passed: bool
    failure_detail: KnowledgeRetrievalEvalFailureDetail | None = None


class KnowledgeRetrievalEvaluationService:
    """Evaluates Knowledge Agent retrieval quality using deterministic cases."""

    def __init__(self, retriever: KnowledgeRetrievalEvalRetriever) -> None:
        """Initialize the evaluator with a retrieval adapter."""
        self._retriever = retriever

    async def evaluate(
        self,
        cases: Sequence[KnowledgeRetrievalEvalCase],
        *,
        run_id: UUID | None = None,
    ) -> KnowledgeRetrievalEvaluationReport:
        """Evaluate multiple retrieval cases and return aggregate metrics."""
        outcomes: list[_CaseEvaluationOutcome] = []

        for case in cases:
            outcome = await self._evaluate_case(case=case, run_id=run_id)
            outcomes.append(outcome)

        total_cases = len(outcomes)
        top_1_hits = sum(1 for outcome in outcomes if outcome.top_1_passed)
        top_k_hits = sum(1 for outcome in outcomes if outcome.top_k_passed)
        failed_details = [
            outcome.failure_detail
            for outcome in outcomes
            if outcome.failure_detail is not None
        ]

        report = KnowledgeRetrievalEvaluationReport(
            total_cases=total_cases,
            passed_cases=top_k_hits,
            failed_cases=total_cases - top_k_hits,
            top_1_accuracy=self._safe_ratio(top_1_hits, total_cases),
            top_k_accuracy=self._safe_ratio(top_k_hits, total_cases),
            failed_case_details=failed_details,
        )

        logger.info(
            "knowledge_retrieval_evaluation_completed",
            extra={
                "run_id": str(run_id) if run_id else None,
                "total_cases": report.total_cases,
                "passed_cases": report.passed_cases,
                "failed_cases": report.failed_cases,
                "top_1_accuracy": report.top_1_accuracy,
                "top_k_accuracy": report.top_k_accuracy,
            },
        )

        return report

    async def _evaluate_case(
        self,
        case: KnowledgeRetrievalEvalCase,
        *,
        run_id: UUID | None,
    ) -> _CaseEvaluationOutcome:
        """Evaluate one retrieval case safely."""
        try:
            candidates = await self._retriever.retrieve_for_evaluation(
                case,
                run_id=run_id,
            )
        except KnowledgeRetrievalEvaluationError as exc:
            logger.warning(
                "knowledge_retrieval_evaluation_case_failed",
                extra={
                    "run_id": str(run_id) if run_id else None,
                    "case_name": case.name,
                    "query_length": len(case.query),
                    "reason": "retrieval_error",
                    "error_type": type(exc).__name__,
                },
            )

            return _CaseEvaluationOutcome(
                top_1_passed=False,
                top_k_passed=False,
                failure_detail=KnowledgeRetrievalEvalFailureDetail(
                    case_name=case.name,
                    expected_document_title=case.expected_document_title,
                    query_length=len(case.query),
                    top_k=case.top_k,
                    reason="retrieval_error",
                    error_type=type(exc).__name__,
                    retrieved_documents=[],
                ),
            )

        ranked_candidates = list(candidates)[: case.top_k]

        if not ranked_candidates:
            return _CaseEvaluationOutcome(
                top_1_passed=False,
                top_k_passed=False,
                failure_detail=self._build_failure_detail(
                    case=case,
                    reason="empty_results",
                    candidates=[],
                ),
            )

        expected_title = self._normalize_title(case.expected_document_title)
        matched_rank = self._find_matched_rank(
            expected_title=expected_title,
            candidates=ranked_candidates,
        )

        top_1_passed = matched_rank == 1
        top_k_passed = matched_rank is not None

        if not top_k_passed:
            return _CaseEvaluationOutcome(
                top_1_passed=False,
                top_k_passed=False,
                failure_detail=self._build_failure_detail(
                    case=case,
                    reason="expected_not_in_top_k",
                    candidates=ranked_candidates,
                ),
            )

        return _CaseEvaluationOutcome(
            top_1_passed=top_1_passed,
            top_k_passed=True,
            failure_detail=None,
        )

    @staticmethod
    def _find_matched_rank(
        *,
        expected_title: str,
        candidates: Sequence[KnowledgeRetrievalEvalCandidate],
    ) -> int | None:
        """Return the 1-based rank where expected title appears."""
        for index, candidate in enumerate(candidates, start=1):
            candidate_title = KnowledgeRetrievalEvaluationService._normalize_title(
                candidate.document_title
            )
            if candidate_title == expected_title:
                return index

        return None

    @staticmethod
    def _build_failure_detail(
        *,
        case: KnowledgeRetrievalEvalCase,
        reason: FailureReason,
        candidates: Sequence[KnowledgeRetrievalEvalCandidate],
    ) -> KnowledgeRetrievalEvalFailureDetail:
        """Build safe failure metadata without raw document content."""
        retrieved_documents = [
            KnowledgeRetrievalEvalRetrievedDocument(
                rank=index,
                document_id=candidate.document_id,
                document_title=candidate.document_title,
                source_type=candidate.source_type,
            )
            for index, candidate in enumerate(candidates, start=1)
        ]

        return KnowledgeRetrievalEvalFailureDetail(
            case_name=case.name,
            expected_document_title=case.expected_document_title,
            query_length=len(case.query),
            top_k=case.top_k,
            reason=reason,
            retrieved_documents=retrieved_documents,
        )

    @staticmethod
    def _normalize_title(title: str) -> str:
        """Normalize document titles for deterministic matching."""
        return " ".join(title.strip().lower().split())

    @staticmethod
    def _safe_ratio(numerator: int, denominator: int) -> float:
        """Return a safe ratio for metric calculations."""
        if denominator == 0:
            return 0.0

        return numerator / denominator
