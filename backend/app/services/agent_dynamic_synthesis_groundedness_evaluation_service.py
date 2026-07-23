"""Deterministic groundedness evaluation for dynamic agent answers."""

from __future__ import annotations

import re
import time
from typing import Final, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.core.logging import get_logger
from app.schemas.agent_dynamic_synthesis import AgentDynamicAnswer
from app.schemas.agent_execution_result import AgentExecutionResult
from app.services.agent_dynamic_synthesis_evidence_index import (
    AgentDynamicSynthesisEvidenceIndex,
)

logger = get_logger(__name__)

DynamicGroundednessFailureReason = Literal[
    "missing_citation",
    "unverified_citation",
    "unsupported_supporting_fact",
    "unsupported_answer",
]

_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-zA-Z0-9_]+")
_STOP_WORDS: Final[frozenset[str]] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "because",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)


class AgentDynamicSynthesisGroundednessConfig(BaseModel):
    """Thresholds for deterministic dynamic-answer groundedness."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_supporting_fact_overlap: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )
    minimum_answer_overlap: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
    )


class AgentDynamicSynthesisGroundednessFailureDetail(BaseModel):
    """Safe metadata describing one dynamic groundedness failure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    citation_index: int | None = Field(default=None, ge=1)
    source_type: str | None = None
    source_id: str | None = None
    reason: DynamicGroundednessFailureReason
    overlap_score: float = Field(ge=0.0, le=1.0)
    claim_token_count: int = Field(ge=0)
    evidence_token_count: int = Field(ge=0)


class AgentDynamicSynthesisGroundednessReport(BaseModel):
    """Aggregate deterministic groundedness metrics for one answer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_citations: int = Field(ge=0)
    verified_citations: int = Field(ge=0)
    grounded_citations: int = Field(ge=0)
    citation_validity_rate: float = Field(ge=0.0, le=1.0)
    citation_groundedness_rate: float = Field(ge=0.0, le=1.0)
    answer_overlap_score: float = Field(ge=0.0, le=1.0)
    answer_grounded: bool
    passed: bool
    duration_ms: float = Field(ge=0.0)
    failure_details: list[
        AgentDynamicSynthesisGroundednessFailureDetail
    ] = Field(default_factory=list)


class AgentDynamicSynthesisGroundednessEvaluationService:
    """Evaluate dynamic claims against citation-specific tool evidence."""

    def __init__(
        self,
        *,
        evidence_index: AgentDynamicSynthesisEvidenceIndex | None = None,
        config: AgentDynamicSynthesisGroundednessConfig | None = None,
    ) -> None:
        """Initialize the evaluator with trusted evidence and thresholds."""
        self._evidence_index = (
            evidence_index or AgentDynamicSynthesisEvidenceIndex()
        )
        self._config = (
            config or AgentDynamicSynthesisGroundednessConfig()
        )

    def evaluate(
        self,
        *,
        answer: AgentDynamicAnswer,
        execution_result: AgentExecutionResult,
        run_id: UUID | str | None = None,
    ) -> AgentDynamicSynthesisGroundednessReport:
        """Return deterministic citation and answer-groundedness metrics.

        Lexical containment is deterministic and inexpensive for CI quality
        gates. It complements semantic and human evaluation rather than
        replacing them.

        Complexity:
            Time: O(e + c + t), where e is trusted evidence, c is citation
            count, and t is the total number of normalized tokens.
            Space: O(e + t) for evidence text and token sets.
        """
        started_at = time.perf_counter()
        trusted_evidence = self._evidence_index.build(execution_result)

        total_citations = len(answer.citations)
        verified_citations = 0
        grounded_citations = 0
        cited_evidence_texts: list[str] = []
        failures: list[
            AgentDynamicSynthesisGroundednessFailureDetail
        ] = []

        if trusted_evidence and not answer.citations:
            failures.append(
                AgentDynamicSynthesisGroundednessFailureDetail(
                    reason="missing_citation",
                    overlap_score=0.0,
                    claim_token_count=len(self._tokenize(answer.answer)),
                    evidence_token_count=0,
                )
            )

        for citation_index, citation in enumerate(
            answer.citations,
            start=1,
        ):
            citation_key = (
                citation.source_type,
                citation.source_id,
            )
            evidence_text = trusted_evidence.get(citation_key)

            if evidence_text is None:
                failures.append(
                    AgentDynamicSynthesisGroundednessFailureDetail(
                        citation_index=citation_index,
                        source_type=citation.source_type,
                        source_id=citation.source_id,
                        reason="unverified_citation",
                        overlap_score=0.0,
                        claim_token_count=len(
                            self._tokenize(citation.supporting_fact)
                        ),
                        evidence_token_count=0,
                    )
                )
                continue

            verified_citations += 1
            cited_evidence_texts.append(evidence_text)

            score, claim_count, evidence_count = self._containment_score(
                claim=citation.supporting_fact,
                evidence=evidence_text,
            )

            if (
                score
                >= self._config.minimum_supporting_fact_overlap
            ):
                grounded_citations += 1
            else:
                failures.append(
                    AgentDynamicSynthesisGroundednessFailureDetail(
                        citation_index=citation_index,
                        source_type=citation.source_type,
                        source_id=citation.source_id,
                        reason="unsupported_supporting_fact",
                        overlap_score=score,
                        claim_token_count=claim_count,
                        evidence_token_count=evidence_count,
                    )
                )

        if cited_evidence_texts:
            answer_score, answer_claim_count, answer_evidence_count = (
                self._containment_score(
                    claim=answer.answer,
                    evidence=" ".join(cited_evidence_texts),
                )
            )
            answer_grounded = (
                answer_score >= self._config.minimum_answer_overlap
            )

            if not answer_grounded:
                failures.append(
                    AgentDynamicSynthesisGroundednessFailureDetail(
                        reason="unsupported_answer",
                        overlap_score=answer_score,
                        claim_token_count=answer_claim_count,
                        evidence_token_count=answer_evidence_count,
                    )
                )
        else:
            answer_score = 1.0 if not trusted_evidence else 0.0
            answer_grounded = not trusted_evidence

        duration_ms = round(
            (time.perf_counter() - started_at) * 1000,
            3,
        )

        evaluation_report = AgentDynamicSynthesisGroundednessReport(
            total_citations=total_citations,
            verified_citations=verified_citations,
            grounded_citations=grounded_citations,
            citation_validity_rate=self._safe_ratio(
                verified_citations,
                total_citations,
            ),
            citation_groundedness_rate=self._safe_ratio(
                grounded_citations,
                total_citations,
            ),
            answer_overlap_score=answer_score,
            answer_grounded=answer_grounded,
            passed=not failures,
            duration_ms=duration_ms,
            failure_details=failures,
        )

        logger.info(
            "agent_dynamic_synthesis_groundedness_evaluation_completed",
            extra={
                "run_id": str(run_id) if run_id is not None else None,
                "total_citations": evaluation_report.total_citations,
                "verified_citations": (
                    evaluation_report.verified_citations
                ),
                "grounded_citations": (
                    evaluation_report.grounded_citations
                ),
                "citation_validity_rate": (
                    evaluation_report.citation_validity_rate
                ),
                "citation_groundedness_rate": (
                    evaluation_report.citation_groundedness_rate
                ),
                "answer_overlap_score": (
                    evaluation_report.answer_overlap_score
                ),
                "answer_grounded": evaluation_report.answer_grounded,
                "passed": evaluation_report.passed,
                "duration_ms": evaluation_report.duration_ms,
            },
        )

        return evaluation_report

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Return meaningful normalized terms for lexical comparison."""
        return {
            token
            for token in _TOKEN_PATTERN.findall(text.casefold())
            if token not in _STOP_WORDS
        }

    @classmethod
    def _containment_score(
        cls,
        *,
        claim: str,
        evidence: str,
    ) -> tuple[float, int, int]:
        """Return claim-term containment within trusted evidence."""
        claim_tokens = cls._tokenize(claim)
        evidence_tokens = cls._tokenize(evidence)

        if not claim_tokens:
            return 0.0, 0, len(evidence_tokens)

        score = len(claim_tokens & evidence_tokens) / len(claim_tokens)
        return round(score, 4), len(claim_tokens), len(evidence_tokens)

    @staticmethod
    def _safe_ratio(numerator: int, denominator: int) -> float:
        """Return one for an empty valid set, otherwise the normal ratio."""
        if denominator == 0:
            return 1.0

        return numerator / denominator
