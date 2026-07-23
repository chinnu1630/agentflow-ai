"""Deterministic groundedness evaluation for release-risk synthesis."""

from __future__ import annotations

import re
import time
from typing import Final, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.core.logging import get_logger
from app.schemas.llm_risk_synthesis import (
    ClaudeReleaseRiskReport,
    SynthesisEvidenceSource,
)
from app.schemas.risk import ReleaseRunRiskResponse
from app.services.release_risk_synthesis_evidence_index import (
    ReleaseRiskSynthesisEvidenceIndex,
)

logger = get_logger(__name__)

GroundednessFailureReason = Literal[
    "unverified_citation",
    "unsupported_supporting_fact",
    "unsupported_risk_explanation",
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


class ReleaseRiskSynthesisGroundednessConfig(BaseModel):
    """Thresholds for deterministic lexical groundedness evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_supporting_fact_overlap: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )
    minimum_explanation_overlap: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
    )


class ReleaseRiskSynthesisGroundednessFailureDetail(BaseModel):
    """Safe metadata describing one synthesis groundedness failure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    risk_rank: int = Field(ge=1)
    citation_index: int | None = Field(default=None, ge=1)
    source: SynthesisEvidenceSource | None = None
    source_id: str | None = None
    reason: GroundednessFailureReason
    overlap_score: float = Field(ge=0.0, le=1.0)
    claim_token_count: int = Field(ge=0)
    evidence_token_count: int = Field(ge=0)


class ReleaseRiskSynthesisGroundednessReport(BaseModel):
    """Aggregate deterministic synthesis-groundedness metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_risks: int = Field(ge=0)
    grounded_risks: int = Field(ge=0)
    total_citations: int = Field(ge=0)
    verified_citations: int = Field(ge=0)
    grounded_citations: int = Field(ge=0)
    citation_validity_rate: float = Field(ge=0.0, le=1.0)
    citation_groundedness_rate: float = Field(ge=0.0, le=1.0)
    risk_groundedness_rate: float = Field(ge=0.0, le=1.0)
    passed: bool
    duration_ms: float = Field(ge=0.0)
    failure_details: list[
        ReleaseRiskSynthesisGroundednessFailureDetail
    ] = Field(default_factory=list)


class ReleaseRiskSynthesisGroundednessEvaluationService:
    """Evaluate synthesized claims against citation-specific trusted evidence."""

    def __init__(
        self,
        *,
        evidence_index: ReleaseRiskSynthesisEvidenceIndex | None = None,
        config: ReleaseRiskSynthesisGroundednessConfig | None = None,
    ) -> None:
        """Initialize the evaluator with trusted evidence and thresholds."""
        self._evidence_index = (
            evidence_index or ReleaseRiskSynthesisEvidenceIndex()
        )
        self._config = config or ReleaseRiskSynthesisGroundednessConfig()

    def evaluate(
        self,
        *,
        report: ClaudeReleaseRiskReport,
        release_risk: ReleaseRunRiskResponse,
        run_id: UUID | str | None = None,
    ) -> ReleaseRiskSynthesisGroundednessReport:
        """Return deterministic citation and claim-groundedness metrics.

        The lexical containment score measures how many meaningful claim terms
        appear in the cited trusted evidence. This is intentionally deterministic
        and inexpensive for CI quality gates; it does not replace semantic or
        human evaluation.

        Complexity:
            Time: O(e + c + t), where e is trusted evidence text, c is citation
            count, and t is the total number of claim and evidence tokens.
            Space: O(e + t) for the evidence index and normalized token sets.
        """
        started_at = time.perf_counter()
        trusted_evidence = self._evidence_index.build(release_risk)

        total_citations = 0
        verified_citations = 0
        grounded_citations = 0
        grounded_risks = 0
        failures: list[
            ReleaseRiskSynthesisGroundednessFailureDetail
        ] = []

        for risk in report.risks:
            risk_failed = False
            cited_evidence_texts: list[str] = []

            for citation_index, citation in enumerate(
                risk.evidence,
                start=1,
            ):
                total_citations += 1
                citation_key = (citation.source, citation.source_id)
                evidence_text = trusted_evidence.get(citation_key)

                if evidence_text is None:
                    risk_failed = True
                    failures.append(
                        ReleaseRiskSynthesisGroundednessFailureDetail(
                            risk_rank=risk.rank,
                            citation_index=citation_index,
                            source=citation.source,
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
                    risk_failed = True
                    failures.append(
                        ReleaseRiskSynthesisGroundednessFailureDetail(
                            risk_rank=risk.rank,
                            citation_index=citation_index,
                            source=citation.source,
                            source_id=citation.source_id,
                            reason="unsupported_supporting_fact",
                            overlap_score=score,
                            claim_token_count=claim_count,
                            evidence_token_count=evidence_count,
                        )
                    )

            if cited_evidence_texts:
                explanation_score, claim_count, evidence_count = (
                    self._containment_score(
                        claim=risk.explanation,
                        evidence=" ".join(cited_evidence_texts),
                    )
                )

                if (
                    explanation_score
                    < self._config.minimum_explanation_overlap
                ):
                    risk_failed = True
                    failures.append(
                        ReleaseRiskSynthesisGroundednessFailureDetail(
                            risk_rank=risk.rank,
                            reason="unsupported_risk_explanation",
                            overlap_score=explanation_score,
                            claim_token_count=claim_count,
                            evidence_token_count=evidence_count,
                        )
                    )

            if not risk_failed:
                grounded_risks += 1

        total_risks = len(report.risks)
        duration_ms = round(
            (time.perf_counter() - started_at) * 1000,
            3,
        )

        evaluation_report = ReleaseRiskSynthesisGroundednessReport(
            total_risks=total_risks,
            grounded_risks=grounded_risks,
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
            risk_groundedness_rate=self._safe_ratio(
                grounded_risks,
                total_risks,
            ),
            passed=not failures,
            duration_ms=duration_ms,
            failure_details=failures,
        )

        logger.info(
            "release_risk_synthesis_groundedness_evaluation_completed",
            extra={
                "run_id": str(run_id) if run_id is not None else None,
                "total_risks": evaluation_report.total_risks,
                "grounded_risks": evaluation_report.grounded_risks,
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
                "risk_groundedness_rate": (
                    evaluation_report.risk_groundedness_rate
                ),
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
