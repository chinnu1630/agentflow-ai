"""Strict structured-output schemas for Claude release-risk synthesis."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.risk_enums import (
    RiskSeverityResponse,
    RiskSummaryActionResponse,
)


class SynthesisEvidenceSource(StrEnum):
    """Trusted evidence sources allowed in a synthesized risk report."""

    GITHUB_PULL_REQUEST = "github_pull_request"
    JIRA_ISSUE = "jira_issue"
    ENGINEERING_DOCUMENT = "engineering_document"
    DETERMINISTIC_RISK_RULE = "deterministic_risk_rule"


class SynthesisEvidenceCitation(BaseModel):
    """One trusted evidence reference supporting a synthesized risk."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    source: SynthesisEvidenceSource
    source_id: str = Field(min_length=1, max_length=255)
    title: str = Field(min_length=1, max_length=500)
    source_url: str | None = Field(default=None, max_length=2_000)
    supporting_fact: str = Field(
        min_length=1,
        max_length=1_000,
        description=(
            "Concise evidence-grounded fact, not hidden reasoning or "
            "chain-of-thought."
        ),
    )


class SynthesizedReleaseRisk(BaseModel):
    """One ranked release risk produced from trusted AgentFlow evidence."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    rank: int = Field(ge=1, le=20)
    title: str = Field(min_length=1, max_length=500)
    severity: RiskSeverityResponse
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(
        min_length=1,
        max_length=2_000,
        description="Concise evidence-based explanation of the release risk.",
    )
    evidence: list[SynthesisEvidenceCitation] = Field(
        min_length=1,
        max_length=10,
    )
    mitigations: list[str] = Field(
        min_length=1,
        max_length=10,
    )

    @field_validator("mitigations")
    @classmethod
    def validate_mitigations(cls, values: list[str]) -> list[str]:
        """Reject blank or duplicate mitigation steps."""
        normalized_values = [value.strip() for value in values]

        if any(not value for value in normalized_values):
            raise ValueError("mitigations must not contain blank values")

        if len(set(normalized_values)) != len(normalized_values):
            raise ValueError("mitigations must not contain duplicate values")

        return normalized_values


class ClaudeReleaseRiskReport(BaseModel):
    """Validated Claude synthesis result for one release assessment."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    schema_version: Literal["claude_release_risk_report_v1"] = (
        "claude_release_risk_report_v1"
    )
    recommendation: RiskSummaryActionResponse
    confidence: float = Field(ge=0.0, le=1.0)
    executive_summary: str = Field(min_length=1, max_length=3_000)
    risks: list[SynthesizedReleaseRisk] = Field(
        default_factory=list,
        max_length=20,
    )
    missing_information: list[str] = Field(
        default_factory=list,
        max_length=20,
    )
    degraded_sources: list[str] = Field(
        default_factory=list,
        max_length=10,
    )
    requires_human_review: bool

    @field_validator("missing_information", "degraded_sources")
    @classmethod
    def validate_string_lists(cls, values: list[str]) -> list[str]:
        """Reject blank or duplicate structured-output values."""
        normalized_values = [value.strip() for value in values]

        if any(not value for value in normalized_values):
            raise ValueError("list values must not be blank")

        if len(set(normalized_values)) != len(normalized_values):
            raise ValueError("list values must not contain duplicates")

        return normalized_values

    @model_validator(mode="after")
    def validate_report_consistency(self) -> ClaudeReleaseRiskReport:
        """Enforce deterministic safety constraints on Claude output."""
        expected_ranks = list(range(1, len(self.risks) + 1))
        actual_ranks = [risk.rank for risk in self.risks]

        if actual_ranks != expected_ranks:
            raise ValueError("risk ranks must be sequential and start at 1")

        if (
            self.recommendation is not RiskSummaryActionResponse.PROCEED
            and not self.risks
        ):
            raise ValueError(
                "non-proceed recommendations must include at least one risk"
            )

        if (
            self.recommendation
            is RiskSummaryActionResponse.PARTIAL_DATA_REVIEW
            and not self.requires_human_review
        ):
            raise ValueError(
                "partial-data recommendations must require human review"
            )

        if any(
            risk.severity
            in {
                RiskSeverityResponse.HIGH,
                RiskSeverityResponse.CRITICAL,
            }
            for risk in self.risks
        ) and not self.requires_human_review:
            raise ValueError(
                "high or critical risks must require human review"
            )

        return self
