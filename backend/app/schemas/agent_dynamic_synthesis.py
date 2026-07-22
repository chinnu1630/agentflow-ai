"""Strict structured-output contracts for dynamic agent synthesis."""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class AgentDynamicAnswerCitation(BaseModel):
    """One trusted tool-evidence citation supporting a dynamic answer."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    source_type: str = Field(min_length=1, max_length=100)
    source_id: str = Field(min_length=1, max_length=255)
    title: str = Field(min_length=1, max_length=500)
    source_url: str | None = Field(default=None, max_length=2_000)
    supporting_fact: str = Field(min_length=1, max_length=1_000)


class AgentDynamicAnswer(BaseModel):
    """Evidence-grounded manager answer synthesized from trusted tool results."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal["agent_dynamic_answer_v1"] = (
        "agent_dynamic_answer_v1"
    )
    answer: str = Field(min_length=1, max_length=5_000)
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[AgentDynamicAnswerCitation] = Field(
        default_factory=list,
        max_length=100,
    )
    missing_information: list[str] = Field(
        default_factory=list,
        max_length=20,
    )
    degraded_steps: list[str] = Field(
        default_factory=list,
        max_length=20,
    )
    requires_human_review: bool

    @field_validator("missing_information", "degraded_steps")
    @classmethod
    def validate_string_lists(cls, values: list[str]) -> list[str]:
        """Reject blank or duplicate synthesis metadata values."""
        normalized_values = [value.strip() for value in values]

        if any(not value for value in normalized_values):
            raise ValueError("list values must not be blank")

        if len(normalized_values) != len(set(normalized_values)):
            raise ValueError("list values must not contain duplicates")

        return normalized_values

    @field_validator("citations")
    @classmethod
    def reject_duplicate_citations(
        cls,
        values: list[AgentDynamicAnswerCitation],
    ) -> list[AgentDynamicAnswerCitation]:
        """Reject duplicate source-type and source-ID citation pairs."""
        citation_keys = [
            (citation.source_type, citation.source_id)
            for citation in values
        ]

        if len(citation_keys) != len(set(citation_keys)):
            raise ValueError(
                "dynamic answer citations must not contain duplicates"
            )

        return values

    @model_validator(mode="after")
    def validate_review_consistency(self) -> AgentDynamicAnswer:
        """Require human review when synthesis reports degraded execution."""
        if self.degraded_steps and not self.requires_human_review:
            raise ValueError(
                "degraded dynamic answers must require human review"
            )

        return self
