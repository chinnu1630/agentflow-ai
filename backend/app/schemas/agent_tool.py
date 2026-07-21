"""Strict contracts for AgentFlow AI planner-selectable tools.

These schemas describe approved tool capabilities, planner-generated
invocations, evidence, and normalized execution results. They do not execute
tools or grant authorization.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)


class AgentToolName(StrEnum):
    """Approved AgentFlow capabilities available to bounded planners."""

    RUN_FRESH_RELEASE_RISK_ANALYSIS = "run_fresh_release_risk_analysis"
    LOAD_CURRENT_RISK_SNAPSHOT = "load_current_risk_snapshot"
    LOOKUP_GITHUB_PULL_REQUEST = "lookup_github_pull_request"
    LOOKUP_JIRA_ISSUE = "lookup_jira_issue"
    SEARCH_ENGINEERING_KNOWLEDGE = "search_engineering_knowledge"
    LOOKUP_RELEASE_HISTORY = "lookup_release_history"
    LOOKUP_SIMILAR_RELEASE = "lookup_similar_release"
    LOOKUP_APPROVAL_STATUS = "lookup_approval_status"
    LOOKUP_SLACK_STATUS = "lookup_slack_status"
    SEND_APPROVED_SLACK_ALERT = "send_approved_slack_alert"


class AgentToolEffect(StrEnum):
    """Operational effect produced by an AgentFlow tool."""

    READ_ONLY = "read_only"
    SIDE_EFFECT = "side_effect"


class AgentToolExecutionStatus(StrEnum):
    """Normalized outcome of one tool execution."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class AgentToolDefinition(BaseModel):
    """Trusted registry metadata describing one available tool."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    name: AgentToolName
    description: str = Field(min_length=1, max_length=500)
    effect: AgentToolEffect
    requires_release_run_context: bool
    requires_human_approval: bool
    default_timeout_seconds: int = Field(ge=1, le=120)

    @model_validator(mode="after")
    def validate_side_effect_policy(self) -> AgentToolDefinition:
        """Require human approval for every side-effecting tool."""
        if (
            self.effect is AgentToolEffect.SIDE_EFFECT
            and not self.requires_human_approval
        ):
            raise ValueError(
                "side-effecting tools must require human approval"
            )

        return self


class AgentToolInvocation(BaseModel):
    """One bounded tool call requested by an execution planner."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    step_id: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$",
    )
    tool_name: AgentToolName
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    timeout_seconds: int = Field(ge=1, le=120)


class AgentToolEvidence(BaseModel):
    """Trusted evidence returned by an AgentFlow tool."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    source_type: str = Field(min_length=1, max_length=100)
    source_id: str = Field(min_length=1, max_length=255)
    title: str = Field(min_length=1, max_length=500)
    source_url: str | None = Field(default=None, max_length=2_000)


class AgentToolResult(BaseModel):
    """Normalized success, partial-success, or failure result."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    step_id: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$",
    )
    tool_name: AgentToolName
    status: AgentToolExecutionStatus
    output: dict[str, JsonValue] = Field(default_factory=dict)
    evidence: list[AgentToolEvidence] = Field(
        default_factory=list,
        max_length=100,
    )
    error_code: str | None = Field(default=None, max_length=100)
    error_message: str | None = Field(default=None, max_length=1_000)
    duration_ms: int = Field(ge=0)

    @field_validator("evidence")
    @classmethod
    def reject_duplicate_evidence(
        cls,
        values: list[AgentToolEvidence],
    ) -> list[AgentToolEvidence]:
        """Reject duplicate evidence references from one tool result."""
        evidence_keys = [
            (value.source_type, value.source_id)
            for value in values
        ]

        if len(evidence_keys) != len(set(evidence_keys)):
            raise ValueError(
                "tool evidence must not contain duplicate references"
            )

        return values

    @model_validator(mode="after")
    def validate_status_consistency(self) -> AgentToolResult:
        """Keep tool status and error metadata internally consistent."""
        has_error = bool(self.error_code or self.error_message)

        if (
            self.status is AgentToolExecutionStatus.FAILED
            and not has_error
        ):
            raise ValueError(
                "failed tool results must include error metadata"
            )

        if (
            self.status is AgentToolExecutionStatus.SUCCESS
            and has_error
        ):
            raise ValueError(
                "successful tool results must not include error metadata"
            )

        return self
