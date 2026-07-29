"""Typed API boundary models for the AgentFlow manager frontend.

These models validate data crossing the Streamlit-to-FastAPI boundary. They do
not calculate risk, make approval decisions, or reproduce backend business
logic.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentEntityReferences(BaseModel):
    """Validated entity context sent between focused follow-up questions."""

    model_config = ConfigDict(extra="forbid")

    service_names: list[str] = Field(default_factory=list)
    pull_request_numbers: list[int] = Field(default_factory=list)
    jira_issue_keys: list[str] = Field(default_factory=list)


class AgentQueryRequest(BaseModel):
    """Natural-language query sent to the AgentFlow backend."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    query: str = Field(min_length=1, max_length=2_000)
    conversation_session_id: UUID | None = None
    release_run_id: UUID | None = None
    context_entity_references: AgentEntityReferences | None = None

    @field_validator("query")
    @classmethod
    def reject_control_only_queries(cls, value: str) -> str:
        """Reject queries containing no meaningful letters or numbers."""
        if not any(character.isalnum() for character in value):
            raise ValueError("Query must contain at least one letter or number.")

        return value


class _APIResponseModel(BaseModel):
    """Forward-compatible base model for backend responses."""

    model_config = ConfigDict(extra="ignore")


class AgentCitation(_APIResponseModel):
    """Trusted evidence citation returned by AgentFlow."""

    source: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_url: str | None = None


class AgentQueryPlan(_APIResponseModel):
    """Safe routing metadata returned with an agent answer."""

    intent: str = Field(min_length=1)
    response_depth: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    release_run_id: UUID | None = None
    conversation_session_id: UUID | None = None
    entity_references: AgentEntityReferences = Field(
        default_factory=AgentEntityReferences
    )
    requires_current_snapshot: bool = False
    requires_historical_lookup: bool = False
    requires_human_approval: bool = False
    may_execute_side_effect: bool = False
    routing_reason_code: str = Field(min_length=1)


class ReleaseRunSummary(_APIResponseModel):
    """Release-run metadata returned inside a risk response."""

    id: UUID
    run_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    requested_by: str = Field(min_length=1)
    status: str = Field(min_length=1)
    created_at: datetime
    completed_at: datetime | None = None


class RiskSummaryItem(_APIResponseModel):
    """One ranked, explainable release risk."""

    source: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_url: str | None = None
    severity: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    title: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    evidence: dict[str, Any] = Field(default_factory=dict)


class ReleaseRiskSummary(_APIResponseModel):
    """Combined deterministic release-risk summary."""

    overall_severity: str = Field(min_length=1)
    recommended_action: str = Field(min_length=1)
    total_signal_count: int = Field(ge=0)
    high_risk_count: int = Field(ge=0)
    top_risks: list[RiskSummaryItem] = Field(default_factory=list)
    summary_text: str = Field(min_length=1)


class GitHubRiskCollection(_APIResponseModel):
    """GitHub collection status used for degradation reporting."""

    status: str = Field(min_length=1)
    error_type: str | None = None
    error_message: str | None = None


class JiraRiskCollection(_APIResponseModel):
    """Jira collection status used for degradation reporting."""

    status: str = Field(min_length=1)
    error_message: str | None = None


class ReleaseRiskScore(_APIResponseModel):
    """Auditable deterministic release-risk score."""

    score: float = Field(ge=0.0, le=1.0)
    risk_level: str = Field(min_length=1)
    recommended_action: str = Field(min_length=1)
    reasons: list[str] = Field(default_factory=list)


class ReleaseRunRiskResponse(_APIResponseModel):
    """Frontend projection of the backend release-risk response."""

    release_run: ReleaseRunSummary
    github: GitHubRiskCollection
    jira: JiraRiskCollection
    release_summary: ReleaseRiskSummary
    risk_score: ReleaseRiskScore | None = None
    synthesis_status: str | None = None
    synthesis_error: str | None = None
    approval_required: bool | None = None
    approval_reason: str | None = None
    approval_request_id: UUID | None = None
    approval_status: str | None = None


class AgentQueryResponse(_APIResponseModel):
    """Validated response from POST /api/v1/agent/query."""

    answer: str = Field(min_length=1)
    plan: AgentQueryPlan
    release_risk: ReleaseRunRiskResponse | None = None
    citations: list[AgentCitation] = Field(default_factory=list)
    approval_required: bool


class ReleaseApprovalDecisionStatus(StrEnum):
    """Allowed manager decisions for a pending release approval."""

    APPROVED = "approved"
    REJECTED = "rejected"


class ReleaseRunApprovalDecisionRequest(BaseModel):
    """Validated approval or rejection payload sent to FastAPI."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    approval_status: ReleaseApprovalDecisionStatus
    decision_note: str | None = Field(default=None, max_length=1_000)


class ReleaseRunApproval(_APIResponseModel):
    """One durable human-approval record."""

    id: UUID
    release_run_id: UUID
    approval_status: str = Field(min_length=1)
    approval_reason: str = Field(min_length=1)
    approval_policy_version: str = Field(min_length=1)
    requested_by: str | None = None
    decided_by: str | None = None
    decision_note: str | None = None
    created_at: datetime
    decided_at: datetime | None = None


class PendingReleaseRunApprovalList(_APIResponseModel):
    """Pending approvals available to the manager dashboard."""

    approval_status: str = Field(min_length=1)
    approvals: list[ReleaseRunApproval] = Field(default_factory=list)


class ReleaseRunApprovalList(_APIResponseModel):
    """Approval history for one release workflow run."""

    release_run_id: UUID
    approvals: list[ReleaseRunApproval] = Field(default_factory=list)


class ReleaseRunStatus(_APIResponseModel):
    """Current persisted status for a release workflow run."""

    id: UUID
    run_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    requested_by: str = Field(min_length=1)
    status: str = Field(min_length=1)
    created_at: datetime
    completed_at: datetime | None = None


class ReleaseRunEvent(_APIResponseModel):
    """One append-only AgentFlow workflow audit event."""

    id: UUID
    release_run_id: UUID
    event_type: str = Field(min_length=1)
    event_status: str = Field(min_length=1)
    message: str = Field(min_length=1)
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ReleaseRunEventList(_APIResponseModel):
    """Audit timeline for one release workflow run."""

    release_run_id: UUID
    events: list[ReleaseRunEvent] = Field(default_factory=list)


class SlackReleaseAlertResult(_APIResponseModel):
    """Result returned after an approved Slack alert is delivered."""

    sent: bool
    slack_channel: str = Field(min_length=1)
    slack_timestamp: str = Field(min_length=1)
    risk_level: str = Field(min_length=1)
    risk_score: float = Field(ge=0.0, le=1.0)
    recommended_action: str = Field(min_length=1)
