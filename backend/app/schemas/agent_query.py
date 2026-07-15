"""Public schemas for natural-language AgentFlow queries.

These schemas define the contract between the natural-language chat interface
and the internal AgentFlow workflow orchestration layer.

A user's free-text question is converted into a validated AgentQueryPlan before
any workflow, retrieval service, approval process, or side-effecting action is
executed.
"""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.risk import ReleaseRunRiskResponse


class AgentIntent(StrEnum):
    """Supported workflow-aware intents for AgentFlow queries."""

    RELEASE_RISK_SUMMARY = "release_risk_summary"
    EXPLAIN_RISK_SCORE = "explain_risk_score"
    EXPLAIN_SPECIFIC_RISK = "explain_specific_risk"
    FILTER_RISKS = "filter_risks"
    GITHUB_PR_QUESTION = "github_pr_question"
    JIRA_TICKET_QUESTION = "jira_ticket_question"
    KNOWLEDGE_DOC_QUESTION = "knowledge_doc_question"
    APPROVAL_STATUS_QUESTION = "approval_status_question"
    SLACK_STATUS_QUESTION = "slack_status_question"
    WORKFLOW_STATUS_QUESTION = "workflow_status_question"
    HISTORICAL_RISK_LOOKUP = "historical_risk_lookup"
    SIMILAR_PAST_RELEASE = "similar_past_release"
    COMPARE_WITH_PREVIOUS_RELEASE = "compare_with_previous_release"
    ACTION_REQUEST = "action_request"
    OUT_OF_SCOPE = "out_of_scope"


class ResponseDepth(StrEnum):
    """Expected level of detail for the generated answer."""

    BRIEF = "brief"
    STANDARD = "standard"
    DEEP = "deep"
    ACTION_CONFIRMATION = "action_confirmation"


class RiskSourceFilter(StrEnum):
    """Trusted AgentFlow sources that may be selected by a query."""

    GITHUB = "github"
    JIRA = "jira"
    KNOWLEDGE = "knowledge"


class AgentQueryFilters(BaseModel):
    """Structured filters inferred from natural-language input."""

    model_config = ConfigDict(extra="forbid")

    sources: list[RiskSourceFilter] = Field(default_factory=list)
    severities: list[str] = Field(default_factory=list)
    blockers_only: bool = False
    open_items_only: bool = False


class AgentEntityReferences(BaseModel):
    """Candidate entity references extracted from a natural-language query.

    These values are not trusted automatically. A later context-resolution
    service must verify them against persisted AgentFlow data such as the
    current risk snapshot, GitHub records, Jira records, and release history.
    """

    model_config = ConfigDict(extra="forbid")

    service_names: list[str] = Field(default_factory=list)
    pull_request_numbers: list[int] = Field(default_factory=list)
    jira_issue_keys: list[str] = Field(default_factory=list)


class AgentQueryRequest(BaseModel):
    """Natural-language query submitted to the AgentFlow agent."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    query: str = Field(
        min_length=1,
        max_length=2_000,
        description="Natural-language release workflow question.",
    )
    conversation_session_id: UUID | None = None
    release_run_id: UUID | None = None

    @field_validator("query")
    @classmethod
    def reject_control_only_queries(cls, value: str) -> str:
        """Reject queries that contain no meaningful letters or numbers."""

        if not any(character.isalnum() for character in value):
            raise ValueError("Query must contain at least one letter or number.")

        return value


class AgentQueryPlan(BaseModel):
    """Validated routing decision produced from a natural-language query.

    This plan describes what AgentFlow should do next. It does not itself
    execute workflows, retrieve data, approve releases, or send Slack alerts.
    """

    model_config = ConfigDict(extra="forbid")

    intent: AgentIntent
    response_depth: ResponseDepth
    confidence: float = Field(ge=0.0, le=1.0)

    release_run_id: UUID | None = None
    conversation_session_id: UUID | None = None

    filters: AgentQueryFilters = Field(default_factory=AgentQueryFilters)
    entity_references: AgentEntityReferences = Field(default_factory=AgentEntityReferences)

    requires_current_snapshot: bool = False
    requires_historical_lookup: bool = False
    requires_human_approval: bool = False
    may_execute_side_effect: bool = False

    routing_reason_code: str = Field(
        min_length=1,
        max_length=100,
        description=(
            "Safe machine-readable routing reason. This field must not "
            "contain raw user input, hidden prompts, or chain-of-thought."
        ),
    )


class AgentCitation(BaseModel):
    """Trusted evidence reference supporting an AgentFlow answer."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1, max_length=50)
    source_type: str = Field(min_length=1, max_length=100)
    source_id: str = Field(min_length=1, max_length=255)
    title: str = Field(min_length=1, max_length=500)
    source_url: str | None = None


class AgentQueryResponse(BaseModel):
    """Conversational response returned after executing an AgentFlow query."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    plan: AgentQueryPlan
    release_risk: ReleaseRunRiskResponse
    citations: list[AgentCitation] = Field(default_factory=list)
    approval_required: bool


class AgentQueryContext(BaseModel):
    """Trusted persisted context resolved for a follow-up agent query."""

    model_config = ConfigDict(extra="forbid")

    release_run_id: UUID
    snapshot_id: UUID
    snapshot_version: int = Field(ge=1)
    release_risk: ReleaseRunRiskResponse
