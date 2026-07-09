"""Workflow state models for AgentFlow AI release-risk orchestration.

This module defines the validated state object that LangGraph nodes will
read from and write to during the release-risk workflow.

Current workflow scope:
FastAPI -> ReleaseRunService -> workflow state -> GitHub/Jira summaries

Future workflow scope:
FastAPI -> LangGraph Orchestrator -> EngOps Agent -> Knowledge Agent
-> ML Scoring -> Risk Synthesis -> HITL Gate -> Slack Agent
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ReleaseRiskWorkflowStatus(StrEnum):
    """High-level execution status for the release-risk workflow."""

    NOT_STARTED = "not_started"
    RUNNING = "running"
    PARTIAL = "partial"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ReleaseRiskWorkflowStage(StrEnum):
    """Current workflow stage for release-risk orchestration."""

    INITIALIZED = "initialized"
    COLLECTING_GITHUB_RISKS = "collecting_github_risks"
    COLLECTING_JIRA_RISKS = "collecting_jira_risks"
    BUILDING_RELEASE_SUMMARY = "building_release_summary"
    RETRIEVING_KNOWLEDGE_CONTEXT = "retrieving_knowledge_context"
    SCORING_RELEASE_RISK = "scoring_release_risk"
    COMPLETED = "completed"
    FAILED = "failed"


class KnowledgeRetrievalStatus(StrEnum):
    """Execution status for Knowledge Agent retrieval."""

    NOT_STARTED = "not_started"
    COMPLETED = "completed"
    NO_RESULTS = "no_results"
    FAILED = "failed"


class ReleaseRiskWorkflowError(BaseModel):
    """Error captured during workflow execution."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Workflow component where the error happened.",
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=1_000,
        description="Safe human-readable error message.",
    )
    recoverable: bool = Field(
        default=True,
        description="Whether the workflow can continue after this error.",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Safe diagnostic metadata. Do not store secrets.",
    )

    @field_validator("source", "message")
    @classmethod
    def validate_non_blank_text(cls, value: str) -> str:
        """Reject blank text after whitespace is stripped."""
        stripped_value = value.strip()

        if not stripped_value:
            raise ValueError("value must not be blank")

        return stripped_value


class ReleaseRiskState(BaseModel):
    """Validated runtime state shared across release-risk workflow nodes.

    This object is the contract that future LangGraph nodes will read from
    and update. For now, GitHub/Jira outputs are stored as flexible dicts
    because the existing service layer already owns those response schemas.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    release_run_id: UUID = Field(
        ...,
        description="Database ID of the release run being analyzed.",
    )
    run_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Request/run correlation ID used in structured logs.",
    )
    manager_query: str = Field(
        default="What are the biggest release risks this week?",
        min_length=1,
        max_length=500,
        description="Original manager question that started the workflow.",
    )
    requested_by: str | None = Field(
        default=None,
        max_length=255,
        description="User or system actor who requested the workflow.",
    )

    status: ReleaseRiskWorkflowStatus = Field(
        default=ReleaseRiskWorkflowStatus.NOT_STARTED,
        description="Current workflow execution status.",
    )
    stage: ReleaseRiskWorkflowStage = Field(
        default=ReleaseRiskWorkflowStage.INITIALIZED,
        description="Current workflow execution stage.",
    )
    
    release_run: dict[str, Any] | None = Field(
        default=None,
        description="Release run metadata from the existing service response.",
    )
    github: dict[str, Any] | None = Field(
        default=None,
        description="GitHub risk collection output.",
    )
    github_summary: dict[str, Any] | None = Field(
        default=None,
        description="GitHub summary output.",
    )
    jira: dict[str, Any] | None = Field(
        default=None,
        description="Jira risk collection output.",
    )
    jira_summary: dict[str, Any] | None = Field(
        default=None,
        description="Jira summary output.",
    )
    release_summary: dict[str, Any] | None = Field(
        default=None,
        description="Combined release summary output.",
    )
    knowledge_query: str | None = Field(
        default=None,
        max_length=1_000,
        description="Query used to retrieve internal engineering knowledge.",
    )
    knowledge_results: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Retrieved Knowledge Agent evidence chunks.",
    )
    knowledge_status: KnowledgeRetrievalStatus = Field(
        default=KnowledgeRetrievalStatus.NOT_STARTED,
        description="Knowledge Agent retrieval execution status.",
    )
    knowledge_error: str | None = Field(
        default=None,
        max_length=1_000,
        description="Safe Knowledge Agent retrieval error message.",
    )

    risk_features: dict[str, Any] | None = Field(
        default=None,
        description="Extracted numeric feature vector used for release-risk scoring.",
    )
    risk_score: dict[str, Any] | None = Field(
        default=None,
        description="Deterministic release-risk score and recommendation.",
    )

    completed_nodes: list[str] = Field(
        default_factory=list,
        description="Workflow node names that completed successfully.",
    )
    failed_nodes: list[str] = Field(
        default_factory=list,
        description="Workflow node names that failed.",
    )
    errors: list[ReleaseRiskWorkflowError] = Field(
        default_factory=list,
        description="Workflow errors captured for graceful degradation.",
    )

    @field_validator("run_id", "manager_query")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """Reject blank required text fields."""
        stripped_value = value.strip()

        if not stripped_value:
            raise ValueError("value must not be blank")

        return stripped_value

    @field_validator("requested_by")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        """Normalize optional text fields and reject blank strings."""
        if value is None:
            return None

        stripped_value = value.strip()

        if not stripped_value:
            raise ValueError("value must not be blank")

        return stripped_value

    @property
    def has_errors(self) -> bool:
        """Return whether the workflow has captured errors."""
        return bool(self.errors)

    @property
    def is_terminal(self) -> bool:
        """Return whether the workflow reached a terminal status."""
        return self.status in {
            ReleaseRiskWorkflowStatus.SUCCEEDED,
            ReleaseRiskWorkflowStatus.FAILED,
        }

    def mark_running(self, stage: ReleaseRiskWorkflowStage) -> Self:
        """Return a copy of the state marked as running at the given stage."""
        return self.model_copy(
            update={
                "status": ReleaseRiskWorkflowStatus.RUNNING,
                "stage": stage,
            }
        )

    def mark_succeeded(self) -> Self:
        """Return a copy of the state marked as successfully completed."""
        return self.model_copy(
            update={
                "status": ReleaseRiskWorkflowStatus.SUCCEEDED,
                "stage": ReleaseRiskWorkflowStage.COMPLETED,
            }
        )

    def add_completed_node(self, node_name: str) -> Self:
        """Return a copy of the state with one completed node added."""
        safe_node_name = node_name.strip()

        if not safe_node_name:
            raise ValueError("node_name must not be blank")

        return self.model_copy(
            update={
                "completed_nodes": [*self.completed_nodes, safe_node_name],
            }
        )

    def add_error(
        self,
        *,
        source: str,
        message: str,
        recoverable: bool = True,
        details: dict[str, Any] | None = None,
    ) -> Self:
        """Return a copy of the state with a workflow error added.

        Recoverable errors move the workflow to PARTIAL status.
        Non-recoverable errors move the workflow to FAILED status.
        """
        workflow_error = ReleaseRiskWorkflowError(
            source=source,
            message=message,
            recoverable=recoverable,
            details=details or {},
        )

        next_status = (
            ReleaseRiskWorkflowStatus.PARTIAL
            if recoverable
            else ReleaseRiskWorkflowStatus.FAILED
        )
        next_stage = self.stage if recoverable else ReleaseRiskWorkflowStage.FAILED

        failed_nodes = (
            [*self.failed_nodes, workflow_error.source]
            if workflow_error.source not in self.failed_nodes
            else self.failed_nodes
        )

        return self.model_copy(
            update={
                "status": next_status,
                "stage": next_stage,
                "errors": [*self.errors, workflow_error],
                "failed_nodes": failed_nodes,
            }
        )