"""Workflow node functions for AgentFlow AI release-risk orchestration.

These functions are the first LangGraph-ready node building blocks.
For now, they only perform safe state transitions.

Current scope:
- Start workflow
- Mark GitHub collection stage
- Mark Jira collection stage
- Mark release summary stage
- Complete workflow
- Fail workflow safely

Future scope:
- Call GitHub/Jira collectors
- Call hybrid RAG
- Call ML risk scoring
- Call Claude synthesis
- Pause for HITL approval
- Send approved Slack alert
"""

from __future__ import annotations

from typing import Any

from app.workflows.release_risk_state import (
    ReleaseRiskState,
    ReleaseRiskWorkflowStage,
)


def start_release_risk_workflow(state: ReleaseRiskState) -> ReleaseRiskState:
    """Mark the release-risk workflow as running.

    This node represents the first step of the future LangGraph workflow.
    It does not perform I/O. It only moves the workflow from initialized
    to running.
    """
    return state.mark_running(ReleaseRiskWorkflowStage.INITIALIZED).add_completed_node(
        "start_release_risk_workflow"
    )


def prepare_github_risk_collection(state: ReleaseRiskState) -> ReleaseRiskState:
    """Move the workflow into the GitHub risk collection stage."""
    return state.mark_running(
        ReleaseRiskWorkflowStage.COLLECTING_GITHUB_RISKS
    ).add_completed_node("prepare_github_risk_collection")


def prepare_jira_risk_collection(state: ReleaseRiskState) -> ReleaseRiskState:
    """Move the workflow into the Jira risk collection stage."""
    return state.mark_running(
        ReleaseRiskWorkflowStage.COLLECTING_JIRA_RISKS
    ).add_completed_node("prepare_jira_risk_collection")


def prepare_release_summary(state: ReleaseRiskState) -> ReleaseRiskState:
    """Move the workflow into the release summary building stage."""
    return state.mark_running(
        ReleaseRiskWorkflowStage.BUILDING_RELEASE_SUMMARY
    ).add_completed_node("prepare_release_summary")


def complete_release_risk_workflow(state: ReleaseRiskState) -> ReleaseRiskState:
    """Mark the release-risk workflow as successfully completed."""
    return state.mark_succeeded().add_completed_node("complete_release_risk_workflow")


def fail_release_risk_workflow(
    state: ReleaseRiskState,
    *,
    source: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> ReleaseRiskState:
    """Mark the release-risk workflow as failed with a safe error message.

    This helper is for non-recoverable workflow failures only.
    Recoverable failures should use state.add_error(..., recoverable=True).
    """
    return state.add_error(
        source=source,
        message=message,
        recoverable=False,
        details=details,
    )