"""Tests for strict AgentFlow agent-tool contracts."""

import pytest
from pydantic import ValidationError

from app.schemas.agent_tool import (
    AgentToolDefinition,
    AgentToolEffect,
    AgentToolEvidence,
    AgentToolExecutionStatus,
    AgentToolName,
    AgentToolResult,
)


def _build_evidence() -> AgentToolEvidence:
    """Create reusable tool evidence."""
    return AgentToolEvidence(
        source_type="jira_issue",
        source_id="PAY-102",
        title="Payment release blocker",
        source_url="https://jira.example.com/browse/PAY-102",
    )


def test_accepts_valid_read_only_tool_definition() -> None:
    """Read-only tools may execute without human approval."""
    definition = AgentToolDefinition(
        name=AgentToolName.SEARCH_ENGINEERING_KNOWLEDGE,
        description="Search trusted engineering documents.",
        effect=AgentToolEffect.READ_ONLY,
        requires_release_run_context=False,
        requires_human_approval=False,
        default_timeout_seconds=20,
    )

    assert definition.effect is AgentToolEffect.READ_ONLY
    assert definition.requires_human_approval is False


def test_rejects_side_effect_tool_without_approval() -> None:
    """Side effects must never bypass human approval."""
    with pytest.raises(
        ValidationError,
        match="side-effecting tools must require human approval",
    ):
        AgentToolDefinition(
            name=AgentToolName.SEND_APPROVED_SLACK_ALERT,
            description="Send an approved Slack release alert.",
            effect=AgentToolEffect.SIDE_EFFECT,
            requires_release_run_context=True,
            requires_human_approval=False,
            default_timeout_seconds=30,
        )


def test_accepts_successful_tool_result_with_evidence() -> None:
    """Successful tool results may contain trusted evidence."""
    result = AgentToolResult(
        step_id="lookup_jira",
        tool_name=AgentToolName.LOOKUP_JIRA_ISSUE,
        status=AgentToolExecutionStatus.SUCCESS,
        output={"issue_key": "PAY-102"},
        evidence=[_build_evidence()],
        duration_ms=42,
    )

    assert result.status is AgentToolExecutionStatus.SUCCESS
    assert result.evidence[0].source_id == "PAY-102"


def test_rejects_failed_result_without_error_metadata() -> None:
    """Failed results must provide a safe machine-readable error."""
    with pytest.raises(
        ValidationError,
        match="failed tool results must include error metadata",
    ):
        AgentToolResult(
            step_id="lookup_jira",
            tool_name=AgentToolName.LOOKUP_JIRA_ISSUE,
            status=AgentToolExecutionStatus.FAILED,
            duration_ms=10,
        )


def test_rejects_successful_result_with_error_metadata() -> None:
    """Successful results must not contain contradictory errors."""
    with pytest.raises(
        ValidationError,
        match="successful tool results must not include error metadata",
    ):
        AgentToolResult(
            step_id="lookup_jira",
            tool_name=AgentToolName.LOOKUP_JIRA_ISSUE,
            status=AgentToolExecutionStatus.SUCCESS,
            error_code="unexpected_error",
            duration_ms=10,
        )


def test_rejects_duplicate_tool_evidence() -> None:
    """One result must not repeat the same evidence reference."""
    evidence = _build_evidence()

    with pytest.raises(
        ValidationError,
        match="tool evidence must not contain duplicate references",
    ):
        AgentToolResult(
            step_id="lookup_jira",
            tool_name=AgentToolName.LOOKUP_JIRA_ISSUE,
            status=AgentToolExecutionStatus.SUCCESS,
            evidence=[evidence, evidence.model_copy()],
            duration_ms=15,
        )
