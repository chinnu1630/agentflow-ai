"""Tests for strict AgentFlow tool-argument contracts."""

import pytest
from pydantic import ValidationError

from app.schemas.agent_tool import AgentToolName
from app.schemas.agent_tool_arguments import (
    LoadCurrentRiskSnapshotArguments,
    LookupGitHubPullRequestArguments,
    LookupJiraIssueArguments,
    SearchEngineeringKnowledgeArguments,
    validate_agent_tool_arguments,
)


def test_validates_github_pull_request_arguments() -> None:
    """A positive PR number is accepted."""
    arguments = validate_agent_tool_arguments(
        tool_name=AgentToolName.LOOKUP_GITHUB_PULL_REQUEST,
        arguments={"pull_request_number": 42},
    )

    assert isinstance(arguments, LookupGitHubPullRequestArguments)
    assert arguments.pull_request_number == 42


def test_normalizes_jira_issue_key() -> None:
    """Jira issue keys are normalized before adapter use."""
    arguments = validate_agent_tool_arguments(
        tool_name=AgentToolName.LOOKUP_JIRA_ISSUE,
        arguments={"issue_key": "pay-102"},
    )

    assert isinstance(arguments, LookupJiraIssueArguments)
    assert arguments.issue_key == "PAY-102"


def test_rejects_unknown_planner_arguments() -> None:
    """Prompt injection cannot add unsupported adapter parameters."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        validate_agent_tool_arguments(
            tool_name=AgentToolName.SEARCH_ENGINEERING_KNOWLEDGE,
            arguments={
                "query": "payment rollback",
                "top_k": 5,
                "include_secrets": True,
            },
        )


def test_rejects_invalid_knowledge_limit() -> None:
    """Knowledge retrieval remains bounded by contract."""
    with pytest.raises(ValidationError):
        validate_agent_tool_arguments(
            tool_name=AgentToolName.SEARCH_ENGINEERING_KNOWLEDGE,
            arguments={
                "query": "payment rollback",
                "top_k": 1_000,
            },
        )


def test_applies_safe_knowledge_defaults() -> None:
    """Knowledge search defaults to five retrieved chunks."""
    arguments = validate_agent_tool_arguments(
        tool_name=AgentToolName.SEARCH_ENGINEERING_KNOWLEDGE,
        arguments={"query": "payment rollback"},
    )

    assert isinstance(arguments, SearchEngineeringKnowledgeArguments)
    assert arguments.top_k == 5


def test_accepts_empty_arguments_for_snapshot_tool() -> None:
    """Context-only tools do not accept planner-controlled identifiers."""
    arguments = validate_agent_tool_arguments(
        tool_name=AgentToolName.LOAD_CURRENT_RISK_SNAPSHOT,
        arguments={},
    )

    assert isinstance(arguments, LoadCurrentRiskSnapshotArguments)


@pytest.mark.parametrize("tool_name", list(AgentToolName))
def test_every_registered_tool_has_argument_contract(
    tool_name: AgentToolName,
) -> None:
    """Every registry tool must have a deterministic argument model."""
    required_arguments: dict[AgentToolName, dict[str, object]] = {
        AgentToolName.RUN_FRESH_RELEASE_RISK_ANALYSIS: {
            "query": "What are the release risks?"
        },
        AgentToolName.LOOKUP_GITHUB_PULL_REQUEST: {
            "pull_request_number": 42
        },
        AgentToolName.LOOKUP_JIRA_ISSUE: {"issue_key": "PAY-102"},
        AgentToolName.SEARCH_ENGINEERING_KNOWLEDGE: {
            "query": "payment rollback"
        },
    }

    validate_agent_tool_arguments(
        tool_name=tool_name,
        arguments=required_arguments.get(tool_name, {}),
    )
