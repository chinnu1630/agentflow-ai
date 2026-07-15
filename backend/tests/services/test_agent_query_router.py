"""Unit tests for the deterministic AgentFlow query router."""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.schemas.agent_query import (
    AgentIntent,
    AgentQueryRequest,
    ResponseDepth,
    RiskSourceFilter,
)
from app.services.agent_query_router import AgentQueryRouter


@pytest.fixture
def router() -> AgentQueryRouter:
    """Return a fresh query router for each test."""

    return AgentQueryRouter()


@pytest.mark.anyio
async def test_routes_release_risk_summary(
    router: AgentQueryRouter,
) -> None:
    """A general release-risk question should use the current snapshot."""

    request = AgentQueryRequest(query="What are the biggest release risks this week?")

    plan = await router.create_plan(request)

    assert plan.intent is AgentIntent.RELEASE_RISK_SUMMARY
    assert plan.response_depth is ResponseDepth.STANDARD
    assert plan.requires_current_snapshot is True
    assert plan.requires_historical_lookup is False
    assert plan.requires_human_approval is False
    assert plan.may_execute_side_effect is False


@pytest.mark.anyio
async def test_routes_short_risk_score_question_as_deep(
    router: AgentQueryRouter,
) -> None:
    """Answer depth should depend on intent rather than query length."""

    request = AgentQueryRequest(
        query="Why high?",
        release_run_id=uuid4(),
    )

    plan = await router.create_plan(request)

    assert plan.intent is AgentIntent.EXPLAIN_RISK_SCORE
    assert plan.response_depth is ResponseDepth.DEEP
    assert plan.requires_current_snapshot is True


@pytest.mark.anyio
async def test_routes_specific_risk_explanation(
    router: AgentQueryRouter,
) -> None:
    """A question about a specific risk should request a deep explanation."""

    request = AgentQueryRequest(
        query="Why is payment risky?",
        release_run_id=uuid4(),
    )

    plan = await router.create_plan(request)

    assert plan.intent is AgentIntent.EXPLAIN_SPECIFIC_RISK
    assert plan.response_depth is ResponseDepth.DEEP
    assert plan.requires_current_snapshot is True


@pytest.mark.anyio
async def test_routes_jira_blockers_filter(
    router: AgentQueryRouter,
) -> None:
    """A Jira-only blocker request should produce structured filters."""

    request = AgentQueryRequest(
        query="Show Jira blockers only.",
        release_run_id=uuid4(),
    )

    plan = await router.create_plan(request)

    assert plan.intent is AgentIntent.FILTER_RISKS
    assert plan.filters.sources == [RiskSourceFilter.JIRA]
    assert plan.filters.blockers_only is True
    assert plan.requires_current_snapshot is True


@pytest.mark.anyio
async def test_routes_historical_question(
    router: AgentQueryRouter,
) -> None:
    """Historical questions should request historical data lookup."""

    request = AgentQueryRequest(
        query="Did this happen before?",
        release_run_id=uuid4(),
    )

    plan = await router.create_plan(request)

    assert plan.intent is AgentIntent.HISTORICAL_RISK_LOOKUP
    assert plan.response_depth is ResponseDepth.DEEP
    assert plan.requires_historical_lookup is True


@pytest.mark.anyio
async def test_routes_previous_release_comparison(
    router: AgentQueryRouter,
) -> None:
    """Previous-release comparison needs current and historical data."""

    request = AgentQueryRequest(
        query="Compare this with the previous release.",
        release_run_id=uuid4(),
    )

    plan = await router.create_plan(request)

    assert plan.intent is AgentIntent.COMPARE_WITH_PREVIOUS_RELEASE
    assert plan.response_depth is ResponseDepth.DEEP
    assert plan.requires_current_snapshot is True
    assert plan.requires_historical_lookup is True


@pytest.mark.anyio
async def test_routes_slack_action_with_human_approval(
    router: AgentQueryRouter,
) -> None:
    """Slack actions must always remain behind the HITL approval gate."""

    request = AgentQueryRequest(
        query="Can you send this to Slack?",
        release_run_id=uuid4(),
    )

    plan = await router.create_plan(request)

    assert plan.intent is AgentIntent.ACTION_REQUEST
    assert plan.response_depth is ResponseDepth.ACTION_CONFIRMATION
    assert plan.requires_human_approval is True
    assert plan.may_execute_side_effect is True


@pytest.mark.anyio
async def test_routes_slack_status_as_read_only(
    router: AgentQueryRouter,
) -> None:
    """Checking Slack status must not be treated as a send action."""

    request = AgentQueryRequest(
        query="Was Slack already sent?",
        release_run_id=uuid4(),
    )

    plan = await router.create_plan(request)

    assert plan.intent is AgentIntent.SLACK_STATUS_QUESTION
    assert plan.response_depth is ResponseDepth.BRIEF
    assert plan.requires_human_approval is False
    assert plan.may_execute_side_effect is False


@pytest.mark.anyio
async def test_extracts_pull_request_and_jira_identifiers(
    router: AgentQueryRouter,
) -> None:
    """The router should extract candidate PR and Jira references."""

    request = AgentQueryRequest(
        query="Why are PR #412 and PAY-102 blocking the release?",
        release_run_id=uuid4(),
    )

    plan = await router.create_plan(request)

    assert plan.entity_references.pull_request_numbers == [412]
    assert plan.entity_references.jira_issue_keys == ["PAY-102"]


@pytest.mark.anyio
async def test_routes_unrelated_question_out_of_scope(
    router: AgentQueryRouter,
) -> None:
    """Unrelated questions should not reach AgentFlow workflows."""

    request = AgentQueryRequest(query="What is the capital of France?")

    plan = await router.create_plan(request)

    assert plan.intent is AgentIntent.OUT_OF_SCOPE
    assert plan.response_depth is ResponseDepth.BRIEF
    assert plan.requires_current_snapshot is False
    assert plan.requires_historical_lookup is False
    assert plan.requires_human_approval is False
    assert plan.may_execute_side_effect is False


@pytest.mark.anyio
async def test_routes_approved_github_pull_requests_as_github_question(
    router: AgentQueryRouter,
) -> None:
    """GitHub PR wording should not be overridden by approval terminology."""

    request = AgentQueryRequest(
        query="Show approved GitHub pull requests.",
        release_run_id=uuid4(),
    )

    plan = await router.create_plan(request)

    assert plan.intent is AgentIntent.GITHUB_PR_QUESTION
    assert plan.filters.sources == [RiskSourceFilter.GITHUB]


@pytest.mark.anyio
async def test_routes_running_workflow_question_as_workflow_status(
    router: AgentQueryRouter,
) -> None:
    """Workflow status wording should override generic 'why is' wording."""

    request = AgentQueryRequest(
        query="Why is the workflow still running?",
        release_run_id=uuid4(),
    )

    plan = await router.create_plan(request)

    assert plan.intent is AgentIntent.WORKFLOW_STATUS_QUESTION
    assert plan.response_depth is ResponseDepth.BRIEF
    assert plan.may_execute_side_effect is False


@pytest.mark.anyio
async def test_does_not_extract_high_severity_from_highlight(
    router: AgentQueryRouter,
) -> None:
    """Severity extraction must match complete words rather than substrings."""

    request = AgentQueryRequest(
        query="Highlight the Jira risks.",
        release_run_id=uuid4(),
    )

    plan = await router.create_plan(request)

    assert plan.filters.severities == []


@pytest.mark.anyio
async def test_routes_credit_score_question_out_of_scope(
    router: AgentQueryRouter,
) -> None:
    """A non-release score question must not enter AgentFlow workflows."""

    request = AgentQueryRequest(query="What is my credit score?")

    plan = await router.create_plan(request)

    assert plan.intent is AgentIntent.OUT_OF_SCOPE


@pytest.mark.anyio
async def test_routes_concert_ticket_question_out_of_scope(
    router: AgentQueryRouter,
) -> None:
    """A non-Jira use of ticket must not be classified as a Jira question."""

    request = AgentQueryRequest(query="How do I buy a concert ticket?")

    plan = await router.create_plan(request)

    assert plan.intent is AgentIntent.OUT_OF_SCOPE


@pytest.mark.anyio
async def test_extracts_complete_word_severity_filter(
    router: AgentQueryRouter,
) -> None:
    """A complete severity word should produce a structured filter."""

    request = AgentQueryRequest(
        query="Show high and critical Jira risks.",
        release_run_id=uuid4(),
    )

    plan = await router.create_plan(request)

    assert plan.filters.severities == ["critical", "high"]


@pytest.mark.anyio
async def test_routes_pr_number_as_github_question(
    router: AgentQueryRouter,
) -> None:
    """A PR abbreviation with a number should route to the GitHub PR intent."""

    request = AgentQueryRequest(
        query="What is happening with PR 42?",
        release_run_id=uuid4(),
    )

    plan = await router.create_plan(request)

    assert plan.intent is AgentIntent.GITHUB_PR_QUESTION
    assert plan.entity_references.pull_request_numbers == [42]
    assert plan.filters.sources == [RiskSourceFilter.GITHUB]
    assert plan.requires_current_snapshot is True


@pytest.mark.anyio
async def test_routes_jira_key_as_jira_ticket_question(
    router: AgentQueryRouter,
) -> None:
    """An explicit Jira key should route to the Jira ticket intent."""

    request = AgentQueryRequest(
        query="What is happening with PAY-102?",
        release_run_id=uuid4(),
    )

    plan = await router.create_plan(request)

    assert plan.intent is AgentIntent.JIRA_TICKET_QUESTION
    assert plan.entity_references.jira_issue_keys == ["PAY-102"]
    assert plan.filters.sources == [RiskSourceFilter.JIRA]
    assert plan.requires_current_snapshot is True
