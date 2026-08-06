"""Unit tests for the deterministic AgentFlow query router."""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.schemas.agent_query import (
    AgentEntityReferences,
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


@pytest.mark.parametrize(
    "query",
    [
        "Explain the highest risk.",
        "Explain the top risk.",
        "Explain the first risk.",
        "Explain the number one risk.",
    ],
)
@pytest.mark.anyio
async def test_routes_ranked_risk_explanation(
    router: AgentQueryRouter,
    query: str,
) -> None:
    """A ranked-risk reference should request one deep risk explanation."""

    request = AgentQueryRequest(
        query=query,
        release_run_id=uuid4(),
    )

    plan = await router.create_plan(request)

    assert plan.intent is AgentIntent.EXPLAIN_SPECIFIC_RISK
    assert plan.response_depth is ResponseDepth.DEEP
    assert plan.requires_current_snapshot is True



@pytest.mark.parametrize(
    ("query", "expected_source"),
    [
        ("Show only GitHub risks.", RiskSourceFilter.GITHUB),
        (
            "Show only GitHub release risks.",
            RiskSourceFilter.GITHUB,
        ),
        ("Show only Jira risks.", RiskSourceFilter.JIRA),
    ],
)
@pytest.mark.anyio
async def test_routes_source_only_risk_wording_as_filter(
    router: AgentQueryRouter,
    query: str,
    expected_source: RiskSourceFilter,
) -> None:
    """Natural source-only wording should not require a specific entity ID."""

    request = AgentQueryRequest(
        query=query,
        release_run_id=uuid4(),
    )

    plan = await router.create_plan(request)

    assert plan.intent is AgentIntent.FILTER_RISKS
    assert plan.filters.sources == [expected_source]
    assert plan.requires_current_snapshot is True


@pytest.mark.parametrize(
    ("query", "expected_source"),
    [
        ("Show all open Jira risks.", RiskSourceFilter.JIRA),
        ("What are the pending Jira tickets?", RiskSourceFilter.JIRA),
        ("Show unresolved Jira issues.", RiskSourceFilter.JIRA),
    ],
)
@pytest.mark.anyio
async def test_routes_open_jira_collection_wording_as_filter(
    router: AgentQueryRouter,
    query: str,
    expected_source: RiskSourceFilter,
) -> None:
    """Open Jira collection wording should produce a bounded risk filter."""

    request = AgentQueryRequest(
        query=query,
        release_run_id=uuid4(),
    )

    plan = await router.create_plan(request)

    assert plan.intent is AgentIntent.FILTER_RISKS
    assert plan.filters.sources == [expected_source]
    assert plan.filters.open_items_only is True
    assert plan.routing_reason_code == "matched_open_jira_filter"


@pytest.mark.anyio
async def test_pending_approval_wording_keeps_approval_intent(
    router: AgentQueryRouter,
) -> None:
    """Open Jira wording must not override a higher-priority approval query."""

    request = AgentQueryRequest(
        query="What is the pending approval status for this Jira release?",
        release_run_id=uuid4(),
    )

    plan = await router.create_plan(request)

    assert plan.intent is AgentIntent.APPROVAL_STATUS_QUESTION
    assert plan.routing_reason_code == "matched_approval_status"


@pytest.mark.anyio
async def test_pending_specific_jira_issue_keeps_ticket_intent(
    router: AgentQueryRouter,
) -> None:
    """An explicit Jira key should remain a specific-ticket question."""

    request = AgentQueryRequest(
        query="Is Jira ticket SCRUM-1 still pending?",
        release_run_id=uuid4(),
    )

    plan = await router.create_plan(request)

    assert plan.intent is AgentIntent.JIRA_TICKET_QUESTION
    assert plan.entity_references.jira_issue_keys == ["SCRUM-1"]


@pytest.mark.anyio
async def test_routes_open_github_and_jira_question_as_filter(
    router: AgentQueryRouter,
) -> None:
    """Combined GitHub and Jira wording should not require a specific PR."""

    request = AgentQueryRequest(
        query=(
            "Which currently open GitHub and Jira items could affect "
            "this week's release?"
        ),
        release_run_id=uuid4(),
    )

    plan = await router.create_plan(request)

    assert plan.intent is AgentIntent.FILTER_RISKS
    assert set(plan.filters.sources) == {
        RiskSourceFilter.GITHUB,
        RiskSourceFilter.JIRA,
    }
    assert plan.filters.open_items_only is True
    assert plan.requires_current_snapshot is True


@pytest.mark.anyio
async def test_routes_deployment_blocker_question_as_risk_filter(
    router: AgentQueryRouter,
) -> None:
    """Natural deployment-blocker wording should request blocker-only risks."""

    request = AgentQueryRequest(
        query="What could block this week's deployment?",
        release_run_id=uuid4(),
    )

    plan = await router.create_plan(request)

    assert plan.intent is AgentIntent.FILTER_RISKS
    assert plan.filters.blockers_only is True
    assert plan.requires_current_snapshot is True
    assert plan.may_execute_side_effect is False


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
async def test_routes_natural_slack_delivery_status_wording(
    router: AgentQueryRouter,
) -> None:
    """Natural delivery wording should use the read-only Slack status route."""

    request = AgentQueryRequest(
        query="Has an alert already been delivered to Slack?",
        release_run_id=uuid4(),
    )

    plan = await router.create_plan(request)

    assert plan.intent is AgentIntent.SLACK_STATUS_QUESTION
    assert plan.response_depth is ResponseDepth.BRIEF
    assert plan.requires_human_approval is False
    assert plan.may_execute_side_effect is False


@pytest.mark.parametrize(
    ("query", "expected_severity"),
    [
        ("Show high severity risks.", "high"),
        ("Show critical risks.", "critical"),
        ("Show only critical Jira risks.", "critical"),
        ("Show only high-severity Jira risks.", "high"),
        ("Show medium severity risks.", "medium"),
        ("Show low severity risks.", "low"),
    ],
)
@pytest.mark.anyio
async def test_routes_natural_severity_filters(
    router: AgentQueryRouter,
    query: str,
    expected_severity: str,
) -> None:
    """Severity-only wording should filter the persisted release snapshot."""

    request = AgentQueryRequest(
        query=query,
        release_run_id=uuid4(),
    )

    plan = await router.create_plan(request)

    assert plan.intent is AgentIntent.FILTER_RISKS
    assert plan.filters.severities == [expected_severity]
    assert plan.requires_current_snapshot is True


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


@pytest.mark.anyio
async def test_routes_lowercase_hyphenated_jira_key(
    router: AgentQueryRouter,
) -> None:
    """Canonical Jira keys should remain case-insensitive."""

    request = AgentQueryRequest(
        query="What is happening with scrum-2?",
        release_run_id=uuid4(),
    )

    plan = await router.create_plan(request)

    assert plan.intent is AgentIntent.JIRA_TICKET_QUESTION
    assert plan.entity_references.jira_issue_keys == ["SCRUM-2"]
    assert plan.filters.sources == [RiskSourceFilter.JIRA]


@pytest.mark.anyio
async def test_routes_spaced_jira_key_as_jira_ticket_question(
    router: AgentQueryRouter,
) -> None:
    """A naturally spaced Jira key should normalize to canonical form."""

    request = AgentQueryRequest(
        query="What about SCRUM 2?",
        release_run_id=uuid4(),
    )

    plan = await router.create_plan(request)

    assert plan.intent is AgentIntent.JIRA_TICKET_QUESTION
    assert plan.entity_references.jira_issue_keys == ["SCRUM-2"]
    assert plan.filters.sources == [RiskSourceFilter.JIRA]
    assert plan.requires_current_snapshot is True


@pytest.mark.anyio
async def test_routes_release_approval_question_as_approval_status(
    router: AgentQueryRouter,
) -> None:
    """Natural approval wording should route to approval-status lookup."""

    release_run_id = uuid4()
    request = AgentQueryRequest(
        query="Has this release been approved?",
        release_run_id=release_run_id,
    )

    plan = await router.create_plan(request)

    assert plan.intent is AgentIntent.APPROVAL_STATUS_QUESTION
    assert plan.response_depth is ResponseDepth.BRIEF
    assert plan.release_run_id == release_run_id
    assert plan.requires_current_snapshot is True
    assert plan.may_execute_side_effect is False

@pytest.mark.anyio
async def test_routes_similar_past_release_question() -> None:
    """Similar-release questions should require current and historical context."""
    router = AgentQueryRouter()

    plan = await router.create_plan(
        AgentQueryRequest(
            query="Which past release was most similar to this one?",
        )
    )

    assert plan.intent is AgentIntent.SIMILAR_PAST_RELEASE
    assert plan.response_depth is ResponseDepth.DEEP
    assert plan.requires_current_snapshot is True
    assert plan.requires_historical_lookup is True
    assert plan.may_execute_side_effect is False

@pytest.mark.anyio
async def test_reuses_pr_context_for_referential_risk_question(
    router: AgentQueryRouter,
) -> None:
    """A pronoun follow-up should reuse one validated PR reference."""

    plan = await router.create_plan(
        AgentQueryRequest(
            query="Why is it risky?",
            release_run_id=uuid4(),
            context_entity_references=AgentEntityReferences(
                pull_request_numbers=[4],
            ),
        )
    )

    assert plan.intent is AgentIntent.EXPLAIN_SPECIFIC_RISK
    assert plan.entity_references.pull_request_numbers == [4]
    assert plan.entity_references.jira_issue_keys == []
    assert plan.requires_current_snapshot is True


@pytest.mark.anyio
async def test_routes_generic_pronoun_follow_up_from_jira_context(
    router: AgentQueryRouter,
) -> None:
    """A generic pronoun question should become a focused risk follow-up."""

    plan = await router.create_plan(
        AgentQueryRequest(
            query="What should the manager do about it?",
            release_run_id=uuid4(),
            context_entity_references=AgentEntityReferences(
                jira_issue_keys=["SCRUM-2"],
            ),
        )
    )

    assert plan.intent is AgentIntent.EXPLAIN_SPECIFIC_RISK
    assert plan.entity_references.pull_request_numbers == []
    assert plan.entity_references.jira_issue_keys == ["SCRUM-2"]
    assert plan.routing_reason_code == "matched_contextual_entity_follow_up"


@pytest.mark.anyio
async def test_explicit_entity_overrides_previous_context(
    router: AgentQueryRouter,
) -> None:
    """A newly named PR must replace, rather than merge with, old context."""

    plan = await router.create_plan(
        AgentQueryRequest(
            query="Tell me about PR 5.",
            release_run_id=uuid4(),
            context_entity_references=AgentEntityReferences(
                pull_request_numbers=[4],
            ),
        )
    )

    assert plan.intent is AgentIntent.GITHUB_PR_QUESTION
    assert plan.entity_references.pull_request_numbers == [5]
    assert plan.entity_references.jira_issue_keys == []


@pytest.mark.parametrize(
    "query",
    [
        "What recovery steps are documented for payment API timeouts?",
        "What monitoring checks should be completed after a rollback?",
    ],
)
@pytest.mark.anyio
async def test_routes_implicit_operational_document_questions_as_knowledge(
    router: AgentQueryRouter,
    query: str,
) -> None:
    """Operational-document wording should route to knowledge retrieval."""

    plan = await router.create_plan(AgentQueryRequest(query=query))

    assert plan.intent is AgentIntent.KNOWLEDGE_DOC_QUESTION
    assert plan.response_depth is ResponseDepth.STANDARD
    assert plan.requires_current_snapshot is False
    assert plan.requires_historical_lookup is False
    assert plan.routing_reason_code == "matched_implicit_knowledge_question"


@pytest.mark.anyio
async def test_does_not_route_generic_rollback_question_as_knowledge(
    router: AgentQueryRouter,
) -> None:
    """Rollback wording alone must not trigger engineering-document retrieval."""

    plan = await router.create_plan(
        AgentQueryRequest(query="Could a rollback affect this release?")
    )

    assert plan.intent is AgentIntent.RELEASE_RISK_SUMMARY
    assert plan.requires_current_snapshot is True
    assert plan.routing_reason_code == "matched_general_release_context"
