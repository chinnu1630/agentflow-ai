"""Define safe fresh-or-reuse policy for AgentFlow manager queries."""

from __future__ import annotations

from typing import Final

from app.schemas.agent_query import (
    AgentIntent,
    AgentQueryPlan,
    AgentQueryRequest,
)

ANALYTICAL_RELEASE_INTENTS: Final[frozenset[AgentIntent]] = frozenset(
    {
        AgentIntent.RELEASE_RISK_SUMMARY,
        AgentIntent.EXPLAIN_RISK_SCORE,
        AgentIntent.EXPLAIN_SPECIFIC_RISK,
        AgentIntent.FILTER_RISKS,
        AgentIntent.GITHUB_PR_QUESTION,
        AgentIntent.JIRA_TICKET_QUESTION,
    }
)

PERSISTED_ONLY_RELEASE_INTENTS: Final[frozenset[AgentIntent]] = frozenset(
    {
        AgentIntent.WORKFLOW_STATUS_QUESTION,
        AgentIntent.APPROVAL_STATUS_QUESTION,
        AgentIntent.SLACK_STATUS_QUESTION,
        AgentIntent.HISTORICAL_RISK_LOOKUP,
        AgentIntent.SIMILAR_PAST_RELEASE,
        AgentIntent.COMPARE_WITH_PREVIOUS_RELEASE,
        AgentIntent.ACTION_REQUEST,
    }
)


def has_explicit_release_run_context(
    request: AgentQueryRequest,
    plan: AgentQueryPlan,
) -> bool:
    """Return whether the caller explicitly identified a release run."""

    return (
        request.release_run_id is not None
        or plan.release_run_id is not None
    )


def should_resolve_persisted_context(
    request: AgentQueryRequest,
    plan: AgentQueryPlan,
) -> bool:
    """Return whether execution must read a persisted release snapshot.

    Analytical queries reuse a snapshot only when the caller explicitly
    supplies a release-run ID. Operational and historical queries always
    require persisted context and therefore enter the strict resolver, which
    rejects missing IDs instead of silently selecting a potentially stale run.
    """

    if plan.intent in PERSISTED_ONLY_RELEASE_INTENTS:
        return True

    return (
        plan.intent in ANALYTICAL_RELEASE_INTENTS
        and has_explicit_release_run_context(request, plan)
    )
