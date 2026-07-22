"""Golden cases for deterministic dynamic-agent evaluation."""

from app.schemas.agent_query import AgentIntent
from app.schemas.agent_tool import AgentToolName
from app.services.agent_dynamic_evaluation_service import (
    DynamicAgentEvalCase,
)


def build_dynamic_agent_eval_cases() -> list[DynamicAgentEvalCase]:
    """Build representative routing, planning, and safety cases."""
    return [
        DynamicAgentEvalCase(
            name="knowledge_runbook_question",
            query="What does the payment runbook say about rollback?",
            expected_intent=AgentIntent.KNOWLEDGE_DOC_QUESTION,
            expected_tool_name=(
                AgentToolName.SEARCH_ENGINEERING_KNOWLEDGE
            ),
        ),
        DynamicAgentEvalCase(
            name="github_pull_request_question",
            query="What is happening with PR 42?",
            expected_intent=AgentIntent.GITHUB_PR_QUESTION,
            expected_tool_name=(
                AgentToolName.LOOKUP_GITHUB_PULL_REQUEST
            ),
            release_run_context_available=True,
        ),
        DynamicAgentEvalCase(
            name="jira_ticket_question",
            query="What is happening with PAY-102?",
            expected_intent=AgentIntent.JIRA_TICKET_QUESTION,
            expected_tool_name=AgentToolName.LOOKUP_JIRA_ISSUE,
            release_run_context_available=True,
        ),
        DynamicAgentEvalCase(
            name="approval_status_question",
            query="Has this release been approved?",
            expected_intent=AgentIntent.APPROVAL_STATUS_QUESTION,
            expected_tool_name=AgentToolName.LOOKUP_APPROVAL_STATUS,
            release_run_context_available=True,
        ),
        DynamicAgentEvalCase(
            name="slack_status_question",
            query="Was Slack already sent?",
            expected_intent=AgentIntent.SLACK_STATUS_QUESTION,
            expected_tool_name=AgentToolName.LOOKUP_SLACK_STATUS,
            release_run_context_available=True,
        ),
        DynamicAgentEvalCase(
            name="side_effect_request_requires_hitl",
            query="Can you send this to Slack?",
            expected_intent=AgentIntent.ACTION_REQUEST,
            dynamic_planning_allowed=False,
            expected_requires_human_approval=True,
            expected_may_execute_side_effect=True,
        ),
        DynamicAgentEvalCase(
            name="unrelated_question_stays_out_of_scope",
            query="What is the capital of France?",
            expected_intent=AgentIntent.OUT_OF_SCOPE,
            dynamic_planning_allowed=False,
        ),
    ]
