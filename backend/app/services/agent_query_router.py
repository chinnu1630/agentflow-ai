"""Deterministic natural-language query routing for AgentFlow AI.

This service converts a manager's natural-language release question into a
validated AgentQueryPlan. It performs classification only and does not execute
workflows, query external APIs, approve releases, or send Slack messages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from app.schemas.agent_query import (
    AgentEntityReferences,
    AgentIntent,
    AgentQueryFilters,
    AgentQueryPlan,
    AgentQueryRequest,
    ResponseDepth,
    RiskSourceFilter,
)


@dataclass(frozen=True, slots=True)
class IntentRule:
    """Immutable routing rule for deterministic intent classification."""

    intent: AgentIntent
    response_depth: ResponseDepth
    phrases: tuple[str, ...]
    routing_reason_code: str
    priority: int
    requires_current_snapshot: bool = False
    requires_historical_lookup: bool = False
    requires_human_approval: bool = False
    may_execute_side_effect: bool = False


class AgentQueryRouter:
    """Convert natural-language AgentFlow questions into safe query plans."""

    _WHITESPACE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\s+")

    _PR_PATTERN: Final[re.Pattern[str]] = re.compile(
        r"\b(?:pr|pull request)\s*#?\s*(\d+)\b",
        re.IGNORECASE,
    )

    _JIRA_KEY_PATTERN: Final[re.Pattern[str]] = re.compile(
        r"\b([A-Z][A-Z0-9]+-\d+)\b",
        re.IGNORECASE,
    )

    _RELEASE_CONTEXT_TERMS: Final[frozenset[str]] = frozenset(
        {
            "release",
            "risk",
            "risky",
            "deploy",
            "deployment",
            "github",
            "pull request",
            "jira",
            "bug",
            "blocker",
            "approval",
            "approved",
            "slack",
            "workflow",
            "incident",
            "runbook",
            "severity",
        }
    )

    _RULES: Final[tuple[IntentRule, ...]] = (
        IntentRule(
            intent=AgentIntent.ACTION_REQUEST,
            response_depth=ResponseDepth.ACTION_CONFIRMATION,
            phrases=(
                "send to slack",
                "send this to slack",
                "post to slack",
                "notify slack",
            ),
            routing_reason_code="matched_slack_action",
            priority=100,
            requires_current_snapshot=True,
            requires_human_approval=True,
            may_execute_side_effect=True,
        ),
        IntentRule(
            intent=AgentIntent.SLACK_STATUS_QUESTION,
            response_depth=ResponseDepth.BRIEF,
            phrases=(
                "was slack sent",
                "was the slack alert sent",
                "slack already sent",
                "was it sent to slack",
                "slack status",
            ),
            routing_reason_code="matched_slack_status",
            priority=95,
            requires_current_snapshot=True,
        ),
        IntentRule(
            intent=AgentIntent.APPROVAL_STATUS_QUESTION,
            response_depth=ResponseDepth.BRIEF,
            phrases=(
                "is it approved",
                "approval status",
                "pending approval",
                "was it rejected",
                "release approved",
                "was it approved",
                "has it been approved",
                "been approved",
            ),
            routing_reason_code="matched_approval_status",
            priority=90,
            requires_current_snapshot=True,
        ),
        IntentRule(
            intent=AgentIntent.COMPARE_WITH_PREVIOUS_RELEASE,
            response_depth=ResponseDepth.DEEP,
            phrases=(
                "compare with previous",
                "compare to previous",
                "previous release",
                "last release",
            ),
            routing_reason_code="matched_previous_release_comparison",
            priority=85,
            requires_current_snapshot=True,
            requires_historical_lookup=True,
        ),
        IntentRule(
            intent=AgentIntent.HISTORICAL_RISK_LOOKUP,
            response_depth=ResponseDepth.DEEP,
            phrases=(
                "did this happen before",
                "what happened last time",
                "happen before",
                "release history",
                "past risk",
            ),
            routing_reason_code="matched_historical_lookup",
            priority=80,
            requires_historical_lookup=True,
        ),
        IntentRule(
            intent=AgentIntent.EXPLAIN_RISK_SCORE,
            response_depth=ResponseDepth.DEEP,
            phrases=(
                "why high",
                "why critical",
                "explain the score",
                "risk score",
                "why was it scored",
            ),
            routing_reason_code="matched_risk_score_explanation",
            priority=75,
            requires_current_snapshot=True,
        ),
        IntentRule(
            intent=AgentIntent.EXPLAIN_SPECIFIC_RISK,
            response_depth=ResponseDepth.DEEP,
            phrases=(
                "why is",
                "why risky",
                "explain this risk",
                "what makes",
            ),
            routing_reason_code="matched_specific_risk_explanation",
            priority=70,
            requires_current_snapshot=True,
        ),
        IntentRule(
            intent=AgentIntent.FILTER_RISKS,
            response_depth=ResponseDepth.STANDARD,
            phrases=(
                "jira blockers only",
                "github risks only",
                "jira only",
                "github only",
                "show blockers",
                "show critical",
            ),
            routing_reason_code="matched_risk_filter",
            priority=65,
            requires_current_snapshot=True,
        ),
        IntentRule(
            intent=AgentIntent.WORKFLOW_STATUS_QUESTION,
            response_depth=ResponseDepth.BRIEF,
            phrases=(
                "workflow status",
                "is workflow complete",
                "analysis status",
                "still running",
            ),
            routing_reason_code="matched_workflow_status",
            priority=72,
            requires_current_snapshot=True,
        ),
        IntentRule(
            intent=AgentIntent.KNOWLEDGE_DOC_QUESTION,
            response_depth=ResponseDepth.STANDARD,
            phrases=(
                "runbook",
                "postmortem",
                "post-mortem",
                "release checklist",
                "engineering document",
            ),
            routing_reason_code="matched_knowledge_question",
            priority=55,
        ),
        IntentRule(
            intent=AgentIntent.GITHUB_PR_QUESTION,
            response_depth=ResponseDepth.STANDARD,
            phrases=(
                "github",
                "pull request",
                "review status",
                "ci status",
            ),
            routing_reason_code="matched_github_question",
            priority=50,
            requires_current_snapshot=True,
        ),
        IntentRule(
            intent=AgentIntent.JIRA_TICKET_QUESTION,
            response_depth=ResponseDepth.STANDARD,
            phrases=(
                "jira",
                "jira ticket",
                "bug",
                "sprint",
            ),
            routing_reason_code="matched_jira_question",
            priority=45,
            requires_current_snapshot=True,
        ),
        IntentRule(
            intent=AgentIntent.RELEASE_RISK_SUMMARY,
            response_depth=ResponseDepth.STANDARD,
            phrases=(
                "biggest release risks",
                "release risks",
                "what are the risks",
                "is it safe to deploy",
                "safe to deploy",
                "release readiness",
            ),
            routing_reason_code="matched_release_risk_summary",
            priority=40,
            requires_current_snapshot=True,
        ),
    )

    async def create_plan(
        self,
        request: AgentQueryRequest,
    ) -> AgentQueryPlan:
        """Convert a validated natural-language request into a query plan.

        Args:
            request: Natural-language query and optional workflow context IDs.

        Returns:
            A validated plan describing the intended AgentFlow operation.
        """

        normalized_query = self._normalize_query(request.query)
        matched_rule = self._find_matching_rule(normalized_query)

        if (
            matched_rule is None
            and self._PR_PATTERN.search(request.query) is not None
        ):
            matched_rule = IntentRule(
                intent=AgentIntent.GITHUB_PR_QUESTION,
                response_depth=ResponseDepth.STANDARD,
                phrases=("explicit_pr_reference",),
                routing_reason_code="matched_github_pr_reference",
                priority=50,
                requires_current_snapshot=True,
            )

        if (
            matched_rule is None
            and self._JIRA_KEY_PATTERN.search(request.query) is not None
        ):
            matched_rule = IntentRule(
                intent=AgentIntent.JIRA_TICKET_QUESTION,
                response_depth=ResponseDepth.STANDARD,
                phrases=("explicit_jira_reference",),
                routing_reason_code="matched_jira_issue_reference",
                priority=45,
                requires_current_snapshot=True,
            )

        if matched_rule is None:
            if not self._contains_release_context(normalized_query):
                return self._create_out_of_scope_plan(request)

            matched_rule = IntentRule(
                intent=AgentIntent.RELEASE_RISK_SUMMARY,
                response_depth=ResponseDepth.STANDARD,
                phrases=("release_context",),
                routing_reason_code="matched_general_release_context",
                priority=10,
                requires_current_snapshot=True,
            )

        return AgentQueryPlan(
            intent=matched_rule.intent,
            response_depth=matched_rule.response_depth,
            confidence=self._calculate_confidence(
                normalized_query=normalized_query,
                matched_rule=matched_rule,
            ),
            release_run_id=request.release_run_id,
            conversation_session_id=request.conversation_session_id,
            filters=self._extract_filters(normalized_query),
            entity_references=self._extract_entities(request.query),
            requires_current_snapshot=(matched_rule.requires_current_snapshot),
            requires_historical_lookup=(matched_rule.requires_historical_lookup),
            requires_human_approval=(matched_rule.requires_human_approval),
            may_execute_side_effect=(matched_rule.may_execute_side_effect),
            routing_reason_code=matched_rule.routing_reason_code,
        )

    def _find_matching_rule(
        self,
        normalized_query: str,
    ) -> IntentRule | None:
        """Return the highest-priority matching routing rule."""

        ordered_rules = sorted(
            self._RULES,
            key=lambda rule: rule.priority,
            reverse=True,
        )

        for rule in ordered_rules:
            if any(phrase in normalized_query for phrase in rule.phrases):
                return rule

        return None

    def _contains_release_context(
        self,
        normalized_query: str,
    ) -> bool:
        """Return whether a query contains AgentFlow domain terminology."""

        return any(term in normalized_query for term in self._RELEASE_CONTEXT_TERMS)

    def _extract_filters(
        self,
        normalized_query: str,
    ) -> AgentQueryFilters:
        """Extract simple source and severity filters from the query."""

        sources: list[RiskSourceFilter] = []

        if (
            "github" in normalized_query
            or "pull request" in normalized_query
            or self._PR_PATTERN.search(normalized_query) is not None
        ):
            sources.append(RiskSourceFilter.GITHUB)

        if (
            "jira" in normalized_query
            or "ticket" in normalized_query
            or self._JIRA_KEY_PATTERN.search(normalized_query) is not None
        ):
            sources.append(RiskSourceFilter.JIRA)

        if (
            "runbook" in normalized_query
            or "postmortem" in normalized_query
            or "post-mortem" in normalized_query
            or "document" in normalized_query
        ):
            sources.append(RiskSourceFilter.KNOWLEDGE)

        severities = [
            severity
            for severity in ("critical", "high", "medium", "low")
            if re.search(
                rf"\b{re.escape(severity)}\b",
                normalized_query,
            )
        ]

        return AgentQueryFilters(
            sources=sources,
            severities=severities,
            blockers_only="blocker" in normalized_query,
            open_items_only=("open only" in normalized_query or "open items" in normalized_query),
        )

    def _extract_entities(
        self,
        original_query: str,
    ) -> AgentEntityReferences:
        """Extract candidate pull-request and Jira identifiers."""

        pull_request_numbers = sorted(
            {int(match.group(1)) for match in self._PR_PATTERN.finditer(original_query)}
        )

        jira_issue_keys = sorted(
            {match.group(1).upper() for match in self._JIRA_KEY_PATTERN.finditer(original_query)}
        )

        return AgentEntityReferences(
            pull_request_numbers=pull_request_numbers,
            jira_issue_keys=jira_issue_keys,
        )

    def _calculate_confidence(
        self,
        normalized_query: str,
        matched_rule: IntentRule,
    ) -> float:
        """Calculate a deterministic confidence score."""

        match_count = sum(phrase in normalized_query for phrase in matched_rule.phrases)

        if match_count >= 2:
            return 0.98

        if match_count == 1:
            return 0.93

        return 0.75

    def _create_out_of_scope_plan(
        self,
        request: AgentQueryRequest,
    ) -> AgentQueryPlan:
        """Create a safe plan for unrelated user questions."""

        return AgentQueryPlan(
            intent=AgentIntent.OUT_OF_SCOPE,
            response_depth=ResponseDepth.BRIEF,
            confidence=0.99,
            release_run_id=request.release_run_id,
            conversation_session_id=request.conversation_session_id,
            requires_current_snapshot=False,
            requires_historical_lookup=False,
            requires_human_approval=False,
            may_execute_side_effect=False,
            routing_reason_code="no_release_workflow_context",
        )

    def _normalize_query(self, query: str) -> str:
        """Normalize whitespace and casing for deterministic matching."""

        normalized_query = query.casefold()
        return self._WHITESPACE_PATTERN.sub(
            " ",
            normalized_query,
        ).strip()
