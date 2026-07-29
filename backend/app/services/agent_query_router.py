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

    _REFERENTIAL_ENTITY_PATTERN: Final[re.Pattern[str]] = re.compile(
        r"\b(?:it|this|that|this risk|that risk|the risk|the issue|"
        r"the pull request|the ticket)\b",
        re.IGNORECASE,
    )

    _PR_PATTERN: Final[re.Pattern[str]] = re.compile(
        r"\b(?:pr|pull request)\s*#?\s*(\d+)\b",
        re.IGNORECASE,
    )

    _JIRA_KEY_PATTERN: Final[re.Pattern[str]] = re.compile(
        r"\b(?:(?P<hyphen_prefix>(?i:[A-Z][A-Z0-9]+))\s*-\s*"
        r"(?P<hyphen_number>\d+)|"
        r"(?P<space_prefix>(?!PR\b)[A-Z][A-Z0-9]{2,})\s+"
        r"(?P<space_number>\d+))\b",
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
                "has an alert already been delivered to slack",
                "has the slack alert already been delivered",
                "has the slack alert been delivered",
                "already been delivered to slack",
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
            intent=AgentIntent.SIMILAR_PAST_RELEASE,
            response_depth=ResponseDepth.DEEP,
            phrases=(
                "most similar",
                "similar past release",
                "similar release",
                "similar to this one",
            ),
            routing_reason_code="matched_similar_past_release",
            priority=83,
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
                "only github risks",
                "only jira risks",
                "jira only",
                "github only",
                "github and jira",
                "jira and github",
                "show blockers",
                "what could block",
                "what is blocking",
                "what's blocking",
                "what blocks",
                "deployment blockers",
                "release blockers",
                "show critical",
                "critical risks",
                "high severity risks",
                "high-severity risks",
                "show high severity",
                "medium severity risks",
                "low severity risks",
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
        query_entities = self._extract_entities(request.query)
        entity_references = self._resolve_entity_references(
            normalized_query=normalized_query,
            query_entities=query_entities,
            context_entities=request.context_entity_references,
        )

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

        if (
            matched_rule is None
            and self._has_single_entity_reference(entity_references)
        ):
            matched_rule = IntentRule(
                intent=AgentIntent.EXPLAIN_SPECIFIC_RISK,
                response_depth=ResponseDepth.DEEP,
                phrases=("contextual_entity_follow_up",),
                routing_reason_code="matched_contextual_entity_follow_up",
                priority=70,
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
            filters=self._extract_filters(request.query),
            entity_references=entity_references,
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
        original_query: str,
    ) -> AgentQueryFilters:
        """Extract simple source and severity filters from the query."""

        normalized_query = self._normalize_query(original_query)
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
            or self._JIRA_KEY_PATTERN.search(original_query) is not None
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
            blockers_only=(
                re.search(
                    r"\bblock(?:s|ed|ing|er|ers)?\b",
                    normalized_query,
                )
                is not None
            ),
            open_items_only=(
                "open only" in normalized_query
                or "open items" in normalized_query
                or "currently open" in normalized_query
            ),
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
            {
                (
                    f"{match.group('hyphen_prefix').upper()}-"
                    f"{match.group('hyphen_number')}"
                    if match.group("hyphen_prefix") is not None
                    else (
                        f"{match.group('space_prefix')}-"
                        f"{match.group('space_number')}"
                    )
                )
                for match in self._JIRA_KEY_PATTERN.finditer(original_query)
            }
        )

        return AgentEntityReferences(
            pull_request_numbers=pull_request_numbers,
            jira_issue_keys=jira_issue_keys,
        )

    def _resolve_entity_references(
        self,
        *,
        normalized_query: str,
        query_entities: AgentEntityReferences,
        context_entities: AgentEntityReferences | None,
    ) -> AgentEntityReferences:
        """Resolve explicit entities before bounded follow-up context.

        Explicit identifiers in the current query always win. Context is used
        only for referential wording and only when it identifies exactly one
        persisted GitHub pull request or Jira issue.
        """

        if (
            query_entities.pull_request_numbers
            or query_entities.jira_issue_keys
            or query_entities.service_names
        ):
            return query_entities

        if (
            context_entities is None
            or self._REFERENTIAL_ENTITY_PATTERN.search(normalized_query) is None
            or not self._has_single_entity_reference(context_entities)
        ):
            return query_entities

        return AgentEntityReferences(
            pull_request_numbers=list(
                context_entities.pull_request_numbers
            ),
            jira_issue_keys=list(context_entities.jira_issue_keys),
        )

    @staticmethod
    def _has_single_entity_reference(
        entity_references: AgentEntityReferences,
    ) -> bool:
        """Return whether exactly one PR or Jira entity is available."""

        return (
            len(entity_references.pull_request_numbers)
            + len(entity_references.jira_issue_keys)
            == 1
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
