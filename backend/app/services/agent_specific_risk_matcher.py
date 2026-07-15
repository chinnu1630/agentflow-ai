"""Match follow-up questions to one trusted persisted release risk."""

from __future__ import annotations

import logging
import re
from typing import Final

from app.schemas.agent_query import AgentQueryPlan
from app.schemas.risk import (
    ReleaseRiskSummaryItemResponse,
    ReleaseRunRiskResponse,
)

logger = logging.getLogger(__name__)


class AgentSpecificRiskMatcherError(RuntimeError):
    """Base error raised while matching a specific persisted risk."""


class AgentSpecificRiskNotFoundError(AgentSpecificRiskMatcherError):
    """Raised when no persisted risk matches the follow-up query."""


class AgentSpecificRiskMatcher:
    """Select one risk from a validated persisted release-risk snapshot."""

    _TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+")
    _ORDINAL_INDEXES: Final[dict[str, int]] = {
        "first": 0,
        "1st": 0,
        "second": 1,
        "2nd": 1,
        "third": 2,
        "3rd": 2,
        "fourth": 3,
        "4th": 3,
        "fifth": 4,
        "5th": 4,
    }
    _IGNORED_QUERY_TERMS: Final[frozenset[str]] = frozenset(
        {
            "a",
            "an",
            "and",
            "dangerous",
            "evidence",
            "explain",
            "for",
            "is",
            "it",
            "of",
            "risk",
            "risky",
            "supports",
            "the",
            "this",
            "what",
            "why",
        }
    )

    def __init__(self, request_id: str) -> None:
        """Initialize the matcher.

        Args:
            request_id: Request identifier included in structured logs.
        """

        self._request_id = request_id

    def match(
        self,
        *,
        query: str,
        plan: AgentQueryPlan,
        release_risk: ReleaseRunRiskResponse,
    ) -> ReleaseRiskSummaryItemResponse:
        """Match a query to one persisted ranked risk.

        Matching priority is:

        1. Explicit GitHub pull-request number.
        2. Explicit Jira issue key.
        3. Ordinal position such as "first risk".
        4. Token overlap with source ID, title, and reason.

        Args:
            query: Original natural-language follow-up question.
            plan: Validated query plan containing extracted entities.
            release_risk: Trusted persisted release-risk snapshot.

        Returns:
            The best matching persisted risk.

        Raises:
            AgentSpecificRiskNotFoundError: When no risk matches the query.
        """

        risks = release_risk.release_summary.top_risks

        matched_risk = (
            self._match_pull_request(plan=plan, risks=risks)
            or self._match_jira_issue(plan=plan, risks=risks)
            or self._match_ordinal(query=query, risks=risks)
            or self._match_text(query=query, risks=risks)
        )

        if matched_risk is None:
            logger.warning(
                "agent_specific_risk_not_found",
                extra={
                    "run_id": self._request_id,
                    "release_run_id": str(release_risk.release_run.id),
                    "risk_count": len(risks),
                    "intent": plan.intent.value,
                },
            )
            raise AgentSpecificRiskNotFoundError(
                "No persisted risk matched the query."
            )

        logger.info(
            "agent_specific_risk_matched",
            extra={
                "run_id": self._request_id,
                "release_run_id": str(release_risk.release_run.id),
                "source_type": matched_risk.source_type,
                "source_id": matched_risk.source_id,
                "intent": plan.intent.value,
            },
        )

        return matched_risk

    @staticmethod
    def _match_pull_request(
        *,
        plan: AgentQueryPlan,
        risks: list[ReleaseRiskSummaryItemResponse],
    ) -> ReleaseRiskSummaryItemResponse | None:
        """Match an explicitly extracted GitHub pull-request number."""

        requested_ids = {
            candidate
            for number in plan.entity_references.pull_request_numbers
            for candidate in (str(number), f"PR-{number}")
        }

        return next(
            (
                risk
                for risk in risks
                if risk.source_type == "github_pull_request"
                and risk.source_id.upper() in requested_ids
            ),
            None,
        )

    @staticmethod
    def _match_jira_issue(
        *,
        plan: AgentQueryPlan,
        risks: list[ReleaseRiskSummaryItemResponse],
    ) -> ReleaseRiskSummaryItemResponse | None:
        """Match an explicitly extracted Jira issue key."""

        requested_ids = {
            issue_key.upper()
            for issue_key in plan.entity_references.jira_issue_keys
        }

        return next(
            (
                risk
                for risk in risks
                if risk.source_type == "jira_issue"
                and risk.source_id.upper() in requested_ids
            ),
            None,
        )

    def _match_ordinal(
        self,
        *,
        query: str,
        risks: list[ReleaseRiskSummaryItemResponse],
    ) -> ReleaseRiskSummaryItemResponse | None:
        """Match an ordinal reference against ranked risk order."""

        normalized_query = query.casefold()

        for ordinal, index in self._ORDINAL_INDEXES.items():
            if re.search(rf"\b{re.escape(ordinal)}\b", normalized_query):
                if index < len(risks):
                    return risks[index]

                return None

        return None

    def _match_text(
        self,
        *,
        query: str,
        risks: list[ReleaseRiskSummaryItemResponse],
    ) -> ReleaseRiskSummaryItemResponse | None:
        """Match meaningful query terms against persisted risk text."""

        query_terms = self._tokenize(query) - self._IGNORED_QUERY_TERMS

        if not query_terms:
            return None

        best_risk: ReleaseRiskSummaryItemResponse | None = None
        best_score = 0

        for risk in risks:
            title_terms = self._tokenize(risk.title)
            reason_terms = self._tokenize(risk.reason)
            source_terms = self._tokenize(risk.source_id)

            score = (
                3 * len(query_terms & title_terms)
                + 2 * len(query_terms & source_terms)
                + len(query_terms & reason_terms)
            )

            if score > best_score:
                best_score = score
                best_risk = risk

        return best_risk if best_score > 0 else None

    def _tokenize(self, value: str) -> set[str]:
        """Return normalized alphanumeric terms from text."""

        return set(self._TOKEN_PATTERN.findall(value.casefold()))
