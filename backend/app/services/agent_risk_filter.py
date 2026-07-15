"""Filter trusted persisted release risks using a validated query plan."""

from __future__ import annotations

import logging
from typing import Final

from app.schemas.agent_query import AgentQueryPlan, RiskSourceFilter
from app.schemas.risk import (
    ReleaseRiskSummaryItemResponse,
    ReleaseRunRiskResponse,
)

logger = logging.getLogger(__name__)


class AgentRiskFilter:
    """Apply deterministic filters to persisted ranked release risks."""

    _CLOSED_STATUSES: Final[frozenset[str]] = frozenset(
        {
            "closed",
            "complete",
            "completed",
            "done",
            "merged",
            "resolved",
        }
    )

    def __init__(self, request_id: str) -> None:
        """Initialize the risk filter.

        Args:
            request_id: Request identifier included in structured logs.
        """

        self._request_id = request_id

    def filter(
        self,
        *,
        plan: AgentQueryPlan,
        release_risk: ReleaseRunRiskResponse,
    ) -> list[ReleaseRiskSummaryItemResponse]:
        """Return persisted risks matching all requested filters.

        Filter categories use AND semantics. Multiple values inside one
        category, such as source or severity, use OR semantics.

        Args:
            plan: Validated query plan containing structured filters.
            release_risk: Trusted persisted release-risk snapshot.

        Returns:
            Ranked risks that satisfy every configured filter.
        """

        source_values = {
            self._source_value(source)
            for source in plan.filters.sources
        }
        severity_values = {
            severity.casefold()
            for severity in plan.filters.severities
        }

        filtered_risks = [
            risk
            for risk in release_risk.release_summary.top_risks
            if self._matches_sources(
                risk=risk,
                source_values=source_values,
            )
            and self._matches_severities(
                risk=risk,
                severity_values=severity_values,
            )
            and self._matches_blocker_filter(
                risk=risk,
                blockers_only=plan.filters.blockers_only,
            )
            and self._matches_open_filter(
                risk=risk,
                open_items_only=plan.filters.open_items_only,
            )
        ]

        logger.info(
            "agent_risks_filtered",
            extra={
                "run_id": self._request_id,
                "release_run_id": str(release_risk.release_run.id),
                "intent": plan.intent.value,
                "input_risk_count": len(
                    release_risk.release_summary.top_risks
                ),
                "filtered_risk_count": len(filtered_risks),
                "source_filters": sorted(source_values),
                "severity_filters": sorted(severity_values),
                "blockers_only": plan.filters.blockers_only,
                "open_items_only": plan.filters.open_items_only,
            },
        )

        return filtered_risks

    @staticmethod
    def _source_value(source: RiskSourceFilter) -> str:
        """Return the persisted source value for a source filter."""

        return source.value

    @staticmethod
    def _matches_sources(
        *,
        risk: ReleaseRiskSummaryItemResponse,
        source_values: set[str],
    ) -> bool:
        """Return whether a risk matches the configured source filters."""

        return not source_values or risk.source in source_values

    @staticmethod
    def _matches_severities(
        *,
        risk: ReleaseRiskSummaryItemResponse,
        severity_values: set[str],
    ) -> bool:
        """Return whether a risk matches the configured severities."""

        return (
            not severity_values
            or risk.severity.value.casefold() in severity_values
        )

    @staticmethod
    def _matches_blocker_filter(
        *,
        risk: ReleaseRiskSummaryItemResponse,
        blockers_only: bool,
    ) -> bool:
        """Return whether a risk satisfies blocker-only filtering."""

        if not blockers_only:
            return True

        evidence_status = str(
            risk.evidence.get("status", "")
        ).casefold()
        is_release_blocker = (
            risk.evidence.get("is_blocking_release") is True
        )

        return evidence_status == "blocked" or is_release_blocker

    def _matches_open_filter(
        self,
        *,
        risk: ReleaseRiskSummaryItemResponse,
        open_items_only: bool,
    ) -> bool:
        """Return whether a risk is still open according to persisted evidence."""

        if not open_items_only:
            return True

        evidence_status = str(
            risk.evidence.get("status", "")
        ).casefold()

        if not evidence_status:
            return True

        return evidence_status not in self._CLOSED_STATUSES
