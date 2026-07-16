"""Compose current persisted release-risk detail responses."""

from __future__ import annotations

import logging

from app.schemas.agent_query import AgentCitation, AgentQueryPlan, AgentQueryResponse
from app.schemas.risk import (
    JiraIssueRiskResponse,
    PullRequestRiskResponse,
    ReleaseRiskSummaryItemResponse,
    ReleaseRunRiskResponse,
)

logger = logging.getLogger(__name__)


class AgentCurrentRiskResponseComposerMixin:
    """Compose deterministic current-risk detail responses."""

    _request_id: str

    def compose_specific_risk(
        self,
        *,
        plan: AgentQueryPlan,
        release_risk: ReleaseRunRiskResponse,
        selected_risk: ReleaseRiskSummaryItemResponse,
    ) -> AgentQueryResponse:
        """Compose a focused explanation for one persisted release risk.

        Args:
            plan: Validated specific-risk query plan.
            release_risk: Trusted persisted release-risk snapshot.
            selected_risk: Risk selected from the persisted ranked risks.

        Returns:
            Focused response containing only the selected risk citation.
        """

        severity = selected_risk.severity.value.replace("_", " ")
        score_percentage = round(selected_risk.score * 100)

        answer = (
            f"{selected_risk.title} is a {severity} severity risk with a "
            f"{score_percentage}% item score. Reason: {selected_risk.reason}"
        )

        if selected_risk.evidence:
            evidence_items = [
                f"{key.replace('_', ' ')}: {value}"
                for key, value in sorted(selected_risk.evidence.items())
            ]
            answer += f" Evidence: {'; '.join(evidence_items)}."

        if release_risk.approval_required is True:
            answer += (
                " Human approval is required before any downstream "
                "release notification or Slack action."
            )

        citation = AgentCitation(
            source=selected_risk.source,
            source_type=selected_risk.source_type,
            source_id=selected_risk.source_id,
            title=selected_risk.title,
            source_url=selected_risk.source_url,
        )

        logger.info(
            "agent_specific_risk_response_composed",
            extra={
                "run_id": self._request_id,
                "release_run_id": str(release_risk.release_run.id),
                "intent": plan.intent.value,
                "source_type": selected_risk.source_type,
                "source_id": selected_risk.source_id,
                "citation_count": 1,
                "approval_required": release_risk.approval_required is True,
            },
        )

        return AgentQueryResponse(
            answer=answer,
            plan=plan,
            release_risk=release_risk,
            citations=[citation],
            approval_required=release_risk.approval_required is True,
        )

    def compose_filtered_risks(
        self,
        *,
        plan: AgentQueryPlan,
        release_risk: ReleaseRunRiskResponse,
        selected_risks: list[ReleaseRiskSummaryItemResponse],
    ) -> AgentQueryResponse:
        """Compose a response containing only risks matching query filters.

        Args:
            plan: Validated risk-filter query plan.
            release_risk: Trusted persisted release-risk snapshot.
            selected_risks: Persisted risks matching all requested filters.

        Returns:
            Filtered conversational response with matching citations.
        """

        risk_count = len(selected_risks)
        risk_label = "risk" if risk_count == 1 else "risks"

        if selected_risks:
            risk_lines = [
                (
                    f"{index}. {risk.title}: {risk.reason} "
                    f"({risk.severity.value} severity)."
                )
                for index, risk in enumerate(selected_risks, start=1)
            ]
            answer = (
                f"Found {risk_count} matching {risk_label}. "
                + " ".join(risk_lines)
            )
        else:
            answer = "No persisted release risks matched the requested filters."

        citations = [
            AgentCitation(
                source=risk.source,
                source_type=risk.source_type,
                source_id=risk.source_id,
                title=risk.title,
                source_url=risk.source_url,
            )
            for risk in selected_risks
        ]

        if release_risk.approval_required is True:
            answer += (
                " Human approval is required before any downstream "
                "release notification or Slack action."
            )

        logger.info(
            "agent_filtered_risk_response_composed",
            extra={
                "run_id": self._request_id,
                "release_run_id": str(release_risk.release_run.id),
                "intent": plan.intent.value,
                "filtered_risk_count": risk_count,
                "citation_count": len(citations),
                "approval_required": release_risk.approval_required is True,
            },
        )

        return AgentQueryResponse(
            answer=answer,
            plan=plan,
            release_risk=release_risk,
            citations=citations,
            approval_required=release_risk.approval_required is True,
        )

    def compose_github_pr(
        self,
        *,
        plan: AgentQueryPlan,
        release_risk: ReleaseRunRiskResponse,
        pull_request: PullRequestRiskResponse,
    ) -> AgentQueryResponse:
        """Compose a focused response for one persisted GitHub pull request.

        Args:
            plan: Validated GitHub PR query plan.
            release_risk: Trusted persisted release-risk snapshot.
            pull_request: Persisted risk result for the requested PR.

        Returns:
            Focused PR response with a single trusted citation.
        """

        severity = (
            pull_request.max_severity.value.replace("_", " ")
            if pull_request.max_severity is not None
            else "no detected"
        )
        score_percentage = round(pull_request.total_score * 100)

        answer = (
            f"PR {pull_request.pull_request_number} has {severity} severity "
            f"with an {score_percentage}% risk score."
        )

        if pull_request.signals:
            signal_lines: list[str] = []

            for index, signal in enumerate(
                pull_request.signals,
                start=1,
            ):
                signal_text = (
                    f"{index}. {signal.title}: {signal.description}"
                )

                if signal.evidence:
                    evidence_items = [
                        f"{key.replace('_', ' ')}: {value}"
                        for key, value in sorted(signal.evidence.items())
                    ]
                    signal_text += (
                        f" Evidence: {'; '.join(evidence_items)}."
                    )

                signal_lines.append(signal_text)

            answer += " Detected signals: " + " ".join(signal_lines)
        else:
            answer += " No persisted risk signals were detected for this PR."

        if release_risk.approval_required is True:
            answer += (
                " Human approval is required before any downstream "
                "release notification or Slack action."
            )

        citation = AgentCitation(
            source="github",
            source_type=pull_request.source_type,
            source_id=pull_request.source_id,
            title=f"GitHub PR {pull_request.pull_request_number}",
            source_url=pull_request.source_url,
        )

        logger.info(
            "agent_github_pr_response_composed",
            extra={
                "run_id": self._request_id,
                "release_run_id": str(release_risk.release_run.id),
                "intent": plan.intent.value,
                "pull_request_number": pull_request.pull_request_number,
                "source_id": pull_request.source_id,
                "signal_count": len(pull_request.signals),
                "citation_count": 1,
                "approval_required": release_risk.approval_required is True,
            },
        )

        return AgentQueryResponse(
            answer=answer,
            plan=plan,
            release_risk=release_risk,
            citations=[citation],
            approval_required=release_risk.approval_required is True,
        )

    def compose_jira_ticket(
        self,
        *,
        plan: AgentQueryPlan,
        release_risk: ReleaseRunRiskResponse,
        jira_issue: JiraIssueRiskResponse,
    ) -> AgentQueryResponse:
        """Compose a focused response for one persisted Jira issue.

        Args:
            plan: Validated Jira ticket query plan.
            release_risk: Trusted persisted release-risk snapshot.
            jira_issue: Persisted risk result for the requested Jira issue.

        Returns:
            Focused Jira response with a single trusted citation.
        """

        severity_rank = {
            "low": 1,
            "medium": 2,
            "high": 3,
            "critical": 4,
        }

        highest_signal = max(
            jira_issue.signals,
            key=lambda signal: (
                severity_rank[signal.severity.value],
                signal.score,
            ),
            default=None,
        )

        if highest_signal is None:
            answer = (
                f"{jira_issue.issue_key}: {jira_issue.title}. "
                "No persisted risk signals were detected for this Jira issue."
            )
        else:
            severity = highest_signal.severity.value.replace("_", " ")
            score_percentage = round(highest_signal.score * 100)

            answer = (
                f"{jira_issue.issue_key}: {jira_issue.title} has "
                f"{severity} severity with a {score_percentage}% risk score."
            )

            signal_lines: list[str] = []

            for index, signal in enumerate(jira_issue.signals, start=1):
                signal_text = (
                    f"{index}. {signal.title}: {signal.description}"
                )

                if signal.evidence:
                    evidence_items = [
                        f"{key.replace('_', ' ')}: {value}"
                        for key, value in sorted(signal.evidence.items())
                    ]
                    signal_text += (
                        f" Evidence: {'; '.join(evidence_items)}."
                    )

                signal_lines.append(signal_text)

            answer += " Detected signals: " + " ".join(signal_lines)

        if release_risk.approval_required is True:
            answer += (
                " Human approval is required before any downstream "
                "release notification or Slack action."
            )

        citation = AgentCitation(
            source="jira",
            source_type="jira_issue",
            source_id=jira_issue.issue_key,
            title=jira_issue.title,
            source_url=jira_issue.issue_url,
        )

        logger.info(
            "agent_jira_ticket_response_composed",
            extra={
                "run_id": self._request_id,
                "release_run_id": str(release_risk.release_run.id),
                "intent": plan.intent.value,
                "jira_issue_key": jira_issue.issue_key,
                "signal_count": len(jira_issue.signals),
                "citation_count": 1,
                "approval_required": release_risk.approval_required is True,
            },
        )

        return AgentQueryResponse(
            answer=answer,
            plan=plan,
            release_risk=release_risk,
            citations=[citation],
            approval_required=release_risk.approval_required is True,
        )
