"""Compose manager-friendly conversational AgentFlow responses."""

from __future__ import annotations

import logging
from typing import Protocol

from app.schemas.agent_query import (
    AgentCitation,
    AgentQueryPlan,
    AgentQueryResponse,
    ResponseDepth,
)
from app.schemas.risk import (
    JiraIssueRiskResponse,
    PullRequestRiskResponse,
    ReleaseRiskSummaryItemResponse,
    ReleaseRunRiskResponse,
)

logger = logging.getLogger(__name__)


class ApprovalStatusRecordProtocol(Protocol):
    """Approval fields required to compose an approval-status response."""

    id: object
    approval_status: str
    approval_reason: str
    approval_policy_version: str
    requested_by: str | None
    decided_by: str | None
    decision_note: str | None
    created_at: object
    decided_at: object


class SlackAlertStatusRecordProtocol(Protocol):
    """Slack delivery fields required to compose a status response."""

    id: object
    delivery_status: str
    slack_channel: str
    slack_timestamp: str
    risk_level: str
    risk_score: float
    recommended_action: str
    created_at: object


class AgentResponseComposer:
    """Create deterministic conversational answers from trusted risk data."""

    def __init__(self, request_id: str) -> None:
        """Initialize the response composer.

        Args:
            request_id: Request identifier used for structured logging.
        """

        self._request_id = request_id

    def compose(
        self,
        *,
        plan: AgentQueryPlan,
        release_risk: ReleaseRunRiskResponse,
    ) -> AgentQueryResponse:
        """Compose a conversational response with evidence citations.

        Args:
            plan: Validated natural-language query plan.
            release_risk: Trusted release-risk workflow response.

        Returns:
            Manager-friendly response containing the answer and citations.
        """

        citations = self._build_citations(release_risk)
        answer = self._build_answer(
            plan=plan,
            release_risk=release_risk,
        )

        logger.info(
            "agent_response_composed",
            extra={
                "run_id": self._request_id,
                "release_run_id": str(release_risk.release_run.id),
                "intent": plan.intent.value,
                "response_depth": plan.response_depth.value,
                "citation_count": len(citations),
                "approval_required": (release_risk.approval_required is True),
            },
        )

        return AgentQueryResponse(
            answer=answer,
            plan=plan,
            release_risk=release_risk,
            citations=citations,
            approval_required=release_risk.approval_required is True,
        )

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

    def compose_approval_status(
        self,
        *,
        plan: AgentQueryPlan,
        release_risk: ReleaseRunRiskResponse,
        latest_approval: ApprovalStatusRecordProtocol | None,
    ) -> AgentQueryResponse:
        """Compose the latest durable HITL approval status.

        Args:
            plan: Validated approval-status query plan.
            release_risk: Trusted persisted release-risk snapshot.
            latest_approval: Latest durable approval record, when one exists.

        Returns:
            Approval-status response without evidence citations.
        """

        if latest_approval is not None:
            approval_status = latest_approval.approval_status
            approval_reason = latest_approval.approval_reason
            decided_by = latest_approval.decided_by
            decision_note = latest_approval.decision_note
            approval_policy_version = latest_approval.approval_policy_version
        else:
            approval_status = (
                release_risk.approval_status
                or (
                    "pending"
                    if release_risk.approval_required is True
                    else "not_required"
                )
            )
            approval_reason = release_risk.approval_reason
            decided_by = None
            decision_note = None
            approval_policy_version = release_risk.approval_policy_version

        readable_status = approval_status.replace("_", " ")
        answer = f"Approval status: {readable_status}."

        if decided_by:
            answer += f" Decided by: {decided_by}."

        if decision_note:
            answer += f" Decision note: {decision_note}"

            if not answer.endswith((".", "!", "?")):
                answer += "."

        if approval_reason:
            answer += f" Approval reason: {approval_reason}"

            if not answer.endswith((".", "!", "?")):
                answer += "."

        logger.info(
            "agent_approval_status_response_composed",
            extra={
                "run_id": self._request_id,
                "release_run_id": str(release_risk.release_run.id),
                "intent": plan.intent.value,
                "approval_status": approval_status,
                "approval_policy_version": approval_policy_version,
                "approval_record_found": latest_approval is not None,
                "decision_recorded": decided_by is not None,
                "citation_count": 0,
            },
        )

        return AgentQueryResponse(
            answer=answer,
            plan=plan,
            release_risk=release_risk,
            citations=[],
            approval_required=release_risk.approval_required is True,
        )

    def compose_historical_risks(
        self,
        *,
        plan: AgentQueryPlan,
        release_risk: ReleaseRunRiskResponse,
        historical_release_risks: list[ReleaseRunRiskResponse],
    ) -> AgentQueryResponse:
        """Compose persisted risk history from previous release runs.

        Args:
            plan: Validated historical-risk query plan.
            release_risk: Trusted current release-risk snapshot.
            historical_release_risks: Trusted previous release snapshots.

        Returns:
            Historical summary with citations from previous persisted risks.
        """
        historical_count = len(historical_release_risks)
        release_label = "release" if historical_count == 1 else "releases"

        if not historical_release_risks:
            answer = "No previous releases with persisted risk history were found."
            citations: list[AgentCitation] = []
        else:
            history_lines: list[str] = []
            citations = []

            for index, historical_risk in enumerate(
                historical_release_risks,
                start=1,
            ):
                severity = (
                    historical_risk.release_summary.overall_severity.value
                    .replace("_", " ")
                )
                risk_score_text = (
                    f"{round(historical_risk.risk_score.score * 100)}% risk score"
                    if historical_risk.risk_score is not None
                    else "no persisted risk score"
                )
                top_risk_titles = [
                    risk.title
                    for risk in historical_risk.release_summary.top_risks[:3]
                ]
                top_risks_text = (
                    ", ".join(top_risk_titles)
                    if top_risk_titles
                    else "no ranked top risks"
                )

                history_lines.append(
                    f"{index}. {historical_risk.release_run.run_id}: "
                    f"{severity} severity, {risk_score_text}. "
                    f"Top risks: {top_risks_text}."
                )
                for citation in self._build_citations(historical_risk):
                    citations.append(
                        citation.model_copy(
                            update={
                                "title": (
                                    f"[{historical_risk.release_run.run_id}] "
                                    f"{citation.title}"
                                ),
                            }
                        )
                    )

            answer = (
                f"Found {historical_count} previous {release_label} "
                "with persisted risk history. "
                + " ".join(history_lines)
            )

        logger.info(
            "agent_historical_risk_response_composed",
            extra={
                "run_id": self._request_id,
                "release_run_id": str(release_risk.release_run.id),
                "intent": plan.intent.value,
                "historical_release_count": historical_count,
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

    def compose_previous_release_comparison(
        self,
        *,
        plan: AgentQueryPlan,
        release_risk: ReleaseRunRiskResponse,
        previous_release_risk: ReleaseRunRiskResponse | None,
    ) -> AgentQueryResponse:
        """Compare the current persisted risk snapshot with the previous release.

        Args:
            plan: Validated previous-release comparison plan.
            release_risk: Trusted current release-risk snapshot.
            previous_release_risk: Immediately previous persisted release snapshot.

        Returns:
            Deterministic comparison with release-qualified citations.
        """
        if previous_release_risk is None:
            answer = (
                "No previous release with persisted risk history was found "
                "for comparison."
            )
            citations: list[AgentCitation] = []
        else:
            current_severity = (
                release_risk.release_summary.overall_severity.value
            )
            previous_severity = (
                previous_release_risk.release_summary.overall_severity.value
            )
            severity_rank = {
                "low": 1,
                "medium": 2,
                "high": 3,
                "critical": 4,
            }

            if severity_rank[current_severity] > severity_rank[previous_severity]:
                severity_change = (
                    f"severity increased from {previous_severity} "
                    f"to {current_severity}"
                )
            elif severity_rank[current_severity] < severity_rank[previous_severity]:
                severity_change = (
                    f"severity decreased from {previous_severity} "
                    f"to {current_severity}"
                )
            else:
                severity_change = f"severity remained {current_severity}"

            current_score = (
                release_risk.risk_score.score
                if release_risk.risk_score is not None
                else None
            )
            previous_score = (
                previous_release_risk.risk_score.score
                if previous_release_risk.risk_score is not None
                else None
            )

            if current_score is not None and previous_score is not None:
                score_delta = round((current_score - previous_score) * 100)

                if score_delta > 0:
                    score_change = (
                        f"risk score increased by {score_delta} "
                        "percentage points"
                    )
                elif score_delta < 0:
                    score_change = (
                        f"risk score decreased by {abs(score_delta)} "
                        "percentage points"
                    )
                else:
                    score_change = "risk score did not change"
            else:
                score_change = "risk score comparison is unavailable"

            current_signal_count = (
                release_risk.release_summary.total_signal_count
            )
            previous_signal_count = (
                previous_release_risk.release_summary.total_signal_count
            )

            if current_signal_count > previous_signal_count:
                signal_change = (
                    f"signal count increased from {previous_signal_count} "
                    f"to {current_signal_count}"
                )
            elif current_signal_count < previous_signal_count:
                signal_change = (
                    f"signal count decreased from {previous_signal_count} "
                    f"to {current_signal_count}"
                )
            else:
                signal_change = (
                    f"signal count remained {current_signal_count}"
                )

            previous_titles = {
                risk.title
                for risk in previous_release_risk.release_summary.top_risks
            }
            new_top_risk_titles = [
                risk.title
                for risk in release_risk.release_summary.top_risks
                if risk.title not in previous_titles
            ]
            new_risks_text = (
                ", ".join(new_top_risk_titles[:3])
                if new_top_risk_titles
                else "no newly ranked top risks"
            )

            answer = (
                f"Compared with "
                f"{previous_release_risk.release_run.run_id}, "
                f"{severity_change}; {score_change}; and {signal_change}. "
                f"New top risks: {new_risks_text}."
            )

            citations = []

            for compared_risk in (
                previous_release_risk,
                release_risk,
            ):
                for citation in self._build_citations(compared_risk):
                    citations.append(
                        citation.model_copy(
                            update={
                                "title": (
                                    f"[{compared_risk.release_run.run_id}] "
                                    f"{citation.title}"
                                ),
                            }
                        )
                    )

        logger.info(
            "agent_previous_release_comparison_composed",
            extra={
                "run_id": self._request_id,
                "release_run_id": str(release_risk.release_run.id),
                "previous_release_run_id": (
                    str(previous_release_risk.release_run.id)
                    if previous_release_risk is not None
                    else None
                ),
                "intent": plan.intent.value,
                "previous_release_found": previous_release_risk is not None,
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

    def compose_slack_status(
        self,
        *,
        plan: AgentQueryPlan,
        release_risk: ReleaseRunRiskResponse,
        slack_alert: SlackAlertStatusRecordProtocol | None,
    ) -> AgentQueryResponse:
        """Compose Slack delivery status from durable persisted state.

        Args:
            plan: Validated Slack-status query plan.
            release_risk: Trusted persisted release-risk snapshot.
            slack_alert: Persisted successful Slack delivery record, if present.

        Returns:
            Slack delivery response without evidence citations.
        """
        if slack_alert is None:
            answer = "No Slack alert has been sent for this release run."
            delivery_status = "not_sent"
        else:
            delivery_status = slack_alert.delivery_status
            readable_action = slack_alert.recommended_action.replace("_", " ")
            risk_percentage = round(slack_alert.risk_score * 100)

            answer = (
                f"Slack alert status: {delivery_status.replace('_', ' ')}. "
                f"Channel: {slack_alert.slack_channel}. "
                f"Risk level: {slack_alert.risk_level.replace('_', ' ')}. "
                f"Risk score: {risk_percentage}%. "
                f"Recommended action: {readable_action}."
            )

        logger.info(
            "agent_slack_status_response_composed",
            extra={
                "run_id": self._request_id,
                "release_run_id": str(release_risk.release_run.id),
                "intent": plan.intent.value,
                "delivery_status": delivery_status,
                "slack_alert_found": slack_alert is not None,
                "slack_channel": (
                    slack_alert.slack_channel
                    if slack_alert is not None
                    else None
                ),
                "citation_count": 0,
                "approval_required": release_risk.approval_required is True,
            },
        )

        return AgentQueryResponse(
            answer=answer,
            plan=plan,
            release_risk=release_risk,
            citations=[],
            approval_required=release_risk.approval_required is True,
        )

    def compose_workflow_status(
        self,
        *,
        plan: AgentQueryPlan,
        release_risk: ReleaseRunRiskResponse,
    ) -> AgentQueryResponse:
        """Compose workflow status from a trusted persisted snapshot.

        Args:
            plan: Validated workflow-status query plan.
            release_risk: Trusted persisted release-risk snapshot.

        Returns:
            Brief workflow-status response without evidence citations.
        """

        workflow_status_value = getattr(
            release_risk.release_run.status,
            "value",
            release_risk.release_run.status,
        )
        github_status_value = getattr(
            release_risk.github.status,
            "value",
            release_risk.github.status,
        )
        jira_status_value = getattr(
            release_risk.jira.status,
            "value",
            release_risk.jira.status,
        )

        workflow_status = str(workflow_status_value).replace("_", " ")
        github_status = str(github_status_value).replace("_", " ")
        jira_status = str(jira_status_value).replace("_", " ")
        knowledge_status = (
            release_risk.knowledge_status.replace("_", " ")
            if release_risk.knowledge_status
            else "not available"
        )
        approval_status = (
            release_risk.approval_status.replace("_", " ")
            if release_risk.approval_status
            else "not required"
        )

        answer = (
            f"Workflow status: {workflow_status}. "
            f"GitHub collection: {github_status}. "
            f"Jira collection: {jira_status}. "
            f"Knowledge retrieval: {knowledge_status}. "
            f"Approval status: {approval_status}."
        )

        logger.info(
            "agent_workflow_status_response_composed",
            extra={
                "run_id": self._request_id,
                "release_run_id": str(release_risk.release_run.id),
                "intent": plan.intent.value,
                "workflow_status": str(workflow_status_value),
                "github_status": str(github_status_value),
                "jira_status": str(jira_status_value),
                "knowledge_status": release_risk.knowledge_status,
                "approval_status": release_risk.approval_status,
                "citation_count": 0,
            },
        )

        return AgentQueryResponse(
            answer=answer,
            plan=plan,
            release_risk=release_risk,
            citations=[],
            approval_required=release_risk.approval_required is True,
        )

    def _build_answer(
        self,
        *,
        plan: AgentQueryPlan,
        release_risk: ReleaseRunRiskResponse,
    ) -> str:
        """Build the answer at the requested response depth."""

        summary = release_risk.release_summary
        severity = summary.overall_severity.value.replace("_", " ")
        action = summary.recommended_action.value.replace("_", " ")

        opening = f"The release risk is {severity}. Recommended action: {action}."

        if plan.response_depth is ResponseDepth.BRIEF:
            return opening

        metrics = (
            f" The analysis found {summary.total_signal_count} risk signals, "
            f"including {summary.high_risk_count} high-severity signals."
        )

        score_text = ""

        if release_risk.risk_score is not None:
            score_percentage = round(release_risk.risk_score.score * 100)
            score_text = f" The deterministic risk score is {score_percentage}%."

        top_risks = summary.top_risks
        risk_limit = 5 if plan.response_depth is ResponseDepth.DEEP else 3

        risk_lines = [
            (f"{index}. {risk.title}: {risk.reason} ({risk.severity.value} severity).")
            for index, risk in enumerate(
                top_risks[:risk_limit],
                start=1,
            )
        ]

        risks_text = ""

        if risk_lines:
            risks_text = " Top risks: " + " ".join(risk_lines)
        else:
            risks_text = " No individual GitHub or Jira risk item was ranked in the final summary."

        approval_text = ""

        if release_risk.approval_required is True:
            approval_text = (
                " Human approval is required before any downstream "
                "release notification or Slack action."
            )

        if plan.response_depth is ResponseDepth.DEEP:
            source_text = (
                f" GitHub status: {release_risk.github.status.value}. "
                f"Jira status: {release_risk.jira.status.value}. "
                f"Knowledge retrieval status: "
                f"{release_risk.knowledge_status or 'not available'}."
            )
        else:
            source_text = ""

        return opening + metrics + score_text + risks_text + source_text + approval_text

    @staticmethod
    def _build_citations(
        release_risk: ReleaseRunRiskResponse,
    ) -> list[AgentCitation]:
        """Build deduplicated citations from trusted workflow evidence."""

        citations: list[AgentCitation] = []
        seen: set[tuple[str, str]] = set()

        for risk in release_risk.release_summary.top_risks:
            key = (risk.source_type, risk.source_id)

            if key in seen:
                continue

            seen.add(key)
            citations.append(
                AgentCitation(
                    source=risk.source,
                    source_type=risk.source_type,
                    source_id=risk.source_id,
                    title=risk.title,
                    source_url=risk.source_url,
                )
            )

        for knowledge_result in release_risk.knowledge_results:
            source_id = str(
                knowledge_result.chunk_id
                or knowledge_result.document_id
                or knowledge_result.title
                or "knowledge-result"
            )
            source_type = knowledge_result.source_type or "engineering_document"
            key = (source_type, source_id)

            if key in seen:
                continue

            seen.add(key)
            citations.append(
                AgentCitation(
                    source="knowledge",
                    source_type=source_type,
                    source_id=source_id,
                    title=(knowledge_result.title or "Engineering knowledge result"),
                )
            )

        return citations
