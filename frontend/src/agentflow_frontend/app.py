"""Streamlit manager experience for AgentFlow AI."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TypeGuard
from urllib.parse import urlsplit

import streamlit as st
from pydantic import SecretStr, ValidationError

from agentflow_frontend.api_client import (
    AgentFlowAPIClient,
    AgentFlowAPIError,
    AgentQueryCallResult,
    ApprovalDecisionCallResult,
    PendingApprovalsCallResult,
    SlackAlertCallResult,
)
from agentflow_frontend.api_models import (
    AgentQueryRequest,
    AgentQueryResponse,
    ReleaseApprovalDecisionStatus,
    ReleaseRunApprovalDecisionRequest,
)
from agentflow_frontend.config import FrontendSettings, get_frontend_settings

DEFAULT_RELEASE_RISK_QUERY = "What are the biggest release risks this week?"
_QUERY_RESULT_STATE_KEY = "agentflow_query_result"
_PENDING_APPROVALS_STATE_KEY = "agentflow_pending_approvals"

_APPROVAL_DECISION_RESULT_STATE_KEY = "agentflow_approval_decision_result"
_APPROVED_RELEASE_RUN_STATE_KEY = "agentflow_approved_release_run"
_SLACK_ALERT_RESULT_STATE_KEY = "agentflow_slack_alert_result"


@dataclass(frozen=True, slots=True)
class ApprovalDecisionIntent:
    """Manager decision selected in the Streamlit approval queue."""

    release_run_id: str
    approval_id: str
    approval_status: ReleaseApprovalDecisionStatus
    decision_note: str | None


def is_safe_http_url(value: str | None) -> TypeGuard[str]:
    """Return whether a citation URL uses an allowed HTTP scheme.

    Args:
        value: Optional backend-provided citation URL.

    Returns:
        True only for complete HTTP or HTTPS URLs.
    """
    if value is None:
        return False

    parsed = urlsplit(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


async def execute_manager_query(
    *,
    settings: FrontendSettings,
    bearer_token: SecretStr,
    query: str,
) -> AgentQueryCallResult:
    """Execute one manager query through the typed FastAPI client.

    Args:
        settings: Validated frontend runtime configuration.
        bearer_token: Signed JWT supplied by the authorized manager.
        query: Natural-language release-risk question.

    Returns:
        Validated backend response and request correlation ID.
    """
    request = AgentQueryRequest(query=query)

    async with AgentFlowAPIClient(
        settings=settings,
        bearer_token=bearer_token,
    ) as client:
        return await client.execute_agent_query(request)


async def load_pending_approvals(
    *,
    settings: FrontendSettings,
    bearer_token: SecretStr,
) -> PendingApprovalsCallResult:
    """Load pending approval requests through the typed FastAPI client.

    Args:
        settings: Validated frontend runtime configuration.
        bearer_token: Signed JWT supplied by an authorized manager.

    Returns:
        Validated pending approval queue and request correlation ID.
    """
    async with AgentFlowAPIClient(
        settings=settings,
        bearer_token=bearer_token,
    ) as client:
        return await client.list_pending_approvals()


async def decide_pending_approval(
    *,
    settings: FrontendSettings,
    bearer_token: SecretStr,
    release_run_id: str,
    approval_id: str,
    approval_status: ReleaseApprovalDecisionStatus,
    decision_note: str | None,
) -> ApprovalDecisionCallResult:
    """Submit one authorized manager approval decision to FastAPI.

    Args:
        settings: Validated frontend runtime configuration.
        bearer_token: Signed JWT containing ``release:approve``.
        release_run_id: Release-run UUID owning the approval request.
        approval_id: Pending approval request UUID.
        approval_status: Approved or rejected terminal decision.
        decision_note: Optional manager-provided audit note.

    Returns:
        Persisted backend approval decision and correlation identifier.
    """
    decision = ReleaseRunApprovalDecisionRequest(
        approval_status=approval_status,
        decision_note=decision_note,
    )

    async with AgentFlowAPIClient(
        settings=settings,
        bearer_token=bearer_token,
    ) as client:
        return await client.decide_release_run_approval(
            release_run_id=release_run_id,
            approval_id=approval_id,
            decision=decision,
        )


async def send_approved_release_slack_alert(
    *,
    settings: FrontendSettings,
    bearer_token: SecretStr,
    release_run_id: str,
) -> SlackAlertCallResult:
    """Request an approval-gated Slack alert through FastAPI.

    Args:
        settings: Validated frontend runtime configuration.
        bearer_token: Signed JWT containing ``release:notify``.
        release_run_id: Backend release-run UUID to notify about.

    Returns:
        Validated Slack delivery result and request correlation identifier.
    """
    async with AgentFlowAPIClient(
        settings=settings,
        bearer_token=bearer_token,
    ) as client:
        return await client.send_release_run_slack_alert(
            release_run_id=release_run_id,
        )


def render_slack_alert_result(
    result: SlackAlertCallResult,
) -> None:
    """Render a validated backend Slack delivery result.

    Args:
        result: Slack delivery response and request correlation identifier.
    """
    response = result.response

    if response.sent:
        st.success(
            f"Approved release alert sent to "
            f"`{response.slack_channel}`."
        )
    else:
        st.warning("The backend did not send the Slack alert.")

    metric_columns = st.columns(2)
    metric_columns[0].metric("Risk level", response.risk_level)
    metric_columns[1].metric(
        "Risk score",
        f"{response.risk_score:.2f}",
    )

    st.write(response.recommended_action)
    st.caption(f"Slack timestamp: {response.slack_timestamp}")
    st.caption(f"Request correlation ID: {result.run_id}")


def render_pending_approvals(
    result: PendingApprovalsCallResult,
) -> ApprovalDecisionIntent | None:
    """Render pending approvals and capture one manager decision.

    Args:
        result: Validated pending approvals and correlation identifier.

    Returns:
        Selected approval decision, or ``None`` when no action was taken.
    """
    approvals = result.response.approvals

    st.caption(f"Request correlation ID: {result.run_id}")

    if not approvals:
        st.info("No release runs are currently waiting for approval.")
        return None

    st.write(f"{len(approvals)} release run(s) require manager review.")

    for approval in approvals:
        with st.container(border=True):
            st.write(f"Release run: `{approval.release_run_id}`")
            st.write(approval.approval_reason)
            st.caption(
                f"Status: {approval.approval_status} · "
                f"Policy: {approval.approval_policy_version}"
            )
            st.caption(
                f"Approval request ID: {approval.id} · "
                f"Created: {approval.created_at.isoformat()}"
            )

            decision_note = st.text_area(
                "Decision note",
                key=f"approval-note-{approval.id}",
                placeholder="Optional audit note for this decision",
                max_chars=2_000,
            )
            normalized_note = decision_note.strip() or None

            approve_column, reject_column = st.columns(2)

            if approve_column.button(
                "Approve",
                key=f"approve-{approval.id}",
                type="primary",
                use_container_width=True,
            ):
                return ApprovalDecisionIntent(
                    release_run_id=str(approval.release_run_id),
                    approval_id=str(approval.id),
                    approval_status=(
                        ReleaseApprovalDecisionStatus.APPROVED
                    ),
                    decision_note=normalized_note,
                )

            if reject_column.button(
                "Reject",
                key=f"reject-{approval.id}",
                use_container_width=True,
            ):
                return ApprovalDecisionIntent(
                    release_run_id=str(approval.release_run_id),
                    approval_id=str(approval.id),
                    approval_status=(
                        ReleaseApprovalDecisionStatus.REJECTED
                    ),
                    decision_note=normalized_note,
                )

    return None


def render_agent_query_response(result: AgentQueryCallResult) -> None:
    """Render a validated AgentFlow response in Streamlit.

    Args:
        result: Typed backend response and correlation identifier.
    """
    response = result.response

    st.subheader("Agent answer")
    st.write(response.answer)
    st.caption(f"Request correlation ID: {result.run_id}")

    release_risk = response.release_risk

    if release_risk is not None:
        st.subheader("Release assessment")

        score_value = (
            f"{release_risk.risk_score.score:.2f}"
            if release_risk.risk_score is not None
            else "Unavailable"
        )

        severity_value = (
            release_risk.risk_score.risk_level
            if release_risk.risk_score is not None
            else release_risk.release_summary.overall_severity
        )

        metric_columns = st.columns(4)
        metric_columns[0].metric("Risk level", severity_value)
        metric_columns[1].metric("Risk score", score_value)
        metric_columns[2].metric(
            "High risks",
            release_risk.release_summary.high_risk_count,
        )
        metric_columns[3].metric(
            "Total signals",
            release_risk.release_summary.total_signal_count,
        )

        st.caption("Release run ID")
        st.code(str(release_risk.release_run.id), language=None)

        degraded_dependencies: list[str] = []

        if release_risk.github.status == "degraded":
            degraded_dependencies.append("GitHub")

        if release_risk.jira.status == "degraded":
            degraded_dependencies.append("Jira")

        if degraded_dependencies:
            st.warning(
                "Degraded dependency data: "
                + ", ".join(degraded_dependencies)
                + ". Review the result before acting."
            )
        else:
            st.success("GitHub and Jira collection completed successfully.")

        if release_risk.synthesis_error:
            st.warning(
                "LLM synthesis was degraded. Deterministic risk evidence "
                "remains available for manager review."
            )

        st.subheader("Ranked risks")
        top_risks = release_risk.release_summary.top_risks

        if not top_risks:
            st.info("No ranked release risks were returned.")
        else:
            for position, risk in enumerate(top_risks, start=1):
                with st.container(border=True):
                    st.caption(
                        f"Rank {position} · {risk.source.upper()} · "
                        f"{risk.severity.upper()} · Score {risk.score:.2f}"
                    )
                    st.write(risk.title)
                    st.write(risk.reason)
                    st.caption(
                        f"Evidence source: {risk.source_type} / {risk.source_id}"
                    )

                    if risk.evidence:
                        with st.expander("Structured evidence"):
                            st.json(risk.evidence)

    _render_human_review_status(response)
    _render_citations(response)


def _render_human_review_status(response: AgentQueryResponse) -> None:
    """Render backend-provided human-review requirements."""
    review_required = response.approval_required or (
        response.release_risk is not None
        and response.release_risk.approval_required is True
    )

    st.subheader("Human review")

    if review_required:
        st.warning("Human approval is required before any Slack alert can be sent.")

        if response.release_risk is not None:
            if response.release_risk.approval_reason:
                st.write(response.release_risk.approval_reason)

            if response.release_risk.approval_status:
                st.caption(
                    "Approval status: "
                    f"{response.release_risk.approval_status}"
                )
    else:
        st.success("The backend did not require human approval for this response.")


def _render_citations(response: AgentQueryResponse) -> None:
    """Render trusted evidence citations returned by FastAPI."""
    st.subheader("Citations")

    if not response.citations:
        st.info("No citations were returned for this answer.")
        return

    for citation in response.citations:
        with st.container(border=True):
            st.write(citation.title)
            st.caption(
                f"{citation.source_type} · {citation.source_id} · "
                f"{citation.source}"
            )

            if is_safe_http_url(citation.source_url):
                st.link_button(
                    "Open evidence",
                    citation.source_url,
                )


def main() -> None:
    """Run the AgentFlow Streamlit manager application."""
    st.set_page_config(
        page_title="AgentFlow AI",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("AgentFlow AI")
    st.caption(
        "Enterprise release-risk analysis with evidence-grounded AI "
        "and human approval."
    )

    try:
        settings = get_frontend_settings()
    except ValidationError:
        st.error(
            "Frontend configuration is incomplete. Set "
            "AGENTFLOW_FRONTEND_BACKEND_BASE_URL before starting Streamlit."
        )
        st.stop()

    with st.sidebar:
        st.header("Secure connection")
        st.caption(f"Backend: {settings.backend_base_url}")
        bearer_token = st.text_input(
            "Signed access token",
            type="password",
            help=(
                "Use an authorized RS256 JWT. The token is sent only to "
                "the configured AgentFlow backend."
            ),
        )

    with st.form("release_risk_query_form"):
        query = st.text_area(
            "Manager question",
            value=DEFAULT_RELEASE_RISK_QUERY,
            height=120,
            max_chars=2_000,
        )
        submitted = st.form_submit_button(
            "Analyze release risks",
            type="primary",
        )

    if submitted:
        if not bearer_token.strip():
            st.error("Enter a signed access token before submitting the query.")
        else:
            try:
                with st.spinner(
                    "Collecting GitHub, Jira, and engineering evidence..."
                ):
                    result = asyncio.run(
                        execute_manager_query(
                            settings=settings,
                            bearer_token=SecretStr(bearer_token),
                            query=query,
                        )
                    )
            except ValidationError:
                st.error("The manager question is invalid.")
            except AgentFlowAPIError as exc:
                st.error(str(exc))

                if exc.run_id:
                    st.caption(f"Request correlation ID: {exc.run_id}")
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state[_QUERY_RESULT_STATE_KEY] = result

    stored_result = st.session_state.get(_QUERY_RESULT_STATE_KEY)

    if isinstance(stored_result, AgentQueryCallResult):
        render_agent_query_response(stored_result)


    st.divider()
    st.subheader("Manager approval queue")
    st.caption(
        "Load durable pending approval requests from the AgentFlow backend."
    )

    stored_decision_result = st.session_state.pop(
        _APPROVAL_DECISION_RESULT_STATE_KEY,
        None,
    )

    if isinstance(stored_decision_result, ApprovalDecisionCallResult):
        decided_approval = stored_decision_result.response
        st.success(
            f"Release run `{decided_approval.release_run_id}` was "
            f"{decided_approval.approval_status}."
        )
        st.caption(
            "Request correlation ID: "
            f"{stored_decision_result.run_id}"
        )

        if (
            decided_approval.approval_status
            == ReleaseApprovalDecisionStatus.APPROVED
        ):
            st.session_state[_APPROVED_RELEASE_RUN_STATE_KEY] = str(
                decided_approval.release_run_id
            )
        else:
            st.session_state.pop(
                _APPROVED_RELEASE_RUN_STATE_KEY,
                None,
            )

    stored_slack_result = st.session_state.pop(
        _SLACK_ALERT_RESULT_STATE_KEY,
        None,
    )

    if isinstance(stored_slack_result, SlackAlertCallResult):
        render_slack_alert_result(stored_slack_result)

    load_approvals_clicked = st.button("Load pending approvals")

    if load_approvals_clicked:
        if not bearer_token.strip():
            st.error(
                "Enter a signed access token before loading approvals."
            )
        else:
            try:
                with st.spinner("Loading pending approval requests..."):
                    approvals_result = asyncio.run(
                        load_pending_approvals(
                            settings=settings,
                            bearer_token=SecretStr(bearer_token),
                        )
                    )
            except AgentFlowAPIError as exc:
                st.error(str(exc))

                if exc.run_id:
                    st.caption(f"Request correlation ID: {exc.run_id}")
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state[_PENDING_APPROVALS_STATE_KEY] = (
                    approvals_result
                )

    stored_approvals = st.session_state.get(
        _PENDING_APPROVALS_STATE_KEY
    )

    decision_intent: ApprovalDecisionIntent | None = None

    if isinstance(stored_approvals, PendingApprovalsCallResult):
        decision_intent = render_pending_approvals(stored_approvals)

    if decision_intent is not None:
        if not bearer_token.strip():
            st.error(
                "Enter a signed access token before deciding an approval."
            )
        else:
            try:
                with st.spinner("Persisting manager decision..."):
                    decision_result = asyncio.run(
                        decide_pending_approval(
                            settings=settings,
                            bearer_token=SecretStr(bearer_token),
                            release_run_id=(
                                decision_intent.release_run_id
                            ),
                            approval_id=decision_intent.approval_id,
                            approval_status=(
                                decision_intent.approval_status
                            ),
                            decision_note=decision_intent.decision_note,
                        )
                    )
            except ValidationError:
                st.error("The approval decision is invalid.")
            except AgentFlowAPIError as exc:
                st.error(str(exc))

                if exc.run_id:
                    st.caption(
                        f"Request correlation ID: {exc.run_id}"
                    )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state[
                    _APPROVAL_DECISION_RESULT_STATE_KEY
                ] = decision_result
                st.session_state.pop(
                    _PENDING_APPROVALS_STATE_KEY,
                    None,
                )
                st.rerun()

    approved_release_run_id = st.session_state.get(
        _APPROVED_RELEASE_RUN_STATE_KEY
    )

    if isinstance(approved_release_run_id, str):
        st.divider()
        st.subheader("Approved release notification")
        st.caption(
            "FastAPI will revalidate approval, authorization, trusted "
            "risk evidence, and duplicate-send protection."
        )

        send_slack_clicked = st.button(
            "Send approved Slack alert",
            key=f"send-slack-{approved_release_run_id}",
            type="primary",
        )

        if send_slack_clicked:
            if not bearer_token.strip():
                st.error(
                    "Enter a signed access token before sending "
                    "the Slack alert."
                )
            else:
                try:
                    with st.spinner(
                        "Requesting approval-gated Slack delivery..."
                    ):
                        slack_result = asyncio.run(
                            send_approved_release_slack_alert(
                                settings=settings,
                                bearer_token=SecretStr(bearer_token),
                                release_run_id=approved_release_run_id,
                            )
                        )
                except AgentFlowAPIError as exc:
                    st.error(str(exc))

                    if exc.run_id:
                        st.caption(
                            f"Request correlation ID: {exc.run_id}"
                        )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.session_state[
                        _SLACK_ALERT_RESULT_STATE_KEY
                    ] = slack_result
                    st.session_state.pop(
                        _APPROVED_RELEASE_RUN_STATE_KEY,
                        None,
                    )
                    st.rerun()


if __name__ == "__main__":
    main()
