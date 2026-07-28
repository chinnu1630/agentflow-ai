"""Streamlit manager experience for AgentFlow AI."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TypeGuard
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import streamlit as st
from pydantic import SecretStr, TypeAdapter, ValidationError

from agentflow_frontend.api_client import (
    AgentFlowAPIClient,
    AgentFlowAPIError,
    AgentQueryCallResult,
    ApprovalDecisionCallResult,
    PendingApprovalsCallResult,
    ReleaseRunEventsCallResult,
    ReleaseRunStatusCallResult,
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
_CHAT_MESSAGES_STATE_KEY = "agentflow_chat_messages"
_CONVERSATION_SESSION_ID_STATE_KEY = "agentflow_conversation_session_id"
_LAST_RELEASE_RUN_ID_STATE_KEY = "agentflow_last_chat_release_run_id"
_PENDING_APPROVALS_STATE_KEY = "agentflow_pending_approvals"

_APPROVAL_DECISION_RESULT_STATE_KEY = "agentflow_approval_decision_result"
_APPROVED_RELEASE_RUN_STATE_KEY = "agentflow_approved_release_run"
_SLACK_ALERT_RESULT_STATE_KEY = "agentflow_slack_alert_result"
_RELEASE_RUN_STATUS_STATE_KEY = "agentflow_release_run_status"
_RELEASE_RUN_EVENTS_STATE_KEY = "agentflow_release_run_events"

_RELEASE_RUN_ID_ADAPTER: TypeAdapter[UUID] = TypeAdapter(UUID)


@dataclass(frozen=True, slots=True)
class ApprovalDecisionIntent:
    """Manager decision selected in the Streamlit approval queue."""

    release_run_id: str
    approval_id: str
    approval_status: ReleaseApprovalDecisionStatus
    decision_note: str | None


@dataclass(frozen=True, slots=True)
class ChatTurn:
    """One completed exchange in the AgentFlow chat transcript."""

    query: str
    result: AgentQueryCallResult | None
    error: str | None


def validate_release_run_id(value: str) -> str:
    """Validate and normalize a manager-provided release-run UUID.

    Args:
        value: Raw Streamlit text input.

    Returns:
        Canonical UUID string accepted by the FastAPI route.

    Raises:
        ValidationError: If the value is not a valid UUID.
    """
    release_run_id = _RELEASE_RUN_ID_ADAPTER.validate_python(value.strip())
    return str(release_run_id)


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
    conversation_session_id: UUID | None = None,
    release_run_id: str | None = None,
) -> AgentQueryCallResult:
    """Execute one manager query through the typed FastAPI client.

    Args:
        settings: Validated frontend runtime configuration.
        bearer_token: Signed JWT supplied by the authorized manager.
        query: Natural-language release-risk question.
        conversation_session_id: Client-generated ID correlating every
            question asked within one chat session.
        release_run_id: Most recent release-run UUID from this chat, reused
            so follow-up questions resolve trusted persisted context.

    Returns:
        Validated backend response and request correlation ID.
    """
    request = AgentQueryRequest(
        query=query,
        conversation_session_id=conversation_session_id,
        release_run_id=(
            UUID(release_run_id) if release_run_id is not None else None
        ),
    )

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


async def load_release_run_status(
    *,
    settings: FrontendSettings,
    bearer_token: SecretStr,
    release_run_id: str,
) -> ReleaseRunStatusCallResult:
    """Load the backend-owned state of one release workflow.

    Args:
        settings: Validated frontend runtime configuration.
        bearer_token: Signed JWT containing ``release:read``.
        release_run_id: Backend release-run UUID.

    Returns:
        Validated workflow status and request correlation identifier.
    """
    async with AgentFlowAPIClient(
        settings=settings,
        bearer_token=bearer_token,
    ) as client:
        return await client.get_release_run_status(
            release_run_id=release_run_id,
        )


async def load_release_run_events(
    *,
    settings: FrontendSettings,
    bearer_token: SecretStr,
    release_run_id: str,
) -> ReleaseRunEventsCallResult:
    """Load the append-only audit timeline for one release workflow.

    Args:
        settings: Validated frontend runtime configuration.
        bearer_token: Signed JWT containing ``release:read``.
        release_run_id: Backend release-run UUID.

    Returns:
        Validated workflow events and request correlation identifier.
    """
    async with AgentFlowAPIClient(
        settings=settings,
        bearer_token=bearer_token,
    ) as client:
        return await client.list_release_run_events(
            release_run_id=release_run_id,
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


def render_release_run_status(
    result: ReleaseRunStatusCallResult,
) -> None:
    """Render the persisted state of one release workflow.

    Args:
        result: Validated workflow status and correlation identifier.
    """
    status = result.response

    with st.container(border=True):
        st.metric("Workflow status", status.status)
        st.write(status.query)
        st.caption(
            f"Requested by: {status.requested_by} · "
            f"Backend run ID: {status.run_id}"
        )
        st.caption(
            f"Created: {status.created_at.isoformat()} · "
            f"Completed: "
            f"{status.completed_at.isoformat() if status.completed_at else 'Pending'}"
        )
        st.caption(f"Request correlation ID: {result.run_id}")


def render_release_run_events(
    result: ReleaseRunEventsCallResult,
) -> None:
    """Render an append-only release workflow audit timeline.

    Args:
        result: Validated workflow events and correlation identifier.
    """
    events = result.response.events

    st.caption(f"Request correlation ID: {result.run_id}")

    if not events:
        st.info("No audit events are available for this release run.")
        return

    st.write(f"{len(events)} audit event(s) recorded.")

    for event in events:
        with st.container(border=True):
            st.write(event.message)
            st.caption(
                f"{event.event_type} · {event.event_status} · "
                f"{event.created_at.isoformat()}"
            )

            if event.metadata_json:
                st.json(event.metadata_json, expanded=False)


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

        if settings.auth_required:
            bearer_token = st.text_input(
                "Signed access token",
                type="password",
                help=(
                    "Use an authorized RS256 JWT. The token is sent only to "
                    "the configured AgentFlow backend."
                ),
            )
        else:
            bearer_token = ""
            st.warning(
                "Local development authentication is enabled. "
                "Do not use this mode in staging or production."
            )

    st.subheader("Chat with AgentFlow")
    st.caption(
        "Ask about release risk, GitHub pull requests, Jira tickets, "
        "engineering docs, or workflow status. Follow-up questions "
        "automatically reuse the most recent release run in this chat."
    )

    if _CONVERSATION_SESSION_ID_STATE_KEY not in st.session_state:
        st.session_state[_CONVERSATION_SESSION_ID_STATE_KEY] = uuid4()

    chat_messages: list[ChatTurn] = st.session_state.setdefault(
        _CHAT_MESSAGES_STATE_KEY,
        [],
    )

    for turn in chat_messages:
        with st.chat_message("user"):
            st.write(turn.query)

        with st.chat_message("assistant"):
            if turn.result is not None:
                render_agent_query_response(turn.result)
            elif turn.error is not None:
                st.error(turn.error)

    chat_query = st.chat_input(DEFAULT_RELEASE_RISK_QUERY)

    if chat_query:
        if settings.auth_required and not bearer_token.strip():
            st.error("Enter a signed access token before chatting with AgentFlow.")
        else:
            with st.chat_message("user"):
                st.write(chat_query)

            with st.chat_message("assistant"):
                new_turn: ChatTurn | None = None

                try:
                    with st.spinner(
                        "Collecting GitHub, Jira, and engineering evidence..."
                    ):
                        result = asyncio.run(
                            execute_manager_query(
                                settings=settings,
                                bearer_token=SecretStr(bearer_token),
                                query=chat_query,
                                conversation_session_id=st.session_state[
                                    _CONVERSATION_SESSION_ID_STATE_KEY
                                ],
                                release_run_id=st.session_state.get(
                                    _LAST_RELEASE_RUN_ID_STATE_KEY
                                ),
                            )
                        )
                except ValidationError:
                    error_message = "The question is invalid."
                    st.error(error_message)
                    new_turn = ChatTurn(
                        query=chat_query,
                        result=None,
                        error=error_message,
                    )
                except AgentFlowAPIError as exc:
                    st.error(str(exc))

                    if exc.run_id:
                        st.caption(f"Request correlation ID: {exc.run_id}")

                    new_turn = ChatTurn(
                        query=chat_query,
                        result=None,
                        error=str(exc),
                    )
                except ValueError as exc:
                    st.error(str(exc))
                    new_turn = ChatTurn(
                        query=chat_query,
                        result=None,
                        error=str(exc),
                    )
                else:
                    render_agent_query_response(result)
                    new_turn = ChatTurn(
                        query=chat_query,
                        result=result,
                        error=None,
                    )

                    if result.response.release_risk is not None:
                        st.session_state[_LAST_RELEASE_RUN_ID_STATE_KEY] = str(
                            result.response.release_risk.release_run.id
                        )

            chat_messages.append(new_turn)


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
        if settings.auth_required and not bearer_token.strip():
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
        if settings.auth_required and not bearer_token.strip():
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
            if settings.auth_required and not bearer_token.strip():
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

    st.divider()
    st.subheader("Workflow status and audit timeline")
    st.caption(
        "Read the persisted workflow state and append-only audit events "
        "from FastAPI."
    )

    release_run_lookup = st.text_input(
        "Release run ID",
        key="workflow-release-run-id",
        placeholder="Paste the release-run UUID shown in the assessment",
    )

    status_column, events_column = st.columns(2)
    load_status_clicked = status_column.button(
        "Load workflow status",
        use_container_width=True,
    )
    load_events_clicked = events_column.button(
        "Load audit timeline",
        use_container_width=True,
    )

    if load_status_clicked or load_events_clicked:
        if settings.auth_required and not bearer_token.strip():
            st.error(
                "Enter a signed access token before loading workflow data."
            )
        else:
            try:
                validated_release_run_id = validate_release_run_id(
                    release_run_lookup
                )
            except ValidationError:
                st.error("Enter a valid release-run UUID.")
            else:
                if load_status_clicked:
                    try:
                        with st.spinner("Loading workflow status..."):
                            status_result = asyncio.run(
                                load_release_run_status(
                                    settings=settings,
                                    bearer_token=SecretStr(bearer_token),
                                    release_run_id=validated_release_run_id,
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
                            _RELEASE_RUN_STATUS_STATE_KEY
                        ] = status_result

                if load_events_clicked:
                    try:
                        with st.spinner("Loading audit timeline..."):
                            events_result = asyncio.run(
                                load_release_run_events(
                                    settings=settings,
                                    bearer_token=SecretStr(bearer_token),
                                    release_run_id=validated_release_run_id,
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
                            _RELEASE_RUN_EVENTS_STATE_KEY
                        ] = events_result

    stored_status_result = st.session_state.get(
        _RELEASE_RUN_STATUS_STATE_KEY
    )

    if isinstance(stored_status_result, ReleaseRunStatusCallResult):
        render_release_run_status(stored_status_result)

    stored_events_result = st.session_state.get(
        _RELEASE_RUN_EVENTS_STATE_KEY
    )

    if isinstance(stored_events_result, ReleaseRunEventsCallResult):
        render_release_run_events(stored_events_result)



if __name__ == "__main__":
    main()
