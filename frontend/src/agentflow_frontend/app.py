"""Streamlit manager experience for AgentFlow AI."""

from __future__ import annotations

import asyncio
from typing import TypeGuard
from urllib.parse import urlsplit

import streamlit as st
from pydantic import SecretStr, ValidationError

from agentflow_frontend.api_client import (
    AgentFlowAPIClient,
    AgentFlowAPIError,
    AgentQueryCallResult,
)
from agentflow_frontend.api_models import AgentQueryRequest, AgentQueryResponse
from agentflow_frontend.config import FrontendSettings, get_frontend_settings

DEFAULT_RELEASE_RISK_QUERY = "What are the biggest release risks this week?"
_QUERY_RESULT_STATE_KEY = "agentflow_query_result"


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


if __name__ == "__main__":
    main()
