"""Tests for the AgentFlow Streamlit manager experience."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import streamlit as st
from pydantic import SecretStr, ValidationError
from streamlit.testing.v1 import AppTest

import agentflow_frontend.app as app_module
from agentflow_frontend.api_client import (
    AgentQueryCallResult,
    ApprovalDecisionCallResult,
    PendingApprovalsCallResult,
    ReleaseRunEventsCallResult,
    ReleaseRunStatusCallResult,
    SlackAlertCallResult,
)
from agentflow_frontend.api_models import (
    AgentEntityReferences,
    AgentQueryRequest,
    AgentQueryResponse,
    PendingReleaseRunApprovalList,
    ReleaseApprovalDecisionStatus,
    ReleaseRunApproval,
    ReleaseRunApprovalDecisionRequest,
    ReleaseRunEventList,
    ReleaseRunStatus,
    SlackReleaseAlertResult,
)
from agentflow_frontend.config import FrontendSettings, get_frontend_settings

_STREAMLIT_APP_PATH = (
    Path(__file__).resolve().parents[1] / "streamlit_app.py"
)


@pytest.fixture(autouse=True)
def isolate_frontend_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Prevent developer-local .env values from leaking into frontend tests."""
    monkeypatch.setenv("AGENTFLOW_FRONTEND_AUTH_REQUIRED", "true")
    get_frontend_settings.cache_clear()

    yield

    get_frontend_settings.cache_clear()


def _text_input_by_label(app: AppTest, label: str) -> Any:
    """Return one Streamlit text input by its stable user-facing label."""
    return next(item for item in app.text_input if item.label == label)


def _text_area_by_label(app: AppTest, label: str) -> Any:
    """Return one Streamlit text area by its stable user-facing label."""
    return next(item for item in app.text_area if item.label == label)


def _button_by_label(app: AppTest, label: str) -> Any:
    """Return one Streamlit button by its stable user-facing label."""
    return next(item for item in app.button if item.label == label)


def test_apply_agentflow_theme_injects_static_brand_css(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply the navy-and-gold stylesheet without dynamic HTML content."""
    captured: dict[str, object] = {}

    def fake_markdown(
        body: str,
        *,
        unsafe_allow_html: bool = False,
    ) -> None:
        captured["body"] = body
        captured["unsafe_allow_html"] = unsafe_allow_html

    monkeypatch.setattr(st, "markdown", fake_markdown)

    app_module.apply_agentflow_theme()

    stylesheet = str(captured["body"])
    assert captured["unsafe_allow_html"] is True
    assert "#071426" in stylesheet
    assert "#F2C14E" in stylesheet
    assert "stChatMessage" in stylesheet
    assert "focus-visible" in stylesheet
    assert "block-container" in stylesheet
    assert "stWidgetLabel" in stylesheet
    assert "-webkit-text-fill-color" in stylesheet
    assert "stBottomBlockContainer" in stylesheet
    assert 'data-baseweb="input"' in stylesheet
    assert ".stButton > button p" in stylesheet



@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://jira.example.test/browse/PAY-102", True),
        ("http://github.example.test/pull/42", True),
        ("javascript:alert(1)", False),
        ("file:///etc/passwd", False),
        ("//malicious.example.test/path", False),
        (None, False),
    ],
)
def test_is_safe_http_url(url: str | None, expected: bool) -> None:
    """Allow only complete HTTP and HTTPS citation links."""
    assert app_module.is_safe_http_url(url) is expected


@pytest.mark.asyncio
async def test_execute_manager_query_uses_typed_api_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delegate the manager question to the existing FastAPI client."""
    captured: dict[str, Any] = {}

    response = AgentQueryResponse.model_validate(
        {
            "answer": "One release blocker requires review.",
            "plan": {
                "intent": "release_risk_summary",
                "response_depth": "detailed",
                "confidence": 0.98,
                "requires_human_approval": True,
                "routing_reason_code": "fresh_release_risk_request",
            },
            "citations": [],
            "approval_required": True,
        }
    )

    class FakeAgentFlowAPIClient:
        """Async context-manager fake for the typed backend client."""

        def __init__(
            self,
            *,
            settings: FrontendSettings,
            bearer_token: SecretStr,
        ) -> None:
            captured["settings"] = settings
            captured["authorization_value"] = bearer_token.get_secret_value()

        async def __aenter__(self) -> FakeAgentFlowAPIClient:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: object | None,
        ) -> None:
            return None

        async def execute_agent_query(
            self,
            request: object,
        ) -> AgentQueryCallResult:
            captured["request"] = request
            return AgentQueryCallResult(
                response=response,
                run_id="frontend-test-run-id",
            )

    monkeypatch.setattr(
        app_module,
        "AgentFlowAPIClient",
        FakeAgentFlowAPIClient,
    )

    settings = FrontendSettings(
        backend_base_url="https://agentflow.example.test",
        _env_file=None,
    )

    result = await app_module.execute_manager_query(
        settings=settings,
        bearer_token=SecretStr("signed-test-jwt"),
        query="What are the biggest release risks this week?",
    )

    assert result.response.answer == "One release blocker requires review."
    assert captured["authorization_value"] == "signed-test-jwt"
    assert captured["settings"] == settings

    captured_request = captured["request"]
    assert isinstance(captured_request, AgentQueryRequest)
    assert captured_request.query == (
        "What are the biggest release risks this week?"
    )


def test_streamlit_app_renders_chat_interface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Render the initial manager screen without making a backend request."""
    monkeypatch.setenv(
        "AGENTFLOW_FRONTEND_BACKEND_BASE_URL",
        "https://agentflow.example.test",
    )
    get_frontend_settings.cache_clear()

    app = AppTest.from_file(_STREAMLIT_APP_PATH)
    app.run()

    assert not app.exception
    assert app.title[0].value == "AgentFlow AI"
    assert any(
        item.label == "Signed access token"
        for item in app.text_input
    )
    assert any(
        item.value == "Chat with AgentFlow"
        for item in app.subheader
    )

    sidebar_captions = [
        item.value for item in app.sidebar.caption
    ]
    assert "Authenticated API connection configured." in sidebar_captions
    assert not any(
        "agentflow.example.test" in caption
        for caption in sidebar_captions
    )

    assert len(app.chat_input) == 1
    assert app.chat_input[0].placeholder == (
        app_module.DEFAULT_RELEASE_RISK_QUERY
    )


def test_streamlit_app_places_release_operations_in_sidebar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep operational release controls outside the chat workspace."""
    monkeypatch.setenv(
        "AGENTFLOW_FRONTEND_BACKEND_BASE_URL",
        "http://127.0.0.1:8000",
    )
    monkeypatch.setenv(
        "AGENTFLOW_FRONTEND_AUTH_REQUIRED",
        "false",
    )
    get_frontend_settings.cache_clear()

    app = AppTest.from_file(_STREAMLIT_APP_PATH)
    app.run()

    assert not app.exception

    sidebar_subheaders = [
        item.value for item in app.sidebar.subheader
    ]
    main_subheaders = [
        item.value for item in app.main.subheader
    ]

    assert "Manager approval queue" in sidebar_subheaders
    assert (
        "Workflow status and audit timeline"
        in sidebar_subheaders
    )
    assert "Manager approval queue" not in main_subheaders
    assert (
        "Workflow status and audit timeline"
        not in main_subheaders
    )


def test_streamlit_chat_contains_only_question_and_agent_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep release dashboards and operational details outside chat."""
    monkeypatch.setenv(
        "AGENTFLOW_FRONTEND_BACKEND_BASE_URL",
        "http://127.0.0.1:8000",
    )
    monkeypatch.setenv(
        "AGENTFLOW_FRONTEND_AUTH_REQUIRED",
        "false",
    )
    get_frontend_settings.cache_clear()

    release_run_id = "14326708-c085-4e6d-9c32-47dc92b24841"

    response = AgentQueryResponse.model_validate(
        {
            "answer": "The release has one high-risk blocker.",
            "plan": {
                "intent": "release_risk_summary",
                "response_depth": "detailed",
                "confidence": 0.98,
                "release_run_id": release_run_id,
                "requires_current_snapshot": True,
                "requires_human_approval": True,
                "routing_reason_code": "fresh_release_risk_request",
            },
            "release_risk": {
                "release_run": {
                    "id": release_run_id,
                    "run_id": "chat-layout-backend-run-id",
                    "query": "What are the biggest release risks this week?",
                    "requested_by": "manager@example.com",
                    "status": "waiting_for_approval",
                    "created_at": "2026-08-01T12:00:00Z",
                },
                "github": {"status": "success"},
                "jira": {"status": "success"},
                "release_summary": {
                    "overall_severity": "high",
                    "recommended_action": "review_required",
                    "total_signal_count": 1,
                    "high_risk_count": 1,
                    "summary_text": "Release requires manager review.",
                    "top_risks": [],
                },
                "risk_score": {
                    "score": 0.65,
                    "risk_level": "high",
                    "recommended_action": "review_required",
                    "reasons": ["One release blocker"],
                },
                "approval_required": True,
                "approval_status": "pending",
            },
            "citations": [],
            "approval_required": True,
        }
    )

    async def fake_execute_manager_query(
        *,
        settings: FrontendSettings,
        bearer_token: SecretStr,
        query: str,
        conversation_session_id: Any = None,
        release_run_id: str | None = None,
        context_entity_references: AgentEntityReferences | None = None,
    ) -> AgentQueryCallResult:
        return AgentQueryCallResult(
            response=response,
            run_id="chat-layout-request-run-id",
        )

    monkeypatch.setattr(
        app_module,
        "execute_manager_query",
        fake_execute_manager_query,
    )

    app = AppTest.from_file(_STREAMLIT_APP_PATH)
    app.run()
    app.chat_input[0].set_value(
        "What are the biggest release risks this week?"
    ).run()

    assert not app.exception
    assert len(app.chat_message) == 2

    assistant_message = app.chat_message[1]

    assert any(
        "The release has one high-risk blocker." in item.value
        for item in assistant_message.markdown
    )
    assert len(assistant_message.subheader) == 0
    assert len(assistant_message.metric) == 0
    assert len(assistant_message.warning) == 0
    assert len(assistant_message.success) == 0

    assert [
        item.value for item in app.main.subheader
    ] == ["Chat with AgentFlow"]


def test_streamlit_sidebar_renders_latest_release_assessment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Show trusted assessment details in the sidebar, not inside chat."""
    monkeypatch.setenv(
        "AGENTFLOW_FRONTEND_BACKEND_BASE_URL",
        "http://127.0.0.1:8000",
    )
    monkeypatch.setenv(
        "AGENTFLOW_FRONTEND_AUTH_REQUIRED",
        "false",
    )
    get_frontend_settings.cache_clear()

    release_run_id = "14326708-c085-4e6d-9c32-47dc92b24841"

    response = AgentQueryResponse.model_validate(
        {
            "answer": "The release has one high-risk blocker.",
            "plan": {
                "intent": "release_risk_summary",
                "response_depth": "detailed",
                "confidence": 0.98,
                "release_run_id": release_run_id,
                "requires_current_snapshot": True,
                "requires_human_approval": True,
                "routing_reason_code": "fresh_release_risk_request",
            },
            "release_risk": {
                "release_run": {
                    "id": release_run_id,
                    "run_id": "sidebar-assessment-backend-run-id",
                    "query": "What are the biggest release risks this week?",
                    "requested_by": "manager@example.com",
                    "status": "waiting_for_approval",
                    "created_at": "2026-08-01T12:00:00Z",
                },
                "github": {"status": "success"},
                "jira": {"status": "success"},
                "release_summary": {
                    "overall_severity": "high",
                    "recommended_action": "review_required",
                    "total_signal_count": 1,
                    "high_risk_count": 1,
                    "summary_text": "Release requires manager review.",
                    "top_risks": [
                        {
                            "source": "jira",
                            "source_type": "jira_issue",
                            "source_id": "PAY-102",
                            "severity": "high",
                            "score": 0.91,
                            "title": "Payment rollback blocker",
                            "reason": "Open issue blocks safe deployment.",
                            "evidence": {"priority": "P1"},
                        }
                    ],
                },
                "risk_score": {
                    "score": 0.65,
                    "risk_level": "high",
                    "recommended_action": "review_required",
                    "reasons": ["One release blocker"],
                },
                "approval_required": True,
                "approval_reason": "High release risk requires review.",
                "approval_status": "pending",
            },
            "citations": [
                {
                    "source": "jira",
                    "source_type": "jira_issue",
                    "source_id": "PAY-102",
                    "title": "Payment rollback blocker",
                }
            ],
            "approval_required": True,
        }
    )

    async def fake_execute_manager_query(
        *,
        settings: FrontendSettings,
        bearer_token: SecretStr,
        query: str,
        conversation_session_id: Any = None,
        release_run_id: str | None = None,
        context_entity_references: AgentEntityReferences | None = None,
    ) -> AgentQueryCallResult:
        return AgentQueryCallResult(
            response=response,
            run_id="sidebar-assessment-request-run-id",
        )

    monkeypatch.setattr(
        app_module,
        "execute_manager_query",
        fake_execute_manager_query,
    )

    app = AppTest.from_file(_STREAMLIT_APP_PATH)
    app.run()
    app.chat_input[0].set_value(
        "What are the biggest release risks this week?"
    ).run()

    assert not app.exception

    assert "Latest release assessment" in [
        item.value for item in app.sidebar.subheader
    ]
    assert any(
        metric.label == "Risk score"
        and metric.value == "0.65"
        for metric in app.sidebar.metric
    )
    assert any(
        "Payment rollback blocker" in item.value
        for item in app.sidebar.markdown
    )

    assistant_message = app.chat_message[1]
    assert len(assistant_message.metric) == 0
    assert len(assistant_message.subheader) == 0


def test_streamlit_app_requires_token_before_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Block chat submission when no signed JWT is provided."""
    monkeypatch.setenv(
        "AGENTFLOW_FRONTEND_BACKEND_BASE_URL",
        "https://agentflow.example.test",
    )
    get_frontend_settings.cache_clear()

    app = AppTest.from_file(_STREAMLIT_APP_PATH)
    app.run()
    app.chat_input[0].set_value(
        "What are the biggest release risks this week?"
    ).run()

    assert not app.exception
    assert app.error[0].value == (
        "Enter a signed access token before chatting with AgentFlow."
    )


def test_streamlit_app_allows_chat_without_token_in_local_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submit a local chat message without weakening the secure default."""
    monkeypatch.setenv(
        "AGENTFLOW_FRONTEND_BACKEND_BASE_URL",
        "http://127.0.0.1:8000",
    )
    monkeypatch.setenv(
        "AGENTFLOW_FRONTEND_AUTH_REQUIRED",
        "false",
    )
    get_frontend_settings.cache_clear()

    captured: dict[str, Any] = {}

    response = AgentQueryResponse.model_validate(
        {
            "answer": "Local release-risk query completed.",
            "plan": {
                "intent": "release_risk_summary",
                "response_depth": "detailed",
                "confidence": 0.95,
                "requires_human_approval": False,
                "routing_reason_code": "fresh_release_risk_request",
            },
            "citations": [],
            "approval_required": False,
        }
    )

    async def fake_execute_manager_query(
        *,
        settings: FrontendSettings,
        bearer_token: SecretStr,
        query: str,
        conversation_session_id: Any = None,
        release_run_id: str | None = None,
        context_entity_references: AgentEntityReferences | None = None,
    ) -> AgentQueryCallResult:
        captured["auth_required"] = settings.auth_required
        captured["token"] = bearer_token.get_secret_value()
        captured["query"] = query
        captured["conversation_session_id"] = conversation_session_id
        captured["release_run_id"] = release_run_id

        return AgentQueryCallResult(
            response=response,
            run_id="local-development-run-id",
        )

    monkeypatch.setattr(
        app_module,
        "execute_manager_query",
        fake_execute_manager_query,
    )

    app = AppTest.from_file(_STREAMLIT_APP_PATH)
    app.run()
    app.chat_input[0].set_value(
        "What are the biggest release risks this week?"
    ).run()

    assert not app.exception
    assert captured["auth_required"] is False
    assert captured["token"] == ""
    assert captured["query"] == (
        "What are the biggest release risks this week?"
    )
    assert captured["conversation_session_id"] is not None
    assert captured["release_run_id"] is None
    assert not any(
        item.label == "Signed access token"
        for item in app.text_input
    )


def test_streamlit_app_renders_release_risk_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submit a manager query and render its validated backend response."""
    monkeypatch.setenv(
        "AGENTFLOW_FRONTEND_BACKEND_BASE_URL",
        "https://agentflow.example.test",
    )
    get_frontend_settings.cache_clear()

    response = AgentQueryResponse.model_validate(
        {
            "answer": "One critical Jira blocker requires manager review.",
            "plan": {
                "intent": "release_risk_summary",
                "response_depth": "detailed",
                "confidence": 0.98,
                "release_run_id": "14326708-c085-4e6d-9c32-47dc92b24841",
                "requires_current_snapshot": True,
                "requires_human_approval": True,
                "routing_reason_code": "fresh_release_risk_request",
            },
            "release_risk": {
                "release_run": {
                    "id": "14326708-c085-4e6d-9c32-47dc92b24841",
                    "run_id": "release-run-demo",
                    "query": "What are the biggest release risks this week?",
                    "requested_by": "manager@example.com",
                    "status": "waiting_for_approval",
                    "created_at": "2026-07-27T12:00:00Z",
                },
                "github": {
                    "status": "degraded",
                    "error_type": "GitHubUnavailableError",
                    "error_message": "GitHub collection was unavailable.",
                },
                "jira": {
                    "status": "success",
                },
                "release_summary": {
                    "overall_severity": "critical",
                    "recommended_action": "hold",
                    "total_signal_count": 4,
                    "high_risk_count": 1,
                    "summary_text": "Release requires manager review.",
                    "top_risks": [
                        {
                            "source": "jira",
                            "source_type": "jira_issue",
                            "source_id": "PAY-102",
                            "source_url": (
                                "https://jira.example.test/browse/PAY-102"
                            ),
                            "severity": "critical",
                            "score": 0.94,
                            "title": "Payment rollback blocker",
                            "reason": "Open P1 issue blocks safe deployment.",
                            "evidence": {
                                "priority": "P1",
                                "status": "open",
                            },
                        }
                    ],
                },
                "risk_score": {
                    "score": 0.91,
                    "risk_level": "critical",
                    "recommended_action": "hold",
                    "reasons": ["Open P1 release blocker"],
                },
                "synthesis_status": "failed",
                "synthesis_error": "Synthesis provider unavailable.",
                "approval_required": True,
                "approval_reason": "Critical risk requires human approval.",
                "approval_request_id": (
                    "3cc48c03-678b-458e-9418-941e914c220b"
                ),
                "approval_status": "pending",
            },
            "citations": [
                {
                    "source": "jira",
                    "source_type": "jira_issue",
                    "source_id": "PAY-102",
                    "title": "Payment rollback blocker",
                    "source_url": (
                        "https://jira.example.test/browse/PAY-102"
                    ),
                }
            ],
            "approval_required": True,
        }
    )

    async def fake_execute_manager_query(
        *,
        settings: FrontendSettings,
        bearer_token: SecretStr,
        query: str,
        conversation_session_id: Any = None,
        release_run_id: str | None = None,
        context_entity_references: AgentEntityReferences | None = None,
    ) -> AgentQueryCallResult:
        assert str(settings.backend_base_url) == (
            "https://agentflow.example.test/"
        )
        assert bearer_token.get_secret_value() == "signed-test-jwt"
        assert query == "What are the biggest release risks this week?"

        return AgentQueryCallResult(
            response=response,
            run_id="frontend-render-test-run-id",
        )

    monkeypatch.setattr(
        app_module,
        "execute_manager_query",
        fake_execute_manager_query,
    )

    app = AppTest.from_file(_STREAMLIT_APP_PATH)
    app.run()
    _text_input_by_label(app, "Signed access token").input("signed-test-jwt")
    app.chat_input[0].set_value(
        "What are the biggest release risks this week?"
    ).run()

    assert not app.exception
    assert len(app.chat_message) == 2

    assistant_message = app.chat_message[1]

    assert any(
        "One critical Jira blocker" in item.value
        for item in assistant_message.markdown
    )
    assert len(assistant_message.subheader) == 0
    assert len(assistant_message.metric) == 0
    assert len(assistant_message.warning) == 0
    assert len(assistant_message.success) == 0

    assert [
        item.value for item in app.main.subheader
    ] == ["Chat with AgentFlow"]

    assert "Manager approval queue" in [
        item.value for item in app.sidebar.subheader
    ]
    assert _text_input_by_label(
        app,
        "Release run ID",
    ).value == "14326708-c085-4e6d-9c32-47dc92b24841"


def test_streamlit_app_chat_defaults_each_query_to_fresh_assessment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not silently attach a previous release-run ID to a new query."""
    monkeypatch.setenv(
        "AGENTFLOW_FRONTEND_BACKEND_BASE_URL",
        "https://agentflow.example.test",
    )
    get_frontend_settings.cache_clear()

    captured_release_run_ids: list[str | None] = []

    release_run_id = "14326708-c085-4e6d-9c32-47dc92b24841"

    first_response = AgentQueryResponse.model_validate(
        {
            "answer": "One critical Jira blocker requires manager review.",
            "plan": {
                "intent": "release_risk_summary",
                "response_depth": "detailed",
                "confidence": 0.98,
                "release_run_id": release_run_id,
                "requires_current_snapshot": True,
                "requires_human_approval": True,
                "routing_reason_code": "fresh_release_risk_request",
            },
            "release_risk": {
                "release_run": {
                    "id": release_run_id,
                    "run_id": "release-run-demo",
                    "query": "What are the biggest release risks this week?",
                    "requested_by": "manager@example.com",
                    "status": "waiting_for_approval",
                    "created_at": "2026-07-27T12:00:00Z",
                },
                "github": {"status": "success"},
                "jira": {"status": "success"},
                "release_summary": {
                    "overall_severity": "critical",
                    "recommended_action": "hold",
                    "total_signal_count": 1,
                    "high_risk_count": 1,
                    "summary_text": "Release requires manager review.",
                    "top_risks": [],
                },
                "approval_required": True,
            },
            "citations": [],
            "approval_required": True,
        }
    )

    second_response = AgentQueryResponse.model_validate(
        {
            "answer": "The top risk is the payment rollback blocker.",
            "plan": {
                "intent": "explain_risk_score",
                "response_depth": "standard",
                "confidence": 0.9,
                "release_run_id": release_run_id,
                "requires_current_snapshot": True,
                "requires_human_approval": False,
                "routing_reason_code": "followup_explain_risk_score",
            },
            "citations": [],
            "approval_required": False,
        }
    )

    responses = [first_response, second_response]

    async def fake_execute_manager_query(
        *,
        settings: FrontendSettings,
        bearer_token: SecretStr,
        query: str,
        conversation_session_id: Any = None,
        release_run_id: str | None = None,
        context_entity_references: AgentEntityReferences | None = None,
    ) -> AgentQueryCallResult:
        captured_release_run_ids.append(release_run_id)

        return AgentQueryCallResult(
            response=responses.pop(0),
            run_id=f"followup-test-run-id-{len(captured_release_run_ids)}",
        )

    monkeypatch.setattr(
        app_module,
        "execute_manager_query",
        fake_execute_manager_query,
    )

    app = AppTest.from_file(_STREAMLIT_APP_PATH)
    app.run()
    _text_input_by_label(app, "Signed access token").input("signed-test-jwt")
    app.chat_input[0].set_value(
        "What are the biggest release risks this week?"
    ).run()
    app.chat_input[0].set_value("What is the top risk about?").run()

    assert not app.exception
    assert captured_release_run_ids == [None, None]
    assert any(
        "The top risk is the payment rollback blocker." in item.value
        for item in app.markdown
    )


def test_streamlit_app_chat_reuses_release_run_when_followup_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit follow-up mode should reuse the latest persisted run."""
    monkeypatch.setenv(
        "AGENTFLOW_FRONTEND_BACKEND_BASE_URL",
        "https://agentflow.example.test",
    )
    get_frontend_settings.cache_clear()

    captured_release_run_ids: list[str | None] = []

    release_run_id = "14326708-c085-4e6d-9c32-47dc92b24841"

    first_response = AgentQueryResponse.model_validate(
        {
            "answer": "One critical Jira blocker requires manager review.",
            "plan": {
                "intent": "release_risk_summary",
                "response_depth": "detailed",
                "confidence": 0.98,
                "release_run_id": release_run_id,
                "requires_current_snapshot": True,
                "requires_human_approval": True,
                "routing_reason_code": "fresh_release_risk_request",
            },
            "release_risk": {
                "release_run": {
                    "id": release_run_id,
                    "run_id": "release-run-demo",
                    "query": "What are the biggest release risks this week?",
                    "requested_by": "manager@example.com",
                    "status": "waiting_for_approval",
                    "created_at": "2026-07-27T12:00:00Z",
                },
                "github": {"status": "success"},
                "jira": {"status": "success"},
                "release_summary": {
                    "overall_severity": "critical",
                    "recommended_action": "hold",
                    "total_signal_count": 1,
                    "high_risk_count": 1,
                    "summary_text": "Release requires manager review.",
                    "top_risks": [],
                },
                "approval_required": True,
            },
            "citations": [],
            "approval_required": True,
        }
    )

    second_response = AgentQueryResponse.model_validate(
        {
            "answer": "The top risk is the payment rollback blocker.",
            "plan": {
                "intent": "explain_risk_score",
                "response_depth": "standard",
                "confidence": 0.9,
                "release_run_id": release_run_id,
                "requires_current_snapshot": True,
                "requires_human_approval": False,
                "routing_reason_code": "followup_explain_risk_score",
            },
            "citations": [],
            "approval_required": False,
        }
    )

    responses = [first_response, second_response]

    async def fake_execute_manager_query(
        *,
        settings: FrontendSettings,
        bearer_token: SecretStr,
        query: str,
        conversation_session_id: Any = None,
        release_run_id: str | None = None,
        context_entity_references: AgentEntityReferences | None = None,
    ) -> AgentQueryCallResult:
        captured_release_run_ids.append(release_run_id)

        return AgentQueryCallResult(
            response=responses.pop(0),
            run_id=f"followup-test-run-id-{len(captured_release_run_ids)}",
        )

    monkeypatch.setattr(
        app_module,
        "execute_manager_query",
        fake_execute_manager_query,
    )

    app = AppTest.from_file(_STREAMLIT_APP_PATH)
    app.run()
    _text_input_by_label(app, "Signed access token").input("signed-test-jwt")
    app.chat_input[0].set_value(
        "What are the biggest release risks this week?"
    ).run()
    app.radio[0].set_value("Follow up on latest run").run()
    app.chat_input[0].set_value("What is the top risk about?").run()

    assert not app.exception
    assert captured_release_run_ids == [None, release_run_id]
    assert any(
        "The top risk is the payment rollback blocker." in item.value
        for item in app.markdown
    )


def test_streamlit_app_rejects_followup_without_release_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Follow-up mode must fail closed when no persisted run is available."""

    monkeypatch.setenv(
        "AGENTFLOW_FRONTEND_BACKEND_BASE_URL",
        "https://agentflow.example.test",
    )
    get_frontend_settings.cache_clear()

    call_count = 0

    async def fake_execute_manager_query(
        *,
        settings: FrontendSettings,
        bearer_token: SecretStr,
        query: str,
        conversation_session_id: Any = None,
        release_run_id: str | None = None,
        context_entity_references: AgentEntityReferences | None = None,
    ) -> AgentQueryCallResult:
        nonlocal call_count
        call_count += 1
        raise AssertionError("Backend must not be called without follow-up context.")

    monkeypatch.setattr(
        app_module,
        "execute_manager_query",
        fake_execute_manager_query,
    )

    app = AppTest.from_file(_STREAMLIT_APP_PATH)
    app.run()
    _text_input_by_label(app, "Signed access token").input("signed-test-jwt")
    app.radio[0].set_value("Follow up on latest run").run()
    app.chat_input[0].set_value("What is the workflow status?").run()

    assert not app.exception
    assert call_count == 0
    assert any(
        "Run a fresh assessment before selecting follow-up mode."
        in item.value
        for item in app.error
    )


@pytest.mark.asyncio
async def test_load_pending_approvals_uses_typed_api_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delegate pending approval retrieval to the existing API client."""
    captured: dict[str, Any] = {}

    pending_response = PendingReleaseRunApprovalList.model_validate(
        {
            "approval_status": "pending",
            "approvals": [],
        }
    )

    class FakeAgentFlowAPIClient:
        """Async context-manager fake for pending approval retrieval."""

        def __init__(
            self,
            *,
            settings: FrontendSettings,
            bearer_token: SecretStr,
        ) -> None:
            captured["settings"] = settings
            captured["authorization_value"] = (
                bearer_token.get_secret_value()
            )

        async def __aenter__(self) -> FakeAgentFlowAPIClient:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: object | None,
        ) -> None:
            return None

        async def list_pending_approvals(
            self,
        ) -> PendingApprovalsCallResult:
            return PendingApprovalsCallResult(
                response=pending_response,
                run_id="pending-approval-test-run-id",
            )

    monkeypatch.setattr(
        app_module,
        "AgentFlowAPIClient",
        FakeAgentFlowAPIClient,
    )

    settings = FrontendSettings(
        backend_base_url="https://agentflow.example.test",
        _env_file=None,
    )

    result = await app_module.load_pending_approvals(
        settings=settings,
        bearer_token=SecretStr("signed-test-jwt"),
    )

    assert result.response.approval_status == "pending"
    assert captured["settings"] == settings
    assert captured["authorization_value"] == "signed-test-jwt"


def test_streamlit_app_renders_pending_approval_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load and render durable pending approval requests."""
    monkeypatch.setenv(
        "AGENTFLOW_FRONTEND_BACKEND_BASE_URL",
        "https://agentflow.example.test",
    )
    get_frontend_settings.cache_clear()

    pending_response = PendingReleaseRunApprovalList.model_validate(
        {
            "approval_status": "pending",
            "approvals": [
                {
                    "id": "3cc48c03-678b-458e-9418-941e914c220b",
                    "release_run_id": (
                        "14326708-c085-4e6d-9c32-47dc92b24841"
                    ),
                    "approval_status": "pending",
                    "approval_reason": (
                        "Critical release risk requires manager review."
                    ),
                    "approval_policy_version": "hitl_policy_v1",
                    "created_at": "2026-07-27T12:00:00Z",
                }
            ],
        }
    )

    async def fake_load_pending_approvals(
        *,
        settings: FrontendSettings,
        bearer_token: SecretStr,
    ) -> PendingApprovalsCallResult:
        assert str(settings.backend_base_url) == (
            "https://agentflow.example.test/"
        )
        assert bearer_token.get_secret_value() == "signed-test-jwt"

        return PendingApprovalsCallResult(
            response=pending_response,
            run_id="pending-queue-render-run-id",
        )

    monkeypatch.setattr(
        app_module,
        "load_pending_approvals",
        fake_load_pending_approvals,
    )

    app = AppTest.from_file(_STREAMLIT_APP_PATH)
    app.run()
    _text_input_by_label(app, "Signed access token").input("signed-test-jwt")
    _button_by_label(app, "Load pending approvals").click().run()

    assert not app.exception
    assert any(
        item.value == "Manager approval queue"
        for item in app.subheader
    )
    assert any(
        "Critical release risk requires manager review." in item.value
        for item in app.markdown
    )
    assert any(
        "pending-queue-render-run-id" in item.value
        for item in app.caption
    )


@pytest.mark.asyncio
async def test_decide_pending_approval_uses_typed_api_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Send only the validated decision and backend-owned identifiers."""
    captured: dict[str, Any] = {}

    approval_response = ReleaseRunApproval.model_validate(
        {
            "id": "3cc48c03-678b-458e-9418-941e914c220b",
            "release_run_id": "14326708-c085-4e6d-9c32-47dc92b24841",
            "approval_status": "approved",
            "approval_reason": "Critical risk requires review.",
            "approval_policy_version": "hitl_policy_v1",
            "decided_by": "manager@example.com",
            "decision_note": "Evidence reviewed.",
            "created_at": "2026-07-27T12:00:00Z",
            "decided_at": "2026-07-27T12:05:00Z",
        }
    )

    class FakeAgentFlowAPIClient:
        """Async fake for one approval decision."""

        def __init__(
            self,
            *,
            settings: FrontendSettings,
            bearer_token: SecretStr,
        ) -> None:
            captured["settings"] = settings
            captured["authorization_value"] = (
                bearer_token.get_secret_value()
            )

        async def __aenter__(self) -> FakeAgentFlowAPIClient:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: object | None,
        ) -> None:
            return None

        async def decide_release_run_approval(
            self,
            *,
            release_run_id: str,
            approval_id: str,
            decision: ReleaseRunApprovalDecisionRequest,
        ) -> ApprovalDecisionCallResult:
            captured["release_run_id"] = release_run_id
            captured["approval_id"] = approval_id
            captured["decision"] = decision

            return ApprovalDecisionCallResult(
                response=approval_response,
                run_id="approval-decision-test-run-id",
            )

    monkeypatch.setattr(
        app_module,
        "AgentFlowAPIClient",
        FakeAgentFlowAPIClient,
    )

    settings = FrontendSettings(
        backend_base_url="https://agentflow.example.test",
        _env_file=None,
    )

    result = await app_module.decide_pending_approval(
        settings=settings,
        bearer_token=SecretStr("signed-test-jwt"),
        release_run_id="14326708-c085-4e6d-9c32-47dc92b24841",
        approval_id="3cc48c03-678b-458e-9418-941e914c220b",
        approval_status=ReleaseApprovalDecisionStatus.APPROVED,
        decision_note="Evidence reviewed.",
    )

    assert result.response.approval_status == "approved"
    assert captured["authorization_value"] == "signed-test-jwt"
    assert captured["release_run_id"] == (
        "14326708-c085-4e6d-9c32-47dc92b24841"
    )
    assert captured["approval_id"] == (
        "3cc48c03-678b-458e-9418-941e914c220b"
    )

    captured_decision = captured["decision"]
    assert isinstance(
        captured_decision,
        ReleaseRunApprovalDecisionRequest,
    )
    assert captured_decision.approval_status is (
        ReleaseApprovalDecisionStatus.APPROVED
    )
    assert captured_decision.decision_note == "Evidence reviewed."
    assert result.run_id == "approval-decision-test-run-id"


@pytest.mark.parametrize(
    ("decision_button_label", "expected_status"),
    [
        ("Approve", ReleaseApprovalDecisionStatus.APPROVED),
        ("Reject", ReleaseApprovalDecisionStatus.REJECTED),
    ],
)
def test_streamlit_app_submits_manager_approval_decision(
    monkeypatch: pytest.MonkeyPatch,
    decision_button_label: str,
    expected_status: ReleaseApprovalDecisionStatus,
) -> None:
    """Persist approved and rejected decisions through the typed helper."""
    monkeypatch.setenv(
        "AGENTFLOW_FRONTEND_BACKEND_BASE_URL",
        "https://agentflow.example.test",
    )
    get_frontend_settings.cache_clear()

    captured: dict[str, Any] = {}

    pending_response = PendingReleaseRunApprovalList.model_validate(
        {
            "approval_status": "pending",
            "approvals": [
                {
                    "id": "3cc48c03-678b-458e-9418-941e914c220b",
                    "release_run_id": (
                        "14326708-c085-4e6d-9c32-47dc92b24841"
                    ),
                    "approval_status": "pending",
                    "approval_reason": (
                        "Critical release risk requires manager review."
                    ),
                    "approval_policy_version": "hitl_policy_v1",
                    "created_at": "2026-07-27T12:00:00Z",
                }
            ],
        }
    )

    async def fake_load_pending_approvals(
        *,
        settings: FrontendSettings,
        bearer_token: SecretStr,
    ) -> PendingApprovalsCallResult:
        assert str(settings.backend_base_url) == (
            "https://agentflow.example.test/"
        )
        assert bearer_token.get_secret_value() == "signed-test-jwt"

        return PendingApprovalsCallResult(
            response=pending_response,
            run_id="pending-decision-queue-run-id",
        )

    async def fake_decide_pending_approval(
        *,
        settings: FrontendSettings,
        bearer_token: SecretStr,
        release_run_id: str,
        approval_id: str,
        approval_status: ReleaseApprovalDecisionStatus,
        decision_note: str | None,
    ) -> ApprovalDecisionCallResult:
        captured["settings"] = settings
        captured["authorization_value"] = (
            bearer_token.get_secret_value()
        )
        captured["release_run_id"] = release_run_id
        captured["approval_id"] = approval_id
        captured["approval_status"] = approval_status
        captured["decision_note"] = decision_note

        approval_response = ReleaseRunApproval.model_validate(
            {
                "id": approval_id,
                "release_run_id": release_run_id,
                "approval_status": approval_status.value,
                "approval_reason": (
                    "Critical release risk requires manager review."
                ),
                "approval_policy_version": "hitl_policy_v1",
                "decided_by": "manager@example.com",
                "decision_note": decision_note,
                "created_at": "2026-07-27T12:00:00Z",
                "decided_at": "2026-07-27T12:05:00Z",
            }
        )

        return ApprovalDecisionCallResult(
            response=approval_response,
            run_id="approval-ui-decision-run-id",
        )

    monkeypatch.setattr(
        app_module,
        "load_pending_approvals",
        fake_load_pending_approvals,
    )
    monkeypatch.setattr(
        app_module,
        "decide_pending_approval",
        fake_decide_pending_approval,
    )

    app = AppTest.from_file(_STREAMLIT_APP_PATH)
    app.run()
    _text_input_by_label(app, "Signed access token").input("signed-test-jwt")
    _button_by_label(app, "Load pending approvals").click().run()

    _text_area_by_label(app, "Decision note").input("Evidence reviewed by manager.")
    _button_by_label(app, decision_button_label).click().run()

    assert not app.exception
    assert captured["authorization_value"] == "signed-test-jwt"
    assert captured["release_run_id"] == (
        "14326708-c085-4e6d-9c32-47dc92b24841"
    )
    assert captured["approval_id"] == (
        "3cc48c03-678b-458e-9418-941e914c220b"
    )
    assert captured["approval_status"] is expected_status
    assert captured["decision_note"] == (
        "Evidence reviewed by manager."
    )
    assert any(
        expected_status.value in item.value
        for item in app.success
    )
    assert any(
        "approval-ui-decision-run-id" in item.value
        for item in app.caption
    )

    slack_button_is_visible = any(
        button.label == "Send approved Slack alert"
        for button in app.button
    )
    assert slack_button_is_visible is (
        expected_status is ReleaseApprovalDecisionStatus.APPROVED
    )


@pytest.mark.asyncio
async def test_send_approved_release_slack_alert_uses_typed_api_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delegate Slack delivery without reproducing backend approval policy."""
    captured: dict[str, Any] = {}

    slack_response = SlackReleaseAlertResult.model_validate(
        {
            "sent": True,
            "slack_channel": "#release-alerts",
            "slack_timestamp": "1753632600.000100",
            "risk_level": "critical",
            "risk_score": 0.91,
            "recommended_action": "Delay deployment pending remediation.",
        }
    )

    class FakeAgentFlowAPIClient:
        """Async fake for one approval-gated Slack request."""

        def __init__(
            self,
            *,
            settings: FrontendSettings,
            bearer_token: SecretStr,
        ) -> None:
            captured["settings"] = settings
            captured["authorization_value"] = (
                bearer_token.get_secret_value()
            )

        async def __aenter__(self) -> FakeAgentFlowAPIClient:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: object | None,
        ) -> None:
            return None

        async def send_release_run_slack_alert(
            self,
            *,
            release_run_id: str,
        ) -> SlackAlertCallResult:
            captured["release_run_id"] = release_run_id

            return SlackAlertCallResult(
                response=slack_response,
                run_id="slack-alert-helper-test-run-id",
            )

    monkeypatch.setattr(
        app_module,
        "AgentFlowAPIClient",
        FakeAgentFlowAPIClient,
    )

    settings = FrontendSettings(
        backend_base_url="https://agentflow.example.test",
        _env_file=None,
    )

    result = await app_module.send_approved_release_slack_alert(
        settings=settings,
        bearer_token=SecretStr("signed-notify-jwt"),
        release_run_id="14326708-c085-4e6d-9c32-47dc92b24841",
    )

    assert captured["settings"] == settings
    assert captured["authorization_value"] == "signed-notify-jwt"
    assert captured["release_run_id"] == (
        "14326708-c085-4e6d-9c32-47dc92b24841"
    )
    assert result.response.sent is True
    assert result.response.slack_channel == "#release-alerts"
    assert result.run_id == "slack-alert-helper-test-run-id"


def test_streamlit_app_sends_slack_after_approved_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose Slack delivery only after an approved backend decision."""
    monkeypatch.setenv(
        "AGENTFLOW_FRONTEND_BACKEND_BASE_URL",
        "https://agentflow.example.test",
    )
    get_frontend_settings.cache_clear()

    captured: dict[str, Any] = {}

    pending_response = PendingReleaseRunApprovalList.model_validate(
        {
            "approval_status": "pending",
            "approvals": [
                {
                    "id": "3cc48c03-678b-458e-9418-941e914c220b",
                    "release_run_id": (
                        "14326708-c085-4e6d-9c32-47dc92b24841"
                    ),
                    "approval_status": "pending",
                    "approval_reason": (
                        "Critical release risk requires manager review."
                    ),
                    "approval_policy_version": "hitl_policy_v1",
                    "created_at": "2026-07-27T12:00:00Z",
                }
            ],
        }
    )

    async def fake_load_pending_approvals(
        *,
        settings: FrontendSettings,
        bearer_token: SecretStr,
    ) -> PendingApprovalsCallResult:
        return PendingApprovalsCallResult(
            response=pending_response,
            run_id="slack-ui-pending-run-id",
        )

    async def fake_decide_pending_approval(
        *,
        settings: FrontendSettings,
        bearer_token: SecretStr,
        release_run_id: str,
        approval_id: str,
        approval_status: ReleaseApprovalDecisionStatus,
        decision_note: str | None,
    ) -> ApprovalDecisionCallResult:
        assert approval_status is (
            ReleaseApprovalDecisionStatus.APPROVED
        )

        approval_response = ReleaseRunApproval.model_validate(
            {
                "id": approval_id,
                "release_run_id": release_run_id,
                "approval_status": "approved",
                "approval_reason": (
                    "Critical release risk requires manager review."
                ),
                "approval_policy_version": "hitl_policy_v1",
                "decided_by": "manager@example.com",
                "decision_note": decision_note,
                "created_at": "2026-07-27T12:00:00Z",
                "decided_at": "2026-07-27T12:05:00Z",
            }
        )

        return ApprovalDecisionCallResult(
            response=approval_response,
            run_id="slack-ui-approval-run-id",
        )

    async def fake_send_approved_release_slack_alert(
        *,
        settings: FrontendSettings,
        bearer_token: SecretStr,
        release_run_id: str,
    ) -> SlackAlertCallResult:
        captured["settings"] = settings
        captured["authorization_value"] = (
            bearer_token.get_secret_value()
        )
        captured["release_run_id"] = release_run_id

        response = SlackReleaseAlertResult.model_validate(
            {
                "sent": True,
                "slack_channel": "#release-alerts",
                "slack_timestamp": "1753632600.000100",
                "risk_level": "critical",
                "risk_score": 0.91,
                "recommended_action": (
                    "Delay deployment pending remediation."
                ),
            }
        )

        return SlackAlertCallResult(
            response=response,
            run_id="slack-ui-delivery-run-id",
        )

    monkeypatch.setattr(
        app_module,
        "load_pending_approvals",
        fake_load_pending_approvals,
    )
    monkeypatch.setattr(
        app_module,
        "decide_pending_approval",
        fake_decide_pending_approval,
    )
    monkeypatch.setattr(
        app_module,
        "send_approved_release_slack_alert",
        fake_send_approved_release_slack_alert,
    )

    app = AppTest.from_file(_STREAMLIT_APP_PATH)
    app.run()
    _text_input_by_label(app, "Signed access token").input("signed-notify-jwt")
    _button_by_label(app, "Load pending approvals").click().run()
    _text_area_by_label(app, "Decision note").input("Approved for Slack notification.")

    approve_button = next(
        button for button in app.button if button.label == "Approve"
    )
    approve_button.click().run()

    assert not app.exception
    assert any(
        button.label == "Send approved Slack alert"
        for button in app.button
    )

    slack_button = next(
        button
        for button in app.button
        if button.label == "Send approved Slack alert"
    )
    slack_button.click().run()

    assert not app.exception
    assert captured["authorization_value"] == "signed-notify-jwt"
    assert captured["release_run_id"] == (
        "14326708-c085-4e6d-9c32-47dc92b24841"
    )
    assert any(
        "#release-alerts" in item.value
        for item in app.success
    )
    assert any(
        "slack-ui-delivery-run-id" in item.value
        for item in app.caption
    )
    assert not any(
        button.label == "Send approved Slack alert"
        for button in app.button
    )


@pytest.mark.asyncio
async def test_load_release_run_status_uses_typed_api_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load persisted workflow state through the typed FastAPI client."""
    captured: dict[str, Any] = {}

    status_response = ReleaseRunStatus.model_validate(
        {
            "id": "14326708-c085-4e6d-9c32-47dc92b24841",
            "run_id": "workflow-status-backend-run-id",
            "query": "What are the biggest release risks this week?",
            "requested_by": "manager@example.com",
            "status": "approved",
            "created_at": "2026-07-27T12:00:00Z",
            "completed_at": "2026-07-27T12:05:00Z",
        }
    )

    class FakeAgentFlowAPIClient:
        """Async fake for release-run status retrieval."""

        def __init__(
            self,
            *,
            settings: FrontendSettings,
            bearer_token: SecretStr,
        ) -> None:
            captured["settings"] = settings
            captured["authorization_value"] = (
                bearer_token.get_secret_value()
            )

        async def __aenter__(self) -> FakeAgentFlowAPIClient:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: object | None,
        ) -> None:
            return None

        async def get_release_run_status(
            self,
            *,
            release_run_id: str,
        ) -> ReleaseRunStatusCallResult:
            captured["release_run_id"] = release_run_id

            return ReleaseRunStatusCallResult(
                response=status_response,
                run_id="status-helper-request-run-id",
            )

    monkeypatch.setattr(
        app_module,
        "AgentFlowAPIClient",
        FakeAgentFlowAPIClient,
    )

    settings = FrontendSettings(
        backend_base_url="https://agentflow.example.test",
        _env_file=None,
    )

    result = await app_module.load_release_run_status(
        settings=settings,
        bearer_token=SecretStr("signed-read-jwt"),
        release_run_id="14326708-c085-4e6d-9c32-47dc92b24841",
    )

    assert captured["settings"] == settings
    assert captured["authorization_value"] == "signed-read-jwt"
    assert captured["release_run_id"] == (
        "14326708-c085-4e6d-9c32-47dc92b24841"
    )
    assert result.response.status == "approved"
    assert result.run_id == "status-helper-request-run-id"


@pytest.mark.asyncio
async def test_load_release_run_events_uses_typed_api_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load the append-only audit trail through the typed FastAPI client."""
    captured: dict[str, Any] = {}

    events_response = ReleaseRunEventList.model_validate(
        {
            "release_run_id": "14326708-c085-4e6d-9c32-47dc92b24841",
            "events": [
                {
                    "id": "a23cfa16-048d-471a-9219-9370365feb66",
                    "release_run_id": (
                        "14326708-c085-4e6d-9c32-47dc92b24841"
                    ),
                    "event_type": "approval_decided",
                    "event_status": "completed",
                    "message": "Release run approved by manager.",
                    "metadata_json": {
                        "approval_policy_version": "hitl_policy_v1"
                    },
                    "created_at": "2026-07-27T12:05:00Z",
                }
            ],
        }
    )

    class FakeAgentFlowAPIClient:
        """Async fake for release-run event retrieval."""

        def __init__(
            self,
            *,
            settings: FrontendSettings,
            bearer_token: SecretStr,
        ) -> None:
            captured["settings"] = settings
            captured["authorization_value"] = (
                bearer_token.get_secret_value()
            )

        async def __aenter__(self) -> FakeAgentFlowAPIClient:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: object | None,
        ) -> None:
            return None

        async def list_release_run_events(
            self,
            *,
            release_run_id: str,
        ) -> ReleaseRunEventsCallResult:
            captured["release_run_id"] = release_run_id

            return ReleaseRunEventsCallResult(
                response=events_response,
                run_id="events-helper-request-run-id",
            )

    monkeypatch.setattr(
        app_module,
        "AgentFlowAPIClient",
        FakeAgentFlowAPIClient,
    )

    settings = FrontendSettings(
        backend_base_url="https://agentflow.example.test",
        _env_file=None,
    )

    result = await app_module.load_release_run_events(
        settings=settings,
        bearer_token=SecretStr("signed-read-jwt"),
        release_run_id="14326708-c085-4e6d-9c32-47dc92b24841",
    )

    assert captured["settings"] == settings
    assert captured["authorization_value"] == "signed-read-jwt"
    assert captured["release_run_id"] == (
        "14326708-c085-4e6d-9c32-47dc92b24841"
    )
    assert len(result.response.events) == 1
    assert result.response.events[0].event_type == "approval_decided"
    assert result.run_id == "events-helper-request-run-id"


def test_validate_release_run_id_normalizes_uuid() -> None:
    """Normalize a valid release-run identifier before API use."""
    assert app_module.validate_release_run_id(
        " 14326708-c085-4e6d-9c32-47dc92b24841 "
    ) == "14326708-c085-4e6d-9c32-47dc92b24841"


def test_validate_release_run_id_rejects_invalid_value() -> None:
    """Reject malformed release-run identifiers at the UI boundary."""
    with pytest.raises(ValidationError):
        app_module.validate_release_run_id("not-a-release-run-id")


def test_streamlit_app_renders_workflow_status_and_audit_timeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load and render persisted workflow state and audit events."""
    monkeypatch.setenv(
        "AGENTFLOW_FRONTEND_BACKEND_BASE_URL",
        "https://agentflow.example.test",
    )
    get_frontend_settings.cache_clear()

    captured: dict[str, Any] = {}

    status_response = ReleaseRunStatus.model_validate(
        {
            "id": "14326708-c085-4e6d-9c32-47dc92b24841",
            "run_id": "workflow-backend-run-id",
            "query": "What are the biggest release risks this week?",
            "requested_by": "manager@example.com",
            "status": "approved",
            "created_at": "2026-07-27T12:00:00Z",
            "completed_at": "2026-07-27T12:05:00Z",
        }
    )

    events_response = ReleaseRunEventList.model_validate(
        {
            "release_run_id": "14326708-c085-4e6d-9c32-47dc92b24841",
            "events": [
                {
                    "id": "a23cfa16-048d-471a-9219-9370365feb66",
                    "release_run_id": (
                        "14326708-c085-4e6d-9c32-47dc92b24841"
                    ),
                    "event_type": "approval_decided",
                    "event_status": "completed",
                    "message": "Release run approved by manager.",
                    "metadata_json": {
                        "approval_policy_version": "hitl_policy_v1"
                    },
                    "created_at": "2026-07-27T12:05:00Z",
                }
            ],
        }
    )

    async def fake_load_release_run_status(
        *,
        settings: FrontendSettings,
        bearer_token: SecretStr,
        release_run_id: str,
    ) -> ReleaseRunStatusCallResult:
        captured["status_settings"] = settings
        captured["status_authorization_value"] = bearer_token.get_secret_value()
        captured["status_release_run_id"] = release_run_id

        return ReleaseRunStatusCallResult(
            response=status_response,
            run_id="workflow-status-ui-run-id",
        )

    async def fake_load_release_run_events(
        *,
        settings: FrontendSettings,
        bearer_token: SecretStr,
        release_run_id: str,
    ) -> ReleaseRunEventsCallResult:
        captured["events_settings"] = settings
        captured["events_authorization_value"] = bearer_token.get_secret_value()
        captured["events_release_run_id"] = release_run_id

        return ReleaseRunEventsCallResult(
            response=events_response,
            run_id="workflow-events-ui-run-id",
        )

    monkeypatch.setattr(
        app_module,
        "load_release_run_status",
        fake_load_release_run_status,
    )
    monkeypatch.setattr(
        app_module,
        "load_release_run_events",
        fake_load_release_run_events,
    )

    app = AppTest.from_file(_STREAMLIT_APP_PATH)
    app.run()

    token_input = _text_input_by_label(
        app,
        "Signed access token",
    )
    release_run_input = _text_input_by_label(
        app,
        "Release run ID",
    )

    token_input.input("signed-read-jwt")
    release_run_input.input(
        "14326708-c085-4e6d-9c32-47dc92b24841"
    )

    status_button = next(
        button
        for button in app.button
        if button.label == "Load workflow status"
    )
    status_button.click().run()

    assert not app.exception
    assert captured["status_authorization_value"] == "signed-read-jwt"
    assert captured["status_release_run_id"] == (
        "14326708-c085-4e6d-9c32-47dc92b24841"
    )
    assert any(
        metric.label == "Workflow status"
        and metric.value == "approved"
        for metric in app.metric
    )
    assert any(
        "workflow-status-ui-run-id" in item.value
        for item in app.caption
    )

    events_button = next(
        button
        for button in app.button
        if button.label == "Load audit timeline"
    )
    events_button.click().run()

    assert not app.exception
    assert captured["events_authorization_value"] == "signed-read-jwt"
    assert captured["events_release_run_id"] == (
        "14326708-c085-4e6d-9c32-47dc92b24841"
    )
    assert any(
        "Release run approved by manager." in item.value
        for item in app.markdown
    )
    assert any(
        "workflow-events-ui-run-id" in item.value
        for item in app.caption
    )


def test_streamlit_app_prefills_workflow_lookup_from_latest_release_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse the latest chat release-run ID in workflow lookup controls."""
    monkeypatch.setenv(
        "AGENTFLOW_FRONTEND_BACKEND_BASE_URL",
        "http://127.0.0.1:8000",
    )
    monkeypatch.setenv(
        "AGENTFLOW_FRONTEND_AUTH_REQUIRED",
        "false",
    )
    get_frontend_settings.cache_clear()

    release_run_id = "14326708-c085-4e6d-9c32-47dc92b24841"

    response = AgentQueryResponse.model_validate(
        {
            "answer": "The release requires manager review.",
            "plan": {
                "intent": "release_risk_summary",
                "response_depth": "detailed",
                "confidence": 0.98,
                "release_run_id": release_run_id,
                "requires_current_snapshot": True,
                "requires_human_approval": True,
                "routing_reason_code": "fresh_release_risk_request",
            },
            "release_risk": {
                "release_run": {
                    "id": release_run_id,
                    "run_id": "workflow-prefill-backend-run-id",
                    "query": "What are the biggest release risks this week?",
                    "requested_by": "manager@example.com",
                    "status": "waiting_for_approval",
                    "created_at": "2026-08-01T12:00:00Z",
                },
                "github": {"status": "success"},
                "jira": {"status": "success"},
                "release_summary": {
                    "overall_severity": "high",
                    "recommended_action": "review_required",
                    "total_signal_count": 1,
                    "high_risk_count": 1,
                    "summary_text": "Release requires manager review.",
                    "top_risks": [],
                },
                "approval_required": True,
                "approval_status": "pending",
            },
            "citations": [],
            "approval_required": True,
        }
    )

    async def fake_execute_manager_query(
        *,
        settings: FrontendSettings,
        bearer_token: SecretStr,
        query: str,
        conversation_session_id: Any = None,
        release_run_id: str | None = None,
        context_entity_references: AgentEntityReferences | None = None,
    ) -> AgentQueryCallResult:
        return AgentQueryCallResult(
            response=response,
            run_id="workflow-prefill-request-run-id",
        )

    monkeypatch.setattr(
        app_module,
        "execute_manager_query",
        fake_execute_manager_query,
    )

    app = AppTest.from_file(_STREAMLIT_APP_PATH)
    app.run()
    app.chat_input[0].set_value(
        "What are the biggest release risks this week?"
    ).run()

    assert not app.exception
    assert _text_input_by_label(app, "Release run ID").value == release_run_id


def test_get_focused_entity_context_returns_single_pr() -> None:
    """Preserve one backend-validated PR for a pronoun follow-up."""

    response = AgentQueryResponse.model_validate(
        {
            "answer": "PR 4 has unresolved release risk.",
            "plan": {
                "intent": "github_pr_question",
                "response_depth": "standard",
                "confidence": 0.98,
                "entity_references": {
                    "pull_request_numbers": [4],
                },
                "routing_reason_code": "matched_github_pr_reference",
            },
            "citations": [],
            "approval_required": False,
        }
    )

    context = app_module.get_focused_entity_context(response)

    assert context == AgentEntityReferences(pull_request_numbers=[4])


def test_get_focused_entity_context_rejects_ambiguous_summary() -> None:
    """Do not infer a pronoun target from a multi-entity summary."""

    response = AgentQueryResponse.model_validate(
        {
            "answer": "Several risks require review.",
            "plan": {
                "intent": "release_risk_summary",
                "response_depth": "standard",
                "confidence": 0.98,
                "entity_references": {
                    "pull_request_numbers": [4],
                    "jira_issue_keys": ["SCRUM-2"],
                },
                "routing_reason_code": "matched_release_risk_summary",
            },
            "citations": [],
            "approval_required": True,
        }
    )

    assert app_module.get_focused_entity_context(response) is None

def test_release_assessment_rendering_is_limited_to_summary_intent() -> None:
    """Focused follow-ups should not redraw the full release dashboard."""

    summary_response = AgentQueryResponse.model_validate(
        {
            "answer": "The release requires manager review.",
            "plan": {
                "intent": "release_risk_summary",
                "response_depth": "standard",
                "confidence": 0.98,
                "routing_reason_code": "matched_release_risk_summary",
            },
            "release_risk": {
                "release_run": {
                    "id": "14326708-c085-4e6d-9c32-47dc92b24841",
                    "run_id": "release-run-render-policy",
                    "query": "What are the biggest release risks?",
                    "requested_by": "manager@example.com",
                    "status": "waiting_for_approval",
                    "created_at": "2026-07-29T12:00:00Z",
                },
                "github": {
                    "status": "success",
                },
                "jira": {
                    "status": "success",
                },
                "release_summary": {
                    "overall_severity": "high",
                    "recommended_action": "review_required",
                    "total_signal_count": 18,
                    "high_risk_count": 6,
                    "summary_text": "Release requires manager review.",
                },
            },
            "citations": [],
            "approval_required": True,
        }
    )

    focused_response = summary_response.model_copy(
        update={
            "plan": summary_response.plan.model_copy(
                update={"intent": "github_pr_question"}
            )
        }
    )

    assert app_module.should_render_release_assessment(summary_response) is True
    assert app_module.should_render_release_assessment(focused_response) is False
