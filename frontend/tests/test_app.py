"""Tests for the AgentFlow Streamlit manager experience."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr
from streamlit.testing.v1 import AppTest

import agentflow_frontend.app as app_module
from agentflow_frontend.api_client import AgentQueryCallResult
from agentflow_frontend.api_models import AgentQueryRequest, AgentQueryResponse
from agentflow_frontend.config import FrontendSettings, get_frontend_settings


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


def test_streamlit_app_renders_secure_query_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Render the initial manager screen without making a backend request."""
    monkeypatch.setenv(
        "AGENTFLOW_FRONTEND_BACKEND_BASE_URL",
        "https://agentflow.example.test",
    )
    get_frontend_settings.cache_clear()

    app = AppTest.from_file("streamlit_app.py")
    app.run()

    assert not app.exception
    assert app.title[0].value == "AgentFlow AI"
    assert app.text_input[0].label == "Signed access token"
    assert app.text_area[0].label == "Manager question"
    assert app.button[0].label == "Analyze release risks"


def test_streamlit_app_requires_token_before_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Block query submission when no signed JWT is provided."""
    monkeypatch.setenv(
        "AGENTFLOW_FRONTEND_BACKEND_BASE_URL",
        "https://agentflow.example.test",
    )
    get_frontend_settings.cache_clear()

    app = AppTest.from_file("streamlit_app.py")
    app.run()
    app.button[0].click().run()

    assert not app.exception
    assert app.error[0].value == (
        "Enter a signed access token before submitting the query."
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

    app = AppTest.from_file("streamlit_app.py")
    app.run()
    app.text_input[0].input("signed-test-jwt")
    app.button[0].click().run()

    assert not app.exception
    assert any(
        item.value == "Agent answer"
        for item in app.subheader
    )
    assert any(
        "One critical Jira blocker" in item.value
        for item in app.markdown
    )
    assert any(
        "Degraded dependency data: GitHub" in item.value
        for item in app.warning
    )
    assert any(
        "Human approval is required" in item.value
        for item in app.warning
    )
    assert any(
        item.value == "Payment rollback blocker"
        for item in app.markdown
    )
    assert any(
        "frontend-render-test-run-id" in item.value
        for item in app.caption
    )
