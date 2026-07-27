"""Tests for the AgentFlow Streamlit manager experience."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr
from streamlit.testing.v1 import AppTest

import agentflow_frontend.app as app_module
from agentflow_frontend.api_client import (
    AgentQueryCallResult,
    ApprovalDecisionCallResult,
    PendingApprovalsCallResult,
    SlackAlertCallResult,
)
from agentflow_frontend.api_models import (
    AgentQueryRequest,
    AgentQueryResponse,
    PendingReleaseRunApprovalList,
    ReleaseApprovalDecisionStatus,
    ReleaseRunApproval,
    ReleaseRunApprovalDecisionRequest,
    SlackReleaseAlertResult,
)
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

    app = AppTest.from_file("streamlit_app.py")
    app.run()
    app.text_input[0].input("signed-test-jwt")
    app.button[1].click().run()

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
    ("decision_button_index", "expected_status"),
    [
        (2, ReleaseApprovalDecisionStatus.APPROVED),
        (3, ReleaseApprovalDecisionStatus.REJECTED),
    ],
)
def test_streamlit_app_submits_manager_approval_decision(
    monkeypatch: pytest.MonkeyPatch,
    decision_button_index: int,
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

    app = AppTest.from_file("streamlit_app.py")
    app.run()
    app.text_input[0].input("signed-test-jwt")
    app.button[1].click().run()

    app.text_area[1].input("Evidence reviewed by manager.")
    app.button[decision_button_index].click().run()

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

    app = AppTest.from_file("streamlit_app.py")
    app.run()
    app.text_input[0].input("signed-notify-jwt")
    app.button[1].click().run()
    app.text_area[1].input("Approved for Slack notification.")

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
