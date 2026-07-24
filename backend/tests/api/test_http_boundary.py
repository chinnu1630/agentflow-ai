"""HTTP boundary security contracts for the AgentFlow FastAPI application."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from app.core.config import get_settings
from app.main import create_app
from app.middleware.request_body_limit import RequestBodyLimitMiddleware


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Iterator[None]:
    """Prevent environment-backed settings from leaking between tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _build_test_app(
    monkeypatch: pytest.MonkeyPatch,
    *,
    allowed_origins: str = "[]",
    trusted_hosts: str = '["testserver"]',
    max_request_body_bytes: str = "1048576",
) -> FastAPI:
    """Build one application with deterministic HTTP boundary settings."""
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", allowed_origins)
    monkeypatch.setenv("TRUSTED_HOSTS", trusted_hosts)
    monkeypatch.setenv(
        "MAX_REQUEST_BODY_BYTES",
        max_request_body_bytes,
    )
    get_settings.cache_clear()

    return create_app()


def test_trusted_host_middleware_rejects_unapproved_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requests with an untrusted Host header must fail before routing."""
    test_app = _build_test_app(monkeypatch)
    client = TestClient(test_app)

    allowed_response = client.get("/api/v1/health")
    rejected_response = client.get(
        "/api/v1/health",
        headers={"Host": "attacker.example.com"},
    )

    assert allowed_response.status_code == 200
    assert rejected_response.status_code == 400
    assert rejected_response.text == "Invalid host header"


def test_cors_allows_only_configured_browser_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only explicitly configured browser origins may read API responses."""
    test_app = _build_test_app(
        monkeypatch,
        allowed_origins='["https://ui.agentflow.example.com"]',
    )
    client = TestClient(test_app)

    allowed_response = client.get(
        "/api/v1/health",
        headers={"Origin": "https://ui.agentflow.example.com"},
    )
    rejected_response = client.get(
        "/api/v1/health",
        headers={"Origin": "https://attacker.example.com"},
    )

    assert (
        allowed_response.headers["access-control-allow-origin"]
        == "https://ui.agentflow.example.com"
    )
    assert "access-control-allow-origin" not in rejected_response.headers


def test_security_headers_are_added_to_api_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every API response should include browser hardening headers."""
    response = TestClient(
        _build_test_app(monkeypatch)
    ).get("/api/v1/health")

    assert response.status_code == 200
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert (
        response.headers["content-security-policy"]
        == "default-src 'none'; frame-ancestors 'none'"
    )
    assert (
        response.headers["permissions-policy"]
        == "camera=(), microphone=(), geolocation=()"
    )


def test_request_body_limit_rejects_oversized_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Oversized bodies must fail before validation or business execution."""
    response = TestClient(
        _build_test_app(
            monkeypatch,
            max_request_body_bytes="16",
        )
    ).post(
        "/api/v1/agent/query-plan",
        content=b"x" * 17,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"


def test_request_body_limit_allows_payload_at_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A body at the configured byte limit must reach normal routing."""
    response = TestClient(
        _build_test_app(
            monkeypatch,
            max_request_body_bytes="16",
        )
    ).post(
        "/api/v1/health",
        content=b"x" * 16,
        headers={"Content-Type": "application/octet-stream"},
    )

    assert response.status_code == 405



@pytest.mark.anyio
async def test_request_body_limit_counts_streamed_body_without_content_length(
) -> None:
    """Chunked bodies must be rejected when their accumulated size is too large."""
    received_messages: list[Message] = [
        {
            "type": "http.request",
            "body": b"x" * 10,
            "more_body": True,
        },
        {
            "type": "http.request",
            "body": b"y" * 7,
            "more_body": False,
        },
    ]
    sent_messages: list[Message] = []

    scope: Scope = {
        "type": "http",
        "asgi": {
            "version": "3.0",
            "spec_version": "2.3",
        },
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/stream",
        "raw_path": b"/stream",
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "state": {"run_id": "stream-body-test"},
    }

    async def receive() -> Message:
        """Return the next simulated request-body chunk."""
        return received_messages.pop(0)

    async def send(message: Message) -> None:
        """Capture middleware response messages."""
        sent_messages.append(message)

    async def downstream_app(
        app_scope: Scope,
        app_receive: Receive,
        app_send: Send,
    ) -> None:
        """Consume the full body before starting a normal response."""
        del app_scope

        while True:
            message = await app_receive()

            if (
                message["type"] == "http.request"
                and not message.get("more_body", False)
            ):
                break

        await app_send(
            {
                "type": "http.response.start",
                "status": 204,
                "headers": [],
            }
        )
        await app_send(
            {
                "type": "http.response.body",
                "body": b"",
            }
        )

    middleware = RequestBodyLimitMiddleware(
        downstream_app,
        max_body_bytes=16,
    )

    await middleware(scope, receive, send)

    assert sent_messages[0]["type"] == "http.response.start"
    assert sent_messages[0]["status"] == 413
    assert sent_messages[1]["type"] == "http.response.body"
    assert b"REQUEST_TOO_LARGE" in sent_messages[1]["body"]
