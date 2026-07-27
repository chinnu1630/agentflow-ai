"""Tests for frontend runtime configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentflow_frontend.config import FrontendSettings


def test_settings_require_backend_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject startup when no backend URL is configured."""
    monkeypatch.delenv("AGENTFLOW_FRONTEND_BACKEND_BASE_URL", raising=False)

    with pytest.raises(ValidationError):
        FrontendSettings(_env_file=None)


def test_authentication_is_required_by_default() -> None:
    """Keep the frontend secure when no auth setting is supplied."""
    settings = FrontendSettings(
        backend_base_url="https://agentflow.example.test",
        _env_file=None,
    )

    assert settings.auth_required is True


def test_authentication_can_be_disabled_explicitly_for_local_development(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow an explicit local-only configuration to omit bearer tokens."""
    monkeypatch.setenv("AGENTFLOW_FRONTEND_AUTH_REQUIRED", "false")

    settings = FrontendSettings(
        backend_base_url="http://127.0.0.1:8000",
        _env_file=None,
    )

    assert settings.auth_required is False


def test_build_api_url_joins_backend_and_api_path() -> None:
    """Build the versioned AgentFlow endpoint without duplicate slashes."""
    settings = FrontendSettings(
        backend_base_url="http://backend.internal:8000/",
        _env_file=None,
    )

    assert (
        settings.build_api_url("/api/v1/agent/query")
        == "http://backend.internal:8000/api/v1/agent/query"
    )


@pytest.mark.parametrize(
    "path",
    [
        "",
        "api/v1/agent/query",
        "//malicious.example/query",
        "https://malicious.example/query",
    ],
)
def test_build_api_url_rejects_untrusted_paths(path: str) -> None:
    """Prevent callers from replacing the configured backend host."""
    settings = FrontendSettings(
        backend_base_url="http://backend.internal:8000",
        _env_file=None,
    )

    with pytest.raises(ValueError, match="without a hostname"):
        settings.build_api_url(path)
