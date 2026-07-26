"""Tests for AgentFlow application configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_settings_use_default_knowledge_model_configuration() -> None:
    """Settings should provide safe local model defaults for hybrid retrieval."""
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert (
        settings.knowledge_embedding_model_name
        == "sentence-transformers/all-MiniLM-L6-v2"
    )
    assert settings.knowledge_embedding_dimension == 384
    assert (
        settings.knowledge_reranker_model_name
        == "cross-encoder/ms-marco-MiniLM-L6-v2"
    )


def test_settings_allow_knowledge_model_name_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model names may change when they preserve the fixed vector dimension."""
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_MODEL_NAME", "local/embedding-model")
    monkeypatch.setenv("KNOWLEDGE_RERANKER_MODEL_NAME", "local/reranker-model")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.knowledge_embedding_model_name == "local/embedding-model"
    assert settings.knowledge_embedding_dimension == 384
    assert settings.knowledge_reranker_model_name == "local/reranker-model"


def test_settings_reject_embedding_dimension_schema_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configuration must reject dimensions incompatible with vector(384)."""
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_DIMENSION", "768")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_settings_use_safe_default_anthropic_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claude synthesis should be disabled unless explicitly configured."""
    monkeypatch.delenv("ANTHROPIC_ENABLED", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.anthropic_enabled is False
    assert settings.agent_dynamic_planning_enabled is False
    assert settings.anthropic_api_key is None
    assert settings.anthropic_model == "claude-sonnet-5"
    assert settings.anthropic_max_tokens == 4_096
    assert settings.agent_dynamic_planner_model is None
    assert settings.agent_dynamic_planner_max_tokens == 1_024
    assert settings.agent_dynamic_synthesis_model is None
    assert settings.agent_dynamic_synthesis_max_tokens == 2_048
    assert (
        settings.agent_dynamic_planner_input_cost_per_million_usd
        == 0
    )
    assert (
        settings.agent_dynamic_planner_output_cost_per_million_usd
        == 0
    )
    assert (
        settings.agent_dynamic_synthesis_input_cost_per_million_usd
        == 0
    )
    assert (
        settings.agent_dynamic_synthesis_output_cost_per_million_usd
        == 0
    )
    assert settings.agent_dynamic_max_estimated_cost_usd is None
    assert settings.anthropic_timeout_seconds == 30.0
    assert settings.anthropic_max_retries == 2


def test_settings_allow_anthropic_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claude configuration should load securely from environment variables."""
    monkeypatch.setenv("ANTHROPIC_ENABLED", "true")
    monkeypatch.setenv("AGENT_DYNAMIC_PLANNING_ENABLED", "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-secret-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "test-claude-model")
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "2048")
    monkeypatch.setenv(
        "AGENT_DYNAMIC_PLANNER_MODEL",
        "test-planner-model",
    )
    monkeypatch.setenv("AGENT_DYNAMIC_PLANNER_MAX_TOKENS", "768")
    monkeypatch.setenv(
        "AGENT_DYNAMIC_SYNTHESIS_MODEL",
        "test-synthesis-model",
    )
    monkeypatch.setenv("AGENT_DYNAMIC_SYNTHESIS_MAX_TOKENS", "3072")
    monkeypatch.setenv(
        "AGENT_DYNAMIC_PLANNER_INPUT_COST_PER_MILLION_USD",
        "3.25",
    )
    monkeypatch.setenv(
        "AGENT_DYNAMIC_PLANNER_OUTPUT_COST_PER_MILLION_USD",
        "15.50",
    )
    monkeypatch.setenv(
        "AGENT_DYNAMIC_SYNTHESIS_INPUT_COST_PER_MILLION_USD",
        "4.00",
    )
    monkeypatch.setenv(
        "AGENT_DYNAMIC_SYNTHESIS_OUTPUT_COST_PER_MILLION_USD",
        "20.00",
    )
    monkeypatch.setenv(
        "AGENT_DYNAMIC_MAX_ESTIMATED_COST_USD",
        "0.25",
    )
    monkeypatch.setenv("ANTHROPIC_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("ANTHROPIC_MAX_RETRIES", "3")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.anthropic_enabled is True
    assert settings.agent_dynamic_planning_enabled is True
    assert settings.anthropic_api_key is not None
    assert (
        settings.anthropic_api_key.get_secret_value()
        == "test-secret-key"
    )
    assert settings.anthropic_model == "test-claude-model"
    assert settings.anthropic_max_tokens == 2_048
    assert settings.agent_dynamic_planner_model == "test-planner-model"
    assert settings.agent_dynamic_planner_max_tokens == 768
    assert settings.agent_dynamic_synthesis_model == "test-synthesis-model"
    assert settings.agent_dynamic_synthesis_max_tokens == 3_072
    assert (
        settings.agent_dynamic_planner_input_cost_per_million_usd
        == 3.25
    )
    assert (
        settings.agent_dynamic_planner_output_cost_per_million_usd
        == 15.50
    )
    assert (
        settings.agent_dynamic_synthesis_input_cost_per_million_usd
        == 4
    )
    assert (
        settings.agent_dynamic_synthesis_output_cost_per_million_usd
        == 20
    )
    assert settings.agent_dynamic_max_estimated_cost_usd == 0.25
    assert settings.anthropic_timeout_seconds == 45.0
    assert settings.anthropic_max_retries == 3


@pytest.mark.parametrize(
    ("environment_name", "invalid_value"),
    [
        ("ANTHROPIC_MAX_TOKENS", "100"),
        ("AGENT_DYNAMIC_PLANNER_MAX_TOKENS", "100"),
        ("AGENT_DYNAMIC_SYNTHESIS_MAX_TOKENS", "9000"),
        (
            "AGENT_DYNAMIC_PLANNER_INPUT_COST_PER_MILLION_USD",
            "-1",
        ),
        (
            "AGENT_DYNAMIC_SYNTHESIS_OUTPUT_COST_PER_MILLION_USD",
            "-0.01",
        ),
        ("AGENT_DYNAMIC_MAX_ESTIMATED_COST_USD", "0"),
        ("ANTHROPIC_TIMEOUT_SECONDS", "0"),
        ("ANTHROPIC_MAX_RETRIES", "6"),
    ],
)
def test_settings_reject_invalid_anthropic_limits(
    monkeypatch: pytest.MonkeyPatch,
    environment_name: str,
    invalid_value: str,
) -> None:
    """Claude request limits must remain inside configured safety bounds."""
    monkeypatch.setenv(environment_name, invalid_value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]

def test_settings_use_safe_default_authentication_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local development should keep authentication explicitly disabled."""
    monkeypatch.delenv("AUTH_ENABLED", raising=False)
    monkeypatch.delenv("AUTH_JWT_ISSUER", raising=False)
    monkeypatch.delenv("AUTH_JWT_AUDIENCE", raising=False)
    monkeypatch.delenv("AUTH_JWT_PUBLIC_KEY", raising=False)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.auth_enabled is False
    assert settings.auth_jwt_algorithm == "RS256"
    assert settings.auth_jwt_issuer is None
    assert settings.auth_jwt_audience is None
    assert settings.auth_jwt_public_key is None


def test_settings_allow_complete_authentication_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authentication settings should load from environment variables."""
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("AUTH_JWT_ISSUER", "https://identity.example.com/")
    monkeypatch.setenv("AUTH_JWT_AUDIENCE", "agentflow-api")
    monkeypatch.setenv(
        "AUTH_JWT_PUBLIC_KEY",
        "-----BEGIN PUBLIC KEY-----\ntest-key\n-----END PUBLIC KEY-----",
    )

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.auth_enabled is True
    assert settings.auth_jwt_algorithm == "RS256"
    assert settings.auth_jwt_issuer == "https://identity.example.com/"
    assert settings.auth_jwt_audience == "agentflow-api"
    assert settings.auth_jwt_public_key is not None
    assert "test-key" in settings.auth_jwt_public_key.get_secret_value()


@pytest.mark.parametrize(
    "missing_environment_variable",
    [
        "AUTH_JWT_ISSUER",
        "AUTH_JWT_AUDIENCE",
        "AUTH_JWT_PUBLIC_KEY",
    ],
)
def test_settings_reject_incomplete_enabled_authentication(
    monkeypatch: pytest.MonkeyPatch,
    missing_environment_variable: str,
) -> None:
    """Enabled authentication must include every JWT verification setting."""
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("AUTH_JWT_ISSUER", "https://identity.example.com/")
    monkeypatch.setenv("AUTH_JWT_AUDIENCE", "agentflow-api")
    monkeypatch.setenv(
        "AUTH_JWT_PUBLIC_KEY",
        "-----BEGIN PUBLIC KEY-----\ntest-key\n-----END PUBLIC KEY-----",
    )
    monkeypatch.delenv(missing_environment_variable, raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.mark.parametrize("environment", ["staging", "production", "prod"])
def test_settings_reject_disabled_authentication_outside_local_environments(
    monkeypatch: pytest.MonkeyPatch,
    environment: str,
) -> None:
    """Deployed environments must fail closed when authentication is disabled."""
    monkeypatch.setenv("ENVIRONMENT", environment)
    monkeypatch.setenv("AUTH_ENABLED", "false")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_settings_use_safe_default_http_boundary_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP boundary defaults should deny CORS and bound request bodies."""
    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("TRUSTED_HOSTS", raising=False)
    monkeypatch.delenv("MAX_REQUEST_BODY_BYTES", raising=False)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.cors_allowed_origins == ()
    assert settings.trusted_hosts == (
        "localhost",
        "127.0.0.1",
        "testserver",
    )
    assert settings.max_request_body_bytes == 1_048_576


def test_settings_allow_http_boundary_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deployment-specific origins, hosts, and size limits should be configurable."""
    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGINS",
        '["https://agentflow.example.com"]',
    )
    monkeypatch.setenv(
        "TRUSTED_HOSTS",
        '["api.agentflow.example.com","*.internal.example.com"]',
    )
    monkeypatch.setenv("MAX_REQUEST_BODY_BYTES", "2097152")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.cors_allowed_origins == (
        "https://agentflow.example.com",
    )
    assert settings.trusted_hosts == (
        "api.agentflow.example.com",
        "*.internal.example.com",
    )
    assert settings.max_request_body_bytes == 2_097_152


@pytest.mark.parametrize(
    ("environment_variable", "invalid_value"),
    [
        ("CORS_ALLOWED_ORIGINS", '["*"]'),
        ("TRUSTED_HOSTS", '["*"]'),
        ("MAX_REQUEST_BODY_BYTES", "0"),
        ("MAX_REQUEST_BODY_BYTES", "104857601"),
    ],
)
def test_settings_reject_unsafe_http_boundary_configuration(
    monkeypatch: pytest.MonkeyPatch,
    environment_variable: str,
    invalid_value: str,
) -> None:
    """Unsafe wildcard and unbounded HTTP settings must fail validation."""
    monkeypatch.setenv(environment_variable, invalid_value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.mark.parametrize("environment", ["staging", "production", "prod"])
def test_settings_reject_local_only_trusted_hosts_in_deployed_environments(
    monkeypatch: pytest.MonkeyPatch,
    environment: str,
) -> None:
    """Deployed environments must explicitly configure trusted API hosts."""
    monkeypatch.setenv("ENVIRONMENT", environment)
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "AUTH_JWT_ISSUER",
        "https://identity.example.com/",
    )
    monkeypatch.setenv("AUTH_JWT_AUDIENCE", "agentflow-api")
    monkeypatch.setenv(
        "AUTH_JWT_PUBLIC_KEY",
        "-----BEGIN PUBLIC KEY-----\ntest-key\n-----END PUBLIC KEY-----",
    )
    monkeypatch.delenv("TRUSTED_HOSTS", raising=False)

    with pytest.raises(
        ValidationError,
        match="Deployed environments require explicit trusted hosts",
    ):
        Settings(_env_file=None)  # type: ignore[call-arg]

def test_settings_use_safe_default_rate_limit_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local development should keep Redis rate limiting disabled by default."""
    environment_variables = (
        "RATE_LIMIT_ENABLED",
        "REDIS_URL",
        "REDIS_MAX_CONNECTIONS",
        "REDIS_CONNECT_TIMEOUT_SECONDS",
        "REDIS_SOCKET_TIMEOUT_SECONDS",
        "RATE_LIMIT_KEY_HMAC_SECRET",
        "RATE_LIMIT_STANDARD_CAPACITY",
        "RATE_LIMIT_STANDARD_REFILL_RATE_PER_SECOND",
        "RATE_LIMIT_EXPENSIVE_CAPACITY",
        "RATE_LIMIT_EXPENSIVE_REFILL_RATE_PER_SECOND",
    )

    for environment_variable in environment_variables:
        monkeypatch.delenv(environment_variable, raising=False)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.rate_limit_enabled is False
    assert settings.redis_url is None
    assert settings.redis_max_connections == 20
    assert settings.redis_connect_timeout_seconds == 1.0
    assert settings.redis_socket_timeout_seconds == 1.0
    assert settings.rate_limit_key_hmac_secret is None
    assert settings.rate_limit_standard_capacity == 60
    assert settings.rate_limit_standard_refill_rate_per_second == 1.0
    assert settings.rate_limit_expensive_capacity == 5
    assert settings.rate_limit_expensive_refill_rate_per_second == 0.1


def test_settings_allow_rate_limit_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Redis and token-bucket limits should load from environment variables."""
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/1")
    monkeypatch.setenv("REDIS_MAX_CONNECTIONS", "40")
    monkeypatch.setenv("REDIS_CONNECT_TIMEOUT_SECONDS", "2")
    monkeypatch.setenv("REDIS_SOCKET_TIMEOUT_SECONDS", "3")
    monkeypatch.setenv(
        "RATE_LIMIT_KEY_HMAC_SECRET",
        "test-rate-limit-hmac-secret",
    )
    monkeypatch.setenv("RATE_LIMIT_STANDARD_CAPACITY", "120")
    monkeypatch.setenv(
        "RATE_LIMIT_STANDARD_REFILL_RATE_PER_SECOND",
        "2.5",
    )
    monkeypatch.setenv("RATE_LIMIT_EXPENSIVE_CAPACITY", "8")
    monkeypatch.setenv(
        "RATE_LIMIT_EXPENSIVE_REFILL_RATE_PER_SECOND",
        "0.25",
    )

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.rate_limit_enabled is True
    assert settings.redis_url is not None
    assert (
        settings.redis_url.get_secret_value()
        == "redis://localhost:6379/1"
    )
    assert settings.redis_max_connections == 40
    assert settings.redis_connect_timeout_seconds == 2.0
    assert settings.redis_socket_timeout_seconds == 3.0
    assert settings.rate_limit_key_hmac_secret is not None
    assert (
        settings.rate_limit_key_hmac_secret.get_secret_value()
        == "test-rate-limit-hmac-secret"
    )
    assert settings.rate_limit_standard_capacity == 120
    assert settings.rate_limit_standard_refill_rate_per_second == 2.5
    assert settings.rate_limit_expensive_capacity == 8
    assert settings.rate_limit_expensive_refill_rate_per_second == 0.25


@pytest.mark.parametrize(
    ("environment_variable", "invalid_value"),
    [
        ("REDIS_MAX_CONNECTIONS", "0"),
        ("REDIS_CONNECT_TIMEOUT_SECONDS", "0"),
        ("REDIS_SOCKET_TIMEOUT_SECONDS", "0"),
        ("RATE_LIMIT_STANDARD_CAPACITY", "0"),
        ("RATE_LIMIT_STANDARD_REFILL_RATE_PER_SECOND", "0"),
        ("RATE_LIMIT_EXPENSIVE_CAPACITY", "0"),
        ("RATE_LIMIT_EXPENSIVE_REFILL_RATE_PER_SECOND", "0"),
    ],
)
def test_settings_reject_invalid_rate_limit_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    environment_variable: str,
    invalid_value: str,
) -> None:
    """Redis pool and token-bucket values must remain strictly positive."""
    monkeypatch.setenv(environment_variable, invalid_value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "invalid_redis_url",
    [
        "http://localhost:6379",
        "redis://",
        "not-a-url",
    ],
)
def test_settings_reject_invalid_redis_urls(
    monkeypatch: pytest.MonkeyPatch,
    invalid_redis_url: str,
) -> None:
    """Redis configuration should accept only usable redis or rediss URLs."""
    monkeypatch.setenv("REDIS_URL", invalid_redis_url)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "missing_environment_variable",
    [
        "REDIS_URL",
        "RATE_LIMIT_KEY_HMAC_SECRET",
    ],
)
def test_settings_reject_incomplete_deployed_rate_limit_configuration(
    monkeypatch: pytest.MonkeyPatch,
    missing_environment_variable: str,
) -> None:
    """Deployed rate limiting requires Redis and a key-derivation secret."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("TRUSTED_HOSTS", '["api.agentflow.example.com"]')
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "AUTH_JWT_ISSUER",
        "https://identity.example.com/",
    )
    monkeypatch.setenv("AUTH_JWT_AUDIENCE", "agentflow-api")
    monkeypatch.setenv(
        "AUTH_JWT_PUBLIC_KEY",
        "-----BEGIN PUBLIC KEY-----\ntest-key\n-----END PUBLIC KEY-----",
    )
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379/0")
    monkeypatch.setenv(
        "RATE_LIMIT_KEY_HMAC_SECRET",
        "production-rate-limit-hmac-secret",
    )
    monkeypatch.delenv(missing_environment_variable, raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_settings_reject_disabled_rate_limiting_in_deployed_environments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deployed environments must enable distributed API rate limiting."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("TRUSTED_HOSTS", '["api.agentflow.example.com"]')
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "AUTH_JWT_ISSUER",
        "https://identity.example.com/",
    )
    monkeypatch.setenv("AUTH_JWT_AUDIENCE", "agentflow-api")
    monkeypatch.setenv(
        "AUTH_JWT_PUBLIC_KEY",
        "-----BEGIN PUBLIC KEY-----\ntest-key\n-----END PUBLIC KEY-----",
    )
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]
