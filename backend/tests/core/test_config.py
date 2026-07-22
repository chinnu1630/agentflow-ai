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
