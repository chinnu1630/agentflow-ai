"""Tests for AgentFlow application configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_settings_use_default_knowledge_model_configuration() -> None:
    """Settings should provide safe local model defaults for hybrid retrieval."""
    settings = Settings(_env_file=None)

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

    settings = Settings(_env_file=None)

    assert settings.knowledge_embedding_model_name == "local/embedding-model"
    assert settings.knowledge_embedding_dimension == 384
    assert settings.knowledge_reranker_model_name == "local/reranker-model"


def test_settings_reject_embedding_dimension_schema_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configuration must reject dimensions incompatible with vector(384)."""
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_DIMENSION", "768")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
