"""Tests for local engineering-document embedding generation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from app.services.engineering_document_embedding_provider import (
    SentenceTransformerEmbeddingProvider,
)


class FakeSentenceTransformer:
    """Small fake model that avoids network and model downloads in tests."""

    def __init__(self) -> None:
        self.encode_calls = 0

    def encode(
        self,
        sentences: Sequence[str],
        **kwargs: Any,
    ) -> list[list[float]]:
        """Return deterministic three-dimensional embeddings."""
        self.encode_calls += 1
        assert kwargs["normalize_embeddings"] is True

        return [
            [float(index + 1), float(len(sentence)), 1.0]
            for index, sentence in enumerate(sentences)
        ]


@pytest.mark.asyncio
async def test_embedding_provider_generates_embeddings_and_reuses_model() -> None:
    """Provider should encode asynchronously and lazily load its model once."""
    fake_model = FakeSentenceTransformer()
    factory_calls = 0

    def model_factory(model_name: str) -> FakeSentenceTransformer:
        nonlocal factory_calls
        factory_calls += 1
        assert model_name == "test-embedding-model"
        return fake_model

    provider = SentenceTransformerEmbeddingProvider(
        model_name="test-embedding-model",
        embedding_dimension=3,
        model_factory=model_factory,
    )

    first_embeddings = await provider.embed_texts(
        ["payment failure", "rollback procedure"],
        run_id="test-run-id",
    )
    second_embeddings = await provider.embed_texts(
        ["release checklist"],
        run_id="test-run-id",
    )

    assert first_embeddings == [
        [1.0, 15.0, 1.0],
        [2.0, 18.0, 1.0],
    ]
    assert second_embeddings == [[1.0, 17.0, 1.0]]
    assert factory_calls == 1
    assert fake_model.encode_calls == 2


def test_get_embedding_provider_returns_cached_instance(monkeypatch) -> None:
    """Production factory should reuse one provider instead of reloading models."""
    from app.core.config import get_settings
    from app.services.engineering_document_embedding_provider import (
        get_engineering_document_embedding_provider,
    )

    monkeypatch.setenv(
        "KNOWLEDGE_EMBEDDING_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_DIMENSION", "384")

    get_settings.cache_clear()
    get_engineering_document_embedding_provider.cache_clear()

    first_provider = get_engineering_document_embedding_provider()
    second_provider = get_engineering_document_embedding_provider()

    assert first_provider is second_provider

    get_engineering_document_embedding_provider.cache_clear()
    get_settings.cache_clear()
