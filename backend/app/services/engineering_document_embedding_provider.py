"""Local embedding provider for AgentFlow engineering-document retrieval."""

from __future__ import annotations

import asyncio
import importlib
import time
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Protocol, cast, runtime_checkable

import structlog

from app.core.config import get_settings

logger = structlog.get_logger(__name__)


class EmbeddingModelProtocol(Protocol):
    """Protocol for sentence-embedding models used by the provider."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> object:
        """Encode text into dense vector representations."""
        ...


@runtime_checkable
class SupportsToList(Protocol):
    """Protocol for NumPy-like values that can convert themselves to lists."""

    def tolist(self) -> object:
        """Return the value as native Python objects."""
        ...


EmbeddingModelFactory = Callable[[str], EmbeddingModelProtocol]


class EngineeringDocumentEmbeddingError(RuntimeError):
    """Raised when local embedding generation fails."""


class SentenceTransformerEmbeddingProvider:
    """Generate normalized embeddings using a locally executed model.

    Model loading and inference are synchronous CPU-bound operations, so they
    run in worker threads to avoid blocking FastAPI's asyncio event loop.
    """

    def __init__(
        self,
        *,
        model_name: str,
        embedding_dimension: int,
        model_factory: EmbeddingModelFactory | None = None,
    ) -> None:
        """Initialize the embedding provider.

        Args:
            model_name: Sentence Transformer model identifier or local path.
            embedding_dimension: Expected number of values in every embedding.
            model_factory: Optional injectable model factory used by tests.

        Raises:
            ValueError: If configuration values are invalid.
        """
        if not model_name.strip():
            raise ValueError("model_name must contain non-whitespace text")

        if embedding_dimension < 1:
            raise ValueError("embedding_dimension must be greater than zero")

        self._model_name = model_name
        self._embedding_dimension = embedding_dimension
        self._model_factory = model_factory or self._default_model_factory
        self._model: EmbeddingModelProtocol | None = None
        self._model_lock = asyncio.Lock()

    async def embed_texts(
        self,
        texts: Sequence[str],
        *,
        run_id: str | None = None,
    ) -> list[list[float]]:
        """Generate normalized embeddings for a batch of texts.

        Args:
            texts: Ordered text values to embed.
            run_id: Optional workflow identifier for structured logs.

        Returns:
            Embeddings in the same order as the supplied texts.

        Raises:
            ValueError: If any supplied text is blank.
            EngineeringDocumentEmbeddingError: If model loading, inference, or
                output validation fails.
        """
        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise ValueError("texts must not contain blank values")

        started_at = time.perf_counter()

        try:
            model = await self._get_model()
            raw_embeddings = await asyncio.to_thread(
                model.encode,
                list(texts),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            embeddings = self._normalize_output(raw_embeddings)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.exception(
                "engineering_document_embedding_failed",
                run_id=run_id,
                model_name=self._model_name,
                text_count=len(texts),
                error_type=exc.__class__.__name__,
            )
            raise EngineeringDocumentEmbeddingError(
                "Engineering document embeddings could not be generated."
            ) from exc

        logger.info(
            "engineering_document_embedding_completed",
            run_id=run_id,
            model_name=self._model_name,
            text_count=len(texts),
            embedding_dimension=self._embedding_dimension,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
        )

        return embeddings

    async def _get_model(self) -> EmbeddingModelProtocol:
        """Load the embedding model once and reuse it across requests."""
        if self._model is not None:
            return self._model

        async with self._model_lock:
            if self._model is None:
                self._model = await asyncio.to_thread(
                    self._model_factory,
                    self._model_name,
                )

        return self._model

    def _normalize_output(self, raw_embeddings: object) -> list[list[float]]:
        """Convert model output to validated Python float lists."""
        output = (
            raw_embeddings.tolist()
            if isinstance(raw_embeddings, SupportsToList)
            else raw_embeddings
        )

        if not isinstance(output, list):
            raise TypeError("embedding model returned an unsupported output type")

        embeddings: list[list[float]] = []

        for raw_embedding in output:
            if not isinstance(raw_embedding, (list, tuple)):
                raise TypeError("embedding model returned an invalid vector")

            embedding = [float(value) for value in raw_embedding]

            if len(embedding) != self._embedding_dimension:
                raise ValueError(
                    "embedding dimension does not match configured dimension"
                )

            embeddings.append(embedding)

        return embeddings

    @staticmethod
    def _default_model_factory(model_name: str) -> EmbeddingModelProtocol:
        """Create the production Sentence Transformer model lazily.

        Lazy loading prevents application startup and static type checking from
        importing the complete Torch, Transformers, and NumPy dependency graph.
        """
        module = importlib.import_module("sentence_transformers")
        model_class = getattr(module, "SentenceTransformer", None)

        if model_class is None or not callable(model_class):
            raise RuntimeError(
                "sentence_transformers does not expose SentenceTransformer"
            )

        return cast(EmbeddingModelProtocol, model_class(model_name))



@lru_cache(maxsize=1)
def get_engineering_document_embedding_provider(
) -> SentenceTransformerEmbeddingProvider:
    """Return the shared production embedding provider.

    The provider and its lazily loaded model are reused across requests to avoid
    repeatedly loading model weights into memory.
    """
    settings = get_settings()

    return SentenceTransformerEmbeddingProvider(
        model_name=settings.knowledge_embedding_model_name,
        embedding_dimension=settings.knowledge_embedding_dimension,
    )
