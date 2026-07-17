"""Local cross-encoder reranker for Knowledge Agent retrieval."""

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


class CrossEncoderModelProtocol(Protocol):
    """Protocol for cross-encoder models used by the reranker."""

    def predict(
        self,
        sentences: Sequence[tuple[str, str]],
        *,
        show_progress_bar: bool,
    ) -> object:
        """Score ordered query-document pairs."""
        ...


@runtime_checkable
class SupportsToList(Protocol):
    """Protocol for NumPy-like values supporting conversion to lists."""

    def tolist(self) -> object:
        """Return the value as native Python objects."""
        ...


CrossEncoderModelFactory = Callable[[str], CrossEncoderModelProtocol]


class EngineeringDocumentRerankerError(RuntimeError):
    """Raised when local cross-encoder reranking fails."""


class CrossEncoderEngineeringDocumentReranker:
    """Score candidate chunks using a locally executed cross-encoder model."""

    def __init__(
        self,
        *,
        model_name: str,
        model_factory: CrossEncoderModelFactory | None = None,
    ) -> None:
        """Initialize the reranker.

        Args:
            model_name: Cross-encoder model identifier or local model path.
            model_factory: Optional model factory used by unit tests.

        Raises:
            ValueError: If the model name is blank.
        """
        if not model_name.strip():
            raise ValueError("model_name must contain non-whitespace text")

        self._model_name = model_name
        self._model_factory = model_factory or self._default_model_factory
        self._model: CrossEncoderModelProtocol | None = None
        self._model_lock = asyncio.Lock()

    async def score_candidates(
        self,
        *,
        query: str,
        candidate_contents: Sequence[str],
        run_id: str | None = None,
    ) -> list[float]:
        """Return one relevance score for each candidate chunk.

        CPU-bound model loading and inference run in worker threads so FastAPI's
        asyncio event loop remains responsive.

        Args:
            query: Natural-language retrieval query.
            candidate_contents: Ordered candidate chunk contents.
            run_id: Optional workflow identifier for structured logs.

        Returns:
            Scores in the same order as ``candidate_contents``.

        Raises:
            ValueError: If the query or any candidate is blank.
            EngineeringDocumentRerankerError: If model execution or output
                validation fails.
        """
        if not query.strip():
            raise ValueError("query must contain non-whitespace text")

        if not candidate_contents:
            return []

        if any(not content.strip() for content in candidate_contents):
            raise ValueError("candidate contents must not contain blank values")

        started_at = time.perf_counter()
        pairs = [(query, content) for content in candidate_contents]

        try:
            model = await self._get_model()
            raw_scores = await asyncio.to_thread(
                model.predict,
                pairs,
                show_progress_bar=False,
            )
            scores = self._normalize_scores(
                raw_scores,
                expected_count=len(candidate_contents),
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.exception(
                "engineering_document_reranking_failed",
                run_id=run_id,
                model_name=self._model_name,
                candidate_count=len(candidate_contents),
                error_type=exc.__class__.__name__,
            )
            raise EngineeringDocumentRerankerError(
                "Engineering document candidates could not be reranked."
            ) from exc

        logger.info(
            "engineering_document_reranking_completed",
            run_id=run_id,
            model_name=self._model_name,
            candidate_count=len(candidate_contents),
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
        )

        return scores

    async def _get_model(self) -> CrossEncoderModelProtocol:
        """Load the cross-encoder model once and reuse it across requests."""
        if self._model is not None:
            return self._model

        async with self._model_lock:
            if self._model is None:
                self._model = await asyncio.to_thread(
                    self._model_factory,
                    self._model_name,
                )

        return self._model

    @staticmethod
    def _normalize_scores(
        raw_scores: object,
        *,
        expected_count: int,
    ) -> list[float]:
        """Convert model output into a validated list of float scores."""
        output = (
            raw_scores.tolist()
            if isinstance(raw_scores, SupportsToList)
            else raw_scores
        )

        if not isinstance(output, (list, tuple)):
            raise TypeError("reranker model returned an unsupported output type")

        scores = [float(score) for score in output]

        if len(scores) != expected_count:
            raise ValueError("reranker returned an unexpected number of scores")

        return scores

    @staticmethod
    def _default_model_factory(model_name: str) -> CrossEncoderModelProtocol:
        """Create the production cross-encoder model lazily."""
        module = importlib.import_module("sentence_transformers")
        model_class = getattr(module, "CrossEncoder", None)

        if model_class is None or not callable(model_class):
            raise RuntimeError("sentence_transformers does not expose CrossEncoder")

        return cast(CrossEncoderModelProtocol, model_class(model_name))


@lru_cache(maxsize=1)
def get_engineering_document_reranker(
) -> CrossEncoderEngineeringDocumentReranker:
    """Return the shared production cross-encoder reranker."""
    settings = get_settings()

    return CrossEncoderEngineeringDocumentReranker(
        model_name=settings.knowledge_reranker_model_name,
    )
