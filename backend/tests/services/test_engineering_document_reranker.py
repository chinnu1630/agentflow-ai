"""Tests for the local engineering-document cross-encoder reranker."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from app.services.engineering_document_reranker import (
    CrossEncoderEngineeringDocumentReranker,
    EngineeringDocumentRerankerError,
)


class FakeCrossEncoderModel:
    """Return deterministic scores for supplied query-document pairs."""

    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []

    def predict(
        self,
        sentences: Sequence[tuple[str, str]],
        *,
        show_progress_bar: bool,
    ) -> list[float]:
        """Return one score per supplied pair."""
        self.calls.append(list(sentences))
        assert show_progress_bar is False
        return [float(index + 1) for index, _pair in enumerate(sentences)]


class FailingCrossEncoderModel:
    """Simulate a local model inference failure."""

    def predict(
        self,
        sentences: Sequence[tuple[str, str]],
        *,
        show_progress_bar: bool,
    ) -> object:
        """Raise an operational model error."""
        raise RuntimeError("Model inference failed.")


@pytest.mark.asyncio
async def test_score_candidates_returns_ordered_scores() -> None:
    """Reranker should preserve candidate order in returned scores."""
    model = FakeCrossEncoderModel()
    factory_calls: list[str] = []

    def model_factory(model_name: str) -> FakeCrossEncoderModel:
        factory_calls.append(model_name)
        return model

    reranker = CrossEncoderEngineeringDocumentReranker(
        model_name="test-cross-encoder",
        model_factory=model_factory,
    )

    scores = await reranker.score_candidates(
        query="Is the release safe?",
        candidate_contents=[
            "Release approval is complete.",
            "Payment rollback is not ready.",
        ],
        run_id="reranker-test",
    )

    assert scores == [1.0, 2.0]
    assert factory_calls == ["test-cross-encoder"]
    assert model.calls == [
        [
            ("Is the release safe?", "Release approval is complete."),
            ("Is the release safe?", "Payment rollback is not ready."),
        ]
    ]


@pytest.mark.asyncio
async def test_reranker_loads_model_only_once() -> None:
    """The shared reranker instance should reuse its loaded model."""
    model = FakeCrossEncoderModel()
    factory_call_count = 0

    def model_factory(model_name: str) -> FakeCrossEncoderModel:
        nonlocal factory_call_count
        factory_call_count += 1
        return model

    reranker = CrossEncoderEngineeringDocumentReranker(
        model_name="test-cross-encoder",
        model_factory=model_factory,
    )

    await reranker.score_candidates(
        query="first query",
        candidate_contents=["first candidate"],
    )
    await reranker.score_candidates(
        query="second query",
        candidate_contents=["second candidate"],
    )

    assert factory_call_count == 1


@pytest.mark.asyncio
async def test_score_candidates_returns_empty_for_no_candidates() -> None:
    """Reranker should avoid model loading when no candidates are supplied."""
    factory_called = False

    def model_factory(model_name: str) -> FakeCrossEncoderModel:
        nonlocal factory_called
        factory_called = True
        return FakeCrossEncoderModel()

    reranker = CrossEncoderEngineeringDocumentReranker(
        model_name="test-cross-encoder",
        model_factory=model_factory,
    )

    scores = await reranker.score_candidates(
        query="release safety",
        candidate_contents=[],
    )

    assert scores == []
    assert factory_called is False


@pytest.mark.asyncio
async def test_score_candidates_rejects_blank_query() -> None:
    """Reranker should reject a blank retrieval query."""
    reranker = CrossEncoderEngineeringDocumentReranker(
        model_name="test-cross-encoder",
        model_factory=lambda _name: FakeCrossEncoderModel(),
    )

    with pytest.raises(ValueError, match="query must contain"):
        await reranker.score_candidates(
            query="   ",
            candidate_contents=["candidate"],
        )


@pytest.mark.asyncio
async def test_model_failure_raises_reranker_error() -> None:
    """Operational model failures should use the domain-specific error."""
    reranker = CrossEncoderEngineeringDocumentReranker(
        model_name="test-cross-encoder",
        model_factory=lambda _name: FailingCrossEncoderModel(),
    )

    with pytest.raises(EngineeringDocumentRerankerError):
        await reranker.score_candidates(
            query="release safety",
            candidate_contents=["candidate"],
        )
