"""Tests for structured dynamic-agent synthesis contracts."""

import pytest
from pydantic import ValidationError

from app.schemas.agent_dynamic_synthesis import (
    AgentDynamicAnswer,
    AgentDynamicAnswerCitation,
)


def _build_citation() -> AgentDynamicAnswerCitation:
    """Create one reusable trusted dynamic-answer citation."""
    return AgentDynamicAnswerCitation(
        source_type="engineering_document_chunk",
        source_id="chunk-123",
        title="Payment Service Runbook",
        source_url="docs/payment-runbook.md",
        supporting_fact="The runbook requires rollback after threshold breach.",
    )


def test_accepts_evidence_grounded_dynamic_answer() -> None:
    """A grounded manager answer should satisfy the strict contract."""
    answer = AgentDynamicAnswer(
        answer=(
            "Rollback the payment service after the documented threshold "
            "is exceeded, then validate recovery."
        ),
        confidence=0.94,
        citations=[_build_citation()],
        requires_human_review=False,
    )

    assert answer.schema_version == "agent_dynamic_answer_v1"
    assert answer.citations[0].source_id == "chunk-123"


def test_rejects_duplicate_dynamic_answer_citations() -> None:
    """Claude must not repeat the same evidence reference."""
    with pytest.raises(
        ValidationError,
        match="citations must not contain duplicates",
    ):
        AgentDynamicAnswer(
            answer="Follow the trusted rollback procedure.",
            confidence=0.9,
            citations=[_build_citation(), _build_citation()],
            requires_human_review=False,
        )


def test_degraded_answer_requires_human_review() -> None:
    """Degraded tool execution must be visible to a human reviewer."""
    with pytest.raises(
        ValidationError,
        match="degraded dynamic answers must require human review",
    ):
        AgentDynamicAnswer(
            answer="Only partial evidence was available.",
            confidence=0.5,
            degraded_steps=["search_knowledge"],
            requires_human_review=False,
        )
