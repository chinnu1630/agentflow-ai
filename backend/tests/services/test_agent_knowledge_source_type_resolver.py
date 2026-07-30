"""Tests for engineering-document source-type resolution."""

from __future__ import annotations

import pytest

from app.models.engineering_document import EngineeringDocumentSourceType
from app.services.agent_knowledge_source_type_resolver import (
    resolve_engineering_document_source_type,
)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "What does the payment-service runbook say?",
            EngineeringDocumentSourceType.RUNBOOK,
        ),
        (
            "According to the release readiness checklist, what is required?",
            EngineeringDocumentSourceType.RELEASE_CHECKLIST,
        ),
        (
            "What did the incident postmortem identify?",
            EngineeringDocumentSourceType.INCIDENT_POSTMORTEM,
        ),
        (
            "What does the architecture document say?",
            EngineeringDocumentSourceType.ARCHITECTURE_DOC,
        ),
    ],
)
def test_resolves_explicit_document_category(
    query: str,
    expected: EngineeringDocumentSourceType,
) -> None:
    """Explicit document wording should produce one typed filter."""

    assert resolve_engineering_document_source_type(query) is expected


@pytest.mark.parametrize(
    "query",
    [
        "What engineering documents discuss rollback?",
        "Compare the runbook with the release checklist.",
    ],
)
def test_avoids_filter_for_generic_or_ambiguous_query(query: str) -> None:
    """Generic or multi-category questions should search all documents."""

    assert resolve_engineering_document_source_type(query) is None
