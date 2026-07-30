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


@pytest.mark.parametrize(
    "query",
    [
        "What recovery steps are documented for payment API timeouts?",
        "What monitoring checks should be completed after a rollback?",
    ],
)
def test_infers_runbook_for_bounded_operational_guidance(
    query: str,
) -> None:
    """Operational recovery guidance should search only trusted runbooks."""

    assert (
        resolve_engineering_document_source_type(query)
        is EngineeringDocumentSourceType.RUNBOOK
    )


@pytest.mark.parametrize(
    "query",
    [
        "Could a rollback affect this release?",
        (
            "Compare documented rollback recovery steps with the "
            "release readiness checklist."
        ),
    ],
)
def test_does_not_overfilter_ambiguous_operational_queries(
    query: str,
) -> None:
    """Generic or multi-category operational wording must remain unfiltered."""

    assert resolve_engineering_document_source_type(query) is None
