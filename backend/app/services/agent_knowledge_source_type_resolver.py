"""Resolve explicit engineering-document categories from user queries."""

from __future__ import annotations

from typing import Final

from app.models.engineering_document import EngineeringDocumentSourceType

_SOURCE_TYPE_PHRASES: Final[tuple[tuple[EngineeringDocumentSourceType, tuple[str, ...]], ...]] = (
    (
        EngineeringDocumentSourceType.RELEASE_CHECKLIST,
        (
            "release readiness checklist",
            "release-readiness checklist",
            "release checklist",
        ),
    ),
    (
        EngineeringDocumentSourceType.INCIDENT_POSTMORTEM,
        (
            "incident postmortem",
            "incident post-mortem",
            "postmortem",
            "post-mortem",
        ),
    ),
    (
        EngineeringDocumentSourceType.ARCHITECTURE_DOC,
        (
            "architecture document",
            "architecture doc",
            "system design document",
        ),
    ),
    (
        EngineeringDocumentSourceType.RUNBOOK,
        ("runbook",),
    ),
)


def resolve_engineering_document_source_type(
    query: str,
) -> EngineeringDocumentSourceType | None:
    """Resolve one explicitly requested document category.

    Ambiguous queries mentioning more than one category intentionally return
    no filter so retrieval can search all relevant engineering documents.

    Args:
        query: Validated natural-language Knowledge Agent question.

    Returns:
        One explicit engineering-document category, or ``None`` when the
        category is absent or ambiguous.
    """

    normalized_query = " ".join(query.casefold().split())
    matches = {
        source_type
        for source_type, phrases in _SOURCE_TYPE_PHRASES
        if any(phrase in normalized_query for phrase in phrases)
    }

    if len(matches) != 1:
        return None

    return next(iter(matches))
