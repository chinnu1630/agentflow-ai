"""Shared release-risk enumeration schemas."""

from enum import StrEnum


class RiskSeverityResponse(StrEnum):
    """API severity level for a detected risk."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskSummaryActionResponse(StrEnum):
    """API recommended action from deterministic risk summary."""

    PROCEED = "proceed"
    REVIEW_REQUIRED = "review_required"
    BLOCK_RELEASE = "block_release"
    PARTIAL_DATA_REVIEW = "partial_data_review"
