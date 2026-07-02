"""Tests for GitHub pull request risk rule evaluation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from app.schemas.github import (
    GitHubCIStatus,
    GitHubPullRequest,
    GitHubPullRequestState,
    GitHubReviewState,
)
from app.services.risk_rules import RiskCategory, RiskRuleEngine, RiskSeverity


def _enum_member(enum_type: type[Enum], *candidates: str) -> Enum:
    normalized_candidates = {candidate.lower() for candidate in candidates}

    for member in enum_type:
        if member.name.lower() in normalized_candidates:
            return member
        if str(member.value).lower() in normalized_candidates:
            return member

    raise AssertionError(
        f"Could not find enum member in {enum_type.__name__}: {candidates}"
    )


def _pull_request(**overrides: Any) -> GitHubPullRequest:
    now = datetime.now(UTC)

    defaults: dict[str, Any] = {
        "number": 42,
        "title": "PAY-123 Fix checkout validation",
        "author": "test-user",
        "state": _enum_member(GitHubPullRequestState, "open"),
        "url": "https://github.com/acme/acme-backend-services/pull/42",
        "created_at": now - timedelta(days=1),
        "updated_at": now,
        "base_branch": "main",
        "head_branch": "PAY-123-fix-checkout-validation",
        "is_draft": False,
        "changed_files": 3,
        "additions": 40,
        "deletions": 5,
        "review_state": _enum_member(GitHubReviewState, "approved"),
        "ci_status": _enum_member(GitHubCIStatus, "success", "passed"),
    }

    defaults.update(overrides)

    allowed_fields = GitHubPullRequest.model_fields
    payload = {
        field_name: value
        for field_name, value in defaults.items()
        if field_name in allowed_fields
    }

    return GitHubPullRequest(**payload)


def test_failing_ci_creates_high_risk_signal() -> None:
    pull_request = _pull_request(
        ci_status=_enum_member(GitHubCIStatus, "failure", "failed"),
    )

    result = RiskRuleEngine().evaluate_pull_request(
        pull_request,
        run_id="test-run-001",
    )

    categories = {signal.category for signal in result.signals}

    assert RiskCategory.CI_FAILURE in categories
    assert result.max_severity == RiskSeverity.HIGH
    assert result.total_score > 0.0


def test_changes_requested_creates_review_blocked_signal() -> None:
    pull_request = _pull_request(
        review_state=_enum_member(
            GitHubReviewState,
            "changes_requested",
            "blocked",
        ),
    )

    result = RiskRuleEngine().evaluate_pull_request(
        pull_request,
        run_id="test-run-002",
    )

    categories = {signal.category for signal in result.signals}

    assert RiskCategory.REVIEW_BLOCKED in categories
    assert result.max_severity == RiskSeverity.HIGH


def test_stale_pull_request_creates_stale_signal() -> None:
    pull_request = _pull_request(
        created_at=datetime.now(UTC) - timedelta(days=8),
    )

    result = RiskRuleEngine().evaluate_pull_request(
        pull_request,
        run_id="test-run-003",
    )

    categories = {signal.category for signal in result.signals}

    assert RiskCategory.STALE_PULL_REQUEST in categories


def test_large_changeset_creates_large_changeset_signal() -> None:
    pull_request = _pull_request(
        changed_files=30,
        additions=700,
        deletions=100,
    )

    result = RiskRuleEngine().evaluate_pull_request(
        pull_request,
        run_id="test-run-004",
    )

    categories = {signal.category for signal in result.signals}

    assert RiskCategory.LARGE_CHANGESET in categories


def test_missing_jira_key_creates_traceability_signal() -> None:
    pull_request = _pull_request(
        title="Fix checkout validation",
        head_branch="fix-checkout-validation",
    )

    result = RiskRuleEngine().evaluate_pull_request(
        pull_request,
        run_id="test-run-005",
    )

    categories = {signal.category for signal in result.signals}

    assert RiskCategory.MISSING_JIRA_LINK in categories


def test_clean_pull_request_returns_no_risk_signals() -> None:
    pull_request = _pull_request()

    result = RiskRuleEngine().evaluate_pull_request(
        pull_request,
        run_id="test-run-006",
    )

    assert result.signals == []
    assert result.total_score == 0.0
    assert result.max_severity is None


def test_evaluate_pull_requests_preserves_input_count() -> None:
    pull_requests = [
        _pull_request(number=1, title="PAY-101 Safe change"),
        _pull_request(
            number=2,
            title="Fix risky change",
            head_branch="fix-risky-change",
            ci_status=_enum_member(GitHubCIStatus, "failure", "failed"),
        ),
    ]

    results = RiskRuleEngine().evaluate_pull_requests(
        pull_requests,
        run_id="test-run-007",
    )

    assert len(results) == 2
    assert results[0].pull_request_number == 1
    assert results[1].pull_request_number == 2