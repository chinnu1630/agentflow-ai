"""Unit tests for resolving one GitHub PR from persisted risk context."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from app.schemas.agent_query import (
    AgentEntityReferences,
    AgentIntent,
    AgentQueryPlan,
    ResponseDepth,
)
from app.schemas.risk import ReleaseRunRiskResponse
from app.services.agent_github_pr_resolver import (
    AgentGitHubPRNotFoundError,
    AgentGitHubPRResolver,
)
from tests.services.test_slack_release_alert_service import (
    build_snapshot_payload,
)


def build_plan(
    *,
    pull_request_numbers: list[int],
) -> AgentQueryPlan:
    """Build a GitHub PR question plan."""

    return AgentQueryPlan(
        intent=AgentIntent.GITHUB_PR_QUESTION,
        response_depth=ResponseDepth.STANDARD,
        confidence=1.0,
        entity_references=AgentEntityReferences(
            pull_request_numbers=pull_request_numbers,
        ),
        requires_current_snapshot=True,
        routing_reason_code="test_github_pr_question",
    )


def build_release_risk_response() -> ReleaseRunRiskResponse:
    """Build a persisted snapshot containing PR 42 risk details."""

    payload = build_snapshot_payload(
        release_run_id=uuid4(),
        approval_request_id=uuid4(),
    )

    payload["github"]["risk_results"] = [
        {
            "source_type": "github_pull_request",
            "source_id": "PR-42",
            "source_url": "https://github.example/pulls/42",
            "pull_request_number": 42,
            "total_score": 0.85,
            "max_severity": "high",
            "signals": [
                {
                    "source_type": "github_pull_request",
                    "source_id": "PR-42",
                    "source_url": "https://github.example/pulls/42",
                    "rule_id": "ci_failure",
                    "category": "ci_failure",
                    "severity": "high",
                    "score": 0.85,
                    "title": "Payment API has failing CI",
                    "description": (
                        "CI failed on a release-critical payment service."
                    ),
                    "evidence": {
                        "ci_status": "failed",
                        "service": "payment-api",
                    },
                }
            ],
            "evaluated_at": datetime.now(UTC).isoformat(),
        }
    ]
    payload["github"]["risk_result_count"] = 1

    return ReleaseRunRiskResponse.model_validate(payload)


def test_resolves_pr_using_extracted_number() -> None:
    """An extracted PR number should resolve the persisted PR result."""

    resolver = AgentGitHubPRResolver(request_id="request-123")

    pull_request = resolver.resolve(
        plan=build_plan(pull_request_numbers=[42]),
        release_risk=build_release_risk_response(),
    )

    assert pull_request.pull_request_number == 42
    assert pull_request.source_id == "PR-42"
    assert pull_request.total_score == 0.85
    assert len(pull_request.signals) == 1


def test_raises_when_pr_number_is_missing() -> None:
    """A PR-level question must identify one pull request."""

    resolver = AgentGitHubPRResolver(request_id="request-123")

    with pytest.raises(
        AgentGitHubPRNotFoundError,
        match="No pull-request number was provided.",
    ):
        resolver.resolve(
            plan=build_plan(pull_request_numbers=[]),
            release_risk=build_release_risk_response(),
        )


def test_raises_when_persisted_pr_does_not_exist() -> None:
    """An unknown PR number should fail without rerunning GitHub."""

    resolver = AgentGitHubPRResolver(request_id="request-123")

    with pytest.raises(
        AgentGitHubPRNotFoundError,
        match="No persisted GitHub pull request matched the query.",
    ):
        resolver.resolve(
            plan=build_plan(pull_request_numbers=[99]),
            release_risk=build_release_risk_response(),
        )
