"""Tests for risk collection services."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Sequence

from app.schemas.github import GitHubPullRequest
from app.services.github_risk_collector import RiskCollectionStatus, RiskCollector
from app.services.github_risk_rules import (
    PullRequestRiskResult,
    RiskCategory,
    RiskSeverity,
    RiskSignal,
)


class FakeGitHubClient:
    """Fake GitHub client for risk collector tests."""

    def __init__(
        self,
        pull_requests: list[GitHubPullRequest] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._pull_requests = pull_requests or []
        self._error = error
        self.was_called = False

    def list_open_pull_requests(self) -> list[GitHubPullRequest]:
        """Return fake pull requests or raise a configured error."""
        self.was_called = True

        if self._error is not None:
            raise self._error

        return self._pull_requests


class FakeRiskRuleEngine:
    """Fake pull request risk evaluator for risk collector tests."""

    def __init__(self, risk_results: list[PullRequestRiskResult]) -> None:
        self._risk_results = risk_results
        self.received_pull_requests: Sequence[GitHubPullRequest] | None = None
        self.received_run_id: str | None = None
        self.was_called = False

    def evaluate_pull_requests(
        self,
        pull_requests: Sequence[GitHubPullRequest],
        *,
        run_id: str,
    ) -> list[PullRequestRiskResult]:
        """Return fake risk results while recording input arguments."""
        self.was_called = True
        self.received_pull_requests = pull_requests
        self.received_run_id = run_id

        return self._risk_results


def _pull_request(number: int) -> GitHubPullRequest:
    return GitHubPullRequest.model_construct(number=number)


def _risk_result(
    pull_request_number: int,
    severity: RiskSeverity,
) -> PullRequestRiskResult:
    signal = RiskSignal(
        source_id=f"PR-{pull_request_number}",
        source_url=None,
        rule_id="test_rule",
        category=RiskCategory.CI_FAILURE,
        severity=severity,
        score=0.85,
        title="Test risk signal",
        description="Test risk signal description.",
        evidence={"test": True},
    )

    return PullRequestRiskResult(
        source_id=f"PR-{pull_request_number}",
        source_url=None,
        pull_request_number=pull_request_number,
        total_score=0.85,
        max_severity=severity,
        signals=[signal],
        evaluated_at=datetime.now(UTC),
    )


def test_collect_github_risks_fetches_and_evaluates_pull_requests() -> None:
    pull_requests = [_pull_request(1), _pull_request(2)]
    risk_results = [
        _risk_result(1, RiskSeverity.HIGH),
        _risk_result(2, RiskSeverity.MEDIUM),
    ]

    github_client = FakeGitHubClient(pull_requests=pull_requests)
    risk_rule_engine = FakeRiskRuleEngine(risk_results=risk_results)
    collector = RiskCollector(
        github_client=github_client,
        risk_rule_engine=risk_rule_engine,
    )

    result = asyncio.run(collector.collect_github_risks(run_id="test-run-001"))

    assert result.status == RiskCollectionStatus.SUCCESS
    assert result.pull_request_count == 2
    assert result.risk_result_count == 2
    assert result.total_signal_count == 2
    assert result.high_risk_count == 1
    assert result.risk_results == risk_results
    assert result.error_type is None
    assert result.error_message is None

    assert github_client.was_called is True
    assert risk_rule_engine.was_called is True
    assert risk_rule_engine.received_pull_requests == pull_requests
    assert risk_rule_engine.received_run_id == "test-run-001"


def test_collect_github_risks_handles_empty_pull_request_list() -> None:
    github_client = FakeGitHubClient(pull_requests=[])
    risk_rule_engine = FakeRiskRuleEngine(risk_results=[])
    collector = RiskCollector(
        github_client=github_client,
        risk_rule_engine=risk_rule_engine,
    )

    result = asyncio.run(collector.collect_github_risks(run_id="test-run-002"))

    assert result.status == RiskCollectionStatus.SUCCESS
    assert result.pull_request_count == 0
    assert result.risk_result_count == 0
    assert result.total_signal_count == 0
    assert result.high_risk_count == 0
    assert result.risk_results == []

    assert github_client.was_called is True
    assert risk_rule_engine.was_called is True


def test_collect_github_risks_degrades_when_github_fails() -> None:
    github_client = FakeGitHubClient(error=ConnectionError("GitHub unavailable"))
    risk_rule_engine = FakeRiskRuleEngine(risk_results=[])
    collector = RiskCollector(
        github_client=github_client,
        risk_rule_engine=risk_rule_engine,
    )

    result = asyncio.run(collector.collect_github_risks(run_id="test-run-003"))

    assert result.status == RiskCollectionStatus.DEGRADED
    assert result.pull_request_count == 0
    assert result.risk_result_count == 0
    assert result.total_signal_count == 0
    assert result.high_risk_count == 0
    assert result.risk_results == []
    assert result.error_type == "ConnectionError"
    assert result.error_message is not None

    assert github_client.was_called is True
    assert risk_rule_engine.was_called is False