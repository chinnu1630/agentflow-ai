"""Reusable trusted evidence for release-risk synthesis tests."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from app.schemas.risk import ReleaseRunRiskResponse
from tests.services.test_slack_release_alert_service import (
    build_snapshot_payload,
)


def build_release_risk_with_full_evidence() -> ReleaseRunRiskResponse:
    """Build realistic trusted evidence across every synthesis source type."""
    payload = build_snapshot_payload(
        release_run_id=uuid4(),
        approval_request_id=uuid4(),
    )
    now = datetime.now(UTC).isoformat()

    github = cast(dict[str, object], payload["github"])
    github["risk_results"] = [
        {
            "source_type": "github_pull_request",
            "source_id": "42",
            "source_url": "https://github.example/pr/42",
            "pull_request_number": 42,
            "total_score": 0.92,
            "max_severity": "high",
            "signals": [
                {
                    "source_type": "github_pull_request",
                    "source_id": "42",
                    "source_url": "https://github.example/pr/42",
                    "rule_id": "CI_FAILURE",
                    "category": "ci_failure",
                    "severity": "high",
                    "score": 0.92,
                    "title": "Payment integration checks failed",
                    "description": (
                        "Three required payment integration checks are failing."
                    ),
                    "evidence": {
                        "failed_check_count": 3,
                        "required_check": True,
                    },
                }
            ],
            "evaluated_at": now,
        }
    ]

    release_summary = cast(dict[str, object], payload["release_summary"])
    release_summary["top_risks"] = [
        {
            "source": "github",
            "source_type": "github_pull_request",
            "source_id": "42",
            "source_url": "https://github.example/pr/42",
            "severity": "high",
            "score": 0.92,
            "title": "Payment API has failing CI",
            "reason": "Required payment integration checks failed.",
            "evidence": {"failed_check_count": 3},
        }
    ]

    jira = cast(dict[str, object], payload["jira"])
    jira_signal = {
        "source_type": "jira_issue",
        "source_id": "PAY-102",
        "source_url": "https://jira.example/browse/PAY-102",
        "rule_id": "OPEN_CRITICAL_BUG",
        "category": "open_critical_bug",
        "severity": "critical",
        "score": 1.0,
        "title": "Critical checkout defect remains open",
        "description": "PAY-102 blocks successful checkout authorization.",
        "evidence": {"priority": "P1", "status": "Open"},
    }
    jira["issues"] = [
        {
            "issue_key": "PAY-102",
            "title": "Checkout authorization fails",
            "issue_url": "https://jira.example/browse/PAY-102",
            "signals": [jira_signal],
        }
    ]
    jira["signals"] = [jira_signal]

    document_id = uuid4()
    chunk_id = uuid4()
    payload["knowledge_status"] = "success"
    payload["knowledge_results"] = [
        {
            "document_id": str(document_id),
            "chunk_id": str(chunk_id),
            "source_type": "runbook",
            "title": "Payment Service Runbook",
            "content": (
                "Rollback requires deploying the previous stable image "
                "and validating checkout health."
            ),
            "score": 0.95,
            "metadata": {"service": "payments"},
        }
    ]

    return ReleaseRunRiskResponse.model_validate(payload)
