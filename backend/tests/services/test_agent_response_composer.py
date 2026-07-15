"""Unit tests for deterministic AgentFlow conversational responses."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

from app.schemas.agent_query import (
    AgentIntent,
    AgentQueryPlan,
    ResponseDepth,
)
from app.schemas.risk import ReleaseRunRiskResponse
from app.services.agent_response_composer import AgentResponseComposer
from tests.services.test_slack_release_alert_service import (
    build_snapshot_payload,
)


def build_plan(
    response_depth: ResponseDepth = ResponseDepth.STANDARD,
) -> AgentQueryPlan:
    """Build a valid release-risk query plan."""

    return AgentQueryPlan(
        intent=AgentIntent.RELEASE_RISK_SUMMARY,
        response_depth=response_depth,
        confidence=1.0,
        requires_current_snapshot=True,
        routing_reason_code="test_release_risk_summary",
    )


def build_release_risk_response() -> ReleaseRunRiskResponse:
    """Build a valid release-risk response for composer tests."""

    return ReleaseRunRiskResponse.model_validate(
        build_snapshot_payload(
            release_run_id=uuid4(),
            approval_request_id=uuid4(),
        )
    )


def test_composes_standard_manager_friendly_answer() -> None:
    """Standard responses should include severity, metrics, score, and risks."""

    composer = AgentResponseComposer(request_id="request-123")
    release_risk = build_release_risk_response()

    response = composer.compose(
        plan=build_plan(),
        release_risk=release_risk,
    )

    assert "The release risk is high." in response.answer
    assert "Recommended action: review required." in response.answer
    assert "1 risk signals" in response.answer
    assert "1 high-severity signals" in response.answer
    assert "deterministic risk score is 78%" in response.answer
    assert "Payment API has failing CI" in response.answer
    assert response.release_risk is release_risk
    assert response.approval_required is True


def test_composes_brief_answer_without_detailed_metrics() -> None:
    """Brief responses should return only the primary decision."""

    composer = AgentResponseComposer(request_id="request-123")

    response = composer.compose(
        plan=build_plan(ResponseDepth.BRIEF),
        release_risk=build_release_risk_response(),
    )

    assert response.answer == ("The release risk is high. Recommended action: review required.")


def test_composes_deep_answer_with_source_statuses() -> None:
    """Deep responses should include source and retrieval status."""

    composer = AgentResponseComposer(request_id="request-123")

    response = composer.compose(
        plan=build_plan(ResponseDepth.DEEP),
        release_risk=build_release_risk_response(),
    )

    assert "GitHub status: success." in response.answer
    assert "Jira status: success." in response.answer
    assert "Knowledge retrieval status:" in response.answer


def test_builds_github_and_jira_citations() -> None:
    """Top release risks should become trusted source citations."""

    payload = build_snapshot_payload(
        release_run_id=uuid4(),
        approval_request_id=uuid4(),
    )
    jira_risk = {
        "source": "jira",
        "source_type": "jira_issue",
        "source_id": "PAY-102",
        "source_url": "https://jira.example/browse/PAY-102",
        "severity": "critical",
        "score": 0.95,
        "title": "Payment release blocker",
        "reason": "Critical production bug remains open.",
        "evidence": {},
    }
    payload["release_summary"]["top_risks"].append(jira_risk)

    release_risk = ReleaseRunRiskResponse.model_validate(payload)
    composer = AgentResponseComposer(request_id="request-123")

    response = composer.compose(
        plan=build_plan(),
        release_risk=release_risk,
    )

    assert len(response.citations) == 2
    assert response.citations[0].source_type == "github_pull_request"
    assert response.citations[0].source_id == "1"
    assert response.citations[1].source_type == "jira_issue"
    assert response.citations[1].source_id == "PAY-102"


def test_builds_knowledge_citations() -> None:
    """Retrieved engineering documents should become citations."""

    payload = build_snapshot_payload(
        release_run_id=uuid4(),
        approval_request_id=uuid4(),
    )
    chunk_id = uuid4()

    payload["knowledge_results"] = [
        {
            "document_id": str(uuid4()),
            "chunk_id": str(chunk_id),
            "source_type": "runbook",
            "title": "Payment Service Runbook",
            "content": "Rollback the payment service when error rates spike.",
            "score": 0.91,
            "metadata": {},
        }
    ]

    release_risk = ReleaseRunRiskResponse.model_validate(payload)
    composer = AgentResponseComposer(request_id="request-123")

    response = composer.compose(
        plan=build_plan(),
        release_risk=release_risk,
    )

    knowledge_citation = response.citations[-1]

    assert knowledge_citation.source == "knowledge"
    assert knowledge_citation.source_type == "runbook"
    assert knowledge_citation.source_id == str(chunk_id)
    assert knowledge_citation.title == "Payment Service Runbook"


def test_deduplicates_repeated_citations() -> None:
    """Repeated evidence should appear only once in the response."""

    payload = build_snapshot_payload(
        release_run_id=uuid4(),
        approval_request_id=uuid4(),
    )
    original_risk = payload["release_summary"]["top_risks"][0]
    payload["release_summary"]["top_risks"].append(deepcopy(original_risk))

    release_risk = ReleaseRunRiskResponse.model_validate(payload)
    composer = AgentResponseComposer(request_id="request-123")

    response = composer.compose(
        plan=build_plan(),
        release_risk=release_risk,
    )

    github_citations = [
        citation
        for citation in response.citations
        if citation.source_type == "github_pull_request" and citation.source_id == "1"
    ]

    assert len(github_citations) == 1


def test_includes_human_approval_warning() -> None:
    """Approval-required releases should include the HITL warning."""

    composer = AgentResponseComposer(request_id="request-123")

    response = composer.compose(
        plan=build_plan(),
        release_risk=build_release_risk_response(),
    )

    assert (
        "Human approval is required before any downstream "
        "release notification or Slack action." in response.answer
    )


def test_composes_focused_specific_risk_explanation() -> None:
    """Specific-risk responses should explain only the matched persisted risk."""

    payload = build_snapshot_payload(
        release_run_id=uuid4(),
        approval_request_id=uuid4(),
    )
    payload["release_summary"]["top_risks"][0]["evidence"] = {
        "ci_status": "failed",
        "service": "payment-api",
    }

    release_risk = ReleaseRunRiskResponse.model_validate(payload)
    selected_risk = release_risk.release_summary.top_risks[0]
    plan = build_plan(ResponseDepth.DEEP).model_copy(
        update={
            "intent": AgentIntent.EXPLAIN_SPECIFIC_RISK,
            "routing_reason_code": "test_specific_risk_explanation",
        }
    )
    composer = AgentResponseComposer(request_id="request-123")

    response = composer.compose_specific_risk(
        plan=plan,
        release_risk=release_risk,
        selected_risk=selected_risk,
    )

    assert "Payment API has failing CI" in response.answer
    assert "CI failed on a release-critical service." in response.answer
    assert "high severity" in response.answer
    assert "78%" in response.answer
    assert "ci status: failed" in response.answer.lower()
    assert "service: payment-api" in response.answer.lower()
    assert len(response.citations) == 1
    assert response.citations[0].source_id == "1"
    assert response.release_risk is release_risk
    assert response.approval_required is True


def test_composes_filtered_risk_response() -> None:
    """Filtered responses should include only matching risks and citations."""

    payload = build_snapshot_payload(
        release_run_id=uuid4(),
        approval_request_id=uuid4(),
    )
    payload["release_summary"]["top_risks"].append(
        {
            "source": "jira",
            "source_type": "jira_issue",
            "source_id": "PAY-102",
            "source_url": "https://jira.example/browse/PAY-102",
            "severity": "critical",
            "score": 0.95,
            "title": "Payment release blocker",
            "reason": "A critical payment issue blocks the release.",
            "evidence": {
                "status": "blocked",
                "is_blocking_release": True,
            },
        }
    )

    release_risk = ReleaseRunRiskResponse.model_validate(payload)
    selected_risks = [
        release_risk.release_summary.top_risks[1],
    ]
    plan = build_plan(ResponseDepth.STANDARD).model_copy(
        update={
            "intent": AgentIntent.FILTER_RISKS,
            "routing_reason_code": "test_risk_filter",
        }
    )
    composer = AgentResponseComposer(request_id="request-123")

    response = composer.compose_filtered_risks(
        plan=plan,
        release_risk=release_risk,
        selected_risks=selected_risks,
    )

    assert "1 matching risk" in response.answer
    assert "Payment release blocker" in response.answer
    assert "A critical payment issue blocks the release." in response.answer
    assert "Payment API has failing CI" not in response.answer

    assert len(response.citations) == 1
    assert response.citations[0].source_type == "jira_issue"
    assert response.citations[0].source_id == "PAY-102"
    assert response.release_risk is release_risk


def test_composes_github_pr_response_from_persisted_signals() -> None:
    """PR responses should explain only the requested persisted pull request."""

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
            "evaluated_at": payload["release_summary"]["generated_at"],
        }
    ]
    payload["github"]["risk_result_count"] = 1

    release_risk = ReleaseRunRiskResponse.model_validate(payload)
    pull_request = release_risk.github.risk_results[0]
    plan = build_plan(ResponseDepth.STANDARD).model_copy(
        update={
            "intent": AgentIntent.GITHUB_PR_QUESTION,
            "routing_reason_code": "test_github_pr_question",
        }
    )
    composer = AgentResponseComposer(request_id="request-123")

    response = composer.compose_github_pr(
        plan=plan,
        release_risk=release_risk,
        pull_request=pull_request,
    )

    assert "PR 42" in response.answer
    assert "high severity" in response.answer
    assert "85%" in response.answer
    assert "Payment API has failing CI" in response.answer
    assert "CI failed on a release-critical payment service." in response.answer
    assert "ci status: failed" in response.answer.lower()

    assert len(response.citations) == 1
    assert response.citations[0].source_type == "github_pull_request"
    assert response.citations[0].source_id == "PR-42"
    assert response.release_risk is release_risk


def test_composes_jira_ticket_response_from_persisted_signals() -> None:
    """Jira responses should explain only the requested persisted ticket."""

    payload = build_snapshot_payload(
        release_run_id=uuid4(),
        approval_request_id=uuid4(),
    )
    payload["jira"]["issues"] = [
        {
            "issue_key": "PAY-102",
            "title": "Payment release blocker",
            "issue_url": "https://jira.example/browse/PAY-102",
            "signals": [
                {
                    "source_type": "jira_issue",
                    "source_id": "PAY-102",
                    "source_url": "https://jira.example/browse/PAY-102",
                    "rule_id": "jira_release_blocker",
                    "category": "release_blocker_issue",
                    "severity": "critical",
                    "score": 0.95,
                    "title": "Jira issue is marked as a release blocker",
                    "description": "The issue explicitly blocks the release.",
                    "evidence": {
                        "status": "blocked",
                        "priority": "P1",
                        "is_blocking_release": True,
                    },
                }
            ],
        }
    ]
    payload["jira"]["total_issues_analyzed"] = 1
    payload["jira"]["total_signals"] = 1
    payload["jira"]["signals"] = payload["jira"]["issues"][0]["signals"]

    release_risk = ReleaseRunRiskResponse.model_validate(payload)
    jira_issue = release_risk.jira.issues[0]
    plan = build_plan(ResponseDepth.STANDARD).model_copy(
        update={
            "intent": AgentIntent.JIRA_TICKET_QUESTION,
            "routing_reason_code": "test_jira_ticket_question",
        }
    )
    composer = AgentResponseComposer(request_id="request-123")

    response = composer.compose_jira_ticket(
        plan=plan,
        release_risk=release_risk,
        jira_issue=jira_issue,
    )

    assert "PAY-102" in response.answer
    assert "Payment release blocker" in response.answer
    assert "critical severity" in response.answer
    assert "95%" in response.answer
    assert "Jira issue is marked as a release blocker" in response.answer
    assert "The issue explicitly blocks the release." in response.answer
    assert "status: blocked" in response.answer.lower()
    assert "priority: P1" in response.answer

    assert len(response.citations) == 1
    assert response.citations[0].source_type == "jira_issue"
    assert response.citations[0].source_id == "PAY-102"
    assert response.release_risk is release_risk


def test_composes_workflow_status_from_persisted_snapshot() -> None:
    """Workflow-status responses should report trusted persisted state."""

    release_risk = build_release_risk_response()
    plan = build_plan(ResponseDepth.BRIEF).model_copy(
        update={
            "intent": AgentIntent.WORKFLOW_STATUS_QUESTION,
            "routing_reason_code": "test_workflow_status_question",
        }
    )
    composer = AgentResponseComposer(request_id="request-123")

    response = composer.compose_workflow_status(
        plan=plan,
        release_risk=release_risk,
    )

    assert "Workflow status: waiting for approval." in response.answer
    assert "GitHub collection: success." in response.answer
    assert "Jira collection: success." in response.answer
    assert "Knowledge retrieval: not available." in response.answer
    assert "Approval status: pending." in response.answer
    assert response.citations == []
    assert response.release_risk is release_risk
    assert response.approval_required is True


def test_composes_latest_approval_status_instead_of_snapshot_status() -> None:
    """Approval responses should prefer the latest durable approval decision."""

    release_risk = build_release_risk_response()
    plan = build_plan(ResponseDepth.BRIEF).model_copy(
        update={
            "intent": AgentIntent.APPROVAL_STATUS_QUESTION,
            "routing_reason_code": "test_approval_status_question",
        }
    )
    latest_approval = SimpleNamespace(
        id=uuid4(),
        approval_status="approved",
        approval_reason="High risk requires manager approval.",
        approval_policy_version="hitl_policy_v1",
        requested_by="agent-query-api",
        decided_by="director@example.com",
        decision_note="Approved after reviewing the rollback plan.",
        created_at=None,
        decided_at=None,
    )
    composer = AgentResponseComposer(request_id="request-123")

    response = composer.compose_approval_status(
        plan=plan,
        release_risk=release_risk,
        latest_approval=latest_approval,
    )

    assert "Approval status: approved." in response.answer
    assert "Decided by: director@example.com." in response.answer
    assert (
        "Decision note: Approved after reviewing the rollback plan."
        in response.answer
    )
    assert "Approval reason: High risk requires manager approval." in response.answer
    assert response.citations == []
    assert response.release_risk is release_risk
    assert response.approval_required is True

def test_composes_historical_risk_lookup_from_previous_snapshots() -> None:
    """Historical responses should summarize trusted previous release snapshots."""
    current_release_risk = build_release_risk_response()
    previous_release_risk = build_release_risk_response().model_copy(
        update={
            "release_run": build_release_risk_response().release_run.model_copy(
                update={
                    "id": uuid4(),
                    "run_id": "release-run-previous",
                }
            ),
        }
    )
    plan = build_plan(ResponseDepth.DEEP).model_copy(
        update={
            "intent": AgentIntent.HISTORICAL_RISK_LOOKUP,
            "requires_historical_lookup": True,
            "routing_reason_code": "test_historical_risk_lookup",
        }
    )
    composer = AgentResponseComposer(request_id="request-123")

    response = composer.compose_historical_risks(
        plan=plan,
        release_risk=current_release_risk,
        historical_release_risks=[previous_release_risk],
    )

    assert "Found 1 previous release with persisted risk history." in response.answer
    assert "release-run-previous" in response.answer
    assert "high severity" in response.answer
    assert "78% risk score" in response.answer
    assert "Payment API has failing CI" in response.answer
    assert response.release_risk is current_release_risk
    assert response.approval_required is True
    assert len(response.citations) == 1
    assert response.citations[0].source_type == "github_pull_request"
    assert response.citations[0].title.startswith(
        "[release-run-previous] "
    )
