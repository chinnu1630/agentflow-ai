"""Unit tests for deterministic AgentFlow conversational responses."""

from __future__ import annotations

from copy import deepcopy
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
