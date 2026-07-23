"""Tests for Claude release-risk synthesis workflow node."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from app.integrations.anthropic_client import (
    AnthropicClientUnavailableError,
    ClaudeSynthesisResult,
)
from app.schemas.llm_risk_synthesis import (
    ClaudeReleaseRiskReport,
    SynthesisEvidenceCitation,
    SynthesisEvidenceSource,
    SynthesizedReleaseRisk,
)
from app.schemas.risk import (
    RiskSeverityResponse,
    RiskSummaryActionResponse,
)
from app.workflows.release_risk_service_graph import (
    build_release_risk_service_graph,
)
from app.workflows.release_risk_service_nodes import (
    create_synthesize_release_risk_node,
)
from app.workflows.release_risk_state import (
    ReleaseRiskState,
    ReleaseRiskWorkflowStage,
    ReleaseRiskWorkflowStatus,
    RiskSynthesisStatus,
)
from tests.services.test_slack_release_alert_service import (
    build_snapshot_payload,
)


class FakeRiskSynthesisService:
    """Return one successful structured Claude synthesis."""

    def __init__(self) -> None:
        """Initialize captured prompt values."""
        self.system_prompt: str | None = None
        self.user_prompt: str | None = None
        self.prompt_version: str | None = None

    async def synthesize_release_risk(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        prompt_version: str,
    ) -> ClaudeSynthesisResult:
        """Return a validated structured release-risk report."""
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.prompt_version = prompt_version

        report = ClaudeReleaseRiskReport(
            recommendation=RiskSummaryActionResponse.REVIEW_REQUIRED,
            confidence=0.91,
            executive_summary="The release requires manager review.",
            risks=[
                SynthesizedReleaseRisk(
                    rank=1,
                    title="Payment API CI failure",
                    severity=RiskSeverityResponse.HIGH,
                    confidence=0.93,
                    explanation="The payment PR has failing CI.",
                    evidence=[
                        SynthesisEvidenceCitation(
                            source=(
                                SynthesisEvidenceSource.GITHUB_PULL_REQUEST
                            ),
                            source_id="1",
                            title="Payment API has failing CI",
                            source_url="https://github.example/pr/1",
                            supporting_fact=(
                                "CI failed on a release-critical service."
                            ),
                        )
                    ],
                    mitigations=["Fix CI before deployment."],
                )
            ],
            requires_human_review=True,
        )

        return ClaudeSynthesisResult(
            report=report,
            message_id="msg-test-001",
            model="test-claude-model",
            input_tokens=500,
            output_tokens=220,
            stop_reason="end_turn",
            duration_ms=750.0,
            prompt_version=prompt_version,
        )


class UnsupportedGroundednessSynthesisService:
    """Return valid citations with claims unsupported by trusted evidence."""

    async def synthesize_release_risk(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        prompt_version: str,
    ) -> ClaudeSynthesisResult:
        """Return structurally valid but lexically unsupported synthesis."""
        report = ClaudeReleaseRiskReport(
            recommendation=RiskSummaryActionResponse.REVIEW_REQUIRED,
            confidence=0.9,
            executive_summary="The release requires human review.",
            risks=[
                SynthesizedReleaseRisk(
                    rank=1,
                    title="Unrelated analytics deployment",
                    severity=RiskSeverityResponse.HIGH,
                    confidence=0.9,
                    explanation=(
                        "The analytics database must be replaced immediately."
                    ),
                    evidence=[
                        SynthesisEvidenceCitation(
                            source=(
                                SynthesisEvidenceSource.GITHUB_PULL_REQUEST
                            ),
                            source_id="1",
                            title="Payment API has failing CI",
                            source_url="https://github.example/pr/1",
                            supporting_fact=(
                                "The analytics database has corrupted backups."
                            ),
                        )
                    ],
                    mitigations=["Replace the analytics database."],
                )
            ],
            requires_human_review=True,
        )

        return ClaudeSynthesisResult(
            report=report,
            message_id="msg-unsupported-groundedness",
            model="test-claude-model",
            input_tokens=410,
            output_tokens=190,
            stop_reason="end_turn",
            duration_ms=510.0,
            prompt_version=prompt_version,
        )


class HallucinatedCitationSynthesisService:
    """Return structured output containing an invented evidence citation."""

    async def synthesize_release_risk(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        prompt_version: str,
    ) -> ClaudeSynthesisResult:
        """Return a valid schema containing an untrusted source ID."""
        report = ClaudeReleaseRiskReport(
            recommendation=RiskSummaryActionResponse.REVIEW_REQUIRED,
            confidence=0.9,
            executive_summary="The release requires human review.",
            risks=[
                SynthesizedReleaseRisk(
                    rank=1,
                    title="Invented payment risk",
                    severity=RiskSeverityResponse.HIGH,
                    confidence=0.9,
                    explanation="The model cited evidence that does not exist.",
                    evidence=[
                        SynthesisEvidenceCitation(
                            source=SynthesisEvidenceSource.GITHUB_PULL_REQUEST,
                            source_id="invented-pr-999",
                            title="Invented pull request",
                            supporting_fact="This evidence is not trusted.",
                        )
                    ],
                    mitigations=["Review trusted release evidence."],
                )
            ],
            requires_human_review=True,
        )

        return ClaudeSynthesisResult(
            report=report,
            message_id="msg-hallucinated-citation",
            model="test-claude-model",
            input_tokens=400,
            output_tokens=180,
            stop_reason="end_turn",
            duration_ms=500.0,
            prompt_version=prompt_version,
        )



class FailingRiskSynthesisService:
    """Simulate an unavailable Claude service."""

    async def synthesize_release_risk(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        prompt_version: str,
    ) -> ClaudeSynthesisResult:
        """Raise a safe Anthropic client error."""
        raise AnthropicClientUnavailableError(
            "Claude service unavailable."
        )


def build_scored_workflow_state() -> ReleaseRiskState:
    """Build workflow state with deterministic risk evidence and score."""
    payload = build_snapshot_payload(
        release_run_id=uuid4(),
        approval_request_id=uuid4(),
    )

    payload.pop("approval_request_id", None)
    payload.pop("approval_status", None)
    payload["knowledge_status"] = "not_started"

    release_run = payload["release_run"]
    assert isinstance(release_run, dict)

    return ReleaseRiskState.model_validate(
        {
            **payload,
            "release_run_id": release_run["id"],
            "run_id": "release-run-001",
            "manager_query": (
                "What are the biggest release risks this week?"
            ),
            "status": "running",
            "stage": "scoring_release_risk",
            "completed_nodes": ["score_release_risk"],
            "failed_nodes": [],
            "errors": [],
        }
    )


@pytest.mark.anyio
async def test_synthesis_node_stores_structured_report_and_usage() -> None:
    """Successful synthesis should store report and LLM metadata."""
    synthesis_service = FakeRiskSynthesisService()
    node = create_synthesize_release_risk_node(synthesis_service)

    result = await node(build_scored_workflow_state())
    final_state = ReleaseRiskState.model_validate(result)

    assert (
        final_state.stage
        is ReleaseRiskWorkflowStage.SYNTHESIZING_RELEASE_RISK
    )
    assert final_state.synthesis_status is RiskSynthesisStatus.COMPLETED
    assert final_state.synthesis_report is not None
    assert final_state.synthesis_report["recommendation"] == (
        "review_required"
    )
    assert final_state.synthesis_model == "test-claude-model"
    assert final_state.synthesis_input_tokens == 500
    assert final_state.synthesis_output_tokens == 220
    assert final_state.synthesis_duration_ms == 750.0
    assert final_state.synthesis_error is None
    assert "synthesize_release_risk" in final_state.completed_nodes

    assert synthesis_service.system_prompt is not None
    assert synthesis_service.user_prompt is not None
    assert (
        synthesis_service.prompt_version
        == "release-risk-synthesis-v1"
    )


@pytest.mark.anyio
async def test_synthesis_node_rejects_unsupported_groundedness() -> None:
    """Unsupported claims should trigger safe deterministic fallback."""
    node = create_synthesize_release_risk_node(
        UnsupportedGroundednessSynthesisService()
    )
    initial_state = build_scored_workflow_state()

    result = await node(initial_state)
    final_state = ReleaseRiskState.model_validate(result)

    assert final_state.status is ReleaseRiskWorkflowStatus.PARTIAL
    assert final_state.synthesis_status is RiskSynthesisStatus.FAILED
    assert final_state.synthesis_report is None
    assert final_state.synthesis_error == "Claude risk synthesis failed."
    assert final_state.risk_score == initial_state.risk_score
    assert final_state.errors[-1].source == "release_risk_synthesis"
    assert final_state.errors[-1].recoverable is True
    assert final_state.errors[-1].details == {
        "error_type": "ReleaseRiskSynthesisGroundednessError"
    }


@pytest.mark.anyio
async def test_synthesis_node_rejects_invented_citation() -> None:
    """Unverified Claude citations should trigger safe deterministic fallback."""
    node = create_synthesize_release_risk_node(
        HallucinatedCitationSynthesisService()
    )
    initial_state = build_scored_workflow_state()

    result = await node(initial_state)
    final_state = ReleaseRiskState.model_validate(result)

    assert final_state.status is ReleaseRiskWorkflowStatus.PARTIAL
    assert final_state.synthesis_status is RiskSynthesisStatus.FAILED
    assert final_state.synthesis_report is None
    assert final_state.synthesis_error == "Claude risk synthesis failed."
    assert final_state.risk_score == initial_state.risk_score
    assert final_state.errors[-1].source == "release_risk_synthesis"
    assert final_state.errors[-1].recoverable is True



@pytest.mark.anyio
async def test_synthesis_node_degrades_gracefully_when_claude_fails() -> None:
    """Claude failure should preserve deterministic workflow evidence."""
    node = create_synthesize_release_risk_node(
        FailingRiskSynthesisService()
    )
    initial_state = build_scored_workflow_state()

    result = await node(initial_state)
    final_state = ReleaseRiskState.model_validate(result)

    assert final_state.status is ReleaseRiskWorkflowStatus.PARTIAL
    assert final_state.synthesis_status is RiskSynthesisStatus.FAILED
    assert final_state.synthesis_report is None
    assert final_state.synthesis_error == "Claude risk synthesis failed."
    assert final_state.risk_score == initial_state.risk_score
    assert final_state.has_errors is True
    assert final_state.errors[-1].source == "release_risk_synthesis"
    assert final_state.errors[-1].recoverable is True



class FakeGraphReleaseRiskCollectionService:
    """Return deterministic release-risk evidence for graph tests."""

    async def collect_release_risks(
        self,
        release_run_id: UUID,
    ) -> dict[str, object]:
        """Return a valid persisted release-risk response payload."""
        payload = build_snapshot_payload(
            release_run_id=release_run_id,
            approval_request_id=uuid4(),
        )

        return {
            "release_run": payload["release_run"],
            "github": payload["github"],
            "github_summary": payload["github_summary"],
            "jira": payload["jira"],
            "jira_summary": payload["jira_summary"],
            "release_summary": payload["release_summary"],
        }


@pytest.mark.anyio
async def test_service_graph_runs_optional_claude_synthesis() -> None:
    """Configured synthesis should execute between scoring and approval."""
    synthesis_service = FakeRiskSynthesisService()
    graph = build_release_risk_service_graph(
        FakeGraphReleaseRiskCollectionService(),
        synthesis_service=synthesis_service,
    )
    initial_state = ReleaseRiskState(
        release_run_id=uuid4(),
        run_id="release-run-graph-001",
    )

    result = await graph.ainvoke(initial_state)
    final_state = ReleaseRiskState.model_validate(result)

    assert final_state.status is ReleaseRiskWorkflowStatus.SUCCEEDED
    assert final_state.stage is ReleaseRiskWorkflowStage.COMPLETED
    assert final_state.synthesis_status is RiskSynthesisStatus.COMPLETED
    assert final_state.synthesis_report is not None
    assert final_state.completed_nodes == [
        "start_release_risk_workflow",
        "collect_release_risks",
        "score_release_risk",
        "synthesize_release_risk",
        "determine_approval_requirement",
        "complete_release_risk_workflow",
    ]


@pytest.mark.anyio
async def test_service_graph_continues_when_claude_synthesis_fails() -> None:
    """Recoverable Claude failure should not bypass approval or completion."""
    graph = build_release_risk_service_graph(
        FakeGraphReleaseRiskCollectionService(),
        synthesis_service=FailingRiskSynthesisService(),
    )
    initial_state = ReleaseRiskState(
        release_run_id=uuid4(),
        run_id="release-run-graph-002",
    )

    result = await graph.ainvoke(initial_state)
    final_state = ReleaseRiskState.model_validate(result)

    assert final_state.status is ReleaseRiskWorkflowStatus.SUCCEEDED
    assert final_state.stage is ReleaseRiskWorkflowStage.COMPLETED
    assert final_state.synthesis_status is RiskSynthesisStatus.FAILED
    assert final_state.synthesis_report is None
    assert final_state.has_errors is True
    assert final_state.errors[-1].source == "release_risk_synthesis"
    assert "determine_approval_requirement" in final_state.completed_nodes
    assert "complete_release_risk_workflow" in final_state.completed_nodes
