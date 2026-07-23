"""Integration tests for the AgentFlow natural-language query API."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from app.api.dependencies.security import get_current_principal
from app.api.routes.agent_queries import (
    get_agent_dynamic_synthesis_client,
    get_agent_execution_planner_client,
    get_agent_github_risk_collector,
    get_agent_jira_risk_collector,
    get_agent_slack_alert_sender,
)
from app.core.config import get_settings
from app.core.security import AuthenticatedPrincipal
from app.db.base import Base
from app.db.session import get_db_session
from app.integrations.anthropic_dynamic_synthesis_client import (
    ClaudeDynamicSynthesisResult,
)
from app.integrations.anthropic_execution_planner_client import (
    ClaudeExecutionPlanResult,
)
from app.integrations.slack_client import SlackPostMessageResult
from app.main import app
from app.schemas.agent_dynamic_synthesis import (
    AgentDynamicAnswer,
    AgentDynamicAnswerCitation,
)
from app.schemas.agent_execution_plan import (
    AgentExecutionPlan,
    AgentExecutionStep,
)
from app.schemas.agent_query import AgentIntent, ResponseDepth
from app.schemas.agent_tool import (
    AgentToolInvocation,
    AgentToolName,
)
from app.schemas.jira import (
    JiraIssue,
    JiraIssuePriority,
    JiraIssueStatus,
    JiraIssueType,
)
from app.services.engineering_document_embedding_provider import (
    get_engineering_document_embedding_provider,
)
from app.services.engineering_document_reranker import (
    get_engineering_document_reranker,
)
from app.services.github_risk_collector import (
    GitHubRiskCollectionResult,
    RiskCollectionStatus,
)
from app.services.github_risk_rules import (
    PullRequestRiskResult,
    RiskCategory,
    RiskSeverity,
    RiskSignal,
)
from app.services.jira_risk_collector import (
    JiraRiskCollectionResult,
    JiraRiskCollectionStatus,
)
from app.services.jira_risk_rules import JiraRiskRuleEngine
from app.services.slack_alert_payload_service import (
    SlackReleaseRiskAlertPayload,
)


class FakeAgentEmbeddingProvider:
    """Generate deterministic embeddings without loading model weights."""

    async def embed_texts(
        self,
        texts: Sequence[str],
        *,
        run_id: str | None = None,
    ) -> list[list[float]]:
        """Return one 384-dimensional embedding per supplied text."""
        return [[float(index + 1)] * 384 for index, _text in enumerate(texts)]



class FakeAgentReranker:
    """Return deterministic scores without loading a cross-encoder."""

    async def score_candidates(
        self,
        *,
        query: str,
        candidate_contents: Sequence[str],
        run_id: str | None = None,
    ) -> list[float]:
        """Return descending scores in candidate order."""
        return [
            float(len(candidate_contents) - index)
            for index, _content in enumerate(candidate_contents)
        ]


class FakeAgentGitHubRiskCollector:
    """Fake GitHub collector for agent execution integration tests."""

    call_count = 0

    async def collect_github_risks(
        self,
        *,
        run_id: str,
    ) -> GitHubRiskCollectionResult:
        """Return deterministic GitHub release-risk data."""

        type(self).call_count += 1

        signal = RiskSignal(
            source_id="PR-42",
            source_url="https://github.example/pulls/42",
            rule_id="integration_ci_failure",
            category=RiskCategory.CI_FAILURE,
            severity=RiskSeverity.HIGH,
            score=0.85,
            title="Payment API has failing CI",
            description="CI failed on a release-critical payment service.",
            evidence={
                "ci_status": "failed",
                "service": "payment-api",
            },
        )
        risk_result = PullRequestRiskResult(
            source_id="PR-42",
            source_url="https://github.example/pulls/42",
            pull_request_number=42,
            total_score=0.85,
            max_severity=RiskSeverity.HIGH,
            signals=[signal],
            evaluated_at=datetime.now(UTC),
        )

        return GitHubRiskCollectionResult(
            status=RiskCollectionStatus.SUCCESS,
            pull_request_count=1,
            risk_result_count=1,
            total_signal_count=1,
            high_risk_count=1,
            risk_results=[risk_result],
            collected_at=datetime.now(UTC),
            duration_ms=10.0,
        )


class FakeAgentJiraRiskCollector:
    """Fake Jira collector for agent execution integration tests."""

    call_count = 0

    async def collect(
        self,
        *,
        run_id: str,
    ) -> JiraRiskCollectionResult:
        """Return deterministic Jira release-risk data."""

        type(self).call_count += 1

        now = datetime.now(UTC)
        issue = JiraIssue(
            issue_key="PAY-102",
            title="Payment release blocker",
            description="A critical payment defect blocks the release.",
            issue_type=JiraIssueType.BUG,
            status=JiraIssueStatus.BLOCKED,
            priority=JiraIssuePriority.P1,
            assignee="payments-team@example.com",
            reporter="release-manager@example.com",
            labels=["release-blocker"],
            components=["payments"],
            affected_services=["payment-service"],
            issue_url="https://jira.example/browse/PAY-102",
            created_at=now,
            updated_at=now,
            is_blocking_release=True,
        )
        issue_result = JiraRiskRuleEngine().evaluate_issue(
            issue,
            run_id=run_id,
            evaluated_at=now,
        )

        return JiraRiskCollectionResult(
            status=JiraRiskCollectionStatus.SUCCESS,
            issues=[issue],
            issue_results=[issue_result],
            signals=list(issue_result.signals),
            error_message=None,
            duration_ms=10.0,
        )


class FakeAgentExecutionPlannerClient:
    """Return deterministic structured plans without calling Claude."""

    call_count = 0

    async def create_execution_plan(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        prompt_version: str,
    ) -> ClaudeExecutionPlanResult:
        """Return one bounded engineering-knowledge execution plan."""
        del system_prompt, user_prompt
        type(self).call_count += 1

        plan = AgentExecutionPlan(
            objective="Search trusted payment rollback guidance.",
            intent=AgentIntent.KNOWLEDGE_DOC_QUESTION,
            response_depth=ResponseDepth.STANDARD,
            steps=[
                AgentExecutionStep(
                    step_id="search_knowledge",
                    invocation=AgentToolInvocation(
                        step_id="search_knowledge",
                        tool_name=(
                            AgentToolName.SEARCH_ENGINEERING_KNOWLEDGE
                        ),
                        arguments={
                            "query": "payment service rollback",
                            "top_k": 5,
                        },
                        timeout_seconds=30,
                    ),
                )
            ],
            plan_reason_code="search_engineering_knowledge",
        )

        return ClaudeExecutionPlanResult(
            plan=plan,
            message_id="msg_dynamic_integration",
            model="test-claude-model",
            input_tokens=200,
            output_tokens=80,
            stop_reason="end_turn",
            duration_ms=20.0,
            prompt_version=prompt_version,
        )



class FakeAgentDynamicSynthesisClient:
    """Return grounded dynamic answers without calling Claude."""

    call_count = 0
    return_invalid_grounding = False

    async def synthesize_dynamic_answer(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        prompt_version: str,
    ) -> ClaudeDynamicSynthesisResult:
        """Build a citation from the exact trusted prompt evidence."""
        assert system_prompt

        type(self).call_count += 1
        evidence_payload = json.loads(user_prompt.split("\n\n", 1)[1])
        trusted_evidence = evidence_payload["trusted_evidence"]
        citations = []

        if trusted_evidence and not type(self).return_invalid_grounding:
            evidence = trusted_evidence[0]
            citations.append(
                AgentDynamicAnswerCitation(
                    source_type=evidence["source_type"],
                    source_id=evidence["source_id"],
                    title=evidence["title"],
                    source_url=evidence["source_url"],
                    supporting_fact=(
                        "The executed AgentFlow tool returned this evidence."
                    ),
                )
            )

        answer_text = (
            "UNTRUSTED_CLAUDE_OUTPUT_SHOULD_NOT_BE_EXPOSED"
            if type(self).return_invalid_grounding
            else (
                "The payment service runbook says to roll back when "
                "authorization failures exceed the release threshold, "
                "then validate transaction success and error-rate recovery."
            )
        )

        return ClaudeDynamicSynthesisResult(
            answer=AgentDynamicAnswer(
                answer=answer_text,
                confidence=0.95,
                citations=citations,
                requires_human_review=False,
            ),
            message_id="msg_dynamic_synthesis_integration",
            model="test-claude-model",
            input_tokens=300,
            output_tokens=120,
            stop_reason="end_turn",
            duration_ms=25.0,
            prompt_version=prompt_version,
        )


class FakeAgentSlackAlertSender:
    """Fake Slack sender for natural-language action tests."""

    call_count = 0
    sent_payloads: list[SlackReleaseRiskAlertPayload] = []

    async def send_release_risk_alert(
        self,
        payload: SlackReleaseRiskAlertPayload,
    ) -> SlackPostMessageResult:
        """Capture one Slack action without network I/O."""
        type(self).call_count += 1
        type(self).sent_payloads.append(payload)

        return SlackPostMessageResult(
            ok=True,
            channel="C1234567890",
            timestamp="12345.6789",
        )


@pytest.fixture
async def agent_query_api_client() -> AsyncIterator[AsyncClient]:
    """Provide an isolated API client for query planning and execution."""

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async def override_get_db_session() -> AsyncIterator[AsyncSession]:
        """Provide an isolated database session."""

        async with session_factory() as session:
            yield session

    async def override_get_current_principal() -> AuthenticatedPrincipal:
        """Return a trusted release manager for protected API tests."""
        return AuthenticatedPrincipal(
            subject="director-123",
            email="director@example.com",
            roles=frozenset({"release_manager"}),
            scopes=frozenset(
                {
                    "agent:query",
                    "release:read",
                    "release:approve",
                    "release:notify",
                }
            ),
        )

    async def override_get_execution_planner(
    ) -> FakeAgentExecutionPlannerClient:
        """Return the deterministic dynamic execution planner."""

        return FakeAgentExecutionPlannerClient()

    async def override_get_dynamic_synthesis(
    ) -> FakeAgentDynamicSynthesisClient:
        """Return deterministic dynamic answer synthesis."""

        return FakeAgentDynamicSynthesisClient()

    async def override_get_github_collector() -> FakeAgentGitHubRiskCollector:
        """Return the fake GitHub collector."""

        return FakeAgentGitHubRiskCollector()

    async def override_get_jira_collector() -> FakeAgentJiraRiskCollector:
        """Return the fake Jira collector."""

        return FakeAgentJiraRiskCollector()

    async def override_get_slack_sender() -> FakeAgentSlackAlertSender:
        """Return the fake Slack sender."""

        return FakeAgentSlackAlertSender()

    FakeAgentExecutionPlannerClient.call_count = 0
    FakeAgentDynamicSynthesisClient.call_count = 0
    FakeAgentDynamicSynthesisClient.return_invalid_grounding = False
    FakeAgentGitHubRiskCollector.call_count = 0
    FakeAgentJiraRiskCollector.call_count = 0
    FakeAgentSlackAlertSender.call_count = 0
    FakeAgentSlackAlertSender.sent_payloads = []

    app.dependency_overrides[
        get_current_principal
    ] = override_get_current_principal
    app.dependency_overrides[
        get_agent_execution_planner_client
    ] = override_get_execution_planner
    app.dependency_overrides[
        get_agent_dynamic_synthesis_client
    ] = override_get_dynamic_synthesis
    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_agent_github_risk_collector] = override_get_github_collector
    app.dependency_overrides[get_agent_jira_risk_collector] = override_get_jira_collector
    app.dependency_overrides[get_agent_slack_alert_sender] = override_get_slack_sender
    app.dependency_overrides[
        get_engineering_document_embedding_provider
    ] = FakeAgentEmbeddingProvider
    app.dependency_overrides[
        get_engineering_document_reranker
    ] = FakeAgentReranker

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            yield client
    finally:
        app.dependency_overrides.clear()

        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)

        await engine.dispose()


@pytest.mark.anyio
async def test_create_agent_query_plan_returns_structured_plan(
    agent_query_api_client: AsyncClient,
) -> None:
    """The API should convert natural language into a query plan."""

    response = await agent_query_api_client.post(
        "/api/v1/agent/query-plan",
        json={"query": "What are the biggest release risks this week?"},
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["intent"] == "release_risk_summary"
    assert payload["response_depth"] == "standard"
    assert payload["requires_current_snapshot"] is True
    assert payload["requires_historical_lookup"] is False
    assert payload["requires_human_approval"] is False
    assert payload["may_execute_side_effect"] is False


@pytest.mark.anyio
async def test_execute_agent_query_runs_release_risk_workflow(
    agent_query_api_client: AsyncClient,
) -> None:
    """The agent query endpoint should return a conversational risk answer."""

    response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "What are the biggest release risks this week?",
        },
    )

    assert response.status_code == 200

    payload = response.json()
    release_risk = payload["release_risk"]
    release_run = release_risk["release_run"]

    assert payload["answer"]
    assert "release risk" in payload["answer"].lower()
    assert payload["plan"]["intent"] == "release_risk_summary"
    assert payload["plan"]["response_depth"] == "standard"
    assert isinstance(payload["citations"], list)
    assert isinstance(payload["approval_required"], bool)

    assert release_run["id"] is not None
    assert release_run["run_id"].startswith("release-run-")
    assert release_run["query"] == ("What are the biggest release risks this week?")
    assert release_run["requested_by"] == "director@example.com"
    assert release_run["status"] in {
        "completed",
        "waiting_for_approval",
    }

    assert release_risk["github"]["status"] == "success"
    assert release_risk["jira"]["status"] == "success"
    assert release_risk["release_summary"]["source"] == "release"
    assert release_risk["risk_features"] is not None
    assert release_risk["risk_score"] is not None
    assert release_risk["risk_score"]["scoring_version"] == ("rule_based_release_risk_v1")

    events_response = await agent_query_api_client.get(
        f"/api/v1/release-runs/{release_run['id']}/events"
    )

    assert events_response.status_code == 200

    event_types = {event["event_type"] for event in events_response.json()["events"]}

    assert "workflow_started" in event_types
    assert "workflow_completed" in event_types
    assert "risk_features_extracted" in event_types
    assert "release_risk_scored" in event_types
    assert "release_risk_snapshot_created" in event_types


@pytest.mark.anyio
async def test_risk_score_follow_up_uses_persisted_snapshot(
    agent_query_api_client: AsyncClient,
) -> None:
    """Risk-score follow-up should not rerun GitHub or Jira collection."""

    initial_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "What are the biggest release risks this week?",
        },
    )

    assert initial_response.status_code == 200

    initial_payload = initial_response.json()
    release_run_id = initial_payload["release_risk"]["release_run"]["id"]

    github_calls = FakeAgentGitHubRiskCollector.call_count
    jira_calls = FakeAgentJiraRiskCollector.call_count

    follow_up_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "Why is the risk score high?",
            "release_run_id": release_run_id,
        },
    )

    assert follow_up_response.status_code == 200

    follow_up_payload = follow_up_response.json()

    assert follow_up_payload["plan"]["intent"] == "explain_risk_score"
    assert follow_up_payload["plan"]["response_depth"] == "deep"
    assert follow_up_payload["release_risk"]["release_run"]["id"] == release_run_id
    assert "risk score" in follow_up_payload["answer"].lower()
    assert FakeAgentGitHubRiskCollector.call_count == github_calls
    assert FakeAgentJiraRiskCollector.call_count == jira_calls


@pytest.mark.anyio
async def test_risk_score_follow_up_requires_release_run_id(
    agent_query_api_client: AsyncClient,
) -> None:
    """Risk-score follow-up must include trusted release-run context."""

    response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "Why is the risk score high?",
        },
    )

    assert response.status_code == 422

    error_payload = response.json()

    assert error_payload["error"]["code"] == "HTTP_ERROR"
    assert error_payload["error"]["message"] == (
        "A release-run ID is required for this follow-up query."
    )
    assert error_payload["run_id"] != "unknown"


@pytest.mark.anyio
async def test_execute_agent_query_rejects_unsupported_intent(
    agent_query_api_client: AsyncClient,
) -> None:
    """Unsupported intents should not enter the release workflow."""

    github_override = app.dependency_overrides.pop(get_agent_github_risk_collector)
    jira_override = app.dependency_overrides.pop(get_agent_jira_risk_collector)

    try:
        response = await agent_query_api_client.post(
            "/api/v1/agent/query",
            json={
                "query": "Write a recipe for chocolate cake.",
            },
        )
    finally:
        app.dependency_overrides[get_agent_github_risk_collector] = github_override
        app.dependency_overrides[get_agent_jira_risk_collector] = jira_override

    assert response.status_code == 422

    error_payload = response.json()["error"]

    assert error_payload["code"] == "HTTP_ERROR"
    assert error_payload["message"] == ("This agent query intent is not executable yet.")


@pytest.mark.anyio
async def test_slack_action_requires_human_approval(
    agent_query_api_client: AsyncClient,
) -> None:
    """Slack action requests must remain behind the HITL gate."""

    response = await agent_query_api_client.post(
        "/api/v1/agent/query-plan",
        json={"query": "Can you send this to Slack?"},
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["intent"] == "action_request"
    assert payload["response_depth"] == "action_confirmation"
    assert payload["requires_human_approval"] is True
    assert payload["may_execute_side_effect"] is True


@pytest.mark.anyio
async def test_unrelated_query_is_marked_out_of_scope(
    agent_query_api_client: AsyncClient,
) -> None:
    """Unrelated questions must not enter the release workflow."""

    response = await agent_query_api_client.post(
        "/api/v1/agent/query-plan",
        json={"query": "Write a recipe for chocolate cake."},
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["intent"] == "out_of_scope"
    assert payload["response_depth"] == "brief"
    assert payload["may_execute_side_effect"] is False


@pytest.mark.anyio
async def test_empty_agent_query_is_rejected(
    agent_query_api_client: AsyncClient,
) -> None:
    """Pydantic should reject an empty natural-language query."""

    response = await agent_query_api_client.post(
        "/api/v1/agent/query-plan",
        json={"query": "   "},
    )

    assert response.status_code == 422


@pytest.mark.anyio
async def test_specific_risk_follow_up_uses_persisted_snapshot(
    agent_query_api_client: AsyncClient,
) -> None:
    """A PR-specific follow-up should explain one persisted risk without recollection."""

    initial_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "What are the biggest release risks this week?",
        },
    )

    assert initial_response.status_code == 200

    initial_payload = initial_response.json()
    release_run_id = initial_payload["release_risk"]["release_run"]["id"]

    github_calls = FakeAgentGitHubRiskCollector.call_count
    jira_calls = FakeAgentJiraRiskCollector.call_count

    follow_up_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "Why is PR 42 dangerous?",
            "release_run_id": release_run_id,
        },
    )

    assert follow_up_response.status_code == 200, follow_up_response.json()

    follow_up_payload = follow_up_response.json()

    assert follow_up_payload["plan"]["intent"] == "explain_specific_risk"
    assert follow_up_payload["plan"]["response_depth"] == "deep"
    assert follow_up_payload["release_risk"]["release_run"]["id"] == release_run_id

    assert "Payment API has failing CI" in follow_up_payload["answer"]
    assert "CI failed on a release-critical payment service." in (
        follow_up_payload["answer"]
    )

    assert len(follow_up_payload["citations"]) == 1
    assert follow_up_payload["citations"][0]["source_type"] == (
        "github_pull_request"
    )
    assert follow_up_payload["citations"][0]["source_id"] == "PR-42"

    assert FakeAgentGitHubRiskCollector.call_count == github_calls
    assert FakeAgentJiraRiskCollector.call_count == jira_calls


@pytest.mark.anyio
async def test_filter_risks_follow_up_uses_persisted_snapshot(
    agent_query_api_client: AsyncClient,
) -> None:
    """Risk filtering should use the persisted snapshot without recollection."""

    initial_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "What are the biggest release risks this week?",
        },
    )

    assert initial_response.status_code == 200

    initial_payload = initial_response.json()
    release_run_id = initial_payload["release_risk"]["release_run"]["id"]

    github_calls = FakeAgentGitHubRiskCollector.call_count
    jira_calls = FakeAgentJiraRiskCollector.call_count

    follow_up_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "Show GitHub risks only.",
            "release_run_id": release_run_id,
        },
    )

    assert follow_up_response.status_code == 200, follow_up_response.json()

    follow_up_payload = follow_up_response.json()

    assert follow_up_payload["plan"]["intent"] == "filter_risks"
    assert follow_up_payload["plan"]["filters"]["sources"] == ["github"]
    assert follow_up_payload["release_risk"]["release_run"]["id"] == release_run_id

    assert "1 matching risk" in follow_up_payload["answer"]
    assert "Payment API has failing CI" in follow_up_payload["answer"]

    assert len(follow_up_payload["citations"]) == 1
    assert follow_up_payload["citations"][0]["source_type"] == (
        "github_pull_request"
    )
    assert follow_up_payload["citations"][0]["source_id"] == "PR-42"

    assert FakeAgentGitHubRiskCollector.call_count == github_calls
    assert FakeAgentJiraRiskCollector.call_count == jira_calls


@pytest.mark.anyio
async def test_github_pr_question_uses_persisted_snapshot(
    agent_query_api_client: AsyncClient,
) -> None:
    """A PR question should use persisted GitHub evidence without recollection."""

    initial_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "What are the biggest release risks this week?",
        },
    )

    assert initial_response.status_code == 200

    initial_payload = initial_response.json()
    release_run_id = initial_payload["release_risk"]["release_run"]["id"]

    github_calls = FakeAgentGitHubRiskCollector.call_count
    jira_calls = FakeAgentJiraRiskCollector.call_count

    follow_up_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "What is happening with PR 42?",
            "release_run_id": release_run_id,
        },
    )

    assert follow_up_response.status_code == 200, follow_up_response.json()

    follow_up_payload = follow_up_response.json()

    assert follow_up_payload["plan"]["intent"] == "github_pr_question"
    assert follow_up_payload["plan"]["entity_references"][
        "pull_request_numbers"
    ] == [42]
    assert follow_up_payload["release_risk"]["release_run"]["id"] == (
        release_run_id
    )

    assert "PR 42" in follow_up_payload["answer"]
    assert "Payment API has failing CI" in follow_up_payload["answer"]
    assert "CI failed on a release-critical payment service." in (
        follow_up_payload["answer"]
    )

    assert len(follow_up_payload["citations"]) == 1
    assert follow_up_payload["citations"][0]["source_type"] == (
        "github_pull_request"
    )
    assert follow_up_payload["citations"][0]["source_id"] == "PR-42"

    assert FakeAgentGitHubRiskCollector.call_count == github_calls
    assert FakeAgentJiraRiskCollector.call_count == jira_calls


@pytest.mark.anyio
async def test_jira_ticket_question_uses_persisted_snapshot(
    agent_query_api_client: AsyncClient,
) -> None:
    """A Jira question should use persisted evidence without recollection."""

    initial_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "What are the biggest release risks this week?",
        },
    )

    assert initial_response.status_code == 200

    initial_payload = initial_response.json()
    release_run_id = initial_payload["release_risk"]["release_run"]["id"]

    github_calls = FakeAgentGitHubRiskCollector.call_count
    jira_calls = FakeAgentJiraRiskCollector.call_count

    follow_up_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "What is happening with PAY-102?",
            "release_run_id": release_run_id,
        },
    )

    assert follow_up_response.status_code == 200, follow_up_response.json()

    follow_up_payload = follow_up_response.json()

    assert follow_up_payload["plan"]["intent"] == "jira_ticket_question"
    assert follow_up_payload["plan"]["entity_references"][
        "jira_issue_keys"
    ] == ["PAY-102"]
    assert follow_up_payload["release_risk"]["release_run"]["id"] == (
        release_run_id
    )

    assert "PAY-102" in follow_up_payload["answer"]
    assert "Payment release blocker" in follow_up_payload["answer"]
    assert "release blocker" in follow_up_payload["answer"].lower()
    assert "status: blocked" in follow_up_payload["answer"].lower()

    assert len(follow_up_payload["citations"]) == 1
    assert follow_up_payload["citations"][0]["source_type"] == "jira_issue"
    assert follow_up_payload["citations"][0]["source_id"] == "PAY-102"

    assert FakeAgentGitHubRiskCollector.call_count == github_calls
    assert FakeAgentJiraRiskCollector.call_count == jira_calls


@pytest.mark.anyio
async def test_workflow_status_question_uses_persisted_snapshot(
    agent_query_api_client: AsyncClient,
) -> None:
    """Workflow-status questions should use persisted state without recollection."""

    initial_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "What are the biggest release risks this week?",
        },
    )

    assert initial_response.status_code == 200

    initial_payload = initial_response.json()
    release_run_id = initial_payload["release_risk"]["release_run"]["id"]

    github_calls = FakeAgentGitHubRiskCollector.call_count
    jira_calls = FakeAgentJiraRiskCollector.call_count

    follow_up_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "What is the workflow status?",
            "release_run_id": release_run_id,
        },
    )

    assert follow_up_response.status_code == 200, follow_up_response.json()

    follow_up_payload = follow_up_response.json()

    assert follow_up_payload["plan"]["intent"] == "workflow_status_question"
    assert follow_up_payload["plan"]["response_depth"] == "brief"
    assert follow_up_payload["release_risk"]["release_run"]["id"] == (
        release_run_id
    )

    assert "Workflow status:" in follow_up_payload["answer"]
    assert "GitHub collection: success." in follow_up_payload["answer"]
    assert "Jira collection: success." in follow_up_payload["answer"]
    assert "Approval status:" in follow_up_payload["answer"]
    assert follow_up_payload["citations"] == []

    assert FakeAgentGitHubRiskCollector.call_count == github_calls
    assert FakeAgentJiraRiskCollector.call_count == jira_calls


@pytest.mark.anyio
async def test_approval_status_question_uses_latest_durable_decision(
    agent_query_api_client: AsyncClient,
) -> None:
    """Approval questions should report the latest durable HITL decision."""

    initial_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "What are the biggest release risks this week?",
        },
    )

    assert initial_response.status_code == 200

    initial_payload = initial_response.json()
    release_risk = initial_payload["release_risk"]
    release_run_id = release_risk["release_run"]["id"]
    approval_request_id = release_risk["approval_request_id"]

    assert approval_request_id is not None
    assert release_risk["approval_status"] == "pending"

    decision_response = await agent_query_api_client.post(
        (
            f"/api/v1/release-runs/{release_run_id}"
            f"/approvals/{approval_request_id}/decision"
        ),
        json={
            "approval_status": "approved",
            "decision_note": "Approved after reviewing the rollback plan.",
        },
    )

    assert decision_response.status_code == 200

    github_calls = FakeAgentGitHubRiskCollector.call_count
    jira_calls = FakeAgentJiraRiskCollector.call_count

    follow_up_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "Has this release been approved?",
            "release_run_id": release_run_id,
        },
    )

    assert follow_up_response.status_code == 200, follow_up_response.json()

    follow_up_payload = follow_up_response.json()

    assert follow_up_payload["plan"]["intent"] == "approval_status_question"
    assert follow_up_payload["plan"]["response_depth"] == "brief"
    assert follow_up_payload["release_risk"]["release_run"]["id"] == (
        release_run_id
    )

    assert "Approval status: approved." in follow_up_payload["answer"]
    assert "Decided by: director@example.com." in follow_up_payload["answer"]
    assert (
        "Decision note: Approved after reviewing the rollback plan."
        in follow_up_payload["answer"]
    )
    assert follow_up_payload["citations"] == []

    assert FakeAgentGitHubRiskCollector.call_count == github_calls
    assert FakeAgentJiraRiskCollector.call_count == jira_calls

@pytest.mark.anyio
async def test_historical_risk_question_uses_previous_persisted_releases(
    agent_query_api_client: AsyncClient,
) -> None:
    """Historical questions should use previous snapshots without recollection."""

    previous_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "What are the biggest release risks this week?",
        },
    )

    assert previous_response.status_code == 200
    previous_payload = previous_response.json()
    previous_run_id = previous_payload["release_risk"]["release_run"]["run_id"]

    current_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "What are the biggest release risks this week?",
        },
    )

    assert current_response.status_code == 200
    current_payload = current_response.json()
    current_release_run_id = current_payload["release_risk"]["release_run"]["id"]

    github_calls = FakeAgentGitHubRiskCollector.call_count
    jira_calls = FakeAgentJiraRiskCollector.call_count

    historical_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "Did this happen before?",
            "release_run_id": current_release_run_id,
        },
    )

    assert historical_response.status_code == 200, historical_response.json()

    historical_payload = historical_response.json()

    assert historical_payload["plan"]["intent"] == "historical_risk_lookup"
    assert historical_payload["plan"]["response_depth"] == "deep"
    assert historical_payload["release_risk"]["release_run"]["id"] == (
        current_release_run_id
    )

    assert (
        "Found 1 previous release with persisted risk history."
        in historical_payload["answer"]
    )
    assert previous_run_id in historical_payload["answer"]
    assert "Payment API has failing CI" in historical_payload["answer"]
    assert len(historical_payload["citations"]) >= 1

    assert FakeAgentGitHubRiskCollector.call_count == github_calls
    assert FakeAgentJiraRiskCollector.call_count == jira_calls

@pytest.mark.anyio
async def test_previous_release_comparison_uses_persisted_snapshots(
    agent_query_api_client: AsyncClient,
) -> None:
    """Comparison questions should use current and previous persisted snapshots."""

    previous_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "What are the biggest release risks this week?",
        },
    )

    assert previous_response.status_code == 200
    previous_payload = previous_response.json()
    previous_run_id = previous_payload["release_risk"]["release_run"]["run_id"]

    current_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "What are the biggest release risks this week?",
        },
    )

    assert current_response.status_code == 200
    current_payload = current_response.json()
    current_release_run_id = current_payload["release_risk"]["release_run"]["id"]

    github_calls = FakeAgentGitHubRiskCollector.call_count
    jira_calls = FakeAgentJiraRiskCollector.call_count

    comparison_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "Compare this with the previous release.",
            "release_run_id": current_release_run_id,
        },
    )

    assert comparison_response.status_code == 200, comparison_response.json()

    comparison_payload = comparison_response.json()

    assert (
        comparison_payload["plan"]["intent"]
        == "compare_with_previous_release"
    )
    assert comparison_payload["plan"]["response_depth"] == "deep"
    assert comparison_payload["release_risk"]["release_run"]["id"] == (
        current_release_run_id
    )

    assert f"Compared with {previous_run_id}" in comparison_payload["answer"]
    assert "severity remained critical" in comparison_payload["answer"]
    assert "risk score did not change" in comparison_payload["answer"]
    assert "signal count remained 5" in comparison_payload["answer"]
    assert len(comparison_payload["citations"]) >= 1

    assert FakeAgentGitHubRiskCollector.call_count == github_calls
    assert FakeAgentJiraRiskCollector.call_count == jira_calls

@pytest.mark.anyio
async def test_slack_status_question_reports_not_sent_without_recollection(
    agent_query_api_client: AsyncClient,
) -> None:
    """Slack-status questions should read durable state without recollection."""
    initial_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "What are the biggest release risks this week?",
        },
    )

    assert initial_response.status_code == 200
    initial_payload = initial_response.json()
    release_run_id = initial_payload["release_risk"]["release_run"]["id"]

    github_calls = FakeAgentGitHubRiskCollector.call_count
    jira_calls = FakeAgentJiraRiskCollector.call_count

    status_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "Was the Slack alert sent?",
            "release_run_id": release_run_id,
        },
    )

    assert status_response.status_code == 200, status_response.json()

    status_payload = status_response.json()

    assert status_payload["plan"]["intent"] == "slack_status_question"
    assert status_payload["plan"]["response_depth"] == "brief"
    assert status_payload["release_risk"]["release_run"]["id"] == release_run_id
    assert status_payload["answer"] == (
        "No Slack alert has been sent for this release run."
    )
    assert status_payload["citations"] == []

    assert FakeAgentGitHubRiskCollector.call_count == github_calls
    assert FakeAgentJiraRiskCollector.call_count == jira_calls

@pytest.mark.anyio
async def test_slack_status_question_reports_sent_alert(
    agent_query_api_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Slack-status questions should report a durable successful delivery."""
    from types import SimpleNamespace

    from app.repositories.release_run_slack_alert_repository import (
        ReleaseRunSlackAlertRepository,
    )

    initial_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "What are the biggest release risks this week?",
        },
    )

    assert initial_response.status_code == 200
    initial_payload = initial_response.json()
    release_run_id = initial_payload["release_risk"]["release_run"]["id"]

    async def fake_get_by_release_run_id(
        self: ReleaseRunSlackAlertRepository,
        queried_release_run_id: object,
    ) -> object:
        """Return a deterministic persisted Slack delivery record."""
        del self
        assert str(queried_release_run_id) == release_run_id

        return SimpleNamespace(
            id="slack-alert-123",
            delivery_status="sent",
            slack_channel="C1234567890",
            slack_timestamp="12345.6789",
            risk_level="critical",
            risk_score=0.9475,
            recommended_action="block_release",
            created_at="2026-07-15T19:00:00Z",
        )

    monkeypatch.setattr(
        ReleaseRunSlackAlertRepository,
        "get_by_release_run_id",
        fake_get_by_release_run_id,
    )

    github_calls = FakeAgentGitHubRiskCollector.call_count
    jira_calls = FakeAgentJiraRiskCollector.call_count

    status_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "Was the Slack alert sent?",
            "release_run_id": release_run_id,
        },
    )

    assert status_response.status_code == 200, status_response.json()

    status_payload = status_response.json()

    assert status_payload["plan"]["intent"] == "slack_status_question"
    assert "Slack alert status: sent." in status_payload["answer"]
    assert "Channel: C1234567890." in status_payload["answer"]
    assert "Risk level: critical." in status_payload["answer"]
    assert "Risk score: 95%." in status_payload["answer"]
    assert "Recommended action: block release." in status_payload["answer"]
    assert status_payload["citations"] == []

    assert FakeAgentGitHubRiskCollector.call_count == github_calls
    assert FakeAgentJiraRiskCollector.call_count == jira_calls

@pytest.mark.anyio
async def test_similar_past_release_uses_persisted_snapshots(
    agent_query_api_client: AsyncClient,
) -> None:
    """Similar-release questions should rank persisted historical snapshots."""
    previous_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "What are the biggest release risks this week?",
        },
    )

    assert previous_response.status_code == 200
    previous_payload = previous_response.json()
    previous_run_id = previous_payload["release_risk"]["release_run"]["run_id"]

    current_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "What are the biggest release risks this week?",
        },
    )

    assert current_response.status_code == 200
    current_payload = current_response.json()
    current_release_run_id = current_payload["release_risk"]["release_run"]["id"]

    github_calls = FakeAgentGitHubRiskCollector.call_count
    jira_calls = FakeAgentJiraRiskCollector.call_count

    similar_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "Which past release was most similar to this one?",
            "release_run_id": current_release_run_id,
        },
    )

    assert similar_response.status_code == 200, similar_response.json()

    similar_payload = similar_response.json()

    assert similar_payload["plan"]["intent"] == "similar_past_release"
    assert similar_payload["plan"]["response_depth"] == "deep"
    assert similar_payload["release_risk"]["release_run"]["id"] == (
        current_release_run_id
    )

    assert previous_run_id in similar_payload["answer"]
    assert "100% similarity" in similar_payload["answer"]
    assert "Payment API has failing CI" in similar_payload["answer"]
    assert len(similar_payload["citations"]) >= 1

    assert FakeAgentGitHubRiskCollector.call_count == github_calls
    assert FakeAgentJiraRiskCollector.call_count == jira_calls

@pytest.mark.anyio
async def test_slack_action_sends_approved_persisted_release(
    agent_query_api_client: AsyncClient,
) -> None:
    """Natural-language Slack action should use approved persisted context."""
    initial_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={"query": "What are the biggest release risks this week?"},
    )

    assert initial_response.status_code == 200

    release_risk = initial_response.json()["release_risk"]
    release_run_id = release_risk["release_run"]["id"]
    approval_request_id = release_risk["approval_request_id"]

    assert approval_request_id is not None

    decision_response = await agent_query_api_client.post(
        (
            f"/api/v1/release-runs/{release_run_id}"
            f"/approvals/{approval_request_id}/decision"
        ),
        json={
            "approval_status": "approved",
            "decision_note": "Approved after reviewing rollback plan.",
        },
    )

    assert decision_response.status_code == 200

    github_calls = FakeAgentGitHubRiskCollector.call_count
    jira_calls = FakeAgentJiraRiskCollector.call_count

    action_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "Can you send this to Slack?",
            "release_run_id": release_run_id,
        },
    )

    assert action_response.status_code == 200, action_response.json()

    action_payload = action_response.json()

    assert action_payload["plan"]["intent"] == "action_request"
    assert action_payload["plan"]["response_depth"] == "action_confirmation"
    assert "Slack alert sent successfully." in action_payload["answer"]
    assert "Channel: C1234567890." in action_payload["answer"]
    assert action_payload["release_risk"]["release_run"]["id"] == release_run_id
    assert FakeAgentSlackAlertSender.call_count == 1
    assert len(FakeAgentSlackAlertSender.sent_payloads) == 1

    assert FakeAgentGitHubRiskCollector.call_count == github_calls
    assert FakeAgentJiraRiskCollector.call_count == jira_calls

@pytest.mark.anyio
async def test_slack_action_is_blocked_before_human_approval(
    agent_query_api_client: AsyncClient,
) -> None:
    """Natural-language Slack action must remain behind the HITL gate."""
    initial_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={"query": "What are the biggest release risks this week?"},
    )

    assert initial_response.status_code == 200

    release_run_id = initial_response.json()["release_risk"]["release_run"]["id"]

    action_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "Can you send this to Slack?",
            "release_run_id": release_run_id,
        },
    )

    assert action_response.status_code == 409
    assert "Slack alert cannot be sent before approval" in action_response.text
    assert FakeAgentSlackAlertSender.call_count == 0


@pytest.mark.anyio
async def test_slack_action_blocks_duplicate_delivery(
    agent_query_api_client: AsyncClient,
) -> None:
    """Repeated natural-language Slack actions must not send twice."""
    initial_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={"query": "What are the biggest release risks this week?"},
    )

    assert initial_response.status_code == 200

    release_risk = initial_response.json()["release_risk"]
    release_run_id = release_risk["release_run"]["id"]
    approval_request_id = release_risk["approval_request_id"]

    decision_response = await agent_query_api_client.post(
        (
            f"/api/v1/release-runs/{release_run_id}"
            f"/approvals/{approval_request_id}/decision"
        ),
        json={
            "approval_status": "approved",
            "decision_note": "Approved after reviewing rollback plan.",
        },
    )

    assert decision_response.status_code == 200

    first_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "Can you send this to Slack?",
            "release_run_id": release_run_id,
        },
    )
    second_response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "Can you send this to Slack?",
            "release_run_id": release_run_id,
        },
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 409
    assert "Slack alert already sent" in second_response.text
    assert FakeAgentSlackAlertSender.call_count == 1


@pytest.mark.anyio
async def test_knowledge_document_question_uses_ingested_documents(
    agent_query_api_client: AsyncClient,
) -> None:
    """Knowledge questions should retrieve documents without running collectors."""
    ingestion_response = await agent_query_api_client.post(
        "/api/v1/engineering-documents/ingest",
        json={
            "title": "Payment Service Production Runbook",
            "source_type": "runbook",
            "source_uri": "docs/payment-service-runbook.md",
            "raw_content": (
                "The payment service must be rolled back when payment failures "
                "exceed the production threshold. After rollback, validate "
                "authorization success and confirm the error rate has recovered."
            ),
            "metadata_json": {
                "service": "payment-service",
                "owner": "Payments Platform Team",
            },
        },
    )

    assert ingestion_response.status_code == 201, ingestion_response.json()

    response = await agent_query_api_client.post(
        "/api/v1/agent/query",
        json={
            "query": "What does the payment service runbook say about rollback?",
        },
    )

    assert response.status_code == 200, response.json()

    payload = response.json()

    assert payload["plan"]["intent"] == "knowledge_doc_question"
    assert payload["release_risk"] is None
    assert payload["approval_required"] is False
    assert "Payment Service Production Runbook" in payload["answer"]
    assert "rolled back" in payload["answer"]
    assert len(payload["citations"]) >= 1
    assert payload["citations"][0]["source"] == "knowledge"
    assert payload["citations"][0]["source_type"] == "runbook"
    assert (
        payload["citations"][0]["source_url"]
        == "docs/payment-service-runbook.md"
    )
    assert FakeAgentGitHubRiskCollector.call_count == 0
    assert FakeAgentJiraRiskCollector.call_count == 0



@pytest.mark.anyio
async def test_dynamic_knowledge_query_executes_read_only_tool_plan(
    agent_query_api_client: AsyncClient,
) -> None:
    """Dynamic queries should plan and execute trusted knowledge retrieval."""
    ingestion_response = await agent_query_api_client.post(
        "/api/v1/engineering-documents/ingest",
        json={
            "title": "Payment Service Dynamic Runbook",
            "source_type": "runbook",
            "source_uri": "docs/payment-dynamic-runbook.md",
            "raw_content": (
                "Rollback the payment service when authorization failures "
                "exceed the release threshold. Validate transaction success "
                "and error-rate recovery after rollback."
            ),
            "metadata_json": {
                "service": "payment-service",
            },
        },
    )

    assert ingestion_response.status_code == 201

    response = await agent_query_api_client.post(
        "/api/v1/agent/query-dynamic",
        json={
            "query": "What does the payment service runbook say about rollback?",
        },
    )

    assert response.status_code == 200, response.json()

    payload = response.json()
    tool_result = payload["execution_result"]["tool_results"][0]

    assert payload["query_plan"]["intent"] == "knowledge_doc_question"
    assert payload["execution_plan"]["steps"][0]["invocation"][
        "tool_name"
    ] == "search_engineering_knowledge"
    assert payload["execution_result"]["status"] == "success"
    assert tool_result["status"] == "success"
    assert tool_result["output"]["result_count"] >= 1
    assert tool_result["evidence"][0]["source_type"] == (
        "engineering_document_chunk"
    )
    assert payload["prompt_version"] == "agent-execution-planner-v1"
    assert payload["model"] == "test-claude-model"
    assert "roll back" in payload["answer"]["answer"].lower()
    assert payload["answer"]["citations"][0]["source_type"] == (
        "engineering_document_chunk"
    )
    assert payload["synthesis_prompt_version"] == (
        "agent-dynamic-synthesis-v1"
    )
    assert payload["synthesis_model"] == "test-claude-model"
    assert FakeAgentExecutionPlannerClient.call_count == 1
    assert FakeAgentDynamicSynthesisClient.call_count == 1
    assert FakeAgentGitHubRiskCollector.call_count == 0
    assert FakeAgentJiraRiskCollector.call_count == 0


@pytest.mark.anyio
async def test_dynamic_query_fails_closed_when_cost_limit_is_exceeded(
    agent_query_api_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The endpoint should hide results that exceed configured cost policy."""
    monkeypatch.setenv(
        "AGENT_DYNAMIC_PLANNER_INPUT_COST_PER_MILLION_USD",
        "1000",
    )
    monkeypatch.setenv(
        "AGENT_DYNAMIC_PLANNER_OUTPUT_COST_PER_MILLION_USD",
        "1000",
    )
    monkeypatch.setenv(
        "AGENT_DYNAMIC_SYNTHESIS_INPUT_COST_PER_MILLION_USD",
        "1000",
    )
    monkeypatch.setenv(
        "AGENT_DYNAMIC_SYNTHESIS_OUTPUT_COST_PER_MILLION_USD",
        "1000",
    )
    monkeypatch.setenv(
        "AGENT_DYNAMIC_MAX_ESTIMATED_COST_USD",
        "0.001",
    )
    get_settings.cache_clear()

    try:
        response = await agent_query_api_client.post(
            "/api/v1/agent/query-dynamic",
            json={
                "query": (
                    "What does the payment service runbook say "
                    "about rollback?"
                ),
            },
        )
    finally:
        get_settings.cache_clear()

    assert response.status_code == 422
    assert (
        "Dynamic agent query exceeded the configured cost limit."
        in response.text
    )
    assert FakeAgentExecutionPlannerClient.call_count == 1
    assert FakeAgentDynamicSynthesisClient.call_count == 1



@pytest.mark.anyio
async def test_dynamic_query_returns_503_when_planning_is_disabled(
    agent_query_api_client: AsyncClient,
) -> None:
    """The dynamic endpoint should fail closed when planning is disabled."""

    async def override_disabled_planner() -> None:
        """Represent a disabled dynamic-planning dependency."""

        return None

    app.dependency_overrides[
        get_agent_execution_planner_client
    ] = override_disabled_planner

    response = await agent_query_api_client.post(
        "/api/v1/agent/query-dynamic",
        json={
            "query": (
                "What does the payment service runbook say about rollback?"
            ),
        },
    )

    assert response.status_code == 503
    assert "Dynamic agent planning is disabled" in response.text
    assert FakeAgentExecutionPlannerClient.call_count == 0


@pytest.mark.anyio
async def test_dynamic_query_blocks_fresh_release_analysis(
    agent_query_api_client: AsyncClient,
) -> None:
    """Fresh analysis must stay outside the read-only dynamic endpoint."""

    response = await agent_query_api_client.post(
        "/api/v1/agent/query-dynamic",
        json={
            "query": "What are the biggest release risks this week?",
        },
    )

    assert response.status_code == 422
    assert (
        "not available through read-only dynamic execution"
        in response.text
    )
    assert FakeAgentExecutionPlannerClient.call_count == 0
    assert FakeAgentGitHubRiskCollector.call_count == 0
    assert FakeAgentJiraRiskCollector.call_count == 0


@pytest.mark.anyio
async def test_dynamic_query_blocks_slack_side_effect_intent(
    agent_query_api_client: AsyncClient,
) -> None:
    """Slack actions must remain on the durable HITL-controlled endpoint."""

    response = await agent_query_api_client.post(
        "/api/v1/agent/query-dynamic",
        json={
            "query": "Can you send this to Slack?",
            "release_run_id": (
                "11111111-1111-1111-1111-111111111111"
            ),
        },
    )

    assert response.status_code == 422
    assert (
        "not available through read-only dynamic execution"
        in response.text
    )
    assert FakeAgentExecutionPlannerClient.call_count == 0
    assert FakeAgentSlackAlertSender.call_count == 0



@pytest.mark.anyio
async def test_dynamic_query_hides_invalid_synthesis_output(
    agent_query_api_client: AsyncClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Grounding failures must return a safe response without Claude output."""
    ingestion_response = await agent_query_api_client.post(
        "/api/v1/engineering-documents/ingest",
        json={
            "title": "Payment Grounding Failure Runbook",
            "source_type": "runbook",
            "source_uri": "docs/payment-grounding-failure.md",
            "raw_content": (
                "Rollback the payment service after the documented "
                "authorization failure threshold is exceeded."
            ),
            "metadata_json": {"service": "payment-service"},
        },
    )

    assert ingestion_response.status_code == 201

    FakeAgentDynamicSynthesisClient.return_invalid_grounding = True
    caplog.set_level(
        "ERROR",
        logger="app.api.routes.agent_queries",
    )

    response = await agent_query_api_client.post(
        "/api/v1/agent/query-dynamic",
        json={
            "query": "What does the payment runbook say about rollback?",
        },
    )

    assert response.status_code == 502
    assert (
        "Dynamic agent synthesis failed grounding verification."
        in response.text
    )
    assert "UNTRUSTED_CLAUDE_OUTPUT_SHOULD_NOT_BE_EXPOSED" not in response.text
    assert FakeAgentDynamicSynthesisClient.call_count == 1

    failure_records = [
        record
        for record in caplog.records
        if record.getMessage()
        == "agent_dynamic_synthesis_grounding_verification_failed"
    ]

    assert len(failure_records) == 1
    failure_record = failure_records[0]
    assert getattr(failure_record, "intent", None) == (
        "knowledge_doc_question"
    )
    assert getattr(failure_record, "error_type", None) == (
        "AgentDynamicSynthesisCitationVerificationError"
    )
    assert (
        "UNTRUSTED_CLAUDE_OUTPUT_SHOULD_NOT_BE_EXPOSED"
        not in caplog.text
    )
