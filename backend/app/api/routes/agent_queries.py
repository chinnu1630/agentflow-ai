"""FastAPI routes for natural-language AgentFlow queries."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.session import get_db_session
from app.integrations.anthropic_client import (
    AnthropicClientConfig,
    AnthropicClientRateLimitError,
    AnthropicClientResponseError,
    AnthropicClientTimeoutError,
    AnthropicClientUnavailableError,
)
from app.integrations.anthropic_dynamic_synthesis_client import (
    AnthropicDynamicSynthesisClient,
)
from app.integrations.anthropic_execution_planner_client import (
    AnthropicExecutionPlannerClient,
)
from app.integrations.github_client import GitHubClient, GitHubClientConfig
from app.integrations.jira_client import JiraClient, JiraClientConfig
from app.integrations.slack_client import SlackClient, SlackClientConfig
from app.repositories.engineering_document_repository import (
    EngineeringDocumentRepository,
    EngineeringDocumentRepositoryError,
)
from app.repositories.release_run_approval_repository import (
    ReleaseRunApprovalRepository,
    ReleaseRunApprovalRepositoryError,
)
from app.repositories.release_run_event_repository import (
    ReleaseRunEventRepository,
    ReleaseRunEventRepositoryError,
)
from app.repositories.release_run_repository import (
    ReleaseRunRepository,
    ReleaseRunRepositoryError,
)
from app.repositories.release_run_risk_snapshot_repository import (
    ReleaseRunRiskSnapshotRepository,
    ReleaseRunRiskSnapshotRepositoryError,
)
from app.repositories.release_run_slack_alert_repository import (
    ReleaseRunSlackAlertAlreadySentError,
    ReleaseRunSlackAlertRepository,
    ReleaseRunSlackAlertRepositoryError,
)
from app.schemas.agent_dynamic_query import AgentDynamicQueryResponse
from app.schemas.agent_query import (
    AgentIntent,
    AgentQueryPlan,
    AgentQueryRequest,
    AgentQueryResponse,
)
from app.schemas.github import GitHubRepositoryConfig
from app.services.agent_dynamic_execution_service import (
    AgentDynamicExecutionService,
)
from app.services.agent_dynamic_query_service import AgentDynamicQueryService
from app.services.agent_dynamic_synthesis_citation_verifier import (
    AgentDynamicSynthesisCitationVerificationError,
)
from app.services.agent_dynamic_synthesis_service import (
    AgentDynamicSynthesisService,
)
from app.services.agent_execution_plan_validator import (
    AgentExecutionPlanValidationError,
    AgentExecutionPlanValidator,
)
from app.services.agent_execution_planner_service import (
    AgentExecutionPlannerService,
    AgentExecutionPlannerServiceError,
)
from app.services.agent_github_pr_resolver import (
    AgentGitHubPRNotFoundError,
    AgentGitHubPRResolver,
)
from app.services.agent_jira_ticket_resolver import (
    AgentJiraTicketNotFoundError,
    AgentJiraTicketResolver,
)
from app.services.agent_llm_cost_estimator import (
    AgentLLMCostEstimator,
    AgentLLMCostRates,
)
from app.services.agent_query_context_resolver import (
    AgentQueryContextConflictError,
    AgentQueryContextRequiredError,
    AgentQueryContextResolver,
    AgentQueryContextResolverError,
    AgentQuerySnapshotNotFoundError,
    AgentQuerySnapshotValidationError,
)
from app.services.agent_query_executor import (
    AgentQueryContextMismatchError,
    AgentQueryExecutor,
    AgentQueryResultError,
    UnsupportedAgentQueryIntentError,
)
from app.services.agent_query_router import AgentQueryRouter
from app.services.agent_read_only_tool_adapters import (
    build_read_only_tool_adapters,
)
from app.services.agent_response_composer import AgentResponseComposer
from app.services.agent_risk_filter import AgentRiskFilter
from app.services.agent_similar_release_matcher import (
    AgentSimilarReleaseMatcher,
)
from app.services.agent_specific_risk_matcher import (
    AgentSpecificRiskMatcher,
    AgentSpecificRiskNotFoundError,
)
from app.services.agent_tool_registry import AgentToolRegistry
from app.services.engineering_document_embedding_provider import (
    SentenceTransformerEmbeddingProvider,
    get_engineering_document_embedding_provider,
)
from app.services.engineering_document_reranker import (
    CrossEncoderEngineeringDocumentReranker,
    get_engineering_document_reranker,
)
from app.services.engineering_document_retrieval_service import (
    EngineeringDocumentRetrievalRequest,
    EngineeringDocumentRetrievalService,
)
from app.services.github_risk_collector import RiskCollector
from app.services.jira_risk_collector import JiraRiskCollector
from app.services.release_risk_execution_finalizer import (
    ReleaseRiskExecutionFinalizer,
)
from app.services.release_run_service import (
    ReleaseRunService,
    ReleaseRunServiceError,
)
from app.services.slack_release_alert_action_service import (
    SlackReleaseAlertActionService,
)
from app.services.slack_release_alert_service import (
    SlackReleaseAlertNotApprovedError,
    SlackReleaseAlertServiceError,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/agent",
    tags=["agent"],
)


def get_agent_query_router() -> AgentQueryRouter:
    """Return the deterministic AgentFlow query router."""

    return AgentQueryRouter()


async def get_agent_execution_planner_client(
    request: Request,
) -> AsyncIterator[AnthropicExecutionPlannerClient | None]:
    """Create the optional bounded Claude execution-planner client."""

    settings = get_settings()

    if not settings.agent_dynamic_planning_enabled:
        yield None
        return

    api_key = settings.anthropic_api_key

    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Dynamic agent planning is enabled but not configured. "
                "Set ANTHROPIC_API_KEY."
            ),
        )

    request_id = str(
        getattr(request.state, "request_id", "unknown-request-id")
    )
    config = AnthropicClientConfig(
        api_key=api_key,
        model=(
            settings.agent_dynamic_planner_model
            or settings.anthropic_model
        ),
        max_tokens=settings.agent_dynamic_planner_max_tokens,
        timeout_seconds=settings.anthropic_timeout_seconds,
        max_retries=settings.anthropic_max_retries,
    )

    async with AnthropicExecutionPlannerClient(
        config=config,
        run_id=request_id,
    ) as planner_client:
        yield planner_client


AgentExecutionPlannerClientDependency = Annotated[
    AnthropicExecutionPlannerClient | None,
    Depends(get_agent_execution_planner_client),
]


async def get_agent_dynamic_synthesis_client(
    request: Request,
) -> AsyncIterator[AnthropicDynamicSynthesisClient | None]:
    """Create the optional Claude dynamic-answer synthesis client."""

    settings = get_settings()

    if not settings.agent_dynamic_planning_enabled:
        yield None
        return

    api_key = settings.anthropic_api_key

    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Dynamic agent synthesis is enabled but not configured. "
                "Set ANTHROPIC_API_KEY."
            ),
        )

    request_id = str(
        getattr(request.state, "request_id", "unknown-request-id")
    )
    config = AnthropicClientConfig(
        api_key=api_key,
        model=(
            settings.agent_dynamic_synthesis_model
            or settings.anthropic_model
        ),
        max_tokens=settings.agent_dynamic_synthesis_max_tokens,
        timeout_seconds=settings.anthropic_timeout_seconds,
        max_retries=settings.anthropic_max_retries,
    )

    async with AnthropicDynamicSynthesisClient(
        config=config,
        run_id=request_id,
    ) as synthesis_client:
        yield synthesis_client


AgentDynamicSynthesisClientDependency = Annotated[
    AnthropicDynamicSynthesisClient | None,
    Depends(get_agent_dynamic_synthesis_client),
]


AgentQueryRouterDependency = Annotated[
    AgentQueryRouter,
    Depends(get_agent_query_router),
]


async def get_executable_agent_query_plan(
    payload: AgentQueryRequest,
    query_router: AgentQueryRouterDependency,
) -> AgentQueryPlan:
    """Create and validate a plan before external dependencies are resolved."""

    plan = await query_router.create_plan(payload)

    if plan.intent not in {
        AgentIntent.RELEASE_RISK_SUMMARY,
        AgentIntent.EXPLAIN_RISK_SCORE,
        AgentIntent.EXPLAIN_SPECIFIC_RISK,
        AgentIntent.FILTER_RISKS,
        AgentIntent.GITHUB_PR_QUESTION,
        AgentIntent.JIRA_TICKET_QUESTION,
        AgentIntent.WORKFLOW_STATUS_QUESTION,
        AgentIntent.APPROVAL_STATUS_QUESTION,
        AgentIntent.SLACK_STATUS_QUESTION,
        AgentIntent.HISTORICAL_RISK_LOOKUP,
        AgentIntent.SIMILAR_PAST_RELEASE,
        AgentIntent.COMPARE_WITH_PREVIOUS_RELEASE,
        AgentIntent.ACTION_REQUEST,
        AgentIntent.KNOWLEDGE_DOC_QUESTION,
    }:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="This agent query intent is not executable yet.",
        )

    return plan


ExecutableAgentQueryPlanDependency = Annotated[
    AgentQueryPlan,
    Depends(get_executable_agent_query_plan),
]


async def get_agent_github_risk_collector(
    request: Request,
    plan: ExecutableAgentQueryPlanDependency,
) -> AsyncIterator[RiskCollector | None]:
    """Create GitHub collector only when fresh collection is required."""

    if plan.intent is not AgentIntent.RELEASE_RISK_SUMMARY:
        yield None
        return

    request_id = str(getattr(request.state, "request_id", "unknown-request-id"))
    settings = get_settings()

    if not settings.github_repository_owner or not settings.github_repository_name:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GitHub risk collection is not configured.",
        )

    repository_config = GitHubRepositoryConfig(
        owner=settings.github_repository_owner,
        repo=settings.github_repository_name,
        default_branch=settings.github_default_branch,
    )
    github_config = GitHubClientConfig(
        repository=repository_config,
        token=settings.github_token,
    )

    async with httpx.AsyncClient() as http_client:
        github_client = GitHubClient(
            http_client=http_client,
            config=github_config,
            request_id=request_id,
        )
        yield RiskCollector(github_client=github_client)


async def get_agent_jira_risk_collector(
    plan: ExecutableAgentQueryPlanDependency,
) -> AsyncIterator[JiraRiskCollector | None]:
    """Create Jira collector only when fresh collection is required."""

    if plan.intent is not AgentIntent.RELEASE_RISK_SUMMARY:
        yield None
        return

    settings = get_settings()

    if (
        settings.jira_base_url is None
        or settings.jira_email is None
        or settings.jira_api_token is None
        or settings.jira_project_key is None
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Jira risk collection is not configured.",
        )

    jira_config = JiraClientConfig(
        base_url=settings.jira_base_url,
        email=settings.jira_email,
        api_token=settings.jira_api_token,
        project_key=settings.jira_project_key,
    )

    async with JiraClient(config=jira_config) as jira_client:
        yield JiraRiskCollector(jira_client=jira_client)


AgentGitHubRiskCollectorDependency = Annotated[
    RiskCollector | None,
    Depends(get_agent_github_risk_collector),
]
AgentJiraRiskCollectorDependency = Annotated[
    JiraRiskCollector | None,
    Depends(get_agent_jira_risk_collector),
]


async def get_agent_slack_alert_sender(
    request: Request,
    plan: ExecutableAgentQueryPlanDependency,
) -> AsyncIterator[SlackClient | None]:
    """Create a Slack sender only for approved action requests."""

    if plan.intent is not AgentIntent.ACTION_REQUEST:
        yield None
        return

    request_id = str(
        getattr(request.state, "request_id", "unknown-request-id")
    )
    settings = get_settings()

    if not settings.slack_bot_token or not settings.slack_channel_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Slack alert delivery is not configured.",
        )

    slack_config = SlackClientConfig(
        bot_token=settings.slack_bot_token,
        channel_id=settings.slack_channel_id,
    )

    async with httpx.AsyncClient() as http_client:
        yield SlackClient(
            http_client=http_client,
            config=slack_config,
            request_id=request_id,
        )


AgentSlackAlertSenderDependency = Annotated[
    SlackClient | None,
    Depends(get_agent_slack_alert_sender),
]


@router.post(
    "/query-plan",
    response_model=AgentQueryPlan,
    status_code=status.HTTP_200_OK,
)
async def create_agent_query_plan(
    payload: AgentQueryRequest,
    request: Request,
    query_router: AgentQueryRouterDependency,
) -> AgentQueryPlan:
    """Convert a natural-language question into a safe workflow plan."""

    plan = await query_router.create_plan(payload)

    logger.info(
        "agent_query_plan_created",
        extra={
            "run_id": getattr(request.state, "run_id", None),
            "intent": plan.intent.value,
            "response_depth": plan.response_depth.value,
            "confidence": plan.confidence,
            "requires_current_snapshot": (plan.requires_current_snapshot),
            "requires_historical_lookup": (plan.requires_historical_lookup),
            "requires_human_approval": (plan.requires_human_approval),
            "may_execute_side_effect": plan.may_execute_side_effect,
            "query_character_count": len(payload.query),
        },
    )

    return plan


@router.post(
    "/query-dynamic",
    response_model=AgentDynamicQueryResponse,
    status_code=status.HTTP_200_OK,
)
async def execute_dynamic_agent_query(
    payload: AgentQueryRequest,
    request: Request,
    plan: ExecutableAgentQueryPlanDependency,
    planner_client: AgentExecutionPlannerClientDependency,
    synthesis_client: AgentDynamicSynthesisClientDependency,
    session: AsyncSession = Depends(get_db_session),
    embedding_provider: SentenceTransformerEmbeddingProvider = Depends(
        get_engineering_document_embedding_provider
    ),
    reranker: CrossEncoderEngineeringDocumentReranker = Depends(
        get_engineering_document_reranker
    ),
) -> AgentDynamicQueryResponse:
    """Execute a bounded read-only dynamic agent plan."""

    if planner_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dynamic agent planning is disabled.",
        )

    if synthesis_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dynamic agent synthesis is disabled.",
        )

    if plan.intent in {
        AgentIntent.RELEASE_RISK_SUMMARY,
        AgentIntent.ACTION_REQUEST,
    }:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "This intent is not available through read-only "
                "dynamic execution."
            ),
        )

    request_id = str(
        getattr(request.state, "request_id", "unknown-request-id")
    )

    risk_snapshot_repository = ReleaseRunRiskSnapshotRepository(
        session=session,
        request_id=request_id,
    )
    approval_repository = ReleaseRunApprovalRepository(
        session=session,
        request_id=request_id,
    )
    slack_alert_repository = ReleaseRunSlackAlertRepository(
        session=session,
        request_id=request_id,
    )
    engineering_document_repository = EngineeringDocumentRepository(
        session=session,
    )
    context_resolver = AgentQueryContextResolver(
        snapshot_repository=risk_snapshot_repository,
        request_id=request_id,
    )
    knowledge_service = EngineeringDocumentRetrievalService(
        repository=engineering_document_repository,
        embedding_provider=embedding_provider,
        reranker=reranker,
    )
    registry = AgentToolRegistry()
    plan_validator = AgentExecutionPlanValidator(
        registry=registry,
        request_id=request_id,
    )
    planner = AgentExecutionPlannerService(
        planner_client=planner_client,
        plan_validator=plan_validator,
        request_id=request_id,
    )
    adapters = build_read_only_tool_adapters(
        request=payload,
        query_plan=plan,
        context_resolver=context_resolver,
        knowledge_retrieval_service=knowledge_service,
        approval_repository=approval_repository,
        slack_alert_repository=slack_alert_repository,
        request_id=request_id,
    )
    executor = AgentDynamicExecutionService(
        adapters=adapters,
        plan_validator=plan_validator,
        request_id=request_id,
    )
    synthesizer = AgentDynamicSynthesisService(
        client=synthesis_client,
        request_id=request_id,
    )
    settings = get_settings()
    cost_estimator = AgentLLMCostEstimator(
        rates=AgentLLMCostRates(
            planning_input_per_million_usd=(
                settings.agent_dynamic_planner_input_cost_per_million_usd
            ),
            planning_output_per_million_usd=(
                settings.agent_dynamic_planner_output_cost_per_million_usd
            ),
            synthesis_input_per_million_usd=(
                settings.agent_dynamic_synthesis_input_cost_per_million_usd
            ),
            synthesis_output_per_million_usd=(
                settings.agent_dynamic_synthesis_output_cost_per_million_usd
            ),
        )
    )
    service = AgentDynamicQueryService(
        planner=planner,
        executor=executor,
        synthesizer=synthesizer,
        request_id=request_id,
        cost_estimator=cost_estimator,
    )

    try:
        response = await service.execute(
            request=payload,
            query_plan=plan,
        )
        await session.commit()
        return response

    except AgentDynamicSynthesisCitationVerificationError as exc:
        await session.rollback()
        logger.error(
            "agent_dynamic_synthesis_grounding_verification_failed",
            extra={
                "run_id": request_id,
                "intent": plan.intent.value,
                "error_type": type(exc).__name__,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "Dynamic agent synthesis failed grounding verification."
            ),
        ) from exc

    except AnthropicClientTimeoutError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Dynamic agent planning timed out.",
        ) from exc

    except AnthropicClientRateLimitError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Dynamic agent planning was rate limited.",
        ) from exc

    except AnthropicClientUnavailableError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dynamic agent planning is unavailable.",
        ) from exc

    except (
        AnthropicClientResponseError,
        AgentExecutionPlannerServiceError,
        AgentExecutionPlanValidationError,
    ) as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Dynamic agent planning returned an invalid plan.",
        ) from exc

    except (
        EngineeringDocumentRepositoryError,
        ReleaseRunApprovalRepositoryError,
        ReleaseRunRiskSnapshotRepositoryError,
        ReleaseRunSlackAlertRepositoryError,
        SQLAlchemyError,
    ) as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute the dynamic AgentFlow query.",
        ) from exc


@router.post(
    "/query",
    response_model=AgentQueryResponse,
    status_code=status.HTTP_200_OK,
)
async def execute_agent_query(
    payload: AgentQueryRequest,
    request: Request,
    plan: ExecutableAgentQueryPlanDependency,
    risk_collector: AgentGitHubRiskCollectorDependency,
    jira_risk_collector: AgentJiraRiskCollectorDependency,
    slack_sender: AgentSlackAlertSenderDependency,
    session: AsyncSession = Depends(get_db_session),
    embedding_provider: SentenceTransformerEmbeddingProvider = Depends(
        get_engineering_document_embedding_provider
    ),
    reranker: CrossEncoderEngineeringDocumentReranker = Depends(
        get_engineering_document_reranker
    ),
) -> AgentQueryResponse:
    """Execute a fresh query or answer from trusted persisted context."""

    request_id = str(getattr(request.state, "request_id", "unknown-request-id"))

    risk_snapshot_repository = ReleaseRunRiskSnapshotRepository(
        session=session,
        request_id=request_id,
    )
    approval_repository = ReleaseRunApprovalRepository(
        session=session,
        request_id=request_id,
    )
    slack_alert_repository = ReleaseRunSlackAlertRepository(
        session=session,
        request_id=request_id,
    )
    event_repository = ReleaseRunEventRepository(
        session=session,
        request_id=request_id,
    )
    response_composer = AgentResponseComposer(request_id=request_id)

    try:
        if plan.intent is AgentIntent.KNOWLEDGE_DOC_QUESTION:
            engineering_document_repository = EngineeringDocumentRepository(
                session=session,
            )
            knowledge_service = EngineeringDocumentRetrievalService(
                repository=engineering_document_repository,
                embedding_provider=embedding_provider,
                reranker=reranker,
            )
            retrieval = await knowledge_service.retrieve_relevant_chunks(
                EngineeringDocumentRetrievalRequest(
                    query=payload.query,
                    top_k=5,
                ),
                run_id=request_id,
            )
            agent_response = response_composer.compose_knowledge_document(
                plan=plan,
                retrieval=retrieval,
            )

            await session.commit()
            return agent_response

        if plan.intent in {
            AgentIntent.EXPLAIN_RISK_SCORE,
            AgentIntent.EXPLAIN_SPECIFIC_RISK,
            AgentIntent.FILTER_RISKS,
            AgentIntent.GITHUB_PR_QUESTION,
            AgentIntent.JIRA_TICKET_QUESTION,
            AgentIntent.WORKFLOW_STATUS_QUESTION,
            AgentIntent.APPROVAL_STATUS_QUESTION,
            AgentIntent.SLACK_STATUS_QUESTION,
            AgentIntent.HISTORICAL_RISK_LOOKUP,
            AgentIntent.SIMILAR_PAST_RELEASE,
            AgentIntent.COMPARE_WITH_PREVIOUS_RELEASE,
            AgentIntent.ACTION_REQUEST,
        }:
            context_resolver = AgentQueryContextResolver(
                snapshot_repository=risk_snapshot_repository,
                request_id=request_id,
            )
            context = await context_resolver.resolve(payload, plan)

            if plan.intent is AgentIntent.EXPLAIN_SPECIFIC_RISK:
                risk_matcher = AgentSpecificRiskMatcher(
                    request_id=request_id,
                )
                selected_risk = risk_matcher.match(
                    query=payload.query,
                    plan=plan,
                    release_risk=context.release_risk,
                )
                agent_response = response_composer.compose_specific_risk(
                    plan=plan,
                    release_risk=context.release_risk,
                    selected_risk=selected_risk,
                )
            elif plan.intent is AgentIntent.FILTER_RISKS:
                risk_filter = AgentRiskFilter(
                    request_id=request_id,
                )
                selected_risks = risk_filter.filter(
                    plan=plan,
                    release_risk=context.release_risk,
                )
                agent_response = response_composer.compose_filtered_risks(
                    plan=plan,
                    release_risk=context.release_risk,
                    selected_risks=selected_risks,
                )
            elif plan.intent is AgentIntent.GITHUB_PR_QUESTION:
                github_pr_resolver = AgentGitHubPRResolver(
                    request_id=request_id,
                )
                pull_request = github_pr_resolver.resolve(
                    plan=plan,
                    release_risk=context.release_risk,
                )
                agent_response = response_composer.compose_github_pr(
                    plan=plan,
                    release_risk=context.release_risk,
                    pull_request=pull_request,
                )
            elif plan.intent is AgentIntent.JIRA_TICKET_QUESTION:
                jira_ticket_resolver = AgentJiraTicketResolver(
                    request_id=request_id,
                )
                jira_issue = jira_ticket_resolver.resolve(
                    plan=plan,
                    release_risk=context.release_risk,
                )
                agent_response = response_composer.compose_jira_ticket(
                    plan=plan,
                    release_risk=context.release_risk,
                    jira_issue=jira_issue,
                )
            elif plan.intent is AgentIntent.WORKFLOW_STATUS_QUESTION:
                agent_response = response_composer.compose_workflow_status(
                    plan=plan,
                    release_risk=context.release_risk,
                )
            elif plan.intent is AgentIntent.APPROVAL_STATUS_QUESTION:
                latest_approval = (
                    await approval_repository.get_latest_by_release_run_id(
                        context.release_run_id,
                    )
                )
                agent_response = response_composer.compose_approval_status(
                    plan=plan,
                    release_risk=context.release_risk,
                    latest_approval=latest_approval,
                )
            elif plan.intent is AgentIntent.SLACK_STATUS_QUESTION:
                slack_alert = (
                    await slack_alert_repository.get_by_release_run_id(
                        context.release_run_id,
                    )
                )
                agent_response = response_composer.compose_slack_status(
                    plan=plan,
                    release_risk=context.release_risk,
                    slack_alert=slack_alert,
                )
            elif plan.intent is AgentIntent.ACTION_REQUEST:
                if slack_sender is None:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Slack alert delivery is unavailable.",
                    )

                action_service = SlackReleaseAlertActionService(
                    approval_repository=approval_repository,
                    risk_snapshot_repository=risk_snapshot_repository,
                    slack_alert_repository=slack_alert_repository,
                    event_repository=event_repository,
                    sender=slack_sender,
                    request_id=request_id,
                )
                slack_result = await action_service.execute(
                    context.release_run_id,
                )
                agent_response = (
                    response_composer.compose_slack_action_confirmation(
                        plan=plan,
                        release_risk=context.release_risk,
                        slack_result=slack_result,
                    )
                )
            elif plan.intent is AgentIntent.HISTORICAL_RISK_LOOKUP:
                historical_release_risks = (
                    await context_resolver.resolve_historical_release_risks(
                        exclude_release_run_id=context.release_run_id,
                        limit=10,
                    )
                )
                agent_response = response_composer.compose_historical_risks(
                    plan=plan,
                    release_risk=context.release_risk,
                    historical_release_risks=historical_release_risks,
                )
            elif plan.intent is AgentIntent.SIMILAR_PAST_RELEASE:
                historical_release_risks = (
                    await context_resolver.resolve_historical_release_risks(
                        exclude_release_run_id=context.release_run_id,
                        limit=100,
                    )
                )
                similar_release_matcher = AgentSimilarReleaseMatcher(
                    request_id=request_id,
                )
                similar_release_match = similar_release_matcher.match(
                    current_release_risk=context.release_risk,
                    historical_release_risks=historical_release_risks,
                )
                agent_response = response_composer.compose_similar_release(
                    plan=plan,
                    release_risk=context.release_risk,
                    similar_release_match=similar_release_match,
                )
            elif plan.intent is AgentIntent.COMPARE_WITH_PREVIOUS_RELEASE:
                previous_release_risks = (
                    await context_resolver.resolve_historical_release_risks(
                        exclude_release_run_id=context.release_run_id,
                        limit=1,
                    )
                )
                previous_release_risk = (
                    previous_release_risks[0]
                    if previous_release_risks
                    else None
                )
                agent_response = (
                    response_composer.compose_previous_release_comparison(
                        plan=plan,
                        release_risk=context.release_risk,
                        previous_release_risk=previous_release_risk,
                    )
                )
            else:
                agent_response = response_composer.compose(
                    plan=plan,
                    release_risk=context.release_risk,
                )

            await session.commit()
            return agent_response

        if risk_collector is None or jira_risk_collector is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Release-risk collectors are unavailable.",
            )

        release_run_repository = ReleaseRunRepository(
            session=session,
            request_id=request_id,
        )
        engineering_document_repository = EngineeringDocumentRepository(
            session=session,
        )
        knowledge_service = EngineeringDocumentRetrievalService(
            repository=engineering_document_repository,
            embedding_provider=embedding_provider,
            reranker=reranker,
        )

        release_run_service = ReleaseRunService(
            repository=release_run_repository,
            request_id=request_id,
            risk_collector=risk_collector,
            jira_risk_collector=jira_risk_collector,
            event_repository=event_repository,
            knowledge_service=knowledge_service,
        )
        executor = AgentQueryExecutor(
            release_run_service=release_run_service,
            request_id=request_id,
        )
        finalizer = ReleaseRiskExecutionFinalizer(
            release_run_repository=release_run_repository,
            approval_repository=approval_repository,
            event_repository=event_repository,
            risk_snapshot_repository=risk_snapshot_repository,
        )

        response = await executor.execute(
            payload,
            plan,
            requested_by="agent-query-api",
        )

        response = await finalizer.finalize(
            release_run_id=response.release_run.id,
            response=response,
        )

        agent_response = response_composer.compose(
            plan=plan,
            release_risk=response,
        )

        await session.commit()
        return agent_response

    except SlackReleaseAlertNotApprovedError as exc:
        await session.commit()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    except ReleaseRunSlackAlertAlreadySentError as exc:
        await session.commit()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    except SlackReleaseAlertServiceError as exc:
        await session.commit()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to send approved release-risk Slack alert.",
        ) from exc

    except AgentJiraTicketNotFoundError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No persisted Jira issue matched the question.",
        ) from exc

    except AgentGitHubPRNotFoundError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No persisted GitHub pull request matched the question.",
        ) from exc

    except AgentSpecificRiskNotFoundError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No persisted risk matched the follow-up question.",
        ) from exc

    except AgentQueryContextRequiredError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="A release-run ID is required for this follow-up query.",
        ) from exc

    except AgentQueryContextConflictError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Agent query context does not match the query plan.",
        ) from exc

    except AgentQuerySnapshotNotFoundError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No persisted release-risk snapshot was found.",
        ) from exc

    except AgentQuerySnapshotValidationError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Persisted release-risk context is invalid.",
        ) from exc

    except AgentQueryContextResolverError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to resolve persisted agent query context.",
        ) from exc

    except UnsupportedAgentQueryIntentError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="This agent query intent is not executable yet.",
        ) from exc

    except AgentQueryContextMismatchError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Agent query context does not match the query plan.",
        ) from exc

    except AgentQueryResultError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No release-risk result was found.",
        ) from exc

    except ValidationError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent workflow returned an invalid response.",
        ) from exc

    except (
        ReleaseRunServiceError,
        ReleaseRunRepositoryError,
        EngineeringDocumentRepositoryError,
        ReleaseRunEventRepositoryError,
        ReleaseRunApprovalRepositoryError,
        ReleaseRunRiskSnapshotRepositoryError,
        ReleaseRunSlackAlertRepositoryError,
        SQLAlchemyError,
    ) as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute the AgentFlow query.",
        ) from exc
