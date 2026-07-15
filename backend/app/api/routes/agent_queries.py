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
from app.integrations.github_client import GitHubClient, GitHubClientConfig
from app.integrations.jira_client import JiraClient, JiraClientConfig
from app.repositories.engineering_document_repository import (
    EngineeringDocumentRepository,
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
from app.schemas.agent_query import (
    AgentIntent,
    AgentQueryPlan,
    AgentQueryRequest,
    AgentQueryResponse,
)
from app.schemas.github import GitHubRepositoryConfig
from app.services.agent_query_executor import (
    AgentQueryContextMismatchError,
    AgentQueryExecutor,
    AgentQueryResultError,
    UnsupportedAgentQueryIntentError,
)
from app.services.agent_query_router import AgentQueryRouter
from app.services.agent_response_composer import AgentResponseComposer
from app.services.engineering_document_retrieval_service import (
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

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/agent",
    tags=["agent"],
)


def get_agent_query_router() -> AgentQueryRouter:
    """Return the deterministic AgentFlow query router."""

    return AgentQueryRouter()


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

    if plan.intent is not AgentIntent.RELEASE_RISK_SUMMARY:
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
) -> AsyncIterator[RiskCollector]:
    """Create the GitHub collector used by agent query execution."""

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
) -> AsyncIterator[JiraRiskCollector]:
    """Create the Jira collector used by agent query execution."""

    del plan

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
    RiskCollector,
    Depends(get_agent_github_risk_collector),
]
AgentJiraRiskCollectorDependency = Annotated[
    JiraRiskCollector,
    Depends(get_agent_jira_risk_collector),
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
    session: AsyncSession = Depends(get_db_session),
) -> AgentQueryResponse:
    """Route, execute, and compose a natural-language query response."""

    request_id = str(getattr(request.state, "request_id", "unknown-request-id"))

    release_run_repository = ReleaseRunRepository(
        session=session,
        request_id=request_id,
    )
    event_repository = ReleaseRunEventRepository(
        session=session,
        request_id=request_id,
    )
    approval_repository = ReleaseRunApprovalRepository(
        session=session,
        request_id=request_id,
    )
    risk_snapshot_repository = ReleaseRunRiskSnapshotRepository(
        session=session,
        request_id=request_id,
    )
    engineering_document_repository = EngineeringDocumentRepository(
        session=session,
    )
    knowledge_service = EngineeringDocumentRetrievalService(
        repository=engineering_document_repository,
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
    response_composer = AgentResponseComposer(
        request_id=request_id,
    )

    try:
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
        ReleaseRunEventRepositoryError,
        ReleaseRunApprovalRepositoryError,
        ReleaseRunRiskSnapshotRepositoryError,
        SQLAlchemyError,
    ) as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute the AgentFlow query.",
        ) from exc
