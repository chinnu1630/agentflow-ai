"""Release run API routes for AgentFlow AI.

This module exposes endpoints for starting release-risk workflow runs,
fetching release runs, and collecting release risks from engineering sources.

Architecture position:
FastAPI route -> ReleaseRunService -> LangGraph workflow -> GitHub/Jira collectors
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Mapping
from typing import Any
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import SecretStr, ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.integrations.github_client import GitHubClient, GitHubClientConfig
from app.repositories.release_run_repository import ReleaseRunRepository
from app.schemas.github import GitHubRepositoryConfig
from app.schemas.risk import ReleaseRunRiskResponse
from app.services.github_risk_collector import RiskCollector
from app.services.jira_risk_collector import JiraRiskCollector
from app.services.release_run_service import (
    ReleaseRunResult,
    ReleaseRunService,
    ReleaseRunServiceError,
    StartReleaseRunCommand,
)

router = APIRouter(prefix="/release-runs", tags=["release-runs"])


async def get_risk_collector(request: Request) -> AsyncIterator[RiskCollector]:
    """Create a GitHub risk collector for the current request."""

    request_id = str(getattr(request.state, "request_id", "unknown-request-id"))

    repository_owner = os.getenv("GITHUB_REPOSITORY_OWNER")
    repository_name = os.getenv("GITHUB_REPOSITORY_NAME")
    repository_default_branch = os.getenv("GITHUB_DEFAULT_BRANCH", "main")
    github_token = os.getenv("GITHUB_TOKEN")

    if not repository_owner or not repository_name:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "GitHub risk collection is not configured. "
                "Set GITHUB_REPOSITORY_OWNER and GITHUB_REPOSITORY_NAME."
            ),
        )

    repository_config = GitHubRepositoryConfig(
        owner=repository_owner,
        repo=repository_name,
        default_branch=repository_default_branch,
    )

    github_config = GitHubClientConfig(
        repository=repository_config,
        token=SecretStr(github_token) if github_token else None,
    )

    async with httpx.AsyncClient() as http_client:
        github_client = GitHubClient(
            http_client=http_client,
            config=github_config,
            request_id=request_id,
        )

        yield RiskCollector(github_client=github_client)


def get_jira_risk_collector() -> JiraRiskCollector:
    """Create a Jira risk collector dependency."""

    return JiraRiskCollector()


@router.post(
    "",
    response_model=ReleaseRunResult,
    status_code=status.HTTP_201_CREATED,
)
async def start_release_run(
    command: StartReleaseRunCommand,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> ReleaseRunResult:
    """Start a new release-risk workflow run."""

    request_id = str(getattr(request.state, "request_id", "unknown-request-id"))

    repository = ReleaseRunRepository(
        session=session,
        request_id=request_id,
    )
    service = ReleaseRunService(
        repository=repository,
        request_id=request_id,
    )

    try:
        result = await service.start_release_run(command)
        await session.commit()
        return result

    except ReleaseRunServiceError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start release-risk workflow.",
        ) from exc

    except SQLAlchemyError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while starting release-risk workflow.",
        ) from exc


@router.get(
    "/{release_run_id}",
    response_model=ReleaseRunResult,
)
async def get_release_run(
    release_run_id: UUID,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> ReleaseRunResult:
    """Fetch a release-risk workflow run by ID."""

    request_id = str(getattr(request.state, "request_id", "unknown-request-id"))

    repository = ReleaseRunRepository(
        session=session,
        request_id=request_id,
    )
    service = ReleaseRunService(
        repository=repository,
        request_id=request_id,
    )

    try:
        result = await service.get_release_run(release_run_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Release run not found.",
            )

        return result

    except HTTPException:
        raise

    except ReleaseRunServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch release run.",
        ) from exc


@router.post(
    "/{release_run_id}/risks",
    response_model=ReleaseRunRiskResponse,
)
async def collect_release_risks(
    release_run_id: UUID,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    risk_collector: RiskCollector = Depends(get_risk_collector),
    jira_risk_collector: JiraRiskCollector = Depends(get_jira_risk_collector),
) -> ReleaseRunRiskResponse:
    """Run the LangGraph release-risk workflow for an existing release run."""

    return await _collect_release_risk_workflow_response(
        release_run_id=release_run_id,
        request=request,
        session=session,
        risk_collector=risk_collector,
        jira_risk_collector=jira_risk_collector,
    )


@router.post(
    "/{release_run_id}/github-risks",
    response_model=ReleaseRunRiskResponse,
)
async def collect_github_risks(
    release_run_id: UUID,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    risk_collector: RiskCollector = Depends(get_risk_collector),
    jira_risk_collector: JiraRiskCollector = Depends(get_jira_risk_collector),
) -> ReleaseRunRiskResponse:
    """Collect release risks for an existing release run using the legacy path."""

    return await _collect_release_risks_response(
        release_run_id=release_run_id,
        request=request,
        session=session,
        risk_collector=risk_collector,
        jira_risk_collector=jira_risk_collector,
    )


async def _collect_release_risk_workflow_response(
    *,
    release_run_id: UUID,
    request: Request,
    session: AsyncSession,
    risk_collector: RiskCollector,
    jira_risk_collector: JiraRiskCollector,
) -> ReleaseRunRiskResponse:
    """Run the LangGraph workflow and convert its final state into an API response."""

    request_id = str(getattr(request.state, "request_id", "unknown-request-id"))

    repository = ReleaseRunRepository(
        session=session,
        request_id=request_id,
    )
    service = ReleaseRunService(
        repository=repository,
        request_id=request_id,
        risk_collector=risk_collector,
        jira_risk_collector=jira_risk_collector,
    )

    try:
        workflow_state = await service.run_release_risk_workflow(release_run_id)
        result = _extract_risk_result_from_workflow_state(workflow_state)

        if result is None:
            await session.rollback()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Release run not found.",
            )

        await session.commit()

        return _to_release_run_risk_response(result)

    except HTTPException:
        raise

    except ValidationError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Workflow returned an invalid release-risk response.",
        ) from exc

    except ReleaseRunServiceError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run release-risk workflow.",
        ) from exc

    except SQLAlchemyError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while running release-risk workflow.",
        ) from exc


async def _collect_release_risks_response(
    *,
    release_run_id: UUID,
    request: Request,
    session: AsyncSession,
    risk_collector: RiskCollector,
    jira_risk_collector: JiraRiskCollector,
) -> ReleaseRunRiskResponse:
    """Collect release risks using the legacy direct service path."""

    request_id = str(getattr(request.state, "request_id", "unknown-request-id"))

    repository = ReleaseRunRepository(
        session=session,
        request_id=request_id,
    )
    service = ReleaseRunService(
        repository=repository,
        request_id=request_id,
        risk_collector=risk_collector,
        jira_risk_collector=jira_risk_collector,
    )

    try:
        result = await service.collect_release_risks(release_run_id)

        if result is None:
            await session.rollback()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Release run not found.",
            )

        await session.commit()

        return _to_release_run_risk_response(result)

    except HTTPException:
        raise

    except ValidationError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Release-risk response validation failed.",
        ) from exc

    except ReleaseRunServiceError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to collect release risks.",
        ) from exc

    except SQLAlchemyError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while collecting release risks.",
        ) from exc


def _extract_risk_result_from_workflow_state(
    workflow_state: Mapping[str, Any] | Any,
) -> Any | None:
    """Extract the release-risk result from the LangGraph workflow state."""

    if isinstance(workflow_state, Mapping):
        return workflow_state.get("risk_result")

    if hasattr(workflow_state, "risk_result"):
        return getattr(workflow_state, "risk_result")

    if hasattr(workflow_state, "model_dump"):
        dumped_state = workflow_state.model_dump()
        return dumped_state.get("risk_result")

    return None


def _to_release_run_risk_response(result: Any) -> ReleaseRunRiskResponse:
    """Convert a service or workflow result into the public API response model."""

    if hasattr(result, "model_dump"):
        return ReleaseRunRiskResponse.model_validate(result.model_dump())

    return ReleaseRunRiskResponse.model_validate(result)
