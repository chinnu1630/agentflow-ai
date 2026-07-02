"""Release run API routes for AgentFlow AI.

This module exposes endpoints for starting release-risk workflow runs,
fetching release runs, and collecting release risks from engineering sources.

Architecture position:
FastAPI route -> ReleaseRunService -> Repository + GitHub Collector + Jira Collector
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import SecretStr
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.integrations.github_client import GitHubClient, GitHubClientConfig
from app.repositories.release_run_repository import ReleaseRunRepository
from app.schemas.github import GitHubRepositoryConfig
from app.schemas.risk import ReleaseRunRiskResponse
from app.services.jira_risk_collector import JiraRiskCollector
from app.services.release_run_service import (
    ReleaseRunResult,
    ReleaseRunService,
    ReleaseRunServiceError,
    StartReleaseRunCommand,
)
from app.services.github_risk_collector import RiskCollector

router = APIRouter(prefix="/release-runs", tags=["release-runs"])


async def get_risk_collector(request: Request) -> AsyncIterator[RiskCollector]:
    """Create a GitHub risk collector for the current request.

    The collector is built from environment configuration so secrets and
    repository settings are not hardcoded in source code.
    """

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
    """Create a Jira risk collector dependency.

    The collector internally uses JiraClient and JiraRiskRuleEngine. Keeping
    this as a FastAPI dependency allows tests to override it with a fake
    collector and prevents API tests from making real Jira network calls.
    """

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
    """Collect release risks from GitHub and Jira for an existing release run.

    The route name and path still mention GitHub because this endpoint started
    as GitHub-only. The response now includes Jira risks too. Later, we can
    rename this endpoint to `/risks` after the workflow is fully stable.
    """

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
        result = await service.collect_github_risks(release_run_id)

        if result is None:
            await session.rollback()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Release run not found.",
            )

        await session.commit()

        return ReleaseRunRiskResponse.model_validate(result)

    except HTTPException:
        raise

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