"""Release run API routes for AgentFlow AI.

This module exposes endpoints for starting release-risk workflow runs,
fetching release runs, and collecting release risks from engineering sources.

Architecture position:
FastAPI route -> ReleaseRunService -> LangGraph workflow -> GitHub/Jira collectors
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.session import get_db_session
from app.integrations.github_client import GitHubClient, GitHubClientConfig
from app.integrations.jira_client import JiraClient, JiraClientConfig
from app.integrations.slack_client import SlackClient, SlackClientConfig
from app.observability.tracing import start_business_span
from app.repositories.engineering_document_repository import EngineeringDocumentRepository
from app.repositories.release_run_approval_repository import (
    CreateReleaseRunApprovalCommand,
    DecideReleaseRunApprovalCommand,
    ReleaseRunApprovalRepository,
    ReleaseRunApprovalRepositoryError,
    ReleaseRunApprovalStatus,
)
from app.repositories.release_run_event_repository import (
    CreateReleaseRunEventCommand,
    ReleaseRunEventRepository,
)
from app.repositories.release_run_repository import (
    ReleaseRunRepository,
    ReleaseRunRepositoryError,
)
from app.repositories.release_run_risk_snapshot_repository import (
    CreateReleaseRunRiskSnapshotCommand,
    ReleaseRunRiskSnapshotRepository,
    ReleaseRunRiskSnapshotRepositoryError,
)
from app.repositories.release_run_slack_alert_repository import (
    CreateReleaseRunSlackAlertCommand,
    ReleaseRunSlackAlertAlreadySentError,
    ReleaseRunSlackAlertRepository,
    ReleaseRunSlackAlertRepositoryError,
)
from app.schemas.github import GitHubRepositoryConfig
from app.schemas.release_run_approval import (
    PendingReleaseRunApprovalListResponse,
    ReleaseRunApprovalDecisionRequest,
    ReleaseRunApprovalListResponse,
    ReleaseRunApprovalResponse,
)
from app.schemas.release_run_event import (
    ReleaseRunEventListResponse,
    ReleaseRunEventResponse,
)
from app.schemas.risk import ReleaseRunRiskResponse
from app.services.engineering_document_retrieval_service import (
    EngineeringDocumentRetrievalService,
)
from app.services.github_risk_collector import RiskCollector
from app.services.jira_risk_collector import JiraRiskCollector
from app.services.release_risk_execution_finalizer import (
    ReleaseRiskExecutionFinalizer,
)
from app.services.release_risk_response_mapper import (
    extract_risk_result_from_workflow_state,
    to_release_run_risk_response,
)
from app.services.release_run_service import (
    ReleaseRunResult,
    ReleaseRunService,
    ReleaseRunServiceError,
    StartReleaseRunCommand,
)
from app.services.slack_release_alert_service import (
    SlackReleaseAlertNotApprovedError,
    SlackReleaseAlertResult,
    SlackReleaseAlertService,
    SlackReleaseAlertServiceError,
)

router = APIRouter(prefix="/release-runs", tags=["release-runs"])


async def get_risk_collector(request: Request) -> AsyncIterator[RiskCollector]:
    """Create a GitHub risk collector for the current request.

    The collector is built from environment configuration so secrets and
    repository settings are not hardcoded in source code.
    """

    request_id = str(getattr(request.state, "request_id", "unknown-request-id"))

    settings = get_settings()
    repository_owner = settings.github_repository_owner
    repository_name = settings.github_repository_name

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


async def get_jira_risk_collector() -> AsyncIterator[JiraRiskCollector]:
    """Create a Jira risk collector dependency.

    The collector internally uses JiraClient and JiraRiskRuleEngine. Keeping
    this as a FastAPI dependency allows tests to override it with a fake
    collector and prevents API tests from making real Jira network calls.
    """

    settings = get_settings()
    jira_base_url = settings.jira_base_url
    jira_email = settings.jira_email
    jira_api_token = settings.jira_api_token
    jira_project_key = settings.jira_project_key

    if (
        jira_base_url is None
        or jira_email is None
        or jira_api_token is None
        or jira_project_key is None
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Jira risk collection is not configured. Set JIRA_BASE_URL, "
                "JIRA_EMAIL, JIRA_API_TOKEN, and JIRA_PROJECT_KEY."
            ),
        )

    jira_config = JiraClientConfig(
        base_url=jira_base_url,
        email=jira_email,
        api_token=jira_api_token,
        project_key=jira_project_key,
    )

    async with JiraClient(config=jira_config) as jira_client:
        yield JiraRiskCollector(jira_client=jira_client)


async def get_slack_alert_sender(request: Request) -> AsyncIterator[SlackClient]:
    """Create a Slack alert sender for the current request.

    Slack credentials are read from environment variables so bot tokens and
    channel IDs are never hardcoded in source code.
    """

    request_id = str(getattr(request.state, "request_id", "unknown-request-id"))

    settings = get_settings()
    slack_bot_token = settings.slack_bot_token
    slack_channel_id = settings.slack_channel_id

    if not slack_bot_token or not slack_channel_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Slack alert delivery is not configured. Set SLACK_BOT_TOKEN and SLACK_CHANNEL_ID."
            ),
        )

    slack_config = SlackClientConfig(
        bot_token=slack_bot_token,
        channel_id=slack_channel_id,
    )

    async with httpx.AsyncClient() as http_client:
        yield SlackClient(
            http_client=http_client,
            config=slack_config,
            request_id=request_id,
        )


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
    event_repository = ReleaseRunEventRepository(
        session=session,
        request_id=request_id,
    )
    service = ReleaseRunService(
        repository=repository,
        request_id=request_id,
        event_repository=event_repository,
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
    "/approvals/pending",
    response_model=PendingReleaseRunApprovalListResponse,
)
async def list_pending_release_run_approvals(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> PendingReleaseRunApprovalListResponse:
    """List pending HITL approval requests across release runs.

    This endpoint powers the future manager review dashboard. It lets the UI
    show all releases waiting for human approval without knowing release IDs
    in advance.
    """

    request_id = str(getattr(request.state, "request_id", "unknown-request-id"))

    approval_repository = ReleaseRunApprovalRepository(
        session=session,
        request_id=request_id,
    )

    try:
        approvals = await approval_repository.list_by_status(
            ReleaseRunApprovalStatus.PENDING,
            limit=limit,
            offset=offset,
        )

        return PendingReleaseRunApprovalListResponse(
            approval_status=ReleaseRunApprovalStatus.PENDING.value,
            approvals=[
                ReleaseRunApprovalResponse.model_validate(approval) for approval in approvals
            ],
        )

    except (ReleaseRunApprovalRepositoryError, SQLAlchemyError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while fetching pending approvals.",
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
    event_repository = ReleaseRunEventRepository(
        session=session,
        request_id=request_id,
    )
    service = ReleaseRunService(
        repository=repository,
        request_id=request_id,
        event_repository=event_repository,
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


@router.get(
    "/{release_run_id}/events",
    response_model=ReleaseRunEventListResponse,
)
async def list_release_run_events(
    release_run_id: UUID,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> ReleaseRunEventListResponse:
    """List audit events for a release-risk workflow run.

    This endpoint returns the append-only audit timeline for one release run.
    It supports enterprise traceability by showing what the workflow did and
    when each step happened.
    """

    request_id = str(getattr(request.state, "request_id", "unknown-request-id"))

    release_run_repository = ReleaseRunRepository(
        session=session,
        request_id=request_id,
    )
    event_repository = ReleaseRunEventRepository(
        session=session,
        request_id=request_id,
    )

    try:
        release_run = await release_run_repository.get_by_id(release_run_id)

        if release_run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Release run not found.",
            )

        events = await event_repository.list_by_release_run_id(release_run_id)

        return ReleaseRunEventListResponse(
            release_run_id=release_run_id,
            events=[ReleaseRunEventResponse.model_validate(event) for event in events],
        )

    except HTTPException:
        raise

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while fetching release-run events.",
        ) from exc


@router.get(
    "/{release_run_id}/approvals",
    response_model=ReleaseRunApprovalListResponse,
)
async def list_release_run_approvals(
    release_run_id: UUID,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> ReleaseRunApprovalListResponse:
    """List HITL approval requests for a release-risk workflow run."""

    request_id = str(getattr(request.state, "request_id", "unknown-request-id"))

    release_run_repository = ReleaseRunRepository(
        session=session,
        request_id=request_id,
    )
    approval_repository = ReleaseRunApprovalRepository(
        session=session,
        request_id=request_id,
    )

    try:
        release_run = await release_run_repository.get_by_id(release_run_id)

        if release_run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Release run not found.",
            )

        approvals = await approval_repository.list_by_release_run_id(release_run_id)

        return ReleaseRunApprovalListResponse(
            release_run_id=release_run_id,
            approvals=[
                ReleaseRunApprovalResponse.model_validate(approval) for approval in approvals
            ],
        )

    except HTTPException:
        raise

    except (ReleaseRunApprovalRepositoryError, SQLAlchemyError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while fetching release-run approvals.",
        ) from exc


@router.post(
    "/{release_run_id}/approvals/{approval_id}/decision",
    response_model=ReleaseRunApprovalResponse,
)
async def decide_release_run_approval(
    release_run_id: UUID,
    approval_id: UUID,
    decision_request: ReleaseRunApprovalDecisionRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> ReleaseRunApprovalResponse:
    """Approve or reject a pending HITL release approval request."""

    request_id = str(getattr(request.state, "request_id", "unknown-request-id"))

    release_run_repository = ReleaseRunRepository(
        session=session,
        request_id=request_id,
    )
    approval_repository = ReleaseRunApprovalRepository(
        session=session,
        request_id=request_id,
    )
    event_repository = ReleaseRunEventRepository(
        session=session,
        request_id=request_id,
    )

    try:
        release_run = await release_run_repository.get_by_id(release_run_id)

        if release_run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Release run not found.",
            )

        approval = await approval_repository.get_by_id(approval_id)

        if approval is None or approval.release_run_id != release_run_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Approval request not found.",
            )

        decided_approval = await approval_repository.decide(
            DecideReleaseRunApprovalCommand(
                approval_id=approval_id,
                approval_status=ReleaseRunApprovalStatus(decision_request.approval_status.value),
                decided_by=decision_request.decided_by,
                decision_note=decision_request.decision_note,
            )
        )

        if decided_approval is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Approval request not found.",
            )

        release_run_status = (
            "approval_approved"
            if decided_approval.approval_status == "approved"
            else "approval_rejected"
        )

        await release_run_repository.update_status(
            release_run_id=release_run_id,
            status=release_run_status,
        )

        await event_repository.create(
            CreateReleaseRunEventCommand(
                release_run_id=release_run_id,
                event_type="approval_request_decided",
                event_status="success",
                message="Release approval request was decided.",
                metadata_json={
                    "approval_request_id": str(decided_approval.id),
                    "approval_status": decided_approval.approval_status,
                    "release_run_status": release_run_status,
                    "decided_by": decided_approval.decided_by,
                    "decision_note_present": (decided_approval.decision_note is not None),
                },
            )
        )

        await session.commit()

        return ReleaseRunApprovalResponse.model_validate(decided_approval)

    except HTTPException:
        raise

    except ValueError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    except (ReleaseRunApprovalRepositoryError, SQLAlchemyError) as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while deciding release-run approval.",
        ) from exc


@router.post(
    "/{release_run_id}/slack-alert",
    response_model=SlackReleaseAlertResult,
)
async def send_release_run_slack_alert(
    release_run_id: UUID,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    sender: SlackClient = Depends(get_slack_alert_sender),
) -> SlackReleaseAlertResult:
    """Manually send an approved release-risk Slack alert.

    This endpoint intentionally accepts only the release_run_id. Slack alert
    content is loaded from the latest backend-trusted risk snapshot after the
    latest HITL approval is verified as approved.
    """

    request_id_for_span = str(getattr(request.state, "request_id", "unknown-request-id"))
    with start_business_span(
        "slack.release_alert.route",
        {
            "release_run_id": str(release_run_id),
            "run_id": request_id_for_span,
            "route": "/api/v1/release-runs/{release_run_id}/slack-alert",
        },
    ):
        request_id = str(getattr(request.state, "request_id", "unknown-request-id"))

        release_run_repository = ReleaseRunRepository(
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
        slack_alert_repository = ReleaseRunSlackAlertRepository(
            session=session,
            request_id=request_id,
        )
        event_repository = ReleaseRunEventRepository(
            session=session,
            request_id=request_id,
        )
        slack_service = SlackReleaseAlertService()

        try:
            release_run = await release_run_repository.get_by_id(release_run_id)

            if release_run is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Release run not found.",
                )

            with start_business_span(
                "slack.release_alert.duplicate_check",
                {
                    "release_run_id": str(release_run_id),
                    "run_id": request_id,
                },
            ) as span:
                existing_slack_alert = await slack_alert_repository.get_by_release_run_id(
                    release_run_id
                )
                duplicate_found = existing_slack_alert is not None
                span.set_attribute("slack.duplicate_found", duplicate_found)

                if existing_slack_alert is not None:
                    await _record_release_slack_alert_event(
                        event_repository=event_repository,
                        release_run_id=release_run_id,
                        event_status="blocked",
                        message="Duplicate Slack alert send was blocked.",
                        metadata_json={
                            "reason": "Slack alert already sent for this release run.",
                            "existing_slack_alert_id": str(existing_slack_alert.id),
                            "existing_slack_channel": existing_slack_alert.slack_channel,
                            "existing_slack_timestamp": existing_slack_alert.slack_timestamp,
                        },
                    )
                    await session.commit()

                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="Slack alert already sent for this release run.",
                    )

            latest_approval = await approval_repository.get_latest_by_release_run_id(release_run_id)
            latest_snapshot = await risk_snapshot_repository.get_latest_by_release_run_id(
                release_run_id
            )

            result = await slack_service.send_approved_release_alert_from_snapshot(
                release_run_id,
                approval_repository=approval_repository,
                risk_snapshot_repository=risk_snapshot_repository,
                sender=sender,
                run_id=request_id,
            )

            alert_record = await slack_alert_repository.create_sent_alert(
                CreateReleaseRunSlackAlertCommand(
                    release_run_id=release_run_id,
                    approval_request_id=(
                        latest_approval.id if latest_approval is not None else None
                    ),
                    snapshot_id=latest_snapshot.id if latest_snapshot is not None else None,
                    snapshot_version=(
                        latest_snapshot.snapshot_version if latest_snapshot is not None else None
                    ),
                    slack_channel=result.slack_channel,
                    slack_timestamp=result.slack_timestamp,
                    risk_level=result.risk_level,
                    risk_score=result.risk_score,
                    recommended_action=result.recommended_action,
                )
            )

            await _record_release_slack_alert_event(
                event_repository=event_repository,
                release_run_id=release_run_id,
                event_status="success",
                message="Approved release-risk Slack alert was sent.",
                metadata_json={
                    "slack_alert_id": str(alert_record.id),
                    "slack_channel": result.slack_channel,
                    "slack_timestamp": result.slack_timestamp,
                    "risk_level": result.risk_level,
                    "risk_score": result.risk_score,
                    "recommended_action": result.recommended_action,
                },
            )

            await session.commit()

            return result

        except HTTPException:
            raise

        except SlackReleaseAlertNotApprovedError as exc:
            await _record_release_slack_alert_event(
                event_repository=event_repository,
                release_run_id=release_run_id,
                event_status="blocked",
                message="Slack alert was blocked because release is not approved.",
                metadata_json={
                    "reason": str(exc),
                },
            )
            await session.commit()

            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc

        except ReleaseRunSlackAlertAlreadySentError as exc:
            await _record_release_slack_alert_event(
                event_repository=event_repository,
                release_run_id=release_run_id,
                event_status="blocked",
                message="Duplicate Slack alert send was blocked.",
                metadata_json={
                    "reason": str(exc),
                },
            )
            await session.commit()

            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc

        except SlackReleaseAlertServiceError as exc:
            await _record_release_slack_alert_event(
                event_repository=event_repository,
                release_run_id=release_run_id,
                event_status="failed",
                message="Approved release-risk Slack alert failed.",
                metadata_json={
                    "error_type": exc.__class__.__name__,
                },
            )
            await session.commit()

            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Failed to send approved release-risk Slack alert.",
            ) from exc

        except ReleaseRunSlackAlertRepositoryError as exc:
            await session.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error while recording Slack alert idempotency.",
            ) from exc

        except (ReleaseRunRepositoryError, SQLAlchemyError) as exc:
            await session.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error while sending release-risk Slack alert.",
            ) from exc


async def _record_release_slack_alert_event(
    *,
    event_repository: ReleaseRunEventRepository,
    release_run_id: UUID,
    event_status: str,
    message: str,
    metadata_json: dict[str, Any],
) -> None:
    """Persist a safe audit event for manual Slack alert attempts."""

    await event_repository.create(
        CreateReleaseRunEventCommand(
            release_run_id=release_run_id,
            event_type="release_slack_alert_sent",
            event_status=event_status,
            message=message,
            metadata_json=metadata_json,
        )
    )


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
    """Run the LangGraph release-risk workflow for an existing release run.

    This is the preferred endpoint because it enters the production workflow
    path:

    FastAPI -> ReleaseRunService -> LangGraph -> GitHub/Jira collectors

    The public API response shape stays unchanged:
    release_run, github, github_summary, jira, jira_summary, release_summary.
    """

    request_id = str(getattr(request.state, "request_id", "unknown-request-id"))

    with start_business_span(
        "release_run.risks_endpoint",
        {
            "release_run_id": str(release_run_id),
            "run_id": request_id,
            "route": "/api/v1/release-runs/{release_run_id}/risks",
        },
    ):
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
    """Collect release risks for an existing release run using the legacy path.

    This endpoint is kept for backward compatibility. It started as a
    GitHub-only route, but the response now includes Jira risks and the combined
    release summary. New clients should use /release-runs/{id}/risks.
    """

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
    """Run the LangGraph workflow and convert its final state into an API response.

    LangGraph returns internal workflow state. The API must still return the
    stable ReleaseRunRiskResponse contract expected by frontend clients and
    integration tests.
    """

    request_id = str(getattr(request.state, "request_id", "unknown-request-id"))

    repository = ReleaseRunRepository(
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

    service = ReleaseRunService(
        repository=repository,
        request_id=request_id,
        risk_collector=risk_collector,
        jira_risk_collector=jira_risk_collector,
        event_repository=event_repository,
        knowledge_service=knowledge_service,
    )
    finalizer = ReleaseRiskExecutionFinalizer(
        release_run_repository=repository,
        approval_repository=approval_repository,
        event_repository=event_repository,
        risk_snapshot_repository=risk_snapshot_repository,
    )

    try:
        workflow_state = await service.run_release_risk_workflow(release_run_id)
        result = extract_risk_result_from_workflow_state(workflow_state)

        if result is None:
            await session.rollback()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Release run not found.",
            )

        response = to_release_run_risk_response(result)

        response = await finalizer.finalize(
            release_run_id=release_run_id,
            response=response,
        )

        await session.commit()

        return response

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

    except ReleaseRunApprovalRepositoryError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create release approval request.",
        ) from exc

    except ReleaseRunRiskSnapshotRepositoryError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist release-risk snapshot.",
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
    """Collect release risks using the legacy direct service path.

    This helper is used by the backward-compatible /github-risks endpoint.
    Do not route the preferred /risks endpoint here anymore.
    """

    request_id = str(getattr(request.state, "request_id", "unknown-request-id"))

    repository = ReleaseRunRepository(
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
    service = ReleaseRunService(
        repository=repository,
        request_id=request_id,
        risk_collector=risk_collector,
        jira_risk_collector=jira_risk_collector,
        event_repository=event_repository,
    )

    try:
        result = await service.collect_release_risks(release_run_id)

        if result is None:
            await session.rollback()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Release run not found.",
            )

        response = to_release_run_risk_response(result)

        await _record_scoring_audit_events(
            event_repository=event_repository,
            release_run_id=release_run_id,
            response=response,
        )
        response = await _ensure_pending_approval_request(
            release_run_repository=repository,
            approval_repository=approval_repository,
            event_repository=event_repository,
            release_run_id=release_run_id,
            response=response,
        )

        await session.commit()

        return response

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

    except ReleaseRunApprovalRepositoryError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create release approval request.",
        ) from exc

    except SQLAlchemyError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while collecting release risks.",
        ) from exc


async def _persist_release_risk_snapshot(
    *,
    risk_snapshot_repository: ReleaseRunRiskSnapshotRepository,
    event_repository: ReleaseRunEventRepository,
    release_run_id: UUID,
    response: ReleaseRunRiskResponse,
) -> None:
    """Persist the final backend-generated release-risk response as a snapshot.

    Slack alerts must later read this trusted backend snapshot instead of
    accepting client-supplied risk payloads. The audit event stores only safe
    metadata and does not duplicate raw PRs, Jira tickets, Knowledge chunks,
    prompts, or stack traces.
    """
    with start_business_span(
        "snapshot.persist",
        {
            "release_run_id": str(release_run_id),
            "approval_required": response.approval_required is True,
            "overall_severity": _safe_enum_value(response.release_summary.overall_severity),
        },
    ):
        approval_required = response.approval_required is True
        approval_status_at_snapshot = response.approval_status

        if approval_status_at_snapshot is None:
            approval_status_at_snapshot = (
                ReleaseRunApprovalStatus.PENDING.value if approval_required else "not_required"
            )

        snapshot = await risk_snapshot_repository.create_snapshot(
            CreateReleaseRunRiskSnapshotCommand(
                release_run_id=release_run_id,
                risk_payload=response.model_dump(mode="json"),
                overall_severity=_safe_enum_value(response.release_summary.overall_severity),
                approval_required=approval_required,
                approval_status_at_snapshot=approval_status_at_snapshot,
            )
        )

        await event_repository.create(
            CreateReleaseRunEventCommand(
                release_run_id=release_run_id,
                event_type="release_risk_snapshot_created",
                event_status="success",
                message="Trusted release-risk report snapshot was persisted.",
                metadata_json={
                    "snapshot_id": str(snapshot.id),
                    "snapshot_version": snapshot.snapshot_version,
                    "overall_severity": snapshot.overall_severity,
                    "approval_required": snapshot.approval_required,
                    "approval_status_at_snapshot": (snapshot.approval_status_at_snapshot),
                },
            )
        )


async def _ensure_pending_approval_request(
    *,
    release_run_repository: ReleaseRunRepository,
    approval_repository: ReleaseRunApprovalRepository,
    event_repository: ReleaseRunEventRepository,
    release_run_id: UUID,
    response: ReleaseRunRiskResponse,
) -> ReleaseRunRiskResponse:
    """Create or reuse a pending approval request when HITL approval is required.

    This function stores only the safe approval reason generated by the
    deterministic HITL policy. It does not store raw PRs, Jira tickets,
    Knowledge chunks, prompts, or stack traces.
    """
    with start_business_span(
        "approval.ensure_pending",
        {
            "release_run_id": str(release_run_id),
            "approval_required": response.approval_required is True,
        },
    ):
        if response.approval_required is not True:
            return response

        latest_approval = await approval_repository.get_latest_by_release_run_id(release_run_id)

        if (
            latest_approval is not None
            and latest_approval.approval_status == ReleaseRunApprovalStatus.PENDING.value
        ):
            await _mark_release_run_waiting_for_approval(
                release_run_repository=release_run_repository,
                event_repository=event_repository,
                release_run_id=release_run_id,
                approval_request_id=latest_approval.id,
                approval_status=latest_approval.approval_status,
                approval_policy_version=latest_approval.approval_policy_version,
            )

            return response.model_copy(
                update={
                    "approval_request_id": latest_approval.id,
                    "approval_status": latest_approval.approval_status,
                }
            )

        approval = await approval_repository.create_pending(
            CreateReleaseRunApprovalCommand(
                release_run_id=release_run_id,
                approval_reason=(
                    response.approval_reason or "Release requires human approval before proceeding."
                ),
                approval_policy_version=(response.approval_policy_version or "hitl_policy_v1"),
                requested_by=response.release_run.requested_by,
            )
        )

        await event_repository.create(
            CreateReleaseRunEventCommand(
                release_run_id=release_run_id,
                event_type="approval_request_created",
                event_status="success",
                message="Pending release approval request was created.",
                metadata_json={
                    "approval_request_id": str(approval.id),
                    "approval_status": approval.approval_status,
                    "approval_policy_version": approval.approval_policy_version,
                    "approval_reason_present": bool(approval.approval_reason),
                },
            )
        )

        await _mark_release_run_waiting_for_approval(
            release_run_repository=release_run_repository,
            event_repository=event_repository,
            release_run_id=release_run_id,
            approval_request_id=approval.id,
            approval_status=approval.approval_status,
            approval_policy_version=approval.approval_policy_version,
        )

        return response.model_copy(
            update={
                "approval_request_id": approval.id,
                "approval_status": approval.approval_status,
                "release_run": response.release_run.model_copy(
                    update={"status": "waiting_for_approval"}
                ),
            }
        )


async def _mark_release_run_waiting_for_approval(
    *,
    release_run_repository: ReleaseRunRepository,
    event_repository: ReleaseRunEventRepository,
    release_run_id: UUID,
    approval_request_id: UUID,
    approval_status: str,
    approval_policy_version: str,
) -> None:
    """Mark release run as waiting for human approval.

    This status makes the parent release run reflect the real business state:
    risk analysis is done, but the release is paused until a human decides.
    """

    await release_run_repository.update_status(
        release_run_id=release_run_id,
        status="waiting_for_approval",
    )

    await event_repository.create(
        CreateReleaseRunEventCommand(
            release_run_id=release_run_id,
            event_type="release_run_waiting_for_approval",
            event_status="success",
            message="Release run is waiting for human approval.",
            metadata_json={
                "approval_request_id": str(approval_request_id),
                "approval_status": approval_status,
                "approval_policy_version": approval_policy_version,
            },
        )
    )


def _count_collection_risks(collection: object) -> int:
    """Return a safe risk count from a risk collection response.

    Observability must never break the release-risk workflow. This helper
    supports multiple response shapes and returns 0 when no known risk list
    field exists.
    """

    for attribute_name in ("risks", "risk_signals", "signals"):
        value = getattr(collection, attribute_name, None)

        if isinstance(value, list):
            return len(value)

    return 0


async def _record_scoring_audit_events(
    *,
    event_repository: ReleaseRunEventRepository,
    release_run_id: UUID,
    response: ReleaseRunRiskResponse,
) -> None:
    """Persist safe audit events for feature extraction and release-risk scoring.

    The metadata intentionally stores only counts, versions, and enum-like
    decisions. It does not store raw PR text, Jira text, Knowledge chunks,
    manager queries, or stack traces.
    """
    with start_business_span(
        "risk.scoring_audit",
        {
            "release_run_id": str(release_run_id),
            "github_risk_count": _count_collection_risks(response.github),
            "jira_risk_count": _count_collection_risks(response.jira),
            "total_risk_count": _count_collection_risks(response.github)
            + _count_collection_risks(response.jira),  # noqa: E501
            "approval_required": response.approval_required is True,
        },
    ):
        if response.risk_features is None or response.risk_score is None:
            return

        risk_features = response.risk_features
        risk_score = response.risk_score

        await event_repository.create(
            CreateReleaseRunEventCommand(
                release_run_id=release_run_id,
                event_type="risk_features_extracted",
                event_status="success",
                message="Release-risk scoring features were extracted.",
                metadata_json={
                    "feature_version": risk_features.feature_version,
                    "total_risk_count": risk_features.total_risk_count,
                    "github_risk_count": risk_features.github_risk_count,
                    "jira_risk_count": risk_features.jira_risk_count,
                    "critical_risk_count": risk_features.critical_risk_count,
                    "high_risk_count": risk_features.high_risk_count,
                    "knowledge_result_count": risk_features.knowledge_result_count,
                    "knowledge_no_results": risk_features.knowledge_no_results,
                    "knowledge_failed": risk_features.knowledge_failed,
                },
            )
        )

        await event_repository.create(
            CreateReleaseRunEventCommand(
                release_run_id=release_run_id,
                event_type="release_risk_scored",
                event_status="success",
                message="Release risk was scored using deterministic rule-based scoring.",
                metadata_json={
                    "scoring_version": risk_score.scoring_version,
                    "feature_version": risk_score.feature_version,
                    "score": risk_score.score,
                    "risk_level": _safe_enum_value(risk_score.risk_level),
                    "recommended_action": _safe_enum_value(risk_score.recommended_action),
                    "reason_count": len(risk_score.reasons),
                    "component_score_count": len(risk_score.component_scores),
                },
            )
        )

        if response.approval_policy_version is not None:
            await event_repository.create(
                CreateReleaseRunEventCommand(
                    release_run_id=release_run_id,
                    event_type="approval_requirement_determined",
                    event_status="success",
                    message="HITL approval requirement was determined.",
                    metadata_json={
                        "approval_policy_version": response.approval_policy_version,
                        "approval_required": response.approval_required,
                        "approval_reason_present": response.approval_reason is not None,
                        "risk_level": _safe_enum_value(risk_score.risk_level),
                        "recommended_action": _safe_enum_value(risk_score.recommended_action),
                    },
                )
            )


def _safe_enum_value(value: object) -> str:
    """Return a safe string value for enum-like audit metadata."""

    enum_value = getattr(value, "value", None)

    if enum_value is not None:
        return str(enum_value)

    return str(value)
