from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.repositories.release_run_repository import ReleaseRunRepository
from app.services.release_run_service import (
    ReleaseRunResult,
    ReleaseRunService,
    ReleaseRunServiceError,
    StartReleaseRunCommand,
)

router = APIRouter(prefix="/release-runs", tags=["release-runs"])


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