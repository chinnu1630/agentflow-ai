from fastapi import APIRouter

from app.api.routes.health import router as health_router
from app.api.routes.release_runs import router as release_runs_router

api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(release_runs_router)