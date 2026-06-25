from fastapi import FastAPI

from app.api.routes.health import router as health_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    fastapi_app = FastAPI(
        title="AgentFlow AI Backend",
        version="0.1.0",
        description="Backend API for enterprise release risk automation.",
    )

    fastapi_app.include_router(health_router, prefix="/api/v1")

    return fastapi_app


app = create_app()