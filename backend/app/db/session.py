from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def create_database_engine(database_url: str | None) -> AsyncEngine:
    """Create an async SQLAlchemy database engine."""
    if not database_url:
        raise ValueError("DATABASE_URL is required to create the database engine.")

    engine_options: dict[str, Any] = {"pool_pre_ping": True}

    if make_url(database_url).get_backend_name() != "sqlite":
        engine_options.update(pool_size=5, max_overflow=10)

    return create_async_engine(database_url, **engine_options)


def create_session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    """Create an async database session factory."""
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


def get_database_engine(settings: Settings | None = None) -> AsyncEngine:
    """Create a database engine from application settings."""
    resolved_settings = settings or get_settings()

    logger.info("database_engine_initializing")

    return create_database_engine(resolved_settings.database_url)


@lru_cache(maxsize=1)
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Create and cache the application database session factory."""
    engine = get_database_engine()
    return create_session_factory(engine)


@asynccontextmanager
async def database_session_scope(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    """Provide a safe async database session scope."""
    async with session_factory() as session:
        try:
            yield session
        except SQLAlchemyError:
            await session.rollback()
            logger.exception("database_session_rollback")
            raise
        finally:
            await session.close()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that provides an async database session.

    The route or service layer should decide when to commit because
    AgentFlow workflows may contain multiple database operations.
    """
    session_factory = get_session_factory()

    async with session_factory() as session:
        try:
            yield session
        except SQLAlchemyError:
            await session.rollback()
            logger.exception("database_session_dependency_rollback")
            raise
        finally:
            await session.close()
