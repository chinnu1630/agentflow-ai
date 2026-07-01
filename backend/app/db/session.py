from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

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

    return create_async_engine(
        database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )


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