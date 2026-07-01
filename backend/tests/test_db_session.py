import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from app.db.session import create_database_engine, create_session_factory


def test_create_database_engine_requires_database_url() -> None:
    """Database engine creation should fail when DATABASE_URL is missing."""
    with pytest.raises(ValueError, match="DATABASE_URL is required"):
        create_database_engine(None)


def test_create_database_engine_returns_async_engine() -> None:
    """Database engine creation should return a SQLAlchemy AsyncEngine."""
    engine = create_database_engine(
        "postgresql+asyncpg://user:password@localhost:5432/agentflow_test"
    )

    assert isinstance(engine, AsyncEngine)


def test_create_session_factory_returns_factory() -> None:
    """Session factory should be created from an async engine."""
    engine = create_database_engine(
        "postgresql+asyncpg://user:password@localhost:5432/agentflow_test"
    )

    session_factory = create_session_factory(engine)

    assert session_factory is not None