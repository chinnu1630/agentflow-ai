"""Tests for the shared asynchronous Redis client."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.core.config import Settings
from app.core.redis_client import (
    close_redis_client,
    create_redis_client,
    ping_redis_client,
)

TEST_RATE_LIMIT_HMAC_SECRET = "test-rate-limit-secret"  # noqa: S105



def _rate_limit_settings() -> Settings:
    """Return valid Redis-backed rate-limit settings for unit tests."""
    return Settings(
        rate_limit_enabled=True,
        redis_url="redis://localhost:6379/0",
        rate_limit_key_hmac_secret=TEST_RATE_LIMIT_HMAC_SECRET,
        _env_file=None,
    )  # type: ignore[call-arg]


def test_create_redis_client_requires_configured_url() -> None:
    """Redis client creation should reject missing connection configuration."""
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    with pytest.raises(
        ValueError,
        match="REDIS_URL is required",
    ):
        create_redis_client(settings)


def test_create_redis_client_uses_validated_pool_and_timeout_settings() -> None:
    """Redis client creation should apply bounded production connection values."""
    settings = _rate_limit_settings()
    fake_client = Mock()

    with patch(
        "app.core.redis_client.Redis.from_url",
        return_value=fake_client,
    ) as from_url:
        client = create_redis_client(settings)

    assert client is fake_client
    from_url.assert_called_once_with(
        "redis://localhost:6379/0",
        encoding="utf-8",
        decode_responses=True,
        max_connections=20,
        socket_connect_timeout=1.0,
        socket_timeout=1.0,
        health_check_interval=30,
    )


@pytest.mark.anyio
async def test_ping_redis_client_checks_connectivity() -> None:
    """Redis startup verification should execute one asynchronous ping."""
    client = Mock()
    client.ping = AsyncMock(return_value=True)

    await ping_redis_client(client)

    client.ping.assert_awaited_once_with()


@pytest.mark.anyio
async def test_close_redis_client_releases_connection_pool() -> None:
    """Application shutdown should close the shared Redis client."""
    client = Mock()
    client.close = AsyncMock()

    await close_redis_client(client)

    client.close.assert_awaited_once_with()
