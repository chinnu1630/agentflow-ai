"""Tests for FastAPI Redis startup and shutdown lifecycle."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError

from app.core.config import Settings
from app.main import create_app

TEST_RATE_LIMIT_HMAC_SECRET = "test-rate-limit-secret"  # noqa: S105



def _enabled_settings() -> Settings:
    """Return local settings with Redis-backed rate limiting enabled."""
    return Settings(
        rate_limit_enabled=True,
        redis_url="redis://localhost:6379/0",
        rate_limit_key_hmac_secret=TEST_RATE_LIMIT_HMAC_SECRET,
        _env_file=None,
    )  # type: ignore[call-arg]


def test_lifespan_skips_redis_when_rate_limiting_is_disabled() -> None:
    """Local apps should not initialize Redis when the feature is disabled."""
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    with (
        patch("app.main.get_settings", return_value=settings),
        patch("app.main.create_redis_client") as create_client,
    ):
        test_app = create_app()

        with TestClient(test_app) as client:
            response = client.get("/api/v1/health")

            assert response.status_code == 200
            assert test_app.state.redis_client is None

    create_client.assert_not_called()


def test_lifespan_initializes_and_closes_one_shared_redis_client() -> None:
    """Enabled rate limiting should share one verified client per process."""
    settings = _enabled_settings()
    redis_client = Mock()

    with (
        patch("app.main.get_settings", return_value=settings),
        patch(
            "app.main.create_redis_client",
            return_value=redis_client,
        ) as create_client,
        patch(
            "app.main.ping_redis_client",
            new=AsyncMock(),
        ) as ping_client,
        patch(
            "app.main.close_redis_client",
            new=AsyncMock(),
        ) as close_client,
    ):
        test_app = create_app()

        with TestClient(test_app):
            assert test_app.state.redis_client is redis_client
            create_client.assert_called_once_with(settings)
            ping_client.assert_awaited_once_with(redis_client)

        close_client.assert_awaited_once_with(redis_client)
        assert test_app.state.redis_client is None


def test_lifespan_degrades_safely_when_redis_is_unavailable() -> None:
    """Redis startup failure should keep health available but limiter disabled."""
    settings = _enabled_settings()
    redis_client = Mock()

    with (
        patch("app.main.get_settings", return_value=settings),
        patch(
            "app.main.create_redis_client",
            return_value=redis_client,
        ),
        patch(
            "app.main.ping_redis_client",
            new=AsyncMock(
                side_effect=RedisConnectionError("redis unavailable")
            ),
        ),
        patch(
            "app.main.close_redis_client",
            new=AsyncMock(),
        ) as close_client,
    ):
        test_app = create_app()

        with TestClient(test_app) as client:
            response = client.get("/api/v1/health")

            assert response.status_code == 200
            assert test_app.state.redis_client is None

        close_client.assert_awaited_once_with(redis_client)
