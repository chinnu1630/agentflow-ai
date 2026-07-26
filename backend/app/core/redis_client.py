"""Shared asynchronous Redis client lifecycle for AgentFlow AI."""

from __future__ import annotations

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def create_redis_client(settings: Settings) -> Redis[str]:
    """Create one configured asynchronous Redis client.

    Args:
        settings: Validated AgentFlow application settings.

    Returns:
        Redis client backed by a bounded asynchronous connection pool.

    Raises:
        ValueError: If no Redis URL has been configured.
    """
    if settings.redis_url is None:
        raise ValueError(
            "REDIS_URL is required to create the Redis client."
        )

    redis_url = settings.redis_url.get_secret_value()

    logger.info(
        "redis_client_initializing",
        extra={
            "max_connections": settings.redis_max_connections,
            "connect_timeout_seconds": (
                settings.redis_connect_timeout_seconds
            ),
            "socket_timeout_seconds": (
                settings.redis_socket_timeout_seconds
            ),
        },
    )

    return Redis.from_url(
        redis_url,
        encoding="utf-8",
        decode_responses=True,
        max_connections=settings.redis_max_connections,
        socket_connect_timeout=settings.redis_connect_timeout_seconds,
        socket_timeout=settings.redis_socket_timeout_seconds,
        health_check_interval=30,
    )


async def ping_redis_client(client: Redis[str]) -> None:
    """Verify that the shared Redis service is reachable.

    Args:
        client: Initialized asynchronous Redis client.

    Raises:
        RedisError: If Redis cannot be reached or rejects the operation.
    """
    try:
        await client.ping()
    except RedisError:
        logger.exception("redis_connectivity_check_failed")
        raise

    logger.info("redis_connectivity_check_succeeded")


async def close_redis_client(client: Redis[str]) -> None:
    """Close the Redis client and release pooled connections.

    Args:
        client: Initialized asynchronous Redis client.

    Raises:
        RedisError: If Redis client shutdown fails.
    """
    try:
        await client.close()
    except RedisError:
        logger.exception("redis_client_shutdown_failed")
        raise

    logger.info("redis_client_shutdown_completed")
