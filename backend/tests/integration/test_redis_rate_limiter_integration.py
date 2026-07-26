"""Real Redis integration tests for distributed token-bucket behavior."""

from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest
from redis.asyncio import Redis

from app.services.redis_rate_limiter import (
    RateLimitPolicy,
    RedisTokenBucketRateLimiter,
)


def _get_test_redis_url() -> str:
    """Return the explicit Redis integration-test URL or skip safely."""
    redis_url = os.getenv("RATE_LIMIT_TEST_REDIS_URL")

    if redis_url is None or not redis_url.strip():
        pytest.skip(
            "RATE_LIMIT_TEST_REDIS_URL is required for Redis integration tests."
        )

    return redis_url.strip()


@pytest.mark.anyio
async def test_concurrent_replicas_cannot_overspend_shared_bucket() -> None:
    """Atomic Lua execution should allow exactly the configured capacity.

    Two independent Redis clients represent separate FastAPI replicas.
    Twenty concurrent requests target one identity-policy key with a capacity
    of five. Because every evaluation uses the same timestamp, no refill can
    occur during the test.

    The expected complexity remains O(1) per request and O(1) Redis storage
    for the shared identity-policy bucket.
    """
    redis_url = _get_test_redis_url()
    first_client = Redis.from_url(
        redis_url,
        encoding="utf-8",
        decode_responses=True,
    )
    second_client = Redis.from_url(
        redis_url,
        encoding="utf-8",
        decode_responses=True,
    )
    redis_key = (
        "agentflow:rate-limit:integration:"
        f"{uuid4().hex}"
    )
    policy = RateLimitPolicy(
        name="expensive",
        capacity=5,
        refill_rate_per_second=0.000001,
    )
    limiters = (
        RedisTokenBucketRateLimiter(first_client),
        RedisTokenBucketRateLimiter(second_client),
    )

    try:
        await first_client.delete(redis_key)

        decisions = await asyncio.gather(
            *(
                limiters[index % len(limiters)].evaluate(
                    key=redis_key,
                    policy=policy,
                    now_ms=1_000,
                )
                for index in range(20)
            )
        )

        allowed_decisions = [
            decision for decision in decisions if decision.allowed
        ]
        denied_decisions = [
            decision for decision in decisions if not decision.allowed
        ]

        assert len(allowed_decisions) == 5
        assert len(denied_decisions) == 15
        assert all(
            decision.retry_after_seconds >= 1
            for decision in denied_decisions
        )

        stored_tokens = await first_client.hget(redis_key, "tokens")
        remaining_ttl_ms = await first_client.pttl(redis_key)

        assert stored_tokens is not None
        assert float(stored_tokens) == pytest.approx(0.0)
        assert remaining_ttl_ms > 0

    finally:
        await first_client.delete(redis_key)
        await first_client.aclose()
        await second_client.aclose()
