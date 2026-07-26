"""Tests for the atomic Redis token-bucket rate limiter."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from app.services.redis_rate_limiter import (
    RateLimitDecision,
    RateLimiterProtocolError,
    RateLimitPolicy,
    RedisTokenBucketRateLimiter,
)


def test_rate_limit_policy_rejects_request_larger_than_capacity() -> None:
    """One request must never consume more tokens than the bucket can hold."""
    with pytest.raises(ValidationError):
        RateLimitPolicy(
            name="expensive",
            capacity=5,
            refill_rate_per_second=0.1,
            requested_tokens=6,
        )


@pytest.mark.anyio
async def test_evaluate_allows_request_and_returns_remaining_tokens() -> None:
    """The limiter should parse one successful atomic Lua result."""
    redis_client = Mock()
    redis_client.eval = AsyncMock(return_value=[1, "4.0", 0])
    limiter = RedisTokenBucketRateLimiter(redis_client)
    policy = RateLimitPolicy(
        name="expensive",
        capacity=5,
        refill_rate_per_second=0.1,
    )

    decision = await limiter.evaluate(
        key="agentflow:rate-limit:expensive:test-key",
        policy=policy,
        now_ms=1_000,
    )

    assert decision == RateLimitDecision(
        allowed=True,
        remaining_tokens=4.0,
        retry_after_seconds=0,
    )
    redis_client.eval.assert_awaited_once()

    arguments = redis_client.eval.await_args.args

    assert arguments[1] == 1
    assert arguments[2] == "agentflow:rate-limit:expensive:test-key"
    assert arguments[3:] == (
        5,
        0.0001,
        1_000,
        1,
        100_000,
    )


@pytest.mark.anyio
async def test_evaluate_denies_request_and_rounds_retry_delay_up() -> None:
    """Clients should receive a whole-second retry delay that is never early."""
    redis_client = Mock()
    redis_client.eval = AsyncMock(return_value=[0, "0.25", 7_501])
    limiter = RedisTokenBucketRateLimiter(redis_client)
    policy = RateLimitPolicy(
        name="expensive",
        capacity=5,
        refill_rate_per_second=0.1,
    )

    decision = await limiter.evaluate(
        key="agentflow:rate-limit:expensive:test-key",
        policy=policy,
        now_ms=2_000,
    )

    assert decision.allowed is False
    assert decision.remaining_tokens == 0.25
    assert decision.retry_after_seconds == 8


@pytest.mark.anyio
async def test_evaluate_rejects_blank_redis_key() -> None:
    """The limiter should never execute Redis with an empty state key."""
    redis_client = Mock()
    redis_client.eval = AsyncMock()
    limiter = RedisTokenBucketRateLimiter(redis_client)
    policy = RateLimitPolicy(
        name="standard",
        capacity=60,
        refill_rate_per_second=1.0,
    )

    with pytest.raises(ValueError, match="Redis rate-limit key"):
        await limiter.evaluate(
            key="   ",
            policy=policy,
            now_ms=1_000,
        )

    redis_client.eval.assert_not_awaited()


@pytest.mark.anyio
async def test_evaluate_rejects_malformed_lua_response() -> None:
    """Unexpected Redis script output must fail closed instead of allowing."""
    redis_client = Mock()
    redis_client.eval = AsyncMock(return_value=["unexpected"])
    limiter = RedisTokenBucketRateLimiter(redis_client)
    policy = RateLimitPolicy(
        name="standard",
        capacity=60,
        refill_rate_per_second=1.0,
    )

    with pytest.raises(RateLimiterProtocolError):
        await limiter.evaluate(
            key="agentflow:rate-limit:standard:test-key",
            policy=policy,
            now_ms=1_000,
        )

@pytest.mark.anyio
async def test_evaluate_uses_redis_server_clock_by_default() -> None:
    """Production evaluations should avoid clock drift across API replicas."""
    redis_client = Mock()
    redis_client.eval = AsyncMock(return_value=[1, "4.0", 0])
    limiter = RedisTokenBucketRateLimiter(redis_client)
    policy = RateLimitPolicy(
        name="expensive",
        capacity=5,
        refill_rate_per_second=0.1,
    )

    await limiter.evaluate(
        key="agentflow:rate-limit:expensive:test-key",
        policy=policy,
    )

    arguments = redis_client.eval.await_args.args

    assert arguments[5] == -1
