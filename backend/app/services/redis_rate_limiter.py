"""Atomic Redis-backed token-bucket rate limiting for AgentFlow AI."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

_RATE_LIMIT_LUA_SCRIPT = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate_per_ms = tonumber(ARGV[2])
local now_ms = tonumber(ARGV[3])
local requested_tokens = tonumber(ARGV[4])
local ttl_ms = tonumber(ARGV[5])

if now_ms < 0 then
    local redis_time = redis.call("TIME")
    local seconds = tonumber(redis_time[1])
    local microseconds = tonumber(redis_time[2])
    now_ms = (seconds * 1000) + math.floor(microseconds / 1000)
end

local state = redis.call("HMGET", key, "tokens", "updated_at_ms")
local tokens = tonumber(state[1])
local updated_at_ms = tonumber(state[2])

if tokens == nil then
    tokens = capacity
end

if updated_at_ms == nil then
    updated_at_ms = now_ms
end

local elapsed_ms = math.max(0, now_ms - updated_at_ms)
tokens = math.min(
    capacity,
    tokens + (elapsed_ms * refill_rate_per_ms)
)

local allowed = 0
local retry_after_ms = 0

if tokens >= requested_tokens then
    tokens = tokens - requested_tokens
    allowed = 1
else
    local missing_tokens = requested_tokens - tokens
    retry_after_ms = math.ceil(
        missing_tokens / refill_rate_per_ms
    )
end

redis.call(
    "HSET",
    key,
    "tokens",
    tostring(tokens),
    "updated_at_ms",
    tostring(now_ms)
)
redis.call("PEXPIRE", key, ttl_ms)

return {
    allowed,
    tostring(tokens),
    retry_after_ms
}
"""


class RedisEvalClient(Protocol):
    """Minimal asynchronous Redis interface required by the limiter."""

    async def eval(
        self,
        script: str,
        numkeys: int,
        *keys_and_args: str | int | float,
    ) -> object:
        """Execute one Redis Lua script and return its raw result."""


class RateLimiterProtocolError(RuntimeError):
    """Raised when Redis returns an unexpected token-bucket response."""


class RateLimitPolicy(BaseModel):
    """Validated configuration for one token-bucket policy class."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    name: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_-]*$",
    )
    capacity: int = Field(ge=1, le=100_000)
    refill_rate_per_second: float = Field(
        gt=0.0,
        le=10_000.0,
    )
    requested_tokens: int = Field(
        default=1,
        ge=1,
        le=100_000,
    )

    @model_validator(mode="after")
    def validate_requested_tokens(self) -> RateLimitPolicy:
        """Ensure one request can fit inside the configured bucket."""
        if self.requested_tokens > self.capacity:
            raise ValueError(
                "requested_tokens must not exceed bucket capacity."
            )

        return self


class RateLimitDecision(BaseModel):
    """Result returned by one atomic token-bucket evaluation."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    allowed: bool
    remaining_tokens: float = Field(ge=0.0)
    retry_after_seconds: int = Field(ge=0)


class RedisTokenBucketRateLimiter:
    """Evaluate distributed token buckets using one atomic Redis script."""

    def __init__(self, redis_client: RedisEvalClient) -> None:
        """Create a limiter using the shared asynchronous Redis client.

        Args:
            redis_client: Process-wide Redis client managed by FastAPI lifespan.
        """
        self._redis_client = redis_client

    async def evaluate(
        self,
        *,
        key: str,
        policy: RateLimitPolicy,
        now_ms: int | None = None,
    ) -> RateLimitDecision:
        """Atomically refill and consume tokens for one identity-policy key.

        Time complexity is O(1) because the Lua script reads and writes one
        Redis hash. Space complexity is O(1) per identity-policy bucket.

        Args:
            key: Pseudonymous Redis state key.
            policy: Validated token-bucket policy.
            now_ms: Optional deterministic Unix timestamp for tests.

        Returns:
            Decision containing allowance, remaining tokens, and retry delay.

        Raises:
            ValueError: If the Redis key is blank or the timestamp is invalid.
            RateLimiterProtocolError: If Redis returns malformed script output.
            redis.exceptions.RedisError: If Redis execution fails.
        """
        normalized_key = key.strip()

        if not normalized_key:
            raise ValueError(
                "Redis rate-limit key must not be blank."
            )

        if now_ms is not None and now_ms < 0:
            raise ValueError("now_ms must not be negative.")

        resolved_now_ms = -1 if now_ms is None else now_ms

        refill_rate_per_ms = (
            policy.refill_rate_per_second / 1_000.0
        )
        bucket_ttl_ms = max(
            1_000,
            math.ceil(
                (
                    policy.capacity
                    / policy.refill_rate_per_second
                )
                * 2_000
            ),
        )

        raw_result = await self._redis_client.eval(
            _RATE_LIMIT_LUA_SCRIPT,
            1,
            normalized_key,
            policy.capacity,
            refill_rate_per_ms,
            resolved_now_ms,
            policy.requested_tokens,
            bucket_ttl_ms,
        )

        return self._parse_result(raw_result)

    @staticmethod
    def _parse_result(raw_result: object) -> RateLimitDecision:
        """Validate and convert the Redis Lua response."""
        if (
            not isinstance(raw_result, Sequence)
            or isinstance(raw_result, (str, bytes))
            or len(raw_result) != 3
        ):
            raise RateLimiterProtocolError(
                "Redis token-bucket response was malformed."
            )

        try:
            allowed_value = int(raw_result[0])
            remaining_tokens = max(
                0.0,
                float(raw_result[1]),
            )
            retry_after_ms = max(
                0,
                int(raw_result[2]),
            )
        except (TypeError, ValueError) as exc:
            raise RateLimiterProtocolError(
                "Redis token-bucket response contained invalid values."
            ) from exc

        if allowed_value not in {0, 1}:
            raise RateLimiterProtocolError(
                "Redis token-bucket allowance value was invalid."
            )

        retry_after_seconds = (
            math.ceil(retry_after_ms / 1_000)
            if retry_after_ms > 0
            else 0
        )

        return RateLimitDecision(
            allowed=allowed_value == 1,
            remaining_tokens=remaining_tokens,
            retry_after_seconds=retry_after_seconds,
        )
