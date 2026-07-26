"""Tests for FastAPI Redis rate-limit dependencies."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.api.dependencies.rate_limit import (
    RateLimitPolicyClass,
    build_identity_rate_limit_key,
    enforce_rate_limit,
)
from app.core.config import Settings
from app.core.exceptions import (
    RateLimitExceededError,
    RateLimitServiceUnavailableError,
)
from app.core.security import AuthenticatedPrincipal
from app.observability.metrics import RateLimitMetricOutcome
from app.services.redis_rate_limiter import RateLimitDecision

TEST_RATE_LIMIT_HMAC_SECRET = "test-rate-limit-secret"  # noqa: S105



def _enabled_settings() -> Settings:
    """Return valid local Redis-backed rate-limit settings."""
    return Settings(
        rate_limit_enabled=True,
        redis_url="redis://localhost:6379/0",
        rate_limit_key_hmac_secret=TEST_RATE_LIMIT_HMAC_SECRET,
        rate_limit_standard_capacity=60,
        rate_limit_standard_refill_rate_per_second=1.0,
        rate_limit_expensive_capacity=5,
        rate_limit_expensive_refill_rate_per_second=0.1,
        _env_file=None,
    )  # type: ignore[call-arg]


def _principal() -> AuthenticatedPrincipal:
    """Return one trusted authenticated principal."""
    return AuthenticatedPrincipal(
        subject="director-123",
        email="director@example.com",
        scopes=frozenset({"agent:query"}),
    )


def _request(redis_client: object | None) -> Mock:
    """Return a request-shaped mock with app and run context."""
    request = Mock()
    request.state.run_id = "11111111-1111-1111-1111-111111111111"
    request.app.state.redis_client = redis_client
    return request


def test_build_identity_rate_limit_key_is_stable_and_pseudonymous() -> None:
    """Redis keys should be deterministic without exposing identity claims."""
    first_key = build_identity_rate_limit_key(
        subject="director-123",
        policy_name="expensive",
        hmac_secret=TEST_RATE_LIMIT_HMAC_SECRET,
    )
    second_key = build_identity_rate_limit_key(
        subject="director-123",
        policy_name="expensive",
        hmac_secret=TEST_RATE_LIMIT_HMAC_SECRET,
    )

    assert first_key == second_key
    assert first_key.startswith("agentflow:rate-limit:expensive:")
    assert "director-123" not in first_key
    assert "director@example.com" not in first_key


def test_build_identity_rate_limit_key_separates_policy_buckets() -> None:
    """Standard and expensive traffic should not share the same token bucket."""
    standard_key = build_identity_rate_limit_key(
        subject="director-123",
        policy_name="standard",
        hmac_secret=TEST_RATE_LIMIT_HMAC_SECRET,
    )
    expensive_key = build_identity_rate_limit_key(
        subject="director-123",
        policy_name="expensive",
        hmac_secret=TEST_RATE_LIMIT_HMAC_SECRET,
    )

    assert standard_key != expensive_key


@pytest.mark.anyio
async def test_disabled_rate_limiting_skips_redis_evaluation() -> None:
    """Local development should preserve existing behavior when disabled."""
    dependency = enforce_rate_limit(RateLimitPolicyClass.EXPENSIVE)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    with patch(
        "app.api.dependencies.rate_limit.get_settings",
        return_value=settings,
    ):
        result = await dependency(
            request=_request(redis_client=None),
            principal=_principal(),
        )

    assert result is None


@pytest.mark.anyio
async def test_expensive_policy_allows_request_with_available_token() -> None:
    """Allowed requests should continue after one Redis token evaluation."""
    dependency = enforce_rate_limit(RateLimitPolicyClass.EXPENSIVE)
    redis_client = Mock()
    limiter = Mock()
    limiter.evaluate = AsyncMock(
        return_value=RateLimitDecision(
            allowed=True,
            remaining_tokens=4.0,
            retry_after_seconds=0,
        )
    )

    with (
        patch(
            "app.api.dependencies.rate_limit.get_settings",
            return_value=_enabled_settings(),
        ),
        patch(
            "app.api.dependencies.rate_limit.RedisTokenBucketRateLimiter",
            return_value=limiter,
        ) as limiter_class,
    ):
        result = await dependency(
            request=_request(redis_client=redis_client),
            principal=_principal(),
        )

    assert result is None
    limiter_class.assert_called_once_with(redis_client)
    limiter.evaluate.assert_awaited_once()

    evaluated_policy = limiter.evaluate.await_args.kwargs["policy"]
    evaluated_key = limiter.evaluate.await_args.kwargs["key"]

    assert evaluated_policy.name == "expensive"
    assert evaluated_policy.capacity == 5
    assert evaluated_policy.refill_rate_per_second == 0.1
    assert "director-123" not in evaluated_key


@pytest.mark.anyio
async def test_denied_request_raises_rate_limit_error() -> None:
    """An empty token bucket should return a stable 429 retry response."""
    dependency = enforce_rate_limit(RateLimitPolicyClass.EXPENSIVE)
    limiter = Mock()
    limiter.evaluate = AsyncMock(
        return_value=RateLimitDecision(
            allowed=False,
            remaining_tokens=0.0,
            retry_after_seconds=8,
        )
    )

    with (
        patch(
            "app.api.dependencies.rate_limit.get_settings",
            return_value=_enabled_settings(),
        ),
        patch(
            "app.api.dependencies.rate_limit.RedisTokenBucketRateLimiter",
            return_value=limiter,
        ),
        pytest.raises(RateLimitExceededError) as exc_info,
    ):
        await dependency(
            request=_request(redis_client=Mock()),
            principal=_principal(),
        )

    assert exc_info.value.response_headers == {
        "Retry-After": "8",
    }


@pytest.mark.anyio
async def test_missing_shared_redis_client_fails_closed() -> None:
    """Protected traffic should fail closed when startup has no Redis client."""
    dependency = enforce_rate_limit(RateLimitPolicyClass.EXPENSIVE)

    with (
        patch(
            "app.api.dependencies.rate_limit.get_settings",
            return_value=_enabled_settings(),
        ),
        pytest.raises(RateLimitServiceUnavailableError),
    ):
        await dependency(
            request=_request(redis_client=None),
            principal=_principal(),
        )


@pytest.mark.anyio
async def test_redis_execution_failure_fails_closed() -> None:
    """Runtime Redis failure should not allow an expensive request through."""
    dependency = enforce_rate_limit(RateLimitPolicyClass.EXPENSIVE)
    limiter = Mock()
    limiter.evaluate = AsyncMock(
        side_effect=RedisConnectionError("redis unavailable")
    )

    with (
        patch(
            "app.api.dependencies.rate_limit.get_settings",
            return_value=_enabled_settings(),
        ),
        patch(
            "app.api.dependencies.rate_limit.RedisTokenBucketRateLimiter",
            return_value=limiter,
        ),
        pytest.raises(RateLimitServiceUnavailableError),
    ):
        await dependency(
            request=_request(redis_client=Mock()),
            principal=_principal(),
        )

@pytest.mark.anyio
async def test_standard_policy_fails_open_when_client_is_missing() -> None:
    """Standard traffic should remain available during a Redis outage."""
    dependency = enforce_rate_limit(RateLimitPolicyClass.STANDARD)

    with patch(
        "app.api.dependencies.rate_limit.get_settings",
        return_value=_enabled_settings(),
    ):
        result = await dependency(
            request=_request(redis_client=None),
            principal=_principal(),
        )

    assert result is None


@pytest.mark.anyio
async def test_standard_policy_fails_open_on_redis_execution_error() -> None:
    """Standard traffic should degrade gracefully after a Redis failure."""
    dependency = enforce_rate_limit(RateLimitPolicyClass.STANDARD)
    limiter = Mock()
    limiter.evaluate = AsyncMock(
        side_effect=RedisConnectionError("redis unavailable")
    )

    with (
        patch(
            "app.api.dependencies.rate_limit.get_settings",
            return_value=_enabled_settings(),
        ),
        patch(
            "app.api.dependencies.rate_limit.RedisTokenBucketRateLimiter",
            return_value=limiter,
        ),
    ):
        result = await dependency(
            request=_request(redis_client=Mock()),
            principal=_principal(),
        )

    assert result is None

@pytest.mark.anyio
async def test_rate_limit_telemetry_does_not_expose_identity(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Logs and trace attributes must contain only safe operational fields."""
    dependency = enforce_rate_limit(RateLimitPolicyClass.EXPENSIVE)
    redis_client = Mock()
    limiter = Mock()
    limiter.evaluate = AsyncMock(
        return_value=RateLimitDecision(
            allowed=True,
            remaining_tokens=4.0,
            retry_after_seconds=0,
        )
    )

    caplog.set_level(
        logging.INFO,
        logger="app.api.dependencies.rate_limit",
    )

    with (
        patch(
            "app.api.dependencies.rate_limit.get_settings",
            return_value=_enabled_settings(),
        ),
        patch(
            "app.api.dependencies.rate_limit.RedisTokenBucketRateLimiter",
            return_value=limiter,
        ),
        patch(
            "app.api.dependencies.rate_limit.set_safe_span_attributes",
        ) as set_span_attributes,
    ):
        result = await dependency(
            request=_request(redis_client=redis_client),
            principal=_principal(),
        )

    assert result is None

    evaluated_key = limiter.evaluate.await_args.kwargs["key"]
    logged_output = caplog.text

    assert "director-123" not in logged_output
    assert "director@example.com" not in logged_output
    assert evaluated_key not in logged_output
    assert TEST_RATE_LIMIT_HMAC_SECRET not in logged_output

    for call in set_span_attributes.call_args_list:
        attributes = call.args[1]
        serialized_attributes = repr(attributes)

        assert "director-123" not in serialized_attributes
        assert "director@example.com" not in serialized_attributes
        assert evaluated_key not in serialized_attributes
        assert TEST_RATE_LIMIT_HMAC_SECRET not in serialized_attributes



@pytest.mark.anyio
async def test_disabled_rate_limiting_records_disabled_metric() -> None:
    """Disabled rate limiting should emit one bounded disabled outcome."""
    dependency = enforce_rate_limit(RateLimitPolicyClass.EXPENSIVE)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    with (
        patch(
            "app.api.dependencies.rate_limit.get_settings",
            return_value=settings,
        ),
        patch(
            "app.api.dependencies.rate_limit.record_rate_limit_outcome",
        ) as record_outcome,
    ):
        await dependency(
            request=_request(redis_client=None),
            principal=_principal(),
        )

    record_outcome.assert_called_once_with(
        policy="expensive",
        outcome=RateLimitMetricOutcome.DISABLED,
    )


@pytest.mark.anyio
async def test_allowed_request_records_allowed_metric() -> None:
    """Successful Redis evaluation should emit one allowed outcome."""
    dependency = enforce_rate_limit(RateLimitPolicyClass.EXPENSIVE)
    limiter = Mock()
    limiter.evaluate = AsyncMock(
        return_value=RateLimitDecision(
            allowed=True,
            remaining_tokens=4.0,
            retry_after_seconds=0,
        )
    )

    with (
        patch(
            "app.api.dependencies.rate_limit.get_settings",
            return_value=_enabled_settings(),
        ),
        patch(
            "app.api.dependencies.rate_limit.RedisTokenBucketRateLimiter",
            return_value=limiter,
        ),
        patch(
            "app.api.dependencies.rate_limit.record_rate_limit_outcome",
        ) as record_outcome,
    ):
        await dependency(
            request=_request(redis_client=Mock()),
            principal=_principal(),
        )

    record_outcome.assert_called_once_with(
        policy="expensive",
        outcome=RateLimitMetricOutcome.ALLOWED,
    )


@pytest.mark.anyio
async def test_denied_request_records_denied_metric() -> None:
    """Exhausted token buckets should emit one denied outcome."""
    dependency = enforce_rate_limit(RateLimitPolicyClass.EXPENSIVE)
    limiter = Mock()
    limiter.evaluate = AsyncMock(
        return_value=RateLimitDecision(
            allowed=False,
            remaining_tokens=0.0,
            retry_after_seconds=8,
        )
    )

    with (
        patch(
            "app.api.dependencies.rate_limit.get_settings",
            return_value=_enabled_settings(),
        ),
        patch(
            "app.api.dependencies.rate_limit.RedisTokenBucketRateLimiter",
            return_value=limiter,
        ),
        patch(
            "app.api.dependencies.rate_limit.record_rate_limit_outcome",
        ) as record_outcome,
        pytest.raises(RateLimitExceededError),
    ):
        await dependency(
            request=_request(redis_client=Mock()),
            principal=_principal(),
        )

    record_outcome.assert_called_once_with(
        policy="expensive",
        outcome=RateLimitMetricOutcome.DENIED,
    )


@pytest.mark.anyio
async def test_missing_redis_records_error_and_fail_closed_metrics() -> None:
    """Expensive traffic should expose safe Redis and fail-closed metrics."""
    dependency = enforce_rate_limit(RateLimitPolicyClass.EXPENSIVE)

    with (
        patch(
            "app.api.dependencies.rate_limit.get_settings",
            return_value=_enabled_settings(),
        ),
        patch(
            "app.api.dependencies.rate_limit.record_rate_limit_outcome",
        ) as record_outcome,
        patch(
            "app.api.dependencies.rate_limit.record_rate_limit_redis_error",
        ) as record_redis_error,
        pytest.raises(RateLimitServiceUnavailableError),
    ):
        await dependency(
            request=_request(redis_client=None),
            principal=_principal(),
        )

    record_redis_error.assert_called_once_with(
        policy="expensive",
        error_type="RedisClientUnavailable",
    )
    record_outcome.assert_called_once_with(
        policy="expensive",
        outcome=RateLimitMetricOutcome.FAIL_CLOSED,
    )


@pytest.mark.anyio
async def test_standard_redis_failure_records_fail_open_metrics() -> None:
    """Standard traffic should expose safe Redis and fail-open metrics."""
    dependency = enforce_rate_limit(RateLimitPolicyClass.STANDARD)
    limiter = Mock()
    limiter.evaluate = AsyncMock(
        side_effect=RedisConnectionError("redis unavailable")
    )

    with (
        patch(
            "app.api.dependencies.rate_limit.get_settings",
            return_value=_enabled_settings(),
        ),
        patch(
            "app.api.dependencies.rate_limit.RedisTokenBucketRateLimiter",
            return_value=limiter,
        ),
        patch(
            "app.api.dependencies.rate_limit.record_rate_limit_outcome",
        ) as record_outcome,
        patch(
            "app.api.dependencies.rate_limit.record_rate_limit_redis_error",
        ) as record_redis_error,
    ):
        await dependency(
            request=_request(redis_client=Mock()),
            principal=_principal(),
        )

    record_redis_error.assert_called_once_with(
        policy="standard",
        error_type="ConnectionError",
    )
    record_outcome.assert_called_once_with(
        policy="standard",
        outcome=RateLimitMetricOutcome.FAIL_OPEN,
    )
