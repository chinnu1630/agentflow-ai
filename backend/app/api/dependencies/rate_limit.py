"""FastAPI dependencies for distributed AgentFlow API rate limiting."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Callable, Coroutine
from enum import StrEnum
from typing import Any, cast

from fastapi import Request
from opentelemetry import trace
from redis.exceptions import RedisError

from app.api.dependencies.security import PrincipalDependency
from app.core.config import Settings, get_settings
from app.core.exceptions import (
    RateLimitExceededError,
    RateLimitServiceUnavailableError,
)
from app.core.logging import get_logger
from app.core.security import AuthenticatedPrincipal
from app.observability.tracing import set_safe_span_attributes
from app.services.redis_rate_limiter import (
    RateLimiterProtocolError,
    RateLimitPolicy,
    RedisEvalClient,
    RedisTokenBucketRateLimiter,
)

logger = get_logger(__name__)


class RateLimitPolicyClass(StrEnum):
    """Stable endpoint classes used for token-bucket configuration."""

    STANDARD = "standard"
    EXPENSIVE = "expensive"


RateLimitDependency = Callable[
    [Request, AuthenticatedPrincipal],
    Coroutine[Any, Any, None],
]


def build_identity_rate_limit_key(
    *,
    subject: str,
    policy_name: str,
    hmac_secret: str,
) -> str:
    """Build a stable pseudonymous Redis key for one principal and policy.

    HMAC-SHA256 prevents raw JWT subjects, emails, or predictable identity
    values from appearing in Redis keys. Time complexity is O(n) in the
    subject length, and the resulting key uses O(1) storage.

    Args:
        subject: Verified JWT subject claim.
        policy_name: Stable rate-limit policy name.
        hmac_secret: Secret key used for pseudonymous key derivation.

    Returns:
        Namespaced Redis key containing only a hexadecimal HMAC digest.

    Raises:
        ValueError: If any input is blank.
    """
    normalized_subject = subject.strip()
    normalized_policy = policy_name.strip()
    normalized_secret = hmac_secret.strip()

    if not normalized_subject:
        raise ValueError("Rate-limit identity subject must not be blank.")

    if not normalized_policy:
        raise ValueError("Rate-limit policy name must not be blank.")

    if not normalized_secret:
        raise ValueError("Rate-limit HMAC secret must not be blank.")

    identity_digest = hmac.new(
        normalized_secret.encode("utf-8"),
        normalized_subject.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return (
        f"agentflow:rate-limit:{normalized_policy}:"
        f"{identity_digest}"
    )


def enforce_rate_limit(
    policy_class: RateLimitPolicyClass,
) -> RateLimitDependency:
    """Create a dependency enforcing one distributed token-bucket policy.

    Authentication runs before this dependency because the principal is a
    dependency input. Redis evaluation therefore occurs before database,
    GitHub, Jira, RAG, LangGraph, or Claude dependencies declared later on
    protected route functions.

    Args:
        policy_class: Standard or expensive endpoint policy class.

    Returns:
        Async FastAPI dependency that permits or rejects one request.
    """

    async def enforce(
        request: Request,
        principal: PrincipalDependency,
    ) -> None:
        """Evaluate one authenticated request against its Redis bucket."""
        settings = get_settings()
        run_id = str(
            getattr(request.state, "run_id", "unknown-run-id")
        )

        if not settings.rate_limit_enabled:
            set_safe_span_attributes(
                trace.get_current_span(),
                {
                    "rate_limit.enabled": False,
                    "rate_limit.policy": policy_class.value,
                },
            )
            return

        redis_client = _get_redis_client(request)

        if redis_client is None:
            _log_unavailable(
                run_id=run_id,
                policy_class=policy_class,
                error_type="RedisClientUnavailable",
            )
            if policy_class is RateLimitPolicyClass.EXPENSIVE:
                raise RateLimitServiceUnavailableError()
            return

        if settings.rate_limit_key_hmac_secret is None:
            _log_unavailable(
                run_id=run_id,
                policy_class=policy_class,
                error_type="RateLimitConfigurationUnavailable",
            )
            if policy_class is RateLimitPolicyClass.EXPENSIVE:
                raise RateLimitServiceUnavailableError()
            return

        policy = _build_policy(
            settings=settings,
            policy_class=policy_class,
        )
        redis_key = build_identity_rate_limit_key(
            subject=principal.subject,
            policy_name=policy.name,
            hmac_secret=(
                settings.rate_limit_key_hmac_secret.get_secret_value()
            ),
        )
        limiter = RedisTokenBucketRateLimiter(redis_client)

        try:
            decision = await limiter.evaluate(
                key=redis_key,
                policy=policy,
            )
        except (RedisError, RateLimiterProtocolError) as exc:
            _log_unavailable(
                run_id=run_id,
                policy_class=policy_class,
                error_type=type(exc).__name__,
            )
            if policy_class is RateLimitPolicyClass.EXPENSIVE:
                raise RateLimitServiceUnavailableError() from exc
            return

        safe_attributes = {
            "rate_limit.enabled": True,
            "rate_limit.policy": policy.name,
            "rate_limit.allowed": decision.allowed,
            "rate_limit.remaining_tokens": (
                decision.remaining_tokens
            ),
            "rate_limit.retry_after_seconds": (
                decision.retry_after_seconds
            ),
            "rate_limit.redis_available": True,
        }
        set_safe_span_attributes(
            trace.get_current_span(),
            safe_attributes,
        )

        if not decision.allowed:
            retry_after_seconds = max(
                1,
                decision.retry_after_seconds,
            )

            logger.warning(
                "api_rate_limit_exceeded",
                extra={
                    "run_id": run_id,
                    "policy": policy.name,
                    "remaining_tokens": (
                        decision.remaining_tokens
                    ),
                    "retry_after_seconds": retry_after_seconds,
                },
            )

            raise RateLimitExceededError(
                retry_after_seconds=retry_after_seconds,
            )

        logger.info(
            "api_rate_limit_allowed",
            extra={
                "run_id": run_id,
                "policy": policy.name,
                "remaining_tokens": decision.remaining_tokens,
            },
        )

    return enforce


def _get_redis_client(
    request: Request,
) -> RedisEvalClient | None:
    """Return a Redis-compatible client from FastAPI application state."""
    client: object = getattr(
        request.app.state,
        "redis_client",
        None,
    )

    if client is None:
        return None

    if callable(getattr(client, "eval", None)):
        return cast(RedisEvalClient, client)

    return None


def _build_policy(
    *,
    settings: Settings,
    policy_class: RateLimitPolicyClass,
) -> RateLimitPolicy:
    """Build one validated token-bucket policy from application settings."""
    if policy_class is RateLimitPolicyClass.EXPENSIVE:
        return RateLimitPolicy(
            name=policy_class.value,
            capacity=settings.rate_limit_expensive_capacity,
            refill_rate_per_second=(
                settings.rate_limit_expensive_refill_rate_per_second
            ),
        )

    return RateLimitPolicy(
        name=policy_class.value,
        capacity=settings.rate_limit_standard_capacity,
        refill_rate_per_second=(
            settings.rate_limit_standard_refill_rate_per_second
        ),
    )


def _log_unavailable(
    *,
    run_id: str,
    policy_class: RateLimitPolicyClass,
    error_type: str,
) -> None:
    """Record safe Redis rate-limit failure metadata."""
    set_safe_span_attributes(
        trace.get_current_span(),
        {
            "rate_limit.enabled": True,
            "rate_limit.policy": policy_class.value,
            "rate_limit.allowed": False,
            "rate_limit.redis_available": False,
            "rate_limit.error_type": error_type,
        },
    )

    logger.error(
        "api_rate_limit_unavailable",
        extra={
            "run_id": run_id,
            "policy": policy_class.value,
            "error_type": error_type,
        },
    )
