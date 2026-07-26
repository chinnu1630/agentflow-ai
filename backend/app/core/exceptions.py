from collections.abc import Mapping


class AppError(Exception):
    """Base application error for expected AgentFlow failures."""

    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 500,
        response_headers: Mapping[str, str] | None = None,
    ) -> None:
        """Create an application error with safe client-facing details.

        Args:
            message: Safe human-readable error message.
            error_code: Stable machine-readable AgentFlow error code.
            status_code: HTTP status returned to the API client.
            response_headers: Optional safe HTTP response headers.
        """
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.response_headers = dict(response_headers or {})
        super().__init__(message)


class ExternalServiceError(AppError):
    """Raised when an external service such as GitHub, Jira, or Slack fails."""

    def __init__(self, service_name: str, message: str) -> None:
        """Create an external service error."""
        super().__init__(
            message=f"{service_name} is currently unavailable: {message}",
            error_code="EXTERNAL_SERVICE_ERROR",
            status_code=503,
        )


class NotFoundError(AppError):
    """Raised when a requested resource cannot be found."""

    def __init__(self, resource_name: str) -> None:
        """Create a not found error."""
        super().__init__(
            message=f"{resource_name} was not found.",
            error_code="NOT_FOUND",
            status_code=404,
        )

class RateLimitExceededError(AppError):
    """Raised when an AgentFlow API token bucket has no available tokens."""

    def __init__(self, *, retry_after_seconds: int) -> None:
        """Create a rate-limit rejection with a safe retry duration.

        Args:
            retry_after_seconds: Whole seconds before the client should retry.

        Raises:
            ValueError: If the retry duration is less than one second.
        """
        if retry_after_seconds < 1:
            raise ValueError(
                "retry_after_seconds must be at least one."
            )

        super().__init__(
            message="API rate limit exceeded.",
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            response_headers={
                "Retry-After": str(retry_after_seconds),
            },
        )


class RateLimitServiceUnavailableError(AppError):
    """Raised when a protected endpoint cannot verify its Redis rate limit."""

    def __init__(self) -> None:
        """Create a fail-closed Redis rate-limit availability error."""
        super().__init__(
            message="API rate limiting is temporarily unavailable.",
            error_code="RATE_LIMIT_SERVICE_UNAVAILABLE",
            status_code=503,
        )
