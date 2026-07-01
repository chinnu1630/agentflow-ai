class AppError(Exception):
    """Base application error for expected AgentFlow failures."""

    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 500,
    ) -> None:
        """Create an application error with safe client-facing details."""
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
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