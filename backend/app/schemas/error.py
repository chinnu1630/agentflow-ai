from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Detailed error information returned by the API."""

    code: str = Field(description="Machine-readable error code.")
    message: str = Field(description="Human-readable error message.")


class ErrorResponse(BaseModel):
    """Standard error response returned by AgentFlow APIs."""

    error: ErrorDetail = Field(description="Error details.")
    run_id: str = Field(description="Request correlation ID for debugging and audit.")