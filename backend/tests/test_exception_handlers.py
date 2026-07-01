from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.exception_handlers import register_exception_handlers
from app.core.exceptions import ExternalServiceError
from app.middleware.request_context import RequestContextMiddleware


def create_test_app() -> FastAPI:
    """Create a test app with exception handlers registered."""
    test_app = FastAPI()
    test_app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(test_app)

    @test_app.get("/external-error")
    async def external_error_route() -> None:
        """Raise a test external service error."""
        raise ExternalServiceError(
            service_name="GitHub",
            message="rate limit exceeded",
        )

    return test_app


def test_app_error_handler_returns_standard_error_response() -> None:
    """AppError should return AgentFlow's standard error format."""
    client = TestClient(create_test_app())

    response = client.get("/external-error")

    assert response.status_code == 503

    response_data = response.json()

    assert response_data["error"]["code"] == "EXTERNAL_SERVICE_ERROR"
    assert "GitHub is currently unavailable" in response_data["error"]["message"]
    assert "run_id" in response_data


def test_not_found_error_returns_standard_error_response() -> None:
    """Unknown routes should return AgentFlow's standard error format."""
    client = TestClient(create_test_app())

    response = client.get("/missing-route")

    assert response.status_code == 404

    response_data = response.json()

    assert response_data["error"]["code"] == "HTTP_ERROR"
    assert response_data["error"]["message"] == "Not Found"
    assert "run_id" in response_data