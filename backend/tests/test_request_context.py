from uuid import UUID

from fastapi.testclient import TestClient

from app.main import app


def test_request_context_adds_run_id_header() -> None:
    """Every API response should include an X-Run-ID header."""
    client = TestClient(app)

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert "X-Run-ID" in response.headers

    run_id = response.headers["X-Run-ID"]
    UUID(run_id)


def test_request_context_preserves_valid_incoming_run_id() -> None:
    """Middleware should preserve a valid incoming X-Run-ID header."""
    client = TestClient(app)

    incoming_run_id = "550e8400-e29b-41d4-a716-446655440000"

    response = client.get(
        "/api/v1/health",
        headers={"X-Run-ID": incoming_run_id},
    )

    assert response.status_code == 200
    assert response.headers["X-Run-ID"] == incoming_run_id


def test_request_context_replaces_invalid_incoming_run_id() -> None:
    """Middleware should replace invalid incoming X-Run-ID values."""
    client = TestClient(app)

    response = client.get(
        "/api/v1/health",
        headers={"X-Run-ID": "bad-run-id"},
    )

    assert response.status_code == 200
    assert response.headers["X-Run-ID"] != "bad-run-id"

    UUID(response.headers["X-Run-ID"])