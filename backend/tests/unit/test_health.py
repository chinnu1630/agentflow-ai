from fastapi.testclient import TestClient

from app.main import app


def test_health_check_returns_ok_status() -> None:
    """Health endpoint should return service status successfully."""
    client = TestClient(app)

    response = client.get("/api/v1/health")

    assert response.status_code == 200

    response_data = response.json()

    assert response_data["status"] == "ok"
    assert response_data["service"] == "AgentFlow AI Backend"
    assert "timestamp" in response_data