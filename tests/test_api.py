from fastapi.testclient import TestClient

from api import app


def test_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction_endpoint():
    client = TestClient(app)
    response = client.post("/api/v1/predict", json={"date": "2025-06-24", "top_k": 3})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["forecast_date"] == "2025-06-24"
    assert len(body["hotspots"]) == 3
    assert {"risk_score", "risk_level", "fraud_type", "city", "lat", "lon"}.issubset(body["hotspots"][0])
