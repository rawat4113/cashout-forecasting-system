import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    import api
    api._PIPE = None
    api._REPLAYER = None
    return TestClient(api.app)


def test_ingest_forecast_audit_flow(client):
    r = client.post("/api/v1/events", json={"events": [
        dict(event_id=1, complaint_time="2025-06-01T08:00:00", cell_id=0, category="UPI / Card Fraud", amount=5000),
        dict(event_id=2, complaint_time="bad", cell_id=0, category="x", amount=1),
    ]})
    assert r.status_code == 200 and r.json()["accepted"] == 1 and len(r.json()["rejected"]) == 1

    r = client.post("/api/v1/live/forecast", json={"date": "2025-06-02", "kind": "OFFICIAL"})
    assert r.status_code == 200, r.text
    assert r.json()["summary"]["n_cells"] == 145 and r.json()["alerts"]

    a = client.get("/api/v1/live/alerts", params={"level": "HIGH", "limit": 3}).json()
    assert all(x["level"] == "HIGH" and x["reasons"] for x in a["alerts"])

    assert client.get("/api/v1/audit/verify").json() == dict(ok=True, entries=1, first_bad_seq=None, detail="chain intact")
    s = client.get("/api/v1/live/stats").json()
    assert s["score_latency"]["count"] == 1 and s["audit"]["ok"]


def test_live_endpoints_before_any_forecast(client):
    assert client.get("/api/v1/live/alerts").json()["alerts"] == []
    assert client.get("/api/v1/live/hotspots").json()["hotspots"] == []
