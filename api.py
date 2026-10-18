"""FastAPI service for the cybercrime cash-out hotspot model.

Start:
    uvicorn api:app --reload

Swagger UI:
    http://127.0.0.1:8000/docs
"""
import os
from datetime import date as date_type
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src import config as C
from src.common import load_data
from src.stream.audit import verify_file
from src.stream.pipeline import StreamPipeline
from src.stream.replayer import DemoReplayer

app = FastAPI(
    title="Cash-Out Hotspot Prediction API + Real-Time Cybercrime Intelligence",
    version="2.0.0",
    description=(
        "Leakage-safe predictive analytics service for prioritising likely cash-withdrawal "
        "hotspots from cybercrime complaint activity. Demo data is synthetic."
    ),
)


class ForecastRequest(BaseModel):
    date: date_type = Field(..., description="Forecast date in YYYY-MM-DD format")
    top_k: int = Field(default=10, ge=1, le=145, description="Number of ranked hotspots to return")


class PredictResponse(BaseModel):
    forecast_date: str
    generated_from: str
    model: str
    hotspots: list[dict]


_DATA_CACHE: Optional[tuple[pd.DataFrame, pd.DataFrame]] = None


def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    global _DATA_CACHE
    if _DATA_CACHE is None:
        _DATA_CACHE = load_data()
    return _DATA_CACHE


def _make_response(date: date_type, top_k: int) -> PredictResponse:
    events, cells = get_data()
    try:
        from src.predict import forecast_from_frames       # batch path needs scikit-learn; the real-time path does not
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="batch endpoint needs scikit-learn/joblib on this node; use /api/v1/live/*") from exc
    try:
        result = forecast_from_frames(events, cells, pd.Timestamp(date), top_k=top_k)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    rows = result.where(pd.notna(result), None).to_dict(orient="records")
    for row in rows:
        for key, value in list(row.items()):
            if hasattr(value, "item"):
                row[key] = value.item()

    return PredictResponse(
        forecast_date=date.isoformat(),
        generated_from="data/events.csv + data/cells.csv",
        model="HistGradientBoostingClassifier",
        hotspots=rows,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "cash-out hotspot prediction",
        "data_source": "synthetic NCRP-style demo data",
        "model_file": str(C.MODEL_DIR / "hgb_model.joblib"),
    }


@app.post("/api/v1/predict", response_model=PredictResponse)
def predict(request: ForecastRequest):
    """Run the full data -> features -> prediction pipeline for one forecast date."""
    return _make_response(request.date, request.top_k)


@app.get("/api/v1/forecast", response_model=PredictResponse)
def forecast(
    date: date_type = Query(..., description="Forecast date in YYYY-MM-DD format"),
    top_k: int = Query(default=10, ge=1, le=145),
):
    return _make_response(date, top_k)


# =====================================================================================================
# Real-time layer (v2): streaming ingestion -> online features -> scoring -> alerts -> audit chain
# =====================================================================================================
_PIPE: Optional[StreamPipeline] = None
_REPLAYER: Optional[DemoReplayer] = None


def get_pipeline() -> StreamPipeline:
    global _PIPE
    if _PIPE is None:
        events, cells = get_data()
        audit = Path(os.environ.get("AUDIT_LOG_PATH", C.AUDIT_DIR / "audit_log.jsonl"))
        _PIPE = StreamPipeline(cells, audit_path=audit)
        _PIPE.store.bulk_load(events, before=pd.Timestamp("2025-06-01"))     # warm start: history before the demo window
    return _PIPE


def _clean(alerts: list[dict], level: Optional[str], limit: int) -> list[dict]:
    if level:
        alerts = [a for a in alerts if a["level"] == level.upper()]
    return alerts[:limit]


class EventBatch(BaseModel):
    events: list[dict] = Field(..., max_length=5000, description="Complaint events: event_id, complaint_time, cell_id, category, amount")


class LiveForecastRequest(BaseModel):
    date: date_type
    kind: str = Field(default="OFFICIAL", pattern="^(OFFICIAL|PROVISIONAL)$")


class ReplayRequest(BaseModel):
    start_date: date_type = date_type(2025, 6, 1)
    days: int = Field(default=7, ge=1, le=30)
    seconds_per_sim_hour: float = Field(default=0.5, ge=0.0, le=10.0)


@app.post("/api/v1/events", tags=["real-time"])
def ingest_events(batch: EventBatch):
    """Ingest complaint events. Malformed events are rejected individually; the batch is never all-or-nothing."""
    pipe = get_pipeline()
    results = [pipe.ingest(e) for e in batch.events]
    return dict(received=len(results), accepted=sum(r["accepted"] for r in results),
                rejected=[dict(index=i, reason=r["reason"]) for i, r in enumerate(results) if not r["accepted"]][:20])


@app.post("/api/v1/live/forecast", tags=["real-time"])
def live_forecast(req: LiveForecastRequest):
    """Issue a forecast from the live online feature store (OFFICIAL is audit-logged)."""
    try:
        res = get_pipeline().issue(pd.Timestamp(req.date), req.kind)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return dict(summary=res["summary"], alerts=res["alerts"][:50])


@app.get("/api/v1/live/alerts", tags=["real-time"])
def live_alerts(level: Optional[str] = Query(default=None, pattern="^(?i)(HIGH|MEDIUM)$"), limit: int = Query(default=20, ge=1, le=200)):
    res = get_pipeline().latest
    if res["date"] is None:
        return dict(date=None, kind=None, alerts=[])
    return dict(date=res["date"], kind=res["kind"], summary=res["summary"], alerts=_clean(res["alerts"], level, limit))


@app.get("/api/v1/live/hotspots", tags=["real-time"])
def live_hotspots():
    """Full risk surface of the latest forecast (all clusters) for map rendering."""
    res = get_pipeline().latest
    if res["table"] is None:
        return dict(date=None, kind=None, hotspots=[])
    t = res["table"][["rank", "cell_id", "city", "state", "lat", "lon", "risk_score", "risk_level", "fraud_type", "r7"]]
    return dict(date=res["date"], kind=res["kind"], hotspots=t.round(4).to_dict("records"))


@app.get("/api/v1/live/stats", tags=["real-time"])
def live_stats():
    return get_pipeline().stats()


@app.post("/api/v1/demo/replay/start", tags=["demo"])
def replay_start(req: ReplayRequest):
    global _REPLAYER
    pipe = get_pipeline()
    events, _ = get_data()
    if _REPLAYER is None:
        _REPLAYER = DemoReplayer(pipe, events)
    try:
        _REPLAYER.start(req.start_date.isoformat(), req.days, req.seconds_per_sim_hour)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return dict(started=True, **req.model_dump(mode="json"))


@app.post("/api/v1/demo/replay/stop", tags=["demo"])
def replay_stop():
    if _REPLAYER:
        _REPLAYER.stop()
    return dict(stopped=True)


@app.get("/api/v1/demo/replay/status", tags=["demo"])
def replay_status():
    return _REPLAYER.status if _REPLAYER else dict(running=False)


@app.get("/api/v1/audit/verify", tags=["audit"])
def audit_verify():
    """Re-compute the whole hash chain and report whether any past entry was altered."""
    return get_pipeline().audit.verify()


@app.get("/api/v1/audit/tail", tags=["audit"])
def audit_tail(n: int = Query(default=10, ge=1, le=100)):
    p = get_pipeline().audit.path
    lines = p.read_text(encoding="utf-8").splitlines()[-n:] if p.exists() else []
    import json as _json
    return [_json.loads(x) for x in lines]
