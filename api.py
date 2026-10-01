"""FastAPI service for the cybercrime cash-out hotspot model.

Start:
    uvicorn api:app --reload

Swagger UI:
    http://127.0.0.1:8000/docs
"""
from datetime import date as date_type
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src import config as C
from src.common import load_data
from src.predict import forecast_from_frames

app = FastAPI(
    title="Cash-Out Hotspot Prediction API",
    version="1.0.0",
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
