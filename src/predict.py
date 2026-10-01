"""
Produce hotspot forecasts from the trained model.

CLI examples:
    python -m src.predict
    python -m src.predict --date 2025-05-20 --top-k 15

The same forecast() function is used by the FastAPI service and the dashboard,
so all three surfaces share the exact same feature engineering + model path.
"""
import argparse
import json
from typing import Optional, Tuple

import joblib
import pandas as pd

from . import config as C
from .common import load_data
from .features import FEATURE_COLUMNS, build_panel


def risk_level(p: float) -> str:
    if p >= C.RISK_HIGH:
        return "HIGH"
    if p >= C.RISK_MEDIUM:
        return "MEDIUM"
    return "LOW"


def _validate_date(date: pd.Timestamp) -> int:
    start = pd.Timestamp(C.START_DATE)
    day_idx = (date - start).days
    last = (start + pd.Timedelta(days=C.N_DAYS - 1)).date()
    if not (C.WARMUP_DAYS <= day_idx <= C.N_DAYS):
        raise ValueError(
            f"date must be between {(start + pd.Timedelta(days=C.WARMUP_DAYS)).date()} "
            f"and {last + pd.Timedelta(days=1)} (next day after the data ends)"
        )
    return day_idx


def _add_fraud_context(out: pd.DataFrame, events: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """Add view-layer fraud context without changing the model features."""
    cutoff = pd.Timestamp(date)
    recent = events[
        events["traced"]
        & (events["complaint_time"] < cutoff)
        & (events["complaint_time"] >= cutoff - pd.Timedelta(days=7))
    ].copy()

    if recent.empty:
        out["fraud_type"] = "No recent classified complaints"
        out["fraud_volume_7d"] = 0
        return out

    category_counts = (
        recent.groupby(["cell_id", "category"])
        .size()
        .rename("category_count")
        .reset_index()
        .sort_values(["cell_id", "category_count", "category"], ascending=[True, False, True])
    )
    dominant = category_counts.drop_duplicates("cell_id").rename(
        columns={"category": "fraud_type", "category_count": "fraud_volume_7d"}
    )[["cell_id", "fraud_type", "fraud_volume_7d"]]
    return out.merge(dominant, on="cell_id", how="left").assign(
        fraud_type=lambda df: df["fraud_type"].fillna("No recent classified complaints"),
        fraud_volume_7d=lambda df: df["fraud_volume_7d"].fillna(0).astype(int),
    )


def forecast_from_frames(
    events: pd.DataFrame,
    cells: pd.DataFrame,
    date: pd.Timestamp,
    top_k: Optional[int] = 10,
) -> pd.DataFrame:
    """Generate a forecast from in-memory data frames.

    This is intentionally shared by the CLI, API, and dashboard so the demo has
    one prediction path instead of three independently implemented models.
    """
    date = pd.Timestamp(date)
    day_idx = _validate_date(date)
    panel = build_panel(events, cells, n_days=max(C.N_DAYS, day_idx + 1))
    day = panel[panel["day_idx"] == day_idx].copy()
    model = joblib.load(C.MODEL_DIR / "hgb_model.joblib")
    day["risk_score"] = model.predict_proba(day[FEATURE_COLUMNS])[:, 1]
    out = day.merge(cells[["cell_id", "city", "state"]], on="cell_id")
    out["risk_level"] = out["risk_score"].map(risk_level)
    out["rank"] = out["risk_score"].rank(ascending=False, method="first").astype(int)
    out = _add_fraud_context(out, events, date)
    cols = [
        "rank", "cell_id", "city", "state", "lat", "lon", "risk_score", "risk_level",
        "fraud_type", "fraud_volume_7d", "r3", "r7", "burst", "atm_count",
    ]
    out = out.sort_values("rank")[cols]
    if top_k is not None:
        out = out.head(int(top_k))
    return out.reset_index(drop=True)


def forecast(date: pd.Timestamp, top_k: Optional[int] = 10) -> pd.DataFrame:
    events, cells = load_data()
    return forecast_from_frames(events, cells, date, top_k=top_k)


def _write_outputs(table: pd.DataFrame, date: pd.Timestamp) -> Tuple[str, str]:
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = date.strftime("%Y-%m-%d")
    table.round(4).to_csv(C.OUT_DIR / f"forecast_{stamp}.csv", index=False)

    alerts = [
        dict(
            rank=int(r["rank"]),
            cell_id=int(r["cell_id"]),
            city=str(r["city"]),
            state=str(r["state"]),
            lat=float(r["lat"]),
            lon=float(r["lon"]),
            risk_score=round(float(r["risk_score"]), 4),
            level=str(r["risk_level"]),
            fraud_type=str(r["fraud_type"]),
            recent_complaints_7d=int(r["r7"]),
            message=(
                f"{r['risk_level']} cash-out risk near {r['city']} "
                f"(cluster {int(r['cell_id'])}) on {stamp}"
            ),
        )
        for _, r in table.iterrows()
    ]
    (C.OUT_DIR / f"alerts_{stamp}.json").write_text(
        json.dumps(dict(date=stamp, alerts=alerts), indent=2)
    )
    return str(C.OUT_DIR / f"forecast_{stamp}.csv"), str(C.OUT_DIR / f"alerts_{stamp}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: day after the data ends)")
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    date = pd.Timestamp(args.date) if args.date else pd.Timestamp(C.START_DATE) + pd.Timedelta(days=C.N_DAYS)
    table = forecast(date, args.top_k)
    _write_outputs(table, date)

    print(f"Top-{args.top_k} predicted cash-out hotspots for {date.strftime('%Y-%m-%d')}")
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
