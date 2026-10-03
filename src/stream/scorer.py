"""Real-time scorer: online features -> portable model -> ranked hotspot table."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .. import config as C
from ..features import FEATURE_COLUMNS
from ..portable_model import PortableGBM
from ..risk import risk_level
from .online_store import OnlineFeatureStore

EXTRA_COLS = ["nbr7", "hv7", "days_since_last", "ewm", "active_days_14", "r14", "r28"]
STD_COLS = ["rank", "cell_id", "city", "state", "lat", "lon", "risk_score", "risk_level",
            "fraud_type", "fraud_volume_7d", "r3", "r7", "burst", "atm_count"]


class RealtimeScorer:
    def __init__(self, store: OnlineFeatureStore, model: PortableGBM):
        if model.feature_names != FEATURE_COLUMNS:
            raise ValueError("model feature order differs from src.features.FEATURE_COLUMNS")
        self.store, self.model = store, model

    def score_day(self, date, top_k=None, timings: dict | None = None) -> pd.DataFrame:
        t0 = time.perf_counter_ns()
        f = self.store.features_for_day(date)
        t1 = time.perf_counter_ns()
        X = np.column_stack([f[c] for c in FEATURE_COLUMNS])
        risk = self.model.predict_proba1(X)
        t2 = time.perf_counter_ns()
        names, vol = self.store.fraud_context(date)
        cells = self.store.cells
        out = pd.DataFrame({"cell_id": cells["cell_id"].to_numpy(), "city": cells["city"].to_numpy(),
                            "state": cells["state"].to_numpy(), "lat": f["lat"], "lon": f["lon"],
                            "risk_score": risk, "fraud_type": names, "fraud_volume_7d": vol,
                            "atm_count": f["atm_count"]})
        for c in ["r3", "r7", "burst"] + EXTRA_COLS:
            if c not in out:
                out[c] = f[c]
        out["risk_level"] = out["risk_score"].map(risk_level)
        out["rank"] = out["risk_score"].rank(ascending=False, method="first").astype(int)
        out = out.sort_values("rank")[STD_COLS + [c for c in EXTRA_COLS if c not in ("r3",)]]
        if top_k is not None:
            out = out.head(int(top_k))
        if timings is not None:
            timings["feature_ms"] = (t1 - t0) / 1e6
            timings["inference_ms"] = (t2 - t1) / 1e6
            timings["total_ms"] = (time.perf_counter_ns() - t0) / 1e6
        return out.reset_index(drop=True)
