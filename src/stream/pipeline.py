"""
StreamPipeline: ingest -> online features -> score -> explain -> audit.

Two kinds of forecast:
  * OFFICIAL    issued at 00:00 for the coming day, from complaints received before midnight
                (exactly what the model was trained for).
  * PROVISIONAL a rolling "what would tomorrow look like right now" refresh during the day. It sees only the
                part of today that has arrived, so it is a nowcast for analysts, never an official forecast.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .. import config as C
from ..features import FEATURE_COLUMNS
from ..portable_model import PortableGBM
from .alerts import AlertEngine
from .audit import AuditLog
from .online_store import OnlineFeatureStore
from .schema import ComplaintEvent, EventValidationError
from .scorer import RealtimeScorer


class LatencyStats:
    def __init__(self, keep: int = 5000):
        self._v = deque(maxlen=keep)

    def add(self, ms: float):
        self._v.append(ms)

    def summary(self) -> dict:
        if not self._v:
            return dict(count=0)
        a = np.array(self._v)
        return dict(count=len(a), p50_ms=round(float(np.percentile(a, 50)), 3),
                    p95_ms=round(float(np.percentile(a, 95)), 3), p99_ms=round(float(np.percentile(a, 99)), 3),
                    max_ms=round(float(a.max()), 3))


class StreamPipeline:
    def __init__(self, cells: pd.DataFrame, model_path: Path = C.PORTABLE_MODEL_PATH,
                 audit_path: Optional[Path] = None, start: str = C.START_DATE):
        self.model = PortableGBM.load(model_path)
        self.store = OnlineFeatureStore(cells, start=start)
        self.scorer = RealtimeScorer(self.store, self.model)
        self.engine = AlertEngine()
        self.audit = AuditLog(audit_path or (C.AUDIT_DIR / "audit_log.jsonl"))
        self.ingest_latency, self.score_latency = LatencyStats(), LatencyStats()
        self.latest: dict = dict(date=None, kind=None, alerts=[], table=None)
        self._lock = threading.RLock()
        self.started = time.time()

    # ---- ingestion --------------------------------------------------------------------------
    def ingest(self, raw: dict) -> dict:
        t0 = time.perf_counter_ns()
        try:
            ev = ComplaintEvent.from_dict(raw)
        except EventValidationError as exc:
            self.store.n_rejected += 1
            return dict(accepted=False, reason=str(exc))
        with self._lock:
            ok = self.store.ingest(ev)
        self.ingest_latency.add((time.perf_counter_ns() - t0) / 1e6)
        return dict(accepted=ok, reason=None if ok else "unknown cell or date before store start")

    def ingest_many(self, events: Iterable[dict]) -> dict:
        acc = sum(1 for e in events if self.ingest(e)["accepted"])
        return dict(accepted=acc)

    # ---- forecasting ------------------------------------------------------------------------
    def issue(self, date, kind: str = "OFFICIAL", min_level: str = "MEDIUM", log: bool = True) -> dict:
        date = pd.Timestamp(date).normalize()
        timings: dict = {}
        with self._lock:
            table = self.scorer.score_day(date, timings=timings)
            alerts = self.engine.build(table, date, min_level=min_level, model_sha=self.model.sha256)
            if kind == "OFFICIAL":
                self.store.mark_issued(date)
        self.score_latency.add(timings["total_ms"])
        summary = dict(date=date.strftime("%Y-%m-%d"), kind=kind, n_cells=len(table),
                       n_high=int((table["risk_level"] == "HIGH").sum()),
                       n_medium=int((table["risk_level"] == "MEDIUM").sum()),
                       n_new_alerts=sum(a["status"] != "UNCHANGED" for a in alerts),
                       timings_ms={k: round(v, 3) for k, v in timings.items()})
        if log:
            self.audit.append("FORECAST_ISSUED", dict(summary, alert_ids=[a["alert_id"] for a in alerts if a["status"] != "UNCHANGED"],
                                                       top_cells=table.head(5)[["cell_id", "risk_score"]].round(4).to_dict("records")),
                              self.model.sha256)
        with self._lock:
            self.latest = dict(date=summary["date"], kind=kind, alerts=alerts, table=table, summary=summary)
        return self.latest

    # ---- replay (demo / benchmark source) ---------------------------------------------------
    def replay(self, events: pd.DataFrame, start_date, n_days: int, provisional_every_h: int = 6, on_tick=None):
        """Replay history as a live stream, in event-time order.

        Everything before `start_date` is warm-loaded; from there each complaint is ingested one by one and an
        OFFICIAL forecast is issued at every midnight (plus provisional refreshes every `provisional_every_h` hours).
        """
        start_date = pd.Timestamp(start_date).normalize()
        self.store.bulk_load(events, before=start_date)
        live = events[events["traced"] & (events["complaint_time"] >= start_date)
                      & (events["complaint_time"] < start_date + pd.Timedelta(days=n_days))]
        live = live.sort_values("complaint_time")
        recs = live[["event_id", "complaint_time", "cell_id", "category", "amount"]].to_dict("records")
        i, results = 0, []
        for d in range(n_days):
            day = start_date + pd.Timedelta(days=d)
            res = self.issue(day, "OFFICIAL")                        # 00:00 forecast for `day`
            results.append(res)
            if on_tick:
                on_tick(res)
            for h in range(provisional_every_h, 25, provisional_every_h):
                boundary = day + pd.Timedelta(hours=h)
                while i < len(recs) and recs[i]["complaint_time"] < boundary:
                    self.ingest(recs[i]); i += 1
                if h < 24 and provisional_every_h:
                    prov = self.issue(day + pd.Timedelta(days=1), "PROVISIONAL", log=False)
                    if on_tick:
                        on_tick(prov)
        return results

    # ---- health -----------------------------------------------------------------------------
    def stats(self) -> dict:
        return dict(uptime_s=round(time.time() - self.started, 1), events_ingested=self.store.n_ingested,
                    events_rejected=self.store.n_rejected, late_events=self.store.n_late,
                    ingest_latency=self.ingest_latency.summary(), score_latency=self.score_latency.summary(),
                    model_sha256=self.model.sha256, latest_forecast=self.latest.get("date"),
                    latest_kind=self.latest.get("kind"), audit=self.audit.verify())
