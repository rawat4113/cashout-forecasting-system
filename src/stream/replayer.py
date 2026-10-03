"""Background replay of history as a live feed (stands in for Kafka / MQ in the demo)."""
from __future__ import annotations

import threading
import time

import pandas as pd

from .pipeline import StreamPipeline


class DemoReplayer:
    def __init__(self, pipeline: StreamPipeline, events: pd.DataFrame):
        self.p, self.events = pipeline, events
        self._thread = None
        self._stop = threading.Event()
        self.status = dict(running=False, sim_time=None, sent=0, end=None)

    def start(self, start_date: str, days: int, seconds_per_sim_hour: float = 0.5):
        if self._thread and self._thread.is_alive():
            raise RuntimeError("replay already running")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(pd.Timestamp(start_date).normalize(), days, seconds_per_sim_hour), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self, start, days, sph):
        p, ev = self.p, self.events
        p.store.bulk_load(ev, before=start)
        live = ev[ev["traced"] & (ev["complaint_time"] >= start) & (ev["complaint_time"] < start + pd.Timedelta(days=days))].sort_values("complaint_time")
        recs = live[["event_id", "complaint_time", "cell_id", "category", "amount"]].to_dict("records")
        i = 0
        end = start + pd.Timedelta(days=days)
        self.status = dict(running=True, sim_time=str(start), sent=0, end=str(end))
        for h in range(days * 24):
            if self._stop.is_set():
                break
            now = start + pd.Timedelta(hours=h)
            if h % 24 == 0:
                p.issue(now, "OFFICIAL")
            elif h % 6 == 0:
                p.issue(now.normalize() + pd.Timedelta(days=1), "PROVISIONAL", log=False)
            boundary = now + pd.Timedelta(hours=1)
            while i < len(recs) and recs[i]["complaint_time"] < boundary:
                p.ingest(recs[i]); i += 1
            self.status.update(sim_time=str(boundary), sent=i)
            time.sleep(sph)
        self.status["running"] = False
