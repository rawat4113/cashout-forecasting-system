"""
Online feature store: O(1) per-event updates, same features as the batch trainer.

State per ATM cluster and day: complaint count, amount, high-value count, per-category count.
`features_for_day()` hands that state to `src.features.features_from_arrays` -- the very same
function the batch trainer uses -- so serving features cannot silently diverge from training.
`tests/test_stream_parity.py` proves it end to end.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .. import config as C
from ..features import HIGH_VALUE_CATEGORIES, features_from_arrays, nearest_neighbours
from .schema import KNOWN_CATEGORIES, ComplaintEvent, category_index

_HV_IDX = {category_index(c) for c in HIGH_VALUE_CATEGORIES}


class OnlineFeatureStore:
    def __init__(self, cells: pd.DataFrame, start: str = C.START_DATE, capacity_days: int = C.N_DAYS + 60):
        self.cells = cells.reset_index(drop=True)
        self.start = pd.Timestamp(start)
        self.n_cells = len(self.cells)
        self._row = {int(c): i for i, c in enumerate(self.cells["cell_id"])}
        self._cap = capacity_days
        z = lambda *s: np.zeros(s)
        self.o_cnt, self.o_amt, self.o_hv = z(self.n_cells, self._cap), z(self.n_cells, self._cap), z(self.n_cells, self._cap)
        self.o_cat = z(self.n_cells, self._cap, len(KNOWN_CATEGORIES))
        self._nbr = nearest_neighbours(self.cells, C.KNN_NEIGHBOURS)     # static geometry, computed once
        self.n_ingested = 0
        self.n_rejected = 0
        self.n_late = 0                    # arrived for a day that was already scored
        self.last_issued_day = -1

    # ---- ingestion --------------------------------------------------------------------------
    def _day(self, ts) -> int:
        return (pd.Timestamp(ts).normalize() - self.start).days

    def _grow(self, day: int):
        extra = max(day + 31 - self._cap, 0)
        if extra:
            for name in ("o_cnt", "o_amt", "o_hv"):
                setattr(self, name, np.pad(getattr(self, name), ((0, 0), (0, extra))))
            self.o_cat = np.pad(self.o_cat, ((0, 0), (0, extra), (0, 0)))
            self._cap += extra

    def ingest(self, ev: ComplaintEvent) -> bool:
        row = self._row.get(ev.cell_id)
        day = self._day(ev.complaint_time)
        if row is None or day < 0:
            self.n_rejected += 1
            return False
        self._grow(day)
        k = category_index(ev.category)
        self.o_cnt[row, day] += 1
        self.o_amt[row, day] += ev.amount
        self.o_cat[row, day, k] += 1
        if k in _HV_IDX:
            self.o_hv[row, day] += 1
        if day < self.last_issued_day:          # day already consumed by an issued forecast
            self.n_late += 1
        self.n_ingested += 1
        return True

    def bulk_load(self, events: pd.DataFrame, before: Optional[pd.Timestamp] = None) -> int:
        """Vectorised warm start from history (traced complaints only). Returns rows loaded."""
        e = events[events["traced"]] if "traced" in events else events
        if before is not None:
            e = e[e["complaint_time"] < before]
        rows = e["cell_id"].map(self._row)
        days = (e["complaint_time"].dt.normalize() - self.start).dt.days
        ok = rows.notna() & (days >= 0)
        e, rows, days = e[ok], rows[ok].to_numpy(int), days[ok].to_numpy(int)
        if len(days):
            self._grow(int(days.max()))
        amt = e["amount"].to_numpy(float)
        cats = np.array([category_index(c) for c in e["category"]], dtype=int)
        hv = np.isin(cats, list(_HV_IDX))
        np.add.at(self.o_cnt, (rows, days), 1.0)
        np.add.at(self.o_amt, (rows, days), amt)
        np.add.at(self.o_cat, (rows, days, cats), 1.0)
        np.add.at(self.o_hv, (rows[hv], days[hv]), 1.0)
        self.n_ingested += len(rows)
        return len(rows)

    # ---- serving ----------------------------------------------------------------------------
    def features_for_day(self, date) -> dict:
        """name -> (n_cells,) vector for the forecast day. Uses ONLY complaints from earlier days."""
        d = self._day(date)
        if d < 0:
            raise ValueError("date precedes store start")
        self._grow(d)
        n = d + 1
        f = features_from_arrays(self.o_cnt[:, :n], self.o_amt[:, :n], self.o_hv[:, :n],
                                 self.cells, self.start, n, nbr=self._nbr)
        return {k: v[:, d] for k, v in f.items()}

    def fraud_context(self, date):
        """(dominant category name, its 7-day count) per cell, over the 7 days before `date`."""
        d = self._day(date)
        lo = max(0, d - 7)
        cat7 = self.o_cat[:, lo:d, :].sum(axis=1)             # (cells, categories)
        vol = cat7.max(axis=1).astype(int)
        names = np.array(KNOWN_CATEGORIES, dtype=object)[cat7.argmax(axis=1)]   # argmax: first max -> alphabetical tie-break
        names = np.where(vol > 0, names, "No recent classified complaints")
        return names, vol

    def mark_issued(self, date):
        self.last_issued_day = max(self.last_issued_day, self._day(date))
